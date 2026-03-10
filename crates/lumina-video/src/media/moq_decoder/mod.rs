//! MoQ decoder implementation with platform-specific hardware acceleration.
//!
//! This module provides VideoDecoderBackend implementations for MoQ (Media over QUIC)
//! live streaming with zero-copy GPU rendering on supported platforms.
//!
//! # Platform-Specific Decoders
//!
//! - [`MoqDecoder`]: Desktop decoder (macOS with VideoToolbox, others FFmpeg fallback)
//! - [`MoqAndroidDecoder`]: Android decoder using MediaCodec with zero-copy HardwareBuffer
//!
//! # Architecture
//!
//! ## macOS (VideoToolbox Zero-Copy)
//! ```text
//! moq:// URL -> moq-native Client (QUIC connection)
//!            -> hang::BroadcastConsumer (catalog + track subscription)
//!            -> hang::TrackConsumer (frame receipt)
//!            -> VTDecompressionSession (decode H.264/H.265)
//!            -> CVPixelBuffer -> IOSurface -> MacOSGpuSurface
//!            -> zero_copy::macos::import_iosurface() -> wgpu::Texture
//! ```
//!
//! ## Android (MediaCodec Zero-Copy)
//! ```text
//! moq:// URL -> moq-native Client (QUIC connection)
//!            -> hang::BroadcastConsumer (catalog + track subscription)
//!            -> hang::TrackConsumer (NAL unit receipt)
//!            -> MoqMediaCodecBridge.submitNalUnit() (JNI)
//!            -> MediaCodec (hardware decode)
//!            -> ImageReader (GPU_SAMPLED_IMAGE usage)
//!            -> HardwareBuffer -> nativeSubmitHardwareBuffer() (JNI)
//!            -> import_ahardwarebuffer_yuv_zero_copy()
//!            -> VkSamplerYcbcrConversion -> wgpu::Texture
//! ```
//!
//! ## Other Platforms (FFmpeg fallback)
//! ```text
//! moq:// URL -> moq-native Client (QUIC connection)
//!            -> hang::BroadcastConsumer (catalog + track subscription)
//!            -> hang::TrackConsumer (frame receipt)
//!            -> FFmpeg (decode H.264/H.265/AV1 frames)
//!            -> VideoFrame -> FrameQueue -> Rendering
//! ```
//!
//! # Live Stream Considerations
//!
//! - `duration()` returns `None` for live streams
//! - `seek()` returns an error (live streams don't support seeking)
//! - `is_eof()` returns true only when the stream actually ends

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::moq::{AudioTrackInfo, MoqTransportConfig, MoqUrl, VideoTrackInfo};
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
use super::video::{CpuFrame, Plane};
use super::video::{
    DecodedFrame, HwAccelType, PixelFormat, VideoDecoderBackend, VideoError, VideoFrame,
    VideoMetadata,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::video_decoder::HwAccelConfig;

use parking_lot::Mutex;

// Apple-specific imports for VTDecompressionSession zero-copy
#[cfg(any(target_os = "macos", target_os = "ios"))]
use super::video::MacOSGpuSurface;

/// Hard failsafe timeout for startup gating (seconds). Used by worker IDR gate
/// (directly) and audio pre-buffer (derived as +1s). Single source of truth.
pub(crate) const MOQ_STARTUP_HARD_FAILSAFE_SECS: u64 = 10;

/// Audio pipeline lifecycle state.
///
/// Defined here (not `moq_audio.rs`) because [`MoqSharedState`] and
/// [`MoqStatsSnapshot`] reference it on all platforms, including Android.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoqAudioStatus {
    /// No AAC track in catalog, audio disabled, or non-desktop platform.
    #[default]
    Unavailable,
    /// Track subscribed, audio thread spawning.
    Starting,
    /// Decoding and playing audio.
    Running,
    /// Initialisation or sustained runtime failure.
    Error,
}

/// Audio-specific shared state passed to `MoqAudioThread` on desktop platforms.
///
/// Defined here (not `moq_audio.rs`) so [`MoqSharedState`] can embed it
/// unconditionally on all platforms — non-desktop stays [`MoqAudioStatus::Unavailable`].
pub(crate) struct MoqAudioShared {
    /// Whether the audio decode/playback thread is running and ready.
    pub internal_audio_ready: std::sync::atomic::AtomicBool,
    /// Audio handle for volume/mute/position control (set by audio thread on init).
    pub moq_audio_handle: parking_lot::Mutex<Option<super::audio::AudioHandle>>,
    /// Current audio pipeline status.
    pub audio_status: parking_lot::Mutex<MoqAudioStatus>,
    /// Liveness flag: set to `true` when audio thread starts, `false` on teardown.
    /// Used by `VideoPlayer::poll_moq_audio_handle()` to detect stale handles
    /// that appear "available" after the thread has been torn down.
    pub alive: std::sync::atomic::AtomicBool,
    /// External audio health watchdog: monotonic timestamp of the last frame
    /// successfully forwarded (read from QUIC + sent to crossbeam) by the audio
    /// forward task. Written by the forward task after each successful `send()`,
    /// read by the main select loop's watchdog arm. `None` = no frames yet.
    ///
    /// Uses monotonic `Instant` (not wall-clock `SystemTime`) so NTP/clock
    /// adjustments can't spuriously trigger or delay stale detection.
    ///
    /// This provides defense-in-depth: even if the forward task's internal
    /// `tokio::time::timeout` fails to fire (observed in production), the main
    /// loop detects the stale heartbeat and forces teardown + resubscribe.
    pub last_audio_forward_frame_at: parking_lot::Mutex<Option<std::time::Instant>>,
    /// Ring buffer metrics updated by the audio thread for observability.
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android"))]
    pub ring_buffer_metrics: parking_lot::Mutex<super::audio_ring_buffer::RingBufferMetrics>,
}

impl MoqAudioShared {
    /// Creates a new `MoqAudioShared` with all fields in their default (unavailable) state.
    pub fn new() -> Self {
        Self {
            internal_audio_ready: std::sync::atomic::AtomicBool::new(false),
            moq_audio_handle: parking_lot::Mutex::new(None),
            audio_status: parking_lot::Mutex::new(MoqAudioStatus::Unavailable),
            alive: std::sync::atomic::AtomicBool::new(false),
            last_audio_forward_frame_at: parking_lot::Mutex::new(None),
            #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android"))]
            ring_buffer_metrics: parking_lot::Mutex::new(Default::default()),
        }
    }
}

/// Decoder state for MoQ streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqDecoderState {
    /// Initial state, not yet connected
    Disconnected,
    /// Connecting to MoQ relay
    Connecting,
    /// Connected, fetching catalog
    FetchingCatalog,
    /// Subscribed to tracks, receiving data
    Streaming,
    /// Stream ended normally
    Ended,
    /// Error occurred
    Error,
}

/// Configuration for the MoQ decoder.
#[derive(Debug, Clone)]
pub struct MoqDecoderConfig {
    /// Transport configuration
    pub transport: MoqTransportConfig,
    /// Hardware acceleration configuration (macOS only)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub hw_accel: HwAccelConfig,
    /// Maximum latency for track subscription (frames older than this are skipped)
    pub max_latency_ms: u64,
    /// Whether to enable audio track subscription
    pub enable_audio: bool,
    /// Audio volume (0.0 to 1.0)
    pub initial_volume: f32,
    /// Size of the MoQ audio handoff buffer (in AAC frames).
    ///
    /// Higher values smooth jitter, lower values reduce latency.
    /// Default: 120 (~2.5 s of AAC at 48 kHz / 1024-sample frames).
    pub audio_buffer_capacity: usize,
}

impl Default for MoqDecoderConfig {
    fn default() -> Self {
        Self {
            transport: MoqTransportConfig::default(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            hw_accel: HwAccelConfig::default(),
            max_latency_ms: 2000, // 2s max latency — prevents OrderedConsumer from
            // skipping during brief decode stalls (observed ~1s on iOS).
            // 500ms was too aggressive: stall > 500ms caused video to jump
            // to live edge while audio continued linearly → permanent A/V gap.
            enable_audio: true,
            initial_volume: 1.0,
            audio_buffer_capacity: 120,
        }
    }
}

impl MoqDecoderConfig {
    /// Auto-enable TLS verification bypass for localhost URLs (self-signed certs).
    fn apply_localhost_tls_bypass(&mut self, moq_url: &MoqUrl) {
        let host = moq_url.host();
        if host == "localhost" || host == "127.0.0.1" || host == "::1" {
            self.transport.disable_tls_verify = true;
        }
    }
}

/// A received video frame from MoQ, ready for decoding.
#[derive(Debug)]
pub(crate) struct MoqVideoFrame {
    /// Presentation timestamp in microseconds
    pub(crate) timestamp_us: u64,
    /// Encoded frame data (H.264/H.265/AV1 NAL units)
    pub(crate) data: bytes::Bytes,
    /// Whether this is a keyframe
    pub(crate) is_keyframe: bool,
}

/// Frame statistics for debugging pipeline issues.
#[derive(Default)]
pub(crate) struct FrameStats {
    /// Frames received from hang crate
    pub(crate) received: AtomicU64,
    /// Frames dropped due to channel backpressure
    pub(crate) dropped_backpressure: AtomicU64,
    /// Frames dropped waiting for IDR
    pub(crate) dropped_waiting_idr: AtomicU64,
    /// Frames skipped by startup IDR gate (before first valid IDR)
    pub(crate) skipped_startup_frames: AtomicU64,
    /// Frames passed to VT decoder
    pub(crate) submitted_to_decoder: AtomicU64,
    /// Frames decoded successfully (VT callback)
    pub(crate) decoded: AtomicU64,
    /// Frames returned to renderer
    pub(crate) rendered: AtomicU64,
    /// Decode errors
    pub(crate) decode_errors: AtomicU64,
    /// Frames dropped during DPB grace (after isolated VT callback error)
    pub(crate) dropped_dpb_grace: AtomicU64,
}

impl FrameStats {
    pub(crate) fn log_summary(&self, label: &str) {
        let received = self.received.load(Ordering::Relaxed);
        let dropped_bp = self.dropped_backpressure.load(Ordering::Relaxed);
        let dropped_idr = self.dropped_waiting_idr.load(Ordering::Relaxed);
        let skipped_startup = self.skipped_startup_frames.load(Ordering::Relaxed);
        let submitted = self.submitted_to_decoder.load(Ordering::Relaxed);
        let decoded = self.decoded.load(Ordering::Relaxed);
        let rendered = self.rendered.load(Ordering::Relaxed);
        let errors = self.decode_errors.load(Ordering::Relaxed);
        let dropped_dpb = self.dropped_dpb_grace.load(Ordering::Relaxed);

        tracing::info!(
            "MoQ FrameStats [{}]: recv={}, drop_bp={}, drop_idr={}, drop_dpb={}, skip_startup={}, submit={}, decoded={}, rendered={}, errors={}",
            label, received, dropped_bp, dropped_idr, dropped_dpb, skipped_startup, submitted, decoded, rendered, errors
        );
    }

    fn snapshot(&self) -> MoqFrameStatsSnapshot {
        MoqFrameStatsSnapshot {
            received: self.received.load(Ordering::Relaxed),
            dropped_backpressure: self.dropped_backpressure.load(Ordering::Relaxed),
            dropped_waiting_idr: self.dropped_waiting_idr.load(Ordering::Relaxed),
            skipped_startup_frames: self.skipped_startup_frames.load(Ordering::Relaxed),
            submitted_to_decoder: self.submitted_to_decoder.load(Ordering::Relaxed),
            decoded: self.decoded.load(Ordering::Relaxed),
            rendered: self.rendered.load(Ordering::Relaxed),
            decode_errors: self.decode_errors.load(Ordering::Relaxed),
            dropped_dpb_grace: self.dropped_dpb_grace.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of MoQ frame pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct MoqFrameStatsSnapshot {
    pub received: u64,
    pub dropped_backpressure: u64,
    pub dropped_waiting_idr: u64,
    pub skipped_startup_frames: u64,
    pub submitted_to_decoder: u64,
    pub decoded: u64,
    pub rendered: u64,
    pub decode_errors: u64,
    pub dropped_dpb_grace: u64,
}

/// Point-in-time snapshot of all MoQ decoder stats for UI display.
#[derive(Debug, Clone)]
pub struct MoqStatsSnapshot {
    pub state: MoqDecoderState,
    pub error_message: Option<String>,
    pub buffering_percent: i32,
    pub frame_stats: MoqFrameStatsSnapshot,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f32,
    pub has_codec_description: bool,
    pub transport_protocol: String,
    /// Current audio pipeline status.
    pub audio_status: MoqAudioStatus,
    /// Ring buffer health metrics (fill level, stall/overflow counts).
    pub ring_buffer_fill_percent: f32,
    pub ring_buffer_stall_count: u64,
    pub ring_buffer_overflow_count: u64,
    /// Audio codec name (e.g. "Opus", "AAC") from catalog, if audio track present.
    pub audio_codec: Option<String>,
}

/// Handle to MoQ shared state for producing snapshots.
///
/// Cheaply cloneable (wraps Arc). Stored in VideoPlayer to expose
/// MoQ-specific stats without requiring trait downcasting.
#[derive(Clone)]
pub struct MoqStatsHandle {
    shared: Arc<MoqSharedState>,
}

impl MoqStatsHandle {
    /// Returns the MoQ audio handle if available and alive.
    ///
    /// Used by `VideoPlayer::poll_moq_audio_handle()` for late binding.
    pub fn audio_handle(&self) -> Option<super::audio::AudioHandle> {
        if !self.shared.audio.alive.load(Ordering::Acquire) {
            return None;
        }
        self.shared.audio.moq_audio_handle.lock().clone()
    }

    /// Returns whether the MoQ audio thread is alive.
    pub fn is_audio_alive(&self) -> bool {
        self.shared.audio.alive.load(Ordering::Acquire)
    }

    pub fn snapshot(&self) -> MoqStatsSnapshot {
        let metadata = self.shared.metadata.lock().clone();

        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android"))]
        let (ring_buffer_fill_percent, ring_buffer_stall_count, ring_buffer_overflow_count) = {
            let rb = self.shared.audio.ring_buffer_metrics.lock().clone();
            let fill_pct = if rb.capacity_samples > 0 {
                rb.fill_samples as f32 * 100.0 / rb.capacity_samples as f32
            } else {
                0.0
            };
            (fill_pct, rb.stall_count, rb.overflow_count)
        };
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux", target_os = "android")))]
        let (ring_buffer_fill_percent, ring_buffer_stall_count, ring_buffer_overflow_count) =
            (0.0f32, 0u64, 0u64);

        MoqStatsSnapshot {
            state: *self.shared.state.lock(),
            error_message: self.shared.error_message.lock().clone(),
            buffering_percent: self.shared.buffering_percent.load(Ordering::Relaxed),
            frame_stats: self.shared.frame_stats.snapshot(),
            codec: metadata.codec,
            width: metadata.width,
            height: metadata.height,
            frame_rate: metadata.frame_rate,
            has_codec_description: self.shared.codec_description.lock().is_some(),
            transport_protocol: self.shared.transport_protocol.lock().clone(),
            audio_status: *self.shared.audio.audio_status.lock(),
            ring_buffer_fill_percent,
            ring_buffer_stall_count,
            ring_buffer_overflow_count,
            audio_codec: self.shared.audio_codec_name.lock().clone(),
        }
    }
}

/// Shared state between decoder and async worker.
pub(crate) struct MoqSharedState {
    /// Current decoder state
    pub(crate) state: Mutex<MoqDecoderState>,
    /// Error message if in error state
    pub(crate) error_message: Mutex<Option<String>>,
    /// Whether EOF has been reached
    pub(crate) eof_reached: AtomicBool,
    /// Buffering percentage (0-100)
    pub(crate) buffering_percent: AtomicI32,
    /// Video metadata (populated after catalog received)
    pub(crate) metadata: Mutex<VideoMetadata>,
    /// Audio track info (if available)
    pub(crate) audio_info: Mutex<Option<AudioTrackInfo>>,
    /// Audio codec name (e.g. "Opus", "AAC") set by worker after catalog parse.
    pub(crate) audio_codec_name: Mutex<Option<String>>,
    /// Codec description from catalog (avcC/hvcC box containing SPS/PPS)
    pub(crate) codec_description: Mutex<Option<bytes::Bytes>>,
    /// Frame statistics for debugging
    pub(crate) frame_stats: FrameStats,
    /// Transport protocol used (QUIC or WebSocket)
    pub(crate) transport_protocol: Mutex<String>,
    /// Audio-specific shared state, passed to MoqAudioThread on desktop.
    /// Present on all platforms (non-desktop stays Unavailable).
    pub(crate) audio: Arc<MoqAudioShared>,
    /// Decoder sets this when it is starved waiting for an IDR after decode errors.
    /// Worker handles it by re-subscribing the video track.
    pub(crate) request_video_resubscribe: AtomicBool,
}

impl MoqSharedState {
    pub(crate) fn new() -> Self {
        let audio = Arc::new(MoqAudioShared::new());
        Self {
            state: Mutex::new(MoqDecoderState::Disconnected),
            error_message: Mutex::new(None),
            eof_reached: AtomicBool::new(false),
            buffering_percent: AtomicI32::new(0),
            metadata: Mutex::new(VideoMetadata {
                width: 0,
                height: 0,
                duration: None,
                frame_rate: 0.0,
                codec: String::new(),
                pixel_aspect_ratio: 1.0,
                start_time: None,
            }),
            audio_info: Mutex::new(None),
            audio_codec_name: Mutex::new(None),
            codec_description: Mutex::new(None),
            frame_stats: FrameStats::default(),
            transport_protocol: Mutex::new("unknown".to_string()),
            audio,
            request_video_resubscribe: AtomicBool::new(false),
        }
    }

    pub(crate) fn set_state(&self, state: MoqDecoderState) {
        *self.state.lock() = state;
    }

    pub(crate) fn set_error(&self, message: String) {
        *self.state.lock() = MoqDecoderState::Error;
        *self.error_message.lock() = Some(message);
        self.eof_reached.store(true, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    fn update_metadata(&self, track: &VideoTrackInfo) {
        let mut metadata = self.metadata.lock();
        metadata.width = track.width;
        metadata.height = track.height;
        metadata.frame_rate = track.frame_rate;
        metadata.codec = format!("{:?}", track.codec);
    }
}

// Platform-specific decoder implementations
mod desktop;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod macos_vt;
#[cfg(target_os = "android")]
pub mod android;
#[cfg(target_os = "linux")]
mod linux_gst;

pub use desktop::MoqDecoder;
#[cfg(target_os = "android")]
pub use android::MoqAndroidDecoder;
#[cfg(target_os = "linux")]
pub use linux_gst::MoqGStreamerDecoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moq_url_detection() {
        assert!(MoqDecoder::is_moq_url("moq://localhost/live/stream"));
        assert!(MoqDecoder::is_moq_url(
            "moqs://relay.example.com/live/stream"
        ));
        assert!(!MoqDecoder::is_moq_url("https://example.com/video.mp4"));
        assert!(!MoqDecoder::is_moq_url("rtmp://stream.example.com/live"));
    }

    #[test]
    fn test_config_default() {
        let config = MoqDecoderConfig::default();
        assert_eq!(config.max_latency_ms, 2000);
        assert!(config.enable_audio);
        assert_eq!(config.initial_volume, 1.0);
    }

    #[test]
    fn test_shared_state() {
        let shared = MoqSharedState::new();
        assert_eq!(*shared.state.lock(), MoqDecoderState::Disconnected);
        assert!(!shared.eof_reached.load(Ordering::Relaxed));

        shared.set_state(MoqDecoderState::Connecting);
        assert_eq!(*shared.state.lock(), MoqDecoderState::Connecting);

        shared.set_error("Test error".to_string());
        assert_eq!(*shared.state.lock(), MoqDecoderState::Error);
        assert!(shared.eof_reached.load(Ordering::Relaxed));
    }

    #[cfg(target_os = "android")]
    mod android_tests {
        use super::android::CodecType;

        #[test]
        fn test_codec_type_parsing() {
            assert_eq!(
                CodecType::from_catalog_codec("avc1.64001f"),
                Some(CodecType::H264)
            );
            assert_eq!(CodecType::from_catalog_codec("h264"), Some(CodecType::H264));
            assert_eq!(
                CodecType::from_catalog_codec("hvc1.1.6.L93.B0"),
                Some(CodecType::H265)
            );
            assert_eq!(CodecType::from_catalog_codec("hevc"), Some(CodecType::H265));
            assert_eq!(CodecType::from_catalog_codec("av1"), None);
            assert_eq!(CodecType::from_catalog_codec("vp9"), None);
        }

        #[test]
        fn test_mime_types() {
            assert_eq!(CodecType::H264.mime_type(), "video/avc");
            assert_eq!(CodecType::H265.mime_type(), "video/hevc");
        }
    }
}
