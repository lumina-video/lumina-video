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

#[cfg(target_os = "macos")]
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use super::moq::{AudioTrackInfo, MoqTransportConfig, MoqUrl, VideoTrackInfo};
#[cfg(not(target_os = "macos"))]
use super::video::{CpuFrame, Plane};
use super::video::{
    DecodedFrame, HwAccelType, PixelFormat, VideoDecoderBackend, VideoError, VideoFrame,
    VideoMetadata,
};
#[cfg(target_os = "macos")]
use super::video_decoder::HwAccelConfig;

use async_channel::{Receiver, Sender};
use parking_lot::Mutex;
use tokio::runtime::Handle;

// macOS-specific imports for VTDecompressionSession zero-copy
#[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
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
            #[cfg(target_os = "macos")]
            hw_accel: HwAccelConfig::default(),
            max_latency_ms: 500, // 500ms max latency for live streaming
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

/// MoQ video decoder using hang crate for media subscription.
///
/// On macOS, this decoder uses VTDecompressionSession for hardware-accelerated
/// zero-copy decoding with IOSurface output for direct GPU rendering.
pub struct MoqDecoder {
    /// Parsed MoQ URL
    #[allow(dead_code)]
    url: MoqUrl,
    /// Configuration
    #[allow(dead_code)]
    config: MoqDecoderConfig,
    /// Shared state with async worker
    shared: Arc<MoqSharedState>,
    /// Receiver for decoded frames from async worker
    frame_rx: Receiver<MoqVideoFrame>,
    /// Active hardware acceleration type
    active_hw_type: HwAccelType,
    /// Owned tokio runtime (created if none exists)
    _owned_runtime: Option<tokio::runtime::Runtime>,
    /// Tokio runtime handle for async operations
    _runtime: Handle,
    /// Whether audio is muted
    audio_muted: bool,
    /// Audio volume (0.0 to 1.0)
    audio_volume: f32,
    /// Cached metadata (updated from shared state to avoid unsafe access)
    cached_metadata: VideoMetadata,
    /// Last frame timestamp for timing
    #[allow(dead_code)]
    last_frame_time: Option<std::time::Instant>,
    /// Start time for PTS calculation
    #[allow(dead_code)]
    start_time: std::time::Instant,
    /// macOS VTDecompressionSession for hardware decoding (zero-copy)
    #[cfg(target_os = "macos")]
    vt_decoder: Option<macos_vt::VTDecoder>,
    /// H.264 AVCC NAL length field size (1, 2, or 4 bytes), from avcC.
    #[cfg(target_os = "macos")]
    h264_nal_length_size: usize,
    /// True if stream is AVCC format (from catalog avcC). Used for correct NAL
    /// parsing in `find_nal_types_for_format`, avoiding the `data_is_annex_b`
    /// heuristic which misclassifies 256-511 byte NALs.
    #[cfg(target_os = "macos")]
    is_avcc: bool,
    /// True after 3+ consecutive decode errors; decoder must wait for next IDR
    /// to resync. Cleared only when a real IDR (NAL type 5) arrives.
    /// The VT session is destroyed on sustained decode failures.
    #[cfg(target_os = "macos")]
    waiting_for_idr_after_error: bool,
    /// Consecutive decode error count. Isolated errors (1-2) skip the frame but
    /// keep the VT session alive. At 3+ consecutive errors, the session is
    /// destroyed and `waiting_for_idr_after_error` is set. Reset to 0 on
    /// successful decode or IDR resync.
    #[cfg(target_os = "macos")]
    consecutive_decode_errors: u32,
    /// Lightweight DPB grace: skip non-IDR frames after an isolated VT callback
    /// error (consecutive_decode_errors == 1 only). Bypasses note_idr_wait_progress()
    /// to avoid premature resubscribe. Bounded by timeout/drop budget;
    /// on expiry, escalates to waiting_for_idr_after_error for normal recovery.
    #[cfg(target_os = "macos")]
    skip_pframes_until_idr: bool,
    #[cfg(target_os = "macos")]
    dpb_grace_started_at: Option<std::time::Instant>,
    #[cfg(target_os = "macos")]
    dpb_grace_dropped_frames: u32,
    /// Observed real-IDR cadence (EMA, microseconds). Used to adapt DPB grace
    /// timeout/drop budget to stream cadence instead of fixed constants.
    #[cfg(target_os = "macos")]
    observed_idr_interval_us: Option<u64>,
    /// Last seen real-IDR PTS (microseconds) for cadence estimation.
    #[cfg(target_os = "macos")]
    last_idr_pts_us: Option<u64>,
    /// Strict recovery gate: after VT session recreation for recovery, only
    /// allow non-IDR frames once a real IDR decodes successfully on the fresh
    /// session.
    #[cfg(target_os = "macos")]
    require_clean_idr_after_recreate: bool,
    /// True until the one-shot VT session recreation has fired.
    #[cfg(target_os = "macos")]
    needs_session_recreation: bool,
    /// If set, defer VT session (re)creation until this instant.
    /// This avoids blocking sleeps on decode paths while still enforcing
    /// a quiesce window after session destruction.
    #[cfg(target_os = "macos")]
    quiesce_until: Option<std::time::Instant>,
    /// Counts VT session creations for lifecycle diagnostics.
    #[cfg(target_os = "macos")]
    vt_session_count: u32,
    /// Opt-in diagnostics for frame-level forensic logging around decode errors.
    #[cfg(target_os = "macos")]
    forensic_enabled: bool,
    /// Rolling window of most-recent submitted frames (for N-3 context on failure).
    #[cfg(target_os = "macos")]
    forensic_recent: VecDeque<ForensicFrameSample>,
    /// Active post-error capture window (+3 frames after failing frame).
    #[cfg(target_os = "macos")]
    forensic_post_error: Option<PostErrorCapture>,
    /// Monotonic frame sequence number for forensic logs.
    #[cfg(target_os = "macos")]
    forensic_seq: u64,
    /// Approximate group index, incremented on keyframe boundaries.
    #[cfg(target_os = "macos")]
    forensic_group_index: u64,
    /// Start of current "waiting for IDR" starvation window.
    #[cfg(target_os = "macos")]
    idr_wait_started_at: Option<std::time::Instant>,
    /// Number of frames dropped in current IDR starvation window.
    #[cfg(target_os = "macos")]
    idr_wait_dropped_frames: u32,
    /// Number of keyframe boundaries seen that still lacked a real IDR while
    /// waiting for IDR recovery.
    #[cfg(target_os = "macos")]
    idr_wait_broken_keyframe_boundaries: u8,
    /// Last time a decoder-side video re-subscribe request was emitted.
    /// Used to prevent rapid request churn when stream metadata is unstable.
    #[cfg(target_os = "macos")]
    idr_last_resubscribe_request_at: Option<std::time::Instant>,
    /// Start of the current RequiredFrameDropped storm-cycle window.
    #[cfg(target_os = "macos")]
    required_drop_window_started_at: Option<std::time::Instant>,
    /// Number of storm cycles observed in the current window.
    #[cfg(target_os = "macos")]
    required_drop_storms_in_window: u8,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
struct ForensicFrameSample {
    seq: u64,
    group_idx: u64,
    pts_us: u64,
    size: usize,
    is_keyframe: bool,
    hash64: u64,
    first16: [u8; 16],
    first16_len: usize,
    nal_types: [u8; MoqDecoder::MAX_NAL_TYPES],
    nal_count: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug)]
struct PostErrorCapture {
    trigger_seq: u64,
    remaining_after: u8,
}

impl MoqDecoder {
    /// Creates a new MoQ decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, MoqDecoderConfig::default())
    }

    /// Creates a new MoQ decoder with explicit configuration.
    pub fn new_with_config(url: &str, mut config: MoqDecoderConfig) -> Result<Self, VideoError> {
        tracing::info!("MoqDecoder::new_with_config: creating decoder for {}", url);

        // Parse the MoQ URL
        let moq_url = MoqUrl::parse(url).map_err(|e| {
            tracing::error!("MoQ: failed to parse URL: {}", e);
            VideoError::OpenFailed(e.to_string())
        })?;

        config.apply_localhost_tls_bypass(&moq_url);

        // Get existing runtime handle or create a new runtime
        let (owned_runtime, runtime) = match Handle::try_current() {
            Ok(handle) => (None, handle),
            Err(_) => {
                // No runtime exists, create one for MoQ async operations
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("moq-runtime")
                    .build()
                    .map_err(|e| {
                        VideoError::OpenFailed(format!("Failed to create tokio runtime: {e}"))
                    })?;
                let handle = rt.handle().clone();
                (Some(rt), handle)
            }
        };

        // Create shared state
        let shared = Arc::new(MoqSharedState::new());

        // Create channel for frames (bounded to limit memory usage)
        let (frame_tx, frame_rx) = async_channel::bounded(30);

        // Spawn the async connection/subscription worker
        let worker_shared = shared.clone();
        let worker_url = moq_url.clone();
        let worker_config = config.clone();

        runtime.spawn(async move {
            tracing::info!("MoQ: worker task starting for {:?}", worker_url);
            if let Err(e) =
                Self::run_moq_worker(worker_shared.clone(), worker_url, worker_config, frame_tx)
                    .await
            {
                tracing::error!("MoQ: worker error: {}", e);
                worker_shared.set_error(format!("MoQ worker error: {e}"));
            }
        });

        let initial_volume = config.initial_volume;
        #[cfg(target_os = "macos")]
        let forensic_enabled = std::env::var("LUMINA_MOQ_ERROR_FORENSICS")
            .map(|v| v != "0")
            .unwrap_or(false);
        Ok(Self {
            url: moq_url,
            config,
            shared,
            frame_rx,
            #[cfg(target_os = "macos")]
            active_hw_type: HwAccelType::VideoToolbox,
            #[cfg(not(target_os = "macos"))]
            active_hw_type: HwAccelType::None,
            _owned_runtime: owned_runtime,
            _runtime: runtime,
            audio_muted: false,
            audio_volume: initial_volume,
            cached_metadata: VideoMetadata {
                width: 0,
                height: 0,
                duration: None,
                frame_rate: 0.0,
                codec: String::new(),
                pixel_aspect_ratio: 1.0,
                start_time: None,
            },
            last_frame_time: None,
            start_time: std::time::Instant::now(),
            #[cfg(target_os = "macos")]
            vt_decoder: None,
            #[cfg(target_os = "macos")]
            h264_nal_length_size: 4,
            #[cfg(target_os = "macos")]
            is_avcc: false, // updated when VT session is created from catalog avcC
            #[cfg(target_os = "macos")]
            waiting_for_idr_after_error: false,
            #[cfg(target_os = "macos")]
            consecutive_decode_errors: 0,
            #[cfg(target_os = "macos")]
            skip_pframes_until_idr: false,
            #[cfg(target_os = "macos")]
            dpb_grace_started_at: None,
            #[cfg(target_os = "macos")]
            dpb_grace_dropped_frames: 0,
            #[cfg(target_os = "macos")]
            observed_idr_interval_us: None,
            #[cfg(target_os = "macos")]
            last_idr_pts_us: None,
            #[cfg(target_os = "macos")]
            require_clean_idr_after_recreate: false,
            #[cfg(target_os = "macos")]
            needs_session_recreation: true,
            #[cfg(target_os = "macos")]
            quiesce_until: None,
            #[cfg(target_os = "macos")]
            vt_session_count: 0,
            #[cfg(target_os = "macos")]
            forensic_enabled,
            #[cfg(target_os = "macos")]
            forensic_recent: VecDeque::with_capacity(8),
            #[cfg(target_os = "macos")]
            forensic_post_error: None,
            #[cfg(target_os = "macos")]
            forensic_seq: 0,
            #[cfg(target_os = "macos")]
            forensic_group_index: 0,
            #[cfg(target_os = "macos")]
            idr_wait_started_at: None,
            #[cfg(target_os = "macos")]
            idr_wait_dropped_frames: 0,
            #[cfg(target_os = "macos")]
            idr_wait_broken_keyframe_boundaries: 0,
            #[cfg(target_os = "macos")]
            idr_last_resubscribe_request_at: None,
            #[cfg(target_os = "macos")]
            required_drop_window_started_at: None,
            #[cfg(target_os = "macos")]
            required_drop_storms_in_window: 0,
        })
    }

    /// Async worker that handles MoQ connection, catalog fetching, and frame receipt.
    async fn run_moq_worker(
        shared: Arc<MoqSharedState>,
        url: MoqUrl,
        config: MoqDecoderConfig,
        frame_tx: Sender<MoqVideoFrame>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        super::moq::worker::run_moq_worker(shared, url, config, frame_tx, "macOS").await
    }

    /// Returns true if this URL is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        MoqUrl::is_moq_url(url)
    }

    /// Returns a handle to the MoQ shared state for producing stats snapshots.
    ///
    /// This handle can be stored separately (e.g. in VideoPlayer) and used to
    /// query MoQ-specific stats without needing a reference to the decoder.
    pub fn stats_handle(&self) -> MoqStatsHandle {
        MoqStatsHandle {
            shared: self.shared.clone(),
        }
    }

    /// Returns the current decoder state.
    pub fn decoder_state(&self) -> MoqDecoderState {
        *self.shared.state.lock()
    }

    /// Returns the error message if in error state.
    pub fn error_message(&self) -> Option<String> {
        self.shared.error_message.lock().clone()
    }

    /// Returns true if audio is muted.
    pub fn is_muted(&self) -> bool {
        self.audio_muted
    }

    /// Returns the current audio volume.
    pub fn volume(&self) -> f32 {
        self.audio_volume
    }

    /// Returns the audio track info, if available.
    pub fn audio_info(&self) -> Option<AudioTrackInfo> {
        self.shared.audio_info.lock().clone()
    }

    /// Checks if data contains an IDR frame (H.264 NAL type 5).
    ///
    /// Auto-detects Annex B (start codes) vs AVCC (length-prefixed) format.
    /// The hang crate's is_keyframe flag can be wrong when joining mid-stream,
    /// so we parse actual NAL types. Returns true if any NAL is type 5.
    #[allow(dead_code)]
    fn is_idr_frame(nal_data: &[u8], nal_length_size: usize) -> bool {
        let (types, count) = Self::find_nal_types(nal_data, nal_length_size);
        types[..count].contains(&5)
    }

    /// Gets the first NAL type from data for logging.
    #[allow(dead_code)]
    fn get_nal_type(nal_data: &[u8], nal_length_size: usize) -> u8 {
        let (types, count) = Self::find_nal_types(nal_data, nal_length_size);
        if count > 0 {
            types[0]
        } else {
            0
        }
    }

    /// Max NAL units we track per sample (AUD + SPS + PPS + SEI + IDR + spare).
    const MAX_NAL_TYPES: usize = 8;

    /// Returns all NAL types found in data, using known format context.
    ///
    /// When `is_avcc` is known from catalog/init context, use this to avoid
    /// the `data_is_annex_b()` heuristic which misclassifies AVCC frames
    /// whose first NAL is 256-511 bytes (length prefix `[0,0,1,X]` looks
    /// like an Annex B start code).
    pub(crate) fn find_nal_types_for_format(
        nal_data: &[u8],
        nal_length_size: usize,
        is_avcc: bool,
    ) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        if is_avcc {
            Self::find_nal_types_avcc(nal_data, nal_length_size)
        } else {
            Self::find_nal_types_annex_b(nal_data)
        }
    }

    /// Returns all NAL types found in data, auto-detecting Annex B vs AVCC format.
    ///
    /// WARNING: The heuristic `data_is_annex_b()` misclassifies AVCC frames whose
    /// first NAL is 256-511 bytes. Prefer `find_nal_types_for_format()` when the
    /// format is known from catalog context.
    pub(crate) fn find_nal_types(
        nal_data: &[u8],
        nal_length_size: usize,
    ) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        if Self::data_is_annex_b(nal_data) {
            Self::find_nal_types_annex_b(nal_data)
        } else {
            Self::find_nal_types_avcc(nal_data, nal_length_size)
        }
    }

    /// Check if data starts with Annex B start codes.
    ///
    /// WARNING: This is a heuristic that can produce false positives for AVCC data
    /// where the first NAL length is 256-511 bytes (prefix `[0,0,1,X]`).
    pub(crate) fn data_is_annex_b(data: &[u8]) -> bool {
        matches!(data, [0, 0, 0, 1, ..] | [0, 0, 1, ..])
    }

    /// Extract NAL types from Annex B bitstream (start-code delimited).
    fn find_nal_types_annex_b(data: &[u8]) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        let mut types = [0u8; Self::MAX_NAL_TYPES];
        let mut count = 0;
        let mut i = 0;
        while i < data.len() && count < Self::MAX_NAL_TYPES {
            let sc_len = if data.get(i..i + 4) == Some(&[0, 0, 0, 1]) {
                4
            } else if data.get(i..i + 3) == Some(&[0, 0, 1]) {
                3
            } else {
                i += 1;
                continue;
            };
            let nal_start = i + sc_len;
            if let Some(&byte) = data.get(nal_start) {
                types[count] = byte & 0x1F;
                count += 1;
            }
            i = nal_start + 1;
        }
        (types, count)
    }

    /// Extract NAL types from AVCC data (length-prefixed).
    fn find_nal_types_avcc(
        nal_data: &[u8],
        nal_length_size: usize,
    ) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        let mut types = [0u8; Self::MAX_NAL_TYPES];
        let mut count = 0;
        if !(1..=4).contains(&nal_length_size) {
            return (types, count);
        }
        let mut offset = 0usize;
        while offset + nal_length_size <= nal_data.len() && count < Self::MAX_NAL_TYPES {
            let len_bytes = match nal_data.get(offset..offset + nal_length_size) {
                Some(b) => b,
                None => break,
            };
            let mut nal_len = 0usize;
            for &byte in len_bytes {
                nal_len = (nal_len << 8) | byte as usize;
            }
            offset += nal_length_size;
            if nal_len == 0 || offset + nal_len > nal_data.len() {
                break;
            }
            types[count] = match nal_data.get(offset) {
                Some(&b) => b & 0x1F,
                None => break,
            };
            count += 1;
            offset += nal_len;
        }
        (types, count)
    }

    /// Returns true if frame should be skipped while waiting for an IDR resync.
    /// Only accepts real IDR (NAL type 5) as valid resync point. I-frames (NAL type 1
    /// with is_keyframe=true) cannot initialize a fresh VT session because they need
    /// existing DPB reference frames.
    #[cfg(target_os = "macos")]
    fn should_wait_for_idr(&self, moq_frame: &MoqVideoFrame) -> bool {
        if !self.waiting_for_idr_after_error {
            return false;
        }
        let (types, count) = Self::find_nal_types_for_format(
            &moq_frame.data,
            self.h264_nal_length_size,
            self.is_avcc,
        );
        let is_idr = types[..count].contains(&5);
        !is_idr // only real IDR can clear the resync gate
    }

    #[cfg(target_os = "macos")]
    fn fnv1a64(data: &[u8]) -> u64 {
        // Deterministic lightweight fingerprint for cross-run frame matching.
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in data {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    #[cfg(target_os = "macos")]
    fn record_forensic_sample(&mut self, moq_frame: &MoqVideoFrame, nal_types: &[u8]) {
        if !self.forensic_enabled {
            return;
        }

        if moq_frame.is_keyframe {
            self.forensic_group_index = self.forensic_group_index.saturating_add(1);
        }
        self.forensic_seq = self.forensic_seq.saturating_add(1);

        let mut first16 = [0u8; 16];
        let first16_len = moq_frame.data.len().min(16);
        first16[..first16_len].copy_from_slice(&moq_frame.data[..first16_len]);

        let mut nal_arr = [0u8; Self::MAX_NAL_TYPES];
        let nal_count = nal_types.len().min(Self::MAX_NAL_TYPES);
        nal_arr[..nal_count].copy_from_slice(&nal_types[..nal_count]);

        let sample = ForensicFrameSample {
            seq: self.forensic_seq,
            group_idx: self.forensic_group_index,
            pts_us: moq_frame.timestamp_us,
            size: moq_frame.data.len(),
            is_keyframe: moq_frame.is_keyframe,
            hash64: Self::fnv1a64(&moq_frame.data),
            first16,
            first16_len,
            nal_types: nal_arr,
            nal_count,
        };

        if self.forensic_recent.len() == 8 {
            let _ = self.forensic_recent.pop_front();
        }
        self.forensic_recent.push_back(sample);

        if let Some(mut post) = self.forensic_post_error {
            if sample.seq > post.trigger_seq && post.remaining_after > 0 {
                tracing::warn!(
                    "MoQ forensic post-error +{}: seq={}, group≈{}, pts={}us, size={}, keyframe={}, hash64={:016x}, nal_types={:?}, first16={:02x?}",
                    (4 - post.remaining_after) as usize,
                    sample.seq,
                    sample.group_idx,
                    sample.pts_us,
                    sample.size,
                    sample.is_keyframe,
                    sample.hash64,
                    &sample.nal_types[..sample.nal_count],
                    &sample.first16[..sample.first16_len]
                );
                post.remaining_after -= 1;
                if post.remaining_after == 0 {
                    tracing::warn!("MoQ forensic: post-error window capture complete");
                    self.forensic_post_error = None;
                } else {
                    self.forensic_post_error = Some(post);
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn log_forensic_error_window(&mut self) {
        if !self.forensic_enabled {
            return;
        }
        let Some(failing) = self.forensic_recent.back().copied() else {
            return;
        };

        tracing::warn!(
            "MoQ forensic trigger: seq={}, group≈{}, pts={}us, size={}, keyframe={}, hash64={:016x}, nal_types={:?}, first16={:02x?}",
            failing.seq,
            failing.group_idx,
            failing.pts_us,
            failing.size,
            failing.is_keyframe,
            failing.hash64,
            &failing.nal_types[..failing.nal_count],
            &failing.first16[..failing.first16_len]
        );

        for sample in self.forensic_recent.iter().rev().skip(1).take(3).rev() {
            tracing::warn!(
                "MoQ forensic pre-error: seq={}, group≈{}, pts={}us, size={}, keyframe={}, hash64={:016x}, nal_types={:?}, first16={:02x?}",
                sample.seq,
                sample.group_idx,
                sample.pts_us,
                sample.size,
                sample.is_keyframe,
                sample.hash64,
                &sample.nal_types[..sample.nal_count],
                &sample.first16[..sample.first16_len]
            );
        }

        self.forensic_post_error = Some(PostErrorCapture {
            trigger_seq: failing.seq,
            remaining_after: 3,
        });
    }

    #[cfg(target_os = "macos")]
    fn reset_idr_wait_tracking(&mut self) {
        self.idr_wait_started_at = None;
        self.idr_wait_dropped_frames = 0;
        self.idr_wait_broken_keyframe_boundaries = 0;
    }

    /// Ensure a VTDecompressionSession exists, creating one if needed.
    ///
    /// Tries catalog avcC description first, falls back to keyframe SPS/PPS
    /// extraction. Respects quiesce window after session destruction.
    #[cfg(target_os = "macos")]
    fn ensure_vt_session(&mut self, moq_frame: &MoqVideoFrame) -> Result<(), VideoError> {
        if self.vt_decoder.is_some() {
            return Ok(());
        }

        // Enforce non-blocking quiesce window after session destruction
        if let Some(quiesce_until) = self.quiesce_until {
            let now = std::time::Instant::now();
            if now < quiesce_until {
                return Err(VideoError::DecodeFailed(
                    "Waiting for VT quiesce".to_string(),
                ));
            }
            self.quiesce_until = None;
            tracing::info!("MoQ: VT quiesce window complete, creating new session");
        }

        let metadata = self.shared.metadata.lock().clone();

        // First, try to use codec description from catalog (avcC/hvcC box)
        if let Some(ref desc) = *self.shared.codec_description.lock() {
            match Self::parse_avcc_box(desc) {
                Ok((sps, pps, nal_length_size)) => {
                    let decoder = macos_vt::VTDecoder::new_h264(
                        &sps,
                        &pps,
                        metadata.width,
                        metadata.height,
                        true, // catalog avcC = AVCC format
                    )?;
                    self.vt_decoder = Some(decoder);
                    self.h264_nal_length_size = nal_length_size;
                    self.is_avcc = true;
                    self.vt_session_count += 1;
                    tracing::info!("MoQ: initialized VTDecoder session #{} from catalog avcC ({} bytes SPS, {} bytes PPS, NAL len size {})", self.vt_session_count, sps.len(), pps.len(), nal_length_size);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("MoQ: failed to parse avcC from catalog: {}", e);
                }
            }
        }

        // Fallback: extract SPS/PPS from keyframe
        if !moq_frame.is_keyframe {
            tracing::debug!("MoQ: waiting for keyframe to initialize VTDecoder");
            return Err(VideoError::DecodeFailed(
                "Waiting for keyframe with SPS/PPS".to_string(),
            ));
        }

        match Self::extract_h264_params(&moq_frame.data) {
            Ok((sps, pps)) => {
                let decoder = macos_vt::VTDecoder::new_h264(
                    &sps,
                    &pps,
                    metadata.width,
                    metadata.height,
                    false, // keyframe extraction = Annex B format
                )?;
                self.vt_decoder = Some(decoder);
                self.h264_nal_length_size = 4;
                self.vt_session_count += 1;
                tracing::info!(
                    "MoQ: initialized VTDecoder session #{} from keyframe SPS/PPS (Annex B)",
                    self.vt_session_count
                );
                Ok(())
            }
            Err(e) => {
                tracing::warn!("MoQ: failed to extract H.264 params: {}", e);
                Err(e)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn request_video_resubscribe_with_cooldown(&mut self) -> bool {
        const RESUBSCRIBE_REQUEST_COOLDOWN: Duration = Duration::from_millis(800);
        let now = std::time::Instant::now();
        if let Some(last) = self.idr_last_resubscribe_request_at {
            if now.saturating_duration_since(last) < RESUBSCRIBE_REQUEST_COOLDOWN {
                return false;
            }
        }
        if self
            .shared
            .request_video_resubscribe
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.idr_last_resubscribe_request_at = Some(now);
            return true;
        }
        false
    }

    #[cfg(target_os = "macos")]
    fn note_idr_wait_progress(&mut self, nal_type: u8, frame_len: usize, is_keyframe: bool) {
        // Keep recovery bounded: if a real IDR doesn't arrive quickly, force
        // re-subscribe so we can rejoin on a fresh group boundary.
        const IDR_WAIT_MAX: Duration = Duration::from_millis(1000);
        const IDR_WAIT_MAX_DROPS: u32 = 24;
        const BROKEN_KEYFRAME_BOUNDARY_THRESHOLD: u8 = 3;

        let start = match self.idr_wait_started_at {
            Some(start) => start,
            None => {
                let now = std::time::Instant::now();
                self.idr_wait_started_at = Some(now);
                now
            }
        };
        self.idr_wait_dropped_frames = self.idr_wait_dropped_frames.saturating_add(1);

        // Metadata keyframe without a real IDR is common in degenerate groups.
        // Do not immediately re-subscribe on a single boundary; require a
        // short sequence of broken boundaries to avoid churn.
        let broken_keyframe_boundary = is_keyframe && nal_type != 5;
        if broken_keyframe_boundary {
            self.idr_wait_broken_keyframe_boundaries =
                self.idr_wait_broken_keyframe_boundaries.saturating_add(1);
            if self.idr_wait_broken_keyframe_boundaries >= BROKEN_KEYFRAME_BOUNDARY_THRESHOLD
                && self.request_video_resubscribe_with_cooldown()
            {
                tracing::warn!(
                    "MoQ: repeated keyframe boundaries without IDR (count={}, nal_type={}, {} bytes) — requesting video re-subscribe",
                    self.idr_wait_broken_keyframe_boundaries,
                    nal_type,
                    frame_len
                );
                self.idr_wait_started_at = Some(std::time::Instant::now());
                self.idr_wait_dropped_frames = 0;
                self.idr_wait_broken_keyframe_boundaries = 0;
                return;
            }
        }

        let elapsed = start.elapsed();
        if elapsed < IDR_WAIT_MAX && self.idr_wait_dropped_frames < IDR_WAIT_MAX_DROPS {
            return;
        }

        if self.request_video_resubscribe_with_cooldown() {
            tracing::warn!(
                "MoQ: IDR starvation ({}ms, {} dropped, broken_keyframes={}, last_nal_type={}, {} bytes) — requesting video re-subscribe",
                elapsed.as_millis(),
                self.idr_wait_dropped_frames,
                self.idr_wait_broken_keyframe_boundaries,
                nal_type,
                frame_len
            );
        }

        // Keep reporting at a bounded cadence if starvation persists.
        self.idr_wait_started_at = Some(std::time::Instant::now());
        self.idr_wait_dropped_frames = 0;
        self.idr_wait_broken_keyframe_boundaries = 0;
    }

    #[cfg(target_os = "macos")]
    fn note_required_drop_storm_cycle(&mut self) -> u8 {
        const STORM_ESCALATION_WINDOW: Duration = Duration::from_millis(4500);

        let now = std::time::Instant::now();
        match self.required_drop_window_started_at {
            Some(start) if now.saturating_duration_since(start) <= STORM_ESCALATION_WINDOW => {
                self.required_drop_storms_in_window =
                    self.required_drop_storms_in_window.saturating_add(1);
            }
            _ => {
                self.required_drop_window_started_at = Some(now);
                self.required_drop_storms_in_window = 1;
            }
        }
        self.required_drop_storms_in_window
    }

    #[cfg(target_os = "macos")]
    fn note_real_idr_timestamp(&mut self, pts_us: u64) {
        // Ignore clearly invalid cadence deltas.
        const MIN_IDR_INTERVAL_US: u64 = 300_000;
        const MAX_IDR_INTERVAL_US: u64 = 8_000_000;

        if let Some(prev_pts) = self.last_idr_pts_us {
            let delta_us = pts_us.saturating_sub(prev_pts);
            if (MIN_IDR_INTERVAL_US..=MAX_IDR_INTERVAL_US).contains(&delta_us) {
                self.observed_idr_interval_us = Some(match self.observed_idr_interval_us {
                    // Smooth noisy group boundaries while still adapting.
                    Some(current) => ((current * 3) + delta_us) / 4,
                    None => delta_us,
                });
            }
        }
        self.last_idr_pts_us = Some(pts_us);
    }

    #[cfg(target_os = "macos")]
    fn dpb_grace_budget(&self) -> (Duration, u32) {
        // Base timeout from observed IDR cadence + margin for jitter/bursting.
        let observed_us = self.observed_idr_interval_us.unwrap_or(2_000_000);
        let timeout_us = observed_us.saturating_add(900_000);
        let timeout_us = timeout_us.clamp(2_500_000, 5_000_000);
        let timeout = Duration::from_micros(timeout_us);

        let fps = if self.cached_metadata.frame_rate.is_finite()
            && self.cached_metadata.frame_rate > 1.0
        {
            self.cached_metadata.frame_rate as f64
        } else {
            24.0
        };
        let max_drops = ((timeout.as_secs_f64() * fps).ceil() as u32)
            .saturating_add(8)
            .clamp(60, 180);

        (timeout, max_drops)
    }

    /// Decodes an encoded frame.
    ///
    /// On macOS, uses VTDecompressionSession for zero-copy hardware decoding.
    /// On other platforms, returns a placeholder (FFmpeg integration TODO).
    #[cfg(target_os = "macos")]
    fn decode_frame(&mut self, moq_frame: &MoqVideoFrame) -> Result<VideoFrame, VideoError> {
        // Initialize VTDecoder lazily (creates from catalog avcC or keyframe SPS/PPS)
        self.ensure_vt_session(moq_frame)?;

        // Check if we're waiting for IDR resync after a decode error
        if self.should_wait_for_idr(moq_frame) {
            let (t, c) = Self::find_nal_types_for_format(
                &moq_frame.data,
                self.h264_nal_length_size,
                self.is_avcc,
            );
            let nal_type = if c > 0 { t[0] } else { 0 };
            self.note_idr_wait_progress(nal_type, moq_frame.data.len(), moq_frame.is_keyframe);
            tracing::debug!(
                "MoQ: waiting for IDR resync after decode error (got NAL type {}, is_keyframe={}, {} bytes)",
                nal_type, moq_frame.is_keyframe, moq_frame.data.len()
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (got NAL type {})",
                nal_type
            )));
        }
        self.reset_idr_wait_tracking();

        // Check NAL types for diagnostics and keyframe validation.
        // Use format-aware parsing (self.is_avcc) to avoid data_is_annex_b() heuristic bug.
        let (nal_types_arr, nal_count) = Self::find_nal_types_for_format(
            &moq_frame.data,
            self.h264_nal_length_size,
            self.is_avcc,
        );
        let nal_types = &nal_types_arr[..nal_count];
        let is_idr = nal_types.contains(&5);
        if is_idr {
            self.note_real_idr_timestamp(moq_frame.timestamp_us);
        }
        self.record_forensic_sample(moq_frame, nal_types);

        // Bounded DPB grace: skip non-IDR frames after isolated VT callback error.
        // The skipped error frame leaves a stale DPB reference — subsequent
        // P-frames decode with status=0 but produce macroblock artifacts.
        // Bypasses note_idr_wait_progress() to avoid premature resubscribe.
        let (dpb_grace_timeout, dpb_grace_max_drops) = self.dpb_grace_budget();
        if self.skip_pframes_until_idr {
            if is_idr {
                let dpb_drops = self.dpb_grace_dropped_frames;
                // Clear DPB grace state
                self.skip_pframes_until_idr = false;
                self.dpb_grace_started_at = None;
                self.dpb_grace_dropped_frames = 0;
                // Reset error tracking — fresh session starts clean
                self.consecutive_decode_errors = 0;
                self.waiting_for_idr_after_error = false;
                self.reset_idr_wait_tracking();
                // Destroy corrupted VT session — IDR alone doesn't clear
                // VT's internal corruption state (r34: SS4 still pixelated
                // after IDR on same session, SS6 clean after recreation).
                self.vt_decoder = None;
                self.require_clean_idr_after_recreate = true;
                self.ensure_vt_session(moq_frame)?;
                tracing::info!(
                    "MoQ: DPB grace cleared by IDR — recreated VT session #{} ({} frames dropped, budget={}ms/{} drops)",
                    self.vt_session_count,
                    dpb_drops,
                    dpb_grace_timeout.as_millis(),
                    dpb_grace_max_drops
                );
                // Fall through to decode this IDR on the fresh session
            } else {
                let start = *self
                    .dpb_grace_started_at
                    .get_or_insert_with(std::time::Instant::now);
                self.dpb_grace_dropped_frames += 1;
                self.shared
                    .frame_stats
                    .dropped_dpb_grace
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let elapsed = start.elapsed();

                if elapsed > dpb_grace_timeout
                    || self.dpb_grace_dropped_frames > dpb_grace_max_drops
                {
                    // Grace expired — escalate to normal IDR-wait with resubscribe.
                    self.skip_pframes_until_idr = false;
                    let drops = self.dpb_grace_dropped_frames;
                    self.dpb_grace_started_at = None;
                    self.dpb_grace_dropped_frames = 0;
                    self.waiting_for_idr_after_error = true;
                    tracing::warn!(
                        "MoQ: DPB grace expired ({}ms, {} drops; budget={}ms/{} drops) — escalating to IDR-wait",
                        elapsed.as_millis(),
                        drops,
                        dpb_grace_timeout.as_millis(),
                        dpb_grace_max_drops
                    );
                }

                // Use distinct message that does NOT contain "Waiting for IDR frame"
                // to avoid double-counting in dropped_waiting_idr (line ~1729).
                return Err(VideoError::DecodeFailed(format!(
                    "DPB grace skip (NAL type {})",
                    nal_types.first().copied().unwrap_or(0)
                )));
            }
        }

        if self.require_clean_idr_after_recreate && !is_idr {
            tracing::debug!(
                "MoQ: post-recreate clean-IDR gate active (NAL type {}, {} bytes)",
                nal_types.first().copied().unwrap_or(0),
                moq_frame.data.len()
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (post-recreate gate, NAL type {})",
                nal_types.first().copied().unwrap_or(0)
            )));
        }

        // One-shot VT session recreation at the second IDR after startup.
        // The initial VT session can produce silently corrupted output (VT
        // status=0 but visibly pixelated). Recreating once at the next IDR
        // boundary clears the corruption. Requires at least 48 frames decoded
        // (one full group at 24fps) to ensure VT hardware has flushed, and
        // only triggers on real IDR (type 5), not I-frames (type 1).
        // A/B test: set to false to skip one-shot recreation and isolate
        // whether mid-session errors are content-dependent or lifecycle-dependent.
        // r31 result: HARMFUL — destroyed working session, caused 44 IDR drops,
        // FPS dropped to 8.0, errors increased from 2→8. Errors are content-dependent
        // (specific BBB P-frames at group position 3), not lifecycle-dependent.
        const ENABLE_ONESHOT_RECREATION: bool = false;
        if ENABLE_ONESHOT_RECREATION
            && self.needs_session_recreation
            && !self.waiting_for_idr_after_error
            && is_idr
        {
            if let Some(ref decoder) = self.vt_decoder {
                let prev_count = decoder.frame_count();
                // Only trigger after at least one full group (48 frames)
                // to give VT hardware time to fully initialize.
                if prev_count >= 48 {
                    self.needs_session_recreation = false;
                    // Drop triggers: WaitForAsync → Invalidate → CFRelease
                    self.vt_decoder = None;
                    self.quiesce_until =
                        Some(std::time::Instant::now() + std::time::Duration::from_millis(50));
                    tracing::info!(
                        "MoQ: one-shot VT recreation scheduled after quiesce window ({} bytes, {} frames on previous session)",
                        moq_frame.data.len(),
                        prev_count
                    );
                    return Err(VideoError::DecodeFailed(
                        "Waiting for VT quiesce".to_string(),
                    ));
                }
            }
        }

        if let Some(ref decoder) = self.vt_decoder {
            let frame_count = decoder.frame_count();

            // Log first 10 frames at INFO level for pipeline diagnostics
            if frame_count < 10 {
                let mut preview = [0u8; 20];
                let preview_len = moq_frame.data.len().min(20);
                preview[..preview_len].copy_from_slice(&moq_frame.data[..preview_len]);
                tracing::info!(
                    "MoQ decode frame #{}: {} bytes, is_keyframe={}, format={}, NAL types={:?}, is_idr={}, first 20 bytes={:02x?}",
                    frame_count,
                    moq_frame.data.len(),
                    moq_frame.is_keyframe,
                    if self.is_avcc { "AVCC" } else { "AnnexB" },
                    nal_types,
                    is_idr,
                    &preview[..preview_len],
                );
            }

            // First submitted frame must be a real IDR. Trusting metadata-only keyframe
            // flags (with non-IDR NAL 1 payloads) can poison VT reference state.
            if frame_count == 0 && !is_idr {
                // New VT sessions must begin on an IDR access unit.
                // Enter the normal IDR-wait path so this is treated as non-fatal,
                // counted as dropped-waiting-IDR, and eligible for bounded re-subscribe.
                self.waiting_for_idr_after_error = true;
                tracing::warn!(
                    "MoQ: dropping frame #{} — first frame is not IDR (NAL types={:?}, is_keyframe={}, {} bytes, format={})",
                    frame_count,
                    nal_types,
                    moq_frame.is_keyframe,
                    moq_frame.data.len(),
                    if self.is_avcc { "AVCC" } else { "AnnexB" },
                );
                return Err(VideoError::DecodeFailed(format!(
                    "Waiting for IDR frame (got NAL types {:?})",
                    nal_types
                )));
            }

            // Only clear IDR resync on a real IDR (NAL type 5). I-frames (NAL type 1
            // with is_keyframe=true) cannot initialize a fresh VT session — they need
            // existing reference frames in the DPB. Accepting I-frames here causes a
            // cascade: fresh session → decode fail → destroy → repeat.
            if self.waiting_for_idr_after_error && is_idr {
                self.waiting_for_idr_after_error = false;
                self.consecutive_decode_errors = 0;
                self.reset_idr_wait_tracking();
                tracing::info!(
                    "MoQ: received real IDR after error, will recreate VT session (session={})",
                    if self.vt_decoder.is_some() {
                        "exists"
                    } else {
                        "None"
                    }
                );
            }
        }

        // Treat metadata keyframe boundaries without a real IDR as discontinuities.
        // Decoding these as plain P-frames can poison visual output without emitting
        // callback errors. Instead, enter IDR wait and let the existing bounded
        // recovery/resubscribe machinery converge on a real IDR boundary.
        if moq_frame.is_keyframe && !is_idr {
            self.waiting_for_idr_after_error = true;
            tracing::warn!(
                "MoQ: keyframe metadata mismatch (is_keyframe=true but NAL types={:?}); entering IDR wait",
                nal_types
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (metadata keyframe without IDR, NAL types {:?})",
                nal_types
            )));
        }

        // Decode the frame using VTDecoder
        let vt_is_keyframe = is_idr;

        let decode_result = if let Some(ref mut decoder) = self.vt_decoder {
            decoder.decode_frame(&moq_frame.data, moq_frame.timestamp_us, vt_is_keyframe)
        } else {
            return Err(VideoError::DecodeFailed(
                "VTDecoder not initialized".to_string(),
            ));
        };

        match decode_result {
            Ok(Some(frame)) => {
                // Track decoded frame in shared stats
                let decoded_count = self
                    .shared
                    .frame_stats
                    .decoded
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;

                // Reset consecutive error counter on success
                self.consecutive_decode_errors = 0;
                if self.require_clean_idr_after_recreate && is_idr {
                    self.require_clean_idr_after_recreate = false;
                    tracing::info!(
                        "MoQ: post-recreate clean-IDR gate satisfied on session #{}",
                        self.vt_session_count
                    );
                }

                // Log first few successful decodes
                if decoded_count <= 5 {
                    tracing::info!(
                        "MoQ: VT decoded frame #{} (pts={}us)",
                        decoded_count,
                        moq_frame.timestamp_us
                    );
                }
                Ok(frame)
            }
            Ok(None) => Err(VideoError::DecodeFailed(
                "VTDecoder: no frame decoded (async?)".to_string(),
            )),
            Err(e) => {
                self.consecutive_decode_errors += 1;
                let total_errors = self
                    .shared
                    .frame_stats
                    .decode_errors
                    .load(std::sync::atomic::Ordering::Relaxed)
                    + 1;
                let error_text = e.to_string();
                let hard_vt_callback_failure = error_text.contains("VT decode callback error:");
                let required_drop_storm = error_text.contains("VT required-frame-drop storm");
                if hard_vt_callback_failure {
                    // Let consecutive_decode_errors increment naturally (+= 1 below).
                    // First 1-2 errors: soft skip (keep session, no resubscribe).
                    // At 3+ consecutive: hard reset + IDR resync + resubscribe.
                    // This avoids destroying a valid VT session for isolated P-frame
                    // errors (-12909) which are common on macOS with certain H.264
                    // content. The session resets to 0 on any successful decode.
                    if self.consecutive_decode_errors >= 2 {
                        // About to hit 3 — request resubscribe for fast IDR delivery
                        if self.request_video_resubscribe_with_cooldown() {
                            tracing::warn!(
                                "MoQ: VT callback failure #{} — requesting video re-subscribe",
                                self.consecutive_decode_errors + 1
                            );
                        }
                    }
                } else if required_drop_storm {
                    // Keep isolated storms soft, but don't loop forever on repeated
                    // storm cycles in a short window.
                    const STORM_ESCALATION_THRESHOLD: u8 = 3;
                    const STORM_ESCALATION_WINDOW_MS: u128 = 4500;
                    let storms_in_window = self.note_required_drop_storm_cycle();
                    if storms_in_window >= STORM_ESCALATION_THRESHOLD {
                        self.consecutive_decode_errors = 3;
                        self.required_drop_window_started_at = Some(std::time::Instant::now());
                        self.required_drop_storms_in_window = 0;
                        let requested_resubscribe = self.request_video_resubscribe_with_cooldown();
                        tracing::warn!(
                            "MoQ: required-frame-drop storm persisted ({} storms within ~{}ms) — escalating to VT session reset{}",
                            storms_in_window,
                            STORM_ESCALATION_WINDOW_MS,
                            if requested_resubscribe {
                                " + video re-subscribe request"
                            } else {
                                ""
                            }
                        );
                    } else {
                        self.consecutive_decode_errors = 0;
                        self.waiting_for_idr_after_error = false;
                        tracing::warn!(
                            "MoQ: required-frame-drop storm detected (window_count={}) — skipping frame without re-subscribe",
                            storms_in_window
                        );
                    }
                }
                self.log_forensic_error_window();

                // Log NAL header bytes for forensic analysis of failing frames
                let data = &moq_frame.data;
                let nal_header_hex = if data.len() >= 16 {
                    format!("{:02x?}", &data[..16])
                } else {
                    format!("{:02x?}", data)
                };
                // Parse first NAL type from AVCC (4-byte length prefix)
                let first_nal_type = if self.is_avcc {
                    match data.get(self.h264_nal_length_size) {
                        Some(&nal_byte) => {
                            format!(
                                "nal_type={} ({})",
                                nal_byte & 0x1f,
                                match nal_byte & 0x1f {
                                    1 => "non-IDR slice",
                                    5 => "IDR slice",
                                    6 => "SEI",
                                    7 => "SPS",
                                    8 => "PPS",
                                    _ => "other",
                                }
                            )
                        }
                        None => "unknown".to_string(),
                    }
                } else {
                    "unknown".to_string()
                };
                tracing::warn!(
                    "MoQ: failing frame forensics: size={}, is_keyframe={}, pts={}us, {}, header={}",
                    data.len(), moq_frame.is_keyframe, moq_frame.timestamp_us,
                    first_nal_type, nal_header_hex
                );

                if self.consecutive_decode_errors >= 3 {
                    // 3+ consecutive errors: destroy VT session and wait for IDR resync.
                    // prepare_for_idr_resync() only clears the output queue — VT's
                    // internal DPB retains stale reference frames. Full session
                    // recreation from catalog SPS/PPS is needed for clean recovery.
                    self.waiting_for_idr_after_error = true;
                    self.skip_pframes_until_idr = false;
                    self.dpb_grace_started_at = None;
                    self.dpb_grace_dropped_frames = 0;
                    self.require_clean_idr_after_recreate = true;
                    self.vt_decoder = None; // Drop: WaitForAsync → Invalidate → CFRelease
                    self.quiesce_until =
                        Some(std::time::Instant::now() + std::time::Duration::from_millis(50));
                    tracing::warn!(
                        "MoQ: VT decode error #{} (consecutive={}), destroyed session for IDR resync: {}",
                        total_errors, self.consecutive_decode_errors, e
                    );
                } else {
                    self.waiting_for_idr_after_error = false;

                    if hard_vt_callback_failure && self.consecutive_decode_errors == 1 {
                        // First isolated VT callback failure: the skipped P-frame was
                        // a DPB reference. Subsequent P-frames will decode successfully
                        // but produce macroblock artifacts. Enter bounded DPB grace
                        // to skip until next natural IDR resets the DPB.
                        self.skip_pframes_until_idr = true;
                        self.dpb_grace_started_at = None;
                        self.dpb_grace_dropped_frames = 0;
                        tracing::warn!(
                            "MoQ: VT callback error #{} (isolated), DPB grace until next IDR: {}",
                            total_errors,
                            e
                        );
                    } else {
                        tracing::warn!(
                            "MoQ: VT decode error #{} (consecutive={}), skipping frame without IDR gate: {}",
                            total_errors,
                            self.consecutive_decode_errors,
                            e
                        );
                    }
                }
                Err(e)
            }
        }
    }

    /// Decodes an encoded frame to YUV (non-macOS fallback).
    ///
    /// In a full implementation, this would use FFmpeg to decode H.264/H.265/AV1.
    /// For now, we return a placeholder frame to demonstrate the pipeline.
    #[cfg(not(target_os = "macos"))]
    fn decode_frame(&mut self, moq_frame: &MoqVideoFrame) -> Result<VideoFrame, VideoError> {
        let metadata = self.shared.metadata.lock();
        let width = metadata.width as usize;
        let height = metadata.height as usize;

        // Create a gray YUV420p frame (placeholder until FFmpeg integration)
        // TODO: Use FFmpeg to decode the actual H.264/H.265/AV1 NAL units
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        let y_plane = vec![128u8; y_size]; // Gray Y
        let u_plane = vec![128u8; uv_size]; // Neutral U
        let v_plane = vec![128u8; uv_size]; // Neutral V

        let cpu_frame = CpuFrame {
            format: PixelFormat::Yuv420p,
            width: metadata.width,
            height: metadata.height,
            planes: vec![
                Plane {
                    data: y_plane,
                    stride: width,
                },
                Plane {
                    data: u_plane,
                    stride: width / 2,
                },
                Plane {
                    data: v_plane,
                    stride: width / 2,
                },
            ],
        };

        // Calculate PTS from MoQ timestamp
        let pts = Duration::from_micros(moq_frame.timestamp_us);

        Ok(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame)))
    }

    /// Parses an avcC box (H.264 decoder configuration record) to extract SPS and PPS.
    ///
    /// avcC format:
    /// - 1 byte: version (always 1)
    /// - 1 byte: profile
    /// - 1 byte: compatibility
    /// - 1 byte: level
    /// - 1 byte: 0xFC | (NAL length size - 1)
    /// - 1 byte: 0xE0 | num_sps
    /// - For each SPS: 2 bytes length (big endian) + SPS data
    /// - 1 byte: num_pps
    /// - For each PPS: 2 bytes length (big endian) + PPS data
    pub(crate) fn parse_avcc_box(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, usize), VideoError> {
        if data.len() < 7 {
            return Err(VideoError::DecodeFailed("avcC too short".to_string()));
        }

        let version = data[0];
        if version != 1 {
            return Err(VideoError::DecodeFailed(format!(
                "Unsupported avcC version: {}",
                version
            )));
        }

        // Extract NAL length size from byte 4: (lengthSizeMinusOne & 0x03) + 1
        let nal_length_size = ((data[4] & 0x03) + 1) as usize;
        if !(1..=4).contains(&nal_length_size) {
            return Err(VideoError::DecodeFailed(format!(
                "Invalid avcC NAL length size: {}",
                nal_length_size
            )));
        }
        tracing::debug!("Parsed avcC: NAL length size {} bytes", nal_length_size);

        let mut offset = 5; // Skip version, profile, compatibility, level, NAL length size

        // Number of SPS (lower 5 bits)
        let num_sps = data[offset] & 0x1F;
        offset += 1;

        if num_sps == 0 {
            return Err(VideoError::DecodeFailed("No SPS in avcC".to_string()));
        }

        // Read first SPS
        if offset + 2 > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at SPS length".to_string(),
            ));
        }
        let sps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + sps_len > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at SPS data".to_string(),
            ));
        }
        let sps = data[offset..offset + sps_len].to_vec();
        offset += sps_len;

        // Skip remaining SPS if any
        for _ in 1..num_sps {
            if offset + 2 > data.len() {
                break;
            }
            let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2 + len;
        }

        // Number of PPS
        if offset >= data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at PPS count".to_string(),
            ));
        }
        let num_pps = data[offset];
        offset += 1;

        if num_pps == 0 {
            return Err(VideoError::DecodeFailed("No PPS in avcC".to_string()));
        }

        // Read first PPS
        if offset + 2 > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at PPS length".to_string(),
            ));
        }
        let pps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + pps_len > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at PPS data".to_string(),
            ));
        }
        let pps = data[offset..offset + pps_len].to_vec();

        tracing::debug!(
            "Parsed avcC: SPS {} bytes, PPS {} bytes, NAL length size {} bytes",
            sps.len(),
            pps.len(),
            nal_length_size
        );
        Ok((sps, pps, nal_length_size))
    }

    /// Extracts SPS and PPS NAL units from H.264 Annex B bitstream.
    ///
    /// SPS NAL type = 7, PPS NAL type = 8
    #[cfg(target_os = "macos")]
    fn extract_h264_params(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>), VideoError> {
        let mut sps: Option<Vec<u8>> = None;
        let mut pps: Option<Vec<u8>> = None;

        let mut i = 0;
        while i < data.len() {
            // Find start code (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
            let start_code_len = if i + 4 <= data.len()
                && data[i] == 0
                && data[i + 1] == 0
                && data[i + 2] == 0
                && data[i + 3] == 1
            {
                4
            } else if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                3
            } else {
                i += 1;
                continue;
            };

            let nal_start = i + start_code_len;
            if nal_start >= data.len() {
                break;
            }

            // Get NAL unit type (lower 5 bits of first byte)
            let nal_type = data[nal_start] & 0x1F;

            // Find end of this NAL unit
            let mut nal_end = data.len();
            for j in nal_start + 1..data.len().saturating_sub(2) {
                if data[j] == 0 && data[j + 1] == 0 {
                    if j + 2 < data.len() && data[j + 2] == 1 {
                        nal_end = j;
                        break;
                    }
                    if j + 3 < data.len() && data[j + 2] == 0 && data[j + 3] == 1 {
                        nal_end = j;
                        break;
                    }
                }
            }

            let nal_data = &data[nal_start..nal_end];

            match nal_type {
                7 => {
                    // SPS
                    sps = Some(nal_data.to_vec());
                    tracing::debug!("Found SPS: {} bytes", nal_data.len());
                }
                8 => {
                    // PPS
                    pps = Some(nal_data.to_vec());
                    tracing::debug!("Found PPS: {} bytes", nal_data.len());
                }
                _ => {}
            }

            i = nal_end;
        }

        match (sps, pps) {
            (Some(s), Some(p)) => Ok((s, p)),
            (None, _) => Err(VideoError::DecodeFailed(
                "No SPS found in keyframe".to_string(),
            )),
            (_, None) => Err(VideoError::DecodeFailed(
                "No PPS found in keyframe".to_string(),
            )),
        }
    }
}

impl Drop for MoqDecoder {
    fn drop(&mut self) {
        tracing::debug!("MoQ: MoqDecoder dropped (frame_rx closing)");
    }
}

impl VideoDecoderBackend for MoqDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Check if we've reached EOF
        if self.shared.eof_reached.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // Check for errors
        let state = *self.shared.state.lock();
        if state == MoqDecoderState::Error {
            return Err(VideoError::DecodeFailed(
                self.error_message()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        // Sync cached metadata from shared state (safe copy under lock)
        {
            let shared_metadata = self.shared.metadata.lock();
            if shared_metadata.width != self.cached_metadata.width
                || shared_metadata.height != self.cached_metadata.height
            {
                self.cached_metadata = shared_metadata.clone();
            }
        }

        // Try to receive a frame (non-blocking)
        match self.frame_rx.try_recv() {
            Ok(moq_frame) => {
                // Track frame submitted to decoder
                self.shared
                    .frame_stats
                    .submitted_to_decoder
                    .fetch_add(1, Ordering::Relaxed);

                // Decode the frame
                match self.decode_frame(&moq_frame) {
                    Ok(frame) => {
                        // Track frame rendered
                        self.shared
                            .frame_stats
                            .rendered
                            .fetch_add(1, Ordering::Relaxed);
                        Ok(Some(frame))
                    }
                    Err(VideoError::DecodeFailed(msg))
                        if msg.contains("Waiting for keyframe")
                            || msg.contains("Waiting for IDR frame")
                            || msg.contains("Waiting for VT quiesce")
                            || msg.contains("DPB grace skip")
                            || msg.contains("no frame decoded") =>
                    {
                        // Track frames dropped waiting for IDR (but NOT DPB grace —
                        // those are already counted via dropped_dpb_grace)
                        if msg.contains("Waiting for IDR") {
                            self.shared
                                .frame_stats
                                .dropped_waiting_idr
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        // Not an error, just need to wait for keyframe or async decode
                        Ok(None)
                    }
                    Err(e) => {
                        // Track decode errors
                        self.shared
                            .frame_stats
                            .decode_errors
                            .fetch_add(1, Ordering::Relaxed);
                        Err(e)
                    }
                }
            }
            Err(async_channel::TryRecvError::Empty) => {
                // No frame available yet
                Ok(None)
            }
            Err(async_channel::TryRecvError::Closed) => {
                // Channel closed, stream ended
                tracing::info!(
                    "MoQ: frame_tx sender dropped (worker ended or shutdown), setting eof"
                );
                self.shared.eof_reached.store(true, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        // Live streams don't support seeking
        Err(VideoError::SeekFailed(
            "Seeking is not supported on live MoQ streams".to_string(),
        ))
    }

    fn metadata(&self) -> &VideoMetadata {
        // Return the locally cached metadata (safe, no lock needed)
        &self.cached_metadata
    }

    fn duration(&self) -> Option<Duration> {
        // Live streams have no duration
        None
    }

    fn is_eof(&self) -> bool {
        self.shared.eof_reached.load(Ordering::Relaxed)
    }

    fn buffering_percent(&self) -> i32 {
        self.shared.buffering_percent.load(Ordering::Relaxed)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.active_hw_type
    }

    fn handles_audio_internally(&self) -> bool {
        self.shared
            .audio
            .internal_audio_ready
            .load(Ordering::Relaxed)
    }

    fn audio_handle(&self) -> Option<super::audio::AudioHandle> {
        self.shared.audio.moq_audio_handle.lock().clone()
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_muted = muted;
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        self.audio_volume = volume.clamp(0.0, 1.0);
        Ok(())
    }
}

// =============================================================================
// macOS VTDecompressionSession Implementation
// =============================================================================
//
// This module provides zero-copy video decoding on macOS using VideoToolbox.
// NAL units from MoQ are fed directly to VTDecompressionSession, which outputs
// CVPixelBuffers backed by IOSurface for zero-copy GPU rendering.

#[cfg(target_os = "macos")]
mod macos_vt {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2_core_video::{
        kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
        kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA,
    };
    use objc2_foundation::{NSCopying, NSMutableDictionary, NSNumber, NSString};
    use parking_lot::Mutex as ParkingMutex;
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

    // ==========================================================================
    // Raw FFI declarations for VideoToolbox and CoreMedia
    //
    // Using raw FFI because objc2 0.6 generated bindings changed from raw pointers
    // to Option<&T>/NonNull which requires significant code restructuring.
    // Raw FFI is more stable across objc2 versions.
    // ==========================================================================

    /// CMTime structure (matches CoreMedia layout)
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct CMTime {
        pub value: i64,
        pub timescale: i32,
        pub flags: u32,
        pub epoch: i64,
    }

    impl CMTime {
        pub const fn invalid() -> Self {
            Self {
                value: 0,
                timescale: 0,
                flags: 0,
                epoch: 0,
            }
        }

        pub const fn new(value: i64, timescale: i32) -> Self {
            Self {
                value,
                timescale,
                flags: 1,
                epoch: 0,
            } // flags=1 is kCMTimeFlags_Valid
        }
    }

    /// CMSampleTimingInfo structure
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct CMSampleTimingInfo {
        pub duration: CMTime,
        pub presentation_time_stamp: CMTime,
        pub decode_time_stamp: CMTime,
    }

    /// VTDecompressionOutputCallback function pointer type.
    /// Field names match Apple's C API naming convention.
    #[allow(non_snake_case)]
    pub type VTDecompressionOutputCallback = extern "C" fn(
        decompressionOutputRefCon: *mut c_void,
        sourceFrameRefCon: *mut c_void,
        status: i32,
        infoFlags: u32,
        imageBuffer: *mut c_void,
        presentationTimeStamp: CMTime,
        presentationDuration: CMTime,
    );

    /// VTDecompressionOutputCallbackRecord structure.
    /// Field names match Apple's C API naming convention.
    #[repr(C)]
    #[allow(non_snake_case)]
    pub struct VTDecompressionOutputCallbackRecord {
        pub decompressionOutputCallback: Option<VTDecompressionOutputCallback>,
        pub decompressionOutputRefCon: *mut c_void,
    }

    // Raw FFI declarations - split by framework for proper linking

    // CoreVideo framework
    #[link(name = "CoreVideo", kind = "framework")]
    extern "C" {
        fn CVPixelBufferGetIOSurface(pixelBuffer: *const c_void) -> *mut c_void;
        fn CVPixelBufferGetWidth(pixelBuffer: *const c_void) -> usize;
        fn CVPixelBufferGetHeight(pixelBuffer: *const c_void) -> usize;
    }

    // CoreMedia framework
    #[link(name = "CoreMedia", kind = "framework")]
    extern "C" {
        fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
            allocator: *const c_void,
            parameterSetCount: usize,
            parameterSetPointers: *const *const u8,
            parameterSetSizes: *const usize,
            NALUnitHeaderLength: i32,
            formatDescriptionOut: *mut *mut c_void,
        ) -> i32;

        // Reserved for H.265 support
        #[allow(dead_code)]
        fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            allocator: *const c_void,
            parameterSetCount: usize,
            parameterSetPointers: *const *const u8,
            parameterSetSizes: *const usize,
            NALUnitHeaderLength: i32,
            extensions: *const c_void,
            formatDescriptionOut: *mut *mut c_void,
        ) -> i32;

        fn CMBlockBufferCreateWithMemoryBlock(
            structureAllocator: *const c_void,
            memoryBlock: *mut c_void,
            blockLength: usize,
            blockAllocator: *const c_void,
            customBlockSource: *const c_void,
            offsetToData: usize,
            dataLength: usize,
            flags: u32,
            blockBufferOut: *mut *mut c_void,
        ) -> i32;

        fn CMSampleBufferCreate(
            allocator: *const c_void,
            dataBuffer: *mut c_void,
            dataReady: bool,
            makeDataReadyCallback: *const c_void,
            makeDataReadyRefcon: *mut c_void,
            formatDescription: *mut c_void,
            numSamples: i64,
            numSampleTimingEntries: i64,
            sampleTimingArray: *const CMSampleTimingInfo,
            numSampleSizeEntries: i64,
            sampleSizeArray: *const usize,
            sampleBufferOut: *mut *mut c_void,
        ) -> i32;

        fn CMSampleBufferGetSampleAttachmentsArray(
            sbuf: *mut c_void,
            createIfNecessary: bool,
        ) -> *const c_void;

        /// kCMSampleAttachmentKey_DependsOnOthers: kCFBooleanFalse = sync sample (keyframe)
        static kCMSampleAttachmentKey_DependsOnOthers: *const c_void;
        static kCMSampleAttachmentKey_NotSync: *const c_void;
    }

    // VideoToolbox framework
    #[link(name = "VideoToolbox", kind = "framework")]
    extern "C" {
        fn VTDecompressionSessionCreate(
            allocator: *const c_void,
            videoFormatDescription: *mut c_void,
            videoDecoderSpecification: *const c_void,
            destinationImageBufferAttributes: *const c_void,
            outputCallback: *const VTDecompressionOutputCallbackRecord,
            decompressionSessionOut: *mut *mut c_void,
        ) -> i32;

        fn VTDecompressionSessionDecodeFrame(
            session: *mut c_void,
            sampleBuffer: *mut c_void,
            decodeFlags: u32,
            sourceFrameRefCon: *mut c_void,
            infoFlagsOut: *mut u32,
        ) -> i32;

        fn VTDecompressionSessionWaitForAsynchronousFrames(session: *mut c_void) -> i32;

        fn VTDecompressionSessionInvalidate(session: *mut c_void);
    }

    // CoreFoundation framework
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRelease(cf: *const c_void);
        fn CFRetain(cf: *const c_void) -> *const c_void;
        fn CFArrayGetValueAtIndex(theArray: *const c_void, idx: isize) -> *const c_void;
        fn CFDictionarySetValue(theDict: *const c_void, key: *const c_void, value: *const c_void);
        /// Null allocator - performs no allocation/deallocation.
        /// Use this for caller-owned memory passed to CM functions.
        static kCFAllocatorNull: *const c_void;
        static kCFBooleanTrue: *const c_void;
        static kCFBooleanFalse: *const c_void;
    }

    /// Wrapper for CVPixelBuffer (raw pointer) that releases on drop.
    struct PixelBufferWrapper(*mut c_void);

    impl PixelBufferWrapper {
        /// Retains and wraps a CVPixelBuffer pointer.
        unsafe fn retain(ptr: *mut c_void) -> Self {
            if !ptr.is_null() {
                CFRetain(ptr);
            }
            Self(ptr)
        }
    }

    impl Drop for PixelBufferWrapper {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CFRelease(self.0) };
            }
        }
    }

    impl std::fmt::Debug for PixelBufferWrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("PixelBufferWrapper")
                .field("ptr", &self.0)
                .finish()
        }
    }

    // SAFETY: CVPixelBuffer is safe to send between threads because:
    // - The pixel data is immutable after creation
    // - CoreFoundation reference counting is thread-safe
    // - The IOSurface backing (if any) is also thread-safe
    unsafe impl Send for PixelBufferWrapper {}
    unsafe impl Sync for PixelBufferWrapper {}

    /// A decoded frame from VTDecompressionSession, ready for rendering.
    struct DecodedVTFrame {
        /// Presentation timestamp in microseconds
        pts_us: u64,
        /// The decoded CVPixelBuffer (retained)
        pixel_buffer: PixelBufferWrapper,
        /// True when callback had kVTDecodeInfo_RequiredFrameDropped.
        required_frame_dropped: bool,
    }

    /// Shared state for decoder callback to push decoded frames.
    struct VTCallbackState {
        /// Queue of decoded frames (protected by mutex)
        decoded_frames: ParkingMutex<VecDeque<DecodedVTFrame>>,
        /// Error flag set by callback on decode failure
        decode_error: AtomicBool,
        /// OSStatus from last callback error (0 = no error)
        decode_error_status: AtomicI32,
        /// Frame counter for debugging
        frame_count: AtomicU32,
    }

    impl VTCallbackState {
        fn new() -> Self {
            Self {
                decoded_frames: ParkingMutex::new(VecDeque::with_capacity(8)),
                decode_error: AtomicBool::new(false),
                decode_error_status: AtomicI32::new(0),
                frame_count: AtomicU32::new(0),
            }
        }
    }

    /// VTDecompressionSession wrapper for zero-copy H.264/H.265 decoding.
    ///
    /// Uses raw FFI pointers for VideoToolbox interop. The session and format_desc
    /// are retained on creation and released on drop.
    pub struct VTDecoder {
        /// The VideoToolbox decompression session (retained)
        session: *mut c_void,
        /// CMFormatDescription for the video stream (retained)
        format_desc: *mut c_void,
        /// Shared callback state (Arc for callback lifetime)
        callback_state: Arc<VTCallbackState>,
        /// Video dimensions (reserved for future use in frame validation)
        #[allow(dead_code)]
        pub width: u32,
        /// Video dimensions (reserved for future use in frame validation)
        #[allow(dead_code)]
        pub height: u32,
        /// Codec type (H.264 or H.265, reserved for H.265 support)
        #[allow(dead_code)]
        codec: VTCodec,
        /// True if NAL data is AVCC format (length-prefixed), false for Annex B (start codes).
        /// Set from catalog info to avoid heuristic detection that misclassifies
        /// AVCC frames with 256-511 byte NALs (length prefix [0,0,1,X] looks like Annex B).
        is_avcc: bool,
        /// Consecutive callbacks flagged with RequiredFrameDropped.
        required_drop_streak: u32,
    }

    /// If this many consecutive decoded callbacks require frame drops, treat as
    /// a corruption storm and force session recovery.
    const VT_REQUIRED_DROP_STORM_THRESHOLD: u32 = 36;

    // SAFETY: VTDecompressionSession is designed for multi-threaded use.
    // The session pointer is only accessed through synchronized methods.
    // The callback_state uses interior mutability with proper synchronization.
    unsafe impl Send for VTDecoder {}
    unsafe impl Sync for VTDecoder {}

    /// Supported codecs for VTDecompressionSession.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(dead_code)]
    pub enum VTCodec {
        /// H.264/AVC codec
        H264,
        /// H.265/HEVC codec (reserved for future support)
        H265,
    }

    impl VTDecoder {
        /// Creates a new VTDecoder for H.264 with the given SPS/PPS NAL units.
        ///
        /// # Arguments
        /// * `sps` - Sequence Parameter Set NAL unit (without start code)
        /// * `pps` - Picture Parameter Set NAL unit (without start code)
        /// * `width` - Video width (hint, may be overridden by SPS)
        /// * `height` - Video height (hint, may be overridden by SPS)
        pub fn new_h264(
            sps: &[u8],
            pps: &[u8],
            width: u32,
            height: u32,
            is_avcc: bool,
        ) -> Result<Self, VideoError> {
            tracing::info!(
                "VTDecoder: Creating H.264 decoder {}x{} (SPS: {} bytes, PPS: {} bytes, avcc={})",
                width,
                height,
                sps.len(),
                pps.len(),
                is_avcc,
            );

            // Create CMFormatDescription from SPS/PPS
            let format_desc = Self::create_h264_format_description(sps, pps)?;

            // Create decoder with format description
            Self::create_decoder(format_desc, width, height, VTCodec::H264, is_avcc)
        }

        /// Creates a new VTDecoder for H.265 with the given VPS/SPS/PPS NAL units.
        ///
        /// # Arguments
        /// * `vps` - Video Parameter Set NAL unit (without start code)
        /// * `sps` - Sequence Parameter Set NAL unit (without start code)
        /// * `pps` - Picture Parameter Set NAL unit (without start code)
        /// * `width` - Video width
        /// * `height` - Video height
        #[allow(dead_code)]
        pub fn new_h265(
            vps: &[u8],
            sps: &[u8],
            pps: &[u8],
            width: u32,
            height: u32,
            is_avcc: bool,
        ) -> Result<Self, VideoError> {
            tracing::info!(
                "VTDecoder: Creating H.265 decoder {}x{} (VPS: {} bytes, SPS: {} bytes, PPS: {} bytes, avcc={})",
                width,
                height,
                vps.len(),
                sps.len(),
                pps.len(),
                is_avcc,
            );

            // Create CMFormatDescription from VPS/SPS/PPS
            let format_desc = Self::create_h265_format_description(vps, sps, pps)?;

            // Create decoder with format description
            Self::create_decoder(format_desc, width, height, VTCodec::H265, is_avcc)
        }

        /// Creates CMVideoFormatDescription for H.264 from SPS/PPS.
        /// Returns a retained pointer that must be released with CFRelease.
        fn create_h264_format_description(
            sps: &[u8],
            pps: &[u8],
        ) -> Result<*mut c_void, VideoError> {
            // Prepare parameter set pointers and sizes
            let parameter_sets: [*const u8; 2] = [sps.as_ptr(), pps.as_ptr()];
            let parameter_set_sizes: [usize; 2] = [sps.len(), pps.len()];

            let mut format_desc_ptr: *mut c_void = ptr::null_mut();

            // Use 4-byte NAL length prefix (standard for Annex B to AVCC conversion)
            let nal_unit_header_length: i32 = 4;

            let status = unsafe {
                CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    ptr::null(),                  // allocator (NULL = default)
                    2,                            // parameter set count
                    parameter_sets.as_ptr(),      // parameter set pointers
                    parameter_set_sizes.as_ptr(), // parameter set sizes
                    nal_unit_header_length,       // NAL unit header length
                    &mut format_desc_ptr,         // output format description
                )
            };

            if status != 0 || format_desc_ptr.is_null() {
                return Err(VideoError::DecoderInit(format!(
                    "Failed to create H.264 format description: OSStatus {}",
                    status
                )));
            }

            Ok(format_desc_ptr)
        }

        /// Creates CMVideoFormatDescription for H.265 from VPS/SPS/PPS.
        /// Returns a retained pointer that must be released with CFRelease.
        #[allow(dead_code)]
        fn create_h265_format_description(
            vps: &[u8],
            sps: &[u8],
            pps: &[u8],
        ) -> Result<*mut c_void, VideoError> {
            // Prepare parameter set pointers and sizes (VPS, SPS, PPS order)
            let parameter_sets: [*const u8; 3] = [vps.as_ptr(), sps.as_ptr(), pps.as_ptr()];
            let parameter_set_sizes: [usize; 3] = [vps.len(), sps.len(), pps.len()];

            let mut format_desc_ptr: *mut c_void = ptr::null_mut();

            // Use 4-byte NAL length prefix
            let nal_unit_header_length: i32 = 4;

            let status = unsafe {
                CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                    ptr::null(),                  // allocator (NULL = default)
                    3,                            // parameter set count
                    parameter_sets.as_ptr(),      // parameter set pointers
                    parameter_set_sizes.as_ptr(), // parameter set sizes
                    nal_unit_header_length,       // NAL unit header length
                    ptr::null(),                  // extensions (NULL for default)
                    &mut format_desc_ptr,         // output format description
                )
            };

            if status != 0 || format_desc_ptr.is_null() {
                return Err(VideoError::DecoderInit(format!(
                    "Failed to create H.265 format description: OSStatus {}",
                    status
                )));
            }

            Ok(format_desc_ptr)
        }

        /// Creates the VTDecompressionSession with IOSurface-compatible output.
        fn create_decoder(
            format_desc: *mut c_void,
            width: u32,
            height: u32,
            codec: VTCodec,
            is_avcc: bool,
        ) -> Result<Self, VideoError> {
            // Create output pixel buffer attributes for IOSurface + Metal compatibility
            let destination_attributes = Self::create_output_attributes()?;

            // Create callback state for receiving decoded frames
            let callback_state = Arc::new(VTCallbackState::new());

            // Create the decompression session
            let session =
                Self::create_session(format_desc, &destination_attributes, &callback_state)?;

            tracing::info!(
                "VTDecoder: Created {:?} session with IOSurface+Metal output (avcc={})",
                codec,
                is_avcc,
            );

            Ok(Self {
                session,
                format_desc,
                callback_state,
                width,
                height,
                codec,
                is_avcc,
                required_drop_streak: 0,
            })
        }

        /// Creates pixel buffer attributes dictionary for IOSurface + Metal output.
        ///
        /// This uses NSMutableDictionary (same pattern as macos_video.rs) to configure:
        /// - kCVPixelBufferPixelFormatTypeKey = kCVPixelFormatType_32BGRA
        /// - kCVPixelBufferIOSurfacePropertiesKey = {} (empty dict enables IOSurface)
        /// - kCVPixelBufferMetalCompatibilityKey = true
        fn create_output_attributes(
        ) -> Result<Retained<NSMutableDictionary<NSString, AnyObject>>, VideoError> {
            unsafe {
                let dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                    NSMutableDictionary::new();

                // Set pixel format to BGRA (matches MacOSVideoDecoder)
                let key_cfstring = kCVPixelBufferPixelFormatTypeKey;
                let pixel_format = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);

                let key_ptr = key_cfstring as *const _ as *const NSString;
                let key: &NSString = &*key_ptr;
                let key_copying: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(key);

                let value_ptr = Retained::as_ptr(&pixel_format) as *mut AnyObject;
                let value: &AnyObject = &*value_ptr;

                dict.setObject_forKey(value, key_copying);

                // Set IOSurface properties (empty dictionary enables IOSurface backing)
                let iosurface_key_cfstring = kCVPixelBufferIOSurfacePropertiesKey;
                let iosurface_key_ptr = iosurface_key_cfstring as *const _ as *const NSString;
                let iosurface_key: &NSString = &*iosurface_key_ptr;
                let iosurface_key_copying: &ProtocolObject<dyn NSCopying> =
                    ProtocolObject::from_ref(iosurface_key);
                let iosurface_props: Retained<NSMutableDictionary<NSString, AnyObject>> =
                    NSMutableDictionary::new();
                let iosurface_value_ptr = Retained::as_ptr(&iosurface_props) as *mut AnyObject;
                let iosurface_value: &AnyObject = &*iosurface_value_ptr;
                dict.setObject_forKey(iosurface_value, iosurface_key_copying);

                // Set Metal compatibility
                let metal_key_cfstring = kCVPixelBufferMetalCompatibilityKey;
                let metal_key_ptr = metal_key_cfstring as *const _ as *const NSString;
                let metal_key: &NSString = &*metal_key_ptr;
                let metal_key_copying: &ProtocolObject<dyn NSCopying> =
                    ProtocolObject::from_ref(metal_key);
                let metal_value = NSNumber::numberWithBool(true);
                let metal_value_ptr = Retained::as_ptr(&metal_value) as *mut AnyObject;
                let metal_value: &AnyObject = &*metal_value_ptr;
                dict.setObject_forKey(metal_value, metal_key_copying);

                tracing::debug!(
                    "VTDecoder: Configured output with IOSurface + Metal compatibility"
                );

                Ok(dict)
            }
        }

        /// Creates the VTDecompressionSession with callback.
        /// Returns a retained session pointer.
        fn create_session(
            format_desc: *mut c_void,
            destination_attributes: &NSMutableDictionary<NSString, AnyObject>,
            callback_state: &Arc<VTCallbackState>,
        ) -> Result<*mut c_void, VideoError> {
            // Create callback record with our decompression output handler
            // The callback_state is passed as refcon (reference context) to the callback
            let callback_state_ptr = Arc::into_raw(Arc::clone(callback_state)) as *mut c_void;

            let callback_record = VTDecompressionOutputCallbackRecord {
                decompressionOutputCallback: Some(vt_decode_callback),
                decompressionOutputRefCon: callback_state_ptr,
            };

            let mut session_ptr: *mut c_void = ptr::null_mut();

            // Get raw pointer to the NSDictionary for FFI
            let dest_attrs_ptr = destination_attributes
                as *const NSMutableDictionary<NSString, AnyObject>
                as *const c_void;

            let status = unsafe {
                VTDecompressionSessionCreate(
                    ptr::null(),      // allocator
                    format_desc,      // video format description
                    ptr::null(),      // decoder specification (NULL = auto)
                    dest_attrs_ptr,   // destination attributes
                    &callback_record, // output callback record
                    &mut session_ptr, // output session
                )
            };

            if status != 0 || session_ptr.is_null() {
                // Clean up the Arc we created for the callback
                unsafe { Arc::from_raw(callback_state_ptr as *const VTCallbackState) };
                return Err(VideoError::DecoderInit(format!(
                    "VTDecompressionSessionCreate failed: OSStatus {}",
                    status
                )));
            }

            Ok(session_ptr)
        }

        /// Decodes a frame from encoded NAL unit data.
        ///
        /// The NAL unit can be in:
        /// - AVCC format (length-prefixed NALs) - used by MoQ/hang
        /// - Annex B format (start code prefixed) - used by raw H.264 streams
        ///
        /// Returns the decoded VideoFrame with IOSurface-backed GPU surface.
        pub fn decode_frame(
            &mut self,
            nal_data: &[u8],
            pts_us: u64,
            is_keyframe: bool,
        ) -> Result<Option<VideoFrame>, VideoError> {
            // Log first few bytes to debug format
            let preview: Vec<u8> = nal_data.iter().take(16).copied().collect();
            tracing::debug!(
                "VTDecoder::decode_frame: {} bytes, keyframe={}, first 16 bytes: {:02x?}",
                nal_data.len(),
                is_keyframe,
                preview
            );

            // Use known format from catalog/init context instead of heuristic detection.
            // The heuristic is_avcc_format() misclassifies AVCC frames with 256-511 byte
            // NAL lengths: the prefix [0x00,0x00,0x01,X] looks like an Annex B start code.
            let is_avcc = self.is_avcc;
            tracing::debug!(
                "VTDecoder: detected format: {}",
                if is_avcc { "AVCC" } else { "Annex B" }
            );

            let avcc_data = if is_avcc {
                // Already in AVCC format, use as-is
                tracing::debug!("VTDecoder: copying {} bytes AVCC data", nal_data.len());
                nal_data.to_vec()
            } else {
                // Annex B format, convert to AVCC
                let converted = Self::annex_b_to_avcc(nal_data);
                tracing::debug!(
                    "VTDecoder: converted Annex B to AVCC: {} -> {} bytes",
                    nal_data.len(),
                    converted.len()
                );
                converted
            };

            tracing::debug!("VTDecoder: avcc_data ready, {} bytes", avcc_data.len());

            if avcc_data.is_empty() {
                return Err(VideoError::DecodeFailed(
                    "Empty AVCC data after conversion".to_string(),
                ));
            }

            tracing::debug!("VTDecoder: creating CMBlockBuffer");

            // Create CMBlockBuffer from the AVCC data
            let mut block_buffer_ptr: *mut c_void = ptr::null_mut();

            let status = unsafe {
                CMBlockBufferCreateWithMemoryBlock(
                    ptr::null(),                       // allocator for CMBlockBuffer structure
                    avcc_data.as_ptr() as *mut c_void, // memory block
                    avcc_data.len(),                   // block length
                    kCFAllocatorNull, // block allocator: kCFAllocatorNull = caller owns memory, don't free
                    ptr::null(),      // custom block source
                    0,                // offset into block
                    avcc_data.len(),  // data length
                    0,                // flags
                    &mut block_buffer_ptr, // output block buffer
                )
            };

            if status != 0 || block_buffer_ptr.is_null() {
                return Err(VideoError::DecodeFailed(format!(
                    "CMBlockBufferCreate failed: OSStatus {}",
                    status
                )));
            }

            tracing::debug!("VTDecoder: CMBlockBuffer created successfully");

            // Create CMSampleBuffer from block buffer
            let mut sample_buffer_ptr: *mut c_void = ptr::null_mut();

            // Create timing info for this frame
            let pts = CMTime::new(pts_us as i64, 1_000_000); // microseconds
            tracing::debug!("VTDecoder: creating timing info, pts_us={}", pts_us);

            let timing_info = CMSampleTimingInfo {
                duration: CMTime::invalid(),
                presentation_time_stamp: pts,
                decode_time_stamp: CMTime::invalid(),
            };

            let sample_size = avcc_data.len();
            tracing::debug!(
                "VTDecoder: creating CMSampleBuffer, sample_size={}, format_desc={:?}",
                sample_size,
                self.format_desc
            );

            let status = unsafe {
                CMSampleBufferCreate(
                    ptr::null(),            // allocator
                    block_buffer_ptr,       // data buffer
                    true,                   // data ready
                    ptr::null(),            // make data ready callback
                    ptr::null_mut(),        // make data ready refcon
                    self.format_desc,       // format description
                    1,                      // num samples
                    1,                      // num sample timing entries
                    &timing_info,           // sample timing array
                    1,                      // num sample size entries
                    &sample_size,           // sample size array
                    &mut sample_buffer_ptr, // output sample buffer
                )
            };
            tracing::debug!("VTDecoder: CMSampleBufferCreate returned status={}", status);

            // Release block buffer (sample buffer retains it if needed)
            unsafe { CFRelease(block_buffer_ptr) };
            tracing::debug!("VTDecoder: released block buffer");

            if status != 0 || sample_buffer_ptr.is_null() {
                return Err(VideoError::DecodeFailed(format!(
                    "CMSampleBufferCreate failed: OSStatus {}",
                    status
                )));
            }

            // SAFETY: Mutate the sample attachment dictionary to set keyframe flags.
            //
            // - `sample_buffer_ptr` is a valid CMSampleBufferRef created by
            //   `CMSampleBufferCreate` above with `num_samples = 1`, so sample
            //   index 0 is the only valid index.
            // - `CMSampleBufferGetSampleAttachmentsArray(sample_buffer_ptr, true)`
            //   returns a retained CFArrayRef with one entry per sample (i.e. one
            //   mutable CFDictionaryRef at index 0), or null on failure.
            // - `CFArrayGetValueAtIndex(attachments, 0)` yields a non-null
            //   CFDictionaryRef that we may mutate via `CFDictionarySetValue`
            //   because the array was obtained with `createIfNecessary = true`.
            // - Both pointers are checked for null before use.
            // - `is_keyframe` guards which keys are set: keyframes get
            //   DependsOnOthers=false + NotSync=false; non-keyframes get
            //   DependsOnOthers=true.
            unsafe {
                let attachments = CMSampleBufferGetSampleAttachmentsArray(sample_buffer_ptr, true);
                if !attachments.is_null() {
                    let dict = CFArrayGetValueAtIndex(attachments, 0);
                    if !dict.is_null() {
                        if is_keyframe {
                            CFDictionarySetValue(
                                dict,
                                kCMSampleAttachmentKey_DependsOnOthers,
                                kCFBooleanFalse,
                            );
                            CFDictionarySetValue(
                                dict,
                                kCMSampleAttachmentKey_NotSync,
                                kCFBooleanFalse,
                            );
                        } else {
                            CFDictionarySetValue(
                                dict,
                                kCMSampleAttachmentKey_DependsOnOthers,
                                kCFBooleanTrue,
                            );
                        }
                    }
                }
            }

            tracing::debug!(
                "VTDecoder: CMSampleBuffer created (keyframe={}), calling VTDecompressionSessionDecodeFrame",
                is_keyframe
            );

            // Decode the frame synchronously for MoQ live streams
            // Use flag 0 to request synchronous decode
            let decode_flags: u32 = 0;

            let mut info_flags_out: u32 = 0;

            let status = unsafe {
                VTDecompressionSessionDecodeFrame(
                    self.session,
                    sample_buffer_ptr,
                    decode_flags,
                    ptr::null_mut(),     // source frame refcon
                    &mut info_flags_out, // info flags out
                )
            };
            tracing::debug!(
                "VTDecoder: VTDecompressionSessionDecodeFrame returned status={}, info_flags={}",
                status,
                info_flags_out
            );

            if status != 0 {
                // Release sample buffer before returning error
                unsafe { CFRelease(sample_buffer_ptr) };
                return Err(VideoError::DecodeFailed(format!(
                    "VTDecompressionSessionDecodeFrame failed: OSStatus {}",
                    status
                )));
            }

            tracing::debug!("VTDecoder: waiting for async frames");

            // Wait for decode to complete BEFORE releasing sample buffer
            // This ensures the memory backing avcc_data stays valid
            let wait_status =
                unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
            tracing::debug!("VTDecoder: wait completed, status={}", wait_status);

            // Now safe to release sample buffer - decode is complete
            // Note: avcc_data is still valid here because we used kCFAllocatorNull,
            // so CMBlockBuffer never tried to free it. Rust will drop it at scope end.
            unsafe { CFRelease(sample_buffer_ptr) };
            tracing::debug!("VTDecoder: released sample buffer");

            let status = wait_status;

            if status != 0 {
                tracing::warn!(
                    "VTDecompressionSessionWaitForAsynchronousFrames: OSStatus {}",
                    status
                );
            }

            // Check for decode errors from callback
            if self
                .callback_state
                .decode_error
                .swap(false, Ordering::AcqRel)
            {
                let cb_status = self
                    .callback_state
                    .decode_error_status
                    .swap(0, Ordering::Relaxed);
                return Err(VideoError::DecodeFailed(format!(
                    "VT decode callback error: OSStatus {}",
                    cb_status
                )));
            }

            // Pop decoded frame from callback queue
            let queue_len = self.callback_state.decoded_frames.lock().len();
            tracing::debug!("VTDecoder: checking callback queue, length={}", queue_len);
            let decoded = self.callback_state.decoded_frames.lock().pop_front();

            match decoded {
                Some(frame) => {
                    if frame.required_frame_dropped {
                        self.required_drop_streak = self.required_drop_streak.saturating_add(1);
                        tracing::debug!(
                            "VTDecoder: callback flagged RequiredFrameDropped on decoded frame (streak={})",
                            self.required_drop_streak
                        );
                        if self.required_drop_streak >= VT_REQUIRED_DROP_STORM_THRESHOLD {
                            let storm_streak = self.required_drop_streak;
                            // Reset so we don't immediately retrigger every frame if
                            // callback flags remain noisy for a short window.
                            self.required_drop_streak = 0;
                            return Err(VideoError::DecodeFailed(format!(
                                "VT required-frame-drop storm (streak={})",
                                storm_streak
                            )));
                        }
                    } else if self.required_drop_streak > 0 {
                        tracing::debug!(
                            "VTDecoder: required-frame-drop streak cleared at {}",
                            self.required_drop_streak
                        );
                        self.required_drop_streak = 0;
                    }

                    tracing::debug!("VTDecoder: got frame from queue, calling create_gpu_frame");
                    // Create MacOSGpuSurface from CVPixelBuffer
                    let video_frame = self.create_gpu_frame(frame)?;
                    tracing::debug!("VTDecoder: create_gpu_frame succeeded");
                    Ok(Some(video_frame))
                }
                None => {
                    // No frame available yet (async decode may not have completed)
                    tracing::debug!("VTDecoder: queue was empty, returning None");
                    Ok(None)
                }
            }
        }

        /// Checks if NAL data is in AVCC format (4-byte length prefix) vs Annex B (start codes).
        ///
        /// AVCC format: first 4 bytes are NAL length (big-endian), followed by NAL data
        /// Annex B format: starts with 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01
        ///
        /// NOTE: This heuristic has known false negatives for AVCC frames with 256-511 byte
        /// NALs (length prefix [0,0,1,X] looks like Annex B). Prefer using VTDecoder::is_avcc
        /// field which is set from catalog info.
        #[allow(dead_code)]
        fn is_avcc_format(data: &[u8]) -> bool {
            if data.len() < 5 {
                return false;
            }

            // Check for Annex B start codes first
            if data[0] == 0 && data[1] == 0 {
                if data[2] == 1 {
                    return false; // 3-byte start code
                }
                if data[2] == 0 && data[3] == 1 {
                    return false; // 4-byte start code
                }
            }

            // Check if first 4 bytes make sense as AVCC length
            let nal_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

            // AVCC length should be reasonable (not zero, not larger than data)
            // and the NAL type byte should be valid (0x01-0x1F for H.264)
            if nal_len > 0 && nal_len <= data.len() - 4 {
                let nal_type = data[4] & 0x1F;
                // Valid H.264 NAL types are 1-23
                if (1..=23).contains(&nal_type) {
                    return true;
                }
            }

            false
        }

        /// Converts Annex B NAL data (start code prefixed) to AVCC format (length prefixed).
        fn annex_b_to_avcc(nal_data: &[u8]) -> Vec<u8> {
            // Find NAL unit boundaries (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
            let mut result = Vec::with_capacity(nal_data.len() + 4);
            let mut i = 0;

            while i < nal_data.len() {
                // Find start code
                let start_code_len = if i + 4 <= nal_data.len()
                    && nal_data[i] == 0
                    && nal_data[i + 1] == 0
                    && nal_data[i + 2] == 0
                    && nal_data[i + 3] == 1
                {
                    4
                } else if i + 3 <= nal_data.len()
                    && nal_data[i] == 0
                    && nal_data[i + 1] == 0
                    && nal_data[i + 2] == 1
                {
                    3
                } else {
                    // No start code at this position, check next byte
                    i += 1;
                    continue;
                };

                // Find end of this NAL unit (next start code or end of data)
                let nal_start = i + start_code_len;
                let mut nal_end = nal_data.len();

                for j in nal_start..nal_data.len().saturating_sub(2) {
                    if nal_data[j] == 0 && nal_data[j + 1] == 0 {
                        if j + 2 < nal_data.len() && nal_data[j + 2] == 1 {
                            nal_end = j;
                            break;
                        }
                        if j + 3 < nal_data.len() && nal_data[j + 2] == 0 && nal_data[j + 3] == 1 {
                            nal_end = j;
                            break;
                        }
                    }
                }

                // Write NAL unit with 4-byte length prefix
                let nal_len = nal_end - nal_start;
                result.extend_from_slice(&(nal_len as u32).to_be_bytes());
                result.extend_from_slice(&nal_data[nal_start..nal_end]);

                i = nal_end;
            }

            // If no start codes found, assume raw NAL unit
            if result.is_empty() && !nal_data.is_empty() {
                result.extend_from_slice(&(nal_data.len() as u32).to_be_bytes());
                result.extend_from_slice(nal_data);
            }

            result
        }

        /// Creates a VideoFrame with MacOSGpuSurface from a decoded CVPixelBuffer.
        fn create_gpu_frame(&self, frame: DecodedVTFrame) -> Result<VideoFrame, VideoError> {
            tracing::debug!(
                "VTDecoder: create_gpu_frame called, pts_us={}",
                frame.pts_us
            );
            let pb_ptr = frame.pixel_buffer.0;
            tracing::debug!("VTDecoder: pixel_buffer ptr={:?}", pb_ptr);
            let width = unsafe { CVPixelBufferGetWidth(pb_ptr) } as u32;
            let height = unsafe { CVPixelBufferGetHeight(pb_ptr) } as u32;
            tracing::debug!("VTDecoder: got dimensions {}x{}", width, height);

            // Check for IOSurface availability (confirms hardware decode)
            let io_surface = unsafe { CVPixelBufferGetIOSurface(pb_ptr) };

            if io_surface.is_null() {
                // Fallback: IOSurface not available, would need CPU copy
                // For now, return error as we require zero-copy
                return Err(VideoError::DecodeFailed(
                    "VTDecoder: IOSurface not available (software decode?)".to_string(),
                ));
            }

            // Create owner wrapper to keep CVPixelBuffer alive
            let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(frame.pixel_buffer);

            // Create MacOSGpuSurface for zero-copy rendering
            let gpu_surface = unsafe {
                MacOSGpuSurface::new(
                    io_surface,
                    width,
                    height,
                    PixelFormat::Bgra,
                    None, // No CPU fallback - zero-copy only
                    owner,
                )
            };

            let pts = Duration::from_micros(frame.pts_us);

            tracing::trace!(
                "VTDecoder: decoded frame {}x{} pts={:?} (zero-copy)",
                width,
                height,
                pts
            );

            Ok(VideoFrame::new(pts, DecodedFrame::MacOS(gpu_surface)))
        }

        /// Returns the codec type (reserved for H.265 support).
        #[allow(dead_code)]
        pub fn codec(&self) -> VTCodec {
            self.codec
        }

        /// Returns the number of frames decoded so far.
        pub fn frame_count(&self) -> u32 {
            self.callback_state
                .frame_count
                .load(std::sync::atomic::Ordering::Relaxed)
        }

        /// Clears queued output and error flag before waiting for a fresh IDR.
        #[allow(dead_code)]
        pub fn prepare_for_idr_resync(&mut self) {
            // Wait for any pending async frames to complete
            let wait_status =
                unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
            if wait_status != 0 {
                tracing::debug!(
                    "VTDecoder: wait during IDR resync returned OSStatus {}",
                    wait_status
                );
            }
            // Clear the output queue and error flag
            self.callback_state.decoded_frames.lock().clear();
            self.callback_state
                .decode_error
                .store(false, Ordering::Release);
            tracing::debug!("VTDecoder: prepared for IDR resync (cleared queue and error flag)");
        }
    }

    impl Drop for VTDecoder {
        fn drop(&mut self) {
            let frame_count = self.callback_state.frame_count.load(Ordering::Relaxed);
            tracing::info!("VTDecoder: dropping after decoding {} frames", frame_count);

            if !self.session.is_null() {
                unsafe {
                    // Drain any in-flight async decode callbacks before invalidating.
                    // Without this, Invalidate can fire while a callback is still writing
                    // to callback_state, and the hardware decoder may not fully quiesce.
                    let wait_status = VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                    if wait_status != 0 {
                        tracing::warn!(
                            "VTDecoder: WaitForAsyncFrames in drop returned OSStatus {}",
                            wait_status
                        );
                    }
                    VTDecompressionSessionInvalidate(self.session);
                    CFRelease(self.session);
                }
            }

            // Release format description
            if !self.format_desc.is_null() {
                unsafe { CFRelease(self.format_desc) };
            }
            tracing::info!("VTDecoder: dropped (session invalidated + released)");
        }
    }

    /// VTDecompressionSession output callback.
    ///
    /// This is called by VideoToolbox when a frame has been decoded.
    /// The decoded CVPixelBuffer is pushed to the callback state queue.
    extern "C" fn vt_decode_callback(
        refcon: *mut c_void,
        _source_frame_refcon: *mut c_void,
        status: i32,
        info_flags: u32,
        image_buffer: *mut c_void, // CVImageBufferRef (same as CVPixelBufferRef for video)
        presentation_time_stamp: CMTime,
        _presentation_duration: CMTime,
    ) {
        tracing::debug!(
            "VT decode callback: status={}, info_flags={}, image_buffer={:?}",
            status,
            info_flags,
            image_buffer
        );

        // Recover the callback state from refcon
        let callback_state = unsafe { &*(refcon as *const VTCallbackState) };

        if status != 0 {
            tracing::error!("VT decode callback error: OSStatus {}", status);
            callback_state
                .decode_error_status
                .store(status, Ordering::Release);
            callback_state.decode_error.store(true, Ordering::Release);
            return;
        }

        // Check for dropped frames
        if info_flags & 0x1 != 0 {
            // kVTDecodeInfo_Asynchronous
            tracing::debug!("VT decode: async frame (info_flags=0x{:x})", info_flags);
        }
        if info_flags & 0x2 != 0 {
            // kVTDecodeInfo_FrameDropped
            tracing::warn!(
                "VT decode: frame dropped by VideoToolbox (info_flags=0x{:x})",
                info_flags
            );
            return;
        }
        if image_buffer.is_null() {
            if info_flags & 0x4 != 0 {
                // kVTDecodeInfo_RequiredFrameDropped with no image — genuine drop
                tracing::warn!(
                    "VT decode: frame dropped (info_flags=0x{:x}, no image buffer)",
                    info_flags
                );
            } else {
                tracing::warn!("VT decode callback: null image buffer");
            }
            return;
        }

        // info_flags bit 0x4 (RequiredFrameDropped) fires on 100% of frames on
        // macOS 15 / Apple Silicon even when status=0 and image_buffer is valid.
        // When VT successfully produces pixels, treat the frame as good — the flag
        // is informational, not an error signal.
        let required_frame_dropped = if info_flags & 0x4 != 0 && !image_buffer.is_null() {
            tracing::trace!(
                "VT decode: info_flags=0x{:x} with valid image — treating as successful",
                info_flags
            );
            false
        } else {
            false
        };

        // CVImageBuffer is the same as CVPixelBuffer for video frames
        // Retain the pixel buffer so it stays valid
        tracing::debug!(
            "VT decode callback: retaining image_buffer {:?}",
            image_buffer
        );
        let pixel_buffer = unsafe { PixelBufferWrapper::retain(image_buffer) };
        tracing::debug!("VT decode callback: retained, ptr={:?}", pixel_buffer.0);

        // Extract PTS from CMTime
        let pts_us = if presentation_time_stamp.timescale > 0 {
            ((presentation_time_stamp.value as f64 / presentation_time_stamp.timescale as f64)
                * 1_000_000.0) as u64
        } else {
            0
        };

        // Push to decoded frame queue
        let frame = DecodedVTFrame {
            pts_us,
            pixel_buffer,
            required_frame_dropped,
        };
        tracing::debug!("VT decode callback: acquiring lock on decoded_frames queue");
        let mut queue = callback_state.decoded_frames.lock();
        queue.push_back(frame);
        let queue_len = queue.len();
        drop(queue);
        let total_count = callback_state.frame_count.fetch_add(1, Ordering::Relaxed) + 1;

        tracing::debug!(
            "VT decode callback: pushed frame pts={}us, queue_len={}, total_count={}",
            pts_us,
            queue_len,
            total_count
        );
    }
}

// =============================================================================
// Android MediaCodec Zero-Copy Implementation
// =============================================================================
//
// This module provides zero-copy video decoding on Android using MediaCodec directly.
// NAL units from MoQ are fed to MediaCodec, which outputs to an ImageReader with
// AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE for zero-copy Vulkan import.
//
// Architecture:
//   MoQ NAL units -> MediaCodec -> ImageReader -> HardwareBuffer -> JNI
//                 -> import_ahardwarebuffer_yuv_zero_copy() -> wgpu texture
//
// Key differences from ExoPlayer path:
// - ExoPlayer: URL-based playback, manages MediaCodec internally
// - This: Raw NAL unit input, direct MediaCodec control, lower latency

#[cfg(target_os = "android")]
pub mod android {
    use super::*;
    use crate::media::android_video::{
        generate_player_id, try_receive_hardware_buffer_for_player, AndroidVideoFrame,
    };
    use crate::media::video::AndroidGpuSurface;
    use jni::objects::{GlobalRef, JClass, JObject, JValue};
    use jni::sys::{jint, jlong};
    use jni::JNIEnv;
    use std::collections::VecDeque;

    /// Android MoQ decoder using MediaCodec with zero-copy HardwareBuffer output.
    ///
    /// This decoder receives NAL units from MoQ and decodes them using Android's
    /// MediaCodec API directly, outputting to an ImageReader configured with
    /// `AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE` for zero-copy Vulkan import.
    ///
    /// # Zero-Copy Pipeline
    ///
    /// ```text
    /// MoQ NAL units -> MediaCodec -> ImageReader -> HardwareBuffer
    ///              -> JNI nativeSubmitHardwareBuffer()
    ///              -> import_ahardwarebuffer_yuv_zero_copy()
    ///              -> wgpu::Texture (GPU-side YUV to RGB)
    /// ```
    pub struct MoqAndroidDecoder {
        /// Parsed MoQ URL
        #[allow(dead_code)]
        url: MoqUrl,
        /// Configuration
        #[allow(dead_code)]
        config: MoqDecoderConfig,
        /// Shared state with async worker
        shared: Arc<MoqSharedState>,
        /// Receiver for encoded NAL units from MoQ worker
        nal_rx: Receiver<MoqVideoFrame>,
        /// Owned tokio runtime (created if none exists)
        _owned_runtime: Option<tokio::runtime::Runtime>,
        /// Tokio runtime handle
        _runtime: Handle,
        /// Whether audio is muted
        audio_muted: bool,
        /// Audio volume (0.0 to 1.0)
        audio_volume: f32,
        /// JNI reference to MoqMediaCodecBridge
        bridge: Option<GlobalRef>,
        /// Unique player ID for frame queue isolation
        player_id: u64,
        /// Pending decoded frames from HardwareBuffer queue
        pending_frames: VecDeque<AndroidVideoFrame>,
        /// Whether MediaCodec has been configured
        codec_configured: bool,
        /// Codec type detected from catalog
        codec_type: Option<CodecType>,
        /// Locally cached metadata (safe copy from shared state)
        cached_metadata: VideoMetadata,
    }

    /// Supported video codec types for MediaCodec.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CodecType {
        /// H.264/AVC
        H264,
        /// H.265/HEVC
        H265,
    }

    impl CodecType {
        /// Returns the MediaCodec MIME type string.
        pub fn mime_type(&self) -> &'static str {
            match self {
                CodecType::H264 => "video/avc",
                CodecType::H265 => "video/hevc",
            }
        }

        /// Parses codec type from hang catalog codec string.
        pub fn from_catalog_codec(codec: &str) -> Option<Self> {
            let lower = codec.to_lowercase();
            if lower.contains("avc") || lower.contains("h264") || lower.contains("h.264") {
                Some(CodecType::H264)
            } else if lower.contains("hevc")
                || lower.contains("hvc1")
                || lower.contains("h265")
                || lower.contains("h.265")
            {
                Some(CodecType::H265)
            } else {
                None
            }
        }
    }

    impl MoqAndroidDecoder {
        /// Creates a new Android MoQ decoder for the given URL.
        pub fn new(url: &str) -> Result<Self, VideoError> {
            Self::new_with_config(url, MoqDecoderConfig::default())
        }

        /// Creates a new Android MoQ decoder with explicit configuration.
        pub fn new_with_config(
            url: &str,
            mut config: MoqDecoderConfig,
        ) -> Result<Self, VideoError> {
            let moq_url = MoqUrl::parse(url).map_err(|e| VideoError::OpenFailed(e.to_string()))?;
            config.apply_localhost_tls_bypass(&moq_url);

            // Get existing runtime handle or create a new runtime
            let (owned_runtime, runtime) = match Handle::try_current() {
                Ok(handle) => (None, handle),
                Err(_) => {
                    let rt = tokio::runtime::Builder::new_multi_thread()
                        .worker_threads(2)
                        .enable_all()
                        .thread_name("moq-android-runtime")
                        .build()
                        .map_err(|e| {
                            VideoError::OpenFailed(format!("Failed to create tokio runtime: {e}"))
                        })?;
                    let handle = rt.handle().clone();
                    (Some(rt), handle)
                }
            };

            let shared = Arc::new(MoqSharedState::new());
            let (nal_tx, nal_rx) = async_channel::bounded(60); // Larger buffer for NAL units

            // Generate unique player ID for frame queue isolation
            let player_id = generate_player_id();
            tracing::info!("MoqAndroidDecoder: Created with player_id={}", player_id);

            // Spawn MoQ worker (reuses same logic as MoqDecoder)
            let worker_shared = shared.clone();
            let worker_url = moq_url.clone();
            let worker_config = config.clone();

            runtime.spawn(async move {
                if let Err(e) =
                    Self::run_moq_worker(worker_shared.clone(), worker_url, worker_config, nal_tx)
                        .await
                {
                    worker_shared.set_error(format!("MoQ worker error: {e}"));
                }
            });

            Ok(Self {
                url: moq_url,
                config,
                shared,
                nal_rx,
                _owned_runtime: owned_runtime,
                _runtime: runtime,
                audio_muted: false,
                audio_volume: config.initial_volume,
                bridge: None,
                player_id,
                pending_frames: VecDeque::new(),
                codec_configured: false,
                codec_type: None,
                cached_metadata: VideoMetadata {
                    codec: String::new(),
                    width: 0,
                    height: 0,
                    frame_rate: 0.0,
                    duration: None,
                    pixel_aspect_ratio: 1.0,
                    start_time: None,
                },
            })
        }

        /// Async worker that handles MoQ connection and NAL unit receipt.
        async fn run_moq_worker(
            shared: Arc<MoqSharedState>,
            url: MoqUrl,
            config: MoqDecoderConfig,
            nal_tx: Sender<MoqVideoFrame>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            crate::media::moq::worker::run_moq_worker(shared, url, config, nal_tx, "Android").await
        }

        /// Initializes the MediaCodec decoder via JNI.
        ///
        /// Creates the MoqMediaCodecBridge Java object which:
        /// 1. Creates MediaCodec for H.264/H.265
        /// 2. Configures ImageReader with GPU_SAMPLED_IMAGE usage
        /// 3. Sets up the output surface for zero-copy frame extraction
        fn initialize_codec(&mut self) -> Result<(), VideoError> {
            if self.codec_configured {
                return Ok(());
            }

            let metadata = self.shared.metadata.lock();
            let width = metadata.width;
            let height = metadata.height;
            let codec_str = metadata.codec.clone();
            drop(metadata);

            // Determine codec type from catalog metadata
            let codec_type = CodecType::from_catalog_codec(&codec_str).ok_or_else(|| {
                VideoError::UnsupportedFormat(format!(
                    "Unsupported codec for Android MediaCodec: {}",
                    codec_str
                ))
            })?;
            self.codec_type = Some(codec_type);

            // Get JVM and create bridge via JNI
            let vm = Self::get_jvm()?;
            let mut env = vm.attach_current_thread().map_err(|e| {
                VideoError::DecoderInit(format!("Failed to attach JNI thread: {}", e))
            })?;

            // Get Android context
            let context =
                unsafe { JObject::from_raw(ndk_context::android_context().context().cast()) };

            // Load MoqMediaCodecBridge class via class loader
            let class_loader = env
                .call_method(&context, "getClassLoader", "()Ljava/lang/ClassLoader;", &[])
                .map_err(|e| VideoError::DecoderInit(format!("Failed to get class loader: {}", e)))?
                .l()
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get class loader object: {}", e))
                })?;

            let class_name = env
                .new_string("com.luminavideo.bridge.MoqMediaCodecBridge")
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to create class name string: {}", e))
                })?;

            let bridge_class = env
                .call_method(
                    &class_loader,
                    "loadClass",
                    "(Ljava/lang/String;)Ljava/lang/Class;",
                    &[JValue::Object(&class_name)],
                )
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to load MoqMediaCodecBridge: {}", e))
                })?
                .l()
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get bridge class: {}", e))
                })?;

            let bridge_class = jni::objects::JClass::from(bridge_class);

            // Create MIME type string for MediaCodec
            let mime_type = env.new_string(codec_type.mime_type()).map_err(|e| {
                VideoError::DecoderInit(format!("Failed to create MIME type string: {}", e))
            })?;

            // Create bridge: MoqMediaCodecBridge(Context, String mimeType, int width, int height, long playerId)
            let bridge = env
                .new_object(
                    bridge_class,
                    "(Landroid/content/Context;Ljava/lang/String;IIJ)V",
                    &[
                        JValue::Object(&context),
                        JValue::Object(&mime_type),
                        JValue::Int(width as i32),
                        JValue::Int(height as i32),
                        JValue::Long(self.player_id as i64),
                    ],
                )
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to create MoqMediaCodecBridge: {}", e))
                })?;

            let bridge_ref = env.new_global_ref(bridge).map_err(|e| {
                VideoError::DecoderInit(format!("Failed to create global ref: {}", e))
            })?;

            // Start the decoder
            env.call_method(&bridge_ref, "start", "()V", &[])
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to start MediaCodec: {}", e))
                })?;

            self.bridge = Some(bridge_ref);
            self.codec_configured = true;

            tracing::info!(
                "MoqAndroidDecoder: MediaCodec initialized for {} ({}x{})",
                codec_type.mime_type(),
                width,
                height
            );

            Ok(())
        }

        /// Submits a NAL unit to MediaCodec for decoding.
        ///
        /// The NAL unit is passed to the Java bridge which queues it in
        /// MediaCodec's input buffer. Decoded frames appear asynchronously
        /// in the HardwareBuffer queue.
        fn submit_nal_unit(&self, nal_data: &[u8], timestamp_us: u64) -> Result<(), VideoError> {
            let Some(bridge) = &self.bridge else {
                return Err(VideoError::DecodeFailed(
                    "MediaCodec not initialized".to_string(),
                ));
            };

            let vm = Self::get_jvm()?;
            let mut env = vm.attach_current_thread().map_err(|e| {
                VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e))
            })?;

            // Create byte array from NAL data
            let byte_array = env.new_byte_array(nal_data.len() as i32).map_err(|e| {
                VideoError::DecodeFailed(format!("Failed to create byte array: {}", e))
            })?;

            // Convert u8 slice to i8 slice for JNI
            let nal_data_i8: Vec<i8> = nal_data.iter().map(|&b| b as i8).collect();
            env.set_byte_array_region(&byte_array, 0, &nal_data_i8)
                .map_err(|e| {
                    VideoError::DecodeFailed(format!("Failed to set byte array data: {}", e))
                })?;

            // Submit to MediaCodec: submitNalUnit(byte[] data, long timestampUs)
            env.call_method(
                bridge,
                "submitNalUnit",
                "([BJ)V",
                &[
                    JValue::Object(&byte_array),
                    JValue::Long(timestamp_us as i64),
                ],
            )
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to submit NAL unit: {}", e)))?;

            Ok(())
        }

        /// Polls for decoded frames from the HardwareBuffer queue.
        ///
        /// MediaCodec outputs to ImageReader, which extracts HardwareBuffers
        /// and submits them via JNI to the per-player queue.
        fn poll_decoded_frames(&mut self) {
            while let Some(frame) = try_receive_hardware_buffer_for_player(self.player_id) {
                self.pending_frames.push_back(frame);
            }
        }

        /// Converts an AndroidVideoFrame to a VideoFrame with zero-copy GPU surface.
        ///
        /// The HardwareBuffer is wrapped in an AndroidGpuSurface which can be
        /// imported into Vulkan via import_ahardwarebuffer_yuv_zero_copy().
        fn convert_to_video_frame(&self, frame: AndroidVideoFrame) -> VideoFrame {
            let pts = Duration::from_nanos(frame.timestamp_ns as u64);

            // Create owner to track HardwareBuffer lifetime
            struct HardwareBufferOwner {
                #[allow(dead_code)]
                buffer: *mut std::ffi::c_void,
            }

            // SAFETY: AHardwareBuffer is thread-safe per Android NDK docs
            unsafe impl Send for HardwareBufferOwner {}
            unsafe impl Sync for HardwareBufferOwner {}

            impl Drop for HardwareBufferOwner {
                fn drop(&mut self) {
                    // Don't release here - AndroidVideoFrame::drop handles it
                    // This owner is just for lifetime tracking
                }
            }

            let owner = Arc::new(HardwareBufferOwner {
                buffer: frame.buffer,
            });

            // Determine pixel format from AHardwareBuffer format
            let pixel_format =
                if crate::media::android_video::is_yuv_hardware_buffer_format(frame.format) {
                    PixelFormat::Nv12 // Most common YUV format from MediaCodec
                } else {
                    PixelFormat::Rgba
                };

            let surface = unsafe {
                AndroidGpuSurface::new(
                    frame.buffer,
                    frame.width,
                    frame.height,
                    pixel_format,
                    None, // No CPU fallback for zero-copy frames
                    owner,
                )
            };

            // Transfer ownership - prevent AndroidVideoFrame from releasing the buffer
            // since the AndroidGpuSurface now owns the reference
            std::mem::forget(frame);

            VideoFrame::new(pts, DecodedFrame::Android(surface))
        }

        /// Gets the Java VM from NDK context.
        fn get_jvm() -> Result<jni::JavaVM, VideoError> {
            unsafe { jni::JavaVM::from_raw(ndk_context::android_context().vm().cast()) }
                .map_err(|e| VideoError::DecoderInit(format!("Failed to get JavaVM: {}", e)))
        }

        /// Returns the current decoder state.
        pub fn decoder_state(&self) -> MoqDecoderState {
            *self.shared.state.lock()
        }

        /// Returns the error message if in error state.
        pub fn error_message(&self) -> Option<String> {
            self.shared.error_message.lock().clone()
        }

        /// Returns the player ID for this decoder instance.
        pub fn player_id(&self) -> u64 {
            self.player_id
        }
    }

    impl Drop for MoqAndroidDecoder {
        fn drop(&mut self) {
            // Release MediaCodec resources via JNI
            if let Some(bridge) = self.bridge.take() {
                if let Ok(vm) = Self::get_jvm() {
                    if let Ok(mut env) = vm.attach_current_thread() {
                        let _ = env.call_method(&bridge, "release", "()V", &[]);
                    }
                }
            }

            // Release player's frame queue
            crate::media::android_video::release_player_queue(self.player_id);

            tracing::info!("MoqAndroidDecoder: Released player_id={}", self.player_id);
        }
    }

    impl VideoDecoderBackend for MoqAndroidDecoder {
        fn open(url: &str) -> Result<Self, VideoError>
        where
            Self: Sized,
        {
            Self::new(url)
        }

        fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
            // Sync cached metadata from shared state (safe copy under lock)
            {
                let shared_metadata = self.shared.metadata.lock();
                if shared_metadata.width != self.cached_metadata.width
                    || shared_metadata.height != self.cached_metadata.height
                {
                    self.cached_metadata = shared_metadata.clone();
                }
            }

            // Check EOF
            if self.shared.eof_reached.load(Ordering::Relaxed) {
                return Ok(None);
            }

            // Check for errors
            let state = *self.shared.state.lock();
            if state == MoqDecoderState::Error {
                return Err(VideoError::DecodeFailed(
                    self.error_message()
                        .unwrap_or_else(|| "Unknown error".to_string()),
                ));
            }

            // Wait for metadata before initializing codec
            if state != MoqDecoderState::Streaming && !self.codec_configured {
                // Try to receive NAL units to trigger state updates
                let _ = self.nal_rx.try_recv();
                return Ok(None);
            }

            // Initialize codec once we have metadata
            if !self.codec_configured {
                self.initialize_codec()?;
            }

            // Submit any pending NAL units to MediaCodec
            while let Ok(nal_frame) = self.nal_rx.try_recv() {
                self.submit_nal_unit(&nal_frame.data, nal_frame.timestamp_us)?;
            }

            // Poll for decoded frames from HardwareBuffer queue
            self.poll_decoded_frames();

            // Return next decoded frame if available
            if let Some(frame) = self.pending_frames.pop_front() {
                return Ok(Some(self.convert_to_video_frame(frame)));
            }

            Ok(None)
        }

        fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
            Err(VideoError::SeekFailed(
                "Seeking is not supported on live MoQ streams".to_string(),
            ))
        }

        fn metadata(&self) -> &VideoMetadata {
            &self.cached_metadata
        }

        fn duration(&self) -> Option<Duration> {
            None // Live streams have no duration
        }

        fn is_eof(&self) -> bool {
            self.shared.eof_reached.load(Ordering::Relaxed)
        }

        fn buffering_percent(&self) -> i32 {
            self.shared.buffering_percent.load(Ordering::Relaxed)
        }

        fn hw_accel_type(&self) -> HwAccelType {
            HwAccelType::MediaCodec
        }

        fn handles_audio_internally(&self) -> bool {
            self.shared
                .audio
                .internal_audio_ready
                .load(Ordering::Relaxed)
        }

        fn audio_handle(&self) -> Option<super::audio::AudioHandle> {
            self.shared.audio.moq_audio_handle.lock().clone()
        }

        fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
            self.audio_muted = muted;
            Ok(())
        }

        fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
            self.audio_volume = volume.clamp(0.0, 1.0);
            Ok(())
        }
    }

    // ========================================================================
    // JNI Entry Points for MoqMediaCodecBridge
    // ========================================================================
    //
    // These functions are called from com.luminavideo.bridge.MoqMediaCodecBridge
    // when decoded frames are available from MediaCodec.

    /// JNI callback when a decoded frame is available from MediaCodec.
    ///
    /// Called by MoqMediaCodecBridge.onOutputBufferAvailable() after acquiring
    /// the HardwareBuffer from ImageReader.
    ///
    /// This reuses the existing nativeSubmitHardwareBuffer infrastructure from
    /// ExoPlayerBridge - the frame is queued by player_id for isolation.
    #[no_mangle]
    pub extern "C" fn Java_com_luminavideo_bridge_MoqMediaCodecBridge_nativeSubmitHardwareBuffer(
        env: JNIEnv,
        class: JClass,
        buffer: JObject,
        timestamp_ns: jlong,
        width: jint,
        height: jint,
        player_id: jlong,
        fence_fd: jint,
    ) {
        // Delegate to the existing ExoPlayerBridge implementation
        // This reuses all the HardwareBuffer acquisition and queue logic
        crate::media::android_video::Java_com_luminavideo_bridge_ExoPlayerBridge_nativeSubmitHardwareBuffer(
            env,
            class,
            buffer,
            timestamp_ns,
            width,
            height,
            player_id,
            fence_fd,
        );
    }

    /// JNI callback for codec errors.
    #[no_mangle]
    pub extern "C" fn Java_com_luminavideo_bridge_MoqMediaCodecBridge_nativeOnError(
        mut env: JNIEnv,
        _class: JClass,
        player_id: jlong,
        error_message: jni::objects::JString,
    ) {
        let error: String = env
            .get_string(&error_message)
            .map(|s| s.into())
            .unwrap_or_else(|_| "Unknown MediaCodec error".to_string());

        tracing::error!(
            "MoqMediaCodecBridge error (player_id={}): {}",
            player_id,
            error
        );
    }

    /// JNI callback when video dimensions change (e.g., adaptive bitrate switch).
    #[no_mangle]
    pub extern "C" fn Java_com_luminavideo_bridge_MoqMediaCodecBridge_nativeOnVideoSizeChanged(
        _env: JNIEnv,
        _class: JClass,
        player_id: jlong,
        width: jint,
        height: jint,
    ) {
        tracing::info!(
            "MoqMediaCodecBridge video size changed (player_id={}): {}x{}",
            player_id,
            width,
            height
        );
    }
}

// =============================================================================
// Linux: MoQ + GStreamer Zero-Copy Decoder
// =============================================================================

/// MoQ video decoder using GStreamer with appsrc for Linux zero-copy rendering.
///
/// This decoder:
/// 1. Receives NAL units from MoQ via the hang crate
/// 2. Pushes NAL units to GStreamer via appsrc element
/// 3. Decodes using VA-API hardware acceleration (vaH264Dec/vaH265Dec)
/// 4. Extracts DMABuf file descriptors from GstBuffer
/// 5. Imports DMABuf into Vulkan via wgpu for zero-copy rendering
///
/// # Pipeline
///
/// ```text
/// appsrc (stream-type=stream, format=time)
///    caps: video/x-h264,stream-format=byte-stream,alignment=au
///    → h264parse
///    → vaH264Dec OR avdec_h264 (fallback)
///    → video/x-raw(memory:DMABuf),format=NV12
///    → appsink (max-buffers=2, drop=true)
/// ```
#[cfg(target_os = "linux")]
pub struct MoqGStreamerDecoder {
    /// GStreamer pipeline
    pipeline: gstreamer::Pipeline,
    /// AppSrc element for pushing NAL units
    appsrc: gstreamer_app::AppSrc,
    /// AppSink element for pulled decoded frames
    appsink: gstreamer_app::AppSink,
    /// Parsed MoQ URL
    #[allow(dead_code)]
    url: MoqUrl,
    /// Configuration
    #[allow(dead_code)]
    config: MoqDecoderConfig,
    /// Shared state with async worker
    shared: Arc<MoqSharedState>,
    /// Receiver for encoded NAL units from async MoQ worker
    nal_rx: Receiver<MoqVideoFrame>,
    /// Active hardware acceleration type
    active_hw_type: HwAccelType,
    /// Owned tokio runtime (created if none exists)
    _owned_runtime: Option<tokio::runtime::Runtime>,
    /// Tokio runtime handle for async operations
    _runtime: Handle,
    /// Whether audio is muted
    audio_muted: bool,
    /// Audio volume (0.0 to 1.0)
    audio_volume: f32,
    /// Current playback position (from last decoded frame)
    position: Duration,
    /// Whether we've received the first keyframe (needed to start decoding)
    received_keyframe: bool,
    /// Codec detected from catalog (H264 or H265)
    #[allow(dead_code)]
    codec: MoqVideoCodec,
    /// Locally cached metadata (safe copy from shared state)
    cached_metadata: VideoMetadata,
}

/// Video codec type for MoQ streams on Linux.
#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqVideoCodec {
    H264,
    H265,
    Unknown,
}

#[cfg(target_os = "linux")]
impl MoqGStreamerDecoder {
    /// Creates a new MoQ GStreamer decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, MoqDecoderConfig::default())
    }

    /// Creates a new MoQ GStreamer decoder with explicit configuration.
    pub fn new_with_config(url: &str, mut config: MoqDecoderConfig) -> Result<Self, VideoError> {
        use gstreamer as gst;
        use gstreamer::prelude::*;
        use gstreamer_app as gst_app;

        // Initialize GStreamer (safe to call multiple times)
        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {e}")))?;

        // Parse the MoQ URL
        let moq_url = MoqUrl::parse(url).map_err(|e| VideoError::OpenFailed(e.to_string()))?;
        config.apply_localhost_tls_bypass(&moq_url);

        // Get existing runtime handle or create a new runtime
        let (owned_runtime, runtime) = match Handle::try_current() {
            Ok(handle) => (None, handle),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("moq-gst-runtime")
                    .build()
                    .map_err(|e| {
                        VideoError::OpenFailed(format!("Failed to create tokio runtime: {e}"))
                    })?;
                let handle = rt.handle().clone();
                (Some(rt), handle)
            }
        };

        // Create shared state
        let shared = Arc::new(MoqSharedState::new());

        // Create channel for NAL units (bounded to limit memory usage)
        let (nal_tx, nal_rx) = async_channel::bounded(60);

        // Start with H264 as default, will be updated from catalog
        let codec = MoqVideoCodec::H264;

        // Build the GStreamer pipeline
        let pipeline = gst::Pipeline::new();

        // AppSrc: Source element for pushing NAL units
        // stream-type=stream: Seekable stream (live, no duration)
        // format=time: We provide PTS timestamps
        // is-live=true: Real-time source, don't block
        let appsrc = gst_app::AppSrc::builder()
            .name("moq_appsrc")
            .stream_type(gst_app::AppStreamType::Stream)
            .format(gst::Format::Time)
            .is_live(true)
            .max_bytes(10 * 1024 * 1024) // 10MB buffer
            .build();

        // Set initial caps for H.264 byte-stream NAL units
        let h264_caps = gst::Caps::builder("video/x-h264")
            .field("stream-format", "byte-stream")
            .field("alignment", "au") // Access unit alignment (complete NALs)
            .build();
        appsrc.set_caps(Some(&h264_caps));

        // Parser element (h264parse)
        let parser = gst::ElementFactory::make("h264parse")
            .name("parser")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create h264parse: {e}")))?;

        // Decoder: Try VA-API first, fallback to software
        let decoder = Self::create_decoder(codec)?;

        // AppSink: Output element for pulling decoded frames
        // Request DMABuf memory type for zero-copy
        let appsink = gst_app::AppSink::builder()
            .name("moq_appsink")
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .features(["memory:DMABuf"])
                    .field("format", "NV12")
                    .build(),
            )
            .max_buffers(2)
            .drop(true) // Drop old frames when buffer is full (live streaming)
            .sync(false) // Don't sync to clock (we handle timing)
            .build();

        // Add elements to pipeline
        pipeline
            .add_many([appsrc.upcast_ref(), &parser, &decoder, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

        // Link elements: appsrc -> parser -> decoder -> appsink
        gst::Element::link_many([appsrc.upcast_ref(), &parser, &decoder, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link elements: {e}")))?;

        // Set pipeline to PLAYING state
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to start pipeline: {e:?}")))?;

        tracing::info!("MoQ GStreamer: Pipeline created with VA-API decoder");

        // Spawn the async MoQ worker
        let worker_shared = shared.clone();
        let worker_url = moq_url.clone();
        let worker_config = config.clone();

        runtime.spawn(async move {
            if let Err(e) =
                Self::run_moq_worker(worker_shared.clone(), worker_url, worker_config, nal_tx).await
            {
                worker_shared.set_error(format!("MoQ worker error: {e}"));
            }
        });

        let initial_volume = config.initial_volume;
        Ok(Self {
            pipeline,
            appsrc,
            appsink,
            url: moq_url,
            config,
            shared,
            nal_rx,
            active_hw_type: HwAccelType::Vaapi,
            _owned_runtime: owned_runtime,
            _runtime: runtime,
            audio_muted: false,
            audio_volume: initial_volume,
            position: Duration::ZERO,
            received_keyframe: false,
            codec,
            cached_metadata: VideoMetadata {
                codec: String::new(),
                width: 0,
                height: 0,
                frame_rate: 0.0,
                duration: None,
                pixel_aspect_ratio: 1.0,
                start_time: None,
            },
        })
    }

    /// Creates the decoder element, preferring VA-API hardware acceleration.
    fn create_decoder(codec: MoqVideoCodec) -> Result<gstreamer::Element, VideoError> {
        use gstreamer as gst;

        let (va_decoder, sw_decoder) = match codec {
            MoqVideoCodec::H264 => ("vaH264Dec", "avdec_h264"),
            MoqVideoCodec::H265 => ("vaH265Dec", "avdec_h265"),
            MoqVideoCodec::Unknown => ("vaH264Dec", "avdec_h264"),
        };

        // Try VA-API decoder first
        if let Ok(decoder) = gst::ElementFactory::make(va_decoder)
            .name("decoder")
            .build()
        {
            tracing::info!("MoQ GStreamer: Using VA-API decoder {}", va_decoder);
            return Ok(decoder);
        }

        // Fallback to software decoder
        tracing::warn!(
            "MoQ GStreamer: VA-API decoder {} not available, falling back to {}",
            va_decoder,
            sw_decoder
        );
        gst::ElementFactory::make(sw_decoder)
            .name("decoder")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create decoder: {e}")))
    }

    /// Async worker that handles MoQ connection, catalog fetching, and NAL unit receipt.
    async fn run_moq_worker(
        shared: Arc<MoqSharedState>,
        url: MoqUrl,
        config: MoqDecoderConfig,
        nal_tx: Sender<MoqVideoFrame>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        super::moq::worker::run_moq_worker(shared, url, config, nal_tx, "GStreamer").await
    }

    /// Pushes a NAL unit to the GStreamer pipeline via appsrc.
    fn push_nal_unit(&mut self, moq_frame: &MoqVideoFrame) -> Result<(), VideoError> {
        use gstreamer as gst;

        // Skip non-keyframes until we receive a keyframe (decoder needs IDR to start)
        if !self.received_keyframe {
            if moq_frame.is_keyframe {
                self.received_keyframe = true;
                tracing::info!("MoQ GStreamer: Received first keyframe, starting decode");
            } else {
                tracing::debug!("MoQ GStreamer: Skipping non-keyframe (waiting for IDR)");
                return Ok(());
            }
        }

        // Create GstBuffer from NAL unit data
        let mut buffer = gst::Buffer::from_slice(moq_frame.data.clone());

        // Set PTS (presentation timestamp)
        {
            let buffer_ref = buffer.get_mut().ok_or_else(|| {
                VideoError::DecodeFailed("Failed to get mutable buffer reference".to_string())
            })?;
            buffer_ref.set_pts(gst::ClockTime::from_useconds(moq_frame.timestamp_us));

            // Mark non-keyframes with DELTA_UNIT (keyframes have no flag — absence of DELTA_UNIT implies sync point)
            if !moq_frame.is_keyframe {
                buffer_ref.set_flags(gst::BufferFlags::DELTA_UNIT);
            }
        }

        // Push buffer to appsrc
        match self.appsrc.push_buffer(buffer) {
            Ok(_) => Ok(()),
            Err(gst::FlowError::Flushing) => {
                tracing::debug!("MoQ GStreamer: Pipeline flushing, skipping buffer");
                Ok(())
            }
            Err(e) => Err(VideoError::DecodeFailed(format!(
                "Failed to push buffer to appsrc: {:?}",
                e
            ))),
        }
    }

    /// Pulls a decoded frame from appsink and converts it to VideoFrame.
    fn pull_decoded_frame(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        use gstreamer as gst;
        use gstreamer_video as gst_video;

        // Try to pull a sample (non-blocking)
        let sample = match self
            .appsink
            .try_pull_sample(gst::ClockTime::from_mseconds(0))
        {
            Some(s) => s,
            None => return Ok(None),
        };

        let buffer = sample
            .buffer()
            .ok_or_else(|| VideoError::DecodeFailed("Sample has no buffer".to_string()))?;

        let caps = sample
            .caps()
            .ok_or_else(|| VideoError::DecodeFailed("Sample has no caps".to_string()))?;

        let video_info = gst_video::VideoInfo::from_caps(caps)
            .map_err(|e| VideoError::DecodeFailed(format!("Invalid video caps: {e}")))?;

        let pts = buffer
            .pts()
            .map(|t| Duration::from_nanos(t.nseconds()))
            .unwrap_or(self.position);

        self.position = pts;

        let width = video_info.width();
        let height = video_info.height();

        // Try zero-copy DMABuf extraction
        if let Some(frame) =
            self.try_dmabuf_frame(buffer, &video_info, pts, width, height, &sample)?
        {
            return Ok(Some(frame));
        }

        // Fallback to CPU copy
        self.buffer_to_cpu_frame(buffer, &video_info, pts, width, height)
    }

    /// Attempts to extract DMABuf file descriptors for zero-copy rendering.
    fn try_dmabuf_frame(
        &self,
        buffer: &gstreamer::BufferRef,
        video_info: &gstreamer_video::VideoInfo,
        pts: Duration,
        width: u32,
        height: u32,
        sample: &gstreamer::Sample,
    ) -> Result<Option<VideoFrame>, VideoError> {
        use super::video::{DmaBufPlane, LinuxGpuSurface};
        use gstreamer_allocators::DmaBufMemory;
        use std::os::fd::RawFd;

        let format = video_info.format();
        let num_planes = video_info.n_planes() as usize;

        // Map GStreamer format to our PixelFormat
        let pixel_format = match format {
            gstreamer_video::VideoFormat::Nv12 => PixelFormat::Nv12,
            gstreamer_video::VideoFormat::I420 => PixelFormat::Yuv420p,
            _ => {
                tracing::debug!(
                    "MoQ GStreamer: Unsupported format {:?} for zero-copy",
                    format
                );
                return Ok(None);
            }
        };

        // Check if memory is DMABuf
        let Some(memory) = buffer.memory(0) else {
            return Ok(None);
        };

        if !memory.is_memory_type::<DmaBufMemory>() {
            tracing::trace!("MoQ GStreamer: Buffer is not DMABuf, using CPU path");
            return Ok(None);
        }

        // Determine if single-FD or multi-FD layout
        let n_memory = buffer.n_memory();
        let multi_fd = n_memory >= num_planes && num_planes > 1;
        let is_single_fd = num_planes > 1 && !multi_fd;

        // Extract per-plane metadata
        let mut planes: Vec<DmaBufPlane> = Vec::with_capacity(num_planes);
        let mut primary_dup_fd: RawFd = -1;

        for plane_idx in 0..num_planes {
            let mem_idx = if multi_fd { plane_idx } else { 0 };
            let Some(plane_memory) = buffer.memory(mem_idx) else {
                tracing::warn!(
                    "MoQ GStreamer: Plane {} missing memory at index {}",
                    plane_idx,
                    mem_idx
                );
                return Ok(None);
            };

            if !plane_memory.is_memory_type::<DmaBufMemory>() {
                tracing::warn!("MoQ GStreamer: Plane {} is not DMABuf", plane_idx);
                return Ok(None);
            }

            let dmabuf_memory = plane_memory
                .downcast_memory_ref::<DmaBufMemory>()
                .ok_or_else(|| {
                    VideoError::DecodeFailed("Failed to downcast to DmaBufMemory".to_string())
                })?;

            let gst_fd: RawFd = dmabuf_memory.fd();
            if gst_fd < 0 {
                tracing::warn!("MoQ GStreamer: Plane {} has invalid fd", plane_idx);
                return Ok(None);
            }

            // Dup the FD (Vulkan takes ownership)
            let fd: RawFd = if is_single_fd {
                if plane_idx == 0 {
                    let dup_fd = unsafe { libc::dup(gst_fd) };
                    if dup_fd < 0 {
                        tracing::warn!("MoQ GStreamer: Failed to dup fd for plane {}", plane_idx);
                        return Ok(None);
                    }
                    primary_dup_fd = dup_fd;
                    dup_fd
                } else {
                    primary_dup_fd
                }
            } else {
                let dup_fd = unsafe { libc::dup(gst_fd) };
                if dup_fd < 0 {
                    // Clean up already-dup'd fds
                    for plane in &planes {
                        unsafe { libc::close(plane.fd) };
                    }
                    return Ok(None);
                }
                dup_fd
            };

            // Get stride and offset from VideoInfo
            let Some(&stride_i32) = video_info.stride().get(plane_idx) else {
                self.close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            };
            let Some(&offset_usize) = video_info.offset().get(plane_idx) else {
                self.close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            };

            if stride_i32 < 0 {
                self.close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            }

            planes.push(DmaBufPlane {
                fd,
                offset: offset_usize as u64,
                stride: stride_i32 as u32,
                size: plane_memory.size() as u64,
            });
        }

        // Parse DRM modifier from caps (GStreamer 1.24+)
        let modifier = Self::parse_drm_modifier_from_sample(sample).unwrap_or(0);

        // Extract CPU fallback data
        let cpu_fallback = self.extract_cpu_fallback(buffer, video_info, width, height);

        // Create LinuxGpuSurface
        // Keep sample alive via Arc
        let sample_owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(sample.clone());

        let surface = unsafe {
            LinuxGpuSurface::new(
                planes,
                width,
                height,
                pixel_format,
                modifier,
                is_single_fd,
                cpu_fallback,
                sample_owner,
            )
        };

        tracing::debug!(
            "MoQ GStreamer: Extracted DMABuf frame {}x{} {:?}",
            width,
            height,
            pixel_format
        );

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Linux(surface))))
    }

    /// Helper to close FDs on error during DMABuf extraction.
    fn close_fds_on_error(
        &self,
        planes: &[super::video::DmaBufPlane],
        current_fd: std::os::fd::RawFd,
        is_single_fd: bool,
    ) {
        if is_single_fd {
            if current_fd >= 0 {
                unsafe { libc::close(current_fd) };
            }
        } else {
            for plane in planes {
                unsafe { libc::close(plane.fd) };
            }
            if current_fd >= 0 {
                unsafe { libc::close(current_fd) };
            }
        }
    }

    /// Parses DRM modifier from GStreamer caps (GStreamer 1.24+).
    fn parse_drm_modifier_from_sample(sample: &gstreamer::Sample) -> Option<u64> {
        let caps = sample.caps()?;
        let structure = caps.structure(0)?;

        let drm_format: String = structure.get("drm-format").ok()?;

        // Parse "NV12:0x0100000000000002" format
        let modifier_str = drm_format.split(':').nth(1)?;

        let modifier = if modifier_str.starts_with("0x") || modifier_str.starts_with("0X") {
            u64::from_str_radix(&modifier_str[2..], 16).ok()?
        } else {
            modifier_str.parse::<u64>().ok()?
        };

        tracing::debug!(
            "MoQ GStreamer: Parsed DRM modifier 0x{:016x} from '{}'",
            modifier,
            drm_format
        );

        Some(modifier)
    }

    /// Extracts CPU frame data for fallback when zero-copy fails.
    fn extract_cpu_fallback(
        &self,
        buffer: &gstreamer::BufferRef,
        video_info: &gstreamer_video::VideoInfo,
        width: u32,
        height: u32,
    ) -> Option<CpuFrame> {
        use gstreamer_video::VideoFormat;

        let map = buffer.map_readable().ok()?;
        let data = map.as_slice();
        let format = video_info.format();

        match format {
            VideoFormat::Nv12 => {
                let strides = video_info.stride();
                let offsets = video_info.offset();

                let y_stride = (*strides.first()?) as usize;
                let uv_stride = (*strides.get(1)?) as usize;
                let y_offset = *offsets.first()?;
                let uv_offset = *offsets.get(1)?;

                let y_size = y_stride * height as usize;
                let uv_size = uv_stride * (height as usize).div_ceil(2);

                if y_offset + y_size > data.len() || uv_offset + uv_size > data.len() {
                    return None;
                }

                let y_data = data[y_offset..y_offset + y_size].to_vec();
                let uv_data = data[uv_offset..uv_offset + uv_size].to_vec();

                Some(CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride: y_stride,
                        },
                        Plane {
                            data: uv_data,
                            stride: uv_stride,
                        },
                    ],
                ))
            }
            VideoFormat::I420 => {
                let strides = video_info.stride();
                let offsets = video_info.offset();

                let y_stride = (*strides.first()?) as usize;
                let u_stride = (*strides.get(1)?) as usize;
                let v_stride = (*strides.get(2)?) as usize;
                let y_offset = *offsets.first()?;
                let u_offset = *offsets.get(1)?;
                let v_offset = *offsets.get(2)?;

                let y_size = y_stride * height as usize;
                let uv_height = (height as usize).div_ceil(2);
                let u_size = u_stride * uv_height;
                let v_size = v_stride * uv_height;

                if y_offset + y_size > data.len()
                    || u_offset + u_size > data.len()
                    || v_offset + v_size > data.len()
                {
                    return None;
                }

                let y_data = data[y_offset..y_offset + y_size].to_vec();
                let u_data = data[u_offset..u_offset + u_size].to_vec();
                let v_data = data[v_offset..v_offset + v_size].to_vec();

                Some(CpuFrame::new(
                    PixelFormat::Yuv420p,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride: y_stride,
                        },
                        Plane {
                            data: u_data,
                            stride: u_stride,
                        },
                        Plane {
                            data: v_data,
                            stride: v_stride,
                        },
                    ],
                ))
            }
            _ => None,
        }
    }

    /// Converts a GStreamer buffer to a CPU frame (fallback path).
    fn buffer_to_cpu_frame(
        &self,
        buffer: &gstreamer::BufferRef,
        video_info: &gstreamer_video::VideoInfo,
        pts: Duration,
        width: u32,
        height: u32,
    ) -> Result<Option<VideoFrame>, VideoError> {
        let cpu_frame = self
            .extract_cpu_fallback(buffer, video_info, width, height)
            .ok_or_else(|| VideoError::DecodeFailed("Failed to extract CPU frame".to_string()))?;

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
    }

    /// Returns true if this URL is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        MoqUrl::is_moq_url(url)
    }

    /// Returns the current decoder state.
    pub fn decoder_state(&self) -> MoqDecoderState {
        *self.shared.state.lock()
    }

    /// Returns the error message if in error state.
    pub fn error_message(&self) -> Option<String> {
        self.shared.error_message.lock().clone()
    }
}

#[cfg(target_os = "linux")]
impl Drop for MoqGStreamerDecoder {
    fn drop(&mut self) {
        use gstreamer::prelude::*;

        tracing::debug!("MoQ: MoqGStreamerDecoder dropped (nal_rx closing)");

        // Signal EOS to appsrc
        let _ = self.appsrc.end_of_stream();

        // Stop pipeline
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}

#[cfg(target_os = "linux")]
impl VideoDecoderBackend for MoqGStreamerDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Sync cached metadata from shared state (safe copy under lock)
        {
            let shared_metadata = self.shared.metadata.lock();
            if shared_metadata.width != self.cached_metadata.width
                || shared_metadata.height != self.cached_metadata.height
            {
                self.cached_metadata = shared_metadata.clone();
            }
        }

        // Check for EOF
        if self.shared.eof_reached.load(Ordering::Relaxed) {
            // Signal EOS to appsrc if not already done
            let _ = self.appsrc.end_of_stream();
            return Ok(None);
        }

        // Check for errors
        let state = *self.shared.state.lock();
        if state == MoqDecoderState::Error {
            return Err(VideoError::DecodeFailed(
                self.error_message()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        // Push any available NAL units to GStreamer
        while let Ok(moq_frame) = self.nal_rx.try_recv() {
            self.push_nal_unit(&moq_frame)?;
        }

        // Pull decoded frame from appsink
        self.pull_decoded_frame()
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        // Live streams don't support seeking
        Err(VideoError::SeekFailed(
            "Seeking is not supported on live MoQ streams".to_string(),
        ))
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.cached_metadata
    }

    fn duration(&self) -> Option<Duration> {
        // Live streams have no duration
        None
    }

    fn is_eof(&self) -> bool {
        self.shared.eof_reached.load(Ordering::Relaxed)
    }

    fn buffering_percent(&self) -> i32 {
        self.shared.buffering_percent.load(Ordering::Relaxed)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.active_hw_type
    }

    fn handles_audio_internally(&self) -> bool {
        self.shared
            .audio
            .internal_audio_ready
            .load(Ordering::Relaxed)
    }

    fn audio_handle(&self) -> Option<super::audio::AudioHandle> {
        self.shared.audio.moq_audio_handle.lock().clone()
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_muted = muted;
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        self.audio_volume = volume.clamp(0.0, 1.0);
        Ok(())
    }
}

// Re-export Android types at module level
#[cfg(target_os = "android")]
pub use android::MoqAndroidDecoder;

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
        assert_eq!(config.max_latency_ms, 500);
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
        use super::super::android::CodecType;

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
