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
#[cfg(not(target_os = "macos"))]
use super::video::{CpuFrame, Plane};
use super::video::{
    DecodedFrame, HwAccelType, PixelFormat, VideoDecoderBackend, VideoError, VideoFrame,
    VideoMetadata,
};
#[cfg(target_os = "macos")]
use super::video_decoder::HwAccelConfig;

use async_channel::{Receiver, Sender};
use bytes::Buf;
use parking_lot::Mutex;
use tokio::runtime::Handle;

// macOS-specific imports for VTDecompressionSession zero-copy
#[cfg(target_os = "macos")]
use super::video::MacOSGpuSurface;

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
}

impl MoqAudioShared {
    /// Creates a new `MoqAudioShared` with all fields in their default (unavailable) state.
    pub fn new() -> Self {
        Self {
            internal_audio_ready: std::sync::atomic::AtomicBool::new(false),
            moq_audio_handle: parking_lot::Mutex::new(None),
            audio_status: parking_lot::Mutex::new(MoqAudioStatus::Unavailable),
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
    /// Default: 60 (~1.2 s of AAC at 48 kHz / 1024-sample frames).
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
            audio_buffer_capacity: 60,
        }
    }
}

/// A received video frame from MoQ, ready for decoding.
#[derive(Debug)]
struct MoqVideoFrame {
    /// Presentation timestamp in microseconds
    timestamp_us: u64,
    /// Encoded frame data (H.264/H.265/AV1 NAL units)
    data: bytes::Bytes,
    /// Whether this is a keyframe
    is_keyframe: bool,
}

/// Frame statistics for debugging pipeline issues.
#[derive(Default)]
struct FrameStats {
    /// Frames received from hang crate
    received: AtomicU64,
    /// Frames dropped due to channel backpressure
    dropped_backpressure: AtomicU64,
    /// Frames dropped waiting for IDR
    dropped_waiting_idr: AtomicU64,
    /// Frames passed to VT decoder
    submitted_to_decoder: AtomicU64,
    /// Frames decoded successfully (VT callback)
    decoded: AtomicU64,
    /// Frames returned to renderer
    rendered: AtomicU64,
    /// Decode errors
    decode_errors: AtomicU64,
}

impl FrameStats {
    fn log_summary(&self, label: &str) {
        let received = self.received.load(Ordering::Relaxed);
        let dropped_bp = self.dropped_backpressure.load(Ordering::Relaxed);
        let dropped_idr = self.dropped_waiting_idr.load(Ordering::Relaxed);
        let submitted = self.submitted_to_decoder.load(Ordering::Relaxed);
        let decoded = self.decoded.load(Ordering::Relaxed);
        let rendered = self.rendered.load(Ordering::Relaxed);
        let errors = self.decode_errors.load(Ordering::Relaxed);

        tracing::info!(
            "MoQ FrameStats [{}]: recv={}, drop_bp={}, drop_idr={}, submit={}, decoded={}, rendered={}, errors={}",
            label, received, dropped_bp, dropped_idr, submitted, decoded, rendered, errors
        );
    }

    fn snapshot(&self) -> MoqFrameStatsSnapshot {
        MoqFrameStatsSnapshot {
            received: self.received.load(Ordering::Relaxed),
            dropped_backpressure: self.dropped_backpressure.load(Ordering::Relaxed),
            dropped_waiting_idr: self.dropped_waiting_idr.load(Ordering::Relaxed),
            submitted_to_decoder: self.submitted_to_decoder.load(Ordering::Relaxed),
            decoded: self.decoded.load(Ordering::Relaxed),
            rendered: self.rendered.load(Ordering::Relaxed),
            decode_errors: self.decode_errors.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of MoQ frame pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct MoqFrameStatsSnapshot {
    pub received: u64,
    pub dropped_backpressure: u64,
    pub dropped_waiting_idr: u64,
    pub submitted_to_decoder: u64,
    pub decoded: u64,
    pub rendered: u64,
    pub decode_errors: u64,
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
    pub fn snapshot(&self) -> MoqStatsSnapshot {
        let metadata = self.shared.metadata.lock().clone();
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
        }
    }
}

/// Shared state between decoder and async worker.
struct MoqSharedState {
    /// Current decoder state
    state: Mutex<MoqDecoderState>,
    /// Error message if in error state
    error_message: Mutex<Option<String>>,
    /// Whether EOF has been reached
    eof_reached: AtomicBool,
    /// Buffering percentage (0-100)
    buffering_percent: AtomicI32,
    /// Video metadata (populated after catalog received)
    metadata: Mutex<VideoMetadata>,
    /// Audio track info (if available)
    audio_info: Mutex<Option<AudioTrackInfo>>,
    /// Codec description from catalog (avcC/hvcC box containing SPS/PPS)
    codec_description: Mutex<Option<bytes::Bytes>>,
    /// Frame statistics for debugging
    frame_stats: FrameStats,
    /// Transport protocol used (QUIC or WebSocket)
    transport_protocol: Mutex<String>,
    /// Audio-specific shared state, passed to MoqAudioThread on desktop.
    /// Present on all platforms (non-desktop stays Unavailable).
    audio: Arc<MoqAudioShared>,
}

impl MoqSharedState {
    fn new() -> Self {
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
            codec_description: Mutex::new(None),
            frame_stats: FrameStats::default(),
            transport_protocol: Mutex::new("unknown".to_string()),
            audio: Arc::new(MoqAudioShared::new()),
        }
    }

    fn set_state(&self, state: MoqDecoderState) {
        *self.state.lock() = state;
    }

    fn set_error(&self, message: String) {
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
    /// True after a decode error; decoder must wait for next IDR to resync.
    #[cfg(target_os = "macos")]
    waiting_for_idr_after_error: bool,
}

impl MoqDecoder {
    /// Creates a new MoQ decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, MoqDecoderConfig::default())
    }

    /// Creates a new MoQ decoder with explicit configuration.
    pub fn new_with_config(url: &str, config: MoqDecoderConfig) -> Result<Self, VideoError> {
        tracing::info!("MoqDecoder::new_with_config: creating decoder for {}", url);

        // Parse the MoQ URL
        let moq_url = MoqUrl::parse(url).map_err(|e| {
            tracing::error!("MoQ: failed to parse URL: {}", e);
            VideoError::OpenFailed(e.to_string())
        })?;

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
            waiting_for_idr_after_error: false,
        })
    }

    /// Async worker that handles MoQ connection, catalog fetching, and frame receipt.
    async fn run_moq_worker(
        shared: Arc<MoqSharedState>,
        url: MoqUrl,
        config: MoqDecoderConfig,
        frame_tx: Sender<MoqVideoFrame>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use moq_lite::{Origin, Path};
        use moq_native::ClientConfig;

        // Update state to connecting
        shared.set_state(MoqDecoderState::Connecting);
        shared.buffering_percent.store(10, Ordering::Relaxed);

        // Build connection URL (moq-native expects https:// or http://)
        //
        // URL structure differs between relay types:
        // 1. cdn.moq.dev style: moqs://cdn.moq.dev/demo/bbb?jwt=xxx
        //    - namespace = "demo" (auth namespace)
        //    - track = "bbb" (specific broadcast)
        //    - Connect to: https://cdn.moq.dev/demo?jwt=xxx
        //    - Look for broadcast at path: "bbb" (if specified) or auto-discover
        //
        // 2. zap.stream style: moq://api-core.zap.stream:1443/537a365c-...
        //    - namespace = "537a365c-..." (actually the broadcast ID!)
        //    - track = None
        //    - Connect to: https://api-core.zap.stream:1443/ (base URL)
        //    - Look for broadcast at path: "537a365c-..." (namespace IS the broadcast)
        //
        // Detection: zap.stream URLs have UUID-like namespaces (broadcast IDs)
        // while cdn.moq.dev uses short namespaces like "anon", "demo"
        //
        // NOTE: QUIC/WebTransport requires TLS. Following zap.stream's approach:
        // - Always use https:// for production servers (even if moq:// scheme was used)
        // - Only use http:// for localhost development (when moq:// scheme is explicitly used)
        let is_localhost =
            url.host() == "localhost" || url.host() == "127.0.0.1" || url.host() == "::1";
        let scheme = if url.use_tls() || !is_localhost {
            "https"
        } else {
            "http"
        };

        // Check if namespace looks like a UUID (zap.stream broadcast ID)
        let namespace_is_broadcast_id = url.namespace().len() >= 32
            && url
                .namespace()
                .chars()
                .all(|c| c.is_ascii_hexdigit() || c == '-');

        let (connect_url, broadcast_path) = if namespace_is_broadcast_id {
            // zap.stream style: connect to base URL, namespace is the broadcast path
            let base = format!("{}://{}", scheme, url.server_addr());
            let connect = match url.query() {
                Some(query) => format!("{}?{}", base, query),
                None => base,
            };
            (connect, Some(Path::from(url.namespace())))
        } else {
            // cdn.moq.dev style: include namespace in connection URL
            let base = format!("{}://{}/{}", scheme, url.server_addr(), url.namespace());
            let connect = match url.query() {
                Some(query) => format!("{}?{}", base, query),
                None => base,
            };
            // Use track (if specified) as the broadcast path
            (connect, url.track().map(Path::from))
        };

        let redacted_connect_url = connect_url
            .split_once('?')
            .map(|(base, _)| format!("{}?<redacted>", base))
            .unwrap_or_else(|| connect_url.clone());
        tracing::info!("MoQ: Connecting to {}", redacted_connect_url);
        if let Some(ref path) = broadcast_path {
            tracing::info!("MoQ: Will look for broadcast at path: {:?}", path);
        }

        let parsed_url: url::Url = connect_url
            .parse()
            .map_err(|e| format!("Invalid URL: {e}"))?;

        // Two-phase connect: try QUIC first, then WebSocket fallback.
        // Each phase creates its own origin since with_consume() takes ownership.
        let quic_probe_timeout =
            Duration::from_millis(config.transport.connect_timeout_ms.min(1500));
        // The session must be kept alive for the entire worker lifetime —
        // dropping it terminates the connection.
        let (mut origin_consumer, transport_protocol, _session) = if config
            .transport
            .websocket_fallback
        {
            // Phase 1: QUIC-only (capped at 1500ms)
            tracing::info!("MoQ: Trying QUIC connection...");
            let quic_result = {
                let mut cfg = ClientConfig::default();
                if config.transport.disable_tls_verify {
                    cfg.tls.disable_verify = Some(true);
                }
                cfg.websocket.enabled = false;
                let origin = Origin::produce();
                let consumer = origin.consume();
                match cfg.init() {
                    Ok(client) => {
                        match tokio::time::timeout(
                            quic_probe_timeout,
                            client.with_consume(origin).connect(parsed_url.clone()),
                        )
                        .await
                        {
                            Ok(Ok(session)) => Ok((consumer, session)),
                            Ok(Err(e)) => {
                                tracing::debug!("MoQ: QUIC connect error: {}", e);
                                Err(())
                            }
                            Err(_) => {
                                tracing::debug!("MoQ: QUIC timed out ({:?})", quic_probe_timeout);
                                Err(())
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("MoQ: QUIC client init failed: {}", e);
                        Err(())
                    }
                }
            };

            match quic_result {
                Ok((consumer, session)) => {
                    tracing::info!("MoQ: Connected via QUIC");
                    (consumer, "QUIC", session)
                }
                Err(()) => {
                    // Phase 2: WebSocket fallback
                    tracing::info!("MoQ: QUIC unavailable, connecting via WebSocket...");
                    let mut cfg = ClientConfig::default();
                    if config.transport.disable_tls_verify {
                        cfg.tls.disable_verify = Some(true);
                    }
                    cfg.websocket.enabled = true;
                    cfg.websocket.delay = Some(Duration::ZERO);
                    let origin = Origin::produce();
                    let consumer = origin.consume();
                    let client = cfg
                        .init()
                        .map_err(|e| format!("WebSocket client init: {e}"))?;
                    let timeout = Duration::from_millis(config.transport.connect_timeout_ms);
                    let session = tokio::time::timeout(
                        timeout,
                        client.with_consume(origin).connect(parsed_url.clone()),
                    )
                    .await
                    .map_err(|_| "Connection timed out")?
                    .map_err(|e| format!("WebSocket connection failed: {e}"))?;
                    tracing::info!("MoQ: Connected via WebSocket");
                    (consumer, "WebSocket", session)
                }
            }
        } else {
            // WebSocket disabled, QUIC only
            let mut cfg = ClientConfig::default();
            if config.transport.disable_tls_verify {
                cfg.tls.disable_verify = Some(true);
            }
            cfg.websocket.enabled = false;
            let origin = Origin::produce();
            let consumer = origin.consume();
            let client = cfg
                .init()
                .map_err(|e| format!("Failed to init MoQ client: {e}"))?;
            let timeout = Duration::from_millis(config.transport.connect_timeout_ms);
            let session = tokio::time::timeout(
                timeout,
                client.with_consume(origin).connect(parsed_url.clone()),
            )
            .await
            .map_err(|_| "Connection timed out")?
            .map_err(|e| format!("Connection failed: {e}"))?;
            tracing::info!("MoQ: Connected via QUIC");
            (consumer, "QUIC", session)
        };

        *shared.transport_protocol.lock() = transport_protocol.to_string();
        tracing::info!(
            "MoQ: Connected ({}), waiting for broadcast announcement",
            transport_protocol
        );

        // Update state to fetching catalog
        shared.set_state(MoqDecoderState::FetchingCatalog);
        shared.buffering_percent.store(30, Ordering::Relaxed);

        // Use the broadcast_path determined earlier (either from namespace for zap.stream
        // or from track for cdn.moq.dev style URLs)
        let specific_broadcast = broadcast_path;

        if let Some(ref path) = specific_broadcast {
            tracing::info!("MoQ: Looking for specific broadcast: {:?}", path);
        } else {
            tracing::info!("MoQ: Auto-discovering first available broadcast...");
        }

        // Wait for the broadcast to be announced, or check if it already exists
        // Use a timeout to avoid hanging forever when no broadcasts are available
        let discovery_timeout = Duration::from_secs(10);
        let discovery_start = std::time::Instant::now();

        let moq_broadcast: moq_lite::BroadcastConsumer = loop {
            // Check if we've exceeded the discovery timeout
            if discovery_start.elapsed() > discovery_timeout {
                let msg = if specific_broadcast.is_some() {
                    format!(
                        "Broadcast discovery timeout - '{}' not found after {:?}",
                        url.track().unwrap_or("unknown"),
                        discovery_timeout
                    )
                } else {
                    format!(
                        "Broadcast discovery timeout - no broadcasts found on '{}' after {:?}",
                        url.namespace(),
                        discovery_timeout
                    )
                };
                return Err(msg.into());
            }

            // If looking for specific broadcast, check if already available
            if let Some(ref path) = specific_broadcast {
                if let Some(broadcast) = origin_consumer.consume_broadcast(path.clone()) {
                    tracing::info!("MoQ: Found specific broadcast at {:?}", path);
                    break broadcast;
                }
            }

            // Wait for announcements with a short timeout to allow checking overall timeout
            let wait_result =
                tokio::time::timeout(Duration::from_secs(2), origin_consumer.announced()).await;

            match wait_result {
                Ok(Some((path, Some(broadcast)))) => {
                    if let Some(ref wanted) = specific_broadcast {
                        // Looking for specific broadcast
                        if path == *wanted {
                            tracing::info!("MoQ: Found matching broadcast at {:?}", path);
                            break broadcast;
                        } else {
                            tracing::debug!(
                                "MoQ: Ignoring broadcast at {:?}, waiting for {:?}",
                                path,
                                wanted
                            );
                            continue;
                        }
                    } else {
                        // Auto-discovery: use the first broadcast we find
                        tracing::info!("MoQ: Auto-selected broadcast: {:?}", path);
                        break broadcast;
                    }
                }
                Ok(Some((path, None))) => {
                    // Broadcast was unannounced, continue waiting
                    tracing::debug!("MoQ: Broadcast unannounced at {:?}", path);
                    continue;
                }
                Ok(None) => {
                    return Err("Origin consumer closed without broadcast".into());
                }
                Err(_) => {
                    // Timeout on this iteration, continue to check overall timeout
                    tracing::debug!("MoQ: Still waiting for broadcast announcement...");
                    continue;
                }
            }
        };

        tracing::info!("MoQ: Found broadcast, subscribing to tracks");
        shared.buffering_percent.store(50, Ordering::Relaxed);

        // Wrap the moq_lite::BroadcastConsumer with hang::BroadcastConsumer
        // which automatically subscribes to the catalog track
        let hang_consumer: hang::BroadcastConsumer = moq_broadcast.into();

        // Wait for catalog to be available (with timeout for stale broadcasts)
        let mut catalog_consumer = hang_consumer.catalog.clone();
        let catalog_timeout = Duration::from_secs(5);
        let catalog = match tokio::time::timeout(catalog_timeout, catalog_consumer.next()).await {
            Ok(Ok(Some(catalog))) => catalog,
            Ok(Ok(None)) => {
                return Err(
                    "Catalog track ended before receiving catalog (broadcast may be offline)"
                        .into(),
                );
            }
            Ok(Err(e)) => {
                return Err(format!("Failed to receive catalog: {e}").into());
            }
            Err(_) => {
                return Err(
                    "Catalog timeout - broadcast may be offline or has no active video".into(),
                );
            }
        };

        tracing::info!("MoQ: Received catalog");

        // Log full catalog contents for debugging
        if let Some(ref video) = catalog.video {
            tracing::info!("MoQ catalog: {} video rendition(s)", video.renditions.len());
            for (name, cfg) in &video.renditions {
                tracing::info!(
                    "  video '{}': codec={:?}, {}x{}, {:.1}fps, bitrate={:?}, description={} bytes, container={:?}, jitter={:?}",
                    name,
                    cfg.codec,
                    cfg.coded_width.unwrap_or(0),
                    cfg.coded_height.unwrap_or(0),
                    cfg.framerate.unwrap_or(0.0),
                    cfg.bitrate,
                    cfg.description.as_ref().map(|d| d.len()).unwrap_or(0),
                    cfg.container,
                    cfg.jitter,
                );
            }
        } else {
            tracing::warn!("MoQ catalog: no video section");
        }
        if let Some(ref audio) = catalog.audio {
            tracing::info!("MoQ catalog: {} audio rendition(s)", audio.renditions.len());
            for (name, cfg) in &audio.renditions {
                tracing::info!(
                    "  audio '{}': codec={:?}, sample_rate={}, channels={}, bitrate={:?}, description={} bytes",
                    name,
                    cfg.codec,
                    cfg.sample_rate,
                    cfg.channel_count,
                    cfg.bitrate,
                    cfg.description.as_ref().map(|d| d.len()).unwrap_or(0),
                );
            }
        } else {
            tracing::info!("MoQ catalog: no audio section");
        }

        // Find the first video rendition in the catalog
        let (video_track_name, video_config) = catalog
            .video
            .as_ref()
            .and_then(|v| v.renditions.iter().next())
            .ok_or("No video track in catalog")?;

        // Validate container format — we only support Legacy (raw NAL units)
        match video_config.container {
            hang::catalog::Container::Legacy => {
                tracing::info!("MoQ: Container format: Legacy (raw frames)");
            }
            hang::catalog::Container::Cmaf {
                timescale,
                track_id,
            } => {
                return Err(format!(
                    "Unsupported container format: CMAF (timescale={}, track_id={}). \
                     lumina-video only supports Legacy (raw NAL units).",
                    timescale, track_id
                )
                .into());
            }
        }

        // Update metadata from catalog
        {
            let mut metadata = shared.metadata.lock();
            metadata.width = video_config.coded_width.unwrap_or(1920);
            metadata.height = video_config.coded_height.unwrap_or(1080);
            metadata.frame_rate = video_config.framerate.unwrap_or(30.0) as f32;
            metadata.codec = format!("{:?}", video_config.codec);
        }

        // Store codec description (avcC/hvcC box with SPS/PPS) if present
        if let Some(ref desc) = video_config.description {
            tracing::info!(
                "MoQ: Got codec description from catalog ({} bytes)",
                desc.len()
            );
            *shared.codec_description.lock() = Some(desc.clone());
        } else {
            tracing::warn!(
                "MoQ: No codec description in catalog, will try to extract from keyframes"
            );
        }

        // Determine max latency: use catalog jitter if available, otherwise config default
        let max_latency = if let Some(jitter) = video_config.jitter {
            let jitter_ms = jitter.as_millis() as u64;
            // Use jitter as floor, but at least the configured value
            let effective_ms = jitter_ms.max(config.max_latency_ms);
            tracing::info!(
                "MoQ: Using catalog jitter {}ms (effective latency: {}ms)",
                jitter_ms,
                effective_ms,
            );
            Duration::from_millis(effective_ms)
        } else {
            tracing::info!(
                "MoQ: No jitter in catalog, using default {}ms",
                config.max_latency_ms,
            );
            Duration::from_millis(config.max_latency_ms)
        };

        // Create the track to subscribe to
        let video_track = moq_lite::Track {
            name: video_track_name.clone(),
            priority: 1,
        };

        // Subscribe to the video track
        let mut video_consumer = hang_consumer.subscribe(&video_track, max_latency);

        // Audio track selection and subscription
        use super::moq_audio::*;

        let (mut audio_consumer_opt, audio_sender_opt, mut moq_audio_thread_opt) =
            if config.enable_audio {
                if let Some((track_name, audio_cfg)) = select_preferred_audio_rendition(&catalog) {
                    *shared.audio.audio_status.lock() = MoqAudioStatus::Starting;

                    let audio_track = moq_lite::Track {
                        name: track_name.to_string(),
                        priority: 2,
                    };
                    let audio_consumer = hang_consumer.subscribe(&audio_track, max_latency);

                    let (tx, rx) = crossbeam_channel::bounded(config.audio_buffer_capacity);
                    let live_sender = LiveEdgeSender::new(tx.clone(), rx.clone());

                    let audio_handle = super::audio::AudioHandle::new();
                    *shared.audio.moq_audio_handle.lock() = Some(audio_handle.clone());

                    match MoqAudioThread::spawn(
                        rx,
                        audio_cfg.sample_rate,
                        audio_cfg.channel_count,
                        audio_cfg.description.clone(),
                        audio_handle,
                        shared.audio.clone(),
                    ) {
                        Ok(thread) => {
                            tracing::info!(
                                "MoQ: Audio subscribed to track '{}' ({}Hz, {}ch)",
                                track_name,
                                audio_cfg.sample_rate,
                                audio_cfg.channel_count,
                            );
                            (Some(audio_consumer), Some(live_sender), Some(thread))
                        }
                        Err(e) => {
                            tracing::warn!("MoQ: Failed to start audio thread: {e}");
                            *shared.audio.audio_status.lock() = MoqAudioStatus::Error;
                            *shared.audio.moq_audio_handle.lock() = None;
                            (None, None, None)
                        }
                    }
                } else {
                    tracing::info!("MoQ: No AAC audio track in catalog");
                    *shared.audio.audio_status.lock() = MoqAudioStatus::Unavailable;
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };

        // Update state to streaming
        shared.set_state(MoqDecoderState::Streaming);
        shared.buffering_percent.store(100, Ordering::Relaxed);

        tracing::info!(
            "MoQ: Streaming started, subscribed to video track '{}'",
            video_track_name
        );

        // Late-join IDR recovery: try fetching the previous group which should
        // start with an IDR frame. The relay caches groups for ~30s (libmoq v0.2.6),
        // so on late-join we can request the previous GOP to get a clean decoder start.
        //
        // Strategy: peek at the first available group's sequence number. If > 0,
        // fetch group (sequence - 1) and send its frames before continuing live.
        if let Some(first_group) = video_consumer.inner.next_group().await.transpose() {
            match first_group {
                Ok(group) => {
                    let seq = group.info.sequence;
                    tracing::info!("MoQ: First group sequence={}", seq);

                    if seq > 0 {
                        // Try fetching previous group for IDR recovery
                        let prev_seq = seq - 1;
                        tracing::info!(
                            "MoQ: Late-join detected (seq={}), fetching previous group {} for IDR",
                            seq,
                            prev_seq,
                        );

                        match tokio::time::timeout(
                            Duration::from_millis(500),
                            video_consumer.inner.get_group(prev_seq),
                        )
                        .await
                        {
                            Ok(Ok(Some(prev_group))) => {
                                tracing::info!(
                                    "MoQ: Got previous group {} for IDR recovery",
                                    prev_seq,
                                );
                                // Read all frames from the previous group and send them
                                let mut prev_hang_group = hang::GroupConsumer::new(prev_group);
                                while let Ok(Some(frame)) = prev_hang_group.read().await {
                                    let mut data =
                                        bytes::BytesMut::with_capacity(frame.payload.remaining());
                                    for chunk in &frame.payload {
                                        data.extend_from_slice(chunk);
                                    }
                                    let moq_frame = MoqVideoFrame {
                                        timestamp_us: frame.timestamp.as_micros() as u64,
                                        data: data.freeze(),
                                        is_keyframe: frame.keyframe,
                                    };
                                    shared.frame_stats.received.fetch_add(1, Ordering::Relaxed);
                                    if frame_tx.send(moq_frame).await.is_err() {
                                        tracing::warn!(
                                            "MoQ: Frame channel closed during IDR recovery"
                                        );
                                        break;
                                    }
                                }
                                tracing::info!(
                                    "MoQ: IDR recovery complete, continuing with live group"
                                );
                            }
                            Ok(Ok(None)) => {
                                tracing::info!("MoQ: Previous group {} not cached (dropped), skipping IDR recovery", prev_seq);
                            }
                            Ok(Err(e)) => {
                                tracing::warn!(
                                    "MoQ: Failed to fetch previous group {}: {}",
                                    prev_seq,
                                    e
                                );
                            }
                            Err(_) => {
                                tracing::warn!("MoQ: Timeout fetching previous group {}", prev_seq);
                            }
                        }
                    }

                    // Now read frames from the current (first) group we already received
                    let mut current_group = hang::GroupConsumer::new(group);
                    while let Ok(Some(frame)) = current_group.read().await {
                        let mut data = bytes::BytesMut::with_capacity(frame.payload.remaining());
                        for chunk in &frame.payload {
                            data.extend_from_slice(chunk);
                        }
                        let moq_frame = MoqVideoFrame {
                            timestamp_us: frame.timestamp.as_micros() as u64,
                            data: data.freeze(),
                            is_keyframe: frame.keyframe,
                        };
                        shared.frame_stats.received.fetch_add(1, Ordering::Relaxed);
                        if frame_tx.send(moq_frame).await.is_err() {
                            tracing::warn!("MoQ: Frame channel closed during first group");
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("MoQ: Error getting first group: {}", e);
                }
            }
        }

        // Main frame receive loop with fair video/audio scheduling
        let mut stats_log_counter = 0u64;
        loop {
            tokio::select! {
                // No biased — fair scheduling prevents audio starvation
                video_result = video_consumer.read_frame() => {
                    match video_result {
                        Ok(Some(frame)) => {
                            let recv_count =
                                shared.frame_stats.received.fetch_add(1, Ordering::Relaxed) + 1;

                            stats_log_counter += 1;
                            if stats_log_counter.is_multiple_of(30) {
                                shared.frame_stats.log_summary("worker");
                            }

                            let mut data = bytes::BytesMut::with_capacity(frame.payload.remaining());
                            for chunk in &frame.payload {
                                data.extend_from_slice(chunk);
                            }

                            let moq_frame = MoqVideoFrame {
                                timestamp_us: frame.timestamp.as_micros() as u64,
                                data: data.freeze(),
                                is_keyframe: frame.keyframe,
                            };

                            if recv_count <= 5 {
                                let nal_type = if moq_frame.data.len() >= 5 {
                                    moq_frame.data[4] & 0x1F
                                } else {
                                    0
                                };
                                tracing::info!(
                                    "MoQ frame #{}: is_keyframe={}, NAL type={}, {} bytes",
                                    recv_count, moq_frame.is_keyframe, nal_type, moq_frame.data.len()
                                );
                            }

                            if frame_tx.send(moq_frame).await.is_err() {
                                tracing::warn!("MoQ: Frame channel closed, stopping worker");
                                break;
                            }
                        }
                        Ok(None) => {
                            tracing::info!("MoQ: Video track ended");
                            shared.frame_stats.log_summary("end");
                            shared.set_state(MoqDecoderState::Ended);
                            shared.eof_reached.store(true, Ordering::Relaxed);
                            break;
                        }
                        Err(e) => {
                            tracing::error!("MoQ: Frame read error: {e}");
                            shared.frame_stats.log_summary("error");
                            shared.set_error(format!("Frame read error: {e}"));
                            break;
                        }
                    }
                }
                audio_result = async {
                    if let Some(consumer) = audio_consumer_opt.as_mut() {
                        consumer.read_frame().await
                    } else {
                        std::future::pending().await
                    }
                } => {
                    if let Some(ref audio_sender) = audio_sender_opt {
                        match audio_result {
                            Ok(Some(frame)) => {
                                let mut data = bytes::BytesMut::with_capacity(frame.payload.remaining());
                                for chunk in &frame.payload {
                                    data.extend_from_slice(chunk);
                                }
                                let moq_frame = MoqAudioFrame {
                                    timestamp_us: frame.timestamp.as_micros() as u64,
                                    data: data.freeze(),
                                };
                                if let Err(ChannelClosed) = audio_sender.send(moq_frame) {
                                    tracing::warn!("MoQ: Audio channel closed");
                                    audio_consumer_opt = None;
                                }
                            }
                            Ok(None) => {
                                tracing::info!("MoQ: Audio track ended");
                                audio_consumer_opt = None;
                            }
                            Err(e) => {
                                tracing::warn!("MoQ: Audio read error: {e}");
                                audio_consumer_opt = None;
                            }
                        }
                    }
                }
            }
        }

        // Worker teardown: deterministic audio thread shutdown
        drop(audio_sender_opt);
        {
            let mut status = shared.audio.audio_status.lock();
            if *status == MoqAudioStatus::Starting {
                *status = MoqAudioStatus::Unavailable;
            }
        }
        if let Some(thread) = moq_audio_thread_opt.take() {
            let shared_for_teardown = shared.clone();
            let teardown_start = std::time::Instant::now();
            let teardown_fut = tokio::task::spawn_blocking(move || drop(thread));
            match tokio::time::timeout(Duration::from_secs(2), teardown_fut).await {
                Ok(Ok(())) => tracing::debug!("MoQ: audio teardown completed"),
                Ok(Err(e)) => {
                    tracing::warn!("MoQ: audio teardown task failed: {e}");
                    *shared_for_teardown.audio.audio_status.lock() = MoqAudioStatus::Error;
                }
                Err(_) => {
                    tracing::warn!("MoQ: audio teardown timed out after 2s, proceeding");
                }
            }
            shared_for_teardown
                .audio
                .internal_audio_ready
                .store(false, Ordering::Relaxed);
            *shared_for_teardown.audio.moq_audio_handle.lock() = None;
            {
                let mut status = shared_for_teardown.audio.audio_status.lock();
                if *status == MoqAudioStatus::Running || *status == MoqAudioStatus::Starting {
                    *status = MoqAudioStatus::Unavailable;
                }
            }
            let teardown_ms = teardown_start.elapsed().as_millis();
            if teardown_ms > 250 {
                tracing::warn!("MoQ: audio teardown took {}ms", teardown_ms);
            }
        }

        Ok(())
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

    /// Checks if AVCC data contains an IDR frame (H.264 NAL type 5).
    /// The hang crate's is_keyframe flag can be wrong when joining mid-stream.
    /// AVCC sample may contain multiple NAL units (SPS/PPS/AUD before IDR),
    /// so we iterate all NALs and return true if any is type 5. The NAL length
    /// prefix size comes from avcC (lengthSizeMinusOne + 1).
    #[allow(dead_code)]
    fn is_idr_frame(nal_data: &[u8], nal_length_size: usize) -> bool {
        if !(1..=4).contains(&nal_length_size) {
            return false;
        }
        let mut offset = 0usize;
        while offset + nal_length_size <= nal_data.len() {
            let mut nal_len = 0usize;
            for i in 0..nal_length_size {
                nal_len = (nal_len << 8) | nal_data[offset + i] as usize;
            }
            offset += nal_length_size;
            if nal_len == 0 || offset + nal_len > nal_data.len() {
                break;
            }
            let nal_type = nal_data[offset] & 0x1F;
            if nal_type == 5 {
                return true;
            }
            offset += nal_len;
        }
        false
    }

    /// Gets the NAL type from AVCC data for logging.
    #[allow(dead_code)]
    fn get_nal_type(nal_data: &[u8], nal_length_size: usize) -> u8 {
        if !(1..=4).contains(&nal_length_size) {
            return 0;
        }
        if nal_data.len() > nal_length_size {
            nal_data[nal_length_size] & 0x1F
        } else {
            0
        }
    }

    /// Returns true if frame should be skipped while waiting for an IDR resync.
    #[cfg(target_os = "macos")]
    fn should_wait_for_idr(&self, moq_frame: &MoqVideoFrame) -> bool {
        if !self.waiting_for_idr_after_error {
            return false;
        }
        !Self::is_idr_frame(&moq_frame.data, self.h264_nal_length_size)
    }

    /// Decodes an encoded frame.
    ///
    /// On macOS, uses VTDecompressionSession for zero-copy hardware decoding.
    /// On other platforms, returns a placeholder (FFmpeg integration TODO).
    #[cfg(target_os = "macos")]
    fn decode_frame(&mut self, moq_frame: &MoqVideoFrame) -> Result<VideoFrame, VideoError> {
        // Initialize VTDecoder lazily
        if self.vt_decoder.is_none() {
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
                        )?;
                        self.vt_decoder = Some(decoder);
                        self.h264_nal_length_size = nal_length_size;
                        tracing::info!("MoQ: initialized VTDecoder from catalog avcC ({} bytes SPS, {} bytes PPS, NAL len size {})", sps.len(), pps.len(), nal_length_size);
                    }
                    Err(e) => {
                        tracing::warn!("MoQ: failed to parse avcC from catalog: {}", e);
                    }
                }
            }

            // If still no decoder, try extracting SPS/PPS from keyframe
            if self.vt_decoder.is_none() {
                if !moq_frame.is_keyframe {
                    tracing::debug!("MoQ: waiting for keyframe to initialize VTDecoder");
                    return Err(VideoError::DecodeFailed(
                        "Waiting for keyframe with SPS/PPS".to_string(),
                    ));
                }

                // Try to extract SPS/PPS from the keyframe data
                match Self::extract_h264_params(&moq_frame.data) {
                    Ok((sps, pps)) => {
                        let decoder = macos_vt::VTDecoder::new_h264(
                            &sps,
                            &pps,
                            metadata.width,
                            metadata.height,
                        )?;
                        self.vt_decoder = Some(decoder);
                        self.h264_nal_length_size = 4; // Default for Annex B extraction
                        tracing::info!("MoQ: initialized VTDecoder from keyframe SPS/PPS");
                    }
                    Err(e) => {
                        tracing::warn!("MoQ: failed to extract H.264 params: {}", e);
                        return Err(e);
                    }
                }
            }
        }

        // Check if we're waiting for IDR resync after a decode error
        if self.should_wait_for_idr(moq_frame) {
            let nal_type = Self::get_nal_type(&moq_frame.data, self.h264_nal_length_size);
            tracing::debug!(
                "MoQ: waiting for IDR resync after decode error (got NAL type {}, is_keyframe={}, {} bytes)",
                nal_type, moq_frame.is_keyframe, moq_frame.data.len()
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (got NAL type {})",
                nal_type
            )));
        }

        // CRITICAL: Check actual NAL type, not just is_keyframe flag.
        // When joining a MoQ stream mid-session, the first frame may be marked
        // as keyframe by hang but actually be a P-frame (NAL type 1).
        // VideoToolbox requires an actual IDR frame (NAL type 5) to start decoding.
        let is_idr = Self::is_idr_frame(&moq_frame.data, self.h264_nal_length_size);
        if let Some(ref decoder) = self.vt_decoder {
            // If we haven't decoded any frames yet, we MUST have an IDR frame
            let frame_count = decoder.frame_count();
            if frame_count == 0 && !is_idr {
                let nal_type = Self::get_nal_type(&moq_frame.data, self.h264_nal_length_size);
                tracing::debug!(
                    "MoQ: waiting for IDR frame (got NAL type {}, is_keyframe={}, {} bytes)",
                    nal_type,
                    moq_frame.is_keyframe,
                    moq_frame.data.len()
                );
                return Err(VideoError::DecodeFailed(format!(
                    "Waiting for IDR frame (got NAL type {})",
                    nal_type
                )));
            }

            // If we were waiting for IDR and got one, clear error state and prepare decoder
            if self.waiting_for_idr_after_error && is_idr {
                if let Some(ref mut decoder) = self.vt_decoder {
                    decoder.prepare_for_idr_resync();
                }
                self.waiting_for_idr_after_error = false;
                tracing::info!("MoQ: received IDR, cleared decoder error state and resyncing");
            }
        }

        // Decode the frame using VTDecoder
        let decode_result = if let Some(ref mut decoder) = self.vt_decoder {
            decoder.decode_frame(
                &moq_frame.data,
                moq_frame.timestamp_us,
                moq_frame.is_keyframe,
            )
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
                // Decode error - enter IDR resync mode
                self.waiting_for_idr_after_error = true;
                if let Some(ref mut decoder) = self.vt_decoder {
                    decoder.prepare_for_idr_resync();
                }
                tracing::warn!("MoQ: VT decode failed, entering IDR resync mode: {}", e);
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
    #[cfg(target_os = "macos")]
    fn parse_avcc_box(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, usize), VideoError> {
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
                            || msg.contains("no frame decoded") =>
                    {
                        // Track frames dropped waiting for IDR
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
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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
        /// Null allocator - performs no allocation/deallocation.
        /// Use this for caller-owned memory passed to CM functions.
        static kCFAllocatorNull: *const c_void;
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
    }

    /// Shared state for decoder callback to push decoded frames.
    struct VTCallbackState {
        /// Queue of decoded frames (protected by mutex)
        decoded_frames: ParkingMutex<VecDeque<DecodedVTFrame>>,
        /// Error flag set by callback on decode failure
        decode_error: AtomicBool,
        /// Frame counter for debugging
        frame_count: AtomicU32,
    }

    impl VTCallbackState {
        fn new() -> Self {
            Self {
                decoded_frames: ParkingMutex::new(VecDeque::with_capacity(8)),
                decode_error: AtomicBool::new(false),
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
    }

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
        ) -> Result<Self, VideoError> {
            tracing::info!(
                "VTDecoder: Creating H.264 decoder {}x{} (SPS: {} bytes, PPS: {} bytes)",
                width,
                height,
                sps.len(),
                pps.len()
            );

            // Create CMFormatDescription from SPS/PPS
            let format_desc = Self::create_h264_format_description(sps, pps)?;

            // Create decoder with format description
            Self::create_decoder(format_desc, width, height, VTCodec::H264)
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
        ) -> Result<Self, VideoError> {
            tracing::info!(
                "VTDecoder: Creating H.265 decoder {}x{} (VPS: {} bytes, SPS: {} bytes, PPS: {} bytes)",
                width,
                height,
                vps.len(),
                sps.len(),
                pps.len()
            );

            // Create CMFormatDescription from VPS/SPS/PPS
            let format_desc = Self::create_h265_format_description(vps, sps, pps)?;

            // Create decoder with format description
            Self::create_decoder(format_desc, width, height, VTCodec::H265)
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
        ) -> Result<Self, VideoError> {
            // Create output pixel buffer attributes for IOSurface + Metal compatibility
            let destination_attributes = Self::create_output_attributes()?;

            // Create callback state for receiving decoded frames
            let callback_state = Arc::new(VTCallbackState::new());

            // Create the decompression session
            let session =
                Self::create_session(format_desc, &destination_attributes, &callback_state)?;

            tracing::info!(
                "VTDecoder: Created {:?} session with IOSurface+Metal output",
                codec
            );

            Ok(Self {
                session,
                format_desc,
                callback_state,
                width,
                height,
                codec,
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

            // Check if data is in AVCC format (length-prefixed) or Annex B (start codes)
            // AVCC: first 4 bytes are NAL length (big-endian)
            // Annex B: starts with 0x00 0x00 0x00 0x01 or 0x00 0x00 0x01
            let is_avcc = Self::is_avcc_format(nal_data);
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

            tracing::debug!(
                "VTDecoder: CMSampleBuffer created, calling VTDecompressionSessionDecodeFrame"
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
            if self.callback_state.decode_error.load(Ordering::Acquire) {
                return Err(VideoError::DecodeFailed(
                    "VT decode callback reported error".to_string(),
                ));
            }

            // Pop decoded frame from callback queue
            let queue_len = self.callback_state.decoded_frames.lock().len();
            tracing::debug!("VTDecoder: checking callback queue, length={}", queue_len);
            let decoded = self.callback_state.decoded_frames.lock().pop_front();

            match decoded {
                Some(frame) => {
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
            tracing::info!("VTDecoder: dropped after decoding {} frames", frame_count);

            // Invalidate and release the session
            if !self.session.is_null() {
                unsafe {
                    VTDecompressionSessionInvalidate(self.session);
                    CFRelease(self.session);
                }
            }

            // Release format description
            if !self.format_desc.is_null() {
                unsafe { CFRelease(self.format_desc) };
            }
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
            tracing::debug!("VT decode: frame dropped (info_flags=0x{:x})", info_flags);
            return;
        }
        if info_flags & 0x4 != 0 {
            // kVTDecodeInfo_RequiredFrameDropped - but we still have an image_buffer
            tracing::debug!(
                "VT decode: required frame drop flagged but continuing (info_flags=0x{:x})",
                info_flags
            );
        }

        if image_buffer.is_null() {
            tracing::warn!("VT decode callback: null image buffer");
            return;
        }

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
        pub fn new_with_config(url: &str, config: MoqDecoderConfig) -> Result<Self, VideoError> {
            let moq_url = MoqUrl::parse(url).map_err(|e| VideoError::OpenFailed(e.to_string()))?;

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
        ///
        /// This is identical to MoqDecoder::run_moq_worker - it connects to
        /// the MoQ relay, fetches the catalog, and streams NAL units.
        async fn run_moq_worker(
            shared: Arc<MoqSharedState>,
            url: MoqUrl,
            config: MoqDecoderConfig,
            nal_tx: Sender<MoqVideoFrame>,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            use moq_lite::{Origin, Path};
            use moq_native::ClientConfig;

            shared.set_state(MoqDecoderState::Connecting);
            shared.buffering_percent.store(10, Ordering::Relaxed);

            // QUIC/WebTransport requires TLS. Only use http:// for localhost development.
            let is_localhost =
                url.host() == "localhost" || url.host() == "127.0.0.1" || url.host() == "::1";
            let scheme = if url.use_tls() || !is_localhost {
                "https"
            } else {
                "http"
            };

            // Match desktop URL behavior: namespace/query handling and UUID-style relays.
            let namespace_is_broadcast_id = url.namespace().len() >= 32
                && url
                    .namespace()
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() || c == '-');

            let (connect_url, broadcast_path) = if namespace_is_broadcast_id {
                // zap.stream style: connect to base URL, namespace is the broadcast path
                let base = format!("{}://{}", scheme, url.server_addr());
                let connect = match url.query() {
                    Some(query) => format!("{}?{}", base, query),
                    None => base,
                };
                (connect, Path::from(url.namespace()))
            } else {
                // cdn.moq.dev style: include namespace in connection URL
                let base = format!("{}://{}/{}", scheme, url.server_addr(), url.namespace());
                let connect = match url.query() {
                    Some(query) => format!("{}?{}", base, query),
                    None => base,
                };
                (connect, Path::from(url.track().unwrap_or(url.namespace())))
            };

            tracing::info!("MoQ Android: Connecting to {}", connect_url);

            let mut client_config = ClientConfig::default();
            if config.transport.disable_tls_verify {
                client_config.tls.disable_verify = Some(true);
            }
            client_config.websocket.enabled = config.transport.websocket_fallback;

            let client = client_config
                .init()
                .map_err(|e| format!("Failed to init MoQ client: {e}"))?;

            let origin_producer = Origin::produce();
            let mut origin_consumer = origin_producer.consume();

            let parsed_url: url::Url = connect_url
                .parse()
                .map_err(|e| format!("Invalid URL: {e}"))?;

            let timeout = Duration::from_millis(config.transport.connect_timeout_ms);
            let _session = tokio::time::timeout(
                timeout,
                client.with_consume(origin_producer).connect(parsed_url),
            )
            .await
            .map_err(|_| "Connection timed out")?
            .map_err(|e| format!("Connection failed: {e}"))?;

            tracing::info!("MoQ Android: Connected, waiting for broadcast");

            shared.set_state(MoqDecoderState::FetchingCatalog);
            shared.buffering_percent.store(30, Ordering::Relaxed);

            // Wait for broadcast with timeout to avoid hanging forever
            let discovery_timeout = Duration::from_secs(10);
            let discovery_start = std::time::Instant::now();

            let moq_broadcast: moq_lite::BroadcastConsumer = loop {
                if discovery_start.elapsed() > discovery_timeout {
                    return Err(format!(
                        "Broadcast discovery timeout on '{}' after {:?}",
                        url.namespace(),
                        discovery_timeout
                    )
                    .into());
                }

                if let Some(broadcast) = origin_consumer.consume_broadcast(broadcast_path.clone()) {
                    break broadcast;
                }

                let wait_result =
                    tokio::time::timeout(Duration::from_secs(2), origin_consumer.announced()).await;

                match wait_result {
                    Ok(Some((_path, Some(broadcast)))) => break broadcast,
                    Ok(Some((_path, None))) => continue,
                    Ok(None) => return Err("Origin consumer closed without broadcast".into()),
                    Err(_) => {
                        tracing::debug!("MoQ Android: Still waiting for broadcast...");
                        continue;
                    }
                }
            };

            tracing::info!("MoQ Android: Found broadcast, subscribing to tracks");
            shared.buffering_percent.store(50, Ordering::Relaxed);

            let hang_consumer: hang::BroadcastConsumer = moq_broadcast.into();

            let mut catalog_consumer = hang_consumer.catalog.clone();
            let catalog = match catalog_consumer.next().await {
                Ok(Some(catalog)) => catalog,
                Ok(None) => return Err("Catalog track ended before receiving catalog".into()),
                Err(e) => return Err(format!("Failed to receive catalog: {e}").into()),
            };

            tracing::info!("MoQ Android: Received catalog");

            let (video_track_name, video_config) = catalog
                .video
                .as_ref()
                .and_then(|v| v.renditions.iter().next())
                .ok_or("No video track in catalog")?;

            // Update metadata from catalog
            {
                let mut metadata = shared.metadata.lock();
                metadata.width = video_config.coded_width.unwrap_or(1920);
                metadata.height = video_config.coded_height.unwrap_or(1080);
                metadata.frame_rate = video_config.framerate.unwrap_or(30.0) as f32;
                metadata.codec = format!("{:?}", video_config.codec);
            }

            let video_track = moq_lite::Track {
                name: video_track_name.clone(),
                priority: 1,
            };

            let max_latency = Duration::from_millis(config.max_latency_ms);
            let mut video_consumer = hang_consumer.subscribe(&video_track, max_latency);

            shared.set_state(MoqDecoderState::Streaming);
            shared.buffering_percent.store(100, Ordering::Relaxed);

            tracing::info!(
                "MoQ Android: Streaming started, video track '{}'",
                video_track_name
            );

            // Main NAL unit receive loop
            loop {
                match video_consumer.read_frame().await {
                    Ok(Some(frame)) => {
                        let mut data = bytes::BytesMut::with_capacity(frame.payload.remaining());
                        for chunk in &frame.payload {
                            data.extend_from_slice(chunk);
                        }

                        let moq_frame = MoqVideoFrame {
                            timestamp_us: frame.timestamp.as_micros() as u64,
                            data: data.freeze(),
                            is_keyframe: frame.keyframe,
                        };

                        // Send NAL unit to decoder (blocking — applies QUIC backpressure)
                        if nal_tx.send(moq_frame).await.is_err() {
                            tracing::warn!("MoQ Android: Frame channel closed, stopping worker");
                            break;
                        }
                    }
                    Ok(None) => {
                        tracing::info!("MoQ Android: Video track ended");
                        shared.set_state(MoqDecoderState::Ended);
                        shared.eof_reached.store(true, Ordering::Relaxed);
                        break;
                    }
                    Err(e) => {
                        tracing::error!("MoQ Android: Frame read error: {e}");
                        shared.set_error(format!("Frame read error: {e}"));
                        break;
                    }
                }
            }

            Ok(())
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
            // Android MoQ audio deferred to JNI AudioTrack
            false
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
    pub fn new_with_config(url: &str, config: MoqDecoderConfig) -> Result<Self, VideoError> {
        use gstreamer as gst;
        use gstreamer::prelude::*;
        use gstreamer_app as gst_app;

        // Initialize GStreamer (safe to call multiple times)
        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {e}")))?;

        // Parse the MoQ URL
        let moq_url = MoqUrl::parse(url).map_err(|e| VideoError::OpenFailed(e.to_string()))?;

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
        use moq_lite::{Origin, Path};
        use moq_native::ClientConfig;

        // Update state to connecting
        shared.set_state(MoqDecoderState::Connecting);
        shared.buffering_percent.store(10, Ordering::Relaxed);

        // Build connection URL (moq-native expects https:// or http://)
        // Include namespace as path and query string for JWT authentication
        // QUIC/WebTransport requires TLS. Only use http:// for localhost development.
        let connect_url = {
            let is_localhost =
                url.host() == "localhost" || url.host() == "127.0.0.1" || url.host() == "::1";
            let scheme = if url.use_tls() || !is_localhost {
                "https"
            } else {
                "http"
            };
            let base = format!(
                "{}://{}:{}/{}",
                scheme,
                url.host(),
                url.port(),
                url.namespace()
            );
            match url.query() {
                Some(query) => format!("{}?{}", base, query),
                None => base,
            }
        };

        let redacted_connect_url = connect_url
            .split_once('?')
            .map(|(base, _)| format!("{}?<redacted>", base))
            .unwrap_or_else(|| connect_url.clone());
        tracing::info!("MoQ GStreamer: Connecting to {}", redacted_connect_url);

        // Configure and create client
        let mut client_config = ClientConfig::default();
        if config.transport.disable_tls_verify {
            client_config.tls.disable_verify = Some(true);
        }
        client_config.websocket.enabled = config.transport.websocket_fallback;

        let client = client_config
            .init()
            .map_err(|e| format!("Failed to init MoQ client: {e}"))?;

        // Create origin for receiving broadcasts
        let origin_producer = Origin::produce();
        let mut origin_consumer = origin_producer.consume();

        // Connect with timeout
        let parsed_url: url::Url = connect_url
            .parse()
            .map_err(|e| format!("Invalid URL: {e}"))?;

        let timeout = Duration::from_millis(config.transport.connect_timeout_ms);
        let _session = tokio::time::timeout(
            timeout,
            client.with_consume(origin_producer).connect(parsed_url),
        )
        .await
        .map_err(|_| "Connection timed out")?
        .map_err(|e| format!("Connection failed: {e}"))?;

        tracing::info!("MoQ GStreamer: Connected, waiting for broadcast announcement");

        // Update state to fetching catalog
        shared.set_state(MoqDecoderState::FetchingCatalog);
        shared.buffering_percent.store(30, Ordering::Relaxed);

        // Create broadcast path from track (or namespace if no track)
        let broadcast_path = if let Some(track) = url.track() {
            Path::from(track)
        } else {
            Path::from(url.namespace())
        };
        tracing::info!(
            "MoQ GStreamer: Looking for broadcast at path: {:?}",
            broadcast_path
        );

        // Wait for the broadcast to be announced with timeout
        let discovery_timeout = Duration::from_secs(10);
        let discovery_start = std::time::Instant::now();

        let moq_broadcast: moq_lite::BroadcastConsumer = loop {
            if discovery_start.elapsed() > discovery_timeout {
                return Err(format!(
                    "Broadcast discovery timeout at {:?} after {:?}",
                    broadcast_path, discovery_timeout
                )
                .into());
            }

            if let Some(broadcast) = origin_consumer.consume_broadcast(broadcast_path.clone()) {
                break broadcast;
            }

            let wait_result =
                tokio::time::timeout(Duration::from_secs(2), origin_consumer.announced()).await;

            match wait_result {
                Ok(Some((_path, Some(broadcast)))) => break broadcast,
                Ok(Some((_path, None))) => continue,
                Ok(None) => return Err("Origin consumer closed without broadcast".into()),
                Err(_) => {
                    tracing::debug!("MoQ GStreamer: Still waiting for broadcast...");
                    continue;
                }
            }
        };

        tracing::info!("MoQ GStreamer: Found broadcast, subscribing to tracks");
        shared.buffering_percent.store(50, Ordering::Relaxed);

        // Wrap with hang::BroadcastConsumer
        let hang_consumer: hang::BroadcastConsumer = moq_broadcast.into();

        // Wait for catalog
        let mut catalog_consumer = hang_consumer.catalog.clone();
        let catalog = match catalog_consumer.next().await {
            Ok(Some(catalog)) => catalog,
            Ok(None) => return Err("Catalog track ended before receiving catalog".into()),
            Err(e) => return Err(format!("Failed to receive catalog: {e}").into()),
        };

        tracing::info!("MoQ GStreamer: Received catalog");

        // Find video track
        let (video_track_name, video_config) = catalog
            .video
            .as_ref()
            .and_then(|v| v.renditions.iter().next())
            .ok_or("No video track in catalog")?;

        // Update metadata
        {
            let mut metadata = shared.metadata.lock();
            metadata.width = video_config.coded_width.unwrap_or(1920);
            metadata.height = video_config.coded_height.unwrap_or(1080);
            metadata.frame_rate = video_config.framerate.unwrap_or(30.0) as f32;
            metadata.codec = format!("{:?}", video_config.codec);
        }

        // Subscribe to video track
        let video_track = moq_lite::Track {
            name: video_track_name.clone(),
            priority: 1,
        };
        let max_latency = Duration::from_millis(config.max_latency_ms);
        let mut video_consumer = hang_consumer.subscribe(&video_track, max_latency);

        // Audio track selection and subscription
        use super::moq_audio::*;

        let (mut audio_consumer_opt, audio_sender_opt, mut moq_audio_thread_opt) =
            if config.enable_audio {
                if let Some((track_name, audio_cfg)) = select_preferred_audio_rendition(&catalog) {
                    *shared.audio.audio_status.lock() = MoqAudioStatus::Starting;

                    let audio_track = moq_lite::Track {
                        name: track_name.to_string(),
                        priority: 2,
                    };
                    let audio_consumer = hang_consumer.subscribe(&audio_track, max_latency);

                    let (tx, rx) = crossbeam_channel::bounded(config.audio_buffer_capacity);
                    let live_sender = LiveEdgeSender::new(tx.clone(), rx.clone());

                    let audio_handle = super::audio::AudioHandle::new();
                    *shared.audio.moq_audio_handle.lock() = Some(audio_handle.clone());

                    match MoqAudioThread::spawn(
                        rx,
                        audio_cfg.sample_rate,
                        audio_cfg.channel_count,
                        audio_cfg.description.clone(),
                        audio_handle,
                        shared.audio.clone(),
                    ) {
                        Ok(thread) => {
                            tracing::info!(
                                "MoQ GStreamer: Audio subscribed to track '{}' ({}Hz, {}ch)",
                                track_name,
                                audio_cfg.sample_rate,
                                audio_cfg.channel_count,
                            );
                            (Some(audio_consumer), Some(live_sender), Some(thread))
                        }
                        Err(e) => {
                            tracing::warn!("MoQ GStreamer: Failed to start audio thread: {e}");
                            *shared.audio.audio_status.lock() = MoqAudioStatus::Error;
                            *shared.audio.moq_audio_handle.lock() = None;
                            (None, None, None)
                        }
                    }
                } else {
                    tracing::info!("MoQ GStreamer: No AAC audio track in catalog");
                    *shared.audio.audio_status.lock() = MoqAudioStatus::Unavailable;
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };

        // Update state to streaming
        shared.set_state(MoqDecoderState::Streaming);
        shared.buffering_percent.store(100, Ordering::Relaxed);

        tracing::info!(
            "MoQ GStreamer: Streaming started, subscribed to video track '{}'",
            video_track_name
        );

        // Main frame receive loop with fair video/audio scheduling
        loop {
            tokio::select! {
                video_result = video_consumer.read_frame() => {
                    match video_result {
                        Ok(Some(frame)) => {
                            let mut data = bytes::BytesMut::with_capacity(frame.payload.remaining());
                            for chunk in &frame.payload {
                                data.extend_from_slice(chunk);
                            }

                            let moq_frame = MoqVideoFrame {
                                timestamp_us: frame.timestamp.as_micros() as u64,
                                data: data.freeze(),
                                is_keyframe: frame.keyframe,
                            };

                            if nal_tx.send(moq_frame).await.is_err() {
                                tracing::warn!("MoQ GStreamer: Frame channel closed, stopping worker");
                                break;
                            }
                        }
                        Ok(None) => {
                            tracing::info!("MoQ GStreamer: Video track ended");
                            shared.set_state(MoqDecoderState::Ended);
                            shared.eof_reached.store(true, Ordering::Relaxed);
                            break;
                        }
                        Err(e) => {
                            tracing::error!("MoQ GStreamer: Frame read error: {e}");
                            shared.set_error(format!("Frame read error: {e}"));
                            break;
                        }
                    }
                }
                audio_result = async {
                    if let Some(consumer) = audio_consumer_opt.as_mut() {
                        consumer.read_frame().await
                    } else {
                        std::future::pending().await
                    }
                } => {
                    if let Some(ref audio_sender) = audio_sender_opt {
                        match audio_result {
                            Ok(Some(frame)) => {
                                let mut data = bytes::BytesMut::with_capacity(frame.payload.remaining());
                                for chunk in &frame.payload {
                                    data.extend_from_slice(chunk);
                                }
                                let moq_frame = MoqAudioFrame {
                                    timestamp_us: frame.timestamp.as_micros() as u64,
                                    data: data.freeze(),
                                };
                                if let Err(ChannelClosed) = audio_sender.send(moq_frame) {
                                    tracing::warn!("MoQ GStreamer: Audio channel closed");
                                    audio_consumer_opt = None;
                                }
                            }
                            Ok(None) => {
                                tracing::info!("MoQ GStreamer: Audio track ended");
                                audio_consumer_opt = None;
                            }
                            Err(e) => {
                                tracing::warn!("MoQ GStreamer: Audio read error: {e}");
                                audio_consumer_opt = None;
                            }
                        }
                    }
                }
            }
        }

        // Worker teardown: deterministic audio thread shutdown
        drop(audio_sender_opt);
        {
            let mut status = shared.audio.audio_status.lock();
            if *status == MoqAudioStatus::Starting {
                *status = MoqAudioStatus::Unavailable;
            }
        }
        if let Some(thread) = moq_audio_thread_opt.take() {
            let shared_for_teardown = shared.clone();
            let teardown_start = std::time::Instant::now();
            let teardown_fut = tokio::task::spawn_blocking(move || drop(thread));
            match tokio::time::timeout(Duration::from_secs(2), teardown_fut).await {
                Ok(Ok(())) => tracing::debug!("MoQ GStreamer: audio teardown completed"),
                Ok(Err(e)) => {
                    tracing::warn!("MoQ GStreamer: audio teardown task failed: {e}");
                    *shared_for_teardown.audio.audio_status.lock() = MoqAudioStatus::Error;
                }
                Err(_) => {
                    tracing::warn!("MoQ GStreamer: audio teardown timed out after 2s, proceeding");
                }
            }
            shared_for_teardown
                .audio
                .internal_audio_ready
                .store(false, Ordering::Relaxed);
            *shared_for_teardown.audio.moq_audio_handle.lock() = None;
            {
                let mut status = shared_for_teardown.audio.audio_status.lock();
                if *status == MoqAudioStatus::Running || *status == MoqAudioStatus::Starting {
                    *status = MoqAudioStatus::Unavailable;
                }
            }
            let teardown_ms = teardown_start.elapsed().as_millis();
            if teardown_ms > 250 {
                tracing::warn!("MoQ GStreamer: audio teardown took {}ms", teardown_ms);
            }
        }

        Ok(())
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
