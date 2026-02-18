//! Video player widget for egui.
//!
//! This module provides a video player widget that can be embedded
//! in egui UIs. It handles:
//! - Video decoding via FFmpeg (or ExoPlayer on Android)
//! - Hardware acceleration (VideoToolbox, VAAPI, D3D11VA, MediaCodec)
//! - GPU texture upload via wgpu
//! - YUV to RGB color space conversion
//! - Frame timing and synchronization
//! - Playback controls (play, pause, seek)
//! - Audio volume/mute control
//!
//! # Usage
//!
//! ```ignore
//! use notedeck::media::{VideoPlayer, VideoPlayerExt};
//!
//! // Create a video player with wgpu render state
//! let mut player = VideoPlayer::with_wgpu(
//!     "https://example.com/video.mp4",
//!     &wgpu_render_state,
//! )
//! .with_autoplay(true)
//! .with_loop(true)
//! .with_controls(true);
//!
//! // In your egui update loop:
//! egui::CentralPanel::default().show(ctx, |ui| {
//!     let size = egui::vec2(640.0, 360.0);
//!     let response = player.show(ui, size);
//!
//!     if response.state_changed {
//!         // Handle playback state changes
//!     }
//! });
//! ```
//!
//! # Platform Support
//!
//! - **macOS**: VideoToolbox hardware acceleration
//! - **Windows**: D3D11VA hardware acceleration
//! - **Linux**: VAAPI hardware acceleration
//! - **Android**: ExoPlayer with MediaCodec

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use egui::{Response, Sense, Ui, Vec2};
use egui_wgpu::wgpu;

#[cfg(target_os = "android")]
use super::android_video::AndroidVideoDecoder;
use super::audio::AudioHandle;
#[cfg(target_os = "macos")]
use super::frame_queue::AudioThread;
use super::frame_queue::{DecodeThread, FrameQueue, FrameScheduler};
#[cfg(target_os = "linux")]
use super::linux_video::ZeroCopyGStreamerDecoder;
#[cfg(target_os = "macos")]
use super::macos_video::MacOSVideoDecoder;
use super::subtitles::{SubtitleError, SubtitleStyle, SubtitleTrack};
use super::sync_metrics::{SyncMetrics, SyncMetricsSnapshot};
use super::triple_buffer::{triple_buffer, TripleBufferReader, TripleBufferWriter};
#[cfg(target_os = "android")]
use super::video::AndroidGpuSurface;
#[cfg(target_os = "linux")]
use super::video::LinuxGpuSurface;
#[cfg(target_os = "macos")]
use super::video::MacOSGpuSurface;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
use super::video::WindowsGpuSurface;
use super::video::{
    CpuFrame, PixelFormat, VideoDecoderBackend, VideoError, VideoMetadata, VideoState,
};
#[cfg(target_os = "macos")]
use super::video_decoder::FfmpegDecoder;

#[cfg(feature = "moq")]
use super::moq_decoder::{MoqDecoder, MoqStatsHandle, MoqStatsSnapshot};

/// Returns true if the URL points to a container format supported by AVFoundation on macOS.
///
/// AVFoundation supports: MP4, MOV, M4V, HLS (m3u8), and some AVI files.
/// It does NOT support: MKV, WebM, FLV, OGG, or Matroska.
///
/// When a URL has an unsupported extension, we skip AVFoundation entirely and use FFmpeg.
/// This prevents AVFoundation from accepting the file but failing to produce frames.
#[cfg(target_os = "macos")]
fn is_avfoundation_supported_container(url: &str) -> bool {
    // Extract the path part (strip query params for remote URLs)
    let path = url.split('?').next().unwrap_or(url);

    // Get the extension (lowercase for comparison)
    let ext = path
        .rsplit('.')
        .next()
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    // AVFoundation-supported containers
    matches!(
        ext.as_str(),
        "mp4" | "m4v" | "mov" | "m4a" | "m3u8" | "ts" | "avi" | "aac" | "mp3" | "wav" | "aiff"
    )
}
use super::video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
use super::video_texture::{VideoRenderCallback, VideoRenderResources, VideoTexture};
use poll_promise::Promise;

/// Shared state for pending frame to be rendered.
/// This allows the prepare callback to access frame data for texture creation/upload.
#[derive(Default, Clone)]
pub struct PendingFrame {
    /// The CPU frame data to upload (used when zero-copy is not available)
    pub frame: Option<CpuFrame>,
    /// Cached pixel format for zero-copy frames.
    /// Used by video_texture.rs to detect format mismatch and trigger texture recreation
    /// when the GPU surface format differs from the current texture.
    pub pixel_format: Option<PixelFormat>,
    /// Android GPU surface for zero-copy import (used when AHardwareBuffer is available)
    #[cfg(target_os = "android")]
    pub android_surface: Option<AndroidGpuSurface>,
    /// macOS GPU surface for zero-copy import (used when IOSurface is available)
    #[cfg(target_os = "macos")]
    pub macos_surface: Option<MacOSGpuSurface>,
    /// Windows GPU surface for zero-copy import (used when D3D11 shared handle is available)
    #[cfg(all(target_os = "windows", feature = "windows-native-video"))]
    pub windows_surface: Option<WindowsGpuSurface>,
    /// Linux GPU surface for zero-copy import (used when DMABuf is available)
    #[cfg(target_os = "linux")]
    pub linux_surface: Option<LinuxGpuSurface>,
    /// Whether the texture needs to be recreated (dimensions/format changed)
    pub needs_recreate: bool,
}

/// A video player widget for egui.
///
/// This widget handles video playback, including:
/// - Decoding video frames on a background thread
/// - Uploading frames to GPU textures
/// - Rendering frames via wgpu
/// - Basic playback controls (play, pause, seek)
pub struct VideoPlayer {
    /// Current playback state
    state: VideoState,
    /// Video metadata
    metadata: Option<VideoMetadata>,
    /// The frame queue for decoded frames
    frame_queue: Arc<FrameQueue>,
    /// The decode thread
    decode_thread: Option<DecodeThread>,
    /// Frame scheduler for timing
    scheduler: FrameScheduler,
    /// Current video texture
    texture: Arc<Mutex<Option<VideoTexture>>>,
    /// Whether the player has been initialized
    initialized: bool,
    /// The URL being played
    url: String,
    /// Whether to autoplay
    autoplay: bool,
    /// Whether to loop playback
    loop_playback: bool,
    /// Guard against infinite seek-loop when decoder doesn't support seeking
    loop_seek_pending: bool,
    /// Whether audio is muted
    muted: bool,
    /// wgpu device for texture creation (internally Arc'd by wgpu)
    device: Option<wgpu::Device>,
    /// wgpu queue for texture upload (internally Arc'd by wgpu)
    queue: Option<wgpu::Queue>,
    /// Triple buffer writer for pending frame (lock-free writes from UI thread)
    pending_frame_writer: TripleBufferWriter<PendingFrame>,
    /// Triple buffer reader for pending frame (lock-free reads from render thread)
    pending_frame_reader: TripleBufferReader<PendingFrame>,
    /// Whether to show controls overlay
    show_controls: bool,
    /// Controls configuration
    controls_config: VideoControlsConfig,
    /// Audio handle for volume/mute control
    audio_handle: AudioHandle,
    /// Audio decode/playback thread
    #[cfg(target_os = "macos")]
    audio_thread: Option<AudioThread>,
    /// Background thread for async initialization
    init_thread: Option<std::thread::JoinHandle<()>>,
    /// Promise for async initialization result (poll_promise eliminates Mutex contention)
    init_promise: Option<Promise<Result<Box<dyn VideoDecoderBackend + Send>, VideoError>>>,
    /// Android player ID for multi-player frame isolation (prevents frame mixing)
    #[cfg(target_os = "android")]
    android_player_id: u64,
    /// Rate-limit "no CPU fallback" warning (log once per player instance, persists across frames)
    #[cfg(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        all(target_os = "windows", feature = "windows-native-video")
    ))]
    fallback_logged: Arc<std::sync::atomic::AtomicBool>,
    /// Linux zero-copy metrics for monitoring DMABuf → Vulkan rendering
    /// Wrapped in Mutex so async init thread can populate it
    #[cfg(target_os = "linux")]
    linux_zero_copy_metrics:
        Arc<parking_lot::Mutex<Option<Arc<super::linux_video::ZeroCopyMetrics>>>>,
    /// MoQ decoder stats handle for monitoring MoQ-specific pipeline state
    /// Wrapped in Mutex so init thread can populate it after MoqDecoder is created
    #[cfg(feature = "moq")]
    moq_stats: Arc<parking_lot::Mutex<Option<MoqStatsHandle>>>,
    /// Whether we've already late-bound the MoQ audio handle to self.audio_handle
    #[cfg(feature = "moq")]
    moq_audio_bound: bool,
    /// Loaded subtitle track for rendering
    subtitle_track: Option<SubtitleTrack>,
    /// Whether subtitles are visible
    show_subtitles: bool,
    /// Subtitle rendering style configuration
    subtitle_style: SubtitleStyle,
}

impl VideoPlayer {
    /// Creates a new video player for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        let (pending_frame_writer, pending_frame_reader) = triple_buffer();
        let audio_handle = AudioHandle::new();
        let scheduler = FrameScheduler::with_audio_handle(audio_handle.clone());
        Self {
            state: VideoState::Loading,
            metadata: None,
            frame_queue: Arc::new(FrameQueue::with_default_capacity()),
            decode_thread: None,
            scheduler,
            texture: Arc::new(Mutex::new(None)),
            initialized: false,
            url: url.into(),
            autoplay: false,
            loop_playback: false,
            loop_seek_pending: false,
            muted: false,
            device: None,
            queue: None,
            pending_frame_writer,
            pending_frame_reader,
            show_controls: true,
            controls_config: VideoControlsConfig::default(),
            audio_handle,
            #[cfg(target_os = "macos")]
            audio_thread: None,
            init_thread: None,
            init_promise: None,
            // Updated from decoder.android_player_id() once init completes.
            #[cfg(target_os = "android")]
            android_player_id: 0,
            #[cfg(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "android",
                all(target_os = "windows", feature = "windows-native-video")
            ))]
            fallback_logged: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(target_os = "linux")]
            linux_zero_copy_metrics: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "moq")]
            moq_stats: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "moq")]
            moq_audio_bound: false,
            subtitle_track: None,
            show_subtitles: true,
            subtitle_style: SubtitleStyle::default(),
        }
    }

    /// Creates a new video player with wgpu render state.
    ///
    /// This is the preferred way to create a video player as it allows
    /// immediate texture creation and upload.
    ///
    /// # Arguments
    ///
    /// * `url` - The video URL to play
    /// * `wgpu_render_state` - The egui wgpu render state
    ///
    /// # Note
    ///
    /// If your app uses a depth buffer, use [`Self::with_wgpu_and_depth`] instead
    /// to avoid wgpu validation errors.
    pub fn with_wgpu(url: impl Into<String>, wgpu_render_state: &egui_wgpu::RenderState) -> Self {
        Self::with_wgpu_and_depth(url, wgpu_render_state, None)
    }

    /// Creates a new video player with wgpu render state and depth format.
    ///
    /// Use this constructor when your app's render pass uses a depth buffer.
    /// Pass the same depth format used by your render pass to avoid wgpu
    /// validation errors.
    ///
    /// # Arguments
    ///
    /// * `url` - The video URL to play
    /// * `wgpu_render_state` - The egui wgpu render state
    /// * `depth_format` - Optional depth format to match the render pass. Use `None` if
    ///   the render pass has no depth attachment, or `Some(format)` to match the host app's
    ///   depth buffer format (e.g., `Depth24Plus`, `Depth32Float`).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For apps using egui's default depth buffer (Depth24Plus)
    /// let player = VideoPlayer::with_wgpu_and_depth(
    ///     "https://example.com/video.m3u8",
    ///     &wgpu_render_state,
    ///     Some(wgpu::TextureFormat::Depth24Plus),
    /// );
    /// ```
    pub fn with_wgpu_and_depth(
        url: impl Into<String>,
        wgpu_render_state: &egui_wgpu::RenderState,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        // Register video render resources if not already done
        {
            let renderer = wgpu_render_state.renderer.read();
            if renderer
                .callback_resources
                .get::<VideoRenderResources>()
                .is_none()
            {
                drop(renderer);
                VideoRenderResources::register(wgpu_render_state, depth_format);
            }
        }

        let (pending_frame_writer, pending_frame_reader) = triple_buffer();
        let audio_handle = AudioHandle::new();
        let scheduler = FrameScheduler::with_audio_handle(audio_handle.clone());
        Self {
            state: VideoState::Loading,
            metadata: None,
            frame_queue: Arc::new(FrameQueue::with_default_capacity()),
            decode_thread: None,
            scheduler,
            texture: Arc::new(Mutex::new(None)),
            initialized: false,
            url: url.into(),
            autoplay: false,
            loop_playback: false,
            loop_seek_pending: false,
            muted: false,
            device: Some(wgpu_render_state.device.clone()),
            queue: Some(wgpu_render_state.queue.clone()),
            pending_frame_writer,
            pending_frame_reader,
            show_controls: true,
            controls_config: VideoControlsConfig::default(),
            audio_handle,
            #[cfg(target_os = "macos")]
            audio_thread: None,
            init_thread: None,
            init_promise: None,
            // Updated from decoder.android_player_id() once init completes.
            #[cfg(target_os = "android")]
            android_player_id: 0,
            #[cfg(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "android",
                all(target_os = "windows", feature = "windows-native-video")
            ))]
            fallback_logged: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(target_os = "linux")]
            linux_zero_copy_metrics: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "moq")]
            moq_stats: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "moq")]
            moq_audio_bound: false,
            subtitle_track: None,
            show_subtitles: true,
            subtitle_style: SubtitleStyle::default(),
        }
    }

    /// Sets whether the video should autoplay.
    pub fn with_autoplay(mut self, autoplay: bool) -> Self {
        self.autoplay = autoplay;
        self
    }

    /// Sets whether the video should loop.
    pub fn with_loop(mut self, loop_playback: bool) -> Self {
        self.loop_playback = loop_playback;
        self
    }

    /// Sets whether audio is muted.
    pub fn with_muted(mut self, muted: bool) -> Self {
        self.muted = muted;
        self.audio_handle.set_muted(muted);
        self
    }

    /// Sets whether to show controls overlay.
    pub fn with_controls(mut self, show_controls: bool) -> Self {
        self.show_controls = show_controls;
        self
    }

    /// Sets whether subtitles should be visible by default.
    pub fn with_subtitles(mut self, show_subtitles: bool) -> Self {
        self.show_subtitles = show_subtitles;
        self
    }

    /// Sets the subtitle rendering style.
    pub fn with_subtitle_style(mut self, style: SubtitleStyle) -> Self {
        self.subtitle_style = style;
        self
    }

    /// Sets the controls configuration.
    pub fn with_controls_config(mut self, config: VideoControlsConfig) -> Self {
        self.controls_config = config;
        self
    }

    /// Starts async initialization of the video player.
    ///
    /// This spawns a background thread to open the video and prepare for playback.
    /// The player will show a loading state until initialization completes.
    ///
    /// Note: On macOS, AVPlayer requires main thread initialization, so the decoder
    /// is created synchronously on the main thread before spawning the background thread.
    pub fn start_async_init(&mut self) {
        if self.initialized || self.init_thread.is_some() {
            return;
        }

        let url = self.url.clone();
        let (sender, promise) = Promise::new();
        self.init_promise = Some(promise);

        // macOS: Initialize decoder on main thread BEFORE spawning background thread
        // AVPlayer/AVFoundation requires main thread for initialization
        // Skip for unsupported containers (MKV, WebM) - use FFmpeg directly
        #[cfg(target_os = "macos")]
        let macos_decoder_result: Option<Result<MacOSVideoDecoder, VideoError>> = {
            if is_avfoundation_supported_container(&url) {
                tracing::info!("Initializing macOS VideoToolbox decoder on main thread");
                Some(MacOSVideoDecoder::new(&url))
            } else {
                tracing::info!(
                    "Skipping macOS decoder for unsupported container, using FFmpeg: {}",
                    url
                );
                None // Signal to use FFmpeg directly
            }
        };

        // Capture Linux metrics holder for async init (so thread can populate it)
        #[cfg(target_os = "linux")]
        let linux_metrics_holder = Arc::clone(&self.linux_zero_copy_metrics);

        // Capture MoQ stats holder for async init
        #[cfg(feature = "moq")]
        let moq_stats_holder = Arc::clone(&self.moq_stats);

        // Spawn background thread for initialization
        let handle = std::thread::spawn(move || {
            // Open the video with platform-specific decoder

            // MoQ URLs - use MoQ decoder for live streaming
            #[cfg(feature = "moq")]
            if MoqDecoder::is_moq_url(&url) {
                tracing::info!("Using MoQ decoder for live stream: {}", url);
                let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> =
                    match MoqDecoder::new(&url) {
                        Ok(decoder) => {
                            // Store stats handle for UI access before boxing
                            *moq_stats_holder.lock() = Some(decoder.stats_handle());
                            Ok(Box::new(decoder) as Box<dyn VideoDecoderBackend + Send>)
                        }
                        Err(e) => Err(e),
                    };
                sender.send(result);
                return;
            }

            // macOS with FFmpeg fallback - use pre-initialized decoder or FFmpeg for unsupported containers
            #[cfg(target_os = "macos")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                match macos_decoder_result {
                    Some(Ok(d)) => {
                        tracing::info!("Using macOS VideoToolbox hardware decoder");
                        Ok(Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    Some(Err(e)) => {
                        tracing::warn!(
                            "macOS VideoToolbox decoder failed, falling back to FFmpeg: {:?}",
                            e
                        );
                        FfmpegDecoder::new(&url)
                            .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    None => {
                        // Unsupported container (MKV, WebM, etc.) - use FFmpeg directly
                        tracing::info!("Using FFmpeg for unsupported container format");
                        FfmpegDecoder::new(&url)
                            .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                }
            };

            #[cfg(target_os = "android")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                tracing::info!("Using Android ExoPlayer decoder for {}", url);
                AndroidVideoDecoder::new(&url)
                    .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
            };

            // Linux: GStreamer is the only supported backend (no FFmpeg fallback)
            // GStreamer handles all containers (MKV, WebM, MP4, etc.) and has built-in
            // fallback from zero-copy DMABuf to CPU copy when VA-API is unavailable
            #[cfg(target_os = "linux")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                match ZeroCopyGStreamerDecoder::new(&url) {
                    Ok(decoder) => {
                        // Store metrics for later access
                        *linux_metrics_holder.lock() = Some(Arc::clone(decoder.metrics()));
                        Ok(Box::new(decoder) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    Err(e) => Err(e),
                }
            };

            // Fallback when no decoder is available at compile time
            // (platforms without native decoder: Windows without feature, WASM, etc.)
            #[cfg(not(any(target_os = "android", target_os = "macos", target_os = "linux",)))]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                let _ = &url; // Silence unused variable warning
                Err(VideoError::DecoderInit(
                    "No video decoder available for this platform".to_string(),
                ))
            };

            sender.send(result);
        });

        self.init_thread = Some(handle);
    }

    /// Checks if async initialization is complete and finishes setup.
    /// Returns true if initialization is complete (success or error).
    fn check_init_complete(&mut self) -> bool {
        if self.initialized {
            return true;
        }

        // Check if promise exists and is ready (non-blocking poll)
        let is_ready = self
            .init_promise
            .as_ref()
            .is_some_and(|p| p.ready().is_some());

        if !is_ready {
            return false;
        }

        // Take the promise and extract the result
        let Some(promise) = self.init_promise.take() else {
            return false;
        };

        // try_take returns Ok(value) if ready, Err(promise) if not ready
        let Ok(result) = promise.try_take() else {
            // This shouldn't happen since we checked ready()
            self.state = VideoState::Error(VideoError::Generic("Init thread crashed".into()));
            self.init_thread = None;
            return true;
        };

        match result {
            Ok(decoder) => {
                // Store metadata
                let metadata = decoder.metadata().clone();
                self.metadata = Some(metadata.clone());

                // Check at RUNTIME if decoder handles audio internally
                // This allows FFmpeg fallback on macOS/Linux to still get FFmpeg audio
                let uses_native_audio = decoder.handles_audio_internally();

                // Extract Android player ID for per-player frame queue routing
                #[cfg(target_os = "android")]
                {
                    self.android_player_id = decoder.android_player_id();
                }

                // Extract MoQ audio handle (if any) before boxing the decoder
                let decoder_audio_handle = decoder.audio_handle();

                // For native decoders, set up audio_handle BEFORE creating decode thread
                // so the decode_loop can update native_position from the decoder's current_time()
                if uses_native_audio {
                    if let Some(ref ah) = decoder_audio_handle {
                        self.audio_handle = ah.clone();
                    }
                    self.audio_handle.set_available(true);
                    self.scheduler.set_audio_handle(self.audio_handle.clone());
                    tracing::info!("Native audio enabled (decoder handles audio internally)");
                }

                // Create and start the decode thread
                let frame_queue = Arc::clone(&self.frame_queue);

                // For native video decoders, pass audio_handle so decode_loop can
                // update native_position from the decoder's current_time()
                #[cfg(target_os = "macos")]
                let decode_thread = if uses_native_audio {
                    DecodeThread::with_audio_handle(
                        decoder,
                        frame_queue,
                        Some(self.audio_handle.clone()),
                    )
                } else {
                    DecodeThread::new(decoder, frame_queue)
                };

                #[cfg(not(target_os = "macos"))]
                let decode_thread = DecodeThread::new(decoder, frame_queue);

                self.decode_thread = Some(decode_thread);

                // Start FFmpeg audio thread only if decoder doesn't handle audio internally
                // AND this is not a MoQ URL (MoQ handles audio via its own pipeline).
                #[cfg(target_os = "macos")]
                {
                    #[cfg(feature = "moq")]
                    let is_moq = super::moq_decoder::MoqDecoder::is_moq_url(&self.url);
                    #[cfg(not(feature = "moq"))]
                    let is_moq = false;
                    if !uses_native_audio && !is_moq {
                        if let Some(audio_thread) = AudioThread::new(&self.url, metadata.start_time)
                        {
                            self.audio_handle = audio_thread.handle();
                            // Update FrameScheduler's audio handle so sync metrics work
                            self.scheduler.set_audio_handle(self.audio_handle.clone());
                            tracing::info!("FFmpeg audio playback initialized for {}", self.url);
                            self.audio_thread = Some(audio_thread);
                        }
                    }
                }

                self.state = VideoState::Ready;
                self.initialized = true;
                self.init_thread = None;

                // Start playback if autoplay is enabled
                if self.autoplay {
                    self.play();
                }

                true
            }
            Err(e) => {
                self.state = VideoState::Error(e);
                self.init_thread = None;
                // Mark as initialized even on error to prevent infinite retry loop
                // (show() would otherwise restart init every frame since init_thread is None)
                self.initialized = true;
                true
            }
        }
    }

    /// Initializes the video player synchronously (legacy, causes UI freeze).
    #[allow(dead_code)]
    pub fn initialize(&mut self) -> Result<(), VideoError> {
        if self.initialized {
            return Ok(());
        }

        // Open the video with platform-specific decoder
        // For unsupported containers (MKV, WebM), skip macOS decoder and use FFmpeg directly
        #[cfg(target_os = "macos")]
        let decoder: Box<dyn VideoDecoderBackend + Send> = {
            if !is_avfoundation_supported_container(&self.url) {
                tracing::info!(
                    "Using FFmpeg for unsupported container format: {}",
                    self.url
                );
                Box::new(FfmpegDecoder::new(&self.url)?)
            } else {
                match MacOSVideoDecoder::new(&self.url) {
                    Ok(d) => {
                        tracing::info!("Using macOS VideoToolbox hardware decoder");
                        Box::new(d)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "macOS VideoToolbox decoder failed, falling back to FFmpeg: {:?}",
                            e
                        );
                        Box::new(FfmpegDecoder::new(&self.url)?)
                    }
                }
            }
        };

        #[cfg(target_os = "android")]
        let decoder: Box<dyn VideoDecoderBackend + Send> = {
            tracing::info!("Using Android ExoPlayer decoder for {}", self.url);
            let d = AndroidVideoDecoder::new(&self.url)?;
            self.android_player_id = d.android_player_id();
            Box::new(d)
        };

        // Linux: GStreamer is the only supported backend (no FFmpeg fallback)
        // GStreamer handles all containers (MKV, WebM, MP4, etc.) and has built-in
        // fallback from zero-copy DMABuf to CPU copy when VA-API is unavailable
        #[cfg(target_os = "linux")]
        let decoder: Box<dyn VideoDecoderBackend + Send> = {
            let gst_decoder = ZeroCopyGStreamerDecoder::new(&self.url)?;
            // Store metrics for later access
            *self.linux_zero_copy_metrics.lock() = Some(Arc::clone(gst_decoder.metrics()));
            Box::new(gst_decoder)
        };

        // Fallback when no decoder is available at compile time
        #[cfg(not(any(target_os = "android", target_os = "macos", target_os = "linux",)))]
        {
            return Err(VideoError::DecoderInit(
                "No video decoder available for this platform".to_string(),
            ));
        }

        // Code that requires a decoder - only compiled when one is available
        #[cfg(any(target_os = "android", target_os = "macos", target_os = "linux",))]
        {
            self.metadata = Some(decoder.metadata().clone());

            let frame_queue = Arc::clone(&self.frame_queue);
            let decode_thread = DecodeThread::new(decoder, frame_queue);

            self.decode_thread = Some(decode_thread);
            self.state = VideoState::Ready;
            self.initialized = true;

            if self.autoplay {
                self.play();
            }

            Ok(())
        }
    }

    /// Starts or resumes playback.
    pub fn play(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            // For native decoders (macOS, Android, GStreamer), sync mute state before playing
            // This ensures audio plays correctly when user hasn't explicitly muted
            #[cfg(any(target_os = "android", target_os = "macos", target_os = "linux"))]
            thread.set_muted(self.muted);

            thread.play();
            self.scheduler.start();
            self.state = VideoState::Playing {
                position: self.scheduler.position(),
            };

            // Start audio playback (FFmpeg audio thread)
            #[cfg(target_os = "macos")]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.play();
            }
        }
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            thread.pause();
            self.scheduler.pause();
            self.state = VideoState::Paused {
                position: self.scheduler.position(),
            };

            // Pause audio playback
            #[cfg(target_os = "macos")]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.pause();
            }
        }
    }

    /// Toggles between play and pause.
    pub fn toggle_playback(&mut self) {
        match self.state {
            VideoState::Playing { .. } => self.pause(),
            VideoState::Paused { .. } | VideoState::Ready | VideoState::Ended => self.play(),
            _ => {}
        }
    }

    /// Seeks to a specific position.
    pub fn seek(&mut self, position: Duration) {
        // Log seek origin for debugging unexpected seeks
        if position == Duration::ZERO {
            tracing::debug!(
                "Seek to ZERO requested from state={:?}, loop_playback={}",
                self.state,
                self.loop_playback
            );
        }

        // Note: EOS is cleared by the decode thread after a successful seek
        // (see frame_queue.rs process_decode_command). We don't clear it eagerly
        // here because if seek fails (e.g. live MoQ streams), EOS should remain
        // set to prevent an infinite seek-EOS loop.

        if let Some(ref thread) = self.decode_thread {
            thread.seek(position);
            self.scheduler.seek(position);

            // Seek audio
            #[cfg(target_os = "macos")]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.seek(position);
            }

            // Update state with new position
            match self.state {
                VideoState::Playing { .. } => {
                    self.state = VideoState::Playing { position };
                }
                VideoState::Paused { .. } => {
                    self.state = VideoState::Paused { position };
                }
                VideoState::Ended => {
                    self.state = VideoState::Paused { position };
                }
                _ => {}
            }
        }
    }

    /// Returns the current playback state.
    pub fn state(&self) -> &VideoState {
        &self.state
    }

    /// Returns the video metadata if available.
    pub fn metadata(&self) -> Option<&VideoMetadata> {
        self.metadata.as_ref()
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        self.scheduler.position()
    }

    /// Returns the video duration if known.
    pub fn duration(&self) -> Option<Duration> {
        // First try to get duration from the decode thread (updated dynamically, e.g., from ExoPlayer callbacks)
        if let Some(ref thread) = self.decode_thread {
            if let Some(dur) = thread.duration() {
                return Some(dur);
            }
        }
        // Fall back to metadata duration
        self.metadata.as_ref().and_then(|m| m.duration)
    }

    /// Returns the A/V sync metrics tracker.
    ///
    /// Use this to monitor audio-video synchronization quality during playback.
    /// The metrics track drift between audio and video presentation timestamps.
    pub fn sync_metrics(&self) -> &SyncMetrics {
        self.scheduler.sync_metrics()
    }

    /// Returns a snapshot of current A/V sync metrics.
    ///
    /// This provides a point-in-time view of sync quality including:
    /// - Current drift (positive = video ahead of audio)
    /// - Maximum drift observed
    /// - Percentage of frames out of sync
    pub fn sync_metrics_snapshot(&self) -> SyncMetricsSnapshot {
        self.scheduler.sync_metrics().snapshot()
    }

    /// Returns a snapshot of Linux zero-copy metrics (Linux GStreamer only).
    ///
    /// This provides visibility into whether DMABuf → Vulkan zero-copy rendering
    /// is working, or if the system has fallen back to CPU copy.
    ///
    /// Returns `None` if:
    /// - Not on Linux
    /// - GStreamer decoder not used
    /// - Decoder not yet initialized
    ///
    /// # Example
    /// ```ignore
    /// # use lumina_video::VideoPlayer;
    /// let player = VideoPlayer::new("video.mp4");
    /// if let Some(metrics) = player.linux_zero_copy_metrics() {
    ///     if metrics.is_zero_copy_active() {
    ///         println!("✓ Zero-copy active: {} frames", metrics.zero_copy_frames);
    ///     } else {
    ///         println!("⚠ Fallback: {:.1}% CPU copy", metrics.fallback_percentage());
    ///     }
    /// }
    /// ```
    #[cfg(target_os = "linux")]
    pub fn linux_zero_copy_metrics(
        &self,
    ) -> Option<super::linux_video::LinuxZeroCopyMetricsSnapshot> {
        self.linux_zero_copy_metrics
            .lock()
            .as_ref()
            .map(|metrics| metrics.snapshot())
    }

    /// Returns a snapshot of MoQ decoder stats (MoQ streams only).
    ///
    /// Returns `None` if the current stream is not a MoQ stream or if the
    /// MoQ decoder hasn't been initialized yet.
    #[cfg(feature = "moq")]
    pub fn moq_stats(&self) -> Option<MoqStatsSnapshot> {
        self.moq_stats
            .lock()
            .as_ref()
            .map(|handle| handle.snapshot())
    }

    /// Returns the video dimensions (width, height).
    ///
    /// This checks the decode thread first for dynamically updated dimensions
    /// (e.g., from ExoPlayer callbacks on Android), then falls back to metadata.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        // First try to get dimensions from the decode thread
        if let Some(ref thread) = self.decode_thread {
            if let Some(dims) = thread.dimensions() {
                return Some(dims);
            }
        }
        // Fall back to metadata dimensions
        self.metadata.as_ref().map(|m| (m.width, m.height))
    }

    /// Returns the video frame rate.
    ///
    /// This checks the decode thread first for dynamically updated frame rate
    /// (e.g., from macOS AVPlayer when metadata becomes ready), then falls back to metadata.
    pub fn frame_rate(&self) -> Option<f32> {
        // First try to get frame rate from the decode thread
        if let Some(ref thread) = self.decode_thread {
            if let Some(fps) = thread.frame_rate() {
                return Some(fps);
            }
        }
        // Fall back to metadata frame rate
        self.metadata.as_ref().map(|m| m.frame_rate)
    }

    /// Syncs the stored metadata with values from the decode thread.
    ///
    /// This should be called periodically to update the metadata with
    /// dynamically discovered values (e.g., macOS AVPlayer lazy metadata).
    fn sync_metadata_from_decode_thread(&mut self) {
        let Some(ref thread) = self.decode_thread else {
            return;
        };

        let Some(ref mut metadata) = self.metadata else {
            return;
        };

        // Update duration
        if let Some(dur) = thread.duration() {
            metadata.duration = Some(dur);
        }

        // Update dimensions
        if let Some((w, h)) = thread.dimensions() {
            if w > 1 && h > 1 {
                metadata.width = w;
                metadata.height = h;
            }
        }

        // Update frame rate and manage pacing for live streams
        if let Some(fps) = thread.frame_rate() {
            if fps > 0.0 {
                metadata.frame_rate = fps;
                if metadata.duration.is_none() {
                    // Live/MoQ: enable pacing to smooth group-boundary bursts
                    self.scheduler.set_frame_rate_pacing(fps);
                } else {
                    // VOD or stream that acquired duration: disable pacing
                    self.scheduler.clear_frame_rate_pacing();
                }
            }
        }
    }

    /// Returns true if the video is currently playing.
    pub fn is_playing(&self) -> bool {
        self.scheduler.is_playing()
    }

    /// Returns the current buffering percentage (0-100).
    ///
    /// For network streams, this indicates how much data has been buffered.
    /// Returns 100 for local files or when buffering state is unknown.
    pub fn buffering_percent(&self) -> i32 {
        if let Some(ref thread) = self.decode_thread {
            thread.buffering_percent()
        } else {
            100 // Assume buffered if no decode thread yet
        }
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &AudioHandle {
        &self.audio_handle
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.audio_handle.volume()
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&mut self, volume: u32) {
        self.audio_handle.set_volume(volume);

        // On Android, audio is controlled by ExoPlayer through the decode thread
        #[cfg(target_os = "android")]
        if let Some(ref decode_thread) = self.decode_thread {
            // Convert 0-100 to 0.0-1.0
            decode_thread.set_volume(volume as f32 / 100.0);
        }
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.audio_handle.is_muted()
    }

    /// Toggles the mute state.
    pub fn toggle_mute(&mut self) {
        self.audio_handle.toggle_mute();
        self.muted = self.audio_handle.is_muted();

        // For native decoders, audio is controlled through the decode thread
        #[cfg(any(target_os = "android", target_os = "macos", target_os = "linux"))]
        if let Some(ref decode_thread) = self.decode_thread {
            decode_thread.set_muted(self.muted);
        }
    }

    /// Loads subtitles from SRT format content.
    ///
    /// # Arguments
    ///
    /// * `content` - The raw SRT file content as a string
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if subtitles were loaded successfully, or an error if parsing failed.
    /// Polls for MoQ audio handle availability and late-binds it.
    ///
    /// The MoQ audio thread creates its AudioHandle asynchronously — it's not
    /// available at `check_init_complete()` time. This method is called each
    /// frame from the `show()` loop. When the handle becomes available:
    ///
    /// 1. Migrates current mute/volume state to the new handle
    /// 2. Replaces `self.audio_handle` so UI controls affect MoQ playback
    /// 3. Binds to FrameScheduler for audio-as-master-clock A/V sync
    ///
    /// Binds audio in metrics-only mode (wall-clock still drives frame pacing)
    /// to avoid a position discontinuity from late-arriving audio.
    ///
    /// When the handle becomes stale (thread torn down), unbinds and reverts.
    #[cfg(feature = "moq")]
    fn poll_moq_audio_handle(&mut self) {
        let moq_stats = self.moq_stats.lock();
        let Some(ref stats) = *moq_stats else {
            return;
        };

        if !self.moq_audio_bound {
            // Try to acquire the audio handle
            if let Some(moq_ah) = stats.audio_handle() {
                // Migrate current mute/volume state
                moq_ah.set_muted(self.audio_handle.is_muted());
                moq_ah.set_volume(self.audio_handle.volume());

                // Replace the player-level handle (for UI mute/volume controls only)
                self.audio_handle = moq_ah;
                self.audio_handle.set_available(true);

                // Bind to FrameScheduler for sync metrics (drift tracking).
                // Use metrics-only mode: wall-clock drives frame pacing, not audio.
                // MoQ audio arrives late; switching to audio-as-master would cause
                // a position discontinuity that freezes or jumps video.
                self.scheduler
                    .set_audio_handle_metrics_only(self.audio_handle.clone());

                // Enable playback epoch so AudioHandle::position() returns non-zero,
                // allowing sync metrics to measure drift.
                if self.scheduler.is_playing() {
                    self.audio_handle.enable_playback_epoch();
                    tracing::info!(
                        "MoQ audio: late bind (metrics only, wall-clock pacing)"
                    );
                }

                self.moq_audio_bound = true;
                tracing::info!("MoQ audio handle bound to VideoPlayer + FrameScheduler");
            }
        } else {
            // Check if handle went stale (thread torn down)
            if !stats.is_audio_alive() {
                tracing::info!("MoQ audio handle stale (thread torn down), unbinding");

                // Create a fresh placeholder handle and migrate state
                let placeholder = AudioHandle::new();
                placeholder.set_muted(self.audio_handle.is_muted());
                placeholder.set_volume(self.audio_handle.volume());
                self.audio_handle = placeholder;

                // Clear FrameScheduler's audio handle so it falls back to wall-clock
                self.scheduler.clear_audio_handle();

                self.moq_audio_bound = false;
            }
        }
    }

    pub fn load_subtitles_srt(&mut self, content: &str) -> Result<(), SubtitleError> {
        let track = SubtitleTrack::from_srt(content)?;
        self.subtitle_track = Some(track);
        Ok(())
    }

    /// Loads subtitles from WebVTT format content.
    ///
    /// # Arguments
    ///
    /// * `content` - The raw VTT file content as a string
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if subtitles were loaded successfully, or an error if parsing failed.
    pub fn load_subtitles_vtt(&mut self, content: &str) -> Result<(), SubtitleError> {
        let track = SubtitleTrack::from_vtt(content)?;
        self.subtitle_track = Some(track);
        Ok(())
    }

    /// Clears the currently loaded subtitle track.
    pub fn clear_subtitles(&mut self) {
        self.subtitle_track = None;
    }

    /// Returns whether subtitles are currently visible.
    pub fn subtitles_visible(&self) -> bool {
        self.show_subtitles
    }

    /// Sets whether subtitles should be visible.
    pub fn set_subtitles_visible(&mut self, visible: bool) {
        self.show_subtitles = visible;
    }

    /// Toggles subtitle visibility.
    pub fn toggle_subtitles(&mut self) {
        self.show_subtitles = !self.show_subtitles;
    }

    /// Returns whether a subtitle track is currently loaded.
    pub fn has_subtitles(&self) -> bool {
        self.subtitle_track.is_some()
    }

    /// Returns the current subtitle text at the current playback position, if any.
    pub fn current_subtitle(&self) -> Option<&str> {
        let track = self.subtitle_track.as_ref()?;
        let position = self.position();
        track.get_cue_at(position).map(|cue| cue.text.as_str())
    }

    /// Sets the subtitle rendering style.
    pub fn set_subtitle_style(&mut self, style: SubtitleStyle) {
        self.subtitle_style = style;
    }

    /// Shows the video player widget.
    ///
    /// This renders the current video frame and handles user interactions.
    pub fn show(&mut self, ui: &mut Ui, size: Vec2) -> VideoPlayerResponse {
        // Allocate space for the video
        let (rect, response) = ui.allocate_exact_size(size, Sense::click());

        // Start async initialization if needed
        if !self.initialized && self.init_thread.is_none() {
            self.start_async_init();
        }

        // Check if async init is complete
        if !self.initialized {
            self.check_init_complete();
        }

        // Sync metadata from decode thread (for lazy metadata like macOS AVPlayer)
        self.sync_metadata_from_decode_thread();

        // Late-bind MoQ audio handle (async — not available at init time)
        #[cfg(feature = "moq")]
        if self.initialized {
            self.poll_moq_audio_handle();
        }

        // Update frame if playback requested (even if buffering), or try to get preview frame when Ready/Paused
        if self.scheduler.is_playback_requested() {
            self.update_frame();
        } else if matches!(self.state, VideoState::Ready | VideoState::Paused { .. }) {
            // Try to get preview frame from queue (non-blocking)
            self.try_get_preview_frame();
        }

        // Check for end of stream
        if self.frame_queue.is_eos() && self.frame_queue.is_empty() {
            if self.loop_playback && !self.loop_seek_pending {
                tracing::debug!("Loop triggered: seeking to start");
                self.loop_seek_pending = true;
                self.seek(Duration::ZERO);
                self.play();
            } else if !self.loop_playback || self.loop_seek_pending {
                // Either looping is disabled, or we already tried to loop and
                // EOS reappeared (meaning seek failed). Give up.
                if self.loop_seek_pending {
                    tracing::debug!("Loop seek failed (EOS reappeared), ending playback");
                    self.loop_seek_pending = false;
                }
                self.state = VideoState::Ended;
            }
        }

        // Render the video frame, loading state, or error
        match &self.state {
            VideoState::Error(err) => {
                self.render_error(ui, rect, err);
            }
            VideoState::Loading => {
                // Show loading indicator while initializing
                self.render_loading(ui, rect);
            }
            _ => {
                self.render(ui, rect);

                // Show buffering overlay only when playing AND buffering < 100%
                // Don't show when paused/ready - let user see the preview frame
                let buffering = self.buffering_percent();
                if buffering < 100 && self.is_playing() {
                    self.render_buffering_overlay(ui, rect, buffering);
                }

                // Render subtitles overlay if enabled and available
                if self.show_subtitles {
                    if let Some(ref track) = self.subtitle_track {
                        if let Some(cue) = track.get_cue_at(self.position()) {
                            self.render_subtitle(ui, rect, &cue.text);
                        }
                    }
                }
            }
        }

        // Show controls overlay and handle interactions
        let mut state_changed = false;
        let mut controls_response = VideoControlsResponse::default();

        if self.show_controls {
            let controls = VideoControls::new(&self.state, self.position(), self.duration())
                .with_config(self.controls_config.clone())
                .with_muted(self.audio_handle.is_muted())
                .with_subtitles(self.show_subtitles, self.subtitle_track.is_some())
                .with_buffering_percent(self.buffering_percent());
            controls_response = controls.show(ui, rect);

            // Handle control interactions
            if controls_response.toggle_playback {
                self.toggle_playback();
                state_changed = true;
            }

            if let Some(seek_pos) = controls_response.seek_to {
                tracing::debug!(
                    "VideoPlayer: UI seek to {:?} (current position: {:?})",
                    seek_pos,
                    self.position()
                );
                self.seek(seek_pos);
                state_changed = true;
            }

            if controls_response.toggle_mute {
                self.toggle_mute();
                state_changed = true;
            }

            if controls_response.toggle_subtitles {
                self.toggle_subtitles();
            }
        }

        // Handle click on video area to toggle playback (only if not handled by controls)
        // Ignore click if any control was interacted with
        let control_was_used = controls_response.toggle_playback
            || controls_response.toggle_mute
            || controls_response.toggle_fullscreen
            || controls_response.toggle_subtitles
            || controls_response.seek_to.is_some()
            || controls_response.is_seeking;
        let clicked = response.clicked() && !control_was_used;
        if clicked {
            self.toggle_playback();
            state_changed = true;
        }

        // Request repaint if playing/buffering, loading, initializing, or have pending frame
        let is_initializing = self.init_thread.is_some();
        let has_pending_frame = self.pending_frame_reader.has_new_frame();
        let is_buffering = self.buffering_percent() < 100;
        if self.scheduler.is_playback_requested()
            || is_initializing
            || has_pending_frame
            || is_buffering
            || matches!(
                self.state,
                VideoState::Loading | VideoState::Buffering { .. }
            )
        {
            ui.ctx().request_repaint();
        }

        VideoPlayerResponse {
            response,
            clicked,
            state_changed,
            toggle_fullscreen: controls_response.toggle_fullscreen,
        }
    }

    /// Updates the current frame from the decode queue.
    ///
    /// This stores the frame in pending_frame for the render callback to process.
    /// The actual texture creation and upload happens in the prepare callback
    /// which has access to VideoRenderResources.
    fn update_frame(&mut self) {
        // Get the next frame to display
        if let Some(frame) = self.scheduler.get_next_frame(&self.frame_queue) {
            // Frame received — loop seek succeeded if one was pending
            self.loop_seek_pending = false;

            // Update state with current position
            self.state = VideoState::Playing {
                position: frame.pts,
            };

            // Check if texture needs to be recreated
            let texture_guard = self.texture.lock();
            let (width, height) = frame.dimensions();
            let format = frame.frame.format();

            let needs_recreate = texture_guard
                .as_ref()
                .map(|t| t.dimensions() != (width, height) || t.format() != format)
                .unwrap_or(true);
            drop(texture_guard);

            // Store the frame for the render callback to process (lock-free write)
            // For zero-copy, prefer GPU surface but include CPU fallback for graceful degradation
            #[cfg(target_os = "android")]
            if let Some(surface) = frame.frame.as_android_surface() {
                self.pending_frame_writer.write(PendingFrame {
                    frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                    pixel_format: Some(surface.format),
                    android_surface: Some(surface.clone()),
                    needs_recreate,
                });
                return;
            }

            #[cfg(target_os = "macos")]
            if let Some(surface) = frame.frame.as_macos_surface() {
                self.pending_frame_writer.write(PendingFrame {
                    frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                    pixel_format: Some(surface.format),
                    macos_surface: Some(surface.clone()),
                    needs_recreate,
                });
                return;
            }

            #[cfg(all(target_os = "windows", feature = "windows-native-video"))]
            if let Some(surface) = frame.frame.as_windows_surface() {
                self.pending_frame_writer.write(PendingFrame {
                    frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                    pixel_format: Some(surface.format),
                    windows_surface: Some(surface.clone()),
                    needs_recreate,
                });
                return;
            }

            #[cfg(target_os = "linux")]
            if let Some(surface) = frame.frame.as_linux_surface() {
                self.pending_frame_writer.write(PendingFrame {
                    frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                    pixel_format: Some(surface.format),
                    linux_surface: Some(surface.clone()),
                    needs_recreate,
                });
                return;
            }

            // CPU fallback path
            if let Some(cpu_frame) = frame.frame.as_cpu() {
                self.pending_frame_writer.write(PendingFrame {
                    frame: Some(cpu_frame.clone()),
                    pixel_format: Some(cpu_frame.format),
                    #[cfg(target_os = "android")]
                    android_surface: None,
                    #[cfg(target_os = "macos")]
                    macos_surface: None,
                    #[cfg(all(target_os = "windows", feature = "windows-native-video"))]
                    windows_surface: None,
                    #[cfg(target_os = "linux")]
                    linux_surface: None,
                    needs_recreate,
                });
            }
        }
    }

    /// Tries to get a preview frame from the queue without advancing playback.
    ///
    /// This is used to display the first frame before playback starts.
    fn try_get_preview_frame(&mut self) {
        // Only try if we don't already have a pending frame
        if self.pending_frame_reader.has_new_frame() {
            return;
        }

        // Peek at the queue - don't pop so we can play from the beginning
        let Some(frame) = self.frame_queue.peek() else {
            return;
        };

        // Check if texture needs to be recreated
        let (width, height) = frame.dimensions();
        let format = frame.frame.format();
        let needs_recreate = self
            .texture
            .lock()
            .as_ref()
            .map(|t| t.dimensions() != (width, height) || t.format() != format)
            .unwrap_or(true);

        // Store the frame for the render callback to process (lock-free write)
        // For zero-copy, prefer GPU surface but include CPU fallback for graceful degradation
        #[cfg(target_os = "android")]
        if let Some(surface) = frame.frame.as_android_surface() {
            self.pending_frame_writer.write(PendingFrame {
                frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                pixel_format: Some(surface.format),
                android_surface: Some(surface.clone()),
                needs_recreate,
            });
            return;
        }

        #[cfg(target_os = "macos")]
        if let Some(surface) = frame.frame.as_macos_surface() {
            self.pending_frame_writer.write(PendingFrame {
                frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                pixel_format: Some(surface.format),
                macos_surface: Some(surface.clone()),
                needs_recreate,
            });
            return;
        }

        #[cfg(all(target_os = "windows", feature = "windows-native-video"))]
        if let Some(surface) = frame.frame.as_windows_surface() {
            self.pending_frame_writer.write(PendingFrame {
                frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                pixel_format: Some(surface.format),
                windows_surface: Some(surface.clone()),
                needs_recreate,
            });
            return;
        }

        #[cfg(target_os = "linux")]
        if let Some(surface) = frame.frame.as_linux_surface() {
            self.pending_frame_writer.write(PendingFrame {
                frame: surface.cpu_fallback.clone(), // CPU fallback for when zero-copy fails
                pixel_format: Some(surface.format),
                linux_surface: Some(surface.clone()),
                needs_recreate,
            });
            return;
        }

        // CPU fallback path
        let Some(cpu_frame) = frame.frame.as_cpu() else {
            return;
        };

        self.pending_frame_writer.write(PendingFrame {
            frame: Some(cpu_frame.clone()),
            pixel_format: Some(cpu_frame.format),
            #[cfg(target_os = "android")]
            android_surface: None,
            #[cfg(target_os = "macos")]
            macos_surface: None,
            #[cfg(all(target_os = "windows", feature = "windows-native-video"))]
            windows_surface: None,
            #[cfg(target_os = "linux")]
            linux_surface: None,
            needs_recreate,
        });
    }

    /// Sets the wgpu render state for this player.
    ///
    /// This must be called before the player can render frames if the player
    /// was created with `new()` instead of `with_wgpu()`.
    /// Sets the wgpu render state for texture creation.
    ///
    /// # Arguments
    ///
    /// * `wgpu_render_state` - The egui wgpu render state
    ///
    /// # Note
    ///
    /// If your app uses a depth buffer, use [`Self::set_wgpu_state_with_depth`] instead.
    pub fn set_wgpu_state(&mut self, wgpu_render_state: &egui_wgpu::RenderState) {
        self.set_wgpu_state_with_depth(wgpu_render_state, None);
    }

    /// Sets the wgpu render state with depth format for texture creation.
    ///
    /// Use this method when your app's render pass uses a depth buffer.
    ///
    /// # Arguments
    ///
    /// * `wgpu_render_state` - The egui wgpu render state
    /// * `depth_format` - Optional depth format to match the render pass
    pub fn set_wgpu_state_with_depth(
        &mut self,
        wgpu_render_state: &egui_wgpu::RenderState,
        depth_format: Option<wgpu::TextureFormat>,
    ) {
        self.device = Some(wgpu_render_state.device.clone());
        self.queue = Some(wgpu_render_state.queue.clone());

        // Register video render resources if not already done
        {
            let renderer = wgpu_render_state.renderer.read();
            if renderer
                .callback_resources
                .get::<VideoRenderResources>()
                .is_none()
            {
                drop(renderer);
                VideoRenderResources::register(wgpu_render_state, depth_format);
            }
        }
    }

    /// Renders an error overlay when video fails to load.
    fn render_error(&self, ui: &mut Ui, rect: egui::Rect, error: &VideoError) {
        use egui::{Align2, Color32, CornerRadius, FontId};

        // Draw dark background
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::from_rgb(30, 30, 30));

        // Draw error icon (X)
        let center = rect.center();
        let icon_size = 40.0;
        let stroke = egui::Stroke::new(4.0, Color32::from_rgb(255, 100, 100));

        ui.painter().line_segment(
            [
                egui::pos2(center.x - icon_size / 2.0, center.y - icon_size / 2.0),
                egui::pos2(center.x + icon_size / 2.0, center.y + icon_size / 2.0),
            ],
            stroke,
        );
        ui.painter().line_segment(
            [
                egui::pos2(center.x + icon_size / 2.0, center.y - icon_size / 2.0),
                egui::pos2(center.x - icon_size / 2.0, center.y + icon_size / 2.0),
            ],
            stroke,
        );

        // Draw error message
        let error_text = format!("Video Error: {error}");
        ui.painter().text(
            egui::pos2(center.x, center.y + icon_size + 10.0),
            Align2::CENTER_TOP,
            error_text,
            FontId::proportional(12.0),
            Color32::from_rgb(200, 200, 200),
        );
    }

    /// Renders a loading indicator while video is initializing.
    fn render_loading(&self, ui: &mut Ui, rect: egui::Rect) {
        use egui::{Align2, Color32, CornerRadius, FontId};

        // Draw dark background
        ui.painter()
            .rect_filled(rect, CornerRadius::ZERO, Color32::from_rgb(30, 30, 30));

        let center = rect.center();

        // Draw animated loading spinner
        let time = ui.input(|i| i.time);
        let spinner_radius = 20.0;
        let num_dots = 8;
        let dot_radius = 4.0;

        for i in 0..num_dots {
            let angle = (i as f64 / num_dots as f64) * std::f64::consts::TAU + time * 2.0;
            let x = center.x + (angle.cos() * spinner_radius as f64) as f32;
            let y = center.y + (angle.sin() * spinner_radius as f64) as f32;

            // Fade dots based on position in rotation
            let alpha = ((i as f64 / num_dots as f64 + time * 2.0).fract() * 255.0) as u8;
            let color = Color32::from_rgba_unmultiplied(200, 200, 200, alpha);

            ui.painter()
                .circle_filled(egui::pos2(x, y), dot_radius, color);
        }

        // Draw "Loading..." text
        ui.painter().text(
            egui::pos2(center.x, center.y + spinner_radius + 20.0),
            Align2::CENTER_TOP,
            "Loading...",
            FontId::proportional(12.0),
            Color32::from_rgb(200, 200, 200),
        );
    }

    /// Renders a buffering progress indicator overlay.
    fn render_buffering_overlay(&self, ui: &mut Ui, rect: egui::Rect, percent: i32) {
        use egui::{Align2, Color32, CornerRadius, FontId, Stroke};

        let center = rect.center();

        // Semi-transparent dark overlay
        ui.painter().rect_filled(
            rect,
            CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(0, 0, 0, 160),
        );

        // Progress ring parameters
        let ring_radius = 30.0;
        let ring_thickness = 4.0;

        // Draw background ring (dark gray)
        ui.painter().circle_stroke(
            center,
            ring_radius,
            Stroke::new(ring_thickness, Color32::from_rgb(60, 60, 60)),
        );

        // Draw progress arc
        let progress = percent as f32 / 100.0;
        let num_segments = 32;
        let segments_to_draw = (num_segments as f32 * progress) as usize;

        if segments_to_draw > 0 {
            let start_angle = -std::f32::consts::FRAC_PI_2; // Start from top

            for i in 0..segments_to_draw {
                let angle1 = start_angle + (i as f32 / num_segments as f32) * std::f32::consts::TAU;
                let angle2 =
                    start_angle + ((i + 1) as f32 / num_segments as f32) * std::f32::consts::TAU;

                let p1 = egui::pos2(
                    center.x + angle1.cos() * ring_radius,
                    center.y + angle1.sin() * ring_radius,
                );
                let p2 = egui::pos2(
                    center.x + angle2.cos() * ring_radius,
                    center.y + angle2.sin() * ring_radius,
                );

                ui.painter().line_segment(
                    [p1, p2],
                    Stroke::new(ring_thickness, Color32::from_rgb(100, 180, 255)),
                );
            }
        }

        // Draw percentage text in center
        ui.painter().text(
            center,
            Align2::CENTER_CENTER,
            format!("{percent}%"),
            FontId::proportional(14.0),
            Color32::WHITE,
        );

        // Draw "Buffering" text below
        ui.painter().text(
            egui::pos2(center.x, center.y + ring_radius + 15.0),
            Align2::CENTER_TOP,
            "Buffering",
            FontId::proportional(12.0),
            Color32::from_rgb(200, 200, 200),
        );
    }

    /// Renders the video frame.
    fn render(&self, ui: &mut Ui, rect: egui::Rect) {
        // Get the current pixel format from pending frame or existing texture
        let format = {
            let pending = self.pending_frame_reader.peek();
            if let Some(ref frame) = pending.frame {
                frame.format
            } else {
                self.texture
                    .lock()
                    .as_ref()
                    .map(|t| t.format())
                    .unwrap_or(PixelFormat::Yuv420p)
            }
        };

        // Create render callback
        let callback = VideoRenderCallback {
            texture: Arc::clone(&self.texture),
            pending_frame_reader: self.pending_frame_reader.clone(),
            format,
            rect,
            #[cfg(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "android",
                all(target_os = "windows", feature = "windows-native-video")
            ))]
            fallback_logged: Arc::clone(&self.fallback_logged),
            #[cfg(target_os = "android")]
            player_id: self.android_player_id,
        };

        // Add paint callback
        ui.painter()
            .add(egui_wgpu::Callback::new_paint_callback(rect, callback));
    }

    /// Renders a subtitle text overlay at the bottom of the video.
    fn render_subtitle(&self, ui: &mut Ui, rect: egui::Rect, text: &str) {
        use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Vec2};

        let style = &self.subtitle_style;

        // Measure text first to determine layout
        let font = FontId::proportional(style.font_size);
        let galley = ui
            .painter()
            .layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE);

        // Position text centered horizontally, offset from bottom by bottom_margin
        let text_pos = Pos2::new(
            rect.center().x - galley.size().x / 2.0,
            rect.max.y - style.bottom_margin - galley.size().y,
        );

        // Draw background if enabled
        if style.show_background {
            let bg_padding = 8.0;
            let bg_rect = Rect::from_min_size(
                Pos2::new(text_pos.x - bg_padding, text_pos.y - bg_padding / 2.0),
                Vec2::new(
                    galley.size().x + bg_padding * 2.0,
                    galley.size().y + bg_padding,
                ),
            );

            let bg_color = Color32::from_rgba_unmultiplied(
                style.background_color[0],
                style.background_color[1],
                style.background_color[2],
                style.background_color[3],
            );

            ui.painter()
                .rect_filled(bg_rect, CornerRadius::same(4), bg_color);
        }

        // Draw text with shadow for readability
        let shadow_color = Color32::from_rgba_unmultiplied(0, 0, 0, 200);
        let text_color = Color32::from_rgba_unmultiplied(
            style.text_color[0],
            style.text_color[1],
            style.text_color[2],
            style.text_color[3],
        );

        // Draw shadow offset
        ui.painter().text(
            Pos2::new(text_pos.x + 1.5, text_pos.y + 1.5),
            Align2::LEFT_TOP,
            text,
            font.clone(),
            shadow_color,
        );

        // Draw main text
        ui.painter()
            .text(text_pos, Align2::LEFT_TOP, text, font, text_color);
    }
}

impl Drop for VideoPlayer {
    fn drop(&mut self) {
        // Stop the decode thread
        if let Some(ref thread) = self.decode_thread {
            thread.stop();
        }

        // Stop the audio thread
        #[cfg(target_os = "macos")]
        if let Some(ref audio_thread) = self.audio_thread {
            audio_thread.stop();
        }
    }
}

/// Response from showing a video player widget.
pub struct VideoPlayerResponse {
    /// The egui response from the widget allocation
    pub response: Response,
    /// Whether the video was clicked
    pub clicked: bool,
    /// Whether the playback state changed
    pub state_changed: bool,
    /// Whether fullscreen was toggled
    pub toggle_fullscreen: bool,
}

/// Extension trait for easily adding video players to egui.
pub trait VideoPlayerExt {
    /// Shows a video player for the given URL.
    fn video_player(&mut self, player: &mut VideoPlayer, size: Vec2) -> VideoPlayerResponse;
}

impl VideoPlayerExt for Ui {
    fn video_player(&mut self, player: &mut VideoPlayer, size: Vec2) -> VideoPlayerResponse {
        player.show(self, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_player_creation() {
        let player = VideoPlayer::new("test.mp4");
        assert!(matches!(player.state(), VideoState::Loading));
        assert!(!player.is_playing());
    }

    #[test]
    fn test_video_player_autoplay() {
        let player = VideoPlayer::new("test.mp4").with_autoplay(true);
        assert!(player.autoplay);
    }
}
