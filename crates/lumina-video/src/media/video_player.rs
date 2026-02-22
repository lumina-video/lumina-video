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

use super::audio::AudioHandle;
use super::player::CorePlayer;
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
use super::video::{CpuFrame, PixelFormat, VideoError, VideoMetadata, VideoState};
#[cfg(feature = "moq")]
use super::video::VideoDecoderBackend;

#[cfg(feature = "moq")]
use super::moq_decoder::{MoqDecoder, MoqStatsHandle, MoqStatsSnapshot};

use super::video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
use super::video_texture::{VideoRenderCallback, VideoRenderResources, VideoTexture};
#[cfg(feature = "moq")]
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
///
/// Delegates core playback logic to [`CorePlayer`].
pub struct VideoPlayer {
    /// Core playback engine (decode pipeline, frame queue, A/V sync)
    core: CorePlayer,
    /// Current video texture
    texture: Arc<Mutex<Option<VideoTexture>>>,
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
    /// Rate-limit "no CPU fallback" warning (log once per player instance, persists across frames)
    #[cfg(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        all(target_os = "windows", feature = "windows-native-video")
    ))]
    fallback_logged: Arc<std::sync::atomic::AtomicBool>,
    /// MoQ decoder stats handle for monitoring MoQ-specific pipeline state
    /// Wrapped in Mutex so init thread can populate it after MoqDecoder is created
    #[cfg(feature = "moq")]
    moq_stats: Arc<parking_lot::Mutex<Option<MoqStatsHandle>>>,
    /// Whether we've already late-bound the MoQ audio handle to self.audio_handle
    #[cfg(feature = "moq")]
    moq_audio_bound: bool,
    /// MoQ-specific init promise (MoQ init bypasses CorePlayer)
    #[cfg(feature = "moq")]
    moq_init_promise: Option<Promise<Result<Box<dyn VideoDecoderBackend + Send>, VideoError>>>,
    /// MoQ-specific init thread handle
    #[cfg(feature = "moq")]
    moq_init_thread: Option<std::thread::JoinHandle<()>>,
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
        let core = CorePlayer::new(url);
        Self {
            core,
            texture: Arc::new(Mutex::new(None)),
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
            #[cfg(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "android",
                all(target_os = "windows", feature = "windows-native-video")
            ))]
            fallback_logged: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "moq")]
            moq_stats: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "moq")]
            moq_audio_bound: false,
            #[cfg(feature = "moq")]
            moq_init_promise: None,
            #[cfg(feature = "moq")]
            moq_init_thread: None,
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
        let core = CorePlayer::new(url);
        Self {
            core,
            texture: Arc::new(Mutex::new(None)),
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
            #[cfg(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "android",
                all(target_os = "windows", feature = "windows-native-video")
            ))]
            fallback_logged: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            #[cfg(feature = "moq")]
            moq_stats: Arc::new(parking_lot::Mutex::new(None)),
            #[cfg(feature = "moq")]
            moq_audio_bound: false,
            #[cfg(feature = "moq")]
            moq_init_promise: None,
            #[cfg(feature = "moq")]
            moq_init_thread: None,
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
        self.core.audio_handle().set_muted(muted);
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
    /// MoQ URLs are handled here (MoQ is egui-layer concern); non-MoQ URLs
    /// delegate to [`CorePlayer::init_decoder`].
    pub fn start_async_init(&mut self) {
        if self.core.is_initialized() || self.core.is_init_pending() {
            return;
        }

        // MoQ init already in progress (bypasses CorePlayer)
        #[cfg(feature = "moq")]
        if self.moq_init_promise.is_some() {
            return;
        }

        // Check for MoQ URL — handle in the egui layer
        #[cfg(feature = "moq")]
        if MoqDecoder::is_moq_url(self.core.url()) {
            let url = self.core.url().to_string();
            let moq_stats_holder = Arc::clone(&self.moq_stats);

            // MoQ init uses its own promise/thread (not CorePlayer's)
            let (sender, promise) = Promise::new();

            let handle = std::thread::spawn(move || {
                tracing::info!("Using MoQ decoder for live stream: {}", url);
                let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> =
                    match MoqDecoder::new(&url) {
                        Ok(decoder) => {
                            *moq_stats_holder.lock() = Some(decoder.stats_handle());
                            Ok(Box::new(decoder) as Box<dyn VideoDecoderBackend + Send>)
                        }
                        Err(e) => Err(e),
                    };
                sender.send(result);
            });

            // Store these for check_init_complete to pick up
            self.moq_init_promise = Some(promise);
            self.moq_init_thread = Some(handle);
            return;
        }

        // Non-MoQ: delegate to CorePlayer
        self.core.init_decoder();
    }

    /// Checks if async initialization is complete and finishes setup.
    /// Returns true if initialization is complete (success or error).
    fn check_init_complete(&mut self) -> bool {
        if self.core.is_initialized() {
            return true;
        }

        // Check MoQ init path first (bypasses CorePlayer)
        #[cfg(feature = "moq")]
        if let Some(ref promise) = self.moq_init_promise {
            if promise.ready().is_some() {
                let Some(promise) = self.moq_init_promise.take() else {
                    return false;
                };
                self.moq_init_thread = None;
                match promise.try_take() {
                    Ok(Ok(decoder)) => {
                        // MoQ decoder ready — wrap in CorePlayer
                        let url = self.core.url().to_string();
                        self.core = CorePlayer::with_decoder(url, decoder);
                        if self.autoplay {
                            self.core.play_with_muted(self.muted);
                        }
                        return true;
                    }
                    Ok(Err(e)) => {
                        self.core.set_state(VideoState::Error(e));
                        return true;
                    }
                    Err(_) => {
                        self.core.set_state(VideoState::Error(VideoError::Generic(
                            "MoQ init thread crashed".into(),
                        )));
                        return true;
                    }
                }
            }
            return false; // MoQ init still pending
        }

        // Non-MoQ: delegate to CorePlayer
        let complete = self.core.check_init_complete();
        if complete && self.autoplay && matches!(self.core.state(), VideoState::Ready) {
            self.core.play_with_muted(self.muted);
        }
        complete
    }

    /// Starts or resumes playback.
    pub fn play(&mut self) {
        self.core.play_with_muted(self.muted);
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        self.core.pause();
    }

    /// Toggles between play and pause.
    pub fn toggle_playback(&mut self) {
        self.core.toggle_playback();
    }

    /// Seeks to a specific position.
    pub fn seek(&mut self, position: Duration) {
        self.core.seek(position);
    }

    /// Returns the current playback state.
    pub fn state(&self) -> &VideoState {
        self.core.state()
    }

    /// Returns the video metadata if available.
    pub fn metadata(&self) -> Option<&VideoMetadata> {
        self.core.metadata()
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        self.core.position()
    }

    /// Returns the video duration if known.
    pub fn duration(&self) -> Option<Duration> {
        self.core.duration()
    }

    /// Returns the A/V sync metrics tracker.
    pub fn sync_metrics(&self) -> &SyncMetrics {
        self.core.sync_metrics()
    }

    /// Returns a snapshot of current A/V sync metrics.
    pub fn sync_metrics_snapshot(&self) -> SyncMetricsSnapshot {
        self.core.sync_metrics_snapshot()
    }

    /// Returns a snapshot of Linux zero-copy metrics (Linux GStreamer only).
    #[cfg(target_os = "linux")]
    pub fn linux_zero_copy_metrics(
        &self,
    ) -> Option<super::linux_video::LinuxZeroCopyMetricsSnapshot> {
        self.core.linux_zero_copy_metrics()
    }

    /// Returns a snapshot of MoQ decoder stats (MoQ streams only).
    #[cfg(feature = "moq")]
    pub fn moq_stats(&self) -> Option<MoqStatsSnapshot> {
        self.moq_stats
            .lock()
            .as_ref()
            .map(|handle| handle.snapshot())
    }

    /// Returns the video dimensions (width, height).
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.core.dimensions()
    }

    /// Returns the video frame rate.
    pub fn frame_rate(&self) -> Option<f32> {
        self.core.frame_rate()
    }

    /// Returns true if the video is currently playing.
    pub fn is_playing(&self) -> bool {
        self.core.is_playing()
    }

    /// Returns the current buffering percentage (0-100).
    pub fn buffering_percent(&self) -> i32 {
        self.core.buffering_percent()
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &AudioHandle {
        self.core.audio_handle()
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.core.audio_handle().volume()
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&mut self, volume: u32) {
        self.core.set_volume(volume);
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.core.audio_handle().is_muted()
    }

    /// Toggles the mute state.
    pub fn toggle_mute(&mut self) {
        self.core.audio_handle().toggle_mute();
        self.muted = self.core.audio_handle().is_muted();
        self.core.set_muted(self.muted);
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
                moq_ah.set_muted(self.core.audio_handle().is_muted());
                moq_ah.set_volume(self.core.audio_handle().volume());
                moq_ah.set_available(true);

                // Bind to FrameScheduler for sync metrics (drift tracking).
                // Use metrics-only mode: wall-clock drives frame pacing, not audio.
                // MoQ audio arrives late; switching to audio-as-master would cause
                // a position discontinuity that freezes or jumps video.
                self.core
                    .scheduler_mut()
                    .set_audio_handle_metrics_only(moq_ah.clone());

                // Replace the player-level handle (for UI mute/volume controls only)
                self.core.set_audio_handle(moq_ah);

                // Don't enable playback epoch here — let the FrameScheduler's
                // clock rebase enable it, so cpal stays gated until video is ready.
                // This prevents audio from playing before the A/V clocks are aligned.
                tracing::info!(
                    "MoQ audio: late bind (metrics only, wall-clock pacing, epoch deferred)"
                );

                self.moq_audio_bound = true;
                tracing::info!("MoQ audio handle bound to VideoPlayer + FrameScheduler");
            }
        } else {
            // Check if handle went stale (thread torn down)
            if !stats.is_audio_alive() {
                tracing::info!("MoQ audio handle stale (thread torn down), unbinding");

                // Create a fresh placeholder handle and migrate state
                let placeholder = AudioHandle::new();
                placeholder.set_muted(self.core.audio_handle().is_muted());
                placeholder.set_volume(self.core.audio_handle().volume());
                self.core.set_audio_handle(placeholder);

                // Clear FrameScheduler's audio handle so it falls back to wall-clock
                self.core.scheduler_mut().clear_audio_handle();

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

        // Start async initialization if needed (idempotent — guards inside)
        self.start_async_init();

        // Check if async init is complete
        if !self.core.is_initialized() {
            self.check_init_complete();
        }

        // Sync metadata from decode thread (for lazy metadata like macOS AVPlayer)
        self.core.sync_metadata_from_decode_thread();

        // Late-bind MoQ audio handle (async — not available at init time)
        #[cfg(feature = "moq")]
        if self.core.is_initialized() {
            self.poll_moq_audio_handle();
        }

        // Update frame if playback requested (even if buffering), or try to get preview frame when Ready/Paused
        if self.core.is_playback_requested() {
            self.update_frame();
        } else if matches!(self.core.state(), VideoState::Ready | VideoState::Paused { .. }) {
            // Try to get preview frame from queue (non-blocking)
            self.try_get_preview_frame();
        }

        // Check for end of stream
        if self.core.is_eos() && self.core.is_queue_empty() {
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
                self.core.set_state(VideoState::Ended);
            }
        }

        // Render the video frame, loading state, or error
        match self.core.state() {
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
                let is_audio_stall = self.core.is_audio_stall();
                if (buffering < 100 || is_audio_stall) && self.is_playing() {
                    let pct = if is_audio_stall { 90 } else { buffering };
                    self.render_buffering_overlay(ui, rect, pct);
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
            let controls = VideoControls::new(self.core.state(), self.position(), self.duration())
                .with_config(self.controls_config.clone())
                .with_muted(self.core.audio_handle().is_muted())
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
        #[cfg(feature = "moq")]
        let is_initializing =
            self.core.is_init_pending() || self.moq_init_promise.is_some();
        #[cfg(not(feature = "moq"))]
        let is_initializing = self.core.is_init_pending();
        let has_pending_frame = self.pending_frame_reader.has_new_frame();
        let is_buffering = self.buffering_percent() < 100;
        if self.core.is_playback_requested()
            || is_initializing
            || has_pending_frame
            || is_buffering
            || matches!(
                self.core.state(),
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
        // Get the next frame to display (CorePlayer updates state internally)
        if let Some(frame) = self.core.poll_frame() {
            // Frame received — loop seek succeeded if one was pending
            self.loop_seek_pending = false;

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
        let Some(frame) = self.core.peek_frame() else {
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
            player_id: self.core.android_player_id(),
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

// Note: No Drop impl needed — CorePlayer has its own Drop that stops
// the decode thread and audio thread.

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
