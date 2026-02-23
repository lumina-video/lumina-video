//! Headless video player (decoder-agnostic, egui-free).
//!
//! [`CorePlayer`] encapsulates the decode pipeline, frame queue, A/V sync,
//! and playback state machine. It is consumed by:
//!
//! - `lumina-video::VideoPlayer` (egui widget wrapper)
//! - `lumina-video-ios` (C FFI for iOS/Swift)
//!
//! CorePlayer is MoQ-agnostic: callers choose the decoder (platform default
//! or MoqDecoder) and pass it via [`CorePlayer::with_decoder`].

use std::sync::Arc;
use std::time::Duration;

#[cfg(target_os = "linux")]
use parking_lot::Mutex;
use poll_promise::Promise;

use crate::audio::AudioHandle;
#[cfg(target_os = "macos")]
use crate::frame_queue::AudioThread;
use crate::frame_queue::{DecodeThread, FrameQueue, FrameScheduler};
#[cfg(target_os = "linux")]
use crate::linux_video::ZeroCopyGStreamerDecoder;
#[cfg(any(target_os = "ios", target_os = "macos"))]
use crate::macos_video::MacOSVideoDecoder;
use crate::sync_metrics::{SyncMetrics, SyncMetricsSnapshot};
use crate::video::{VideoDecoderBackend, VideoError, VideoFrame, VideoMetadata, VideoState};

/// Returns true if the URL points to a container format supported by AVFoundation.
#[cfg(any(target_os = "ios", target_os = "macos"))]
fn is_avfoundation_supported_container(url: &str) -> bool {
    let path = url.split('?').next().unwrap_or(url);
    let ext = path
        .rsplit('.')
        .next()
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    matches!(
        ext.as_str(),
        "mp4" | "m4v" | "mov" | "m4a" | "m3u8" | "ts" | "avi" | "aac" | "mp3" | "wav" | "aiff"
    )
}

/// Headless, egui-free video player.
///
/// Manages the decode pipeline, frame queue, A/V synchronization, and
/// playback state machine. Does NOT handle GPU texture upload or UI —
/// that belongs in the egui layer (`VideoPlayer`).
///
/// # State Machine
///
/// ```text
/// new(url) / with_decoder(decoder)
///   → Loading
///
/// init completes successfully → Ready
/// init fails                  → Error (terminal)
///
/// play()  [Ready/Paused/Ended] → Playing
/// pause() [Playing]            → Paused
/// seek()  [Playing/Paused/Ended] → stays in current state
///
/// EOS detected → Ended
/// ```
pub struct CorePlayer {
    /// Current playback state
    state: VideoState,
    /// Video metadata (populated after init)
    metadata: Option<VideoMetadata>,
    /// Frame queue for decoded frames
    frame_queue: Arc<FrameQueue>,
    /// Background decode thread
    decode_thread: Option<DecodeThread>,
    /// Frame timing and A/V sync
    scheduler: FrameScheduler,
    /// Audio handle for volume/mute control
    audio_handle: AudioHandle,
    /// Audio decode/playback thread (macOS FFmpeg audio)
    #[cfg(target_os = "macos")]
    audio_thread: Option<AudioThread>,
    /// Whether the player has completed initialization
    initialized: bool,
    /// The URL being played
    url: String,
    /// Background thread for async initialization
    init_thread: Option<std::thread::JoinHandle<()>>,
    /// Promise for async initialization result
    init_promise: Option<Promise<Result<Box<dyn VideoDecoderBackend + Send>, VideoError>>>,
    /// Android player ID for multi-player frame isolation
    #[cfg(target_os = "android")]
    android_player_id: u64,
    /// Linux zero-copy metrics
    #[cfg(target_os = "linux")]
    linux_zero_copy_metrics: Arc<Mutex<Option<Arc<crate::linux_video::ZeroCopyMetrics>>>>,
}

impl CorePlayer {
    /// Creates a new player for the given URL.
    ///
    /// The player starts in [`VideoState::Loading`]. Call [`init_decoder`] to
    /// begin async initialization, then poll with [`check_init_complete`].
    pub fn new(url: impl Into<String>) -> Self {
        let audio_handle = AudioHandle::new();
        let scheduler = FrameScheduler::with_audio_handle(audio_handle.clone());
        Self {
            state: VideoState::Loading,
            metadata: None,
            frame_queue: Arc::new(FrameQueue::with_default_capacity()),
            decode_thread: None,
            scheduler,
            audio_handle,
            #[cfg(target_os = "macos")]
            audio_thread: None,
            initialized: false,
            url: url.into(),
            init_thread: None,
            init_promise: None,
            #[cfg(target_os = "android")]
            android_player_id: 0,
            #[cfg(target_os = "linux")]
            linux_zero_copy_metrics: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a player with a pre-created decoder.
    ///
    /// Use this when the caller manages decoder creation (e.g., MoQ).
    /// The player transitions directly to [`VideoState::Ready`].
    pub fn with_decoder(
        url: impl Into<String>,
        decoder: Box<dyn VideoDecoderBackend + Send>,
    ) -> Self {
        let audio_handle = AudioHandle::new();
        let scheduler = FrameScheduler::with_audio_handle(audio_handle.clone());
        let metadata = decoder.metadata().clone();
        let frame_queue = Arc::new(FrameQueue::with_default_capacity());

        let decode_thread = DecodeThread::new(decoder, Arc::clone(&frame_queue));

        Self {
            state: VideoState::Ready,
            metadata: Some(metadata),
            frame_queue,
            decode_thread: Some(decode_thread),
            scheduler,
            audio_handle,
            #[cfg(target_os = "macos")]
            audio_thread: None,
            initialized: true,
            url: url.into(),
            init_thread: None,
            init_promise: None,
            #[cfg(target_os = "android")]
            android_player_id: 0,
            #[cfg(target_os = "linux")]
            linux_zero_copy_metrics: Arc::new(Mutex::new(None)),
        }
    }

    /// Starts async initialization of the decoder.
    ///
    /// Spawns a background thread to open the video and prepare for playback.
    /// Poll with [`check_init_complete`] to detect when init finishes.
    ///
    /// On macOS, AVPlayer requires main thread initialization, so the macOS
    /// decoder is created synchronously before spawning the background thread.
    pub fn init_decoder(&mut self) {
        if self.initialized || self.init_thread.is_some() {
            return;
        }

        let url = self.url.clone();
        let (sender, promise) = Promise::new();
        self.init_promise = Some(promise);

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
                None
            }
        };

        // iOS: dispatch decoder creation to main queue for any-thread safety.
        // run_on_main: if already on main thread, runs inline (no dispatch).
        // Otherwise dispatches synchronously to main queue and waits.
        // MacOSVideoDecoder::new() internally calls MainThreadMarker::new()
        // which succeeds because run_on_main guarantees main-thread execution.
        #[cfg(target_os = "ios")]
        let ios_decoder_result: Option<Result<MacOSVideoDecoder, VideoError>> = {
            if is_avfoundation_supported_container(&url) {
                tracing::info!("Initializing iOS VideoToolbox decoder (dispatching to main queue)");
                let url_clone = url.clone();
                Some(dispatch2::run_on_main(move |_mtm| {
                    MacOSVideoDecoder::new(&url_clone)
                }))
            } else {
                None
            }
        };

        #[cfg(target_os = "linux")]
        let linux_metrics_holder = Arc::clone(&self.linux_zero_copy_metrics);

        let handle = std::thread::spawn(move || {
            #[cfg(target_os = "macos")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                use crate::video_decoder::FfmpegDecoder;
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
                        tracing::info!("Using FFmpeg for unsupported container format");
                        FfmpegDecoder::new(&url)
                            .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                }
            };

            #[cfg(target_os = "ios")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                match ios_decoder_result {
                    Some(Ok(d)) => {
                        tracing::info!("Using iOS VideoToolbox hardware decoder");
                        Ok(Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    Some(Err(e)) => Err(e),
                    None => Err(VideoError::DecoderInit(format!(
                        "Unsupported container format on iOS: {}",
                        url
                    ))),
                }
            };

            #[cfg(target_os = "android")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                use crate::android_video::AndroidVideoDecoder;
                tracing::info!("Using Android ExoPlayer decoder for {}", url);
                AndroidVideoDecoder::new(&url)
                    .map(|d| Box::new(d) as Box<dyn VideoDecoderBackend + Send>)
            };

            #[cfg(target_os = "linux")]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                match ZeroCopyGStreamerDecoder::new(&url) {
                    Ok(decoder) => {
                        *linux_metrics_holder.lock() = Some(Arc::clone(decoder.metrics()));
                        Ok(Box::new(decoder) as Box<dyn VideoDecoderBackend + Send>)
                    }
                    Err(e) => Err(e),
                }
            };

            #[cfg(not(any(
                target_os = "android",
                target_os = "ios",
                target_os = "linux",
                target_os = "macos",
            )))]
            let result: Result<Box<dyn VideoDecoderBackend + Send>, VideoError> = {
                let _ = &url;
                Err(VideoError::DecoderInit(
                    "No video decoder available for this platform".to_string(),
                ))
            };

            sender.send(result);
        });

        self.init_thread = Some(handle);
    }

    /// Checks if async initialization is complete and finishes setup.
    ///
    /// Returns `true` if initialization is complete (success or error).
    pub fn check_init_complete(&mut self) -> bool {
        if self.initialized {
            return true;
        }

        let is_ready = self
            .init_promise
            .as_ref()
            .is_some_and(|p| p.ready().is_some());

        if !is_ready {
            return false;
        }

        let Some(promise) = self.init_promise.take() else {
            return false;
        };

        let Ok(result) = promise.try_take() else {
            self.state = VideoState::Error(VideoError::Generic("Init thread crashed".into()));
            self.init_thread = None;
            return true;
        };

        match result {
            Ok(decoder) => {
                let metadata = decoder.metadata().clone();
                self.metadata = Some(metadata.clone());

                let uses_native_audio = decoder.handles_audio_internally();

                #[cfg(target_os = "android")]
                {
                    self.android_player_id = decoder.android_player_id();
                }

                let decoder_audio_handle = decoder.audio_handle();

                if uses_native_audio {
                    if let Some(ref ah) = decoder_audio_handle {
                        self.audio_handle = ah.clone();
                    }
                    self.audio_handle.set_available(true);
                    self.scheduler.set_audio_handle(self.audio_handle.clone());
                    tracing::info!("Native audio enabled (decoder handles audio internally)");
                }

                let frame_queue = Arc::clone(&self.frame_queue);

                #[cfg(any(target_os = "macos", target_os = "ios"))]
                let decode_thread = if uses_native_audio {
                    DecodeThread::with_audio_handle(
                        decoder,
                        frame_queue,
                        Some(self.audio_handle.clone()),
                    )
                } else {
                    DecodeThread::new(decoder, frame_queue)
                };

                #[cfg(not(any(target_os = "macos", target_os = "ios")))]
                let decode_thread = DecodeThread::new(decoder, frame_queue);

                self.decode_thread = Some(decode_thread);

                // Start FFmpeg audio thread on macOS if decoder doesn't handle audio
                #[cfg(target_os = "macos")]
                if !uses_native_audio {
                    if let Some(audio_thread) = AudioThread::new(&self.url, metadata.start_time) {
                        self.audio_handle = audio_thread.handle();
                        self.scheduler.set_audio_handle(self.audio_handle.clone());
                        tracing::info!("FFmpeg audio playback initialized for {}", self.url);
                        self.audio_thread = Some(audio_thread);
                    }
                }

                self.state = VideoState::Ready;
                self.initialized = true;
                self.init_thread = None;
                true
            }
            Err(e) => {
                self.state = VideoState::Error(e);
                self.init_thread = None;
                self.initialized = true;
                true
            }
        }
    }

    /// Returns whether the player has completed initialization.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns whether initialization is in progress.
    pub fn is_init_pending(&self) -> bool {
        self.init_thread.is_some()
    }

    // =========================================================================
    // Playback control
    // =========================================================================

    /// Starts or resumes playback.
    ///
    /// Preserves the current mute state. Use [`play_with_muted`] to set an
    /// explicit mute state at the same time.
    pub fn play(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            #[cfg(any(
                target_os = "android",
                target_os = "ios",
                target_os = "linux",
                target_os = "macos"
            ))]
            thread.set_muted(self.audio_handle.is_muted());
            thread.play();
            self.scheduler.start();
            self.state = VideoState::Playing {
                position: self.scheduler.position(),
            };
            #[cfg(target_os = "macos")]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.play();
            }
        }
    }

    /// Starts playback with a specific mute state.
    pub fn play_with_muted(&mut self, muted: bool) {
        if let Some(ref thread) = self.decode_thread {
            #[cfg(any(
                target_os = "android",
                target_os = "ios",
                target_os = "linux",
                target_os = "macos"
            ))]
            thread.set_muted(muted);
            thread.play();
            self.scheduler.start();
            self.state = VideoState::Playing {
                position: self.scheduler.position(),
            };
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
            #[cfg(target_os = "macos")]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.pause();
            }
        }
    }

    /// Seeks to a specific position.
    pub fn seek(&mut self, position: Duration) {
        if position == Duration::ZERO {
            tracing::debug!("Seek to ZERO requested from state={:?}", self.state,);
        }

        if let Some(ref thread) = self.decode_thread {
            thread.seek(position);
            self.scheduler.seek(position);

            #[cfg(target_os = "macos")]
            if let Some(ref audio_thread) = self.audio_thread {
                audio_thread.seek(position);
            }

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

    /// Toggles between play and pause.
    pub fn toggle_playback(&mut self) {
        match self.state {
            VideoState::Playing { .. } => self.pause(),
            VideoState::Paused { .. } | VideoState::Ready | VideoState::Ended => self.play(),
            _ => {}
        }
    }

    // =========================================================================
    // State queries
    // =========================================================================

    /// Returns the current playback state.
    pub fn state(&self) -> &VideoState {
        &self.state
    }

    /// Sets the state (used by the egui layer for EOS/loop handling).
    pub fn set_state(&mut self, state: VideoState) {
        self.state = state;
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
        if let Some(ref thread) = self.decode_thread {
            if let Some(dur) = thread.duration() {
                return Some(dur);
            }
        }
        self.metadata.as_ref().and_then(|m| m.duration)
    }

    /// Returns true if the video is currently playing.
    pub fn is_playing(&self) -> bool {
        self.scheduler.is_playing()
    }

    /// Returns true if playback has been requested (even if waiting for first frame).
    pub fn is_playback_requested(&self) -> bool {
        self.scheduler.is_playback_requested()
    }

    /// Returns the A/V sync metrics tracker.
    pub fn sync_metrics(&self) -> &SyncMetrics {
        self.scheduler.sync_metrics()
    }

    /// Returns a snapshot of current A/V sync metrics.
    pub fn sync_metrics_snapshot(&self) -> SyncMetricsSnapshot {
        self.scheduler.sync_metrics().snapshot()
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &AudioHandle {
        &self.audio_handle
    }

    /// Swaps the audio handle (used by MoQ late-binding in VideoPlayer).
    pub fn set_audio_handle(&mut self, ah: AudioHandle) {
        self.audio_handle = ah;
    }

    /// Returns the frame scheduler (for audio handle binding).
    pub fn scheduler(&self) -> &FrameScheduler {
        &self.scheduler
    }

    /// Returns a mutable reference to the frame scheduler.
    pub fn scheduler_mut(&mut self) -> &mut FrameScheduler {
        &mut self.scheduler
    }

    /// Returns the frame queue.
    pub fn frame_queue(&self) -> &Arc<FrameQueue> {
        &self.frame_queue
    }

    /// Returns the decode thread if active.
    pub fn decode_thread(&self) -> Option<&DecodeThread> {
        self.decode_thread.as_ref()
    }

    /// Returns the URL being played.
    pub fn url(&self) -> &str {
        &self.url
    }

    // =========================================================================
    // Frame retrieval
    // =========================================================================

    /// Gets the next frame to display from the scheduler.
    ///
    /// Returns `None` if no frame is ready. Updates internal position tracking.
    pub fn poll_frame(&mut self) -> Option<VideoFrame> {
        let frame = self.scheduler.get_next_frame(&self.frame_queue);
        if let Some(ref f) = frame {
            match self.state {
                VideoState::Playing { .. } => {
                    self.state = VideoState::Playing { position: f.pts };
                }
                VideoState::Paused { .. } => {
                    self.state = VideoState::Paused { position: f.pts };
                }
                VideoState::Ready | VideoState::Buffering { .. } | VideoState::Loading => {
                    self.state = VideoState::Playing { position: f.pts };
                }
                // Don't override Ended/Error — frame may be stale or spurious
                VideoState::Ended | VideoState::Error(_) => {}
            }
        }
        frame
    }

    /// Peeks at the next frame without removing it from the queue.
    pub fn peek_frame(&self) -> Option<VideoFrame> {
        self.frame_queue.peek()
    }

    /// Returns true if the frame queue has reached end-of-stream.
    pub fn is_eos(&self) -> bool {
        self.frame_queue.is_eos()
    }

    /// Returns true if the frame queue is empty.
    pub fn is_queue_empty(&self) -> bool {
        self.frame_queue.is_empty()
    }

    // =========================================================================
    // Metadata sync
    // =========================================================================

    /// Syncs metadata from the decode thread (dynamic values like lazy macOS AVPlayer metadata).
    pub fn sync_metadata_from_decode_thread(&mut self) {
        let Some(ref thread) = self.decode_thread else {
            return;
        };
        let Some(ref mut metadata) = self.metadata else {
            return;
        };

        if let Some(dur) = thread.duration() {
            metadata.duration = Some(dur);
        }
        if let Some((w, h)) = thread.dimensions() {
            if w > 1 && h > 1 {
                metadata.width = w;
                metadata.height = h;
            }
        }
        if let Some(fps) = thread.frame_rate() {
            if fps > 0.0 {
                metadata.frame_rate = fps;
                if metadata.duration.is_none() {
                    self.scheduler.set_frame_rate_pacing(fps);
                } else {
                    self.scheduler.clear_frame_rate_pacing();
                }
            }
        }
    }

    /// Returns the video dimensions (width, height).
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        if let Some(ref thread) = self.decode_thread {
            if let Some(dims) = thread.dimensions() {
                return Some(dims);
            }
        }
        self.metadata.as_ref().map(|m| (m.width, m.height))
    }

    /// Returns the video frame rate.
    pub fn frame_rate(&self) -> Option<f32> {
        if let Some(ref thread) = self.decode_thread {
            if let Some(fps) = thread.frame_rate() {
                return Some(fps);
            }
        }
        self.metadata.as_ref().map(|m| m.frame_rate)
    }

    /// Returns the current buffering percentage (0-100).
    pub fn buffering_percent(&self) -> i32 {
        if let Some(ref thread) = self.decode_thread {
            thread.buffering_percent()
        } else {
            100
        }
    }

    /// Returns whether there's an audio stall.
    pub fn is_audio_stall(&self) -> bool {
        self.scheduler.is_audio_stall()
    }

    // =========================================================================
    // Platform-specific accessors
    // =========================================================================

    /// Returns the Android player ID.
    #[cfg(target_os = "android")]
    pub fn android_player_id(&self) -> u64 {
        self.android_player_id
    }

    /// Returns Linux zero-copy metrics snapshot.
    #[cfg(target_os = "linux")]
    pub fn linux_zero_copy_metrics(
        &self,
    ) -> Option<crate::linux_video::LinuxZeroCopyMetricsSnapshot> {
        self.linux_zero_copy_metrics
            .lock()
            .as_ref()
            .map(|metrics| metrics.snapshot())
    }

    // =========================================================================
    // Audio control
    // =========================================================================

    /// Sets the muted state and syncs to the decode thread.
    pub fn set_muted(&mut self, muted: bool) {
        self.audio_handle.set_muted(muted);
        #[cfg(any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos"
        ))]
        if let Some(ref thread) = self.decode_thread {
            thread.set_muted(muted);
        }
    }

    /// Sets volume and syncs to the decode thread.
    pub fn set_volume(&mut self, volume: u32) {
        self.audio_handle.set_volume(volume);
        #[cfg(any(
            target_os = "android",
            target_os = "ios",
            target_os = "linux",
            target_os = "macos"
        ))]
        if let Some(ref decode_thread) = self.decode_thread {
            decode_thread.set_volume(volume as f32 / 100.0);
        }
    }

    /// Stops the decode thread and joins the init thread (called during cleanup).
    fn stop(&mut self) {
        if let Some(ref thread) = self.decode_thread {
            thread.stop();
        }
        #[cfg(target_os = "macos")]
        if let Some(ref audio_thread) = self.audio_thread {
            audio_thread.stop();
        }
        // Join init thread if still running to avoid leaking the handle
        if let Some(handle) = self.init_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CorePlayer {
    fn drop(&mut self) {
        self.stop();
    }
}
