//! Frame queue for video playback.
//!
//! This module provides a thread-safe ring buffer for decoded video frames,
//! enabling smooth playback by decoupling decoding from rendering.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use super::audio::AudioHandle;
#[cfg(target_os = "macos")]
use super::audio_decoder::AudioDecoder;
use super::sync_metrics::{StallType, SyncMetrics};
use super::video::{VideoDecoderBackend, VideoFrame};

/// Default number of frames to buffer ahead.
const DEFAULT_BUFFER_SIZE: usize = 5;

/// Commands sent to the decode thread.
#[derive(Debug, Clone)]
pub enum DecodeCommand {
    /// Start or resume decoding
    Play,
    /// Pause decoding
    Pause,
    /// Seek to a specific position
    Seek(Duration),
    /// Stop the decode thread
    Stop,
    /// Set muted state (Android only - audio controlled by ExoPlayer)
    SetMuted(bool),
    /// Set volume level (Android only - audio controlled by ExoPlayer)
    SetVolume(f32),
}

/// A thread-safe queue of decoded video frames.
///
/// The FrameQueue manages a ring buffer of decoded frames with a producer
/// (decode thread) that fills the buffer and a consumer (render thread)
/// that takes frames for display.
pub struct FrameQueue {
    /// The decoded frames ready for display
    frames: Arc<Mutex<VecDeque<VideoFrame>>>,
    /// Maximum number of frames to buffer
    capacity: usize,
    /// Condition variable for signaling when frames are available
    frame_available: Arc<Condvar>,
    /// Condition variable for signaling when space is available
    space_available: Arc<Condvar>,
    /// Flag indicating the queue is being flushed (for seeking)
    flushing: Arc<AtomicBool>,
    /// Flag indicating end of stream reached
    eos: Arc<AtomicBool>,
    /// Flag indicating the queue has been stopped (for shutdown)
    stopped: Arc<AtomicBool>,
}

impl FrameQueue {
    /// Creates a new frame queue with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
            frame_available: Arc::new(Condvar::new()),
            space_available: Arc::new(Condvar::new()),
            flushing: Arc::new(AtomicBool::new(false)),
            eos: Arc::new(AtomicBool::new(false)),
            stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a new frame queue with the default capacity.
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_BUFFER_SIZE)
    }

    /// Pushes a frame onto the queue.
    ///
    /// This will block if the queue is full, unless the queue is being flushed
    /// or stopped. Returns false if the queue is being flushed/stopped and the
    /// frame should be discarded.
    pub fn push(&self, frame: VideoFrame) -> bool {
        let mut frames = self.frames.lock();

        // Wait for space if queue is full
        while frames.len() >= self.capacity {
            // Check both flushing and stopped to avoid deadlock on shutdown
            if self.flushing.load(Ordering::Acquire) || self.stopped.load(Ordering::Acquire) {
                return false;
            }
            self.space_available.wait(&mut frames);
        }

        // Check again after waiting
        if self.flushing.load(Ordering::Acquire) || self.stopped.load(Ordering::Acquire) {
            return false;
        }

        frames.push_back(frame);
        self.frame_available.notify_one();
        true
    }

    /// Pushes a frame without blocking.
    ///
    /// Returns false if the queue is full, being flushed, or stopped.
    pub fn try_push(&self, frame: VideoFrame) -> bool {
        if self.flushing.load(Ordering::Acquire) || self.stopped.load(Ordering::Acquire) {
            return false;
        }

        let mut frames = self.frames.lock();
        if frames.len() >= self.capacity {
            return false;
        }

        frames.push_back(frame);
        self.frame_available.notify_one();
        true
    }

    /// Takes the next frame from the queue.
    ///
    /// Returns None if the queue is empty or end-of-stream has been reached.
    pub fn pop(&self) -> Option<VideoFrame> {
        let mut frames = self.frames.lock();
        tracing::trace!("pop(): queue len before pop = {}", frames.len());

        let frame = frames.pop_front();
        if frame.is_some() {
            self.space_available.notify_one();
        }
        frame
    }

    /// Takes the next frame, blocking until one is available.
    ///
    /// Returns None if end-of-stream is reached and the queue is empty.
    pub fn pop_blocking(&self, timeout: Duration) -> Option<VideoFrame> {
        let mut frames = self.frames.lock();

        // Wait for a frame if the queue is empty
        if frames.is_empty() {
            if self.eos.load(Ordering::Acquire) {
                return None;
            }

            let timeout_result = self.frame_available.wait_for(&mut frames, timeout);

            if timeout_result.timed_out() && frames.is_empty() {
                return None;
            }
        }

        let frame = frames.pop_front();
        if frame.is_some() {
            self.space_available.notify_one();
        }
        frame
    }

    /// Peeks at the next frame without removing it.
    pub fn peek(&self) -> Option<VideoFrame> {
        let frames = self.frames.lock();
        frames.front().cloned()
    }

    /// Returns the presentation timestamp of the next frame without removing it.
    pub fn peek_pts(&self) -> Option<Duration> {
        let frames = self.frames.lock();
        frames.front().map(|f| f.pts)
    }

    /// Returns the number of frames currently in the queue.
    pub fn len(&self) -> usize {
        self.frames.lock().len()
    }

    /// Returns true if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the queue is full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Clears all frames from the queue for seeking.
    ///
    /// This sets the flushing flag to prevent new frames from being added,
    /// clears the queue, then resets both eos and flushing flags.
    ///
    /// The ordering of clearing eos before flushing is intentional:
    /// 1. Set flushing=true - blocks producers from pushing new frames
    /// 2. Clear the queue
    /// 3. Clear eos=false - reset end-of-stream state for new content
    /// 4. Clear flushing=false - allow producers to push new frames
    ///
    /// This ordering ensures that when flushing=false is visible, eos=false
    /// is also visible (Release ordering guarantees this). Producers check
    /// flushing before pushing, so they won't push until step 4, by which
    /// time eos has already been cleared.
    pub fn flush(&self) {
        self.flushing.store(true, Ordering::Release);

        // Wake up any blocked producers
        self.space_available.notify_all();

        let dropped_count = {
            let mut frames = self.frames.lock();
            let count = frames.len();
            frames.clear();
            count
        };

        tracing::debug!("FrameQueue::flush: dropped {} frames", dropped_count);

        // Clear eos before flushing so consumers see consistent state
        self.eos.store(false, Ordering::Release);
        self.flushing.store(false, Ordering::Release);
    }

    /// Marks that end-of-stream has been reached.
    pub fn set_eos(&self) {
        self.eos.store(true, Ordering::Release);
        self.frame_available.notify_all();
    }

    /// Returns true if end-of-stream has been reached.
    pub fn is_eos(&self) -> bool {
        self.eos.load(Ordering::Acquire)
    }

    /// Resets the end-of-stream flag.
    pub fn clear_eos(&self) {
        self.eos.store(false, Ordering::Release);
    }

    /// Stops the queue, waking any blocked producers.
    ///
    /// This is called during shutdown to ensure the decode thread doesn't
    /// deadlock while waiting for space in push().
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        // Wake up any blocked producers so they can exit
        self.space_available.notify_all();
        // Also wake consumers in case they're waiting
        self.frame_available.notify_all();
    }

    /// Returns true if the queue has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }
}

impl Default for FrameQueue {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

/// A video decode thread that fills a frame queue.
///
/// This runs decoding on a separate thread to avoid blocking the render thread.
pub struct DecodeThread {
    /// Handle to the decode thread
    handle: Option<JoinHandle<()>>,
    /// Channel to send commands to the decode thread
    command_tx: crossbeam_channel::Sender<DecodeCommand>,
    /// The frame queue being filled
    frame_queue: Arc<FrameQueue>,
    /// Flag to signal the thread should stop
    stop_flag: Arc<AtomicBool>,
    /// Shared duration (updated by decode thread, read by UI thread)
    duration: Arc<Mutex<Option<Duration>>>,
    /// Shared dimensions (updated by decode thread, read by UI thread)
    dimensions: Arc<Mutex<Option<(u32, u32)>>>,
    /// Shared frame rate (updated by decode thread, read by UI thread)
    frame_rate: Arc<Mutex<Option<f32>>>,
    /// Shared buffering percentage (0-100, updated by decode thread)
    buffering_percent: Arc<std::sync::atomic::AtomicI32>,
}

impl DecodeThread {
    /// Creates and starts a new decode thread.
    ///
    /// The thread will start in a paused state.
    pub fn new<D: VideoDecoderBackend + Send + 'static>(
        decoder: D,
        frame_queue: Arc<FrameQueue>,
    ) -> Self {
        Self::with_audio_handle(decoder, frame_queue, None)
    }

    /// Creates and starts a new decode thread with optional audio handle.
    ///
    /// When an audio handle is provided, the decode thread will update
    /// the native position from the decoder's current_time() for A/V sync.
    /// This is used for native decoders (macOS AVPlayer, Android ExoPlayer)
    /// that have integrated playback control.
    pub fn with_audio_handle<D: VideoDecoderBackend + Send + 'static>(
        decoder: D,
        frame_queue: Arc<FrameQueue>,
        audio_handle: Option<AudioHandle>,
    ) -> Self {
        use std::sync::atomic::AtomicI32;

        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let duration = Arc::new(Mutex::new(None));
        let dimensions = Arc::new(Mutex::new(None));
        let frame_rate = Arc::new(Mutex::new(None));
        let buffering_percent = Arc::new(AtomicI32::new(0)); // Start unbuffered, decoder will update

        let queue = Arc::clone(&frame_queue);
        let stop = Arc::clone(&stop_flag);
        let dur = Arc::clone(&duration);
        let dims = Arc::clone(&dimensions);
        let fps = Arc::clone(&frame_rate);
        let buf = Arc::clone(&buffering_percent);
        let audio = audio_handle.clone();

        let handle = thread::spawn(move || {
            decode_loop(decoder, queue, command_rx, stop, dur, dims, fps, buf, audio);
        });

        Self {
            handle: Some(handle),
            command_tx,
            frame_queue,
            stop_flag,
            duration,
            dimensions,
            frame_rate,
            buffering_percent,
        }
    }

    /// Starts or resumes decoding.
    pub fn play(&self) {
        let _ = self.command_tx.send(DecodeCommand::Play);
    }

    /// Pauses decoding.
    pub fn pause(&self) {
        let _ = self.command_tx.send(DecodeCommand::Pause);
    }

    /// Seeks to a specific position.
    ///
    /// This will flush the frame queue and start decoding from the new position.
    pub fn seek(&self, position: Duration) {
        self.frame_queue.flush();
        // Immediately show buffering indicator - HTTP streams need to rebuffer after seek
        self.buffering_percent.store(0, Ordering::Relaxed);
        let _ = self.command_tx.send(DecodeCommand::Seek(position));
    }

    /// Stops the decode thread.
    ///
    /// This stops the frame queue first to wake any blocked push() calls,
    /// preventing deadlock during shutdown.
    pub fn stop(&self) {
        // Stop the frame queue first to wake any blocked producers
        self.frame_queue.stop();
        self.stop_flag.store(true, Ordering::Release);
        let _ = self.command_tx.send(DecodeCommand::Stop);
    }

    /// Sets the muted state (Android only - audio is controlled by ExoPlayer).
    pub fn set_muted(&self, muted: bool) {
        let _ = self.command_tx.send(DecodeCommand::SetMuted(muted));
    }

    /// Sets the volume level (Android only - audio is controlled by ExoPlayer).
    pub fn set_volume(&self, volume: f32) {
        let _ = self.command_tx.send(DecodeCommand::SetVolume(volume));
    }

    /// Returns a reference to the frame queue.
    pub fn frame_queue(&self) -> &Arc<FrameQueue> {
        &self.frame_queue
    }

    /// Returns the current known duration (updated by decode thread).
    pub fn duration(&self) -> Option<Duration> {
        *self.duration.lock()
    }

    /// Returns the current known dimensions (updated by decode thread).
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        *self.dimensions.lock()
    }

    /// Returns the current known frame rate (updated by decode thread).
    pub fn frame_rate(&self) -> Option<f32> {
        *self.frame_rate.lock()
    }

    /// Returns the current buffering percentage (0-100).
    pub fn buffering_percent(&self) -> i32 {
        self.buffering_percent.load(Ordering::Relaxed)
    }
}

impl Drop for DecodeThread {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Result of processing a decode command.
#[allow(dead_code)] // Seeking variant only used with ffmpeg feature
enum CommandResult {
    /// Continue processing, optionally updating playing state
    Continue(Option<bool>),
    /// Stop the decode loop
    Stop,
    /// Seek in progress - keep playing but tolerate empty decodes
    #[allow(dead_code)]
    Seeking,
}

/// Processes a single decode command. Returns the result to apply.
fn process_decode_command<D: VideoDecoderBackend>(
    cmd: DecodeCommand,
    decoder: &mut D,
    frame_queue: &FrameQueue,
) -> CommandResult {
    match cmd {
        DecodeCommand::Stop => return CommandResult::Stop,
        DecodeCommand::Play => {
            frame_queue.clear_eos();
            if let Err(e) = decoder.resume() {
                tracing::error!("Failed to resume decoder: {}", e);
            }
            return CommandResult::Continue(Some(true));
        }
        DecodeCommand::Pause => {
            if let Err(e) = decoder.pause() {
                tracing::error!("Failed to pause decoder: {}", e);
            }
            return CommandResult::Continue(Some(false));
        }
        DecodeCommand::Seek(position) => {
            frame_queue.flush();
            if let Err(e) = decoder.seek(position) {
                tracing::error!("Seek failed: {}", e);
                // Don't clear EOS if seek failed — prevents infinite loop
                // when loop_playback tries to seek on live streams
            } else {
                frame_queue.clear_eos();
            }
        }
        DecodeCommand::SetMuted(muted) => {
            if let Err(e) = decoder.set_muted(muted) {
                tracing::error!("Failed to set muted: {}", e);
            }
        }
        DecodeCommand::SetVolume(volume) => {
            if let Err(e) = decoder.set_volume(volume) {
                tracing::error!("Failed to set volume: {}", e);
            }
        }
    }
    CommandResult::Continue(None)
}

/// The main decode loop running on the decode thread.
#[allow(clippy::too_many_arguments)]
fn decode_loop<D: VideoDecoderBackend>(
    mut decoder: D,
    frame_queue: Arc<FrameQueue>,
    command_rx: crossbeam_channel::Receiver<DecodeCommand>,
    stop_flag: Arc<AtomicBool>,
    shared_duration: Arc<Mutex<Option<Duration>>>,
    shared_dimensions: Arc<Mutex<Option<(u32, u32)>>>,
    shared_frame_rate: Arc<Mutex<Option<f32>>>,
    shared_buffering: Arc<std::sync::atomic::AtomicI32>,
    audio_handle: Option<AudioHandle>,
) {
    let mut playing = false;
    let mut last_metadata_check = std::time::Instant::now();
    let mut last_position_update = std::time::Instant::now();

    // Decode one frame immediately for preview (before waiting for Play command)
    // This allows showing the first frame without starting playback
    // Try multiple times since streaming decoders (HTTP, ExoPlayer) need time to buffer
    let mut preview_attempts = 0;
    let max_preview_attempts = 30; // Try for up to ~3 seconds for slow HTTP streams
    let mut preview_dims: Option<(u32, u32)> = None;

    loop {
        // Check for early termination (user closed video)
        if stop_flag.load(Ordering::Acquire) {
            tracing::debug!("Preview loop interrupted by stop signal");
            return;
        }

        match decoder.decode_next() {
            Ok(Some(frame)) => {
                // Check if this is a real frame (not a 1x1 placeholder)
                let (w, h) = frame.dimensions();
                if w > 1 && h > 1 {
                    tracing::info!("Decoded preview frame at {:?} ({}x{})", frame.pts, w, h);
                    preview_dims = Some((w, h));
                    let _ = frame_queue.try_push(frame);
                    break;
                } else {
                    // Placeholder frame, keep trying
                    preview_attempts += 1;
                    if preview_attempts >= max_preview_attempts {
                        tracing::debug!("Max preview attempts reached, using placeholder");
                        let _ = frame_queue.try_push(frame);
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
            Ok(None) => {
                // For HTTP streams, None often means "still buffering" not "EOS"
                preview_attempts += 1;
                if preview_attempts >= max_preview_attempts {
                    tracing::debug!(
                        "No preview frame available after {} attempts",
                        preview_attempts
                    );
                    break;
                }
                // Wait a bit before retrying
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                tracing::warn!("Failed to decode preview frame: {}", e);
                break;
            }
        }
    }

    // Wait for metadata to become available (ExoPlayer needs time to determine duration/dimensions)
    // This is important because pausing too early may prevent ExoPlayer from reporting metadata
    //
    // For live streams (no duration), skip this wait entirely if we already have dimensions
    // from the preview frame. Live stream decoders (MoQ) never report duration, and their
    // cached_metadata only syncs in decode_next() — which isn't called during this loop.
    // Without this bypass, MoQ streams always stall for the full 3-second timeout.
    let is_live = decoder.duration().is_none();
    if is_live && preview_dims.is_some() {
        tracing::info!(
            "Live stream: using preview dimensions {:?}, skipping metadata wait",
            preview_dims
        );
        *shared_dimensions.lock() = preview_dims;
        let fps = decoder.metadata().frame_rate;
        if fps > 0.0 {
            *shared_frame_rate.lock() = Some(fps);
        }
    } else {
        let metadata_wait_start = std::time::Instant::now();
        let metadata_timeout = Duration::from_secs(3);

        loop {
            // Check for early termination (user closed video)
            if stop_flag.load(Ordering::Acquire) {
                tracing::debug!("Metadata loop interrupted by stop signal");
                return;
            }

            let duration_opt = decoder.duration();
            let has_duration = duration_opt.is_some();
            let dims = decoder.dimensions();
            let has_dimensions = dims.0 > 1 && dims.1 > 1; // >1 to exclude placeholder

            if has_duration && has_dimensions {
                *shared_duration.lock() = duration_opt;
                *shared_dimensions.lock() = Some(dims);
                let fps = decoder.metadata().frame_rate;
                if fps > 0.0 {
                    *shared_frame_rate.lock() = Some(fps);
                }
                break;
            }

            if metadata_wait_start.elapsed() > metadata_timeout {
                tracing::warn!("Timeout waiting for video metadata");
                // Store whatever we have
                if let Some(dur) = duration_opt {
                    *shared_duration.lock() = Some(dur);
                }
                if dims.0 > 0 && dims.1 > 0 {
                    *shared_dimensions.lock() = Some(dims);
                }
                let fps = decoder.metadata().frame_rate;
                if fps > 0.0 {
                    *shared_frame_rate.lock() = Some(fps);
                }
                break;
            }

            thread::sleep(Duration::from_millis(100));
        }
    }

    // Pause the decoder after getting preview frame (for decoders like ExoPlayer that auto-play)
    if let Err(e) = decoder.pause() {
        tracing::debug!("Failed to pause after preview: {}", e);
    }

    // Note: We no longer count consecutive Nones for EOS detection.
    // Instead, we rely on decoder.is_eof() which checks actual decoder state.

    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands (non-blocking)
        while let Ok(cmd) = command_rx.try_recv() {
            match process_decode_command(cmd, &mut decoder, &frame_queue) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) | CommandResult::Seeking => {}
            }
        }

        // Update buffering percentage immediately (important for UI feedback)
        shared_buffering.store(decoder.buffering_percent(), Ordering::Relaxed);

        // Periodically update shared metadata (every 500ms)
        if last_metadata_check.elapsed() > Duration::from_millis(500) {
            if let Some(dur) = decoder.duration() {
                *shared_duration.lock() = Some(dur);
            }
            let dims = decoder.dimensions();
            if dims.0 > 0 && dims.1 > 0 {
                *shared_dimensions.lock() = Some(dims);
            }
            let fps = decoder.metadata().frame_rate;
            if fps > 0.0 {
                *shared_frame_rate.lock() = Some(fps);
            }
            last_metadata_check = std::time::Instant::now();
        }

        // Update native position from decoder's current_time() for A/V sync (every 16ms ~ 60fps)
        // This is used for native decoders (macOS AVPlayer, Android ExoPlayer) that
        // have integrated playback and can report their actual playback position.
        if let Some(ref audio) = audio_handle {
            if playing && last_position_update.elapsed() > Duration::from_millis(16) {
                if let Some(pos) = decoder.current_time() {
                    audio.set_native_position(pos);
                }
                last_position_update = std::time::Instant::now();
            }
        }

        // When paused, wait for commands
        if !playing {
            tracing::info!("decode_loop: PAUSED branch (playing=false), waiting on recv_timeout");
            let cmd = match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(cmd) => cmd,
                Err(_) => continue,
            };
            match process_decode_command(cmd, &mut decoder, &frame_queue) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) | CommandResult::Seeking => {}
            }
            continue;
        }

        // Don't decode if queue is full
        if frame_queue.is_full() {
            tracing::debug!(
                "decode_loop: QUEUE FULL branch, sleeping 5ms (queue len={})",
                frame_queue.len()
            );
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        // Decode the next frame
        let frame = match decoder.decode_next() {
            Ok(Some(frame)) => frame,
            Ok(None) if decoder.is_eof() => {
                frame_queue.set_eos();
                playing = false;
                tracing::debug!("End of stream confirmed by decoder");
                continue;
            }
            Ok(None) => {
                tracing::trace!("decode_next returned None (buffering)");
                continue;
            }
            Err(e) => {
                tracing::error!("Decode error: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        tracing::trace!("Decoded frame at {:?}", frame.pts);
        if !frame_queue.push(frame) {
            tracing::debug!("Frame rejected by queue (flushing)");
        }
    }
}

// ============================================================================
// Audio decoding thread
// ============================================================================

/// An audio decode thread that decodes audio and sends samples to a channel.
/// The actual audio playback happens on this thread to avoid Send/Sync issues.
#[cfg(target_os = "macos")]
pub struct AudioThread {
    /// Handle to the audio thread
    handle: Option<JoinHandle<()>>,
    /// Channel to send commands to the audio thread
    command_tx: crossbeam_channel::Sender<DecodeCommand>,
    /// Flag to signal the thread should stop
    stop_flag: Arc<AtomicBool>,
    /// Audio handle for volume/mute control (shared with UI)
    audio_handle: super::audio::AudioHandle,
}

#[cfg(target_os = "macos")]
impl AudioThread {
    /// Creates and starts a new audio decode thread.
    ///
    /// # Arguments
    /// * `url` - The video URL or file path
    /// * `video_start_time` - The video stream's start time (for PTS offset calculation)
    pub fn new(url: &str, video_start_time: Option<Duration>) -> Option<Self> {
        let (command_tx, command_rx) = crossbeam_channel::unbounded();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let audio_handle = super::audio::AudioHandle::new();
        audio_handle.set_available(true);

        // Set video start time BEFORE thread spawn to avoid race with finalize_stream_pts_offset
        audio_handle.set_video_start_time(video_start_time);

        let stop = Arc::clone(&stop_flag);
        let handle_clone = audio_handle.clone();
        let url_owned = url.to_string();

        let handle = thread::spawn(move || {
            audio_thread_main(url_owned, handle_clone, command_rx, stop);
        });

        Some(Self {
            handle: Some(handle),
            command_tx,
            stop_flag,
            audio_handle,
        })
    }

    /// Returns the audio handle for UI control.
    pub fn handle(&self) -> super::audio::AudioHandle {
        self.audio_handle.clone()
    }

    /// Starts or resumes audio playback.
    pub fn play(&self) {
        let _ = self.command_tx.send(DecodeCommand::Play);
    }

    /// Pauses audio playback.
    pub fn pause(&self) {
        let _ = self.command_tx.send(DecodeCommand::Pause);
    }

    /// Seeks to a specific position.
    pub fn seek(&self, position: Duration) {
        let _ = self.command_tx.send(DecodeCommand::Seek(position));
    }

    /// Stops the audio thread.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
        let _ = self.command_tx.send(DecodeCommand::Stop);
    }
}

#[cfg(target_os = "macos")]
impl Drop for AudioThread {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Processes a single audio command. Returns the result to apply.
#[cfg(target_os = "macos")]
fn process_audio_command(
    cmd: DecodeCommand,
    player: &mut super::audio::AudioPlayer,
    decoder: &mut AudioDecoder,
) -> CommandResult {
    match cmd {
        DecodeCommand::Stop => CommandResult::Stop,
        DecodeCommand::Play => {
            player.play();
            CommandResult::Continue(Some(true))
        }
        DecodeCommand::Pause => {
            player.pause();
            CommandResult::Continue(Some(false))
        }
        DecodeCommand::Seek(position) => {
            player.clear();
            player.play(); // Re-arm sink after clear (rodio may auto-pause on empty queue)
            if let Err(e) = decoder.seek(position) {
                tracing::error!("Audio seek failed: {}", e);
            }
            // Return special marker to indicate seeking state
            CommandResult::Seeking
        }
        // SetMuted and SetVolume are handled by the video decoder thread
        DecodeCommand::SetMuted(_) | DecodeCommand::SetVolume(_) => CommandResult::Continue(None),
    }
}

/// The main audio thread function - creates player and runs decode loop.
#[cfg(target_os = "macos")]
fn audio_thread_main(
    url: String,
    handle: super::audio::AudioHandle,
    command_rx: crossbeam_channel::Receiver<DecodeCommand>,
    stop_flag: Arc<AtomicBool>,
) {
    use super::audio::{AudioConfig, AudioPlayer};

    // Create audio player on this thread (OutputStream is not Send)
    let mut player =
        match AudioPlayer::new_with_handle(AudioConfig::default(), Some(handle.clone())) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to create audio player: {}", e);
                handle.set_available(false);
                return;
            }
        };

    // Get device sample rate and create decoder with it
    let device_sample_rate = player.device_sample_rate();
    let mut decoder = match AudioDecoder::new(&url, device_sample_rate) {
        Ok(d) => d,
        Err(e) => {
            tracing::error!("Failed to create audio decoder: {}", e);
            handle.set_available(false);
            return;
        }
    };

    // Finalize stream PTS offset now that we have audio metadata
    // (video start time was set by video_player before audio thread started)
    handle.finalize_stream_pts_offset(decoder.metadata().start_time);

    let mut playing = false;
    let mut seeking = false; // True while seeking - tolerate empty decodes
    let mut empty_decode_count = 0;
    const MAX_EMPTY_DECODES_SEEKING: u32 = 100; // Allow ~1s of empty decodes during seek
    const MAX_EMPTY_DECODES_NORMAL: u32 = 10; // Only ~100ms for normal EOF detection

    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands (non-blocking)
        while let Ok(cmd) = command_rx.try_recv() {
            match process_audio_command(cmd, &mut player, &mut decoder) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) => {}
                CommandResult::Seeking => {
                    // Seek keeps playing but enters seeking state
                    seeking = true;
                    empty_decode_count = 0;
                    tracing::debug!("Audio: entering seeking state");
                }
            }
        }

        // When paused, wait for commands
        if !playing {
            let cmd = match command_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(cmd) => cmd,
                Err(_) => continue,
            };
            match process_audio_command(cmd, &mut player, &mut decoder) {
                CommandResult::Stop => return,
                CommandResult::Continue(Some(new_playing)) => playing = new_playing,
                CommandResult::Continue(None) => {}
                CommandResult::Seeking => {
                    seeking = true;
                    empty_decode_count = 0;
                    tracing::debug!("Audio: entering seeking state (from paused)");
                }
            }
            continue;
        }

        // Decode the next audio samples
        let samples = match decoder.decode_next() {
            Ok(Some(samples)) => {
                if seeking {
                    tracing::debug!("Audio: got frame after seek, exiting seeking state");
                    seeking = false;
                }
                empty_decode_count = 0;
                samples
            }
            Ok(None) => {
                empty_decode_count += 1;
                let max_empty = if seeking {
                    MAX_EMPTY_DECODES_SEEKING
                } else {
                    MAX_EMPTY_DECODES_NORMAL
                };

                if empty_decode_count >= max_empty {
                    if seeking {
                        tracing::warn!(
                            "Audio: still no frames after {} empty decodes during seek",
                            empty_decode_count
                        );
                        // Don't stop - keep trying during seek
                    } else {
                        // True EOF - stop playing
                        tracing::debug!(
                            "Audio: EOF reached after {} empty decodes",
                            empty_decode_count
                        );
                        playing = false;
                    }
                }
                thread::sleep(Duration::from_millis(10));
                continue;
            }
            Err(e) => {
                tracing::error!("Audio decode error: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        };

        player.queue_samples(samples);
        thread::sleep(Duration::from_millis(5));
    }
}

/// A simple frame scheduler that determines which frame to display.
///
/// This handles frame timing based on presentation timestamps.
/// Number of smooth frames required to end recovery mode.
const RECOVERY_FRAME_THRESHOLD: u32 = 30;

/// Wall-clock duration before forcing a resync when stuck.
/// This handles videos with sparse keyframes where gaps > position clamp can occur.
const STUCK_TIMEOUT: Duration = Duration::from_secs(3);
/// Near-boundary reject windows should recover faster than generic stuck handling.
const NEAR_BOUNDARY_FORCE_TIMEOUT: Duration = Duration::from_millis(900);
/// Startup tolerance while waiting for audio to begin.
const AUDIO_STARTUP_AHEAD_TOLERANCE: Duration = Duration::from_millis(500);
/// Base live tolerance once audio is running.
const LIVE_BASE_AHEAD_TOLERANCE: Duration = Duration::from_millis(2000);
/// Maximum adaptive live tolerance to avoid unbounded A/V divergence.
const LIVE_MAX_AHEAD_TOLERANCE: Duration = Duration::from_millis(4200);
/// Margin added when adapting to near-boundary gaps.
const LIVE_AHEAD_MARGIN: Duration = Duration::from_millis(180);
/// Only treat this much extra gap beyond base as near-boundary.
const LIVE_NEAR_BOUNDARY_RANGE: Duration = Duration::from_millis(1700);
/// Reject windows starting this frequently indicate a scheduler thrash burst.
const REJECT_BURST_WINDOW: Duration = Duration::from_millis(140);
/// Number of rapid reject-window starts before enabling burst tolerance.
const REJECT_BURST_THRESHOLD: u32 = 8;
/// Duration to keep elevated tolerance during a detected burst.
const REJECT_BURST_TOLERANCE_HOLD: Duration = Duration::from_secs(2);
/// How long to wait for audio clock movement before treating timing as "started".
const AUDIO_CLOCK_START_GRACE: Duration = Duration::from_secs(2);
/// Minimum interval between repeated audio-clock-zero diagnostics.
const AUDIO_ZERO_DIAG_COOLDOWN: Duration = Duration::from_secs(2);
/// Larger gaps are treated as stale and dropped.
const STALE_GAP_THRESHOLD: Duration = Duration::from_secs(10);
/// If repeated forced-resync attempts fail in one reject window, drop one head frame.
const MAX_FORCED_RESYNCS_PER_WINDOW: u32 = 2;

/// The scheduler only advances position when frames are actually being delivered,
/// preventing the scroll bar from advancing during buffering.
pub struct FrameScheduler {
    /// The current playback position (updated from frame PTS)
    current_position: Duration,
    /// The last frame that was displayed
    current_frame: Option<VideoFrame>,
    /// Time when playback started (or was resumed) - only set after first frame arrives
    playback_start_time: Option<std::time::Instant>,
    /// Position when playback started (synced to frame PTS)
    playback_start_position: Duration,
    /// True if we're waiting for the first frame after play/seek
    waiting_for_first_frame: bool,
    /// True if playback has been requested (even if waiting for first frame)
    playback_requested: bool,
    /// True if we're stalled (queue empty during playback)
    stalled: bool,
    /// A/V sync metrics tracker
    sync_metrics: SyncMetrics,
    /// Audio handle for getting playback position
    audio_handle: Option<AudioHandle>,
    /// Number of frames displayed since recovery started (for ending recovery)
    frames_since_recovery: u32,
    /// Wall-clock time when rejection streak started (for stuck detection)
    rejection_start_time: Option<std::time::Instant>,
    /// Monotonic seek generation to identify stale frames
    seek_generation: u64,
    /// Time when audio first started (position > 0) - for accurate elapsed measurement
    audio_start_time: Option<std::time::Instant>,
    /// Audio position when audio_start_time was recorded (for seek-aware drift calculation)
    audio_start_pos: Duration,
    /// Number of consecutive rejections in the current reject window.
    rejection_count: u32,
    /// Peak observed A/V gap in the current reject window.
    rejection_peak_gap: Duration,
    /// Number of force-resync attempts in the current reject window.
    forced_resyncs_in_window: u32,
    /// Last wall-clock time when a reject window started.
    last_reject_window_start: Option<std::time::Instant>,
    /// Number of rapid reject-window starts in the current burst.
    reject_burst_count: u32,
    /// Elevated tolerance window to break repeated reject thrash.
    burst_tolerance_until: Option<std::time::Instant>,
    /// Last time we logged detailed "audio clock still zero" diagnostics.
    last_audio_zero_diag: Option<std::time::Instant>,
}

impl FrameScheduler {
    /// Creates a new frame scheduler.
    pub fn new() -> Self {
        Self {
            current_position: Duration::ZERO,
            current_frame: None,
            playback_start_time: None,
            playback_start_position: Duration::ZERO,
            waiting_for_first_frame: false,
            playback_requested: false,
            stalled: false,
            sync_metrics: SyncMetrics::new(),
            audio_handle: None,
            frames_since_recovery: 0,
            rejection_start_time: None,
            seek_generation: 0,
            audio_start_time: None,
            audio_start_pos: Duration::ZERO,
            rejection_count: 0,
            rejection_peak_gap: Duration::ZERO,
            forced_resyncs_in_window: 0,
            last_reject_window_start: None,
            reject_burst_count: 0,
            burst_tolerance_until: None,
            last_audio_zero_diag: None,
        }
    }

    /// Creates a new frame scheduler with audio handle for sync tracking.
    pub fn with_audio_handle(audio_handle: AudioHandle) -> Self {
        let sync_metrics = SyncMetrics::new();
        // Audio handle presence implies using audio clock for sync metrics
        sync_metrics.set_using_audio_clock(true);

        Self {
            current_position: Duration::ZERO,
            current_frame: None,
            playback_start_time: None,
            playback_start_position: Duration::ZERO,
            waiting_for_first_frame: false,
            playback_requested: false,
            stalled: false,
            sync_metrics,
            audio_handle: Some(audio_handle),
            frames_since_recovery: 0,
            rejection_start_time: None,
            seek_generation: 0,
            audio_start_time: None,
            audio_start_pos: Duration::ZERO,
            rejection_count: 0,
            rejection_peak_gap: Duration::ZERO,
            forced_resyncs_in_window: 0,
            last_reject_window_start: None,
            reject_burst_count: 0,
            burst_tolerance_until: None,
            last_audio_zero_diag: None,
        }
    }

    /// Sets the audio handle for sync tracking.
    pub fn set_audio_handle(&mut self, audio_handle: AudioHandle) {
        self.audio_handle = Some(audio_handle);
        self.last_audio_zero_diag = None;
        self.sync_metrics.set_using_audio_clock(true);
    }

    /// Returns the sync metrics tracker.
    pub fn sync_metrics(&self) -> &SyncMetrics {
        &self.sync_metrics
    }

    /// Returns the current audio position normalized for sync comparison.
    ///
    /// Uses `position_for_sync()` which adjusts for any PTS offset between
    /// audio and video streams, making it directly comparable to video PTS.
    fn audio_position(&self) -> Duration {
        self.audio_handle
            .as_ref()
            .map(|h| h.position_for_sync())
            .unwrap_or(Duration::ZERO)
    }

    /// Returns the position to use for frame selection (A/V sync).
    ///
    /// # Audio as Master Clock
    ///
    /// When audio is available and playing (position > 0), we use audio position
    /// as the master clock. This ensures video stays synchronized with actual
    /// audio playback, accounting for audio buffer latency.
    ///
    /// Falls back to wall-clock time when:
    /// - No audio handle is set
    /// - Audio is not available
    /// - Audio hasn't started yet (position == 0)
    fn sync_position(&self) -> Duration {
        // Get wall-clock position as baseline
        let wall_clock_pos = self.position();

        // Try to use audio as master clock
        if let Some(ref audio) = self.audio_handle {
            if audio.is_available() {
                let audio_pos = audio.position_for_sync();
                // Only use audio position if audio has actually started
                if audio_pos > Duration::ZERO {
                    // Once audio has started, always use it as master clock.
                    // Don't fall back to wall-clock even if audio is behind -
                    // video should slow down to match audio, not race ahead.
                    return audio_pos;
                } else {
                    // Audio available but not started yet (position == 0).
                    // Return playback_start_position to hold video at first frame
                    // while waiting for audio to buffer and start.
                    // This prevents video from racing ahead of audio.
                    let elapsed = self
                        .playback_start_time
                        .map(|t| t.elapsed())
                        .unwrap_or(Duration::ZERO);
                    if elapsed < Duration::from_millis(500) {
                        // During initial startup, hold at start position
                        return self.playback_start_position;
                    }
                    // If we've been waiting too long (>500ms), something is wrong
                    // with audio - fall back to wall-clock to avoid stall
                    tracing::warn!(
                        "sync_position: audio not started after {:?}ms, falling back to wall-clock",
                        elapsed.as_millis()
                    );
                }
            }
        }

        // Fallback to wall-clock position
        wall_clock_pos
    }

    /// Starts or resumes playback.
    /// Note: The clock doesn't actually start until the first frame arrives.
    pub fn start(&mut self) {
        self.playback_requested = true;
        self.waiting_for_first_frame = true;
        self.stalled = false;
        self.reset_rejection_tracking();
        // Don't set playback_start_time yet - wait for first frame
    }

    /// Pauses playback.
    pub fn pause(&mut self) {
        self.playback_requested = false;
        self.waiting_for_first_frame = false;
        self.stalled = false;
        self.reset_rejection_tracking();
        self.audio_start_time = None;
        self.audio_start_pos = Duration::ZERO;
        if let Some(start) = self.playback_start_time.take() {
            // Calculate wall-clock position
            let wall_clock_pos = self.playback_start_position + start.elapsed();

            // Use frame PTS if wall-clock has run away (e.g., after loops)
            if let Some(ref frame) = self.current_frame {
                let max_pos = frame.pts + Duration::from_secs(1);
                self.current_position = if wall_clock_pos > max_pos {
                    frame.pts
                } else {
                    wall_clock_pos
                };
            } else {
                self.current_position = wall_clock_pos;
            }
        }
    }

    /// Seeks to a new position.
    pub fn seek(&mut self, position: Duration) {
        tracing::debug!(
            "FrameScheduler::seek: position={:?}, playback_requested={}, prev_current_frame={:?}",
            position,
            self.playback_requested,
            self.current_frame.as_ref().map(|f| f.pts)
        );

        self.current_position = position;
        self.current_frame = None;
        self.stalled = false;
        self.reset_rejection_tracking();
        self.seek_generation = self.seek_generation.wrapping_add(1);

        // Reset sync metrics on seek to clear max_drift from transient spikes
        self.sync_metrics.reset();
        // Set grace period to filter drift spikes during seek warmup
        self.sync_metrics.set_grace_period(30);

        if self.playback_requested {
            // Wait for first frame at new position before resuming clock
            self.waiting_for_first_frame = true;
            self.playback_start_time = None;
            self.audio_start_time = None;
            self.audio_start_pos = Duration::ZERO;

            // Clear audio epoch and native position so they get re-synchronized
            if let Some(ref audio) = self.audio_handle {
                audio.clear_playback_epoch();
                audio.clear_native_position();
            }
        }
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        // If stalled (queue empty during playback), return the last known position
        // to prevent the scroll bar from advancing during buffering
        if self.stalled {
            return self.current_position;
        }

        // Use wall-clock for smooth updates, but don't exceed the last frame's PTS
        // This prevents position from running ahead of actual video playback
        let wall_clock_pos = match self.playback_start_time {
            Some(start) => self.playback_start_position + start.elapsed(),
            None => return self.current_position,
        };

        // Clamp to current frame PTS to prevent runaway position
        if let Some(ref frame) = self.current_frame {
            // Allow wall-clock to be ahead for smooth scrubbing.
            // Use 3 seconds to handle keyframe gaps after seeking - keyframe-based
            // seeking can create gaps > 1s between the first decoded frame and
            // the next available frame in the stream.
            let max_pos = frame.pts + Duration::from_secs(3);
            if wall_clock_pos > max_pos {
                return frame.pts;
            }
        }

        wall_clock_pos
    }

    /// Returns true if playback is active (clock is running).
    pub fn is_playing(&self) -> bool {
        self.playback_start_time.is_some()
    }

    /// Returns true if playback has been requested (even if buffering).
    pub fn is_playback_requested(&self) -> bool {
        self.playback_requested
    }

    /// Called when a frame is received to sync the clock.
    /// If we were waiting for the first frame, this starts the clock.
    fn on_frame_received(&mut self, frame_pts: Duration) {
        if self.waiting_for_first_frame && self.playback_requested {
            // First frame after play/seek - start the clock synced to frame PTS
            self.playback_start_time = Some(std::time::Instant::now());
            self.playback_start_position = frame_pts;
            self.waiting_for_first_frame = false;

            // Also start the shared audio playback epoch so A/V clocks are synchronized
            if let Some(ref audio) = self.audio_handle {
                audio.start_playback_epoch();
                tracing::debug!(
                    "Clock started at frame PTS {:?}, audio epoch synchronized",
                    frame_pts
                );
            } else {
                tracing::debug!("Clock started at frame PTS {:?}", frame_pts);
            }
        }
    }

    fn audio_started_for_timing(&self) -> bool {
        self.audio_handle
            .as_ref()
            .map(|h| {
                if !h.is_available() {
                    return true;
                }
                if h.position_for_sync() > Duration::ZERO {
                    return true;
                }
                self.playback_start_time
                    .map(|t| t.elapsed() >= AUDIO_CLOCK_START_GRACE)
                    .unwrap_or(false)
            })
            .unwrap_or(true)
    }

    fn maybe_log_audio_zero_diag(
        &mut self,
        now: std::time::Instant,
        current_pos: Duration,
        next_pts: Duration,
        gap: Duration,
        ahead_tolerance: Duration,
    ) {
        let Some(audio) = self.audio_handle.clone() else {
            return;
        };
        let sync_pos = audio.position_for_sync();
        if sync_pos > Duration::ZERO {
            return;
        }
        if self
            .last_audio_zero_diag
            .map(|last| now.duration_since(last) < AUDIO_ZERO_DIAG_COOLDOWN)
            .unwrap_or(false)
        {
            return;
        }
        self.last_audio_zero_diag = Some(now);
        let raw_pos = audio.position();
        let samples_played = audio.samples_played();
        tracing::warn!(
            "get_next_frame: audio clock still zero during reject window (current_pos={:?}, next_pts={:?}, gap={}ms, tolerance={}ms, audio_available={}, epoch_set={}, raw_audio_pos={:?}, sync_audio_pos={:?}, samples_played={}, samples_ms={}, offset_us={})",
            current_pos,
            next_pts,
            gap.as_millis(),
            ahead_tolerance.as_millis(),
            audio.is_available(),
            audio.playback_epoch().is_some(),
            raw_pos,
            sync_pos,
            samples_played,
            audio.samples_played_duration().as_millis(),
            audio.stream_pts_offset_us()
        );
    }

    fn compute_ahead_tolerance(
        &self,
        audio_started: bool,
        gap: Duration,
        now: std::time::Instant,
    ) -> Duration {
        if !audio_started {
            return AUDIO_STARTUP_AHEAD_TOLERANCE;
        }

        let mut tolerance = LIVE_BASE_AHEAD_TOLERANCE;

        if self
            .burst_tolerance_until
            .map(|until| now < until)
            .unwrap_or(false)
        {
            return LIVE_MAX_AHEAD_TOLERANCE;
        }

        let near_boundary = gap > LIVE_BASE_AHEAD_TOLERANCE
            && gap <= LIVE_BASE_AHEAD_TOLERANCE + LIVE_NEAR_BOUNDARY_RANGE;
        if near_boundary {
            tolerance = tolerance.max(gap.saturating_add(LIVE_AHEAD_MARGIN));
        }

        if let Some(start) = self.rejection_start_time {
            let stuck_duration = now.duration_since(start);
            let steps = stuck_duration.as_millis() / 500;
            let escalated_ms = LIVE_BASE_AHEAD_TOLERANCE
                .as_millis()
                .saturating_add(steps.saturating_mul(LIVE_AHEAD_MARGIN.as_millis()))
                .min(LIVE_MAX_AHEAD_TOLERANCE.as_millis());
            tolerance = tolerance.max(Duration::from_millis(escalated_ms as u64));
        }

        tolerance.min(LIVE_MAX_AHEAD_TOLERANCE)
    }

    fn reset_rejection_tracking(&mut self) {
        self.rejection_start_time = None;
        self.rejection_count = 0;
        self.rejection_peak_gap = Duration::ZERO;
        self.forced_resyncs_in_window = 0;
    }

    fn record_reject_window_start(&mut self, now: std::time::Instant) {
        if let Some(last_start) = self.last_reject_window_start {
            if now.duration_since(last_start) <= REJECT_BURST_WINDOW {
                self.reject_burst_count = self.reject_burst_count.saturating_add(1);
            } else {
                self.reject_burst_count = 1;
            }
        } else {
            self.reject_burst_count = 1;
        }
        self.last_reject_window_start = Some(now);

        if self.reject_burst_count >= REJECT_BURST_THRESHOLD {
            let hold_until = now + REJECT_BURST_TOLERANCE_HOLD;
            let already_active = self
                .burst_tolerance_until
                .map(|until| now < until)
                .unwrap_or(false);
            self.burst_tolerance_until = Some(hold_until);
            if !already_active {
                tracing::warn!(
                    "get_next_frame: reject burst detected (count={}), enabling burst tolerance for {:?}",
                    self.reject_burst_count,
                    REJECT_BURST_TOLERANCE_HOLD
                );
            }
        }
    }

    /// Gets the next frame to display from the queue.
    ///
    /// This will return the appropriate frame based on the current playback
    /// position, dropping frames if we're behind schedule.
    ///
    /// # A/V Sync Strategy (Audio as Master Clock)
    ///
    /// When audio is available, we use audio position as the master clock for
    /// frame selection. This ensures video presentation stays synchronized with
    /// actual audio playback, accounting for audio buffer latency.
    ///
    /// - If video behind audio → skip frames to catch up
    /// - If video ahead of audio → hold current frame (don't advance)
    pub fn get_next_frame(&mut self, queue: &FrameQueue) -> Option<VideoFrame> {
        // If waiting for first frame, accept any frame to start the clock
        if self.waiting_for_first_frame {
            let queue_len = queue.len();
            let Some(frame) = queue.pop() else {
                tracing::trace!(
                    "get_next_frame: waiting_for_first_frame=true, queue empty, returning current_frame={:?}",
                    self.current_frame.as_ref().map(|f| f.pts)
                );
                return self.current_frame.clone();
            };
            tracing::debug!(
                "get_next_frame: FIRST FRAME accepted, pts={:?}, queue_len={}",
                frame.pts,
                queue_len
            );
            self.on_frame_received(frame.pts);
            self.current_frame = Some(frame.clone());
            self.stalled = false;
            // Record sync metrics for first frame
            self.record_sync(frame.pts);
            return Some(frame);
        }

        // Use AUDIO POSITION as master clock when available.
        // This is the key to A/V sync: video presentation follows audio playback.
        // Wall-clock is only used as fallback when audio is not available.
        let current_pos = self.sync_position();

        // Keep popping frames until we find one that should be displayed now
        loop {
            let Some(next_pts) = queue.peek_pts() else {
                // Queue is empty - we're stalled (buffering)
                self.handle_stall();
                return self.current_frame.clone();
            };

            // We have frames - clear stall state and resync clock if needed
            self.clear_stall_if_needed();

            let now = std::time::Instant::now();
            // Accept frame if:
            // 1. It's at or before current position (normal case), OR
            // 2. It's within tolerance AHEAD of current position (adaptive for live jitter), OR
            // 3. We have no current frame (after seek) - accept ANY frame to restart clock.
            let audio_started = self.audio_started_for_timing();
            let gap = next_pts.abs_diff(current_pos);
            let ahead_tolerance = self.compute_ahead_tolerance(audio_started, gap, now);
            let should_accept =
                next_pts <= current_pos + ahead_tolerance || self.current_frame.is_none();

            if !should_accept {
                let rejection_start = *self.rejection_start_time.get_or_insert(now);
                let stuck_duration = now.duration_since(rejection_start);
                self.rejection_count = self.rejection_count.saturating_add(1);
                self.rejection_peak_gap = self.rejection_peak_gap.max(gap);
                let audio_pos = self.audio_position();

                if self.rejection_count == 1 {
                    self.record_reject_window_start(now);
                    tracing::warn!(
                        "get_next_frame: reject window start current_pos={:?}, next_pts={:?}, gap={}ms, tolerance={}ms, audio_pos={:?}, seek_gen={}",
                        current_pos,
                        next_pts,
                        gap.as_millis(),
                        ahead_tolerance.as_millis(),
                        audio_pos,
                        self.seek_generation
                    );
                    if audio_pos == Duration::ZERO {
                        self.maybe_log_audio_zero_diag(
                            now,
                            current_pos,
                            next_pts,
                            gap,
                            ahead_tolerance,
                        );
                    }
                }

                let near_boundary = audio_started
                    && gap > LIVE_BASE_AHEAD_TOLERANCE
                    && gap <= LIVE_BASE_AHEAD_TOLERANCE + LIVE_NEAR_BOUNDARY_RANGE;
                let force_timeout = if near_boundary {
                    NEAR_BOUNDARY_FORCE_TIMEOUT
                } else {
                    STUCK_TIMEOUT
                };
                let gap_is_stale = gap >= STALE_GAP_THRESHOLD;

                if stuck_duration >= force_timeout && gap_is_stale {
                    tracing::warn!(
                        "get_next_frame: dropping stale frame after reject window stuck={:?}, gap={}ms, current_pos={:?}, next_pts={:?}, seek_gen={}",
                        stuck_duration,
                        gap.as_millis(),
                        current_pos,
                        next_pts,
                        self.seek_generation
                    );
                    let _ = queue.pop();
                    self.reset_rejection_tracking();
                    continue;
                }

                if stuck_duration >= force_timeout {
                    if self.forced_resyncs_in_window < MAX_FORCED_RESYNCS_PER_WINDOW {
                        self.forced_resyncs_in_window += 1;
                        tracing::warn!(
                            "get_next_frame: forcing resync attempt {}/{} (stuck={:?}, near_boundary={}, gap={}ms, tolerance={}ms, audio_pos={:?})",
                            self.forced_resyncs_in_window,
                            MAX_FORCED_RESYNCS_PER_WINDOW,
                            stuck_duration,
                            near_boundary,
                            gap.as_millis(),
                            ahead_tolerance.as_millis(),
                            audio_pos
                        );
                        if let Some(frame) = queue.pop() {
                            self.current_position = frame.pts;
                            self.current_frame = Some(frame.clone());
                            self.playback_start_time = Some(std::time::Instant::now());
                            self.playback_start_position = frame.pts;
                            return Some(frame);
                        }
                    } else if let Some(dropped_frame) = queue.pop() {
                        tracing::warn!(
                            "get_next_frame: forced-resync limit reached; dropping head frame pts={:?} after {:?} (rejects={}, peak_gap={}ms, audio_pos={:?})",
                            dropped_frame.pts,
                            stuck_duration,
                            self.rejection_count,
                            self.rejection_peak_gap.as_millis(),
                            audio_pos
                        );
                        self.reset_rejection_tracking();
                        continue;
                    }
                }

                // Log rejection details periodically (every ~1 second)
                if stuck_duration.as_millis() % 1000 < 20 {
                    tracing::debug!(
                        "get_next_frame: rejecting, stuck={:?}, current_pos={:?}, next_pts={:?}, gap={:?}ms, tolerance={}ms, audio_pos={:?}, rejects={}",
                        stuck_duration,
                        current_pos,
                        next_pts,
                        gap.as_millis(),
                        ahead_tolerance.as_millis(),
                        audio_pos,
                        self.rejection_count
                    );
                }
                // We're ahead of schedule, return current frame
                return self.current_frame.clone();
            }

            // Frame accepted - reset rejection tracking
            if let Some(rejection_start) = self.rejection_start_time {
                let stuck_duration = now.duration_since(rejection_start);
                if stuck_duration >= Duration::from_millis(200) {
                    tracing::info!(
                        "get_next_frame: reject window ended after {:?} (rejects={}, peak_gap={}ms, tolerance={}ms, audio_pos={:?})",
                        stuck_duration,
                        self.rejection_count,
                        self.rejection_peak_gap.as_millis(),
                        ahead_tolerance.as_millis(),
                        self.audio_position()
                    );
                }
            }
            self.reset_rejection_tracking();

            let Some(frame) = queue.pop() else { continue };

            // Skip if this frame is older than what we already have
            if let Some(ref current) = self.current_frame {
                if frame.pts < current.pts {
                    continue;
                }
            }

            self.current_position = frame.pts;
            self.current_frame = Some(frame.clone());

            // NOTE: We previously tried setting native_position from frame.pts for macOS,
            // but this creates a circular dependency: sync_position() reads audio.position
            // which we just set to the current frame's PTS, so no future frames can ever
            // be selected (next_pts > current_pos is always true).
            //
            // For native platforms where audio is handled internally (macOS AVPlayer,
            // GStreamer), we should either:
            // 1. Query the native player's audio position (AVPlayer.currentTime), OR
            // 2. Fall back to wall-clock timing (current approach)
            //
            // Until we integrate proper audio position queries from native players,
            // wall-clock timing provides reasonable results.

            // Record sync metrics for displayed frame
            self.record_sync(frame.pts);
            // Track frame for recovery completion
            self.track_recovery_frame();
            return Some(frame);
        }
    }

    /// Records A/V sync metrics for a displayed frame.
    fn record_sync(&mut self, _video_pts: Duration) {
        // Always copy stream PTS offset for reporting (even before audio starts)
        if let Some(ref handle) = self.audio_handle {
            let offset = handle.stream_pts_offset_us();
            if self.sync_metrics.stream_pts_offset_us() != offset {
                self.sync_metrics.set_stream_pts_offset(offset);
            }
        }

        // Only record drift if audio has started (position > 0)
        let audio_pos = self.audio_position();
        if audio_pos > Duration::ZERO {
            // Choose video position metric based on audio source:
            let uses_native_audio = self
                .audio_handle
                .as_ref()
                .map(|h| h.is_using_native_position())
                .unwrap_or(false);

            if uses_native_audio {
                // Native audio (AVPlayer, GStreamer): A/V sync is handled internally
                // by the native player. We can't meaningfully measure drift since
                // our frame display may lag behind the native player's internal state.
                // Mark sync as externally managed so UI shows appropriate message.
                self.sync_metrics.set_sync_externally_managed(true);
                self.sync_metrics.record_frame(audio_pos, audio_pos);
            } else {
                self.sync_metrics.set_sync_externally_managed(false);
                // FFmpeg audio: audio_pos is sample-counted from base_pts after seek.
                // Track when audio actually started and its position at that time.
                if self.audio_start_time.is_none() {
                    self.audio_start_time = Some(std::time::Instant::now());
                    self.audio_start_pos = audio_pos;
                }
                // Calculate drift as: elapsed_time - audio_progress_since_start
                // This is seek-aware: after seek to 50s, audio_pos=50s, audio_start_pos=50s
                // so audio_delta=0, and we compare against elapsed=0 → drift=0
                let elapsed = self
                    .audio_start_time
                    .map(|t| t.elapsed())
                    .unwrap_or(Duration::ZERO);
                let audio_delta = audio_pos.saturating_sub(self.audio_start_pos);
                self.sync_metrics.record_frame(elapsed, audio_delta);
            }
        }
    }

    /// Handles entering stall state when queue is empty.
    fn handle_stall(&mut self) {
        // Only mark as stalled during active playback, not during pause/buffering
        if self.stalled || !self.playback_requested {
            return;
        }
        // Only update position if we have a valid playback start time
        if let Some(start_time) = self.playback_start_time {
            self.current_position = self.playback_start_position + start_time.elapsed();
        }
        self.stalled = true;

        // Record buffer underrun in sync metrics
        self.sync_metrics.record_underrun();

        // Record stall - we classify as decode stall by default since we can't
        // easily distinguish from network stall at this level. Higher-level code
        // (e.g., DecodeThread) could call record_stall(StallType::Network) if it
        // has more context about why frames aren't available.
        self.sync_metrics.record_stall(StallType::Decode);

        // Start recovery tracking
        self.sync_metrics.start_recovery();

        tracing::debug!("Stalled at {:?} (queue empty)", self.current_position);
    }

    /// Clears stall state and resyncs clock when frames become available.
    fn clear_stall_if_needed(&mut self) {
        if !self.stalled {
            return;
        }
        self.stalled = false;
        self.playback_start_time = Some(std::time::Instant::now());
        self.playback_start_position = self.current_position;
        // Reset frame counter for tracking recovery completion
        self.frames_since_recovery = 0;
        tracing::debug!("Resuming from stall at {:?}", self.current_position);
    }

    /// Tracks a frame displayed during recovery and ends recovery when stabilized.
    fn track_recovery_frame(&mut self) {
        if !self.sync_metrics.is_in_recovery() {
            return;
        }

        self.frames_since_recovery += 1;

        // End recovery after enough smooth frames have been displayed
        if self.frames_since_recovery >= RECOVERY_FRAME_THRESHOLD {
            self.sync_metrics.end_recovery();
            self.frames_since_recovery = 0;
            tracing::debug!(
                "Recovery complete after {} frames",
                RECOVERY_FRAME_THRESHOLD
            );
        }
    }

    /// Returns the current frame without advancing.
    pub fn current_frame(&self) -> Option<&VideoFrame> {
        self.current_frame.as_ref()
    }
}

impl Default for FrameScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::video::{CpuFrame, DecodedFrame, PixelFormat, Plane};

    fn make_test_frame(pts: Duration) -> VideoFrame {
        let plane = Plane {
            data: vec![128; 100],
            stride: 10,
        };
        let cpu_frame = CpuFrame::new(PixelFormat::Yuv420p, 10, 10, vec![plane]);
        VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))
    }

    #[test]
    fn test_frame_queue_push_pop() {
        let queue = FrameQueue::new(3);

        queue.push(make_test_frame(Duration::from_millis(0)));
        queue.push(make_test_frame(Duration::from_millis(33)));
        queue.push(make_test_frame(Duration::from_millis(66)));

        assert_eq!(queue.len(), 3);
        assert!(queue.is_full());

        let Some(frame) = queue.pop() else {
            panic!("Expected frame from queue");
        };
        assert_eq!(frame.pts, Duration::from_millis(0));

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_full());
    }

    #[test]
    fn test_frame_queue_flush() {
        let queue = FrameQueue::new(5);

        queue.push(make_test_frame(Duration::from_millis(0)));
        queue.push(make_test_frame(Duration::from_millis(33)));

        assert_eq!(queue.len(), 2);

        queue.flush();

        assert!(queue.is_empty());
        assert!(!queue.is_eos());
    }

    #[test]
    fn test_frame_scheduler_position() {
        let mut scheduler = FrameScheduler::new();

        assert_eq!(scheduler.position(), Duration::ZERO);

        scheduler.seek(Duration::from_secs(10));
        assert_eq!(scheduler.position(), Duration::from_secs(10));

        scheduler.start();
        std::thread::sleep(Duration::from_millis(50));
        assert!(scheduler.position() >= Duration::from_secs(10));

        scheduler.pause();
        let pos = scheduler.position();
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(scheduler.position(), pos);
    }
}
