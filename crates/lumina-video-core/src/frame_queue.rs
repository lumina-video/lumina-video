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

use crate::audio::AudioHandle;
#[cfg(target_os = "macos")]
use crate::audio_decoder::AudioDecoder;
use crate::sync_metrics::{StallType, SyncMetrics};
use crate::video::{VideoDecoderBackend, VideoFrame};

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
    audio_handle: crate::audio::AudioHandle,
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
        let audio_handle = crate::audio::AudioHandle::new();
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
    pub fn handle(&self) -> crate::audio::AudioHandle {
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
    player: &mut crate::audio::AudioPlayer,
    decoder: &mut AudioDecoder,
    producer: &crate::audio_ring_buffer::RingBufferProducer,
    first_samples: &mut bool,
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
            producer.request_flush();
            *first_samples = true; // Next decoded frame reseeds base PTS
            if let Err(e) = decoder.seek(position) {
                tracing::error!("Audio seek failed: {}", e);
            }
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
    handle: crate::audio::AudioHandle,
    command_rx: crossbeam_channel::Receiver<DecodeCommand>,
    stop_flag: Arc<AtomicBool>,
) {
    use crate::audio::AudioPlayer;
    use crate::audio_ring_buffer::RingBufferConfig;

    // Query device sample rate first so ring buffer capacity matches actual output rate
    let device_sample_rate = match AudioPlayer::query_device_sample_rate() {
        Ok(rate) => rate,
        Err(e) => {
            tracing::error!("Failed to query audio device: {}", e);
            handle.set_available(false);
            return;
        }
    };

    // Create audio player backed by ring buffer sized for the actual device rate
    let ring_config = RingBufferConfig::for_vod(device_sample_rate, 2);
    let (mut player, producer) =
        match AudioPlayer::new_ring_buffer(ring_config, Some(handle.clone()), None) {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("Failed to create audio player: {}", e);
                handle.set_available(false);
                return;
            }
        };
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
    let mut first_samples = true;
    const MAX_EMPTY_DECODES_SEEKING: u32 = 100; // Allow ~1s of empty decodes during seek
    const MAX_EMPTY_DECODES_NORMAL: u32 = 10; // Only ~100ms for normal EOF detection

    loop {
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        // Process commands (non-blocking)
        while let Ok(cmd) = command_rx.try_recv() {
            match process_audio_command(
                cmd,
                &mut player,
                &mut decoder,
                &producer,
                &mut first_samples,
            ) {
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
            match process_audio_command(
                cmd,
                &mut player,
                &mut decoder,
                &producer,
                &mut first_samples,
            ) {
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

        // Write decoded samples to ring buffer
        if first_samples {
            first_samples = false;
            handle.set_audio_format(samples.sample_rate, samples.channels as u32);
            handle.set_audio_base_pts(samples.pts);
        }
        producer.write(&samples.data);

        // Adaptive backpressure: the ring buffer is non-blocking (overwrites on overflow),
        // so we must throttle the decode thread. When the buffer is comfortably full,
        // sleep for roughly one audio frame duration. When low, decode at max speed.
        let fill = producer.fill_level();
        let cap = producer.capacity();
        if fill > cap / 2 {
            thread::sleep(Duration::from_millis(20));
        } else if fill > cap / 4 {
            thread::sleep(Duration::from_millis(5));
        }
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
const LIVE_MAX_AHEAD_TOLERANCE: Duration = Duration::from_millis(4600);
/// Margin added when adapting to near-boundary gaps.
const LIVE_AHEAD_MARGIN: Duration = Duration::from_millis(180);
/// Only treat this much extra gap beyond base as near-boundary.
const LIVE_NEAR_BOUNDARY_RANGE: Duration = Duration::from_millis(2600);
/// Small jitter allowance to avoid thrashing exactly at tolerance boundaries.
const LIVE_ACCEPT_JITTER_TOLERANCE: Duration = Duration::from_millis(90);
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
/// Lead target after a discontinuity rebase.
const OFFSET_REBASE_TARGET_LEAD: Duration = Duration::from_millis(150);
/// Minimum stable lead required before applying a one-shot rebase.
const OFFSET_REBASE_MIN_LEAD: Duration = Duration::from_millis(2500);
/// Consecutive stable reject samples required before rebasing.
const OFFSET_REBASE_MIN_STABLE_SAMPLES: u32 = 6;
/// Maximum per-sample lead wobble still considered stable.
const OFFSET_REBASE_STABILITY_TOLERANCE: Duration = Duration::from_millis(220);
/// Clamp to avoid unbounded scheduler bias.
const MAX_VIDEO_PTS_BIAS: Duration = Duration::from_secs(8);
/// Rendering gap threshold: if get_next_frame() hasn't been called for this long,
/// the app was likely backgrounded (alt-tabbed). Audio continues via cpal but video
/// frame consumption stalls, causing A/V drift equal to the gap duration.
/// Normal call interval is ~16ms (60fps vsync); 100ms catches throttled rendering
/// (macOS backgrounds apps at ~5-10fps) without false-triggering during normal playback.
const RENDERING_GAP_THRESHOLD: Duration = Duration::from_millis(100);
/// Stale frame threshold: skip frames whose PTS lags audio by more than this.
/// Prevents displaying old content (and recording false drift) during catch-up
/// after app backgrounding. Matches SYNC_DRIFT_THRESHOLD_MS (100ms) so we never
/// display a frame that would register as out-of-sync.
const STALE_FRAME_LAG_THRESHOLD: Duration = Duration::from_millis(100);
/// Max number of aggressive catch-up frame admissions before escalating.
const CATCH_UP_MAX_FRAMES: u32 = 10;
/// If catch-up does not recover quickly, escalate to forced resync.
const CATCH_UP_MAX_DURATION: Duration = Duration::from_millis(1300);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RejectHandlingState {
    Normal,
    CatchUp,
    Resync,
}

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
    /// Last time we logged 10s A/V clock delta diagnostics.
    last_clock_delta_log: Option<std::time::Instant>,
    /// Wall-clock elapsed and audio_delta at last log point (for computing 10s deltas).
    last_clock_delta_values: Option<(Duration, Duration)>,
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
    /// Current reject-handling phase.
    reject_state: RejectHandlingState,
    /// Number of frames aggressively accepted while in catch-up.
    catch_up_frames_in_window: u32,
    /// One-shot guard: only apply one offset rebase per reject window.
    offset_rebased_in_window: bool,
    /// Last lead seen during the current reject window.
    last_reject_lead: Option<Duration>,
    /// Number of consecutive stable lead samples in the reject window.
    stable_reject_lead_samples: u32,
    /// Scheduler-only bias to account for persistent video PTS lead over audio.
    video_pts_bias: Duration,
    /// Accumulated clock-drift correction in microseconds (signed).
    /// Applied alongside `video_pts_bias` at the frame acceptance point.
    /// Combined total is clamped to ±MAX_VIDEO_PTS_BIAS.
    clock_drift_correction_us: i64,
    /// Last wall-clock time the drift controller ran.
    last_drift_update: Option<std::time::Instant>,
    /// True when drift correction is actively slewing (hysteresis state).
    drift_correction_active: bool,
    /// EMA-smoothed drift signal (microseconds) for controller input.
    smoothed_drift_us: f64,
    /// Deadline accumulator for frame-rate pacing.
    /// Advances by `frame_pacing_interval` on each accept, producing a natural
    /// 2-3 tick cadence (e.g., 24fps on 60Hz) instead of hard 3-tick quantization.
    next_frame_due: Option<std::time::Instant>,
    /// Minimum interval between frame acceptances (1/fps), derived from metadata.
    /// Zero disables pacing (e.g., for VOD or unknown frame rate).
    frame_pacing_interval: Duration,
    /// When true, audio position drives frame selection (VOD/FFmpeg).
    /// When false, wall-clock drives frame selection but audio is still
    /// tracked for sync metrics (MoQ live).
    use_audio_as_sync_master: bool,
    /// True when frozen due to audio ring buffer underrun (separate from queue-empty `stalled`).
    audio_stalled: bool,
    /// Wall-clock time when audio stall started (for timeout).
    audio_stall_start: Option<std::time::Instant>,
    /// Latched bypass: after a stall timeout, skip re-entering stall until audio
    /// truly recovers (i.e. `is_audio_stalled()` returns false).
    ignore_audio_stall_until_recovered: bool,
    /// True once the initial audio-video PTS bias has been computed (one-shot).
    initial_pts_bias_applied: bool,
    /// Deferred epoch: the PTS threshold at which to enable the cpal playback epoch.
    /// Set during MoQ rebase so audio stays gated while old GOP frames are consumed.
    /// Cleared once a frame with PTS >= this value is accepted.
    deferred_epoch_pts: Option<Duration>,
    /// Last time get_next_frame() was called (for rendering gap detection).
    /// A gap > RENDERING_GAP_THRESHOLD indicates the app was backgrounded.
    last_get_next_frame_time: Option<std::time::Instant>,
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
            last_clock_delta_log: None,
            last_clock_delta_values: None,
            rejection_count: 0,
            rejection_peak_gap: Duration::ZERO,
            forced_resyncs_in_window: 0,
            last_reject_window_start: None,
            reject_burst_count: 0,
            burst_tolerance_until: None,
            last_audio_zero_diag: None,
            reject_state: RejectHandlingState::Normal,
            catch_up_frames_in_window: 0,
            offset_rebased_in_window: false,
            last_reject_lead: None,
            stable_reject_lead_samples: 0,
            video_pts_bias: Duration::ZERO,
            clock_drift_correction_us: 0,
            last_drift_update: None,
            drift_correction_active: false,
            smoothed_drift_us: 0.0,
            next_frame_due: None,
            frame_pacing_interval: Duration::ZERO,
            use_audio_as_sync_master: true,
            audio_stalled: false,
            audio_stall_start: None,
            ignore_audio_stall_until_recovered: false,
            initial_pts_bias_applied: false,
            deferred_epoch_pts: None,
            last_get_next_frame_time: None,
        }
    }

    /// Creates a new frame scheduler with audio handle for sync tracking.
    pub fn with_audio_handle(audio_handle: AudioHandle) -> Self {
        let mut s = Self::new();
        s.set_audio_handle(audio_handle);
        s
    }

    /// Sets the audio handle for sync tracking and uses audio as master clock.
    pub fn set_audio_handle(&mut self, audio_handle: AudioHandle) {
        self.audio_handle = Some(audio_handle);
        self.last_audio_zero_diag = None;
        self.sync_metrics.set_using_audio_clock(true);
        self.use_audio_as_sync_master = true;
    }

    /// Sets the audio handle for sync metrics only (wall-clock drives frame pacing).
    /// Used for MoQ live where audio handle arrives late and the handoff to
    /// audio-as-master-clock would cause a position discontinuity.
    pub fn set_audio_handle_metrics_only(&mut self, audio_handle: AudioHandle) {
        self.audio_handle = Some(audio_handle);
        self.last_audio_zero_diag = None;
        self.sync_metrics.set_using_audio_clock(true);
        self.use_audio_as_sync_master = false;
    }

    /// Clears the audio handle, falling back to wall-clock for frame pacing.
    pub fn clear_audio_handle(&mut self) {
        self.audio_handle = None;
        self.sync_metrics.set_using_audio_clock(false);
        self.use_audio_as_sync_master = true; // reset to default
        self.audio_start_time = None;
        self.audio_start_pos = Duration::ZERO;
        self.initial_pts_bias_applied = false;
        self.deferred_epoch_pts = None;
        self.audio_stalled = false;
        self.audio_stall_start = None;
        self.ignore_audio_stall_until_recovered = false;
        self.last_get_next_frame_time = None;
        self.reset_drift_correction();
    }

    /// Resets drift correction state: zeroes the accumulated correction and
    /// clears the EMA filter so the controller restarts cleanly.
    fn reset_drift_correction(&mut self) {
        self.clock_drift_correction_us = 0;
        self.last_drift_update = None;
        self.drift_correction_active = false;
        self.smoothed_drift_us = 0.0;
    }

    /// Returns the PTS used to anchor the playback clock (first video frame PTS).
    pub fn playback_start_position(&self) -> Duration {
        self.playback_start_position
    }

    /// Sets the initial video PTS bias to compensate for audio-video PTS offset
    /// when subscribing mid-stream (e.g., MoQ join where first video frame is at
    /// the start of a GOP but first audio frame is at the live edge).
    pub fn set_initial_pts_bias(&mut self, bias: Duration) {
        let clamped = std::cmp::min(bias, MAX_VIDEO_PTS_BIAS);
        if !clamped.is_zero() {
            tracing::info!(
                "Initial video PTS bias: {}ms (audio ahead of video at join)",
                clamped.as_millis()
            );
            self.video_pts_bias = clamped;
        }
    }

    /// Sets frame-rate pacing for live streams. When fps > 0, the scheduler
    /// won't accept a new frame sooner than 1/fps after the last acceptance.
    /// This prevents burst consumption of group-boundary deliveries.
    /// Updates pacing if fps changes (e.g., provisional → accurate metadata).
    pub fn set_frame_rate_pacing(&mut self, fps: f32) {
        if fps <= 0.0 {
            return;
        }
        let new_interval = Duration::from_secs_f64(1.0 / fps as f64);
        if self.frame_pacing_interval == new_interval {
            return; // No change
        }
        let was_zero = self.frame_pacing_interval.is_zero();
        self.frame_pacing_interval = new_interval;
        if was_zero {
            tracing::info!(
                "Frame-rate pacing enabled: {:.1} fps ({:.1}ms interval)",
                fps,
                1000.0 / fps as f64,
            );
        } else {
            tracing::info!(
                "Frame-rate pacing updated: {:.1} fps ({:.1}ms interval)",
                fps,
                1000.0 / fps as f64,
            );
        }
    }

    /// Disables frame-rate pacing (e.g., when a stream acquires duration metadata
    /// and is no longer treated as live).
    pub fn clear_frame_rate_pacing(&mut self) {
        if !self.frame_pacing_interval.is_zero() {
            tracing::info!("Frame-rate pacing disabled");
            self.frame_pacing_interval = Duration::ZERO;
            self.next_frame_due = None;
        }
    }

    /// Advances the frame-pacing deadline accumulator.
    /// Uses `due + interval` (not `now + interval`) to maintain a smooth cadence.
    /// Snaps forward if the deadline fell behind by more than one interval
    /// (e.g., after a stall) to prevent burst catch-up.
    fn advance_frame_pacing(&mut self) {
        if self.frame_pacing_interval.is_zero() {
            return;
        }
        let now = std::time::Instant::now();
        self.next_frame_due = Some(match self.next_frame_due {
            Some(due) => {
                let next = due + self.frame_pacing_interval;
                // Snap forward if we fell behind by more than one interval
                if next + self.frame_pacing_interval < now {
                    now + self.frame_pacing_interval
                } else {
                    next
                }
            }
            None => now + self.frame_pacing_interval,
        });
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

        // Only use audio as master clock when explicitly enabled (VOD/FFmpeg).
        // MoQ live uses wall-clock for frame pacing to avoid late-bind discontinuity.
        if !self.use_audio_as_sync_master {
            return wall_clock_pos;
        }

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
        self.last_clock_delta_log = None;
        self.last_clock_delta_values = None;
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
        self.last_clock_delta_log = None;
        self.last_clock_delta_values = None;
        // Preserve clock_drift_correction_us across pause/resume — the
        // accumulated correction is still valid since the clock pair hasn't changed.
        self.last_drift_update = None;
        self.drift_correction_active = false;
        self.smoothed_drift_us = 0.0;
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
        self.playback_start_position = position;
        self.current_frame = None;
        self.stalled = false;
        self.reset_rejection_tracking();
        self.seek_generation = self.seek_generation.wrapping_add(1);
        self.video_pts_bias = Duration::ZERO;
        self.reset_drift_correction();
        self.initial_pts_bias_applied = false;
        self.deferred_epoch_pts = None;
        self.last_get_next_frame_time = None;

        // Reset sync metrics on seek to clear max_drift from transient spikes
        self.sync_metrics.reset();
        // Set grace period to filter drift spikes during seek warmup
        self.sync_metrics.set_grace_period(30);

        // Always clear audio clock state on seek — even while paused — so stale
        // epoch/base_pts don't leak into position() when playback later resumes.
        if let Some(ref audio) = self.audio_handle {
            audio.clear_playback_epoch();
            audio.clear_native_position();
            audio.reset_samples_played();
            audio.clear_audio_base_pts();
        }

        if self.playback_requested {
            // Wait for first frame at new position before resuming clock
            self.waiting_for_first_frame = true;
            self.playback_start_time = None;
            self.audio_start_time = None;
            self.audio_start_pos = Duration::ZERO;
            self.last_clock_delta_log = None;
            self.last_clock_delta_values = None;
        }
    }

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        // If stalled (queue empty or audio underrun), return the last known position
        // to prevent the scroll bar / subtitles from advancing during buffering
        if self.stalled || self.audio_stalled {
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

    /// Returns true when stalled due to either queue-empty or audio ring buffer underrun.
    pub fn is_stalled(&self) -> bool {
        self.stalled || self.audio_stalled
    }

    /// Returns true when frozen specifically due to audio ring buffer underrun.
    pub fn is_audio_stall(&self) -> bool {
        self.audio_stalled
    }

    /// Clears audio stall state (called when audio handle becomes stale/unavailable).
    pub fn clear_audio_stall(&mut self) {
        self.ignore_audio_stall_until_recovered = false;
        if self.audio_stalled {
            self.audio_stalled = false;
            self.audio_stall_start = None;
            if self.playback_requested {
                self.playback_start_time = Some(std::time::Instant::now());
                self.playback_start_position = self.current_position;
            }
            self.frames_since_recovery = 0;
            tracing::debug!(
                "Cleared audio stall (handle removed) at {:?}",
                self.current_position
            );
        }
    }

    /// Called when a frame is received to sync the clock.
    /// If we were waiting for the first frame, this starts the clock.
    fn on_frame_received(&mut self, frame_pts: Duration) {
        if self.waiting_for_first_frame && self.playback_requested {
            // First frame after play/seek - start the clock synced to frame PTS
            self.playback_start_time = Some(std::time::Instant::now());
            self.playback_start_position = frame_pts;
            self.waiting_for_first_frame = false;

            // Start the shared audio playback epoch so A/V clocks are synchronized.
            // Skip for MoQ (wall-clock mode): the rebase block in get_next_frame()
            // will enable the epoch after aligning the video clock to audio_base_pts.
            // Setting it here would let cpal consume ring buffer samples before the
            // rebase, creating a permanent A/V content offset.
            if self.use_audio_as_sync_master {
                if let Some(ref audio) = self.audio_handle {
                    audio.start_playback_epoch();
                    tracing::debug!(
                        "Clock started at frame PTS {:?}, audio epoch synchronized",
                        frame_pts
                    );
                } else {
                    tracing::debug!("Clock started at frame PTS {:?}", frame_pts);
                }
            } else {
                tracing::debug!(
                    "Clock started at frame PTS {:?} (epoch deferred for MoQ rebase)",
                    frame_pts
                );
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

    /// Time-based proportional controller that compensates for clock drift
    /// between the system monotonic clock (video pacing) and the cpal hardware
    /// clock (audio position). Updates at 200ms cadence with hysteresis deadband.
    fn update_clock_drift_correction(&mut self) {
        if self.use_audio_as_sync_master {
            return;
        }

        // Skip for native audio (externally managed sync).
        let uses_native = self
            .audio_handle
            .as_ref()
            .map(|h| h.is_using_native_position())
            .unwrap_or(false);
        if uses_native || self.sync_metrics.is_sync_externally_managed() {
            return;
        }

        // Gate: no correction during stall.
        // Reset EMA/hysteresis so stale drift doesn't bias post-stall correction.
        // Note: is_in_recovery() intentionally excluded — recovery can persist if
        // stalls recur faster than RECOVERY_FRAME_THRESHOLD, creating a chicken-and-egg
        // where the controller can never run to fix the drift causing the stalls.
        if self.stalled || self.audio_stalled {
            self.last_drift_update = None;
            self.drift_correction_active = false;
            self.smoothed_drift_us = 0.0;
            return;
        }

        if self.audio_position() == Duration::ZERO {
            return;
        }

        // Warmup: 5s after audio starts
        match self.audio_start_time {
            Some(start) if start.elapsed() >= Duration::from_secs(5) => {}
            _ => return,
        }

        // Time-based rate limiting: every 200ms, dt capped at 500ms
        let now = std::time::Instant::now();
        let dt = match self.last_drift_update {
            Some(last) => {
                let elapsed = now.duration_since(last);
                if elapsed < Duration::from_millis(200) {
                    return;
                }
                std::cmp::min(elapsed, Duration::from_millis(500))
            }
            None => {
                self.last_drift_update = Some(now);
                return;
            }
        };
        self.last_drift_update = Some(now);

        // EMA-smooth the raw drift signal (alpha=0.3, ~1s window at 200ms intervals)
        let raw_drift = self.sync_metrics.current_drift_us() as f64;

        // Step detection: large drift (>200ms) indicates sudden desync from app
        // backgrounding or similar disruption. The proportional controller's 10ms/s
        // max slew would take ~100s to correct a 1s step. Instead, resync the wall
        // clock to the audio position immediately.
        const STEP_THRESHOLD_US: f64 = 100_000.0; // 100ms — matches SYNC_DRIFT_THRESHOLD_MS
        if raw_drift.abs() > STEP_THRESHOLD_US {
            let audio_pos = self.audio_position();
            if audio_pos > Duration::ZERO {
                tracing::info!(
                    "Drift controller: step drift {}ms detected, resyncing wall clock to audio ({:?})",
                    raw_drift as i64 / 1000,
                    audio_pos
                );
                self.playback_start_position = audio_pos;
                self.playback_start_time = Some(now);
                self.current_position = audio_pos;
                self.reset_drift_correction();
                self.sync_metrics.set_grace_period(10);
                return;
            }
        }

        const EMA_ALPHA: f64 = 0.3;
        self.smoothed_drift_us = self.smoothed_drift_us * (1.0 - EMA_ALPHA) + raw_drift * EMA_ALPHA;
        let drift_us = self.smoothed_drift_us as i64;

        // Hysteresis deadband: enter at |drift| > 40ms, exit at < 20ms
        const ENTER_US: i64 = 40_000;
        const EXIT_US: i64 = 20_000;

        if self.drift_correction_active {
            if drift_us.abs() < EXIT_US {
                self.drift_correction_active = false;
                return;
            }
        } else {
            if drift_us.abs() < ENTER_US {
                return;
            }
            self.drift_correction_active = true;
        }

        // Proportional controller: slew at most 10ms/s (1% speed change)
        const MAX_SLEW_US_PER_SEC: f64 = 10_000.0;
        const GAIN: f64 = 0.1;

        let dt_secs = dt.as_secs_f64();
        let desired_rate = -(drift_us as f64) * GAIN;
        let clamped_rate = desired_rate.clamp(-MAX_SLEW_US_PER_SEC, MAX_SLEW_US_PER_SEC);
        let delta = (clamped_rate * dt_secs) as i64;

        self.clock_drift_correction_us += delta;

        let max_us = MAX_VIDEO_PTS_BIAS.as_micros() as i64;
        self.clock_drift_correction_us = self.clock_drift_correction_us.clamp(-max_us, max_us);
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
        self.reject_state = RejectHandlingState::Normal;
        self.catch_up_frames_in_window = 0;
        self.offset_rebased_in_window = false;
        self.last_reject_lead = None;
        self.stable_reject_lead_samples = 0;
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

    fn maybe_rebase_offset_for_stable_lead(
        &mut self,
        current_pos: Duration,
        next_pts: Duration,
        lead: Duration,
    ) -> bool {
        let stable = match self.last_reject_lead {
            Some(prev_lead) => lead.abs_diff(prev_lead) <= OFFSET_REBASE_STABILITY_TOLERANCE,
            None => false,
        };

        self.last_reject_lead = Some(lead);
        if stable {
            self.stable_reject_lead_samples = self.stable_reject_lead_samples.saturating_add(1);
        } else {
            self.stable_reject_lead_samples = 1;
        }

        if self.offset_rebased_in_window
            || lead < OFFSET_REBASE_MIN_LEAD
            || self.stable_reject_lead_samples < OFFSET_REBASE_MIN_STABLE_SAMPLES
        {
            return false;
        }

        let bias_delta = lead.saturating_sub(OFFSET_REBASE_TARGET_LEAD);
        if bias_delta.is_zero() {
            return false;
        }

        let previous_bias = self.video_pts_bias;
        self.video_pts_bias = std::cmp::min(
            self.video_pts_bias.saturating_add(bias_delta),
            MAX_VIDEO_PTS_BIAS,
        );
        self.offset_rebased_in_window = true;
        self.reject_state = RejectHandlingState::CatchUp;
        // Filter reset — bias jumped, stale EMA/hysteresis could produce wrong-way correction.
        self.last_drift_update = None;
        self.drift_correction_active = false;
        self.smoothed_drift_us = 0.0;

        tracing::warn!(
            "get_next_frame: applied one-shot offset rebase (lead={}ms, stable_samples={}, current_pos={:?}, next_pts={:?}, bias {}ms -> {}ms)",
            lead.as_millis(),
            self.stable_reject_lead_samples,
            current_pos,
            next_pts,
            previous_bias.as_millis(),
            self.video_pts_bias.as_millis()
        );

        true
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
            self.advance_frame_pacing();
            // Record sync metrics for first frame
            self.record_sync(frame.pts);
            return Some(frame);
        }

        // Rendering gap detection: when the app is backgrounded (alt-tab), egui stops
        // or throttles repainting, but cpal audio continues playing. This creates A/V
        // drift equal to the gap duration. Detect the gap and resync immediately.
        if !self.use_audio_as_sync_master && self.playback_start_time.is_some() {
            let now = std::time::Instant::now();
            if let Some(last) = self.last_get_next_frame_time {
                let gap = now.duration_since(last);
                if gap > RENDERING_GAP_THRESHOLD {
                    let audio_pos = self.audio_position();
                    if audio_pos > Duration::ZERO {
                        // Drain stale frames — they'd show old content while audio is
                        // at the live edge, creating a drift spike during catch-up.
                        let mut drained = 0;
                        while queue.pop().is_some() {
                            drained += 1;
                        }
                        tracing::info!(
                            "Rendering gap {}ms: drained {} stale frames, resyncing wall clock to audio ({:?})",
                            gap.as_millis(),
                            drained,
                            audio_pos
                        );
                        // Resync wall clock to audio position
                        self.playback_start_position = audio_pos;
                        self.playback_start_time = Some(now);
                        self.current_position = audio_pos;
                        self.reset_drift_correction();
                        // Reset pacing so next fresh frame is accepted immediately
                        self.next_frame_due = None;
                        // Grace period to filter transient drift spikes during recovery
                        self.sync_metrics.set_grace_period(10);
                    }
                }
            }
            self.last_get_next_frame_time = Some(now);
        }

        // Frame-rate pacing: prevent burst consumption of group-boundary deliveries.
        // Uses a deadline accumulator (next_due += interval) instead of elapsed-since-last
        // to avoid UI refresh aliasing. At 24fps on 60Hz, the old elapsed check quantized
        // to 50ms/frame (20fps); the accumulator produces a 2-3 tick cadence averaging 24fps.
        if !self.frame_pacing_interval.is_zero() {
            let now = std::time::Instant::now();
            if let Some(due) = self.next_frame_due {
                if now < due {
                    return self.current_frame.clone();
                }
            }
        }

        // Deferred clock rebase: when subscribing mid-stream, the first video frame
        // comes from the GOP start (older PTS) while the first audio frame is at the
        // live edge (newer PTS). Rebase the playback clock to audio's start PTS so that
        // all old GOP frames are consumed instantly. The queue will empty during catch-up,
        // triggering handle_stall() which freezes the clock until real-time frames arrive.
        // This prevents wall-clock from advancing while the decode pipeline chews through
        // the old GOP, keeping video aligned with audio when playback resumes.
        if !self.initial_pts_bias_applied && !self.use_audio_as_sync_master {
            if let Some(ref ah) = self.audio_handle {
                if let Some(audio_start) = ah.audio_base_pts() {
                    self.initial_pts_bias_applied = true;
                    let video_start = self.playback_start_position;
                    let offset = audio_start.saturating_sub(video_start);
                    if offset > Duration::from_millis(100) {
                        tracing::info!(
                            "MoQ join rebase: video_start={:?}, audio_start={:?}, offset={}ms — rebasing clock",
                            video_start, audio_start, offset.as_millis()
                        );
                        self.playback_start_position = audio_start;
                        self.playback_start_time = Some(std::time::Instant::now());
                        self.current_position = audio_start;
                    }
                    // Defer the playback epoch until video catches up to the live edge.
                    // Old GOP frames (PTS < audio_start) are still being consumed at paced
                    // rate. If we enable audio now, cpal plays live-edge audio while video
                    // shows stale GOP content, creating a permanent offset equal to the
                    // catch-up duration. Instead, gate cpal until the first frame with
                    // PTS >= audio_start is accepted.
                    self.deferred_epoch_pts = Some(audio_start);
                    self.reset_drift_correction();
                    tracing::info!("MoQ: epoch deferred until video PTS >= {:?}", audio_start);
                }
            }
        }

        // Audio-stall gate: if audio ring buffer is underrunning and we're using
        // wall-clock pacing (MoQ live), freeze video to prevent A/V drift.
        // VOD (audio-as-master) naturally pauses via stale sync_position().
        // The position_for_sync() > ZERO guard prevents false stalls during startup:
        // set_available(true) fires at bind time before first decoded audio arrives,
        // so empty callbacks are expected until position advances.
        if !self.use_audio_as_sync_master {
            if let Some(ref ah) = self.audio_handle {
                let audio_producing = ah.is_available() && ah.position_for_sync() > Duration::ZERO;
                if audio_producing && ah.is_audio_stalled() {
                    if self.ignore_audio_stall_until_recovered {
                        // Latch active: skip stall gate until audio truly recovers.
                    } else {
                        // Timeout: if audio stall lasts >3s, give up and resume wall-clock.
                        // This handles publisher stream loops or permanent audio loss.
                        let timed_out = self
                            .audio_stall_start
                            .map(|t| t.elapsed() > Duration::from_secs(3))
                            .unwrap_or(false);
                        if timed_out {
                            tracing::warn!(
                                "Audio stall timeout (>3s) at {:?}, resuming wall-clock",
                                self.current_position
                            );
                            self.exit_audio_stall();
                            self.ignore_audio_stall_until_recovered = true;
                        } else {
                            if !self.audio_stalled {
                                self.enter_audio_stall();
                            }
                            return self.current_frame.clone();
                        }
                    }
                } else {
                    // Audio recovered (no longer stalled) — clear latch and stall state.
                    if self.ignore_audio_stall_until_recovered {
                        self.ignore_audio_stall_until_recovered = false;
                    }
                    if self.audio_stalled {
                        self.exit_audio_stall();
                    }
                }
            }
        }

        // Use AUDIO POSITION as master clock when available.
        // This is the key to A/V sync: video presentation follows audio playback.
        // Wall-clock is only used as fallback when audio is not available.
        // Keep popping frames until we find one that should be displayed now
        loop {
            let Some(next_pts) = queue.peek_pts() else {
                // Queue is empty - we're stalled (buffering)
                self.handle_stall();
                return self.current_frame.clone();
            };

            // We have frames - clear stall state and resync clock if needed
            self.clear_stall_if_needed();
            self.update_clock_drift_correction();
            let raw_sync_pos = self.sync_position();
            let total_bias_us =
                self.video_pts_bias.as_micros() as i64 + self.clock_drift_correction_us;
            let max_bias_us = MAX_VIDEO_PTS_BIAS.as_micros() as i64;
            let clamped_bias_us = total_bias_us.clamp(-max_bias_us, max_bias_us);
            let effective_bias_ms = clamped_bias_us / 1000;
            let current_pos = if clamped_bias_us >= 0 {
                raw_sync_pos.saturating_add(Duration::from_micros(clamped_bias_us as u64))
            } else {
                raw_sync_pos.saturating_sub(Duration::from_micros((-clamped_bias_us) as u64))
            };

            let now = std::time::Instant::now();
            // Accept frame if:
            // 1. It's at or before current position (normal case), OR
            // 2. It's within tolerance AHEAD of current position (adaptive for live jitter), OR
            // 3. We have no current frame (after seek) - accept ANY frame to restart clock.
            let audio_started = self.audio_started_for_timing();
            let gap = next_pts.abs_diff(current_pos);
            let ahead_tolerance = self.compute_ahead_tolerance(audio_started, gap, now);
            let accept_tolerance = ahead_tolerance.saturating_add(LIVE_ACCEPT_JITTER_TOLERANCE);
            let should_accept =
                next_pts <= current_pos + accept_tolerance || self.current_frame.is_none();

            if !should_accept {
                let rejection_start = *self.rejection_start_time.get_or_insert(now);
                let stuck_duration = now.duration_since(rejection_start);
                self.rejection_count = self.rejection_count.saturating_add(1);
                self.rejection_peak_gap = self.rejection_peak_gap.max(gap);
                let lead = next_pts.saturating_sub(current_pos);
                let audio_pos = self.audio_position();

                if self.rejection_count == 1 {
                    self.record_reject_window_start(now);
                    tracing::warn!(
                        "get_next_frame: reject window start sync_pos={:?}, effective_pos={:?}, next_pts={:?}, gap={}ms, lead={}ms, tolerance={}ms, eff_bias={}ms, state={:?}, audio_pos={:?}, seek_gen={}",
                        raw_sync_pos,
                        current_pos,
                        next_pts,
                        gap.as_millis(),
                        lead.as_millis(),
                        accept_tolerance.as_millis(),
                        effective_bias_ms,
                        self.reject_state,
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

                if self.maybe_rebase_offset_for_stable_lead(current_pos, next_pts, lead) {
                    continue;
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
                    if self.reject_state == RejectHandlingState::Normal {
                        self.reject_state = RejectHandlingState::CatchUp;
                        tracing::warn!(
                            "get_next_frame: entering catch-up mode (stuck={:?}, lead={}ms, gap={}ms, tolerance={}ms, eff_bias={}ms)",
                            stuck_duration,
                            lead.as_millis(),
                            gap.as_millis(),
                            accept_tolerance.as_millis(),
                            effective_bias_ms
                        );
                    }

                    if self.reject_state == RejectHandlingState::CatchUp {
                        let catch_up_timed_out =
                            stuck_duration >= force_timeout.saturating_add(CATCH_UP_MAX_DURATION);
                        if catch_up_timed_out
                            || self.catch_up_frames_in_window >= CATCH_UP_MAX_FRAMES
                        {
                            self.reject_state = RejectHandlingState::Resync;
                            tracing::warn!(
                                "get_next_frame: escalating catch-up to resync (stuck={:?}, catch_up_frames={}, lead={}ms)",
                                stuck_duration,
                                self.catch_up_frames_in_window,
                                lead.as_millis()
                            );
                        } else if let Some(frame) = queue.pop() {
                            self.catch_up_frames_in_window =
                                self.catch_up_frames_in_window.saturating_add(1);
                            tracing::debug!(
                                "get_next_frame: catch-up accepted frame pts={:?} (count={}, lead={}ms, eff_bias={}ms)",
                                frame.pts,
                                self.catch_up_frames_in_window,
                                lead.as_millis(),
                                effective_bias_ms
                            );
                            self.current_position = frame.pts;
                            self.current_frame = Some(frame.clone());
                            self.playback_start_time = Some(std::time::Instant::now());
                            self.playback_start_position = frame.pts;
                            self.advance_frame_pacing();
                            self.record_sync(frame.pts);
                            self.track_recovery_frame();
                            return Some(frame);
                        }
                    }

                    if self.reject_state == RejectHandlingState::Resync
                        && self.forced_resyncs_in_window < MAX_FORCED_RESYNCS_PER_WINDOW
                    {
                        self.forced_resyncs_in_window += 1;
                        tracing::warn!(
                            "get_next_frame: forcing resync attempt {}/{} (stuck={:?}, near_boundary={}, gap={}ms, tolerance={}ms, audio_pos={:?})",
                            self.forced_resyncs_in_window,
                            MAX_FORCED_RESYNCS_PER_WINDOW,
                            stuck_duration,
                            near_boundary,
                            gap.as_millis(),
                            accept_tolerance.as_millis(),
                            audio_pos
                        );
                        if let Some(frame) = queue.pop() {
                            self.current_position = frame.pts;
                            self.current_frame = Some(frame.clone());
                            self.playback_start_time = Some(std::time::Instant::now());
                            self.playback_start_position = frame.pts;
                            self.advance_frame_pacing();
                            return Some(frame);
                        }
                    } else if self.reject_state == RejectHandlingState::Resync {
                        if let Some(dropped_frame) = queue.pop() {
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
                }

                // Log rejection details periodically (every ~1 second)
                if stuck_duration.as_millis() % 1000 < 20 {
                    tracing::debug!(
                        "get_next_frame: rejecting, stuck={:?}, current_pos={:?}, next_pts={:?}, gap={:?}ms, tolerance={}ms, eff_bias={}ms, state={:?}, audio_pos={:?}, rejects={}",
                        stuck_duration,
                        current_pos,
                        next_pts,
                        gap.as_millis(),
                        accept_tolerance.as_millis(),
                        effective_bias_ms,
                        self.reject_state,
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
                        accept_tolerance.as_millis(),
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

            // Skip stale frames in wall-clock mode: if frame PTS lags audio by more
            // than STALE_FRAME_LAG_THRESHOLD, the frame would show old content while
            // audio is at the live edge. Drop it and try the next queued frame.
            // This is a safety net for cases the rendering gap detection didn't catch
            // (e.g., reduced-rate rendering during background, gradual accumulation).
            if !self.use_audio_as_sync_master {
                let audio_pos = self.audio_position();
                if audio_pos > Duration::ZERO {
                    let lag = audio_pos.saturating_sub(frame.pts);
                    if lag > STALE_FRAME_LAG_THRESHOLD {
                        tracing::debug!(
                            "Skipping stale frame PTS={:?} (audio at {:?}, lag={}ms)",
                            frame.pts,
                            audio_pos,
                            lag.as_millis()
                        );
                        continue;
                    }
                }
            }

            self.current_position = frame.pts;
            self.current_frame = Some(frame.clone());
            self.advance_frame_pacing();

            // Deferred epoch: enable audio once video reaches the live edge.
            // This fires once, when the first frame with PTS >= the rebase target
            // is accepted, ensuring audio and video start from the same content point.
            if let Some(threshold) = self.deferred_epoch_pts {
                if frame.pts >= threshold {
                    if let Some(ref ah) = self.audio_handle {
                        ah.start_playback_epoch();
                        tracing::info!(
                            "MoQ: epoch enabled at video PTS {:?} (threshold {:?})",
                            frame.pts,
                            threshold
                        );
                    }
                    self.deferred_epoch_pts = None;
                }
            }

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
    fn record_sync(&mut self, video_pts: Duration) {
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
            } else if !self.use_audio_as_sync_master {
                // MoQ wall-clock mode: compare displayed video PTS directly against
                // audio content position. This captures both rate drift AND any
                // constant content offset (e.g., audio started ahead of video).
                // The previous approach compared wall-clock vs audio-clock rate,
                // which only detected rate drift and missed constant offsets.
                self.sync_metrics.set_sync_externally_managed(false);

                if self.audio_start_time.is_none() {
                    self.audio_start_time = Some(std::time::Instant::now());
                    self.audio_start_pos = audio_pos;
                    tracing::info!(
                        "record_sync(MoQ): first measurement video_pts={:?}, audio_pos={:?}, offset={}ms",
                        video_pts,
                        audio_pos,
                        video_pts.as_millis() as i64 - audio_pos.as_millis() as i64,
                    );
                }

                self.sync_metrics.record_frame(video_pts, audio_pos);

                // 10s diagnostic: log content alignment and clock rates
                let now = std::time::Instant::now();
                let elapsed = self
                    .audio_start_time
                    .map(|t| t.elapsed())
                    .unwrap_or(Duration::ZERO);
                let should_log = match self.last_clock_delta_log {
                    None => elapsed > Duration::from_secs(1),
                    Some(last) => now.duration_since(last) >= Duration::from_secs(10),
                };
                if should_log {
                    let drift_ms = video_pts.as_millis() as i64 - audio_pos.as_millis() as i64;
                    tracing::info!(
                        "A/V content sync (10s): video_pts={:?}, audio_pos={:?}, drift={}ms, drift_corr={}ms, eff_bias={}ms",
                        video_pts,
                        audio_pos,
                        drift_ms,
                        self.clock_drift_correction_us / 1000,
                        (self.video_pts_bias.as_micros() as i64 + self.clock_drift_correction_us)
                            .clamp(-(MAX_VIDEO_PTS_BIAS.as_micros() as i64), MAX_VIDEO_PTS_BIAS.as_micros() as i64) / 1000,
                    );
                    self.last_clock_delta_log = Some(now);
                }
            } else {
                self.sync_metrics.set_sync_externally_managed(false);
                // VOD (audio-as-master): audio drives video pacing, so measure
                // audio clock rate accuracy vs wall-clock. Any rate drift here
                // IS the A/V drift since video follows audio by construction.
                if self.audio_start_time.is_none() {
                    self.audio_start_time = Some(std::time::Instant::now());
                    self.audio_start_pos = audio_pos;
                    if let Some(ref h) = self.audio_handle {
                        let ch = h.channels();
                        if ch > 0 {
                            tracing::info!(
                                "record_sync: first measurement audio_pos={:?}, samples_played={}, raw_pos={:?}, samples_dur={:?}, channels={}",
                                audio_pos,
                                h.samples_played(),
                                h.position(),
                                h.samples_played_duration(),
                                ch,
                            );
                        } else {
                            tracing::info!(
                                "record_sync: first measurement audio_pos={:?}, samples_played={}, raw_pos={:?}, samples_dur={:?} (channels not yet set)",
                                audio_pos,
                                h.samples_played(),
                                h.position(),
                                h.samples_played_duration(),
                            );
                        }
                    }
                }

                // Re-baseline drift measurement when it exceeds threshold.
                // Handles: (1) initial ring buffer prefill offset (~200ms at startup),
                // (2) publisher restarts / IDR drops that stall audio delivery,
                // (3) complete audio stalls (ring buffer underrun).
                // Cooldown: audio_start_time resets on each re-baseline, and we require
                // 2s since last baseline before checking again.
                if let Some(start) = self.audio_start_time {
                    let since_start = start.elapsed();
                    if since_start >= Duration::from_secs(2) {
                        let audio_delta = audio_pos.saturating_sub(self.audio_start_pos);
                        let drift_abs = since_start.abs_diff(audio_delta);
                        if drift_abs > Duration::from_millis(200) {
                            let old_pos = self.audio_start_pos;
                            self.audio_start_time = Some(std::time::Instant::now());
                            self.audio_start_pos = audio_pos;
                            self.sync_metrics.reset();
                            tracing::info!(
                                "record_sync: re-baselined drift (was {:?}, old_start={:?}, new_start={:?})",
                                drift_abs,
                                old_pos,
                                audio_pos,
                            );
                        }
                    }
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

                // 10s clock-source diagnostic: log wall-clock vs audio-clock deltas
                // to determine whether drift is from audio sample counting or video pacing.
                let now = std::time::Instant::now();
                let should_log = match self.last_clock_delta_log {
                    None => elapsed > Duration::from_secs(1),
                    Some(last) => now.duration_since(last) >= Duration::from_secs(10),
                };
                if should_log {
                    let (prev_elapsed, prev_audio) = self
                        .last_clock_delta_values
                        .unwrap_or((Duration::ZERO, Duration::ZERO));
                    let d_wall = elapsed.saturating_sub(prev_elapsed);
                    let d_audio = audio_delta.saturating_sub(prev_audio);
                    let d_wall_s = d_wall.as_secs_f64();
                    let d_audio_s = d_audio.as_secs_f64();
                    let rate = if d_wall_s > 0.0 {
                        d_audio_s / d_wall_s
                    } else {
                        1.0
                    };
                    tracing::info!(
                        "A/V clock delta (10s): wall={}ms, audio={}ms, rate={:.4} (1.0=perfect), drift_now={}ms",
                        d_wall.as_millis(),
                        d_audio.as_millis(),
                        rate,
                        elapsed.as_millis() as i64 - audio_delta.as_millis() as i64
                    );
                    self.last_clock_delta_log = Some(now);
                    self.last_clock_delta_values = Some((elapsed, audio_delta));
                }
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

        // MoQ wall-clock mode: pause audio during video stalls so they stay in sync.
        // Without this, audio continues playing through the ring buffer while the
        // video clock is frozen, creating a permanent A/V offset equal to the stall
        // duration. Clearing the epoch gates the cpal callback (outputs silence).
        if !self.use_audio_as_sync_master {
            if let Some(ref ah) = self.audio_handle {
                ah.clear_playback_epoch();
            }
        }

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

        // MoQ wall-clock mode: resume audio after stall. Re-enable the epoch
        // but preserve samples_played so audio position stays correct.
        // start_playback_epoch() would reset samples_played to 0, making
        // audio jump back to base_pts while video continues from current_position.
        if !self.use_audio_as_sync_master {
            if let Some(ref ah) = self.audio_handle {
                ah.enable_playback_epoch();
            }
        }

        tracing::debug!("Resuming from stall at {:?}", self.current_position);
    }

    /// Enters audio-induced stall (ring buffer underrun during MoQ live).
    fn enter_audio_stall(&mut self) {
        if self.audio_stalled || !self.playback_requested {
            return;
        }
        if let Some(start_time) = self.playback_start_time {
            self.current_position = self.playback_start_position + start_time.elapsed();
        }
        self.playback_start_time = None;
        self.audio_stalled = true;
        self.audio_stall_start = Some(std::time::Instant::now());
        self.sync_metrics.record_underrun();
        self.sync_metrics.record_stall(StallType::Network);
        self.sync_metrics.start_recovery();
        tracing::debug!(
            "Audio stall at {:?} (ring buffer underrun)",
            self.current_position
        );
    }

    /// Exits audio stall — called when cpal callback reports data flowing again.
    fn exit_audio_stall(&mut self) {
        self.audio_stalled = false;
        self.audio_stall_start = None;
        self.playback_start_time = Some(std::time::Instant::now());
        self.playback_start_position = self.current_position;
        self.frames_since_recovery = 0;
        tracing::debug!("Resuming from audio stall at {:?}", self.current_position);
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
    use crate::video::{CpuFrame, DecodedFrame, PixelFormat, Plane};

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

    /// Bind an AudioHandle with a specific position for drift controller tests.
    fn bind_test_audio_metrics_only(s: &mut FrameScheduler, pos: Duration) {
        let h = crate::audio::AudioHandle::new();
        h.set_available(true);
        h.set_audio_format(48_000, 2);
        h.set_audio_base_pts(Duration::ZERO);
        h.enable_playback_epoch();
        let samples = ((pos.as_secs_f64() * 48_000.0) * 2.0) as u64;
        h.add_samples_played(samples);
        s.set_audio_handle_metrics_only(h);
    }

    #[test]
    fn test_drift_controller_deadband_enter_exit() {
        let mut s = FrameScheduler::new();
        s.use_audio_as_sync_master = false;
        s.audio_start_time = Some(std::time::Instant::now() - Duration::from_secs(10));
        s.last_drift_update = Some(std::time::Instant::now() - Duration::from_secs(1));
        bind_test_audio_metrics_only(&mut s, Duration::from_secs(10));

        // Drive drift to 30ms (below enter threshold of 40ms)
        s.sync_metrics
            .record_frame(Duration::from_millis(10_030), Duration::from_millis(10_000));
        s.update_clock_drift_correction();
        // EMA starts at 0, first update: 0.0 * 0.7 + 30000.0 * 0.3 = 9000 (9ms) < 40ms
        assert_eq!(s.clock_drift_correction_us, 0);
        assert!(!s.drift_correction_active);

        // Drive drift to 80ms, enough for EMA to cross enter threshold (40ms)
        // but below step detection threshold (100ms)
        for _ in 0..20 {
            s.sync_metrics
                .record_frame(Duration::from_millis(10_080), Duration::from_millis(10_000));
            s.last_drift_update = Some(std::time::Instant::now() - Duration::from_secs(1));
            s.update_clock_drift_correction();
        }
        assert!(s.drift_correction_active);
        assert!(s.clock_drift_correction_us < 0); // video ahead → negative correction

        // Phase 2: exit — drive drift below EXIT_US (20ms)
        for _ in 0..30 {
            s.sync_metrics
                .record_frame(Duration::from_millis(10_010), Duration::from_millis(10_000));
            s.last_drift_update = Some(std::time::Instant::now() - Duration::from_secs(1));
            s.update_clock_drift_correction();
        }
        // EMA converges toward 10ms (10_000us) < EXIT_US → should deactivate
        assert!(!s.drift_correction_active);

        // Capture correction at deactivation, then verify no further changes
        let correction_at_deactivation = s.clock_drift_correction_us;
        for _ in 0..5 {
            s.sync_metrics
                .record_frame(Duration::from_millis(10_010), Duration::from_millis(10_000));
            s.last_drift_update = Some(std::time::Instant::now() - Duration::from_secs(1));
            s.update_clock_drift_correction();
        }
        assert_eq!(s.clock_drift_correction_us, correction_at_deactivation);
    }

    #[test]
    fn test_drift_controller_stall_gating() {
        let mut s = FrameScheduler::new();
        s.use_audio_as_sync_master = false;

        // Set up some existing state
        s.last_drift_update = Some(std::time::Instant::now());
        s.drift_correction_active = true;
        s.smoothed_drift_us = 50_000.0;

        s.stalled = true;
        s.update_clock_drift_correction();
        assert!(s.last_drift_update.is_none());
        assert!(!s.drift_correction_active);
        assert_eq!(s.smoothed_drift_us, 0.0);

        // Also test audio_stalled
        s.stalled = false;
        s.audio_stalled = true;
        s.last_drift_update = Some(std::time::Instant::now());
        s.drift_correction_active = true;
        s.smoothed_drift_us = 50_000.0;
        s.update_clock_drift_correction();
        assert!(s.last_drift_update.is_none());
        assert!(!s.drift_correction_active);
        assert_eq!(s.smoothed_drift_us, 0.0);
    }

    #[test]
    fn test_drift_controller_dt_cap() {
        let mut s = FrameScheduler::new();
        s.use_audio_as_sync_master = false;
        s.audio_start_time = Some(std::time::Instant::now() - Duration::from_secs(10));
        bind_test_audio_metrics_only(&mut s, Duration::from_secs(10));

        // Pre-seed controller state to active with drift below step threshold (100ms)
        // so proportional controller runs instead of step correction.
        s.smoothed_drift_us = -80_000.0; // -80ms
        s.drift_correction_active = true;
        s.sync_metrics
            .record_frame(Duration::from_millis(9_920), Duration::from_millis(10_000));

        // Set last update 5s ago (will be capped to 500ms)
        s.last_drift_update = Some(std::time::Instant::now() - Duration::from_secs(5));
        s.update_clock_drift_correction();

        // dt capped at 500ms, max slew 10ms/s → max delta per step = 5000us
        assert!(s.clock_drift_correction_us <= 5_000);
        assert!(s.clock_drift_correction_us > 0); // audio ahead → positive correction
    }

    #[test]
    fn test_drift_controller_step_detection() {
        let mut s = FrameScheduler::new();
        s.use_audio_as_sync_master = false;
        s.audio_start_time = Some(std::time::Instant::now() - Duration::from_secs(10));
        s.playback_start_time = Some(std::time::Instant::now());
        s.playback_start_position = Duration::from_millis(9_000);
        bind_test_audio_metrics_only(&mut s, Duration::from_secs(10));

        // Set drift > 100ms step threshold — should trigger immediate resync
        s.drift_correction_active = true;
        s.sync_metrics
            .record_frame(Duration::from_millis(9_000), Duration::from_millis(10_000));
        s.last_drift_update = Some(std::time::Instant::now() - Duration::from_secs(1));
        s.update_clock_drift_correction();

        // Step detection should resync and reset correction to 0
        assert_eq!(s.clock_drift_correction_us, 0);
        assert!(!s.drift_correction_active);
        // Wall clock should be resynced near audio position
        assert!(s.playback_start_position > Duration::from_millis(9_900));
    }

    #[test]
    fn test_drift_controller_total_bias_clamp() {
        let mut s = FrameScheduler::new();
        s.video_pts_bias = Duration::from_secs(7);
        s.clock_drift_correction_us = 5_000_000; // +5s

        let total = s.video_pts_bias.as_micros() as i64 + s.clock_drift_correction_us;
        let max_us = MAX_VIDEO_PTS_BIAS.as_micros() as i64;
        let clamped = total.clamp(-max_us, max_us);
        assert_eq!(clamped, max_us); // 12s clamped to 8s
    }
}
