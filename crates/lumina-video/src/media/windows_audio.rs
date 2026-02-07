//! Native Windows audio support via Media Foundation.
//!
//! This module provides:
//! - `AudioFrame`: A decoded audio frame with PCM samples and timestamp
//! - `AudioQueue`: A bounded, non-blocking queue for audio frames
//! - `QueueAudioSource`: A rodio Source that pulls from the queue
//! - `AudioClock`: Sample-based audio clock for A/V sync
//!
//! # Architecture
//!
//! Audio is decoded by the same `IMFSourceReader` that decodes video, using
//! `MF_SOURCE_READER_FIRST_AUDIO_STREAM`. The decode thread reads both streams
//! in an interleaved fashion, pushing audio frames to an `AudioQueue`.
//!
//! A `QueueAudioSource` pulls from the queue and feeds rodio/WASAPI.
//! The `AudioClock` tracks samples sent for accurate A/V synchronization
//! (more reliable than rodio's `Sink::get_pos()`).

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Audio Frame
// ============================================================================

/// A decoded audio frame containing PCM samples.
#[derive(Clone)]
pub struct AudioFrame {
    /// Presentation timestamp from Media Foundation (converted from 100ns units)
    pub pts: Duration,
    /// Interleaved PCM samples (i16)
    pub data: Vec<i16>,
    /// Number of audio channels
    pub channels: u16,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl AudioFrame {
    /// Creates a new audio frame.
    pub fn new(pts: Duration, data: Vec<i16>, channels: u16, sample_rate: u32) -> Self {
        Self {
            pts,
            data,
            channels,
            sample_rate,
        }
    }

    /// Returns the number of samples per channel in this frame.
    pub fn samples_per_channel(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.data.len() / self.channels as usize
        }
    }

    /// Returns the duration of this frame.
    pub fn duration(&self) -> Duration {
        if self.sample_rate == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(self.samples_per_channel() as f64 / self.sample_rate as f64)
        }
    }
}

// ============================================================================
// Audio Format Info
// ============================================================================

/// Audio format information resolved from Media Foundation.
#[derive(Debug, Clone)]
pub struct AudioFormatInfo {
    /// Sample rate in Hz (e.g., 48000)
    pub sample_rate: u32,
    /// Number of channels (e.g., 2 for stereo)
    pub channels: u16,
    /// Bits per sample (e.g., 16)
    pub bits_per_sample: u16,
    /// Bytes per audio frame (channels * bits_per_sample / 8)
    pub block_align: u16,
    /// Average bytes per second (sample_rate * block_align)
    pub avg_bytes_per_sec: u32,
    /// Whether the audio data is 32-bit float (f32) vs integer PCM.
    pub is_float: bool,
}

impl Default for AudioFormatInfo {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 16,
            block_align: 4, // 2 channels * 16 bits / 8
            avg_bytes_per_sec: 48000 * 4,
            is_float: false,
        }
    }
}

// ============================================================================
// Audio Queue (bounded, non-blocking)
// ============================================================================

/// A bounded, non-blocking queue for audio frames.
///
/// This queue is designed to prevent the decode thread from blocking:
/// - `push()` returns `false` if the queue is full (frame is dropped)
/// - `pop()` returns `None` if the queue is empty
///
/// This prevents backpressure from stalling video decode.
pub struct AudioQueue {
    /// Maximum number of frames to buffer
    max_frames: usize,
    /// The frame queue
    queue: Mutex<VecDeque<AudioFrame>>,
    /// Total samples pushed (for statistics)
    total_pushed: AtomicU64,
    /// Total samples dropped due to full queue
    total_dropped: AtomicU64,
}

impl AudioQueue {
    /// Creates a new audio queue with the specified capacity.
    pub fn new(max_frames: usize) -> Arc<Self> {
        Arc::new(Self {
            max_frames,
            queue: Mutex::new(VecDeque::with_capacity(max_frames)),
            total_pushed: AtomicU64::new(0),
            total_dropped: AtomicU64::new(0),
        })
    }

    /// Pushes a frame to the queue. Returns `false` if the queue is full.
    ///
    /// This method never blocks. If the queue is full, the frame is dropped
    /// and `false` is returned.
    pub fn push(&self, frame: AudioFrame) -> bool {
        let samples = frame.data.len() as u64;
        let mut queue = self.queue.lock();

        if queue.len() >= self.max_frames {
            // Queue full - drop frame to prevent blocking
            self.total_dropped.fetch_add(samples, Ordering::Relaxed);
            return false;
        }

        queue.push_back(frame);
        self.total_pushed.fetch_add(samples, Ordering::Relaxed);
        true
    }

    /// Pops a frame from the queue. Returns `None` if empty.
    pub fn pop(&self) -> Option<AudioFrame> {
        let mut queue = self.queue.lock();
        queue.pop_front()
    }

    /// Returns the number of frames currently in the queue.
    pub fn len(&self) -> usize {
        self.queue.lock().len()
    }

    /// Returns true if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all frames from the queue (used on seek).
    pub fn clear(&self) {
        let mut queue = self.queue.lock();
        queue.clear();
    }

    /// Returns the total number of samples pushed.
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed.load(Ordering::Relaxed)
    }

    /// Returns the total number of samples dropped due to full queue.
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Audio Clock (sample-based, for A/V sync)
// ============================================================================

/// Audio clock based on samples sent to the output device.
///
/// This provides a more reliable clock than `rodio::Sink::get_pos()`,
/// which can drift on Windows due to buffering.
///
/// The clock tracks:
/// - `samples_sent`: Total samples sent to rodio
/// - `output_latency`: Estimated latency of the audio output path
///
/// The effective audio position is:
/// ```text
/// position = (samples_sent / sample_rate) - output_latency
/// ```
pub struct AudioClock {
    /// Total samples sent to output
    samples_sent: AtomicU64,
    /// Sample rate in Hz
    sample_rate: u32,
    /// Estimated output latency
    output_latency: Duration,
}

impl AudioClock {
    /// Default output latency estimate (50ms is reasonable for WASAPI)
    pub const DEFAULT_OUTPUT_LATENCY: Duration = Duration::from_millis(50);

    /// Creates a new audio clock.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            samples_sent: AtomicU64::new(0),
            sample_rate,
            output_latency: Self::DEFAULT_OUTPUT_LATENCY,
        }
    }

    /// Creates a new audio clock with custom output latency.
    pub fn with_latency(sample_rate: u32, output_latency: Duration) -> Self {
        Self {
            samples_sent: AtomicU64::new(0),
            sample_rate,
            output_latency,
        }
    }

    /// Adds samples to the clock (called when samples are sent to rodio).
    pub fn add_samples(&self, count: u64) {
        self.samples_sent.fetch_add(count, Ordering::Relaxed);
    }

    /// Returns the current audio position, accounting for output latency.
    pub fn position(&self) -> Duration {
        let samples = self.samples_sent.load(Ordering::Relaxed);
        let raw_position = Duration::from_secs_f64(samples as f64 / self.sample_rate as f64);

        // Subtract output latency, but don't go negative
        raw_position.saturating_sub(self.output_latency)
    }

    /// Returns the raw position without latency adjustment.
    pub fn raw_position(&self) -> Duration {
        let samples = self.samples_sent.load(Ordering::Relaxed);
        Duration::from_secs_f64(samples as f64 / self.sample_rate as f64)
    }

    /// Resets the clock (called on seek).
    pub fn reset(&self) {
        self.samples_sent.store(0, Ordering::Relaxed);
    }

    /// Returns the total samples sent.
    pub fn samples_sent(&self) -> u64 {
        self.samples_sent.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Queue Audio Source (rodio Source implementation)
// ============================================================================

/// A rodio `Source` that pulls audio from an `AudioQueue`.
///
/// This bridges the decode thread (which pushes to the queue) with
/// rodio's playback thread (which pulls via this source).
///
/// Handles PTS gaps by inserting silence when a frame's PTS is ahead
/// of the expected clock position (prevents desync on stream ticks/seeks).
pub struct QueueAudioSource {
    /// The audio queue to pull from
    queue: Arc<AudioQueue>,
    /// Audio clock to update when samples are consumed
    clock: Arc<AudioClock>,
    /// Current frame being consumed
    current_frame: Option<AudioFrame>,
    /// Position within current frame
    frame_pos: usize,
    /// Number of channels
    channels: u16,
    /// Sample rate
    sample_rate: u32,
    /// Silence samples remaining to insert for PTS gap
    silence_samples: u64,
}

impl QueueAudioSource {
    /// Creates a new queue audio source.
    pub fn new(
        queue: Arc<AudioQueue>,
        clock: Arc<AudioClock>,
        channels: u16,
        sample_rate: u32,
    ) -> Self {
        Self {
            queue,
            clock,
            current_frame: None,
            frame_pos: 0,
            channels,
            sample_rate,
            silence_samples: 0,
        }
    }

    /// Calculates silence samples needed when frame PTS is ahead of clock.
    fn calculate_pts_gap(&self, frame_pts: Duration) -> u64 {
        let clock_pos = self.clock.raw_position();

        // Only insert silence if frame is significantly ahead (>10ms threshold)
        let threshold = Duration::from_millis(10);
        if frame_pts > clock_pos + threshold {
            let gap = frame_pts - clock_pos;
            let gap_samples = (gap.as_secs_f64() * self.sample_rate as f64) as u64;
            // Multiply by channels since we output interleaved samples
            gap_samples * self.channels as u64
        } else {
            0
        }
    }
}

impl Iterator for QueueAudioSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        // First, emit any pending silence samples for PTS gap
        if self.silence_samples > 0 {
            self.silence_samples -= 1;
            self.clock.add_samples(1);
            return Some(0.0);
        }

        // Try to get sample from current frame
        if let Some(ref frame) = self.current_frame {
            if self.frame_pos < frame.data.len() {
                let sample = frame.data[self.frame_pos] as f32 / 32768.0;
                self.frame_pos += 1;
                self.clock.add_samples(1);
                return Some(sample);
            }
        }

        // Current frame exhausted, try to get next frame
        self.current_frame = self.queue.pop();
        self.frame_pos = 0;

        if let Some(ref frame) = self.current_frame {
            // Calculate PTS gap before playing this frame
            self.silence_samples = self.calculate_pts_gap(frame.pts);

            // If we need to insert silence, do it before the frame's samples
            if self.silence_samples > 0 {
                self.silence_samples -= 1;
                self.clock.add_samples(1);
                return Some(0.0);
            }

            if !frame.data.is_empty() {
                let sample = frame.data[0] as f32 / 32768.0;
                self.frame_pos = 1;
                self.clock.add_samples(1);
                return Some(sample);
            }
        }

        // No more samples available - return silence to keep stream alive
        // This prevents rodio from ending the stream during buffer underruns
        // Advance the clock even for silence so playback time keeps moving
        self.clock.add_samples(1);
        Some(0.0)
    }
}

impl rodio::Source for QueueAudioSource {
    fn current_span_len(&self) -> Option<usize> {
        // We don't know the total length
        None
    }

    fn channels(&self) -> u16 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        // Streaming source, unknown duration
        None
    }
}

// ============================================================================
// Windows Audio Playback (rodio integration)
// ============================================================================

/// Windows audio playback manager.
///
/// This integrates the AudioQueue with rodio for playback:
/// - Creates a rodio Sink with a QueueAudioSource
/// - Manages the AudioClock for A/V sync
/// - Provides play/pause/seek controls
///
/// # Usage
///
/// 1. Create with `WindowsAudioPlayback::new()`
/// 2. Pass `audio_queue()` to the decode thread
/// 3. Decode thread pushes AudioFrames to the queue
/// 4. Call `play()`/`pause()`/`seek()` as needed
/// 5. Use `clock_position()` for A/V sync
pub struct WindowsAudioPlayback {
    /// The audio queue that receives decoded frames
    queue: Arc<AudioQueue>,
    /// Sample-based audio clock for A/V sync
    clock: Arc<AudioClock>,
    /// The rodio output stream (must keep alive)
    _stream: rodio::OutputStream,
    /// The rodio sink for playback control
    sink: rodio::Sink,
    /// Audio format info
    format: AudioFormatInfo,
    /// Whether playback is active
    playing: bool,
}

impl WindowsAudioPlayback {
    /// Maximum audio frames to buffer (about 500ms at typical frame sizes)
    const MAX_AUDIO_FRAMES: usize = 50;

    /// Creates a new Windows audio playback manager with external queue and clock.
    ///
    /// This is the preferred constructor - it allows the decode thread to share
    /// the same queue/clock so audio frames actually reach playback.
    ///
    /// # Arguments
    /// * `format` - Audio format info from the decoder
    /// * `queue` - The audio queue (shared with decode thread)
    /// * `clock` - The audio clock (for A/V sync)
    ///
    /// # Returns
    /// `Ok(WindowsAudioPlayback)` on success, `Err` if audio device unavailable.
    pub fn new_with_queue(
        format: AudioFormatInfo,
        queue: Arc<AudioQueue>,
        clock: Arc<AudioClock>,
    ) -> Result<Self, String> {
        let stream = rodio::OutputStreamBuilder::open_default_stream()
            .map_err(|e| format!("Failed to open audio output: {}", e))?;

        let sink = rodio::Sink::connect_new(stream.mixer());

        let source = QueueAudioSource::new(
            Arc::clone(&queue),
            Arc::clone(&clock),
            format.channels,
            format.sample_rate,
        );

        sink.append(source);
        sink.pause(); // Start paused

        Ok(Self {
            queue,
            clock,
            _stream: stream,
            sink,
            format,
            playing: false,
        })
    }

    /// Creates a new Windows audio playback manager (creates its own queue/clock).
    ///
    /// **Warning**: This creates a disconnected queue. Use `new_with_queue` instead
    /// when you need to share the queue with a decode thread.
    #[allow(dead_code)]
    pub fn new(format: AudioFormatInfo) -> Result<Self, String> {
        let queue = AudioQueue::new(Self::MAX_AUDIO_FRAMES);
        let clock = Arc::new(AudioClock::new(format.sample_rate));
        Self::new_with_queue(format, queue, clock)
    }

    /// Returns the audio queue for the decode thread to push frames to.
    pub fn audio_queue(&self) -> Arc<AudioQueue> {
        Arc::clone(&self.queue)
    }

    /// Returns the audio clock for A/V synchronization.
    pub fn audio_clock(&self) -> Arc<AudioClock> {
        Arc::clone(&self.clock)
    }

    /// Starts or resumes audio playback.
    pub fn play(&mut self) {
        self.sink.play();
        self.playing = true;
    }

    /// Pauses audio playback.
    pub fn pause(&mut self) {
        self.sink.pause();
        self.playing = false;
    }

    /// Handles seek: clears the queue and resets the clock.
    ///
    /// The decode thread should call this before seeking the decoder,
    /// then start pushing new frames at the seek position.
    pub fn seek(&mut self) {
        self.queue.clear();
        self.clock.reset();
    }

    /// Returns the current audio playback position (accounting for latency).
    ///
    /// This is the position that video should sync to.
    pub fn clock_position(&self) -> Duration {
        self.clock.position()
    }

    /// Returns true if audio is playing.
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Returns the audio format info.
    pub fn format(&self) -> &AudioFormatInfo {
        &self.format
    }

    /// Sets the volume (0.0 to 1.0).
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(volume.clamp(0.0, 1.0));
    }

    /// Returns the current volume (0.0 to 1.0).
    pub fn volume(&self) -> f32 {
        self.sink.volume()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_frame_duration() {
        let frame = AudioFrame::new(
            Duration::ZERO,
            vec![0i16; 4800], // 2400 samples per channel at stereo
            2,
            48000,
        );
        assert_eq!(frame.samples_per_channel(), 2400);
        assert_eq!(frame.duration(), Duration::from_millis(50));
    }

    #[test]
    fn test_audio_queue_push_pop() {
        let queue = AudioQueue::new(10);
        let frame = AudioFrame::new(Duration::ZERO, vec![1, 2, 3, 4], 2, 48000);

        assert!(queue.push(frame.clone()));
        assert_eq!(queue.len(), 1);

        let popped = queue.pop().unwrap();
        assert_eq!(popped.data, vec![1, 2, 3, 4]);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_audio_queue_bounded() {
        let queue = AudioQueue::new(2);

        // Fill the queue
        assert!(queue.push(AudioFrame::new(Duration::ZERO, vec![1], 1, 48000)));
        assert!(queue.push(AudioFrame::new(Duration::ZERO, vec![2], 1, 48000)));

        // Queue is full - should return false (not block)
        assert!(!queue.push(AudioFrame::new(Duration::ZERO, vec![3], 1, 48000)));
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_audio_clock() {
        let clock = AudioClock::new(48000);

        // Add 48000 samples = 1 second
        clock.add_samples(48000);

        // Raw position should be 1 second
        assert_eq!(clock.raw_position(), Duration::from_secs(1));

        // Adjusted position should be 1s - 50ms = 950ms
        assert_eq!(clock.position(), Duration::from_millis(950));
    }

    #[test]
    fn test_audio_clock_reset() {
        let clock = AudioClock::new(48000);
        clock.add_samples(48000);
        assert_eq!(clock.samples_sent(), 48000);

        clock.reset();
        assert_eq!(clock.samples_sent(), 0);
    }
}
