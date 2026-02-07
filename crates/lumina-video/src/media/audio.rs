//! Audio playback for video player.
//!
//! This module handles audio decoding and playback for video files.
//! It provides A/V synchronization, volume control, and mute toggle.
//!
//! # Architecture
//!
//! The audio system consists of:
//! - `AudioDecoder`: Extracts audio frames from video stream via FFmpeg
//! - `AudioPlayer`: Handles audio output via rodio
//! - `AudioSync`: Synchronizes audio playback with video presentation
//!
//! Audio serves as the master clock for A/V sync - video frames are
//! presented relative to the audio playback position.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Audio playback state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioState {
    /// Audio is not initialized
    Uninitialized,
    /// Audio is playing
    Playing,
    /// Audio is paused
    Paused,
    /// Audio playback error
    Error,
}

/// Configuration for audio playback.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Buffer size in samples
    pub buffer_size: u32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            buffer_size: 1024,
        }
    }
}

/// Audio player handle for volume and mute control.
///
/// This is a lightweight handle that can be cloned and shared
/// between the video player and UI controls.
#[derive(Clone)]
pub struct AudioHandle {
    inner: Arc<AudioHandleInner>,
}

struct AudioHandleInner {
    /// Volume level (0-100)
    volume: AtomicU32,
    /// Whether audio is muted
    muted: AtomicBool,
    /// Whether audio is available for this video
    available: AtomicBool,
    /// Shared playback epoch (microseconds since UNIX epoch, 0 = not started)
    /// Set by video system when first frame is displayed, used by audio for position calculation
    playback_epoch_us: AtomicU64,
    /// Total samples played (consumed by audio callback) - for accurate position tracking
    samples_played: AtomicU64,
    /// Sample rate for converting samples to time
    sample_rate: AtomicU32,
    /// Number of channels (for sample-to-frame conversion)
    channels: AtomicU32,
    /// Base PTS of audio stream (first queued sample's PTS) + 1, for position calculation.
    /// Stored as value+1 so that PTS=0 is distinguishable from "unset" (which is 0).
    audio_base_pts_us_plus1: AtomicU64,
    /// Stream PTS offset in microseconds: (audio_start_time - video_start_time).
    /// Used to normalize audio position when comparing against video PTS.
    /// Positive = audio stream starts after video, negative = audio starts before video.
    stream_pts_offset_us: AtomicI64,
    /// Video stream start time (stored as us+1, 0 = not set).
    /// Set by video player after video decoder init.
    video_start_time_us_plus1: AtomicU64,
    /// Direct position override in microseconds (for native platforms like macOS/GStreamer).
    /// When non-zero, position() returns this value instead of calculating from samples.
    /// This allows native players (AVPlayer, GStreamer) to directly report their position.
    native_position_us: AtomicU64,
}

impl AudioHandle {
    /// Creates a new audio handle.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AudioHandleInner {
                volume: AtomicU32::new(100),
                muted: AtomicBool::new(false),
                available: AtomicBool::new(false),
                playback_epoch_us: AtomicU64::new(0),
                samples_played: AtomicU64::new(0),
                sample_rate: AtomicU32::new(48000), // Default, updated when audio starts
                channels: AtomicU32::new(2),        // Default stereo
                audio_base_pts_us_plus1: AtomicU64::new(0),
                stream_pts_offset_us: AtomicI64::new(0),
                video_start_time_us_plus1: AtomicU64::new(0),
                native_position_us: AtomicU64::new(0),
            }),
        }
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.inner.volume.load(Ordering::Relaxed)
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&self, volume: u32) {
        self.inner.volume.store(volume.min(100), Ordering::Relaxed);
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.inner.muted.load(Ordering::Relaxed)
    }

    /// Sets the mute state.
    pub fn set_muted(&self, muted: bool) {
        self.inner.muted.store(muted, Ordering::Relaxed);
    }

    /// Toggles the mute state.
    pub fn toggle_mute(&self) {
        // Use fetch_xor for atomic toggle to avoid TOCTOU race condition
        self.inner.muted.fetch_xor(true, Ordering::Relaxed);
    }

    /// Returns the effective volume (0.0-1.0) accounting for mute.
    pub fn effective_volume(&self) -> f32 {
        if self.is_muted() {
            0.0
        } else {
            self.volume() as f32 / 100.0
        }
    }

    /// Returns the current playback position.
    ///
    /// For native platforms (macOS AVPlayer, GStreamer), uses directly set position.
    /// For FFmpeg+rodio path, computes as: base_pts + samples_played_duration.
    /// Returns Duration::ZERO if playback epoch hasn't started (video not ready)
    /// or if base PTS hasn't been set yet.
    pub fn position(&self) -> Duration {
        // Gate on playback epoch - don't advance until video starts
        if self.inner.playback_epoch_us.load(Ordering::Acquire) == 0 {
            return Duration::ZERO;
        }

        // For native platforms, use directly set position if available
        let native_pos = self.inner.native_position_us.load(Ordering::Relaxed);
        if native_pos > 0 {
            return Duration::from_micros(native_pos);
        }

        // base_pts stored as value+1, so 0 means "unset" and PTS=0 works correctly
        let base_plus1 = self.inner.audio_base_pts_us_plus1.load(Ordering::Relaxed);
        if base_plus1 == 0 {
            return Duration::ZERO;
        }

        let base = Duration::from_micros(base_plus1 - 1);
        base + self.samples_played_duration()
    }

    /// Returns true if position is coming from native player (AVPlayer, GStreamer).
    ///
    /// Native players set position directly via set_native_position().
    /// FFmpeg audio calculates position from samples_played.
    pub fn is_using_native_position(&self) -> bool {
        self.inner.native_position_us.load(Ordering::Relaxed) > 0
    }

    /// Sets the audio position directly (for native platforms like macOS/GStreamer).
    ///
    /// Native video players handle audio internally, so we can't track samples played.
    /// Instead, the video player updates this position based on its internal clock.
    /// For AVPlayer, this should be set from video frame PTS since AVPlayer keeps A/V in sync.
    pub fn set_native_position(&self, position: Duration) {
        self.inner
            .native_position_us
            .store(position.as_micros() as u64, Ordering::Relaxed);
    }

    /// Clears the native position (for seek/stop).
    pub fn clear_native_position(&self) {
        self.inner.native_position_us.store(0, Ordering::Relaxed);
    }

    /// Sets the base PTS for audio position calculation (first sample's PTS).
    pub fn set_audio_base_pts(&self, pts: Duration) {
        let us = pts.as_micros() as u64;
        self.inner
            .audio_base_pts_us_plus1
            .store(us.saturating_add(1), Ordering::Relaxed);
    }

    /// Clears the base PTS (for seek/stop).
    pub fn clear_audio_base_pts(&self) {
        self.inner
            .audio_base_pts_us_plus1
            .store(0, Ordering::Relaxed);
    }

    /// Sets the stream PTS offset (audio_start_time - video_start_time).
    ///
    /// This offset is used to normalize audio position when comparing against video PTS
    /// for A/V sync calculation. Call this after opening both audio and video streams.
    ///
    /// # Arguments
    /// * `audio_start` - Start time of the audio stream (first PTS)
    /// * `video_start` - Start time of the video stream (first PTS)
    pub fn set_stream_pts_offset(
        &self,
        audio_start: Option<Duration>,
        video_start: Option<Duration>,
    ) {
        let offset_us = match (audio_start, video_start) {
            (Some(a), Some(v)) => a.as_micros() as i64 - v.as_micros() as i64,
            _ => 0, // No offset if either stream lacks start time
        };
        self.inner
            .stream_pts_offset_us
            .store(offset_us, Ordering::Relaxed);

        if offset_us != 0 {
            tracing::info!(
                "Stream PTS offset: {}ms (audio_start={:?}, video_start={:?})",
                offset_us / 1000,
                audio_start,
                video_start
            );
        }
    }

    /// Returns the stream PTS offset in microseconds.
    pub fn stream_pts_offset_us(&self) -> i64 {
        self.inner.stream_pts_offset_us.load(Ordering::Relaxed)
    }

    /// Returns the audio position normalized to the video timeline.
    ///
    /// This adjusts the raw audio position by subtracting the stream PTS offset,
    /// making it directly comparable to video PTS for drift calculation.
    /// Use this for A/V sync comparison instead of raw `position()`.
    pub fn position_for_sync(&self) -> Duration {
        let raw_pos = self.position();
        let offset_us = self.inner.stream_pts_offset_us.load(Ordering::Relaxed);

        if offset_us == 0 {
            return raw_pos;
        }

        // Normalize: audio_position - offset = position_in_video_timeline
        let raw_us = raw_pos.as_micros() as i64;
        let normalized_us = raw_us - offset_us;
        Duration::from_micros(normalized_us.max(0) as u64)
    }

    /// Clears the stream PTS offset (for new video).
    pub fn clear_stream_pts_offset(&self) {
        self.inner.stream_pts_offset_us.store(0, Ordering::Relaxed);
    }

    /// Sets the video stream start time (call from video player after decoder init).
    ///
    /// This is stored so the audio thread can compute the PTS offset when it
    /// has the audio start time.
    pub fn set_video_start_time(&self, start_time: Option<Duration>) {
        let us_plus1 = start_time.map(|d| d.as_micros() as u64 + 1).unwrap_or(0);
        self.inner
            .video_start_time_us_plus1
            .store(us_plus1, Ordering::Relaxed);
    }

    /// Finalizes the stream PTS offset using the stored video start time.
    ///
    /// Call this from the audio thread after the audio decoder is initialized.
    /// This computes offset = audio_start - video_start.
    pub fn finalize_stream_pts_offset(&self, audio_start: Option<Duration>) {
        let video_us_plus1 = self.inner.video_start_time_us_plus1.load(Ordering::Relaxed);
        let video_start = if video_us_plus1 > 0 {
            Some(Duration::from_micros(video_us_plus1 - 1))
        } else {
            None
        };

        self.set_stream_pts_offset(audio_start, video_start);
    }

    /// Returns whether audio is available for this video.
    pub fn is_available(&self) -> bool {
        self.inner.available.load(Ordering::Relaxed)
    }

    /// Sets whether audio is available (internal use).
    pub fn set_available(&self, available: bool) {
        self.inner.available.store(available, Ordering::Relaxed);
    }

    /// Starts the shared playback epoch (call when first video frame is displayed).
    /// This coordinates audio and video clocks so they start from the same moment.
    ///
    /// Resets samples_played to 0 so we only count samples from this point forward,
    /// avoiding drift from samples played while waiting for video's first frame.
    pub fn start_playback_epoch(&self) {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Reset sample counter - only count samples from epoch start
        self.inner.samples_played.store(0, Ordering::Release);

        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.inner
            .playback_epoch_us
            .store(now_us, Ordering::Release);
    }

    /// Returns the playback epoch as an Instant-like duration since UNIX epoch.
    /// Returns None if epoch hasn't been set yet.
    pub fn playback_epoch(&self) -> Option<u64> {
        let epoch_us = self.inner.playback_epoch_us.load(Ordering::Acquire);
        if epoch_us == 0 {
            None
        } else {
            Some(epoch_us)
        }
    }

    /// Clears the playback epoch (for seek/stop).
    pub fn clear_playback_epoch(&self) {
        self.inner.playback_epoch_us.store(0, Ordering::Release);
    }

    // ========================================================================
    // Sample-based position tracking (callback-accurate)
    // ========================================================================

    /// Sets the audio format for sample-to-time conversion.
    pub fn set_audio_format(&self, sample_rate: u32, channels: u32) {
        self.inner.sample_rate.store(sample_rate, Ordering::Relaxed);
        self.inner.channels.store(channels, Ordering::Relaxed);
    }

    /// Increments the samples-played counter (called by audio callback).
    /// This should be called for each sample consumed by the audio device.
    #[inline]
    pub fn add_samples_played(&self, samples: u64) {
        self.inner
            .samples_played
            .fetch_add(samples, Ordering::Relaxed);
    }

    /// Returns the total samples played since start/seek.
    pub fn samples_played(&self) -> u64 {
        self.inner.samples_played.load(Ordering::Relaxed)
    }

    /// Resets the samples-played counter (for seek/stop).
    pub fn reset_samples_played(&self) {
        self.inner.samples_played.store(0, Ordering::Relaxed);
    }

    /// Returns playback duration based on samples played (more accurate than wall-clock).
    pub fn samples_played_duration(&self) -> Duration {
        let samples = self.inner.samples_played.load(Ordering::Relaxed);
        let sample_rate = self.inner.sample_rate.load(Ordering::Relaxed) as u64;
        let channels = self.inner.channels.load(Ordering::Relaxed) as u64;

        if sample_rate == 0 || channels == 0 {
            return Duration::ZERO;
        }

        // samples is total individual samples, divide by channels for frames
        let frames = samples / channels;
        // Convert frames to microseconds: frames * 1_000_000 / sample_rate
        let us = frames * 1_000_000 / sample_rate;
        Duration::from_micros(us)
    }
}

impl Default for AudioHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio synchronization helper.
///
/// Uses audio playback position as the master clock for video frame timing.
/// When audio is not available, falls back to wall-clock time.
pub struct AudioSync {
    /// Audio handle for getting playback position
    audio: AudioHandle,
    /// Whether to use audio as master clock
    use_audio_clock: bool,
    /// Fallback start time when audio is not available
    fallback_start: std::time::Instant,
}

impl AudioSync {
    /// Creates a new audio sync helper.
    pub fn new(audio: AudioHandle) -> Self {
        Self {
            audio,
            use_audio_clock: true,
            fallback_start: std::time::Instant::now(),
        }
    }

    /// Returns the current playback position for frame timing.
    pub fn position(&self) -> Duration {
        if self.use_audio_clock && self.audio.is_available() {
            self.audio.position()
        } else {
            // Fallback to wall-clock time from start
            self.fallback_start.elapsed()
        }
    }

    /// Sets whether to use audio as the master clock.
    pub fn set_use_audio_clock(&mut self, use_audio: bool) {
        self.use_audio_clock = use_audio;
    }

    /// Returns whether audio clock is being used.
    pub fn using_audio_clock(&self) -> bool {
        self.use_audio_clock && self.audio.is_available()
    }
}

/// Decoded audio samples ready for playback.
#[derive(Clone)]
pub struct AudioSamples {
    /// Interleaved samples (f32, -1.0 to 1.0)
    pub data: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Presentation timestamp
    pub pts: Duration,
}

impl std::fmt::Debug for AudioSamples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioSamples")
            .field("data_len", &self.data.len())
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("pts", &self.pts)
            .finish()
    }
}

// ============================================================================
// Rodio-based audio player (when ffmpeg feature is enabled)
// ============================================================================

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
mod rodio_impl {
    use super::*;
    use parking_lot::Mutex;
    use rodio::Source;
    use rodio::{buffer::SamplesBuffer, OutputStream, OutputStreamBuilder, Sink};

    // ========================================================================
    // Sample-counting source wrapper for accurate position tracking
    // ========================================================================

    /// Number of samples to accumulate before flushing to atomic counter.
    /// Batching avoids per-sample atomic ops which can cause glitches at 48-96kHz.
    const FLUSH_SAMPLES: u64 = 256;

    /// A Source wrapper that counts samples as they're consumed by the audio device.
    /// This provides accurate playback position tracking by counting actual samples
    /// that have been sent to the DAC, rather than relying on wall-clock time.
    ///
    /// Uses batched counting (flushes every FLUSH_SAMPLES) to avoid per-sample
    /// atomic operations which can cause audio glitches on the real-time thread.
    struct SampleCountingSource<S> {
        inner: S,
        handle: AudioHandle,
        /// Pending samples not yet flushed to atomic counter
        pending: u64,
    }

    impl<S> SampleCountingSource<S> {
        fn new(source: S, handle: AudioHandle) -> Self {
            Self {
                inner: source,
                handle,
                pending: 0,
            }
        }

        /// Flush pending samples to the atomic counter if threshold reached.
        #[inline]
        fn flush_if_needed(&mut self) {
            if self.pending >= FLUSH_SAMPLES {
                let n = self.pending;
                self.pending = 0;
                self.handle.add_samples_played(n);
            }
        }
    }

    impl<S: Source<Item = f32>> Iterator for SampleCountingSource<S> {
        type Item = f32;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            let sample = self.inner.next();
            if sample.is_some() {
                self.pending += 1;
                self.flush_if_needed();
            } else if self.pending != 0 {
                // Source exhausted - flush remaining samples
                let n = self.pending;
                self.pending = 0;
                self.handle.add_samples_played(n);
            }
            sample
        }

        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    impl<S: Source<Item = f32>> Source for SampleCountingSource<S> {
        #[inline]
        fn current_span_len(&self) -> Option<usize> {
            self.inner.current_span_len()
        }

        #[inline]
        fn channels(&self) -> u16 {
            self.inner.channels()
        }

        #[inline]
        fn sample_rate(&self) -> u32 {
            self.inner.sample_rate()
        }

        #[inline]
        fn total_duration(&self) -> Option<Duration> {
            self.inner.total_duration()
        }
    }

    impl<S> Drop for SampleCountingSource<S> {
        fn drop(&mut self) {
            // Flush any remaining samples on drop
            if self.pending != 0 {
                self.handle.add_samples_played(self.pending);
                self.pending = 0;
            }
        }
    }

    // ========================================================================
    // AudioPlayer implementation
    // ========================================================================

    /// Rodio-based audio player.
    pub struct AudioPlayer {
        /// Audio handle for control
        handle: AudioHandle,
        /// Rodio output stream (must be kept alive)
        _stream: OutputStream,
        /// Rodio sink for playback control
        sink: Arc<Mutex<Sink>>,
        /// Current state
        state: AudioState,
        /// Device sample rate
        device_sample_rate: u32,
        /// PTS of first audio sample queued (for position calculation)
        initial_pts: Option<Duration>,
    }

    impl AudioPlayer {
        /// Creates a new audio player.
        ///
        /// If `external_handle` is provided, the player will use it for volume/mute control.
        /// Otherwise, it creates its own handle.
        pub fn new_with_handle(
            _config: AudioConfig,
            external_handle: Option<AudioHandle>,
        ) -> Result<Self, String> {
            // Create audio output stream (rodio 0.21 API)
            // On Linux, set explicit ALSA buffer size to avoid extreme default latency (cpal#446).
            // macOS CoreAudio (512 frames) and Android Oboe have reasonable defaults.
            #[cfg(target_os = "linux")]
            let stream = OutputStreamBuilder::from_default_device()
                .map(|b| b.with_buffer_size(rodio::cpal::BufferSize::Fixed(1024)))
                .and_then(|b| b.open_stream_or_fallback())
                .or_else(|e| {
                    tracing::warn!("Audio: explicit buffer setup failed ({e}), trying default");
                    OutputStreamBuilder::open_default_stream()
                })
                .map_err(|e| format!("Failed to create audio output: {e}"))?;

            #[cfg(not(target_os = "linux"))]
            let stream = OutputStreamBuilder::open_default_stream()
                .map_err(|e| format!("Failed to create audio output: {e}"))?;

            // Get the device sample rate from the stream config
            let device_sample_rate = stream.config().sample_rate();

            tracing::info!("Audio device sample rate: {}Hz", device_sample_rate);

            // Use external handle or create our own
            let handle = external_handle.unwrap_or_default();
            handle.set_available(true);

            // Create sink connected to the stream's mixer (rodio 0.21 API)
            let sink = Sink::connect_new(stream.mixer());
            sink.pause(); // Start paused

            tracing::info!(
                "Audio player initialized at {}Hz stereo",
                device_sample_rate
            );

            Ok(Self {
                handle,
                _stream: stream,
                sink: Arc::new(Mutex::new(sink)),
                state: AudioState::Paused,
                device_sample_rate,
                initial_pts: None,
            })
        }

        /// Creates a new audio player with its own handle.
        pub fn new(config: AudioConfig) -> Result<Self, String> {
            Self::new_with_handle(config, None)
        }

        /// Returns the device sample rate.
        pub fn device_sample_rate(&self) -> u32 {
            self.device_sample_rate
        }

        /// Returns the audio handle for control.
        pub fn handle(&self) -> AudioHandle {
            self.handle.clone()
        }

        /// Queues audio samples for playback using SamplesBuffer.
        pub fn queue_samples(&mut self, samples: AudioSamples) {
            if samples.data.is_empty() {
                return;
            }

            // Track initial PTS and set audio format for position calculation
            if self.initial_pts.is_none() {
                self.initial_pts = Some(samples.pts);
                // Set audio format and base PTS for position calculation
                self.handle
                    .set_audio_format(samples.sample_rate, samples.channels as u32);
                self.handle.set_audio_base_pts(samples.pts);
                tracing::debug!(
                    "Audio: first samples queued, base_pts={:?}, {} samples, {}Hz {}ch",
                    samples.pts,
                    samples.data.len(),
                    samples.sample_rate,
                    samples.channels
                );
            }

            // Create a SamplesBuffer wrapped with sample counting for accurate position
            let buffer =
                SamplesBuffer::new(samples.channels, samples.sample_rate, samples.data.clone());
            let counting_source = SampleCountingSource::new(buffer, self.handle.clone());

            // Append to sink and update volume dynamically
            let sink = self.sink.lock();
            // Apply current volume/mute state to sink (dynamic, affects all queued audio)
            sink.set_volume(self.handle.effective_volume());
            sink.append(counting_source);
            // Position is now computed on-demand in AudioHandle::position()
        }

        /// Starts audio playback.
        pub fn play(&mut self) {
            let sink = self.sink.lock();
            sink.play();
            self.state = AudioState::Playing;
        }

        /// Pauses audio playback.
        pub fn pause(&mut self) {
            let sink = self.sink.lock();
            sink.pause();
            self.state = AudioState::Paused;
        }

        /// Returns the current state.
        pub fn state(&self) -> AudioState {
            self.state
        }

        /// Clears the audio buffer (for seeking).
        ///
        /// Note: After clear(), the sink may be paused. Caller should call play()
        /// if playback should continue (e.g., during seek).
        pub fn clear(&mut self) {
            let sink = self.sink.lock();
            sink.clear();
            drop(sink);
            // Reset all position tracking so next queued samples establish new timeline
            self.initial_pts = None;
            self.handle.clear_playback_epoch();
            self.handle.reset_samples_played();
            self.handle.clear_audio_base_pts();
        }
    }
}

// ============================================================================
// Placeholder implementation (when ffmpeg feature is disabled)
// ============================================================================

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
mod placeholder_impl {
    use super::*;

    /// Placeholder audio player.
    pub struct AudioPlayer {
        /// Audio handle for control
        handle: AudioHandle,
        /// Current state (stored for API consistency with the real implementation,
        /// though unused in this placeholder since audio is not functional)
        #[allow(dead_code)]
        state: AudioState,
    }

    impl AudioPlayer {
        /// Creates a new audio player with optional external handle (placeholder).
        pub fn new_with_handle(
            _config: AudioConfig,
            external_handle: Option<AudioHandle>,
        ) -> Result<Self, String> {
            Ok(Self {
                handle: external_handle.unwrap_or_default(),
                state: AudioState::Uninitialized,
            })
        }

        /// Creates a new audio player (placeholder).
        pub fn new(config: AudioConfig) -> Result<Self, String> {
            Self::new_with_handle(config, None)
        }

        /// Returns the device sample rate (placeholder: returns 48000Hz).
        pub fn device_sample_rate(&self) -> u32 {
            48000
        }

        /// Returns the audio handle for control.
        pub fn handle(&self) -> AudioHandle {
            self.handle.clone()
        }

        /// Queues audio samples for playback (no-op).
        pub fn queue_samples(&mut self, _samples: AudioSamples) {
            // No-op
        }

        /// Starts audio playback (no-op).
        pub fn play(&mut self) {}

        /// Pauses audio playback (no-op).
        pub fn pause(&mut self) {}

        /// Returns the current state.
        pub fn state(&self) -> AudioState {
            AudioState::Uninitialized
        }

        /// Clears the audio buffer (no-op).
        pub fn clear(&mut self) {}
    }
}

// Re-export the appropriate implementation
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
pub use rodio_impl::AudioPlayer;

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]
pub use placeholder_impl::AudioPlayer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_handle_volume() {
        let handle = AudioHandle::new();
        assert_eq!(handle.volume(), 100);

        handle.set_volume(50);
        assert_eq!(handle.volume(), 50);

        handle.set_volume(150); // Should clamp to 100
        assert_eq!(handle.volume(), 100);
    }

    #[test]
    fn test_audio_handle_mute() {
        let handle = AudioHandle::new();
        assert!(!handle.is_muted());
        assert_eq!(handle.effective_volume(), 1.0);

        handle.set_muted(true);
        assert!(handle.is_muted());
        assert_eq!(handle.effective_volume(), 0.0);

        handle.toggle_mute();
        assert!(!handle.is_muted());
    }
}
