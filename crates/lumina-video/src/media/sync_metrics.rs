//! A/V synchronization metrics and drift tracking.
//!
//! This module provides instrumentation for measuring audio-video synchronization
//! quality. It tracks the drift between audio and video presentation timestamps
//! and provides statistics for debugging and automated testing.
//!
//! # Usage
//!
//! ```ignore
//! let metrics = SyncMetrics::new();
//!
//! // During playback, record each video frame presentation
//! metrics.record_frame(video_pts, audio_position);
//!
//! // Get current sync status
//! let snapshot = metrics.snapshot();
//! println!("Current drift: {:?}", snapshot.current_drift);
//! println!("Max drift: {:?}", snapshot.max_drift);
//! ```

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Threshold for acceptable A/V sync drift (±100ms is acceptable for streaming).
/// Note: ±40ms is imperceptible, but streaming/network variability makes 100ms more practical.
pub const SYNC_DRIFT_THRESHOLD_MS: i64 = 100;

/// Threshold for warning-level drift (±150ms is noticeable but tolerable).
pub const SYNC_DRIFT_WARNING_MS: i64 = 150;

/// Threshold for severe drift (±200ms is clearly out of sync).
pub const SYNC_DRIFT_SEVERE_MS: i64 = 200;

/// Type of stall event for differentiated tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StallType {
    /// Decoder couldn't produce frames (decode stall)
    Decode,
    /// Network/buffering issues prevented frame delivery
    Network,
}

/// A/V synchronization metrics tracker.
///
/// Thread-safe structure for recording and analyzing audio-video sync drift.
/// Uses atomic operations for lock-free updates from multiple threads.
#[derive(Clone)]
pub struct SyncMetrics {
    inner: Arc<SyncMetricsInner>,
}

struct SyncMetricsInner {
    /// Whether metrics collection is enabled
    enabled: AtomicBool,

    /// Current drift in microseconds (video_pts - audio_pos, positive = video ahead)
    current_drift_us: AtomicI64,

    /// Maximum positive drift seen (video ahead of audio)
    max_drift_ahead_us: AtomicI64,

    /// Maximum negative drift seen (video behind audio)
    max_drift_behind_us: AtomicI64,

    /// Sum of absolute drift values for average calculation
    total_drift_us: AtomicU64,

    /// Number of samples recorded
    sample_count: AtomicU64,

    /// Number of frames where drift exceeded threshold
    out_of_sync_count: AtomicU64,

    /// Last video PTS in microseconds
    last_video_pts_us: AtomicU64,

    /// Last audio position in microseconds
    last_audio_pos_us: AtomicU64,

    /// Whether audio clock is being used (vs wall-clock fallback)
    using_audio_clock: AtomicBool,

    /// Start time for session duration tracking
    start_time_us: AtomicU64,

    // ========================================================================
    // Underrun, stall, and recovery tracking
    // ========================================================================
    /// Number of buffer underrun events (buffer empty when frame needed)
    underrun_count: AtomicU64,

    /// Timestamp of last underrun in microseconds (for time-to-recover calculation)
    last_underrun_time_us: AtomicU64,

    /// Number of stall events (extended periods with no frames)
    stall_count: AtomicU64,

    /// Number of decode stalls (decoder couldn't produce frames)
    decode_stall_count: AtomicU64,

    /// Number of network stalls (network/buffering issues)
    network_stall_count: AtomicU64,

    /// Whether currently in recovery mode after a stall
    in_recovery: AtomicBool,

    /// Number of samples recorded during recovery periods
    recovery_samples: AtomicU64,

    /// Total drift accumulated during recovery periods (microseconds)
    recovery_total_drift_us: AtomicU64,

    /// Timestamp when recovery started (for time-to-recover)
    recovery_start_time_us: AtomicU64,

    /// Total time spent in recovery (microseconds)
    total_recovery_time_us: AtomicU64,

    /// Number of completed recovery periods
    recovery_count: AtomicU64,

    /// Number of out-of-sync frames during recovery (for computing steady-state)
    recovery_out_of_sync_count: AtomicU64,

    /// PTS offset between audio and video streams (audio_start - video_start) in microseconds
    stream_pts_offset_us: AtomicI64,

    // ========================================================================
    // Real-time FPS tracking
    // ========================================================================
    /// Wall-clock time when FPS measurement window started (microseconds since epoch)
    fps_window_start_us: AtomicU64,

    /// Number of frames delivered in current FPS measurement window
    fps_window_frames: AtomicU64,

    /// Current measured FPS (multiplied by 100 for precision, e.g. 2400 = 24.00 fps)
    current_fps_x100: AtomicU64,

    /// Grace period samples remaining after seek/reset (skip max_drift during this time)
    grace_samples: AtomicU64,

    /// Whether A/V sync is managed externally (native player like AVPlayer)
    /// When true, drift measurements are not meaningful and UI should indicate this
    sync_externally_managed: AtomicBool,
}

impl SyncMetrics {
    /// Creates a new sync metrics tracker.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SyncMetricsInner {
                enabled: AtomicBool::new(true),
                current_drift_us: AtomicI64::new(0),
                max_drift_ahead_us: AtomicI64::new(0),
                max_drift_behind_us: AtomicI64::new(0),
                total_drift_us: AtomicU64::new(0),
                sample_count: AtomicU64::new(0),
                out_of_sync_count: AtomicU64::new(0),
                last_video_pts_us: AtomicU64::new(0),
                last_audio_pos_us: AtomicU64::new(0),
                using_audio_clock: AtomicBool::new(false),
                start_time_us: AtomicU64::new(0),
                // Underrun, stall, and recovery tracking
                underrun_count: AtomicU64::new(0),
                last_underrun_time_us: AtomicU64::new(0),
                stall_count: AtomicU64::new(0),
                decode_stall_count: AtomicU64::new(0),
                network_stall_count: AtomicU64::new(0),
                in_recovery: AtomicBool::new(false),
                recovery_samples: AtomicU64::new(0),
                recovery_total_drift_us: AtomicU64::new(0),
                recovery_start_time_us: AtomicU64::new(0),
                total_recovery_time_us: AtomicU64::new(0),
                recovery_count: AtomicU64::new(0),
                recovery_out_of_sync_count: AtomicU64::new(0),
                stream_pts_offset_us: AtomicI64::new(0),
                // FPS tracking
                fps_window_start_us: AtomicU64::new(0),
                fps_window_frames: AtomicU64::new(0),
                current_fps_x100: AtomicU64::new(0),
                // Grace period (skip max_drift tracking for first N samples after seek)
                grace_samples: AtomicU64::new(0),
                // External sync management (native players)
                sync_externally_managed: AtomicBool::new(false),
            }),
        }
    }

    /// Enables or disables metrics collection.
    pub fn set_enabled(&self, enabled: bool) {
        self.inner.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Returns whether metrics collection is enabled.
    pub fn is_enabled(&self) -> bool {
        self.inner.enabled.load(Ordering::Relaxed)
    }

    /// Sets whether audio clock is being used.
    pub fn set_using_audio_clock(&self, using: bool) {
        self.inner.using_audio_clock.store(using, Ordering::Relaxed);
    }

    /// Sets whether A/V sync is managed externally by a native player.
    ///
    /// When true, drift measurements are not meaningful because the native player
    /// (AVPlayer, GStreamer) handles A/V sync internally. The UI should indicate
    /// this rather than showing 0ms drift.
    pub fn set_sync_externally_managed(&self, managed: bool) {
        self.inner
            .sync_externally_managed
            .store(managed, Ordering::Relaxed);
    }

    /// Returns whether A/V sync is managed externally.
    pub fn is_sync_externally_managed(&self) -> bool {
        self.inner.sync_externally_managed.load(Ordering::Relaxed)
    }

    /// Sets the PTS offset between audio and video streams.
    ///
    /// The offset is calculated as (audio_start - video_start) in microseconds.
    /// Positive means audio stream starts later than video stream.
    pub fn set_stream_pts_offset(&self, offset_us: i64) {
        self.inner
            .stream_pts_offset_us
            .store(offset_us, Ordering::Relaxed);
    }

    /// Returns the PTS offset between audio and video streams.
    pub fn stream_pts_offset_us(&self) -> i64 {
        self.inner.stream_pts_offset_us.load(Ordering::Relaxed)
    }

    /// Records a video frame presentation for sync analysis.
    ///
    /// # Arguments
    /// * `video_pts` - The presentation timestamp of the video frame
    /// * `audio_position` - The current audio playback position
    ///
    /// # Returns
    /// The non-negative drift as a Duration (video ahead of audio).
    /// Negative drift (video behind audio) returns `Duration::ZERO` since
    /// Duration cannot represent negative values. Use `current_drift_ms()`
    /// on the snapshot for signed drift values.
    pub fn record_frame(&self, video_pts: Duration, audio_position: Duration) -> Duration {
        if !self.inner.enabled.load(Ordering::Relaxed) {
            return Duration::ZERO;
        }

        let video_us = video_pts.as_micros() as i64;
        let audio_us = audio_position.as_micros() as i64;
        let drift_us = video_us - audio_us;

        // Check grace period - skip drift updates during seek warmup to avoid transient spikes
        // Use fetch_update with checked_sub to prevent underflow on concurrent calls
        let in_grace = self
            .inner
            .grace_samples
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1))
            .is_ok();

        // Update current values (skip during grace period to avoid showing transient spikes)
        if !in_grace {
            self.inner
                .current_drift_us
                .store(drift_us, Ordering::Relaxed);
        }
        self.inner
            .last_video_pts_us
            .store(video_us as u64, Ordering::Relaxed);
        self.inner
            .last_audio_pos_us
            .store(audio_us as u64, Ordering::Relaxed);

        // Update max drift (positive = video ahead), also skip during grace period
        if !in_grace {
            if drift_us > 0 {
                self.inner
                    .max_drift_ahead_us
                    .fetch_max(drift_us, Ordering::Relaxed);
            } else {
                self.inner
                    .max_drift_behind_us
                    .fetch_min(drift_us, Ordering::Relaxed);
            }
        }

        // Update running totals
        self.inner
            .total_drift_us
            .fetch_add(drift_us.unsigned_abs(), Ordering::Relaxed);
        self.inner.sample_count.fetch_add(1, Ordering::Relaxed);

        // Track out-of-sync frames
        let drift_ms = drift_us.abs() / 1000;
        let is_out_of_sync = drift_ms > SYNC_DRIFT_THRESHOLD_MS;

        // Track recovery metrics separately when in recovery mode
        let in_recovery = self.inner.in_recovery.load(Ordering::Relaxed);
        if in_recovery {
            self.inner.recovery_samples.fetch_add(1, Ordering::Relaxed);
            self.inner
                .recovery_total_drift_us
                .fetch_add(drift_us.unsigned_abs(), Ordering::Relaxed);
            if is_out_of_sync {
                self.inner
                    .recovery_out_of_sync_count
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        if is_out_of_sync {
            self.inner.out_of_sync_count.fetch_add(1, Ordering::Relaxed);

            // Log significant drift
            if drift_ms > SYNC_DRIFT_WARNING_MS {
                let direction = if drift_us > 0 { "ahead" } else { "behind" };
                tracing::warn!(
                    "A/V sync: video {}ms {} audio (video_pts={:?}, audio_pos={:?})",
                    drift_ms,
                    direction,
                    video_pts,
                    audio_position
                );
            }
        }

        // Update FPS tracking
        self.update_fps_tracking();

        // Return signed drift as Duration (negative represented as zero for Duration)
        if drift_us >= 0 {
            Duration::from_micros(drift_us as u64)
        } else {
            Duration::ZERO // Caller should use snapshot() for signed drift
        }
    }

    /// Updates FPS tracking based on wall-clock time.
    /// Called internally by record_frame().
    fn update_fps_tracking(&self) {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;

        let window_start = self.inner.fps_window_start_us.load(Ordering::Relaxed);

        // Initialize window on first call
        if window_start == 0 {
            self.inner
                .fps_window_start_us
                .store(now_us, Ordering::Relaxed);
            self.inner.fps_window_frames.store(1, Ordering::Relaxed);
            return;
        }

        let elapsed_us = now_us.saturating_sub(window_start);
        let frames = self.inner.fps_window_frames.fetch_add(1, Ordering::Relaxed) + 1;

        // Calculate FPS every 500ms (enough samples for accuracy)
        if elapsed_us >= 500_000 {
            // Calculate FPS * 100 for precision (e.g., 2400 = 24.00 fps)
            let fps_x100 = if elapsed_us > 0 {
                (frames * 100_000_000) / elapsed_us
            } else {
                0
            };
            self.inner
                .current_fps_x100
                .store(fps_x100, Ordering::Relaxed);

            // Reset window
            self.inner
                .fps_window_start_us
                .store(now_us, Ordering::Relaxed);
            self.inner.fps_window_frames.store(0, Ordering::Relaxed);
        }
    }

    /// Returns the current measured FPS (real-time frame delivery rate).
    pub fn current_fps(&self) -> f32 {
        self.inner.current_fps_x100.load(Ordering::Relaxed) as f32 / 100.0
    }

    // ========================================================================
    // Underrun, stall, and recovery instrumentation
    // ========================================================================

    /// Records a buffer underrun event (buffer was empty when a frame was needed).
    ///
    /// This tracks when the frame queue couldn't provide a frame during active playback,
    /// which indicates the decoder/network isn't keeping up with playback.
    pub fn record_underrun(&self) {
        if !self.inner.enabled.load(Ordering::Relaxed) {
            return;
        }

        let now = now_micros();
        self.inner.underrun_count.fetch_add(1, Ordering::Relaxed);
        self.inner
            .last_underrun_time_us
            .store(now, Ordering::Relaxed);

        tracing::warn!(
            "Buffer underrun #{} at t={}ms",
            self.inner.underrun_count.load(Ordering::Relaxed),
            now / 1000
        );
    }

    /// Records a stall event with the type of stall.
    ///
    /// # Arguments
    /// * `stall_type` - The type of stall that occurred
    pub fn record_stall(&self, stall_type: StallType) {
        if !self.inner.enabled.load(Ordering::Relaxed) {
            return;
        }

        self.inner.stall_count.fetch_add(1, Ordering::Relaxed);

        match stall_type {
            StallType::Decode => {
                self.inner
                    .decode_stall_count
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    "Decode stall #{} (total stalls: {})",
                    self.inner.decode_stall_count.load(Ordering::Relaxed),
                    self.inner.stall_count.load(Ordering::Relaxed)
                );
            }
            StallType::Network => {
                self.inner
                    .network_stall_count
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    "Network stall #{} (total stalls: {})",
                    self.inner.network_stall_count.load(Ordering::Relaxed),
                    self.inner.stall_count.load(Ordering::Relaxed)
                );
            }
        }
    }

    /// Starts recovery mode after a stall.
    ///
    /// During recovery, drift metrics are tracked separately to distinguish
    /// between steady-state performance and post-stall behavior.
    pub fn start_recovery(&self) {
        if !self.inner.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Only start if not already in recovery
        if self
            .inner
            .in_recovery
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let now = now_micros();
            self.inner
                .recovery_start_time_us
                .store(now, Ordering::Relaxed);

            tracing::debug!("Starting recovery mode at t={}ms", now / 1000);
        }
    }

    /// Ends recovery mode and records the recovery duration.
    ///
    /// Call this when playback has stabilized after a stall (e.g., after
    /// a certain number of frames have been displayed smoothly).
    pub fn end_recovery(&self) {
        if !self.inner.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Only end if currently in recovery
        if self
            .inner
            .in_recovery
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let now = now_micros();
            let start = self.inner.recovery_start_time_us.load(Ordering::Relaxed);
            let recovery_duration = now.saturating_sub(start);

            self.inner
                .total_recovery_time_us
                .fetch_add(recovery_duration, Ordering::Relaxed);
            self.inner.recovery_count.fetch_add(1, Ordering::Relaxed);

            // Calculate time-to-recover: time from last underrun to now
            let last_underrun = self.inner.last_underrun_time_us.load(Ordering::Relaxed);
            let time_to_recover = if last_underrun > 0 && last_underrun <= now {
                now.saturating_sub(last_underrun)
            } else {
                recovery_duration // Fall back to recovery duration if no underrun recorded
            };

            tracing::debug!(
                "Recovery complete: duration={}ms, time-to-recover={}ms, recovery #{}",
                recovery_duration / 1000,
                time_to_recover / 1000,
                self.inner.recovery_count.load(Ordering::Relaxed)
            );
        }
    }

    /// Returns whether currently in recovery mode.
    pub fn is_in_recovery(&self) -> bool {
        self.inner.in_recovery.load(Ordering::Relaxed)
    }

    /// Returns the number of underrun events.
    pub fn underrun_count(&self) -> u64 {
        self.inner.underrun_count.load(Ordering::Relaxed)
    }

    /// Returns the number of stall events.
    pub fn stall_count(&self) -> u64 {
        self.inner.stall_count.load(Ordering::Relaxed)
    }

    /// Resets all metrics.
    pub fn reset(&self) {
        self.inner.current_drift_us.store(0, Ordering::Relaxed);
        self.inner.max_drift_ahead_us.store(0, Ordering::Relaxed);
        self.inner.max_drift_behind_us.store(0, Ordering::Relaxed);
        self.inner.total_drift_us.store(0, Ordering::Relaxed);
        self.inner.sample_count.store(0, Ordering::Relaxed);
        self.inner.out_of_sync_count.store(0, Ordering::Relaxed);
        self.inner.last_video_pts_us.store(0, Ordering::Relaxed);
        self.inner.last_audio_pos_us.store(0, Ordering::Relaxed);
        self.inner
            .start_time_us
            .store(now_micros(), Ordering::Relaxed);
        // Reset underrun, stall, and recovery tracking
        self.inner.underrun_count.store(0, Ordering::Relaxed);
        self.inner.last_underrun_time_us.store(0, Ordering::Relaxed);
        self.inner.stall_count.store(0, Ordering::Relaxed);
        self.inner.decode_stall_count.store(0, Ordering::Relaxed);
        self.inner.network_stall_count.store(0, Ordering::Relaxed);
        self.inner.in_recovery.store(false, Ordering::Relaxed);
        self.inner.recovery_samples.store(0, Ordering::Relaxed);
        self.inner
            .recovery_total_drift_us
            .store(0, Ordering::Relaxed);
        self.inner
            .recovery_start_time_us
            .store(0, Ordering::Relaxed);
        self.inner
            .total_recovery_time_us
            .store(0, Ordering::Relaxed);
        self.inner.recovery_count.store(0, Ordering::Relaxed);
        self.inner
            .recovery_out_of_sync_count
            .store(0, Ordering::Relaxed);
        // Note: stream_pts_offset_us is not reset here as it's set externally
        // Reset grace period to zero; callers can re-arm via set_grace_period() if needed
        self.inner.grace_samples.store(0, Ordering::Relaxed);
        // Reset FPS tracking to avoid stale values after seek
        self.inner.fps_window_start_us.store(0, Ordering::Relaxed);
        self.inner.fps_window_frames.store(0, Ordering::Relaxed);
        self.inner.current_fps_x100.store(0, Ordering::Relaxed);
        // Reset external sync flag (will be re-set on next record_sync call)
        self.inner
            .sync_externally_managed
            .store(false, Ordering::Relaxed);
    }

    /// Sets a grace period to skip max_drift updates for the next N samples.
    /// Use after seeks to filter transient drift spikes during warmup.
    pub fn set_grace_period(&self, samples: u64) {
        self.inner.grace_samples.store(samples, Ordering::Relaxed);
    }

    /// Returns a snapshot of current sync metrics.
    pub fn snapshot(&self) -> SyncMetricsSnapshot {
        let sample_count = self.inner.sample_count.load(Ordering::Relaxed);
        let total_drift = self.inner.total_drift_us.load(Ordering::Relaxed);
        let avg_drift_us = if sample_count > 0 {
            (total_drift / sample_count) as i64
        } else {
            0
        };

        // Calculate recovery average drift
        let recovery_samples = self.inner.recovery_samples.load(Ordering::Relaxed);
        let recovery_total_drift = self.inner.recovery_total_drift_us.load(Ordering::Relaxed);
        let recovery_avg_drift_us = if recovery_samples > 0 {
            (recovery_total_drift / recovery_samples) as i64
        } else {
            0
        };

        // Calculate time since last underrun
        let last_underrun = self.inner.last_underrun_time_us.load(Ordering::Relaxed);
        let time_since_last_underrun_us = if last_underrun > 0 {
            now_micros().saturating_sub(last_underrun)
        } else {
            0
        };

        // Calculate steady-state metrics (excluding recovery periods)
        let steady_samples = sample_count.saturating_sub(recovery_samples);
        let recovery_out_of_sync = self
            .inner
            .recovery_out_of_sync_count
            .load(Ordering::Relaxed);
        let out_of_sync_count = self.inner.out_of_sync_count.load(Ordering::Relaxed);
        let steady_out_of_sync_count = out_of_sync_count.saturating_sub(recovery_out_of_sync);
        let steady_total_drift = total_drift.saturating_sub(recovery_total_drift);
        let steady_avg_drift_us = if steady_samples > 0 {
            (steady_total_drift / steady_samples) as i64
        } else {
            0
        };

        // Get stream PTS offset
        let stream_pts_offset_us = self.inner.stream_pts_offset_us.load(Ordering::Relaxed);

        SyncMetricsSnapshot {
            current_drift_us: self.inner.current_drift_us.load(Ordering::Relaxed),
            max_drift_ahead_us: self.inner.max_drift_ahead_us.load(Ordering::Relaxed),
            max_drift_behind_us: self.inner.max_drift_behind_us.load(Ordering::Relaxed),
            avg_drift_us,
            sample_count,
            out_of_sync_count: self.inner.out_of_sync_count.load(Ordering::Relaxed),
            last_video_pts: Duration::from_micros(
                self.inner.last_video_pts_us.load(Ordering::Relaxed),
            ),
            last_audio_pos: Duration::from_micros(
                self.inner.last_audio_pos_us.load(Ordering::Relaxed),
            ),
            using_audio_clock: self.inner.using_audio_clock.load(Ordering::Relaxed),
            // Underrun, stall, and recovery metrics
            underrun_count: self.inner.underrun_count.load(Ordering::Relaxed),
            stall_count: self.inner.stall_count.load(Ordering::Relaxed),
            decode_stall_count: self.inner.decode_stall_count.load(Ordering::Relaxed),
            network_stall_count: self.inner.network_stall_count.load(Ordering::Relaxed),
            in_recovery: self.inner.in_recovery.load(Ordering::Relaxed),
            recovery_samples,
            recovery_avg_drift_us,
            recovery_count: self.inner.recovery_count.load(Ordering::Relaxed),
            total_recovery_time_us: self.inner.total_recovery_time_us.load(Ordering::Relaxed),
            time_since_last_underrun_us,
            // Steady-state metrics
            steady_samples,
            steady_avg_drift_us,
            steady_out_of_sync_count,
            // Stream offset
            stream_pts_offset_us,
            // Real-time FPS
            current_fps: self.current_fps(),
            // External sync management
            sync_externally_managed: self.inner.sync_externally_managed.load(Ordering::Relaxed),
        }
    }

    /// Returns true if sync is currently within acceptable threshold.
    pub fn is_in_sync(&self) -> bool {
        let drift_us = self.inner.current_drift_us.load(Ordering::Relaxed);
        drift_us.abs() <= SYNC_DRIFT_THRESHOLD_MS * 1000
    }

    /// Logs current sync status at debug level.
    pub fn log_status(&self) {
        let snap = self.snapshot();
        tracing::debug!(
            "A/V Sync: current={:+}ms, max_ahead={:+}ms, max_behind={:+}ms, avg={:+}ms, samples={}, out_of_sync={}, audio_clock={}",
            snap.current_drift_us / 1000,
            snap.max_drift_ahead_us / 1000,
            snap.max_drift_behind_us / 1000,
            snap.avg_drift_us / 1000,
            snap.sample_count,
            snap.out_of_sync_count,
            snap.using_audio_clock
        );
    }
}

impl Default for SyncMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of sync metrics at a point in time.
#[derive(Debug, Clone)]
pub struct SyncMetricsSnapshot {
    /// Current drift in microseconds (positive = video ahead of audio)
    pub current_drift_us: i64,

    /// Maximum drift where video was ahead of audio (microseconds)
    pub max_drift_ahead_us: i64,

    /// Maximum drift where video was behind audio (microseconds, negative)
    pub max_drift_behind_us: i64,

    /// Average absolute drift in microseconds
    pub avg_drift_us: i64,

    /// Total number of frames measured
    pub sample_count: u64,

    /// Number of frames where drift exceeded threshold
    pub out_of_sync_count: u64,

    /// Last video PTS recorded
    pub last_video_pts: Duration,

    /// Last audio position recorded
    pub last_audio_pos: Duration,

    /// Whether audio clock was being used
    pub using_audio_clock: bool,

    // ========================================================================
    // Underrun, stall, and recovery metrics
    // ========================================================================
    /// Number of buffer underrun events
    pub underrun_count: u64,

    /// Total number of stall events
    pub stall_count: u64,

    /// Number of decode stalls (decoder couldn't produce frames)
    pub decode_stall_count: u64,

    /// Number of network stalls (network/buffering issues)
    pub network_stall_count: u64,

    /// Whether currently in recovery mode
    pub in_recovery: bool,

    /// Number of samples recorded during recovery periods
    pub recovery_samples: u64,

    /// Average drift during recovery periods (microseconds)
    pub recovery_avg_drift_us: i64,

    /// Number of completed recovery periods
    pub recovery_count: u64,

    /// Total time spent in recovery (microseconds)
    pub total_recovery_time_us: u64,

    /// Time since last underrun (microseconds), 0 if no underruns
    pub time_since_last_underrun_us: u64,

    // ========================================================================
    // Steady-state metrics (excluding recovery periods)
    // ========================================================================
    /// Number of samples recorded during steady-state (total - recovery)
    pub steady_samples: u64,

    /// Average absolute drift during steady-state only (microseconds)
    pub steady_avg_drift_us: i64,

    /// Number of out-of-sync frames during steady-state only
    pub steady_out_of_sync_count: u64,

    // ========================================================================
    // Stream offset tracking
    // ========================================================================
    /// PTS offset between audio and video streams (audio_start - video_start) in microseconds
    /// Positive means audio stream starts later than video stream
    pub stream_pts_offset_us: i64,

    // ========================================================================
    // Real-time FPS
    // ========================================================================
    /// Current measured FPS (real-time frame delivery rate)
    pub current_fps: f32,

    // ========================================================================
    // External sync management
    // ========================================================================
    /// Whether A/V sync is managed externally by a native player (AVPlayer, GStreamer)
    /// When true, drift values are not meaningful and UI should show "externally managed"
    pub sync_externally_managed: bool,
}

impl SyncMetricsSnapshot {
    /// Returns the current drift as a signed duration in milliseconds.
    pub fn current_drift_ms(&self) -> i64 {
        self.current_drift_us / 1000
    }

    /// Returns the maximum absolute drift in milliseconds.
    pub fn max_drift_ms(&self) -> i64 {
        self.max_drift_ahead_us
            .abs()
            .max(self.max_drift_behind_us.abs())
            / 1000
    }

    /// Returns the percentage of frames that were out of sync.
    pub fn out_of_sync_percentage(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            (self.out_of_sync_count as f64 / self.sample_count as f64) * 100.0
        }
    }

    /// Returns the percentage of steady-state frames that were out of sync.
    pub fn steady_out_of_sync_percentage(&self) -> f64 {
        if self.steady_samples == 0 {
            0.0
        } else {
            (self.steady_out_of_sync_count as f64 / self.steady_samples as f64) * 100.0
        }
    }

    /// Returns the average absolute drift during steady-state in milliseconds.
    pub fn steady_avg_drift_ms(&self) -> i64 {
        self.steady_avg_drift_us / 1000
    }

    /// Returns the stream PTS offset in milliseconds.
    pub fn stream_pts_offset_ms(&self) -> i64 {
        self.stream_pts_offset_us / 1000
    }

    /// Minimum samples required for a valid sync test.
    const MIN_SYNC_SAMPLES: u64 = 10;

    /// Returns true if the session passed sync quality criteria.
    ///
    /// Criteria:
    /// - At least `MIN_SYNC_SAMPLES` frames recorded (prevents false positives)
    /// - Max drift < SYNC_DRIFT_SEVERE_MS
    /// - Less than 5% of frames out of sync
    pub fn passed_sync_test(&self) -> bool {
        self.sample_count >= Self::MIN_SYNC_SAMPLES
            && self.max_drift_ms() < SYNC_DRIFT_SEVERE_MS
            && self.out_of_sync_percentage() < 5.0
    }

    /// Returns a human-readable summary of sync quality.
    pub fn quality_summary(&self) -> String {
        let max_drift = self.max_drift_ms();
        let quality = if max_drift < SYNC_DRIFT_THRESHOLD_MS {
            "Excellent"
        } else if max_drift < SYNC_DRIFT_WARNING_MS {
            "Good"
        } else if max_drift < SYNC_DRIFT_SEVERE_MS {
            "Fair"
        } else {
            "Poor"
        };

        // Use steady-state metrics if available, otherwise fall back to totals
        let (avg_drift_ms, out_of_sync_pct) = if self.steady_samples > 0 {
            (
                self.steady_avg_drift_ms(),
                self.steady_out_of_sync_percentage(),
            )
        } else {
            (self.avg_drift_us / 1000, self.out_of_sync_percentage())
        };

        format!(
            "{quality} (max drift: {max_drift:+}ms, steady-state avg: {avg_drift_ms:+}ms, {out_of_sync_pct:.1}% out of sync)"
        )
    }
}

impl std::fmt::Display for SyncMetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "A/V Sync: drift={:+}ms (max ahead={:+}ms, behind={:+}ms), {} samples, {:.1}% out of sync",
            self.current_drift_ms(),
            self.max_drift_ahead_us / 1000,
            self.max_drift_behind_us / 1000,
            self.sample_count,
            self.out_of_sync_percentage()
        )
    }
}

/// Returns current time in microseconds since epoch.
fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_metrics_basic() {
        let metrics = SyncMetrics::new();

        // Perfect sync
        metrics.record_frame(Duration::from_millis(1000), Duration::from_millis(1000));
        assert!(metrics.is_in_sync());

        let snap = metrics.snapshot();
        assert_eq!(snap.current_drift_us, 0);
        assert_eq!(snap.sample_count, 1);
    }

    #[test]
    fn test_sync_metrics_drift() {
        let metrics = SyncMetrics::new();

        // Video 120ms ahead of audio
        metrics.record_frame(Duration::from_millis(1120), Duration::from_millis(1000));

        let snap = metrics.snapshot();
        assert_eq!(snap.current_drift_ms(), 120);
        assert_eq!(snap.max_drift_ahead_us, 120_000);
        assert!(!metrics.is_in_sync()); // 120ms > 100ms threshold
    }

    #[test]
    fn test_sync_metrics_behind() {
        let metrics = SyncMetrics::new();

        // Video 50ms behind audio
        metrics.record_frame(Duration::from_millis(950), Duration::from_millis(1000));

        let snap = metrics.snapshot();
        assert_eq!(snap.current_drift_ms(), -50);
        assert_eq!(snap.max_drift_behind_us, -50_000);
        assert!(metrics.is_in_sync()); // 50ms < 100ms threshold
    }

    #[test]
    fn test_sync_quality_rating() {
        let metrics = SyncMetrics::new();

        // Record frames with varying drift
        for i in 0..100 {
            let drift = if i % 10 == 0 { 120 } else { 20 }; // 10% at 120ms, 90% at 20ms
            metrics.record_frame(
                Duration::from_millis(1000 + drift),
                Duration::from_millis(1000),
            );
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.sample_count, 100);
        // 10% were at 120ms (> 100ms threshold)
        assert_eq!(snap.out_of_sync_count, 10);
        assert!((snap.out_of_sync_percentage() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_sync_test_pass_fail() {
        let metrics = SyncMetrics::new();

        // All frames in sync
        for _ in 0..100 {
            metrics.record_frame(Duration::from_millis(1010), Duration::from_millis(1000));
        }
        assert!(metrics.snapshot().passed_sync_test());

        // Reset and add severe drift
        metrics.reset();
        metrics.record_frame(Duration::from_millis(1250), Duration::from_millis(1000));
        assert!(!metrics.snapshot().passed_sync_test()); // 250ms > 200ms severe threshold
    }
}
