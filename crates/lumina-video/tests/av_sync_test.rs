//! A/V Synchronization Integration Tests
//!
//! These tests verify that audio and video remain synchronized during playback.
//! They measure the drift between audio PTS and video PTS over time and assert
//! that it stays within acceptable thresholds.
//!
//! # Test Videos
//!
//! The tests use publicly available test videos with audio tracks:
//! - Tears of Steel: Has dialogue, good for sync verification
//! - Big Buck Bunny: Has music/sound effects
//!
//! # Running Tests
//!
//! ```bash
//! cargo test --package lumina-video --test av_sync_test
//! ```
//!
//! For verbose sync metrics output:
//! ```bash
//! RUST_LOG=lumina_video=debug cargo test --test av_sync_test -- --nocapture
//! ```

// Integration tests verify sync metrics tracking works correctly
// Full playback tests require a graphics context and are better suited for the demo app

#[cfg(not(target_arch = "wasm32"))]
mod sync_tests {
    use lumina_video::media::sync_metrics::{
        SyncMetrics, SYNC_DRIFT_SEVERE_MS, SYNC_DRIFT_THRESHOLD_MS, SYNC_DRIFT_WARNING_MS,
    };
    use std::time::Duration;

    /// Test that SyncMetrics correctly tracks perfect sync.
    #[test]
    fn test_perfect_sync() {
        let metrics = SyncMetrics::new();

        // Simulate 100 frames with perfect sync
        for i in 0..100 {
            let pts = Duration::from_millis(i * 33); // ~30fps
            metrics.record_frame(pts, pts); // Video and audio at same position
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.current_drift_ms(), 0);
        assert_eq!(snap.max_drift_ms(), 0);
        assert_eq!(snap.out_of_sync_count, 0);
        assert!(snap.passed_sync_test());
    }

    /// Test that SyncMetrics correctly tracks video ahead of audio.
    #[test]
    fn test_video_ahead() {
        let metrics = SyncMetrics::new();

        // Video is 120ms ahead of audio (exceeds 100ms threshold)
        for i in 0..100 {
            let video_pts = Duration::from_millis(i * 33 + 120);
            let audio_pos = Duration::from_millis(i * 33);
            metrics.record_frame(video_pts, audio_pos);
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.current_drift_ms(), 120);
        assert!(snap.max_drift_ahead_us > 0);
        assert!(!snap.passed_sync_test()); // 120ms > 100ms threshold triggers out_of_sync
    }

    /// Test that SyncMetrics correctly tracks video behind audio.
    #[test]
    fn test_video_behind() {
        let metrics = SyncMetrics::new();

        // Video is 30ms behind audio (within threshold)
        for i in 0..100 {
            let video_pts = Duration::from_millis(i * 33);
            let audio_pos = Duration::from_millis(i * 33 + 30);
            metrics.record_frame(video_pts, audio_pos);
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.current_drift_ms(), -30);
        assert!(snap.max_drift_behind_us < 0);
        // 30ms is within threshold, so should pass
        assert!(snap.passed_sync_test());
    }

    /// Test drift thresholds are correctly applied.
    #[test]
    fn test_drift_thresholds() {
        // Verify threshold constants (relaxed for streaming: 100/150/200ms)
        assert_eq!(SYNC_DRIFT_THRESHOLD_MS, 100);
        assert_eq!(SYNC_DRIFT_WARNING_MS, 150);
        assert_eq!(SYNC_DRIFT_SEVERE_MS, 200);

        let metrics = SyncMetrics::new();

        // Record frames with varying drift (adjusted for new thresholds)
        let drifts = [0, 50, 99, 100, 101, 149, 150, 151, 199, 200, 201];
        for (i, &drift) in drifts.iter().enumerate() {
            let video_pts = Duration::from_millis(1000 + drift as u64);
            let audio_pos = Duration::from_millis(1000);
            metrics.record_frame(video_pts, audio_pos);

            let _snap = metrics.snapshot();

            // Check if in_sync status matches threshold
            let expected_in_sync = drift <= SYNC_DRIFT_THRESHOLD_MS;
            assert_eq!(
                metrics.is_in_sync(),
                expected_in_sync,
                "Frame {i} with drift {drift}ms should be in_sync={expected_in_sync}"
            );
        }
    }

    /// Test that reset clears all metrics.
    #[test]
    fn test_reset() {
        let metrics = SyncMetrics::new();

        // Record some frames with drift
        metrics.record_frame(Duration::from_millis(1100), Duration::from_millis(1000));
        metrics.record_frame(Duration::from_millis(1200), Duration::from_millis(1000));

        let snap = metrics.snapshot();
        assert!(snap.sample_count > 0);
        assert!(snap.max_drift_ms() > 0);

        // Reset
        metrics.reset();

        let snap = metrics.snapshot();
        assert_eq!(snap.sample_count, 0);
        assert_eq!(snap.max_drift_ms(), 0);
        assert_eq!(snap.current_drift_us, 0);
    }

    /// Test quality summary strings.
    /// Quality bands based on thresholds (100/150/200ms):
    /// - Excellent: < 100ms (threshold)
    /// - Good: 100-149ms (threshold to warning)
    /// - Fair: 150-199ms (warning to severe)
    /// - Poor: >= 200ms (severe)
    #[test]
    fn test_quality_summary() {
        let metrics = SyncMetrics::new();

        // Perfect sync -> Excellent
        metrics.record_frame(Duration::from_millis(1000), Duration::from_millis(1000));
        assert!(metrics.snapshot().quality_summary().contains("Excellent"));

        metrics.reset();

        // Small drift (50ms, within 100ms threshold) -> Excellent
        metrics.record_frame(Duration::from_millis(1050), Duration::from_millis(1000));
        assert!(metrics.snapshot().quality_summary().contains("Excellent"));

        metrics.reset();

        // Threshold drift (120ms, 100-150ms range) -> Good
        metrics.record_frame(Duration::from_millis(1120), Duration::from_millis(1000));
        assert!(metrics.snapshot().quality_summary().contains("Good"));

        metrics.reset();

        // Warning drift (160ms, 150-200ms range) -> Fair
        metrics.record_frame(Duration::from_millis(1160), Duration::from_millis(1000));
        assert!(metrics.snapshot().quality_summary().contains("Fair"));

        metrics.reset();

        // Severe drift (250ms, >= 200ms) -> Poor
        metrics.record_frame(Duration::from_millis(1250), Duration::from_millis(1000));
        assert!(metrics.snapshot().quality_summary().contains("Poor"));
    }

    /// Test out of sync percentage calculation.
    #[test]
    fn test_out_of_sync_percentage() {
        let metrics = SyncMetrics::new();

        // 90 frames in sync (50ms drift, within 100ms threshold)
        for i in 0..90 {
            metrics.record_frame(
                Duration::from_millis(i * 33 + 50),
                Duration::from_millis(i * 33),
            );
        }

        // 10 frames out of sync (120ms drift, exceeds 100ms threshold)
        for i in 90..100 {
            metrics.record_frame(
                Duration::from_millis(i * 33 + 120),
                Duration::from_millis(i * 33),
            );
        }

        let snap = metrics.snapshot();
        assert_eq!(snap.sample_count, 100);
        assert_eq!(snap.out_of_sync_count, 10);
        assert!((snap.out_of_sync_percentage() - 10.0).abs() < 0.01);
    }

    /// Test display formatting.
    #[test]
    fn test_display() {
        let metrics = SyncMetrics::new();
        metrics.record_frame(Duration::from_millis(1050), Duration::from_millis(1000));

        let snap = metrics.snapshot();
        let display = format!("{snap}");
        assert!(display.contains("A/V Sync"));
        assert!(display.contains("+50ms"));
    }
}

/// Baseline A/V sync test configuration.
/// These constants define acceptable sync quality for automated testing.
#[allow(dead_code)]
mod baseline {
    /// Maximum acceptable average drift in milliseconds.
    pub const MAX_AVG_DRIFT_MS: i64 = 50;

    /// Maximum acceptable peak drift in milliseconds.
    pub const MAX_PEAK_DRIFT_MS: i64 = 100;

    /// Maximum acceptable percentage of frames out of sync.
    pub const MAX_OUT_OF_SYNC_PERCENT: f64 = 5.0;

    /// Minimum number of samples required for a valid test.
    pub const MIN_SAMPLES: u64 = 100;
}
