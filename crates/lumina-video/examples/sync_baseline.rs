//! A/V Sync Baseline Measurement Tool
//!
//! Runs an automated playback test and records sync metrics.
//! No GUI interaction required - runs for a set duration and outputs results.
//!
//! Usage:
//!   # Default test video (Tears of Steel), 30 second duration
//!   cargo run -p lumina-video --example sync_baseline --features "ffmpeg,macos-native-video"
//!
//!   # Local file test
//!   cargo run -p lumina-video --example sync_baseline --features "ffmpeg,macos-native-video" -- /path/to/video.mp4
//!
//!   # Local file with custom duration (seconds)
//!   cargo run -p lumina-video --example sync_baseline --features "ffmpeg,macos-native-video" -- /path/to/video.mp4 60
//!
//!   # URL with custom duration
//!   cargo run -p lumina-video --example sync_baseline --features "ffmpeg,macos-native-video" -- https://example.com/video.mp4 120

use std::time::{Duration, Instant};

use eframe::egui;
use lumina_video::{VideoPlayer, VideoPlayerExt};

/// Test video with dialogue for A/V sync verification
const TEST_VIDEO: &str =
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4";

/// Default test duration in seconds
const DEFAULT_DURATION_SECS: u64 = 30;

fn main() -> eframe::Result<()> {
    // Parse arguments: [video_path] [duration_secs]
    // - If first arg is a number, treat it as duration (backward compat)
    // - If first arg is a path/URL, use it as video source
    // - Second arg (if present) is always duration
    let args: Vec<String> = std::env::args().skip(1).collect();

    let (video_source, duration_secs) = match args.len() {
        0 => (TEST_VIDEO.to_string(), DEFAULT_DURATION_SECS),
        1 => {
            // Check if it's a number (duration) or a path/URL
            if let Ok(dur) = args[0].parse::<u64>() {
                (TEST_VIDEO.to_string(), dur)
            } else {
                (args[0].clone(), DEFAULT_DURATION_SECS)
            }
        }
        _ => {
            // First arg is video, second is duration
            let dur = args[1].parse().unwrap_or(DEFAULT_DURATION_SECS);
            (args[0].clone(), dur)
        }
    };

    // Determine display name for video
    let video_display = if video_source.starts_with("http") {
        if video_source.contains("TearsOfSteel") {
            "Tears of Steel (remote)".to_string()
        } else {
            // Truncate long URLs
            if video_source.len() > 40 {
                format!("{}...", &video_source[..40])
            } else {
                video_source.clone()
            }
        }
    } else {
        // Local file - show just the filename
        std::path::Path::new(&video_source)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| video_source.clone())
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         A/V SYNC BASELINE MEASUREMENT                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Video: {video_display:<53} ║");
    println!("║ Duration: {duration_secs:>3} seconds                                       ║");
    println!("║ Starting playback...                                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([854.0, 480.0])
            .with_title("A/V Sync Baseline Test"),
        ..Default::default()
    };

    eframe::run_native(
        "sync_baseline",
        options,
        Box::new(move |cc| {
            Ok(Box::new(SyncBaselineApp::new(
                cc,
                &video_source,
                duration_secs,
            )))
        }),
    )
}

/// Baseline A/V sync test application for automated drift measurement.
struct SyncBaselineApp {
    /// VideoPlayer instance, or None if wgpu not available.
    player: Option<VideoPlayer>,
    /// Instant when test playback began.
    start_time: Option<Instant>,
    /// Total duration to run the test.
    duration: Duration,
    /// Whether the test has finished.
    finished: bool,
    /// Instant of the last progress report.
    last_report: Instant,
    /// Duration between progress reports.
    report_interval: Duration,
}

impl SyncBaselineApp {
    fn new(cc: &eframe::CreationContext<'_>, video_source: &str, duration_secs: u64) -> Self {
        let player = cc.wgpu_render_state.as_ref().map(|render_state| {
            VideoPlayer::with_wgpu(video_source, render_state)
                .with_autoplay(true)
                .with_loop(false)
                .with_controls(false)
        });

        Self {
            player,
            start_time: None,
            duration: Duration::from_secs(duration_secs),
            finished: false,
            last_report: Instant::now(),
            report_interval: Duration::from_secs(5),
        }
    }

    fn print_progress(&mut self) {
        let Some(ref player) = self.player else {
            return;
        };

        let snap = player.sync_metrics_snapshot();
        let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();

        // Use steady-state metrics if available
        let (avg_drift, out_of_sync_pct, samples_label) = if snap.steady_samples > 0 {
            (
                snap.steady_avg_drift_ms(),
                snap.steady_out_of_sync_percentage(),
                format!("{}/{}", snap.steady_samples, snap.sample_count),
            )
        } else {
            (
                snap.avg_drift_us / 1000,
                snap.out_of_sync_percentage(),
                format!("{}", snap.sample_count),
            )
        };

        let recovery_indicator = if snap.in_recovery { " [R]" } else { "" };

        println!(
            "[{:>5.1}s] Drift: {:>+4}ms | Avg: {:>+4}ms | Steady/Total: {:>11} | OOS: {:>5.1}%{}",
            elapsed.as_secs_f64(),
            snap.current_drift_ms(),
            avg_drift,
            samples_label,
            out_of_sync_pct,
            recovery_indicator
        );
    }

    fn print_final_report(&self) {
        let Some(ref player) = self.player else {
            println!("ERROR: No player available");
            return;
        };

        let snap = player.sync_metrics_snapshot();
        let elapsed = self.start_time.map(|t| t.elapsed()).unwrap_or_default();

        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                   BASELINE RESULTS                           ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║ Test Duration:        {:>6.1} seconds                         ║",
            elapsed.as_secs_f64()
        );
        println!(
            "║ Frames Measured:      {:>6} (total)                          ║",
            snap.sample_count
        );
        println!(
            "║ Steady-State Frames:  {:>6} (excl. recovery)                 ║",
            snap.steady_samples
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ DRIFT METRICS (all samples)                                  ║");
        println!(
            "║   Current Drift:      {:>+6} ms                               ║",
            snap.current_drift_ms()
        );
        println!(
            "║   Max Drift (ahead):  {:>+6} ms                               ║",
            snap.max_drift_ahead_us / 1000
        );
        println!(
            "║   Max Drift (behind): {:>+6} ms                               ║",
            snap.max_drift_behind_us / 1000
        );
        println!(
            "║   Avg Abs Drift:      {:>+6} ms                               ║",
            snap.avg_drift_us / 1000
        );
        println!(
            "║   Out of Sync:        {:>6} ({:>5.1}%)                        ║",
            snap.out_of_sync_count,
            snap.out_of_sync_percentage()
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ STEADY-STATE METRICS (excluding recovery periods)            ║");
        println!(
            "║   Avg Abs Drift:      {:>+6} ms                               ║",
            snap.steady_avg_drift_ms()
        );
        println!(
            "║   Out of Sync:        {:>6} ({:>5.1}%)                        ║",
            snap.steady_out_of_sync_count,
            snap.steady_out_of_sync_percentage()
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ STALL/RECOVERY STATISTICS                                    ║");
        println!(
            "║   Underrun Events:    {:>6}                                  ║",
            snap.underrun_count
        );
        println!(
            "║   Stall Events:       {:>6} (decode: {}, network: {})       ║",
            snap.stall_count, snap.decode_stall_count, snap.network_stall_count
        );
        println!(
            "║   Recovery Periods:   {:>6}                                  ║",
            snap.recovery_count
        );
        println!(
            "║   Recovery Samples:   {:>6}                                  ║",
            snap.recovery_samples
        );
        if snap.recovery_samples > 0 {
            println!(
                "║   Recovery Avg Drift: {:>+6} ms                               ║",
                snap.recovery_avg_drift_us / 1000
            );
        }
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!("║ STREAM INFO                                                  ║");
        println!(
            "║   Audio Clock Used:   {:>6}                                  ║",
            if snap.using_audio_clock { "Yes" } else { "No" }
        );
        println!(
            "║   PTS Offset (A-V):   {:>+6} ms                               ║",
            snap.stream_pts_offset_ms()
        );
        println!("╠══════════════════════════════════════════════════════════════╣");

        let quality = snap.quality_summary();
        let passed = if snap.passed_sync_test() {
            "PASS"
        } else {
            "FAIL"
        };
        println!("║ Quality: {quality:<52} ║");
        println!("║ Result:  {passed:<52} ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Output JSON for programmatic parsing
        println!();
        println!("--- JSON Output ---");
        println!(
            r#"{{"duration_s":{:.1},"samples":{},"steady_samples":{},"current_drift_ms":{},"max_drift_ms":{},"avg_drift_ms":{},"steady_avg_drift_ms":{},"out_of_sync_pct":{:.2},"steady_out_of_sync_pct":{:.2},"underruns":{},"stalls":{},"recovery_count":{},"pts_offset_ms":{},"passed":{}}}"#,
            elapsed.as_secs_f64(),
            snap.sample_count,
            snap.steady_samples,
            snap.current_drift_ms(),
            snap.max_drift_ms(),
            snap.avg_drift_us / 1000,
            snap.steady_avg_drift_ms(),
            snap.out_of_sync_percentage(),
            snap.steady_out_of_sync_percentage(),
            snap.underrun_count,
            snap.stall_count,
            snap.recovery_count,
            snap.stream_pts_offset_ms(),
            snap.passed_sync_test()
        );
    }
}

impl eframe::App for SyncBaselineApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Start timer on first frame
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
            println!("Playback started, measuring sync...\n");
        }

        let elapsed = self.start_time.unwrap().elapsed();

        // Check if test is complete
        if elapsed >= self.duration && !self.finished {
            self.finished = true;
            self.print_final_report();
            println!("\nTest complete. Close window to exit.");
        }

        // Print progress periodically
        if !self.finished && self.last_report.elapsed() >= self.report_interval {
            self.print_progress();
            self.last_report = Instant::now();
        }

        // Render video
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref mut player) = self.player {
                let size = ui.available_size();
                ui.video_player(player, size);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("Initializing...");
                });
            }
        });

        // Request continuous repaints for video playback
        ctx.request_repaint();
    }
}
