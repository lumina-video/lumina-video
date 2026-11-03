use std::time::{Duration, Instant};

use lumina_video::VideoPlayer;

struct PlayUrl {
    player: VideoPlayer,
    started: Instant,
    last_report: Instant,
    frame_times: Vec<f32>,
    ui_times: Vec<f32>,
    last_frame: Instant,
    exit_after: Option<Duration>,
}

impl eframe::App for PlayUrl {
    fn ui(&mut self, root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let now = Instant::now();
        self.frame_times
            .push(now.duration_since(self.last_frame).as_secs_f32() * 1000.0);
        self.last_frame = now;

        egui::CentralPanel::default().show(root, |ui| {
            let size = ui.available_size();
            self.player.show(ui, size);
        });
        self.ui_times.push(now.elapsed().as_secs_f32() * 1000.0);

        if now.duration_since(self.last_report) >= Duration::from_secs(5) {
            self.report();
            self.last_report = now;
        }
        if let Some(limit) = self.exit_after {
            if now.duration_since(self.started) >= limit {
                self.report();
                std::process::exit(0);
            }
        }
        root.ctx().request_repaint();
    }
}

impl PlayUrl {
    fn report(&mut self) {
        let mut t = std::mem::take(&mut self.frame_times);
        let mut u = std::mem::take(&mut self.ui_times);
        if t.is_empty() || u.is_empty() {
            return;
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        u.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = t.iter().sum::<f32>() / t.len() as f32;
        let p = |q: f32| t[((t.len() - 1) as f32 * q) as usize];
        let ui_mean = u.iter().sum::<f32>() / u.len() as f32;
        let ui_p95 = u[((u.len() - 1) as f32 * 0.95) as usize];
        let sync = self.player.sync_metrics_snapshot();
        #[cfg(target_os = "linux")]
        let zc = self
            .player
            .linux_zero_copy_metrics()
            .map(|m| {
                format!(
                    " zero_copy={} fallback={}",
                    m.zero_copy_frames, m.fallback_frames
                )
            })
            .unwrap_or_default();
        #[cfg(not(target_os = "linux"))]
        let zc = String::new();
        println!(
            "frames={} mean={:.2}ms p50={:.2}ms p95={:.2}ms p99={:.2}ms max={:.2}ms ui_mean={:.3}ms ui_p95={:.3}ms state={:?} dims={:?} {:?}{}",
            t.len(),
            mean,
            p(0.5),
            p(0.95),
            p(0.99),
            p(1.0),
            ui_mean,
            ui_p95,
            self.player.state(),
            self.player.dimensions(),
            sync,
            zc
        );
    }
}

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let url = std::env::args()
        .nth(1)
        .expect("usage: play_url <url> [seconds]");
    let exit_after = std::env::args()
        .nth(2)
        .map(|s| Duration::from_secs(s.parse().expect("seconds")));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "play_url",
        options,
        Box::new(move |cc| {
            let rs = cc.wgpu_render_state.as_ref().expect("wgpu");
            let player = VideoPlayer::with_wgpu(url, rs).with_autoplay(true);
            let now = Instant::now();
            Ok(Box::new(PlayUrl {
                player,
                started: now,
                last_report: now,
                frame_times: Vec::new(),
                ui_times: Vec::new(),
                last_frame: now,
                exit_after,
            }))
        }),
    )
}
