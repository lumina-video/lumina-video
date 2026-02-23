//! lumina-video Demo Application
//!
//! A simple demo showcasing the lumina-video video player capabilities.

use eframe::{egui, egui_wgpu};
use lumina_video::{VideoPlayer, VideoPlayerExt};
use std::sync::mpsc;

#[cfg(feature = "moq")]
use lumina_video::media::{DiscoveryEvent, MoqStream, NostrDiscovery};

/// Sample videos for testing: (name, video_url, subtitle_url)
///
/// MP4/MOV → macOS AVPlayer native path (audio handled by AVPlayer, not cpal);
///           on Linux and Android these route through FFmpeg+cpal instead
/// MKV/WebM → FFmpeg decode path (audio through cpal ring buffer)
const SAMPLE_VIDEOS: &[(&str, &str, Option<&str>)] = &[
    (
        "Big Buck Bunny (MP4)",
        "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4",
        Some("https://raw.githubusercontent.com/demuxed/big-buck-captions/main/big-buck-bunny.srt"),
    ),
    (
        "Sintel (MP4)",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
        Some("https://durian.blender.org/wp-content/content/subtitles/sintel_en.srt"),
    ),
    (
        "Elephant's Dream (MP4)",
        "https://archive.org/download/ElephantsDream/ed_hd.mp4",
        None,
    ),
    // Tears of Steel has dialogue - good for A/V sync testing
    (
        "Tears of Steel (English)",
        "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        Some("https://download.blender.org/demo/movies/ToS/subtitles/TOS-en.srt"),
    ),
    // Non-Latin script subtitle tests
    (
        "Tears of Steel (日本語)",
        "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        Some("https://download.blender.org/demo/movies/ToS/subtitles/TOS-JP.srt"),
    ),
    (
        "Tears of Steel (中文)",
        "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        Some("https://download.blender.org/demo/movies/ToS/subtitles/TOS-CH-traditional.srt"),
    ),
    (
        "Tears of Steel (Русский)",
        "https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        Some("https://download.blender.org/demo/movies/ToS/subtitles/TOS-ru.srt"),
    ),
    (
        "Sample (MKV)",
        "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mkv-file.mkv",
        None,
    ),
];

/// Builds the list of MoQ test streams at runtime.
///
/// Uses the anonymous public endpoint (cdn.moq.dev/anon/) which doesn't require
/// authentication. The `demo` namespace requires JWT tokens from MOQ_DEMO_JWT.
fn get_moq_test_streams() -> Vec<(String, String)> {
    let mut streams = Vec::new();

    // Auto-discover: connect to /anon without specifying a broadcast
    // The decoder will automatically pick the first available broadcast
    streams.push((
        "cdn.moq.dev (auto-discover)".to_string(),
        "moqs://cdn.moq.dev:443/anon".to_string(),
    ));

    // Add the authenticated demo stream if JWT is available
    if let Ok(jwt) = std::env::var("MOQ_DEMO_JWT") {
        if !jwt.is_empty() {
            let demo_url = format!("moqs://cdn.moq.dev:443/demo/bbb?jwt={}", jwt);
            streams.push(("moq.dev BBB (official)".to_string(), demo_url));
        }
    }

    streams
}

/// Known MoQ relay endpoints for manual entry
const MOQ_RELAYS: &[(&str, &str)] = &[
    ("localhost", "moq://localhost:4443/anon"),
    ("cdn.moq.dev (anon)", "moqs://cdn.moq.dev:443/anon"),
    ("cdn.moq.dev", "moqs://cdn.moq.dev:443"),
    ("localhost (anon)", "moq://localhost:4443/anon"),
    ("zap.stream (US)", "moqs://api-core.zap.stream:1443"),
    ("zap.stream (UK)", "moqs://api-uk.zap.stream:1443"),
];

fn main() -> eframe::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("lumina_video=debug".parse().unwrap())
                .add_directive("lumina_video_demo=debug".parse().unwrap()),
        )
        .init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_title("lumina-video Demo"),
        ..Default::default()
    };

    eframe::run_native(
        "lumina-video Demo",
        options,
        Box::new(|cc| Ok(Box::new(DemoApp::new(cc)))),
    )
}

/// Video source type for unified UI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceType {
    Sample,
    MoqLive,
    CustomUrl,
}

/// Main application state for the lumina-video video demo.
///
/// Manages the video player instance, UI state, and user preferences.
struct DemoApp {
    /// The currently active VideoPlayer instance, or None if no video is loaded.
    player: Option<VideoPlayer>,
    /// The display URL of the currently loaded video, or None if no video is loaded.
    loaded_url: Option<String>,
    /// Selected source type
    source_type: SourceType,
    /// Previous source type (for auto-load on change)
    prev_source_type: SourceType,
    /// Custom URL input field
    custom_url: String,
    /// Selected sample video index
    selected_sample: usize,
    /// Show metadata panel
    show_metadata: bool,
    /// Channel receiver for subtitle content
    subtitle_receiver: Option<mpsc::Receiver<String>>,
    /// Measured FPS tracking
    fps_tracker: FpsTracker,
    /// Available MoQ test streams (built at runtime from env vars)
    moq_test_streams: Vec<(String, String)>,
    /// Selected MoQ test stream index
    selected_moq_stream: usize,
    /// Previous MoQ stream index (for auto-load on change)
    prev_selected_moq_stream: usize,
    /// Selected MoQ relay index (for manual entry)
    selected_moq_relay: usize,
    /// Manual broadcast name entry
    moq_broadcast_name: String,
    /// Show MoQ help window
    show_moq_help: bool,
    /// Nostr stream discovery service (kept alive; events come via nostr_event_rx)
    #[cfg(feature = "moq")]
    #[allow(dead_code)]
    nostr_discovery: Option<NostrDiscovery>,
    /// Receiver for nostr discovery events
    #[cfg(feature = "moq")]
    nostr_event_rx: Option<mpsc::Receiver<DiscoveryEvent>>,
    /// Discovered MoQ streams from Nostr
    #[cfg(feature = "moq")]
    discovered_streams: Vec<MoqStream>,
    /// Selected discovered stream index
    #[cfg(feature = "moq")]
    selected_discovered_stream: usize,
    /// Has Nostr discovery completed initial fetch?
    #[cfg(feature = "moq")]
    nostr_discovery_done: bool,
}

/// Tracks measured frame rate (actual rendering FPS)
struct FpsTracker {
    /// Frame timestamps for FPS calculation
    frame_times: Vec<std::time::Instant>,
    /// Last calculated FPS
    measured_fps: f32,
    /// Last frame number from player (to detect new frames)
    last_frame_count: u64,
}

impl FpsTracker {
    fn new() -> Self {
        Self {
            frame_times: Vec::with_capacity(120),
            measured_fps: 0.0,
            last_frame_count: 0,
        }
    }

    /// Update FPS measurement. Call once per UI frame with current player frame count.
    fn update(&mut self, current_frame_count: u64) {
        let now = std::time::Instant::now();

        // Only count if a new video frame was rendered
        if current_frame_count > self.last_frame_count {
            self.frame_times.push(now);
            self.last_frame_count = current_frame_count;
        }

        // Remove timestamps older than 1 second
        let one_second_ago = now - std::time::Duration::from_secs(1);
        self.frame_times.retain(|t| *t > one_second_ago);

        // Calculate FPS from frame count in last second
        self.measured_fps = self.frame_times.len() as f32;
    }

    /// Update FPS by counting every UI frame (simpler, doesn't need frame count from player)
    fn update_ui_frame(&mut self) {
        self.update(self.last_frame_count + 1);
    }

    fn fps(&self) -> f32 {
        self.measured_fps
    }

    /// Returns true if we have recorded any frame samples
    fn has_samples(&self) -> bool {
        self.last_frame_count > 0
    }

    fn reset(&mut self) {
        self.frame_times.clear();
        self.measured_fps = 0.0;
        self.last_frame_count = 0;
    }
}

impl DemoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Load font with CJK/Thai/Cyrillic support for subtitles
        // GoNotoKurrent covers Latin, Cyrillic, CJK, Thai, and most living scripts
        // License: SIL Open Font License (OFL)
        // Source: https://github.com/satbyy/go-noto-universal
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "noto".into(),
            egui::FontData::from_static(include_bytes!("../assets/GoNotoKurrent-Regular.ttf"))
                .into(),
        );
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "noto".into());
        cc.egui_ctx.set_fonts(fonts);

        // Start with the first sample video (with safe fallback)
        let first_sample = SAMPLE_VIDEOS.first();
        let player = cc.wgpu_render_state.as_ref().and_then(|render_state| {
            first_sample.map(|(_, url, _)| {
                VideoPlayer::with_wgpu(*url, render_state)
                    .with_autoplay(false)
                    .with_loop(true)
                    .with_controls(true)
            })
        });

        // Start fetching subtitles for the first video
        let subtitle_receiver =
            first_sample.and_then(|(_, _, sub_url)| sub_url.map(Self::fetch_subtitles));

        // Build MoQ test streams from environment variable
        let moq_test_streams = get_moq_test_streams();

        // Start Nostr discovery for MoQ streams
        #[cfg(feature = "moq")]
        let (nostr_discovery, nostr_event_rx) = {
            let mut discovery = NostrDiscovery::new();
            let rx = discovery.start();
            tracing::info!("Started Nostr MoQ stream discovery");
            (Some(discovery), Some(rx))
        };

        Self {
            player,
            loaded_url: first_sample.map(|(_, url, _)| url.to_string()),
            source_type: SourceType::Sample,
            prev_source_type: SourceType::Sample,
            custom_url: String::new(),
            selected_sample: 0,
            show_metadata: true,
            subtitle_receiver,
            fps_tracker: FpsTracker::new(),
            moq_test_streams,
            selected_moq_stream: 0,
            prev_selected_moq_stream: 0,
            selected_moq_relay: 0,
            moq_broadcast_name: String::new(),
            show_moq_help: false,
            #[cfg(feature = "moq")]
            nostr_discovery,
            #[cfg(feature = "moq")]
            nostr_event_rx,
            #[cfg(feature = "moq")]
            discovered_streams: Vec::new(),
            #[cfg(feature = "moq")]
            selected_discovered_stream: 0,
            #[cfg(feature = "moq")]
            nostr_discovery_done: false,
        }
    }

    fn fetch_subtitles(url: &str) -> mpsc::Receiver<String> {
        let (tx, rx) = mpsc::channel();
        let request = ehttp::Request::get(url);
        ehttp::fetch(request, move |result| match result {
            Ok(response) if response.ok => {
                if let Some(content) = response.text() {
                    let _ = tx.send(content.to_string());
                }
            }
            Ok(response) => {
                tracing::warn!("Failed to fetch subtitles: HTTP {}", response.status);
            }
            Err(e) => {
                tracing::warn!("Failed to fetch subtitles: {}", e);
            }
        });
        rx
    }

    fn load_video(
        &mut self,
        url: &str,
        subtitle_url: Option<&str>,
        render_state: &egui_wgpu::RenderState,
    ) {
        let is_live = url.starts_with("moq://") || url.starts_with("moqs://");
        self.player = Some(
            VideoPlayer::with_wgpu(url, render_state)
                .with_autoplay(true)
                .with_loop(!is_live)
                .with_controls(true),
        );

        // Track what URL is loaded for display
        self.loaded_url = Some(url.to_string());

        // Reset FPS tracker for new video
        self.fps_tracker.reset();

        // Start fetching subtitles if URL provided
        self.subtitle_receiver = subtitle_url.map(Self::fetch_subtitles);
    }

    /// Determine the type of media from URL
    fn url_source_type(url: &str) -> &'static str {
        if url.starts_with("moq://") || url.starts_with("moqs://") {
            "MoQ Live"
        } else if url.ends_with(".m3u8") {
            "HLS"
        } else if url.ends_with(".mpd") {
            "DASH"
        } else if url.starts_with("http://") || url.starts_with("https://") {
            "HTTP"
        } else if url.starts_with("file://") || url.starts_with("/") {
            "Local File"
        } else {
            "Unknown"
        }
    }

    fn check_subtitle_fetch(&mut self) {
        let Some(ref receiver) = self.subtitle_receiver else {
            return;
        };

        // Non-blocking check for subtitle content
        match receiver.try_recv() {
            Ok(content) => {
                if let Some(ref mut player) = self.player {
                    if let Err(e) = player.load_subtitles_srt(&content) {
                        tracing::warn!("Failed to parse subtitles: {:?}", e);
                    } else {
                        tracing::info!("Subtitles loaded successfully ({} bytes)", content.len());
                    }
                }
                // Clear the receiver after processing
                self.subtitle_receiver = None;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                // Channel closed without receiving content (fetch failed)
                self.subtitle_receiver = None;
            }
            Err(mpsc::TryRecvError::Empty) => {
                // Still waiting for content, keep polling
            }
        }
    }

    /// Poll for Nostr discovery events and update discovered streams list.
    #[cfg(feature = "moq")]
    fn poll_nostr_discovery(&mut self) {
        if let Some(ref rx) = self.nostr_event_rx {
            // Non-blocking poll for discovery events
            while let Ok(event) = rx.try_recv() {
                match event {
                    DiscoveryEvent::StreamUpdated(stream) => {
                        tracing::info!(
                            "Discovered MoQ stream: {} at {}",
                            stream.title.as_deref().unwrap_or(&stream.id),
                            stream.url
                        );
                        // Update or add stream
                        if let Some(existing) = self
                            .discovered_streams
                            .iter_mut()
                            .find(|s| s.id == stream.id)
                        {
                            *existing = stream;
                        } else {
                            self.discovered_streams.push(stream);
                        }
                        // Mark discovery as done since we found at least one stream
                        self.nostr_discovery_done = true;
                    }
                    DiscoveryEvent::StreamEnded(id) => {
                        tracing::info!("Stream ended: {}", id);
                        self.discovered_streams.retain(|s| s.id != id);
                    }
                    DiscoveryEvent::Connected(connected) => {
                        tracing::info!("Nostr discovery connected: {}", connected);
                        // Initial fetch is done when we get the connected event
                        self.nostr_discovery_done = true;
                    }
                    DiscoveryEvent::Error(e) => {
                        tracing::error!("Nostr discovery error: {}", e);
                        // Also mark as done on error so we don't show "searching" forever
                        self.nostr_discovery_done = true;
                    }
                }
            }
        }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Poll for Nostr discovery events
        #[cfg(feature = "moq")]
        self.poll_nostr_discovery();

        // Check if subtitle fetch completed
        self.check_subtitle_fetch();

        // Update FPS tracker when video is playing
        if let Some(ref player) = self.player {
            if player.is_playing() {
                // Use actual decoded frame count when available (MoQ stats),
                // otherwise fall back to UI frame counting.
                #[cfg(feature = "moq")]
                {
                    if let Some(moq) = player.moq_stats() {
                        self.fps_tracker.update(moq.frame_stats.rendered);
                    } else {
                        self.fps_tracker.update_ui_frame();
                    }
                }
                #[cfg(not(feature = "moq"))]
                {
                    self.fps_tracker.update_ui_frame();
                }
            }
        }

        // Top panel with controls
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(format!(
                    "lumina-video Demo  [{}]",
                    option_env!("LUMINA_BUILD_ID").unwrap_or("dev")
                ));
                ui.separator();

                // Source type selector
                ui.label("Source:");
                egui::ComboBox::from_id_salt("source_type")
                    .selected_text(match self.source_type {
                        SourceType::Sample => "Sample Videos",
                        SourceType::MoqLive => "MoQ Live Stream",
                        SourceType::CustomUrl => "Custom URL",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.source_type,
                            SourceType::Sample,
                            "Sample Videos",
                        );
                        ui.selectable_value(
                            &mut self.source_type,
                            SourceType::MoqLive,
                            "MoQ Live Stream",
                        );
                        ui.selectable_value(
                            &mut self.source_type,
                            SourceType::CustomUrl,
                            "Custom URL",
                        );
                    });

                ui.separator();
                ui.checkbox(&mut self.show_metadata, "Show Metadata");
            });

            // Source-specific controls
            ui.horizontal(|ui| {
                match self.source_type {
                    SourceType::Sample => {
                        // Sample video selector
                        egui::ComboBox::from_id_salt("sample_selector")
                            .selected_text(SAMPLE_VIDEOS[self.selected_sample].0)
                            .show_ui(ui, |ui| {
                                for (i, (name, _, _)) in SAMPLE_VIDEOS.iter().enumerate() {
                                    ui.selectable_value(&mut self.selected_sample, i, *name);
                                }
                            });

                        if ui.button("Load").clicked() {
                            let (_, url, subtitle_url) = SAMPLE_VIDEOS[self.selected_sample];
                            if let Some(render_state) = frame.wgpu_render_state() {
                                self.load_video(url, subtitle_url, render_state);
                            }
                        }
                    }
                    SourceType::MoqLive => {
                        // Show test streams first (auto-discover)
                        if !self.moq_test_streams.is_empty() {
                            let selected_name = self
                                .moq_test_streams
                                .get(self.selected_moq_stream)
                                .map(|(name, _)| name.as_str())
                                .unwrap_or("(none)");

                            egui::ComboBox::from_id_salt("moq_stream_selector")
                                .selected_text(selected_name)
                                .show_ui(ui, |ui| {
                                    for (i, (name, _)) in self.moq_test_streams.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut self.selected_moq_stream,
                                            i,
                                            name.as_str(),
                                        );
                                    }
                                });
                        }

                        // Show discovered streams from Nostr (NIP-53)
                        #[cfg(feature = "moq")]
                        {
                            let stream_count = self.discovered_streams.len();
                            ui.separator();

                            if stream_count > 0 {
                                let selected_name = self
                                    .discovered_streams
                                    .get(self.selected_discovered_stream)
                                    .map(|s| s.title.as_deref().unwrap_or(&s.id))
                                    .unwrap_or("(none)");

                                ui.label(format!("Nostr Live ({}):", stream_count));
                                egui::ComboBox::from_id_salt("discovered_stream_selector")
                                    .selected_text(selected_name)
                                    .show_ui(ui, |ui| {
                                        for (i, stream) in
                                            self.discovered_streams.iter().enumerate()
                                        {
                                            let label =
                                                stream.title.as_deref().unwrap_or(&stream.id);
                                            ui.selectable_value(
                                                &mut self.selected_discovered_stream,
                                                i,
                                                label,
                                            );
                                        }
                                    });

                                if ui.button("Play").clicked() {
                                    if let Some(stream) =
                                        self.discovered_streams.get(self.selected_discovered_stream)
                                    {
                                        let url = stream.url.clone();
                                        if let Some(render_state) = frame.wgpu_render_state() {
                                            self.load_video(&url, None, render_state);
                                        }
                                    }
                                }
                            } else if self.nostr_discovery_done {
                                ui.label("No live MoQ streams on Nostr");
                            } else {
                                ui.label("Searching Nostr...");
                            }
                        }

                        ui.separator();

                        // Manual broadcast entry
                        ui.label("Or enter broadcast:");
                        let selected_relay_name = MOQ_RELAYS
                            .get(self.selected_moq_relay)
                            .map(|(name, _)| *name)
                            .unwrap_or("Unknown");
                        egui::ComboBox::from_id_salt("moq_relay_selector")
                            .selected_text(selected_relay_name)
                            .width(120.0)
                            .show_ui(ui, |ui| {
                                for (i, (name, _)) in MOQ_RELAYS.iter().enumerate() {
                                    ui.selectable_value(&mut self.selected_moq_relay, i, *name);
                                }
                            });

                        ui.add(
                            egui::TextEdit::singleline(&mut self.moq_broadcast_name)
                                .hint_text("broadcast name")
                                .desired_width(100.0),
                        );

                        let can_load = !self.moq_broadcast_name.trim().is_empty();
                        if ui.add_enabled(can_load, egui::Button::new("Go")).clicked() {
                            let relay = MOQ_RELAYS
                                .get(self.selected_moq_relay)
                                .map(|(_, url)| *url)
                                .or_else(|| MOQ_RELAYS.first().map(|(_, url)| *url))
                                .expect("MOQ_RELAYS is empty");
                            let url = format!("{}/{}", relay, self.moq_broadcast_name.trim());
                            if let Some(render_state) = frame.wgpu_render_state() {
                                self.load_video(&url, None, render_state);
                            }
                        }

                        // Help button
                        if ui.button("?").clicked() {
                            self.show_moq_help = true;
                        }
                    }
                    SourceType::CustomUrl => {
                        // Custom URL input
                        let response = ui.add(
                            egui::TextEdit::singleline(&mut self.custom_url)
                                .hint_text("https://example.com/video.mp4")
                                .desired_width(400.0),
                        );

                        if ui.button("Load").clicked()
                            || (response.lost_focus()
                                && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        {
                            if let Some(render_state) = frame.wgpu_render_state() {
                                self.load_video(&self.custom_url.clone(), None, render_state);
                            }
                        }
                    }
                }
            });
        });

        // Auto-load MoQ stream when source type changes to MoQ or stream selection changes
        let switched_to_moq =
            self.source_type == SourceType::MoqLive && self.prev_source_type != SourceType::MoqLive;
        let moq_stream_changed = self.source_type == SourceType::MoqLive
            && self.selected_moq_stream != self.prev_selected_moq_stream;

        if switched_to_moq || moq_stream_changed {
            if let Some((_, url)) = self.moq_test_streams.get(self.selected_moq_stream) {
                let url = url.clone();
                if let Some(render_state) = frame.wgpu_render_state() {
                    self.load_video(&url, None, render_state);
                }
            }
        }

        // Update previous values for next frame
        self.prev_source_type = self.source_type;
        self.prev_selected_moq_stream = self.selected_moq_stream;

        // Right panel with metadata (optional)
        if self.show_metadata {
            egui::SidePanel::right("metadata_panel")
                .min_width(200.0)
                .show(ctx, |ui| {
                    ui.heading("Video Info");
                    ui.separator();

                    if let Some(ref player) = self.player {
                        if let Some(metadata) = player.metadata() {
                            egui::Grid::new("metadata_grid")
                                .num_columns(2)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    // Show source type based on loaded URL
                                    ui.label("Source:");
                                    if let Some(ref url) = self.loaded_url {
                                        let source_type = Self::url_source_type(url);
                                        let color = if source_type == "MoQ Live" {
                                            egui::Color32::from_rgb(0, 200, 100)
                                        } else {
                                            egui::Color32::GRAY
                                        };
                                        ui.colored_label(color, source_type);
                                    } else {
                                        ui.label("Unknown");
                                    }
                                    ui.end_row();

                                    ui.label("Resolution:");
                                    ui.label(format!("{}x{}", metadata.width, metadata.height));
                                    ui.end_row();

                                    ui.label("Aspect Ratio:");
                                    let aspect = metadata.aspect_ratio();
                                    // Show common aspect ratios by name
                                    let aspect_name = if (aspect - 16.0/9.0).abs() < 0.01 {
                                        "16:9".to_string()
                                    } else if (aspect - 4.0/3.0).abs() < 0.01 {
                                        "4:3".to_string()
                                    } else if (aspect - 21.0/9.0).abs() < 0.05 {
                                        "21:9".to_string()
                                    } else if (aspect - 1.0).abs() < 0.01 {
                                        "1:1".to_string()
                                    } else {
                                        format!("{:.2}:1", aspect)
                                    };
                                    ui.label(aspect_name);
                                    ui.end_row();

                                    ui.label("Codec:");
                                    ui.label(&metadata.codec);
                                    ui.end_row();

                                    ui.label("Pixel Format:");
                                    // Derive format info from codec name
                                    let format = if metadata.codec.contains("h264") || metadata.codec.contains("avc") {
                                        "YUV 4:2:0 (H.264)"
                                    } else if metadata.codec.contains("hevc") || metadata.codec.contains("h265") {
                                        "YUV 4:2:0 (HEVC)"
                                    } else if metadata.codec.contains("vp9") {
                                        "YUV 4:2:0 (VP9)"
                                    } else if metadata.codec.contains("av1") {
                                        "YUV 4:2:0 (AV1)"
                                    } else {
                                        "YUV 4:2:0"
                                    };
                                    ui.label(format);
                                    ui.end_row();

                                    ui.label("Frame Rate:");
                                    ui.label(format!("{:.2} fps", metadata.frame_rate));
                                    ui.end_row();

                                    // Measured FPS (actual frames rendered)
                                    if self.fps_tracker.has_samples() {
                                        ui.label("Measured FPS:");
                                        let measured = self.fps_tracker.fps();
                                        let fps_color = if measured >= metadata.frame_rate * 0.95 {
                                            egui::Color32::GREEN
                                        } else if measured >= metadata.frame_rate * 0.8 {
                                            egui::Color32::from_rgb(255, 165, 0) // orange
                                        } else {
                                            egui::Color32::RED
                                        };
                                        ui.colored_label(fps_color, format!("{:.1} fps", measured));
                                        ui.end_row();
                                    }

                                    if let Some(duration) = metadata.duration {
                                        ui.label("Duration:");
                                        let secs = duration.as_secs();
                                        let mins = secs / 60;
                                        let hours = mins / 60;
                                        if hours > 0 {
                                            ui.label(format!("{}:{:02}:{:02}", hours, mins % 60, secs % 60));
                                        } else {
                                            ui.label(format!("{}:{:02}", mins, secs % 60));
                                        }
                                        ui.end_row();
                                    }

                                    // Bitrate estimate (if duration known)
                                    // This is approximate since we don't have file size
                                    if metadata.pixel_aspect_ratio != 1.0 {
                                        ui.label("Pixel AR:");
                                        ui.label(format!("{:.2}", metadata.pixel_aspect_ratio));
                                        ui.end_row();
                                    }
                                });
                        } else {
                            ui.label("Loading metadata...");
                        }

                        ui.separator();
                        ui.heading("Playback");

                        egui::Grid::new("playback_grid")
                            .num_columns(2)
                            .spacing([10.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("State:");
                                let state_str = format!("{:?}", player.state());
                                // Truncate position in state to 2 decimals
                                let state_display = if state_str.contains("position:") {
                                    state_str.split("position:").collect::<Vec<_>>().into_iter().enumerate().map(|(i, part)| {
                                        if i == 0 {
                                            part.to_string()
                                        } else {
                                            let trimmed = part.trim_start();
                                            if let Some(pos) = trimmed.split_whitespace().next() {
                                                // Try parsing as simple float (e.g., "10.5s")
                                                if let Ok(num) = pos.trim_end_matches('s').parse::<f64>() {
                                                    format!("position: {:.2}s{}", num, &trimmed[pos.len()..])
                                                } else if trimmed.contains("Duration {") {
                                                    // Try parsing Debug format: "Duration { secs: X, nanos: Y }"
                                                    let secs = trimmed
                                                        .split("secs:")
                                                        .nth(1)
                                                        .and_then(|s| s.split(',').next())
                                                        .and_then(|s| s.trim().parse::<u64>().ok());
                                                    let nanos = trimmed
                                                        .split("nanos:")
                                                        .nth(1)
                                                        .and_then(|s| s.split([',', '}']).next())
                                                        .and_then(|s| s.trim().parse::<u32>().ok());
                                                    if let (Some(s), Some(n)) = (secs, nanos) {
                                                        let seconds = s as f64 + n as f64 / 1_000_000_000.0;
                                                        // Find where Duration block ends
                                                        let end_pos = trimmed.find('}').map(|p| p + 1).unwrap_or(trimmed.len());
                                                        format!("position: {:.2}s{}", seconds, &trimmed[end_pos..])
                                                    } else {
                                                        format!("position:{}", part)
                                                    }
                                                } else {
                                                    format!("position:{}", part)
                                                }
                                            } else {
                                                part.to_string()
                                            }
                                        }
                                    }).collect::<Vec<_>>().join("")
                                } else {
                                    state_str
                                };
                                ui.label(state_display);
                                ui.end_row();

                                ui.label("Position:");
                                ui.label(format!("{:.2}s", player.position().as_secs_f64()));
                                ui.end_row();

                                ui.label("Playing:");
                                ui.label(if player.is_playing() { "Yes" } else { "No" });
                                ui.end_row();

                                ui.label("Buffering:");
                                ui.label(format!("{}%", player.buffering_percent()));
                                ui.end_row();
                            });

                        // MoQ Pipeline Stats
                        #[cfg(feature = "moq")]
                        if let Some(moq) = player.moq_stats() {
                            ui.separator();
                            ui.heading("MoQ Pipeline");

                            egui::Grid::new("moq_grid")
                                .num_columns(2)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    ui.label("State:");
                                    let (state_text, state_color) = match moq.state {
                                        lumina_video::media::MoqDecoderState::Disconnected => {
                                            ("Disconnected", egui::Color32::GRAY)
                                        }
                                        lumina_video::media::MoqDecoderState::Connecting => {
                                            ("Connecting...", egui::Color32::from_rgb(200, 120, 0))
                                        }
                                        lumina_video::media::MoqDecoderState::FetchingCatalog => {
                                            ("Fetching Catalog", egui::Color32::from_rgb(200, 120, 0))
                                        }
                                        lumina_video::media::MoqDecoderState::Streaming => {
                                            ("Streaming", egui::Color32::GREEN)
                                        }
                                        lumina_video::media::MoqDecoderState::Ended => {
                                            ("Ended", egui::Color32::GRAY)
                                        }
                                        lumina_video::media::MoqDecoderState::Error => {
                                            ("Error", egui::Color32::RED)
                                        }
                                    };
                                    ui.colored_label(state_color, state_text);
                                    ui.end_row();

                                    // Transport protocol
                                    ui.label("Transport:");
                                    let proto_color = match moq.transport_protocol.as_str() {
                                        "QUIC" => egui::Color32::GREEN,
                                        "WebSocket" => egui::Color32::from_rgb(200, 120, 0),
                                        _ => egui::Color32::GRAY,
                                    };
                                    ui.colored_label(proto_color, &moq.transport_protocol);
                                    ui.end_row();

                                    if let Some(ref err) = moq.error_message {
                                        ui.label("Error:");
                                        ui.colored_label(egui::Color32::RED, err);
                                        ui.end_row();
                                    }

                                    if moq.has_codec_description {
                                        ui.label("SPS/PPS:");
                                        ui.colored_label(egui::Color32::GREEN, "present");
                                    } else {
                                        ui.label("SPS/PPS:");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(200, 100, 0),
                                            "missing (extracting from stream)",
                                        );
                                    }
                                    ui.end_row();

                                    let fs = &moq.frame_stats;
                                    ui.label("Received:");
                                    ui.label(format!("{}", fs.received));
                                    ui.end_row();

                                    ui.label("Submitted:");
                                    ui.label(format!("{}", fs.submitted_to_decoder));
                                    ui.end_row();

                                    ui.label("Decoded:");
                                    ui.label(format!("{}", fs.decoded));
                                    ui.end_row();

                                    ui.label("Rendered:");
                                    ui.label(format!("{}", fs.rendered));
                                    ui.end_row();

                                    // Show drops/errors only when non-zero
                                    if fs.dropped_backpressure > 0 {
                                        ui.label("Drop (backpressure):");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(255, 140, 0),
                                            format!("{}", fs.dropped_backpressure),
                                        );
                                        ui.end_row();
                                    }
                                    if fs.dropped_waiting_idr > 0 {
                                        ui.label("Drop (no IDR):");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(255, 140, 0),
                                            format!("{}", fs.dropped_waiting_idr),
                                        );
                                        ui.end_row();
                                    }
                                    if fs.dropped_dpb_grace > 0 {
                                        ui.label("DPB Grace:");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(255, 200, 100),
                                            format!("{}", fs.dropped_dpb_grace),
                                        );
                                        ui.end_row();
                                    }
                                    if fs.skipped_startup_frames > 0 {
                                        ui.label("Skip (startup):");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(255, 140, 0),
                                            format!("{}", fs.skipped_startup_frames),
                                        );
                                        ui.end_row();
                                    }
                                    if fs.decode_errors > 0 {
                                        ui.label("Decode errors:");
                                        ui.colored_label(
                                            egui::Color32::RED,
                                            format!("{}", fs.decode_errors),
                                        );
                                        ui.end_row();
                                    }
                                    // Ring buffer metrics
                                    if moq.ring_buffer_fill_percent > 0.0 {
                                        ui.label("Ring buffer:");
                                        let fill_color = if moq.ring_buffer_fill_percent > 90.0 {
                                            egui::Color32::from_rgb(255, 140, 0)
                                        } else {
                                            egui::Color32::GREEN
                                        };
                                        ui.colored_label(
                                            fill_color,
                                            format!("{:.0}%", moq.ring_buffer_fill_percent),
                                        );
                                        ui.end_row();
                                    }
                                    if moq.ring_buffer_overflow_count > 0 {
                                        ui.label("RB overflows:");
                                        ui.colored_label(
                                            egui::Color32::from_rgb(255, 140, 0),
                                            format!("{}", moq.ring_buffer_overflow_count),
                                        );
                                        ui.end_row();
                                    }

                                    // Audio status
                                    ui.label("Audio:");
                                    let (audio_label, audio_color) = match moq.audio_status {
                                        lumina_video::media::MoqAudioStatus::Unavailable => {
                                            ("Unavailable", egui::Color32::GRAY)
                                        }
                                        lumina_video::media::MoqAudioStatus::Starting => {
                                            ("Starting...", egui::Color32::YELLOW)
                                        }
                                        lumina_video::media::MoqAudioStatus::Running => {
                                            ("Running", egui::Color32::GREEN)
                                        }
                                        lumina_video::media::MoqAudioStatus::Error => {
                                            ("Error", egui::Color32::RED)
                                        }
                                    };
                                    ui.colored_label(audio_color, audio_label);
                                    ui.end_row();

                                    // Audio codec
                                    if let Some(ref codec) = moq.audio_codec {
                                        ui.label("Audio codec:");
                                        ui.label(codec.as_str());
                                        ui.end_row();
                                    }
                                });
                        }

                        // A/V Sync Metrics
                        ui.separator();
                        ui.heading("A/V Sync");

                        let sync = player.sync_metrics_snapshot();
                        egui::Grid::new("sync_grid")
                            .num_columns(2)
                            .spacing([10.0, 4.0])
                            .show(ui, |ui| {
                                ui.label("Drift:");
                                if sync.sync_externally_managed {
                                    // Native player handles A/V sync internally
                                    ui.colored_label(egui::Color32::GREEN, "native (external)");
                                } else {
                                    let drift = sync.current_drift_ms();
                                    let drift_color = if drift.abs() < 40 {
                                        egui::Color32::GREEN
                                    } else if drift.abs() < 80 {
                                        // Use orange instead of yellow for better contrast on light backgrounds
                                        egui::Color32::from_rgb(255, 140, 0)
                                    } else {
                                        egui::Color32::RED
                                    };
                                    ui.colored_label(drift_color, format!("{:+}ms", drift));
                                }
                                ui.end_row();

                                ui.label("Max Drift:");
                                if sync.sync_externally_managed {
                                    ui.label("n/a");
                                } else {
                                    ui.label(format!("{:+}ms", sync.max_drift_ms()));
                                }
                                ui.end_row();

                                ui.label("Samples:");
                                ui.label(format!("{}", sync.sample_count));
                                ui.end_row();

                                ui.label("Out of Sync:");
                                ui.label(format!("{:.1}%", sync.out_of_sync_percentage()));
                                ui.end_row();

                                ui.label("Quality:");
                                ui.label(if sync.sample_count == 0 {
                                    "N/A (native)" // Native decoder handles A/V sync internally
                                } else if sync.passed_sync_test() {
                                    "PASS"
                                } else {
                                    "FAIL"
                                });
                                ui.end_row();

                                ui.label("Real FPS:");
                                ui.label(format!("{:.1}", sync.current_fps));
                                ui.end_row();
                            });

                        // Zero-Copy Status (Linux only)
                        #[cfg(target_os = "linux")]
                        {
                            ui.separator();
                            ui.heading("Zero-Copy Status");

                            if let Some(metrics) = player.linux_zero_copy_metrics() {
                                let total = metrics.total_frames();

                                // Update measured FPS tracker with total frame count
                                self.fps_tracker.update(total);
                                let (status_color, status_text, detail_text) = if total == 0 {
                                    (
                                        egui::Color32::GRAY,
                                        "Initializing",
                                        "Waiting for frames...".to_string(),
                                    )
                                } else if metrics.is_zero_copy_active() {
                                    (
                                        egui::Color32::from_rgb(0, 200, 0),
                                        "✓ Active",
                                        format!(
                                            "{} frames via DMABuf → Vulkan\n100% zero-copy, no CPU copies",
                                            metrics.zero_copy_frames
                                        ),
                                    )
                                } else if metrics.fallback_frames == total {
                                    (
                                        egui::Color32::from_rgb(255, 140, 0),
                                        "⚠ CPU Fallback",
                                        format!(
                                            "{} frames via CPU copy\nDMABuf import failed",
                                            metrics.fallback_frames
                                        ),
                                    )
                                } else {
                                    (
                                        egui::Color32::from_rgb(255, 200, 0),
                                        "⚠ Partial",
                                        format!(
                                            "{}/{} zero-copy ({:.1}%)\n{} CPU fallback frames",
                                            metrics.zero_copy_frames,
                                            total,
                                            metrics.zero_copy_percentage(),
                                            metrics.fallback_frames
                                        ),
                                    )
                                };

                                ui.horizontal(|ui| {
                                    // Draw colored square indicator
                                    let size = egui::Vec2::splat(12.0);
                                    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                                    ui.painter().rect_filled(rect, 2.0, status_color);

                                    ui.label(egui::RichText::new(status_text).strong().color(status_color));
                                });

                                ui.label(egui::RichText::new(detail_text).small().weak());
                            } else {
                                ui.label(egui::RichText::new("Metrics unavailable").small().weak());
                            }

                            // Audio Sink Info
                            ui.separator();
                            ui.heading("Audio Sink");

                            // Default is now alsasink for reliable HTTP seek
                            let audio_sink = if std::env::var("EGUI_VID_NO_AUDIO").is_ok() {
                                ("disabled", egui::Color32::GRAY)
                            } else if std::env::var("EGUI_VID_PULSE_AUDIO").is_ok() {
                                ("pulsesink", egui::Color32::from_rgb(255, 140, 0))
                            } else if std::env::var("EGUI_VID_PIPEWIRE_AUDIO").is_ok() {
                                ("pipewiresink", egui::Color32::YELLOW)
                            } else if std::env::var("EGUI_VID_FAKE_AUDIO").is_ok() {
                                ("fakesink", egui::Color32::GRAY)
                            } else {
                                ("alsasink (default)", egui::Color32::GREEN)
                            };

                            ui.horizontal(|ui| {
                                let size = egui::Vec2::splat(12.0);
                                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                                ui.painter().rect_filled(rect, 2.0, audio_sink.1);
                                ui.label(egui::RichText::new(audio_sink.0).strong().color(audio_sink.1));
                            });

                            ui.label(egui::RichText::new(
                                "alsasink: reliable seek, shares audio via dmix"
                            ).small().weak());
                        }
                    } else {
                        ui.label("No video loaded");
                    }
                });
        }

        // MoQ help window
        if self.show_moq_help {
            egui::Window::new("MoQ Live Streaming Help")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label("MoQ (Media over QUIC) enables ultra-low-latency live streaming.");
                    ui.add_space(8.0);

                    ui.heading("How it works");
                    ui.label("• MoQ provides <1 second latency vs 10-30s for HLS");
                    ui.label("• Streams are live and ephemeral - they disappear when offline");
                    ui.add_space(8.0);

                    ui.heading("Quick Start");
                    ui.label("1. Select 'cdn.moq.dev (anon)' relay");
                    ui.label("2. Visit https://moq.dev/watch in browser");
                    ui.label("3. Find an active broadcast name (e.g., 'silly-mink')");
                    ui.label("4. Enter that name in the Broadcast field");
                    ui.label("5. Click Go");
                    ui.add_space(8.0);

                    ui.heading("Authentication");
                    ui.label("• /anon - Public anonymous broadcasts (any user)");
                    ui.label("• /demo - Official demos (requires MOQ_DEMO_JWT env var)");
                    ui.label("• Set MOQ_DEMO_JWT to watch demo/bbb stream");
                    ui.add_space(8.0);

                    ui.heading("Relays");
                    ui.label("• cdn.moq.dev (anon) - Public anonymous access");
                    ui.label("• zap.stream - Nostr streaming (NIP-53)");
                    ui.add_space(8.0);

                    if ui.button("Close").clicked() {
                        self.show_moq_help = false;
                    }
                });
        }

        // Central panel with video
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref mut player) = self.player {
                let available_size = ui.available_size();

                // Maintain aspect ratio
                let video_size = if let Some(metadata) = player.metadata() {
                    // Handle case where metadata dimensions aren't yet available (MoQ streams)
                    let aspect = if metadata.width > 0 && metadata.height > 0 {
                        metadata.width as f32 / metadata.height as f32
                    } else {
                        16.0 / 9.0 // Default aspect ratio until metadata is available
                    };
                    let max_width = available_size.x;
                    let max_height = available_size.y;

                    if max_width / aspect <= max_height {
                        egui::vec2(max_width, max_width / aspect)
                    } else {
                        egui::vec2(max_height * aspect, max_height)
                    }
                } else {
                    // Default 16:9 while loading
                    let aspect = 16.0 / 9.0;
                    egui::vec2(available_size.x, available_size.x / aspect).min(available_size)
                };

                // Center the video
                ui.centered_and_justified(|ui| {
                    ui.video_player(player, video_size);
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("No video loaded");
                    ui.label("Select a sample video or enter a URL above");
                });
            }
        });
    }
}
