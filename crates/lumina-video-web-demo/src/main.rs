//! lumina-video Web Demo Application
//!
//! A browser-based demo showcasing the lumina-video video player capabilities.
//! Follows Apple Human Interface Guidelines (HIG) for video players.
//!
//! Build and run with trunk:
//! ```bash
//! trunk serve --open
//! ```

use std::sync::Arc;

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use lumina_video::media::{WebVideoPlayer, WebVideoRenderResources, WebVideoTexture};
use parking_lot::Mutex;
use wasm_bindgen::JsCast;

/// Shared state for zero-copy video rendering, stored in CallbackResources.
pub struct WebVideoState {
    /// GPU resources (pipeline, sampler, bind group layout)
    pub resources: WebVideoRenderResources,
    /// Current video texture (updated each frame)
    pub texture: Option<WebVideoTexture>,
    /// Texture dimensions
    pub dims: (u32, u32),
}

/// Paint callback that renders video from shared state.
pub struct WebVideoCallback;

impl egui_wgpu::CallbackTrait for WebVideoCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        // Get shared video state from resources
        let Some(state) = resources.get::<Arc<Mutex<WebVideoState>>>() else {
            return;
        };
        let state = state.lock();

        // Get texture to render
        let Some(texture) = &state.texture else {
            return;
        };

        // Set viewport
        let viewport = info.viewport_in_pixels();
        render_pass.set_viewport(
            viewport.left_px as f32,
            viewport.top_px as f32,
            viewport.width_px as f32,
            viewport.height_px as f32,
            0.0,
            1.0,
        );

        // Set scissor rect
        let clip = info.clip_rect_in_pixels();
        render_pass.set_scissor_rect(
            clip.left_px.max(0) as u32,
            clip.top_px.max(0) as u32,
            clip.width_px.max(0) as u32,
            clip.height_px.max(0) as u32,
        );

        // Render
        render_pass.set_pipeline(state.resources.pipeline());
        render_pass.set_bind_group(0, texture.bind_group(), &[]);
        render_pass.draw(0..6, 0..1);
    }
}

/// Sample HLS streams for testing
const SAMPLE_VIDEOS: &[(&str, &str)] = &[
    (
        "Big Buck Bunny (HLS)",
        "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
    ),
    (
        "Apple Test Stream",
        "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_ts/master.m3u8",
    ),
    (
        "Tears of Steel (MP4)",
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
    ),
];

/// Time in seconds before controls auto-hide
const CONTROLS_HIDE_DELAY: f64 = 3.0;

/// Minimum touch target size per Apple HIG (44pt)
const MIN_TOUCH_TARGET: f32 = 44.0;

/// Skip amount in seconds for skip buttons
const SKIP_SECONDS: f64 = 10.0;

/// Icon drawing helpers using egui shape primitives
mod icons {
    use eframe::egui::{self, Color32, Pos2, Stroke};

    /// Draw a play triangle (pointing right)
    pub fn draw_play(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
        let half = size / 2.0;
        // Offset slightly right for visual centering
        let cx = center.x + size * 0.1;
        let points = vec![
            Pos2::new(cx - half * 0.7, center.y - half),
            Pos2::new(cx + half * 0.8, center.y),
            Pos2::new(cx - half * 0.7, center.y + half),
        ];
        painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
    }

    /// Draw pause bars (two vertical rectangles)
    pub fn draw_pause(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
        let bar_width = size * 0.25;
        let bar_height = size * 0.8;
        let gap = size * 0.15;

        // Left bar
        painter.rect_filled(
            egui::Rect::from_center_size(
                Pos2::new(center.x - gap - bar_width / 2.0, center.y),
                egui::vec2(bar_width, bar_height),
            ),
            2.0,
            color,
        );
        // Right bar
        painter.rect_filled(
            egui::Rect::from_center_size(
                Pos2::new(center.x + gap + bar_width / 2.0, center.y),
                egui::vec2(bar_width, bar_height),
            ),
            2.0,
            color,
        );
    }

    /// Draw skip back icon (counter-clockwise arrow + "10")
    pub fn draw_skip_back(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
        let r = size * 0.32;
        let arc_center = Pos2::new(center.x, center.y - size * 0.08);

        // Counter-clockwise arc (from right side going up and around to left)
        let segments = 12;
        let start_angle = 0.3_f32; // Start from right
        let end_angle = std::f32::consts::PI + 0.8; // Go to left side
        let mut points = Vec::with_capacity(segments + 1);
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let angle = start_angle + t * (end_angle - start_angle);
            points.push(Pos2::new(
                arc_center.x + r * angle.cos(),
                arc_center.y - r * angle.sin(), // Negative for counter-clockwise visual
            ));
        }

        // Get arrow tip before consuming points (return early if empty)
        let Some(arrow_tip) = points.last().copied() else {
            return;
        };
        painter.add(egui::Shape::line(points, Stroke::new(2.5, color)));

        // Arrow head at end of arc (pointing left/down)
        let arrow_size = size * 0.18;
        let arrow_points = vec![
            Pos2::new(arrow_tip.x - arrow_size, arrow_tip.y + arrow_size * 0.3),
            Pos2::new(
                arrow_tip.x + arrow_size * 0.2,
                arrow_tip.y - arrow_size * 0.5,
            ),
            Pos2::new(
                arrow_tip.x + arrow_size * 0.4,
                arrow_tip.y + arrow_size * 0.6,
            ),
        ];
        painter.add(egui::Shape::convex_polygon(
            arrow_points,
            color,
            Stroke::NONE,
        ));

        // "10" text centered in circle
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            "10",
            egui::FontId::proportional(size * 0.36),
            color,
        );
    }

    /// Draw skip forward icon (clockwise arrow + "10")
    pub fn draw_skip_forward(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
        let r = size * 0.32;
        let arc_center = Pos2::new(center.x, center.y - size * 0.08);

        // Clockwise arc (from left side going up and around to right)
        let segments = 12;
        let start_angle = std::f32::consts::PI - 0.3; // Start from left
        let end_angle = -0.8_f32; // Go to right side
        let mut points = Vec::with_capacity(segments + 1);
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let angle = start_angle + t * (end_angle - start_angle);
            points.push(Pos2::new(
                arc_center.x + r * angle.cos(),
                arc_center.y - r * angle.sin(),
            ));
        }

        // Get arrow tip before consuming points (return early if empty)
        let Some(arrow_tip) = points.last().copied() else {
            return;
        };
        painter.add(egui::Shape::line(points, Stroke::new(2.5, color)));

        // Arrow head at end of arc (pointing right/down)
        let arrow_size = size * 0.18;
        let arrow_points = vec![
            Pos2::new(arrow_tip.x + arrow_size, arrow_tip.y + arrow_size * 0.3),
            Pos2::new(
                arrow_tip.x - arrow_size * 0.2,
                arrow_tip.y - arrow_size * 0.5,
            ),
            Pos2::new(
                arrow_tip.x - arrow_size * 0.4,
                arrow_tip.y + arrow_size * 0.6,
            ),
        ];
        painter.add(egui::Shape::convex_polygon(
            arrow_points,
            color,
            Stroke::NONE,
        ));

        // "10" text centered in circle
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            "10",
            egui::FontId::proportional(size * 0.36),
            color,
        );
    }

    /// Draw volume/speaker icon
    pub fn draw_volume(
        painter: &egui::Painter,
        center: Pos2,
        size: f32,
        color: Color32,
        level: f32,
    ) {
        let s = size * 0.4;

        // Speaker body (trapezoid-ish shape)
        let body = vec![
            Pos2::new(center.x - s * 0.6, center.y - s * 0.3),
            Pos2::new(center.x - s * 0.2, center.y - s * 0.3),
            Pos2::new(center.x - s * 0.2, center.y + s * 0.3),
            Pos2::new(center.x - s * 0.6, center.y + s * 0.3),
        ];
        painter.add(egui::Shape::convex_polygon(body, color, Stroke::NONE));

        // Speaker cone
        let cone = vec![
            Pos2::new(center.x - s * 0.2, center.y - s * 0.3),
            Pos2::new(center.x + s * 0.3, center.y - s * 0.7),
            Pos2::new(center.x + s * 0.3, center.y + s * 0.7),
            Pos2::new(center.x - s * 0.2, center.y + s * 0.3),
        ];
        painter.add(egui::Shape::convex_polygon(cone, color, Stroke::NONE));

        // Sound waves (arcs) based on volume level
        if level > 0.01 {
            let wave_x = center.x + s * 0.5;
            painter.add(egui::Shape::line(
                vec![
                    Pos2::new(wave_x, center.y - s * 0.3),
                    Pos2::new(wave_x + s * 0.2, center.y),
                    Pos2::new(wave_x, center.y + s * 0.3),
                ],
                Stroke::new(2.0, color),
            ));
        }
        if level > 0.5 {
            let wave_x = center.x + s * 0.8;
            painter.add(egui::Shape::line(
                vec![
                    Pos2::new(wave_x, center.y - s * 0.5),
                    Pos2::new(wave_x + s * 0.25, center.y),
                    Pos2::new(wave_x, center.y + s * 0.5),
                ],
                Stroke::new(2.0, color),
            ));
        }
    }

    /// Draw muted speaker icon (speaker with X)
    pub fn draw_muted(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
        let s = size * 0.4;

        // Speaker body
        let body = vec![
            Pos2::new(center.x - s * 0.8, center.y - s * 0.3),
            Pos2::new(center.x - s * 0.4, center.y - s * 0.3),
            Pos2::new(center.x - s * 0.4, center.y + s * 0.3),
            Pos2::new(center.x - s * 0.8, center.y + s * 0.3),
        ];
        painter.add(egui::Shape::convex_polygon(body, color, Stroke::NONE));

        // Speaker cone
        let cone = vec![
            Pos2::new(center.x - s * 0.4, center.y - s * 0.3),
            Pos2::new(center.x + s * 0.1, center.y - s * 0.7),
            Pos2::new(center.x + s * 0.1, center.y + s * 0.7),
            Pos2::new(center.x - s * 0.4, center.y + s * 0.3),
        ];
        painter.add(egui::Shape::convex_polygon(cone, color, Stroke::NONE));

        // X mark
        let x_center = Pos2::new(center.x + s * 0.6, center.y);
        let x_size = s * 0.4;
        painter.add(egui::Shape::line(
            vec![
                Pos2::new(x_center.x - x_size, x_center.y - x_size),
                Pos2::new(x_center.x + x_size, x_center.y + x_size),
            ],
            Stroke::new(2.5, color),
        ));
        painter.add(egui::Shape::line(
            vec![
                Pos2::new(x_center.x + x_size, x_center.y - x_size),
                Pos2::new(x_center.x - x_size, x_center.y + x_size),
            ],
            Stroke::new(2.5, color),
        ));
    }

    /// Draw replay icon (circular arrow)
    pub fn draw_replay(painter: &egui::Painter, center: Pos2, size: f32, color: Color32) {
        let r = size * 0.35;

        // Full circle arc
        let segments = 12;
        let mut points = Vec::with_capacity(segments + 1);
        for i in 0..=segments {
            let angle = -std::f32::consts::FRAC_PI_2
                + (i as f32 / segments as f32) * std::f32::consts::TAU * 0.85;
            points.push(Pos2::new(
                center.x + r * angle.cos(),
                center.y + r * angle.sin(),
            ));
        }
        painter.add(egui::Shape::line(points, Stroke::new(2.5, color)));

        // Arrow head at the top
        let arrow_tip = Pos2::new(center.x, center.y - r);
        let arrow_points = vec![
            Pos2::new(arrow_tip.x, arrow_tip.y - size * 0.12),
            Pos2::new(arrow_tip.x - size * 0.1, arrow_tip.y + size * 0.06),
            Pos2::new(arrow_tip.x + size * 0.1, arrow_tip.y + size * 0.06),
        ];
        painter.add(egui::Shape::convex_polygon(
            arrow_points,
            color,
            Stroke::NONE,
        ));
    }
}

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    wasm_bindgen_futures::spawn_local(async {
        let web_options = eframe::WebOptions::default();

        // Get window, document, and canvas with graceful error handling
        let Some(window) = web_sys::window() else {
            web_sys::console::error_1(&"Failed to get window object".into());
            return;
        };
        let Some(document) = window.document() else {
            web_sys::console::error_1(&"Failed to get document".into());
            return;
        };
        let Some(element) = document.get_element_by_id("lumina-video-canvas") else {
            web_sys::console::error_1(&"No canvas element with id 'lumina-video-canvas'".into());
            return;
        };
        let Ok(canvas) = element.dyn_into::<web_sys::HtmlCanvasElement>() else {
            web_sys::console::error_1(&"Element 'lumina-video-canvas' is not a canvas".into());
            return;
        };

        if let Err(e) = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(DemoApp::new(cc)))),
            )
            .await
        {
            web_sys::console::error_1(&format!("Failed to start eframe: {:?}", e).into());
        }
    });
}

/// UI state for the lumina-video web demo application.
struct DemoApp {
    /// Active video player instance
    player: Option<WebVideoPlayer>,
    /// Shared state for zero-copy rendering (stored in CallbackResources)
    video_state: Option<Arc<Mutex<WebVideoState>>>,
    /// User-entered URL for custom video loading
    url_input: String,
    /// Index of currently selected sample video
    selected_sample: usize,
    /// Last error message to display to user
    error: Option<String>,
    /// Whether the video picker panel is visible
    show_picker: bool,
    /// Timestamp of last user interaction (for auto-hide controls)
    last_interaction: f64,
    /// Whether playback controls are currently visible
    controls_visible: bool,
    /// Whether mouse is hovering over video area
    mouse_over_video: bool,
    /// Whether user is actively seeking (dragging seek bar)
    is_seeking: bool,
    /// Current seek position while dragging (0.0-1.0)
    seek_position: f32,
}

impl DemoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Initialize zero-copy render resources
        let video_state = cc.wgpu_render_state.as_ref().map(|render_state| {
            let resources = WebVideoRenderResources::new(render_state);
            let state = Arc::new(Mutex::new(WebVideoState {
                resources,
                texture: None,
                dims: (0, 0),
            }));
            // Register in CallbackResources for access during paint
            render_state
                .renderer
                .write()
                .callback_resources
                .insert(Arc::clone(&state));
            state
        });

        Self {
            player: None,
            video_state,
            url_input: SAMPLE_VIDEOS
                .first()
                .map(|(_, url)| (*url).to_string())
                .unwrap_or_default(),
            selected_sample: 0,
            error: None,
            show_picker: true,
            last_interaction: 0.0,
            controls_visible: true,
            mouse_over_video: false,
            is_seeking: false,
            seek_position: 0.0,
        }
    }

    fn load_video(&mut self, url: &str) {
        self.error = None;
        self.show_picker = false;
        self.controls_visible = true;
        self.last_interaction = current_time();

        // Clear existing texture
        if let Some(state) = &self.video_state {
            let mut state = state.lock();
            state.texture = None;
            state.dims = (0, 0);
        }

        match WebVideoPlayer::new(url) {
            Ok(mut player) => {
                // Start frame callbacks for frame-accurate sync
                if let Err(e) = player.start_frame_callbacks() {
                    web_sys::console::warn_1(&format!("Frame callbacks failed: {:?}", e).into());
                }
                self.player = Some(player);
            }
            Err(e) => {
                self.error = Some(format!("Failed to load: {:?}", e));
                self.show_picker = true;
            }
        }
    }

    /// Updates the zero-copy video texture using GPU-to-GPU transfer.
    ///
    /// Uses WebGPU's copyExternalImageToTexture for efficient rendering
    /// without CPU readback. This is ~50x faster than Canvas 2D getImageData.
    fn update_video_texture(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        video_width: u32,
        video_height: u32,
    ) {
        let Some(player) = &self.player else { return };
        let Some(video_state) = &self.video_state else {
            return;
        };

        let mut state = video_state.lock();
        let dims_changed = state.dims != (video_width, video_height);
        let texture_created = state.texture.is_none();

        // Recreate texture if dimensions changed
        if dims_changed || texture_created {
            state.texture = Some(WebVideoTexture::new(
                &render_state.device,
                state.resources.bind_group_layout(),
                state.resources.sampler(),
                video_width,
                video_height,
            ));
            state.dims = (video_width, video_height);
        }

        // Upload frame when:
        // 1. Texture was just created (need initial frame)
        // 2. A new frame is ready from video callbacks
        let should_upload = texture_created || dims_changed || player.is_frame_ready();
        if should_upload {
            if let Some(texture) = &state.texture {
                match texture.upload_frame(&render_state.queue, player.video_element()) {
                    Ok(true) => {
                        // Upload succeeded, clear frame_ready
                        player.clear_frame_ready();
                    }
                    Ok(false) => {
                        // Dimension mismatch - texture needs recreation, don't clear frame_ready
                        // so we retry on next frame after texture is recreated
                        state.texture = None;
                        state.dims = (0, 0);
                    }
                    Err(_) => {
                        // Upload failed, clear frame_ready to avoid spam
                        player.clear_frame_ready();
                    }
                }
            }
        }
    }

    fn update_controls_visibility(&mut self) {
        let now = current_time();

        if self.mouse_over_video || self.is_seeking {
            self.last_interaction = now;
            self.controls_visible = true;
        } else if let Some(player) = &self.player {
            if player.is_playing() && now - self.last_interaction > CONTROLS_HIDE_DELAY {
                self.controls_visible = false;
            }
        }
    }

    fn handle_keyboard(&mut self, ctx: &egui::Context) {
        if self.player.is_none() {
            return;
        }

        // Don't handle shortcuts when a text field is focused
        if ctx.wants_keyboard_input() {
            return;
        }

        ctx.input(|i| {
            // Space: play/pause
            if i.key_pressed(egui::Key::Space) {
                if let Some(player) = &mut self.player {
                    player.toggle_playback();
                    self.last_interaction = current_time();
                    self.controls_visible = true;
                }
            }

            // Left arrow: skip back 10s
            if i.key_pressed(egui::Key::ArrowLeft) {
                if let Some(player) = &mut self.player {
                    let pos = player.position().as_secs_f64();
                    let new_pos = (pos - SKIP_SECONDS).max(0.0);
                    player.seek(std::time::Duration::from_secs_f64(new_pos));
                    self.last_interaction = current_time();
                    self.controls_visible = true;
                }
            }

            // Right arrow: skip forward 10s
            if i.key_pressed(egui::Key::ArrowRight) {
                if let Some(player) = &mut self.player {
                    let pos = player.position().as_secs_f64();
                    let duration = player
                        .duration()
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(f64::MAX);
                    let new_pos = (pos + SKIP_SECONDS).min(duration);
                    player.seek(std::time::Duration::from_secs_f64(new_pos));
                    self.last_interaction = current_time();
                    self.controls_visible = true;
                }
            }

            // M: toggle mute
            if i.key_pressed(egui::Key::M) {
                if let Some(player) = &mut self.player {
                    player.toggle_mute();
                    self.last_interaction = current_time();
                    self.controls_visible = true;
                }
            }

            // Escape: back to picker
            if i.key_pressed(egui::Key::Escape) {
                self.player = None;
                // Clear texture but keep state (Arc is shared with CallbackResources)
                if let Some(state) = &self.video_state {
                    let mut state = state.lock();
                    state.texture = None;
                    state.dims = (0, 0);
                }
                self.show_picker = true;
            }
        });
    }

    fn show_video_picker(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::from_rgb(20, 20, 20)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 4.0);

                    ui.heading(
                        egui::RichText::new("lumina-video")
                            .size(32.0)
                            .color(egui::Color32::WHITE),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Web Video Player Demo")
                            .size(16.0)
                            .color(egui::Color32::GRAY),
                    );
                    ui.add_space(32.0);

                    ui.label(
                        egui::RichText::new("Select a video:").color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(12.0);

                    for (i, (name, url)) in SAMPLE_VIDEOS.iter().enumerate() {
                        let selected = self.selected_sample == i;
                        let btn = egui::Button::new(egui::RichText::new(*name).size(16.0).color(
                            if selected {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::LIGHT_GRAY
                            },
                        ))
                        .fill(if selected {
                            egui::Color32::from_rgb(0, 122, 255)
                        } else {
                            egui::Color32::from_rgb(50, 50, 50)
                        })
                        .min_size(egui::vec2(280.0, MIN_TOUCH_TARGET))
                        .corner_radius(10.0);

                        if ui.add(btn).clicked() {
                            self.selected_sample = i;
                            self.url_input = url.to_string();
                        }
                        ui.add_space(8.0);
                    }

                    ui.add_space(16.0);
                    ui.label(egui::RichText::new("Or enter a URL:").color(egui::Color32::GRAY));
                    ui.add_space(8.0);

                    let text_edit = egui::TextEdit::singleline(&mut self.url_input)
                        .desired_width(280.0)
                        .text_color(egui::Color32::WHITE)
                        .hint_text("https://...");
                    ui.add(text_edit);

                    ui.add_space(24.0);

                    let play_btn = egui::Button::new(
                        egui::RichText::new("Play Video")
                            .size(18.0)
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(0, 122, 255))
                    .min_size(egui::vec2(200.0, 50.0))
                    .corner_radius(12.0);

                    if ui.add(play_btn).clicked() {
                        let url = self.url_input.clone();
                        self.load_video(&url);
                    }

                    if let Some(error) = &self.error {
                        ui.add_space(16.0);
                        ui.label(
                            egui::RichText::new(error).color(egui::Color32::from_rgb(255, 69, 58)),
                        );
                    }

                    ui.add_space(24.0);
                    ui.label(
                        egui::RichText::new("Keyboard: Space=play/pause, Arrows=seek, M=mute")
                            .size(12.0)
                            .color(egui::Color32::DARK_GRAY),
                    );
                });
            });
    }

    fn show_video_player(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        self.update_controls_visibility();
        self.handle_keyboard(ctx);

        // Request repaint for video updates (playing or loading initial frame)
        if self.player.is_some() {
            let needs_texture = self
                .video_state
                .as_ref()
                .map(|s| s.lock().texture.is_none())
                .unwrap_or(true);

            let is_playing = self
                .player
                .as_ref()
                .map(|p| p.is_playing())
                .unwrap_or(false);

            // Repaint cadence based on state:
            // - Playing: 60fps for smooth video
            // - Waiting for texture (loading): 4fps to check for first frame without spinning CPU
            // - Paused with texture: no automatic repaint needed
            if is_playing {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            } else if needs_texture {
                ctx.request_repaint_after(std::time::Duration::from_millis(250));
            }
        }

        // Update zero-copy texture before rendering
        if let Some(render_state) = frame.wgpu_render_state() {
            if let Some(player) = &self.player {
                let (video_width, video_height) = player.dimensions();
                if video_width > 0 && video_height > 0 {
                    self.update_video_texture(render_state, video_width, video_height);
                }
            }
        }

        // Check if we have a valid texture to render
        let has_texture = self
            .video_state
            .as_ref()
            .map(|s| s.lock().texture.is_some())
            .unwrap_or(false);

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::BLACK))
            .show(ctx, |ui| {
                let available_size = ui.available_size();

                if let Some(player) = &self.player {
                    let (video_width, video_height) = player.dimensions();

                    if video_width > 0 && video_height > 0 {
                        let aspect = video_width as f32 / video_height as f32;
                        let fit_width = available_size.x.min(available_size.y * aspect);
                        let fit_height = fit_width / aspect;
                        let video_size = egui::vec2(fit_width, fit_height);
                        let video_offset = (available_size - video_size) / 2.0;
                        let video_rect =
                            egui::Rect::from_min_size(ui.min_rect().min + video_offset, video_size);

                        // Render video using zero-copy PaintCallback
                        if has_texture {
                            let paint_callback = egui_wgpu::Callback::new_paint_callback(
                                video_rect,
                                WebVideoCallback,
                            );
                            ui.painter().add(paint_callback);
                        }

                        let response =
                            ui.interact(video_rect, ui.id().with("video"), egui::Sense::click());
                        self.mouse_over_video = response.hovered();

                        if response.hovered() {
                            self.last_interaction = current_time();
                            self.controls_visible = true;
                        }

                        if response.clicked()
                            && !self.is_control_area(response.interact_pointer_pos(), video_rect)
                        {
                            if let Some(player) = &mut self.player {
                                player.toggle_playback();
                                self.last_interaction = current_time();
                            }
                        }

                        self.draw_controls_overlay(ui, video_rect);
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.spinner();
                        });
                    }
                }
            });
    }

    fn is_control_area(&self, pointer_pos: Option<egui::Pos2>, video_rect: egui::Rect) -> bool {
        if let Some(pos) = pointer_pos {
            pos.y > video_rect.max.y - 56.0
        } else {
            false
        }
    }

    fn draw_controls_overlay(&mut self, ui: &mut egui::Ui, video_rect: egui::Rect) {
        let player = match &mut self.player {
            Some(p) => p,
            None => return,
        };

        let is_playing = player.is_playing();
        let is_ended = player.is_ended();
        let alpha = if self.controls_visible || !is_playing {
            255u8
        } else {
            0u8
        };

        if alpha == 0 {
            return;
        }

        // Center overlay with play/pause and skip buttons (shown when paused or controls visible)
        if !is_playing || self.controls_visible {
            let center = video_rect.center();
            let btn_size = 70.0;
            let skip_btn_size = 50.0;
            let spacing = 24.0;

            // Skip back button (left of center)
            let skip_back_center = egui::pos2(
                center.x - btn_size / 2.0 - spacing - skip_btn_size / 2.0,
                center.y,
            );
            ui.painter().circle_filled(
                skip_back_center,
                skip_btn_size / 2.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150),
            );
            icons::draw_skip_back(
                ui.painter(),
                skip_back_center,
                skip_btn_size,
                egui::Color32::WHITE,
            );

            let skip_back_rect = egui::Rect::from_center_size(
                skip_back_center,
                egui::vec2(skip_btn_size, skip_btn_size),
            );
            let skip_back_response = ui
                .interact(
                    skip_back_rect,
                    ui.id().with("skip_back"),
                    egui::Sense::click(),
                )
                .on_hover_text("Skip back 10 seconds");
            if skip_back_response.clicked() {
                let pos = player.position().as_secs_f64();
                player.seek(std::time::Duration::from_secs_f64(
                    (pos - SKIP_SECONDS).max(0.0),
                ));
                self.last_interaction = current_time();
            }

            // Center play/pause button
            ui.painter().circle_filled(
                center,
                btn_size / 2.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 180),
            );

            if is_ended {
                icons::draw_replay(ui.painter(), center, btn_size * 0.6, egui::Color32::WHITE);
            } else if is_playing {
                icons::draw_pause(ui.painter(), center, btn_size * 0.4, egui::Color32::WHITE);
            } else {
                icons::draw_play(ui.painter(), center, btn_size * 0.45, egui::Color32::WHITE);
            }

            let center_btn_rect =
                egui::Rect::from_center_size(center, egui::vec2(btn_size, btn_size));
            let center_response = ui
                .interact(
                    center_btn_rect,
                    ui.id().with("center_play"),
                    egui::Sense::click(),
                )
                .on_hover_text(if is_playing { "Pause" } else { "Play" });
            if center_response.clicked() {
                player.toggle_playback();
                self.last_interaction = current_time();
            }

            // Skip forward button (right of center)
            // Skip forward button (right of center)
            let skip_fwd_center = egui::pos2(
                center.x + btn_size / 2.0 + spacing + skip_btn_size / 2.0,
                center.y,
            );
            ui.painter().circle_filled(
                skip_fwd_center,
                skip_btn_size / 2.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 150),
            );
            icons::draw_skip_forward(
                ui.painter(),
                skip_fwd_center,
                skip_btn_size,
                egui::Color32::WHITE,
            );

            let skip_fwd_rect = egui::Rect::from_center_size(
                skip_fwd_center,
                egui::vec2(skip_btn_size, skip_btn_size),
            );
            let skip_fwd_response = ui
                .interact(
                    skip_fwd_rect,
                    ui.id().with("skip_fwd"),
                    egui::Sense::click(),
                )
                .on_hover_text("Skip forward 10 seconds");
            if skip_fwd_response.clicked() {
                let pos = player.position().as_secs_f64();
                let duration = player
                    .duration()
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(f64::MAX);
                player.seek(std::time::Duration::from_secs_f64(
                    (pos + SKIP_SECONDS).min(duration),
                ));
                self.last_interaction = current_time();
            }
        }

        // Bottom control bar (56pt per HIG)
        let bar_height = 56.0;
        let bar_rect = egui::Rect::from_min_max(
            egui::pos2(video_rect.min.x, video_rect.max.y - bar_height),
            video_rect.max,
        );

        ui.painter().rect_filled(
            bar_rect,
            0.0,
            egui::Color32::from_rgba_unmultiplied(0, 0, 0, (0.7 * alpha as f32) as u8),
        );

        let padding = 16.0;
        let center_y = bar_rect.center().y;
        let left_x = bar_rect.min.x + padding;
        let right_x = bar_rect.max.x - padding;

        // Play/pause button (44pt touch target)
        let play_btn_rect = egui::Rect::from_center_size(
            egui::pos2(left_x + MIN_TOUCH_TARGET / 2.0, center_y),
            egui::vec2(MIN_TOUCH_TARGET, MIN_TOUCH_TARGET),
        );

        if is_playing {
            icons::draw_pause(
                ui.painter(),
                play_btn_rect.center(),
                18.0,
                egui::Color32::WHITE,
            );
        } else {
            icons::draw_play(
                ui.painter(),
                play_btn_rect.center(),
                20.0,
                egui::Color32::WHITE,
            );
        }

        let play_response = ui
            .interact(
                play_btn_rect,
                ui.id().with("play_btn"),
                egui::Sense::click(),
            )
            .on_hover_text(if is_playing {
                "Pause (Space)"
            } else {
                "Play (Space)"
            });
        if play_response.clicked() {
            player.toggle_playback();
            self.last_interaction = current_time();
        }

        // Volume controls (right side)
        let vol_slider_width = 70.0;
        let vol_btn_width = MIN_TOUCH_TARGET;

        let volume = player.volume();
        let is_muted = player.is_muted() || volume < 0.01;

        // Volume/mute button
        let vol_btn_rect = egui::Rect::from_center_size(
            egui::pos2(right_x - vol_slider_width - vol_btn_width / 2.0, center_y),
            egui::vec2(vol_btn_width, MIN_TOUCH_TARGET),
        );

        if is_muted {
            icons::draw_muted(
                ui.painter(),
                vol_btn_rect.center(),
                32.0,
                egui::Color32::WHITE,
            );
        } else {
            icons::draw_volume(
                ui.painter(),
                vol_btn_rect.center(),
                32.0,
                egui::Color32::WHITE,
                volume,
            );
        }

        let vol_btn_response = ui
            .interact(vol_btn_rect, ui.id().with("vol_btn"), egui::Sense::click())
            .on_hover_text("Mute (M)");
        if vol_btn_response.clicked() {
            player.toggle_mute();
            self.last_interaction = current_time();
        }

        // Volume slider
        let vol_slider_rect = egui::Rect::from_min_max(
            egui::pos2(right_x - vol_slider_width, center_y - 3.0),
            egui::pos2(right_x, center_y + 3.0),
        );

        ui.painter()
            .rect_filled(vol_slider_rect, 3.0, egui::Color32::from_gray(80));
        let vol_fill = egui::Rect::from_min_max(
            vol_slider_rect.min,
            egui::pos2(
                vol_slider_rect.min.x + vol_slider_rect.width() * volume,
                vol_slider_rect.max.y,
            ),
        );
        ui.painter()
            .rect_filled(vol_fill, 3.0, egui::Color32::WHITE);

        let vol_response = ui.interact(
            vol_slider_rect.expand(8.0),
            ui.id().with("vol_slider"),
            egui::Sense::click_and_drag(),
        );
        if vol_response.dragged() || vol_response.clicked() {
            self.last_interaction = current_time();
            if let Some(pos) = vol_response.interact_pointer_pos() {
                let v = ((pos.x - vol_slider_rect.min.x) / vol_slider_rect.width()).clamp(0.0, 1.0);
                player.set_volume(v);
            }
        }

        // Time display
        let time_x = vol_btn_rect.min.x - 12.0;

        // Seek bar
        let seek_left = left_x + MIN_TOUCH_TARGET + 12.0;
        let seek_right = time_x - 90.0;
        let seek_bar_rect = egui::Rect::from_min_max(
            egui::pos2(seek_left, center_y - 4.0),
            egui::pos2(seek_right, center_y + 4.0),
        );

        if let Some(duration) = player.duration() {
            let duration_secs = duration.as_secs_f32();
            let current_pos = if self.is_seeking {
                self.seek_position
            } else {
                player.position().as_secs_f32()
            };
            let progress = (current_pos / duration_secs).clamp(0.0, 1.0);

            ui.painter()
                .rect_filled(seek_bar_rect, 4.0, egui::Color32::from_gray(80));
            let progress_rect = egui::Rect::from_min_max(
                seek_bar_rect.min,
                egui::pos2(
                    seek_bar_rect.min.x + seek_bar_rect.width() * progress,
                    seek_bar_rect.max.y,
                ),
            );
            ui.painter()
                .rect_filled(progress_rect, 4.0, egui::Color32::WHITE);

            // Seek handle (larger for touch)
            let handle_x = seek_bar_rect.min.x + seek_bar_rect.width() * progress;
            ui.painter()
                .circle_filled(egui::pos2(handle_x, center_y), 8.0, egui::Color32::WHITE);

            let seek_response = ui.interact(
                seek_bar_rect.expand(12.0),
                ui.id().with("seek_bar"),
                egui::Sense::click_and_drag(),
            );

            if seek_response.dragged() || seek_response.clicked() {
                self.is_seeking = true;
                self.last_interaction = current_time();
                if let Some(pos) = seek_response.interact_pointer_pos() {
                    let t = ((pos.x - seek_bar_rect.min.x) / seek_bar_rect.width()).clamp(0.0, 1.0);
                    self.seek_position = t * duration_secs;
                }
            }

            if seek_response.drag_stopped() || (self.is_seeking && !seek_response.dragged()) {
                player.seek(std::time::Duration::from_secs_f32(self.seek_position));
                self.is_seeking = false;
            }

            // Time display
            let time_text = format!(
                "{} / {}",
                format_time(current_pos),
                format_time(duration_secs)
            );
            ui.painter().text(
                egui::pos2(time_x - 4.0, center_y),
                egui::Align2::RIGHT_CENTER,
                time_text,
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );
        }

        // Close button (top left, 44pt touch target)
        if self.controls_visible {
            let close_btn_rect = egui::Rect::from_min_size(
                egui::pos2(video_rect.min.x + 12.0, video_rect.min.y + 12.0),
                egui::vec2(MIN_TOUCH_TARGET, MIN_TOUCH_TARGET),
            );

            ui.painter().circle_filled(
                close_btn_rect.center(),
                MIN_TOUCH_TARGET / 2.0,
                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 120),
            );
            ui.painter().text(
                close_btn_rect.center(),
                egui::Align2::CENTER_CENTER,
                "X",
                egui::FontId::proportional(18.0),
                egui::Color32::WHITE,
            );

            let close_response = ui
                .interact(
                    close_btn_rect,
                    ui.id().with("close_btn"),
                    egui::Sense::click(),
                )
                .on_hover_text("Close (Escape)");
            if close_response.clicked() {
                self.player = None;
                // Clear texture but keep state (Arc is shared with CallbackResources)
                if let Some(state) = &self.video_state {
                    let mut state = state.lock();
                    state.texture = None;
                    state.dims = (0, 0);
                }
                self.show_picker = true;
            }
        }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.show_picker {
            self.show_video_picker(ctx);
        } else {
            self.show_video_player(ctx, frame);
        }
    }
}

fn format_time(secs: f32) -> String {
    let mins = (secs / 60.0) as u32;
    let secs = (secs % 60.0) as u32;
    format!("{}:{:02}", mins, secs)
}

fn current_time() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0)
}

mod console_error_panic_hook {
    use std::panic::PanicHookInfo;

    pub fn hook(info: &PanicHookInfo<'_>) {
        let msg = info.to_string();
        web_sys::console::error_1(&msg.into());
    }
}
