//! Video player controls UI.
//!
//! This module provides overlay controls for video playback including:
//! - Play/Pause button
//! - Seek bar with drag support
//! - Time display (current position / duration)
//! - Loading/buffering indicator

use std::time::Duration;

use egui::{Align2, Color32, CornerRadius, FontId, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use super::video::VideoState;

/// Configuration for video controls appearance.
#[derive(Clone)]
pub struct VideoControlsConfig {
    /// Height of the controls bar
    pub bar_height: f32,
    /// Background color of the controls bar
    pub bar_color: Color32,
    /// Icon color for buttons
    pub icon_color: Color32,
    /// Progress bar fill color
    pub progress_color: Color32,
    /// Progress bar background color
    pub progress_bg_color: Color32,
    /// Text color for time display
    pub text_color: Color32,
    /// Font size for time display
    pub font_size: f32,
    /// Whether to auto-hide controls after a period of inactivity.
    ///
    /// **NOT YET IMPLEMENTED** - This field is reserved for future use and
    /// currently has no effect.
    ///
    /// Planned behavior: When enabled, the controls bar will fade out after
    /// `hide_delay` duration of no mouse movement over the video area,
    /// providing an unobstructed view. Moving the mouse back over the video
    /// will show the controls again.
    #[allow(dead_code)]
    pub auto_hide: bool,
    /// Duration of inactivity before auto-hiding controls.
    ///
    /// **NOT YET IMPLEMENTED** - This field is reserved for future use and
    /// currently has no effect.
    ///
    /// Planned behavior: Only applies when `auto_hide` is enabled. After this
    /// duration without mouse movement over the video area, the controls bar
    /// will fade out.
    #[allow(dead_code)]
    pub hide_delay: Duration,
}

impl Default for VideoControlsConfig {
    fn default() -> Self {
        Self {
            bar_height: 40.0,
            bar_color: Color32::from_rgba_unmultiplied(0, 0, 0, 180),
            icon_color: Color32::WHITE,
            progress_color: Color32::from_rgb(255, 100, 100),
            progress_bg_color: Color32::from_rgba_unmultiplied(255, 255, 255, 100),
            text_color: Color32::WHITE,
            font_size: 12.0,
            auto_hide: true,
            hide_delay: Duration::from_secs(3),
        }
    }
}

/// Response from video controls interaction.
#[derive(Default)]
pub struct VideoControlsResponse {
    /// Whether the play/pause button was clicked
    pub toggle_playback: bool,
    /// New seek position if user dragged the seek bar
    pub seek_to: Option<Duration>,
    /// Whether fullscreen was toggled
    pub toggle_fullscreen: bool,
    /// Whether the user is currently dragging the seek bar
    pub is_seeking: bool,
    /// Whether the mute button was clicked
    pub toggle_mute: bool,
    /// Whether the subtitles/CC button was clicked
    pub toggle_subtitles: bool,
}

/// Video player controls widget.
pub struct VideoControls<'a> {
    /// Current playback state
    state: &'a VideoState,
    /// Current position
    position: Duration,
    /// Total duration (None if unknown)
    duration: Option<Duration>,
    /// Whether the video is loading/buffering
    is_loading: bool,
    /// Buffering percentage (0-100)
    buffering_percent: i32,
    /// Configuration
    config: VideoControlsConfig,
    /// Whether controls are visible
    visible: bool,
    /// Whether audio is muted
    muted: bool,
    /// Whether subtitles are currently visible
    subtitles_visible: bool,
    /// Whether subtitles are available (a track is loaded)
    subtitles_available: bool,
}

impl<'a> VideoControls<'a> {
    /// Creates new video controls.
    pub fn new(state: &'a VideoState, position: Duration, duration: Option<Duration>) -> Self {
        Self {
            state,
            position,
            duration,
            is_loading: matches!(state, VideoState::Loading),
            buffering_percent: 0,
            config: VideoControlsConfig::default(),
            visible: true,
            muted: false,
            subtitles_visible: false,
            subtitles_available: false,
        }
    }

    /// Sets the buffering percentage for the progress ring display.
    pub fn with_buffering_percent(mut self, percent: i32) -> Self {
        self.buffering_percent = percent.clamp(0, 100);
        self
    }

    /// Sets the configuration.
    pub fn with_config(mut self, config: VideoControlsConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the muted state for the mute button display.
    pub fn with_muted(mut self, muted: bool) -> Self {
        self.muted = muted;
        self
    }

    /// Sets whether controls are visible.
    pub fn with_visibility(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Sets the subtitle state for the CC button display.
    pub fn with_subtitles(mut self, visible: bool, available: bool) -> Self {
        self.subtitles_visible = visible;
        self.subtitles_available = available;
        self
    }

    /// Shows the video controls.
    ///
    /// The controls are rendered as an overlay at the bottom of the given rect.
    pub fn show(&self, ui: &mut Ui, video_rect: Rect) -> VideoControlsResponse {
        let mut response = VideoControlsResponse::default();

        if !self.visible {
            return response;
        }

        // Calculate controls rect at bottom of video
        let controls_rect = Rect::from_min_size(
            Pos2::new(video_rect.min.x, video_rect.max.y - self.config.bar_height),
            Vec2::new(video_rect.width(), self.config.bar_height),
        );

        // Draw background
        ui.painter()
            .rect_filled(controls_rect, CornerRadius::ZERO, self.config.bar_color);

        // Layout: [Play/Pause] [Seek Bar] [Time Display] [CC] [Mute] [Fullscreen]
        let padding = 8.0;
        let button_size = self.config.bar_height - padding * 2.0;

        // Play/Pause button area
        let play_button_rect = Rect::from_min_size(
            Pos2::new(controls_rect.min.x + padding, controls_rect.min.y + padding),
            Vec2::splat(button_size),
        );

        // Fullscreen button area (rightmost)
        let fullscreen_button_rect = Rect::from_min_size(
            Pos2::new(
                controls_rect.max.x - button_size - padding,
                controls_rect.min.y + padding,
            ),
            Vec2::splat(button_size),
        );

        // Mute button area (left of fullscreen)
        let mute_button_rect = Rect::from_min_size(
            Pos2::new(
                fullscreen_button_rect.min.x - button_size - padding / 2.0,
                controls_rect.min.y + padding,
            ),
            Vec2::splat(button_size),
        );

        // CC/Subtitles button area (left of mute)
        let cc_button_rect = Rect::from_min_size(
            Pos2::new(
                mute_button_rect.min.x - button_size - padding / 2.0,
                controls_rect.min.y + padding,
            ),
            Vec2::splat(button_size),
        );

        // Time display area (left of CC button)
        let time_width = 80.0;
        let time_rect = Rect::from_min_size(
            Pos2::new(
                cc_button_rect.min.x - time_width - padding,
                controls_rect.min.y + padding,
            ),
            Vec2::new(time_width, button_size),
        );

        // Seek bar area (between play button and time display)
        let seek_bar_x = play_button_rect.max.x + padding;
        let seek_bar_width = time_rect.min.x - seek_bar_x - padding;
        let seek_bar_height = 6.0;
        let seek_bar_rect = Rect::from_min_size(
            Pos2::new(seek_bar_x, controls_rect.center().y - seek_bar_height / 2.0),
            Vec2::new(seek_bar_width, seek_bar_height),
        );

        // Draw play/pause button
        response.toggle_playback = self.draw_play_button(ui, play_button_rect);

        // Draw seek bar
        let (seek_to, is_seeking) = self.draw_seek_bar(ui, seek_bar_rect);
        response.seek_to = seek_to;
        response.is_seeking = is_seeking;

        // Draw time display
        self.draw_time_display(ui, time_rect);

        // Draw CC/subtitles button (only if subtitles are available)
        if self.subtitles_available {
            response.toggle_subtitles = self.draw_cc_button(ui, cc_button_rect);
        }

        // Draw mute button
        response.toggle_mute = self.draw_mute_button(ui, mute_button_rect);

        // Draw fullscreen button
        response.toggle_fullscreen = self.draw_fullscreen_button(ui, fullscreen_button_rect);

        // Draw loading indicator only for Loading state (buffering handled by VideoPlayer)
        if self.is_loading {
            self.draw_loading_indicator(ui, video_rect);
        }

        response
    }

    /// Draws the play/pause button.
    fn draw_play_button(&self, ui: &mut Ui, rect: Rect) -> bool {
        let response = ui.allocate_rect(rect, Sense::click());

        let is_playing = matches!(self.state, VideoState::Playing { .. });

        // Draw button background on hover
        if response.hovered() {
            ui.painter().rect_filled(
                rect,
                CornerRadius::same(4),
                Color32::from_rgba_unmultiplied(255, 255, 255, 30),
            );
        }

        // Draw play or pause icon
        let center = rect.center();
        let icon_size = rect.width() * 0.5;

        if is_playing {
            // Draw pause icon (two vertical bars)
            let bar_width = icon_size * 0.25;
            let bar_height = icon_size;
            let gap = icon_size * 0.25;

            let left_bar = Rect::from_center_size(
                Pos2::new(center.x - gap / 2.0 - bar_width / 2.0, center.y),
                Vec2::new(bar_width, bar_height),
            );
            let right_bar = Rect::from_center_size(
                Pos2::new(center.x + gap / 2.0 + bar_width / 2.0, center.y),
                Vec2::new(bar_width, bar_height),
            );

            ui.painter()
                .rect_filled(left_bar, CornerRadius::same(2), self.config.icon_color);
            ui.painter()
                .rect_filled(right_bar, CornerRadius::same(2), self.config.icon_color);
        } else {
            // Draw play icon (triangle pointing right)
            let points = vec![
                Pos2::new(center.x - icon_size * 0.4, center.y - icon_size * 0.5),
                Pos2::new(center.x - icon_size * 0.4, center.y + icon_size * 0.5),
                Pos2::new(center.x + icon_size * 0.5, center.y),
            ];
            ui.painter().add(egui::Shape::convex_polygon(
                points,
                self.config.icon_color,
                Stroke::NONE,
            ));
        }

        response.clicked()
    }

    /// Draws the seek bar and handles seeking.
    fn draw_seek_bar(&self, ui: &mut Ui, rect: Rect) -> (Option<Duration>, bool) {
        // Expand the clickable area for easier interaction
        let hit_rect = rect.expand2(Vec2::new(0.0, 10.0));
        let response = ui.allocate_rect(hit_rect, Sense::click_and_drag());

        // Draw background
        ui.painter()
            .rect_filled(rect, CornerRadius::same(3), self.config.progress_bg_color);

        // Calculate and draw progress
        let progress = if let Some(duration) = self.duration {
            if duration.as_secs_f32() > 0.0 {
                (self.position.as_secs_f32() / duration.as_secs_f32()).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let progress_width = rect.width() * progress;
        let progress_rect = Rect::from_min_size(rect.min, Vec2::new(progress_width, rect.height()));

        ui.painter().rect_filled(
            progress_rect,
            CornerRadius::same(3),
            self.config.progress_color,
        );

        // Draw scrubber handle
        if progress > 0.0 {
            let handle_center = Pos2::new(rect.min.x + progress_width, rect.center().y);
            let handle_radius = if response.hovered() || response.dragged() {
                7.0
            } else {
                5.0
            };
            ui.painter()
                .circle_filled(handle_center, handle_radius, self.config.progress_color);
        }

        // Handle seeking
        let mut seek_to = None;
        let is_seeking = response.dragged() || response.drag_stopped();

        if let Some(duration) = self.duration {
            if response.clicked() || response.dragged() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let relative_x = (pos.x - rect.min.x).clamp(0.0, rect.width());
                    let seek_progress = relative_x / rect.width();
                    let seek_pos = Duration::from_secs_f32(duration.as_secs_f32() * seek_progress);

                    // Debug: log suspicious seeks near zero
                    if seek_pos < Duration::from_secs(2) {
                        tracing::debug!(
                            "Progress bar seek near zero: pos={:?}, rect.min.x={}, relative_x={}, progress={:.3}, duration={:?}",
                            pos,
                            rect.min.x,
                            relative_x,
                            seek_progress,
                            duration
                        );
                    }

                    seek_to = Some(seek_pos);
                }
            }
        }

        (seek_to, is_seeking)
    }

    /// Draws the time display.
    fn draw_time_display(&self, ui: &mut Ui, rect: Rect) {
        let current = format_duration(self.position);
        let total = self
            .duration
            .map(format_duration)
            .unwrap_or_else(|| "--:--".to_string());

        let text = format!("{current} / {total}");

        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(self.config.font_size),
            self.config.text_color,
        );
    }

    /// Draws a loading indicator with progress ring.
    fn draw_loading_indicator(&self, ui: &mut Ui, video_rect: Rect) {
        // Draw semi-transparent overlay
        ui.painter().rect_filled(
            video_rect,
            CornerRadius::ZERO,
            Color32::from_rgba_unmultiplied(0, 0, 0, 100),
        );

        let center = video_rect.center();
        let radius = 24.0;
        let stroke_width = 4.0;

        // Draw background ring (gray track)
        let bg_stroke = Stroke::new(
            stroke_width,
            Color32::from_rgba_unmultiplied(255, 255, 255, 60),
        );
        let segments = 32;
        for i in 0..segments {
            let t = i as f32 / segments as f32;
            let a1 = -std::f32::consts::FRAC_PI_2 + t * std::f32::consts::TAU;
            let a2 =
                -std::f32::consts::FRAC_PI_2 + (t + 1.0 / segments as f32) * std::f32::consts::TAU;

            let p1 = Pos2::new(center.x + radius * a1.cos(), center.y + radius * a1.sin());
            let p2 = Pos2::new(center.x + radius * a2.cos(), center.y + radius * a2.sin());

            ui.painter().line_segment([p1, p2], bg_stroke);
        }

        // Draw progress arc (fills clockwise from top)
        let progress = self.buffering_percent as f32 / 100.0;
        let _progress_arc_length = progress * std::f32::consts::TAU;
        let progress_stroke = Stroke::new(stroke_width, Color32::from_rgb(76, 175, 80)); // Green

        if progress > 0.0 {
            let progress_segments = (segments as f32 * progress).max(1.0) as i32;
            for i in 0..progress_segments {
                let t = i as f32 / segments as f32;
                let a1 = -std::f32::consts::FRAC_PI_2 + t * std::f32::consts::TAU;
                let a2 = -std::f32::consts::FRAC_PI_2
                    + ((i + 1) as f32 / segments as f32).min(progress) * std::f32::consts::TAU;

                let p1 = Pos2::new(center.x + radius * a1.cos(), center.y + radius * a1.sin());
                let p2 = Pos2::new(center.x + radius * a2.cos(), center.y + radius * a2.sin());

                ui.painter().line_segment([p1, p2], progress_stroke);
            }
        }

        // Add subtle spinning animation on top when not at 100%
        if self.buffering_percent < 100 {
            let time = ui.ctx().input(|i| i.time);
            let spin_angle = (time * 3.0) % std::f64::consts::TAU;
            let spin_radius = radius + 6.0;

            // Small spinning dot
            let dot_pos = Pos2::new(
                center.x + spin_radius * (spin_angle as f32).cos(),
                center.y + spin_radius * (spin_angle as f32).sin(),
            );
            ui.painter().circle_filled(dot_pos, 3.0, Color32::WHITE);
        }

        // Draw percentage text in center
        let text = format!("{}%", self.buffering_percent);
        ui.painter().text(
            center,
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(14.0),
            Color32::WHITE,
        );

        // Draw "Buffering" label below
        ui.painter().text(
            Pos2::new(center.x, center.y + radius + 16.0),
            Align2::CENTER_CENTER,
            "Buffering",
            FontId::proportional(12.0),
            Color32::from_rgba_unmultiplied(255, 255, 255, 180),
        );

        // Request repaint for animation
        ui.ctx().request_repaint();
    }

    /// Draws the mute/unmute button.
    fn draw_mute_button(&self, ui: &mut Ui, rect: Rect) -> bool {
        let response = ui.allocate_rect(rect, Sense::click());

        // Draw button background on hover
        if response.hovered() {
            ui.painter().rect_filled(
                rect,
                CornerRadius::same(4),
                Color32::from_rgba_unmultiplied(255, 255, 255, 30),
            );
        }

        // Draw speaker icon
        let center = rect.center();
        let icon_size = rect.width() * 0.4;

        // Speaker body (trapezoid approximation using rectangle)
        let speaker_width = icon_size * 0.3;
        let speaker_height = icon_size * 0.5;
        let speaker_rect = Rect::from_center_size(
            Pos2::new(center.x - icon_size * 0.2, center.y),
            Vec2::new(speaker_width, speaker_height),
        );
        ui.painter()
            .rect_filled(speaker_rect, CornerRadius::same(1), self.config.icon_color);

        // Speaker cone (triangle)
        let cone_points = vec![
            Pos2::new(center.x - icon_size * 0.05, center.y - speaker_height / 2.0),
            Pos2::new(center.x - icon_size * 0.05, center.y + speaker_height / 2.0),
            Pos2::new(center.x + icon_size * 0.3, center.y + icon_size * 0.5),
            Pos2::new(center.x + icon_size * 0.3, center.y - icon_size * 0.5),
        ];
        ui.painter().add(egui::Shape::convex_polygon(
            cone_points,
            self.config.icon_color,
            Stroke::NONE,
        ));

        if self.muted {
            // Draw X through the speaker when muted
            let stroke = Stroke::new(2.0, Color32::from_rgb(255, 100, 100));
            let x_offset = icon_size * 0.1;
            ui.painter().line_segment(
                [
                    Pos2::new(
                        center.x - icon_size * 0.5 + x_offset,
                        center.y - icon_size * 0.5,
                    ),
                    Pos2::new(
                        center.x + icon_size * 0.5 + x_offset,
                        center.y + icon_size * 0.5,
                    ),
                ],
                stroke,
            );
            ui.painter().line_segment(
                [
                    Pos2::new(
                        center.x - icon_size * 0.5 + x_offset,
                        center.y + icon_size * 0.5,
                    ),
                    Pos2::new(
                        center.x + icon_size * 0.5 + x_offset,
                        center.y - icon_size * 0.5,
                    ),
                ],
                stroke,
            );
        } else {
            // Draw sound waves when not muted
            let wave_stroke = Stroke::new(1.5, self.config.icon_color);
            let wave_offset = icon_size * 0.45;
            for i in 0..2 {
                let wave_x = center.x + wave_offset + (i as f32) * icon_size * 0.2;
                let wave_height = icon_size * (0.3 + (i as f32) * 0.15);

                // Draw arc using line segments
                let segments = 6;
                for j in 0..segments {
                    let t1 = (j as f32) / (segments as f32) - 0.5;
                    let t2 = ((j + 1) as f32) / (segments as f32) - 0.5;
                    let angle1 = t1 * std::f32::consts::PI;
                    let angle2 = t2 * std::f32::consts::PI;

                    let p1 = Pos2::new(wave_x, center.y + angle1.sin() * wave_height);
                    let p2 = Pos2::new(
                        wave_x + icon_size * 0.05,
                        center.y + angle2.sin() * wave_height,
                    );

                    ui.painter().line_segment([p1, p2], wave_stroke);
                }
            }
        }

        response.clicked()
    }

    /// Draws the fullscreen button.
    fn draw_fullscreen_button(&self, ui: &mut Ui, rect: Rect) -> bool {
        let response = ui.allocate_rect(rect, Sense::click());

        // Draw button background on hover
        if response.hovered() {
            ui.painter().rect_filled(
                rect,
                CornerRadius::same(4),
                Color32::from_rgba_unmultiplied(255, 255, 255, 30),
            );
        }

        // Draw fullscreen icon (four corners pointing outward)
        let center = rect.center();
        let icon_size = rect.width() * 0.35;
        let corner_len = icon_size * 0.4;
        let stroke = Stroke::new(2.0, self.config.icon_color);

        // Top-left corner
        let tl = Pos2::new(center.x - icon_size, center.y - icon_size);
        ui.painter()
            .line_segment([tl, Pos2::new(tl.x + corner_len, tl.y)], stroke);
        ui.painter()
            .line_segment([tl, Pos2::new(tl.x, tl.y + corner_len)], stroke);

        // Top-right corner
        let tr = Pos2::new(center.x + icon_size, center.y - icon_size);
        ui.painter()
            .line_segment([tr, Pos2::new(tr.x - corner_len, tr.y)], stroke);
        ui.painter()
            .line_segment([tr, Pos2::new(tr.x, tr.y + corner_len)], stroke);

        // Bottom-left corner
        let bl = Pos2::new(center.x - icon_size, center.y + icon_size);
        ui.painter()
            .line_segment([bl, Pos2::new(bl.x + corner_len, bl.y)], stroke);
        ui.painter()
            .line_segment([bl, Pos2::new(bl.x, bl.y - corner_len)], stroke);

        // Bottom-right corner
        let br = Pos2::new(center.x + icon_size, center.y + icon_size);
        ui.painter()
            .line_segment([br, Pos2::new(br.x - corner_len, br.y)], stroke);
        ui.painter()
            .line_segment([br, Pos2::new(br.x, br.y - corner_len)], stroke);

        response.clicked()
    }

    /// Draws the CC/subtitles toggle button.
    fn draw_cc_button(&self, ui: &mut Ui, rect: Rect) -> bool {
        let response = ui.allocate_rect(rect, Sense::click());

        // Draw button background on hover or when subtitles are active
        let bg_color = if self.subtitles_visible {
            Color32::from_rgba_unmultiplied(255, 255, 255, 50)
        } else if response.hovered() {
            Color32::from_rgba_unmultiplied(255, 255, 255, 30)
        } else {
            Color32::TRANSPARENT
        };

        if bg_color != Color32::TRANSPARENT {
            ui.painter()
                .rect_filled(rect, CornerRadius::same(4), bg_color);
        }

        // Draw CC text icon
        let center = rect.center();
        let font_size = rect.width() * 0.4;

        // Use different color when subtitles are active vs inactive
        let text_color = if self.subtitles_visible {
            Color32::from_rgb(100, 200, 255) // Highlighted blue when active
        } else {
            self.config.icon_color
        };

        ui.painter().text(
            center,
            Align2::CENTER_CENTER,
            "CC",
            FontId::proportional(font_size),
            text_color,
        );

        response.clicked()
    }
}

/// Formats a duration as MM:SS or HH:MM:SS.
fn format_duration(d: Duration) -> String {
    let total_secs = d.as_secs();
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    if hours > 0 {
        format!("{hours}:{mins:02}:{secs:02}")
    } else {
        format!("{mins}:{secs:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0:00");
        assert_eq!(format_duration(Duration::from_secs(65)), "1:05");
        assert_eq!(format_duration(Duration::from_secs(3661)), "1:01:01");
    }
}
