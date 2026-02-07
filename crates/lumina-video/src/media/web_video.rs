//! Web video playback using browser's native HTMLVideoElement.
//!
//! This module provides hardware-accelerated video playback on web browsers
//! by leveraging the browser's built-in video decoder. Unlike native platforms
//! where we implement `VideoDecoderBackend`, web uses a fundamentally different
//! architecture:
//!
//! - Browser's `<video>` element handles decoding (hardware-accelerated)
//! - `requestVideoFrameCallback` provides frame-accurate timing
//! - Texture upload via `copyExternalImageToTexture` (WebGPU) or `texImage2D` (WebGL)
//! - HLS streaming via native support (Safari) or hls.js (Chrome/Firefox/Edge)
//!
//! # Architecture
//!
//! ```text
//! Rust/WASM (this module)          JavaScript (video-bridge.js)
//! ┌─────────────────────┐          ┌─────────────────────────────┐
//! │ WebVideoPlayer      │◄────────►│ Hidden <video> element      │
//! │   - state           │          │ requestVideoFrameCallback   │
//! │   - frame_ready     │          │ HLS.js for adaptive streaming│
//! │   - dimensions      │          │ Audio sync (native)         │
//! └─────────────────────┘          └─────────────────────────────┘
//! ```

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use egui_wgpu::wgpu;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{window, HtmlVideoElement};

use super::video::{VideoError, VideoMetadata, VideoState};

// Re-export egui for convenience in web builds
pub use egui;

/// Callback type for frame-ready notifications from JavaScript.
pub type FrameReadyCallback = Rc<RefCell<Option<Box<dyn FnMut()>>>>;

/// Web video player that wraps browser's HTMLVideoElement.
///
/// This provides a Rust-friendly interface to browser video playback,
/// handling HLS streaming, frame synchronization, and state management.
pub struct WebVideoPlayer {
    /// The underlying HTML video element
    video: HtmlVideoElement,
    /// Current playback state
    state: VideoState,
    /// Video metadata (populated after loadedmetadata event)
    metadata: Option<VideoMetadata>,
    /// HLS.js instance handle (for non-Safari browsers)
    hls_handle: Option<JsValue>,
    /// Callback closure for requestVideoFrameCallback
    frame_callback: Option<Closure<dyn FnMut(f64, JsValue)>>,
    /// Flag indicating a new frame is ready for texture upload
    frame_ready: Rc<RefCell<bool>>,
    /// Current video time in seconds (updated by frame callback).
    /// Reserved for future A/V sync - provides more precise timing than video.currentTime.
    #[allow(dead_code)]
    current_time: Rc<RefCell<f64>>,
    /// Whether this is an HLS stream
    is_hls: bool,
}

impl WebVideoPlayer {
    /// Creates a new web video player for the given URL.
    ///
    /// Automatically detects HLS streams (.m3u8) and uses hls.js on
    /// browsers without native HLS support.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        let window = window()
            .ok_or_else(|| VideoError::DecoderInit("No window object available".to_string()))?;

        let document = window
            .document()
            .ok_or_else(|| VideoError::DecoderInit("No document available".to_string()))?;

        // Create hidden video element
        let video: HtmlVideoElement = document
            .create_element("video")
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to create video element: {:?}", e))
            })?
            .dyn_into()
            .map_err(|_| VideoError::DecoderInit("Element is not a video".to_string()))?;

        // Configure video element for optimal playback
        video.set_cross_origin(Some("anonymous")); // Enable CORS for texture upload
        video.set_preload("auto");
        // set_plays_inline is the correct method name in web-sys
        video.set_attribute("playsinline", "true").ok(); // Required for iOS
                                                         // Start muted to comply with browser autoplay policies.
                                                         // User must explicitly unmute via UI or call set_muted(false) after user gesture.
        video.set_muted(true);

        // Hide from DOM but keep functional
        let style = video.style();
        let _ = style.set_property("position", "absolute");
        let _ = style.set_property("width", "1px");
        let _ = style.set_property("height", "1px");
        let _ = style.set_property("opacity", "0");
        let _ = style.set_property("pointer-events", "none");

        // Append to document body
        document
            .body()
            .ok_or_else(|| VideoError::DecoderInit("No document body".to_string()))?
            .append_child(&video)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to append video: {:?}", e)))?;

        let is_hls = url.contains(".m3u8") || url.contains("application/vnd.apple.mpegurl");
        let frame_ready = Rc::new(RefCell::new(false));
        let current_time = Rc::new(RefCell::new(0.0));

        let mut player = Self {
            video,
            state: VideoState::Loading,
            metadata: None,
            hls_handle: None,
            frame_callback: None,
            frame_ready,
            current_time,
            is_hls,
        };

        player.setup_event_listeners()?;
        player.load_source(url)?;

        Ok(player)
    }

    /// Sets up event listeners for video state changes.
    ///
    /// Currently relies on polling via `update_state()`. Event-based listeners
    /// for loadedmetadata, canplay, error, and ended events would improve
    /// responsiveness but are not yet implemented.
    fn setup_event_listeners(&mut self) -> Result<(), VideoError> {
        // Polling-based state updates are sufficient for current use cases.
        // Event listeners could be added for more responsive state changes.
        Ok(())
    }

    /// Loads the video source, using HLS.js if needed.
    fn load_source(&mut self, url: &str) -> Result<(), VideoError> {
        if self.is_hls {
            // Check for native HLS support (Safari)
            if !self
                .video
                .can_play_type("application/vnd.apple.mpegurl")
                .is_empty()
            {
                // Safari: native HLS
                self.video.set_src(url);
            } else {
                // Chrome/Firefox/Edge: use hls.js
                self.hls_handle = Some(init_hls_js(&self.video, url)?);
            }
        } else {
            // Direct video source (MP4, WebM, etc.)
            self.video.set_src(url);
        }

        Ok(())
    }

    /// Starts frame callback registration for accurate frame timing.
    pub fn start_frame_callbacks(&mut self) -> Result<(), VideoError> {
        let frame_ready = self.frame_ready.clone();
        let current_time = self.current_time.clone();

        let callback = Closure::new(move |now: f64, metadata: JsValue| {
            *frame_ready.borrow_mut() = true;
            // Extract mediaTime from metadata for accurate video time, fallback to now
            let media_time = js_sys::Reflect::get(&metadata, &"mediaTime".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(now / 1000.0);
            *current_time.borrow_mut() = media_time;
        });

        // Register the callback
        request_video_frame_callback(&self.video, &callback)?;

        self.frame_callback = Some(callback);
        Ok(())
    }

    /// Returns true if a new frame is ready for texture upload.
    pub fn is_frame_ready(&self) -> bool {
        *self.frame_ready.borrow()
    }

    /// Clears the frame-ready flag after texture upload.
    pub fn clear_frame_ready(&self) {
        *self.frame_ready.borrow_mut() = false;
    }

    /// Returns a reference to the underlying video element for texture upload.
    pub fn video_element(&self) -> &HtmlVideoElement {
        &self.video
    }

    /// Starts or resumes video playback.
    pub fn play(&self) -> Result<(), VideoError> {
        let promise = self
            .video
            .play()
            .map_err(|e| VideoError::Generic(format!("Play failed: {:?}", e)))?;

        // Fire and forget - we handle state via events
        let _ = promise;
        Ok(())
    }

    /// Pauses video playback.
    pub fn pause(&self) {
        self.video.pause().ok();
    }

    /// Seeks to a specific position.
    pub fn seek(&self, position: Duration) {
        self.video.set_current_time(position.as_secs_f64());
    }

    /// Sets the volume (0.0 to 1.0).
    pub fn set_volume(&self, volume: f32) {
        self.video.set_volume(volume.clamp(0.0, 1.0) as f64);
    }

    /// Returns the current volume (0.0 to 1.0).
    pub fn volume(&self) -> f32 {
        self.video.volume() as f32
    }

    /// Sets the muted state.
    pub fn set_muted(&self, muted: bool) {
        self.video.set_muted(muted);
    }

    /// Returns true if the video is muted.
    pub fn is_muted(&self) -> bool {
        self.video.muted()
    }

    /// Toggles the muted state.
    pub fn toggle_mute(&self) {
        self.video.set_muted(!self.video.muted());
    }

    // ========================================================================
    // Audio/Video Synchronization
    // ========================================================================
    //
    // HTMLVideoElement handles A/V sync natively - the browser's media pipeline
    // ensures audio and video frames are presented in sync. No additional
    // synchronization code is needed.
    //
    // ## Autoplay Policy
    //
    // Modern browsers require user interaction before playing audio. The video
    // element is created with `muted: true` initially to allow autoplay. Call
    // `set_muted(false)` after a user gesture (click/tap) to enable audio.
    //
    // ## Best Practice
    //
    // 1. Start videos muted (done automatically)
    // 2. Show an "unmute" button in the UI
    // 3. Call `set_muted(false)` when user clicks the unmute button
    // 4. If play() fails with NotAllowedError, video needs user gesture

    /// Returns the current playback position.
    pub fn position(&self) -> Duration {
        Duration::from_secs_f64(self.video.current_time())
    }

    /// Returns the total duration if known.
    pub fn duration(&self) -> Option<Duration> {
        let dur = self.video.duration();
        if dur.is_nan() || dur.is_infinite() {
            None
        } else {
            Some(Duration::from_secs_f64(dur))
        }
    }

    /// Returns the video dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.video.video_width(), self.video.video_height())
    }

    /// Returns true if the video is currently playing.
    pub fn is_playing(&self) -> bool {
        !self.video.paused() && !self.video.ended()
    }

    /// Returns true if the video has ended.
    pub fn is_ended(&self) -> bool {
        self.video.ended()
    }

    /// Returns the current buffering percentage (0-100).
    ///
    /// Note: This uses the end of the last buffered range, which may overestimate
    /// progress for sparse buffering (multiple non-contiguous ranges). For more
    /// accurate buffering info with HLS, use `hls_buffer_info()` instead.
    pub fn buffering_percent(&self) -> i32 {
        let buffered = self.video.buffered();
        let duration = self.video.duration();

        if duration.is_nan() || duration <= 0.0 || buffered.length() == 0 {
            return 0;
        }

        // Get the end of the last buffered range (may overestimate for sparse buffering)
        if let Ok(end) = buffered.end(buffered.length() - 1) {
            ((end / duration) * 100.0) as i32
        } else {
            0
        }
    }

    /// Returns the video metadata.
    pub fn metadata(&self) -> Option<&VideoMetadata> {
        self.metadata.as_ref()
    }

    /// Updates metadata from the video element (call after loadedmetadata event).
    pub fn update_metadata(&mut self) {
        let (width, height) = self.dimensions();
        if width > 0 && height > 0 {
            self.metadata = Some(VideoMetadata {
                width,
                height,
                duration: self.duration(),
                frame_rate: 30.0, // Browsers don't expose frame rate directly
                codec: "browser-native".to_string(),
                pixel_aspect_ratio: 1.0,
                start_time: None, // Browser doesn't expose stream start time
            });
        }
    }

    /// Returns the current playback state.
    pub fn state(&self) -> &VideoState {
        &self.state
    }

    /// Updates the playback state based on video element state.
    pub fn update_state(&mut self) {
        self.state = if self.video.ended() {
            VideoState::Ended
        } else if self.video.paused() {
            VideoState::Paused {
                position: self.position(),
            }
        } else if self.video.ready_state() < 3 {
            // HAVE_FUTURE_DATA = 3
            VideoState::Buffering {
                position: self.position(),
            }
        } else {
            VideoState::Playing {
                position: self.position(),
            }
        };
    }
}

impl Drop for WebVideoPlayer {
    fn drop(&mut self) {
        // Clean up HLS.js instance
        if let Some(hls) = self.hls_handle.take() {
            destroy_hls_js(&hls);
        }

        // Remove video element from DOM
        if let Some(parent) = self.video.parent_node() {
            let _ = parent.remove_child(&self.video);
        }
    }
}

// ============================================================================
// JavaScript interop functions
// ============================================================================

/// Initializes HLS.js for the given video element.
fn init_hls_js(video: &HtmlVideoElement, url: &str) -> Result<JsValue, VideoError> {
    js_init_hls(video, url)
        .map_err(|e| VideoError::DecoderInit(format!("HLS.js initialization failed: {:?}", e)))
}

/// Destroys an HLS.js instance.
fn destroy_hls_js(hls: &JsValue) {
    js_destroy_hls(hls);
}

/// Registers a requestVideoFrameCallback.
fn request_video_frame_callback(
    video: &HtmlVideoElement,
    callback: &Closure<dyn FnMut(f64, JsValue)>,
) -> Result<(), VideoError> {
    js_request_video_frame_callback(video, callback.as_ref().unchecked_ref())
        .map_err(|e| VideoError::Generic(format!("requestVideoFrameCallback failed: {:?}", e)))?;
    Ok(())
}

// ============================================================================
// wasm-bindgen extern declarations
// ============================================================================

#[wasm_bindgen(module = "/web/video-bridge.js")]
extern "C" {
    /// Initializes HLS.js and attaches it to the video element.
    /// Returns the Hls instance handle.
    #[wasm_bindgen(catch, js_name = "initHls")]
    fn js_init_hls(video: &HtmlVideoElement, url: &str) -> Result<JsValue, JsValue>;

    /// Destroys an HLS.js instance and cleans up resources.
    #[wasm_bindgen(js_name = "destroyHls")]
    fn js_destroy_hls(hls: &JsValue);

    /// Registers a requestVideoFrameCallback on the video element.
    /// The callback receives (now, metadata) where now is timestamp in ms.
    #[wasm_bindgen(catch, js_name = "requestVideoFrameCallback")]
    fn js_request_video_frame_callback(
        video: &HtmlVideoElement,
        callback: &js_sys::Function,
    ) -> Result<(), JsValue>;

    /// Gets the current HLS quality levels.
    #[wasm_bindgen(catch, js_name = "getHlsLevels")]
    fn js_get_hls_levels(hls: &JsValue) -> Result<JsValue, JsValue>;

    /// Sets the current HLS quality level (-1 for auto).
    #[wasm_bindgen(catch, js_name = "setHlsLevel")]
    fn js_set_hls_level(hls: &JsValue, level: i32) -> Result<(), JsValue>;

    /// Gets HLS.js buffer statistics.
    #[wasm_bindgen(catch, js_name = "getHlsBufferInfo")]
    fn js_get_hls_buffer_info(hls: &JsValue) -> Result<JsValue, JsValue>;
}

// ============================================================================
// HLS quality level management
// ============================================================================

/// Represents an HLS quality level.
#[derive(Debug, Clone)]
pub struct HlsQualityLevel {
    /// Level index (0-based)
    pub index: i32,
    /// Bitrate in bits per second
    pub bitrate: u32,
    /// Resolution width
    pub width: u32,
    /// Resolution height
    pub height: u32,
    /// Codec string
    pub codec: String,
}

impl WebVideoPlayer {
    /// Returns available HLS quality levels.
    pub fn hls_quality_levels(&self) -> Vec<HlsQualityLevel> {
        if let Some(hls) = &self.hls_handle {
            if let Ok(levels) = js_get_hls_levels(hls) {
                // Parse levels from JsValue (array of level objects)
                parse_hls_levels(&levels)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    }

    /// Sets the HLS quality level. Use -1 for automatic bitrate selection.
    pub fn set_hls_quality_level(&self, level: i32) {
        if let Some(hls) = &self.hls_handle {
            let _ = js_set_hls_level(hls, level);
        }
    }

    /// Returns the current HLS buffer info (buffer length, etc.).
    pub fn hls_buffer_info(&self) -> Option<HlsBufferInfo> {
        if let Some(hls) = &self.hls_handle {
            if let Ok(info) = js_get_hls_buffer_info(hls) {
                return parse_hls_buffer_info(&info);
            }
        }
        None
    }
}

/// HLS buffer statistics.
#[derive(Debug, Clone)]
pub struct HlsBufferInfo {
    /// Length of buffered content in seconds
    pub buffer_length: f64,
    /// Estimated bandwidth in bits per second
    pub bandwidth: u32,
    /// Current quality level index
    pub current_level: i32,
}

fn parse_hls_levels(levels: &JsValue) -> Vec<HlsQualityLevel> {
    let mut result = Vec::new();

    // Check if levels is an array
    if !js_sys::Array::is_array(levels) {
        return result;
    }

    let array = js_sys::Array::from(levels);
    for i in 0..array.length() {
        let item = array.get(i);
        if item.is_undefined() || item.is_null() {
            continue;
        }

        // Extract fields from the level object
        let index = js_sys::Reflect::get(&item, &"index".into())
            .ok()
            .and_then(|v| v.as_f64())
            .map(|v| v as i32)
            .unwrap_or(i as i32);

        let bitrate = js_sys::Reflect::get(&item, &"bitrate".into())
            .ok()
            .and_then(|v| v.as_f64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let width = js_sys::Reflect::get(&item, &"width".into())
            .ok()
            .and_then(|v| v.as_f64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let height = js_sys::Reflect::get(&item, &"height".into())
            .ok()
            .and_then(|v| v.as_f64())
            .map(|v| v as u32)
            .unwrap_or(0);

        let codec = js_sys::Reflect::get(&item, &"codec".into())
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| "unknown".to_string());

        result.push(HlsQualityLevel {
            index,
            bitrate,
            width,
            height,
            codec,
        });
    }

    result
}

// ============================================================================
// egui Integration
// ============================================================================

/// Response from showing a web video player in egui.
#[derive(Default)]
pub struct WebVideoPlayerResponse {
    /// Whether the video state changed (play/pause/seek).
    pub state_changed: bool,
    /// Whether user clicked on the video area.
    pub clicked: bool,
}

impl WebVideoPlayer {
    /// Shows the video player in an egui UI.
    ///
    /// Returns a texture ID that can be used with `ui.image()` or a custom
    /// paint callback to display the video frame.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let response = player.show(ui, available_size, &device, &queue, &bind_group_layout, &sampler);
    /// if response.clicked {
    ///     player.toggle_playback();
    /// }
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        size: egui::Vec2,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &mut Option<WebVideoTexture>,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
    ) -> WebVideoPlayerResponse {
        let mut response = WebVideoPlayerResponse::default();

        // Get video dimensions
        let (video_width, video_height) = self.dimensions();

        // Create or recreate texture if needed
        if video_width > 0 && video_height > 0 {
            let needs_recreate = texture
                .as_ref()
                .map(|t| t.dimensions() != (video_width, video_height))
                .unwrap_or(true);

            if needs_recreate {
                *texture = Some(WebVideoTexture::new(
                    device,
                    bind_group_layout,
                    sampler,
                    video_width,
                    video_height,
                ));
            }

            // Upload frame if ready
            if self.is_frame_ready() {
                if let Some(tex) = texture.as_ref() {
                    if let Ok(true) = tex.upload_frame(queue, &self.video) {
                        self.clear_frame_ready();
                    }
                }
            }
        }

        // Draw video area
        let (rect, ui_response) = ui.allocate_exact_size(size, egui::Sense::click());

        if ui_response.clicked() {
            response.clicked = true;
        }

        // If we have a valid texture, we could render it here using a paint callback
        // For now, this is a placeholder - actual rendering requires egui_wgpu integration
        if texture.is_some() && video_width > 0 && video_height > 0 {
            // The actual rendering would be done via egui's paint callback system
            // or by returning the bind_group for use in a custom shader
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
        } else {
            // Loading placeholder
            ui.painter()
                .rect_filled(rect, 0.0, egui::Color32::DARK_GRAY);
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Loading...",
                egui::FontId::default(),
                egui::Color32::WHITE,
            );
        }

        response
    }

    /// Toggles between play and pause.
    pub fn toggle_playback(&mut self) {
        if self.is_playing() {
            self.pause();
        } else {
            let _ = self.play();
        }
    }
}

fn parse_hls_buffer_info(info: &JsValue) -> Option<HlsBufferInfo> {
    if info.is_undefined() || info.is_null() {
        return None;
    }

    let buffer_length = js_sys::Reflect::get(info, &"bufferLength".into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let bandwidth = js_sys::Reflect::get(info, &"bandwidth".into())
        .ok()
        .and_then(|v| v.as_f64())
        .map(|v| v as u32)
        .unwrap_or(0);

    let current_level = js_sys::Reflect::get(info, &"currentLevel".into())
        .ok()
        .and_then(|v| v.as_f64())
        .map(|v| v as i32)
        .unwrap_or(-1);

    Some(HlsBufferInfo {
        buffer_length,
        bandwidth,
        current_level,
    })
}

// ============================================================================
// Web Video Texture Pipeline
// ============================================================================

/// Texture pipeline for uploading video frames to GPU on web.
///
/// Uses `wgpu::Queue::copy_external_image_to_texture` for efficient GPU-to-GPU
/// copy from the browser's video decoder to a wgpu texture. This is NOT true
/// zero-copy (WebGPU has no external memory import), but avoids CPU-side pixel
/// access.
///
/// # Performance
/// The browser compositor already has the decoded video frame in GPU memory.
/// `copyExternalImageToTexture` performs a GPU-to-GPU blit, typically sub-1ms
/// for 1080p content.
pub struct WebVideoTexture {
    /// The wgpu texture for rendering
    texture: wgpu::Texture,
    /// Texture view for shader access
    view: wgpu::TextureView,
    /// Bind group for the texture + sampler
    bind_group: wgpu::BindGroup,
    /// Current texture dimensions
    width: u32,
    height: u32,
}

impl WebVideoTexture {
    /// Creates a new web video texture with the given dimensions.
    pub fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("web_video_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // RGBA8 is the format browsers provide via copyExternalImageToTexture
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("web_video_bind_group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            texture,
            view,
            bind_group,
            width,
            height,
        }
    }

    /// Returns the bind group for rendering.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Returns the texture view.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Returns current dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Uploads a video frame from the HTMLVideoElement to the texture.
    ///
    /// Uses `copy_external_image_to_texture` for efficient GPU-to-GPU copy.
    /// Returns true if upload succeeded, false if video dimensions changed
    /// (caller should recreate the texture).
    pub fn upload_frame(
        &self,
        queue: &wgpu::Queue,
        video: &HtmlVideoElement,
    ) -> Result<bool, VideoError> {
        let video_width = video.video_width();
        let video_height = video.video_height();

        // Check if dimensions changed
        if video_width != self.width || video_height != self.height {
            return Ok(false); // Signal that texture needs recreation
        }

        // Skip upload if video has no content yet
        if video_width == 0 || video_height == 0 {
            return Ok(true);
        }

        // Use wgpu's copy_external_image_to_texture for efficient GPU-to-GPU copy
        // This is available on WebGPU backend
        queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::HTMLVideoElement(video.clone()),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d {
                width: video_width,
                height: video_height,
                depth_or_array_layers: 1,
            },
        );

        Ok(true)
    }
}

/// Inline WGSL shader for simple RGBA texture rendering.
/// This is a minimal shader that just samples an RGBA texture and outputs it.
const WEB_VIDEO_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle (covers -1 to 1 in NDC)
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var tex_coords = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.tex_coord = tex_coords[vertex_index];
    return output;
}

@group(0) @binding(0) var video_texture: texture_2d<f32>;
@group(0) @binding(1) var video_sampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(video_texture, video_sampler, input.tex_coord);
}
"#;

/// GPU resources for zero-copy web video rendering.
///
/// This is simpler than the native VideoRenderResources because web video
/// frames are already in RGBA format (no YUV conversion needed).
pub struct WebVideoRenderResources {
    /// Render pipeline for RGBA texture
    pipeline: wgpu::RenderPipeline,
    /// Bind group layout
    bind_group_layout: wgpu::BindGroupLayout,
    /// Texture sampler
    sampler: wgpu::Sampler,
}

impl WebVideoRenderResources {
    /// Creates web video render resources.
    pub fn new(wgpu_render_state: &egui_wgpu::RenderState) -> Self {
        let device = &wgpu_render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("web_video_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(WEB_VIDEO_SHADER)),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("web_video_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("web_video_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("web_video_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("web_video_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu_render_state.target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// Returns the render pipeline.
    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    /// Returns the bind group layout for creating WebVideoTexture.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Returns the sampler for creating WebVideoTexture.
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}

/// Paint callback for zero-copy web video rendering.
///
/// Use this with `egui::PaintCallback` to render video frames directly
/// from HTMLVideoElement to the screen without CPU readback.
pub struct WebVideoRenderCallback {
    /// The bind group containing the video texture
    bind_group: wgpu::BindGroup,
}

impl WebVideoRenderCallback {
    /// Creates a new render callback with the given texture bind group.
    pub fn new(bind_group: wgpu::BindGroup) -> Self {
        Self { bind_group }
    }
}

impl egui_wgpu::CallbackTrait for WebVideoRenderCallback {
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
        let Some(render_resources): Option<&WebVideoRenderResources> = resources.get() else {
            return;
        };

        // Set viewport to fill the callback rect
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

        render_pass.set_pipeline(&render_resources.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}
