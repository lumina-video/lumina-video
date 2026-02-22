//! Web MoQ decoder using WebCodecs VideoDecoder.
//!
//! This module provides MoQ (Media over QUIC) video decoding on web browsers using
//! the WebCodecs API. Unlike the native MoQ decoder which uses FFmpeg, this web
//! implementation leverages browser-native hardware decoding.
//!
//! # Architecture
//!
//! ```text
//! MoQ NAL units (via JS moq-lite) → WebCodecs VideoDecoder → VideoFrame
//!                                                         → copyExternalImageToTexture → wgpu
//! ```
//!
//! # Browser Requirements
//!
//! - WebCodecs: Chrome 94+, Firefox 130+, Safari 16.4+
//! - WebGPU: Chrome 113+, Firefox 141+, Safari 26+
//! - WebTransport: Chrome 97+, Firefox 114+ (or WebSocket fallback)
//!
//! # Zero-Copy Rendering
//!
//! Web doesn't have true zero-copy like native platforms (no external memory import),
//! but `copyExternalImageToTexture` performs a GPU-to-GPU blit that avoids CPU
//! readback, typically completing in sub-1ms for 1080p content.
//!
//! # Usage
//!
//! ```ignore
//! use lumina_video::media::web_moq_decoder::WebMoqDecoder;
//!
//! // Create decoder for a MoQ stream
//! let decoder = WebMoqDecoder::new("moqs://relay.example.com/live/stream")?;
//!
//! // In render loop, check for new frames
//! if let Some(frame_info) = decoder.poll_frame() {
//!     // Upload to wgpu texture via copyExternalImageToTexture
//!     decoder.copy_to_texture(&device, &queue, &texture)?;
//! }
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use web_sys::VideoFrame;

use super::video::{VideoError, VideoMetadata};

// ============================================================================
// wasm-bindgen extern declarations for moq-bridge.js
// ============================================================================

#[wasm_bindgen(module = "/web/moq-bridge.js")]
extern "C" {
    /// Creates a new WebCodecs VideoDecoder for MoQ streams.
    #[wasm_bindgen(catch, js_name = "createMoqDecoder")]
    fn js_create_moq_decoder(
        codec: &str,
        width: u32,
        height: u32,
        description: Option<js_sys::Uint8Array>,
        on_frame: &js_sys::Function,
        on_error: &js_sys::Function,
    ) -> Result<u32, JsValue>;

    /// Checks if a codec configuration is supported.
    #[wasm_bindgen(catch, js_name = "isCodecSupported")]
    async fn js_is_codec_supported(
        codec: &str,
        width: u32,
        height: u32,
    ) -> Result<JsValue, JsValue>;

    /// Decodes an encoded video chunk.
    #[wasm_bindgen(catch, js_name = "decodeChunk")]
    fn js_decode_chunk(
        decoder_id: u32,
        data: &js_sys::Uint8Array,
        timestamp_us: f64,
        is_keyframe: bool,
    ) -> Result<(), JsValue>;

    /// Gets the last decoded VideoFrame.
    #[wasm_bindgen(js_name = "getLastFrame")]
    fn js_get_last_frame(decoder_id: u32) -> Option<VideoFrame>;

    /// Gets decoder state information.
    #[wasm_bindgen(js_name = "getDecoderState")]
    fn js_get_decoder_state(decoder_id: u32) -> JsValue;

    /// Flushes the decoder.
    #[wasm_bindgen(catch, js_name = "flushDecoder")]
    async fn js_flush_decoder(decoder_id: u32) -> Result<(), JsValue>;

    /// Resets the decoder.
    #[wasm_bindgen(js_name = "resetDecoder")]
    fn js_reset_decoder(decoder_id: u32);

    /// Closes and cleans up a decoder.
    #[wasm_bindgen(js_name = "closeDecoder")]
    fn js_close_decoder(decoder_id: u32);

    /// Copies a VideoFrame to a WebGPU texture.
    #[wasm_bindgen(catch, js_name = "copyFrameToTexture")]
    fn js_copy_frame_to_texture(
        frame: &VideoFrame,
        device: &JsValue,
        texture: &JsValue,
    ) -> Result<bool, JsValue>;

    /// Closes a VideoFrame.
    #[wasm_bindgen(js_name = "closeFrame")]
    fn js_close_frame(frame: &VideoFrame);

    /// Checks if WebCodecs is supported.
    #[wasm_bindgen(js_name = "isWebCodecsSupported")]
    fn js_is_webcodecs_supported() -> bool;

    /// Gets WebCodecs capabilities.
    #[wasm_bindgen(js_name = "getWebCodecsCapabilities")]
    fn js_get_webcodecs_capabilities() -> JsValue;
}

// ============================================================================
// MoQ URL types (duplicated for wasm32 since moq module is not available)
// ============================================================================

/// Parsed MoQ URL for web.
///
/// URL format: `moq[s]://host[:port][/auth_path]/namespace`
/// - `moq://` = no TLS, `moqs://` = TLS
/// - `auth_path` = optional relay auth path (e.g., "anon" for anonymous)
/// - `namespace` = broadcast name (last path segment)
///
/// Examples:
/// - `moq://localhost:4443/anon/bbb` → relay `http://localhost:4443/anon`, broadcast `bbb`
/// - `moqs://relay.example.com/live` → relay `https://relay.example.com`, broadcast `live`
#[derive(Debug, Clone)]
pub struct WebMoqUrl {
    /// Remote host
    host: String,
    /// Remote port
    port: u16,
    /// Whether to use TLS
    use_tls: bool,
    /// Auth path (e.g., "anon" for anonymous access on dev relays)
    auth_path: Option<String>,
    /// Namespace / broadcast name (last path segment)
    namespace: String,
    /// Query string (e.g., "jwt=xxx" for authentication)
    query: Option<String>,
    /// Original URL (reserved for debugging/logging)
    #[allow(dead_code)]
    original: String,
}

impl WebMoqUrl {
    /// Parses a MoQ URL string.
    pub fn parse(url: &str) -> Result<Self, VideoError> {
        let original = url.to_string();

        // Check scheme
        let (use_tls, rest) = if let Some(rest) = url.strip_prefix("moqs://") {
            (true, rest)
        } else if let Some(rest) = url.strip_prefix("moq://") {
            (false, rest)
        } else {
            return Err(VideoError::OpenFailed(
                "URL must start with moq:// or moqs://".to_string(),
            ));
        };

        // Split off query string before parsing path
        let (rest, query) = match rest.find('?') {
            Some(idx) => (&rest[..idx], Some(rest[idx + 1..].to_string())),
            None => (rest, None),
        };

        // Split host:port from path
        let (authority, path) = match rest.find('/') {
            Some(idx) => (&rest[..idx], &rest[idx + 1..]),
            None => (rest, ""),
        };

        if authority.is_empty() {
            return Err(VideoError::OpenFailed("Missing host".to_string()));
        }

        // Parse host and port
        let (host, port) = if let Some(colon_idx) = authority.rfind(':') {
            let host = &authority[..colon_idx];
            let port: u16 = authority[colon_idx + 1..]
                .parse()
                .map_err(|_| VideoError::OpenFailed("Invalid port number".to_string()))?;
            (host.to_string(), port)
        } else {
            (authority.to_string(), 443)
        };

        // Parse path: last segment = namespace, preceding segments = auth_path
        let path = path.trim_matches('/');
        if path.is_empty() {
            return Err(VideoError::OpenFailed("Missing namespace".to_string()));
        }

        let (auth_path, namespace) = match path.rfind('/') {
            Some(idx) => (Some(path[..idx].to_string()), path[idx + 1..].to_string()),
            None => (None, path.to_string()),
        };

        if namespace.is_empty() {
            return Err(VideoError::OpenFailed("Missing namespace".to_string()));
        }

        Ok(WebMoqUrl {
            host,
            port,
            use_tls,
            auth_path,
            namespace,
            query,
            original,
        })
    }

    /// Returns the WebTransport URL for connection (includes auth path if present).
    pub fn webtransport_url(&self) -> String {
        let scheme = if self.use_tls { "https" } else { "http" };
        let path = match &self.auth_path {
            Some(p) => format!("/{}", p),
            None => String::new(),
        };
        match &self.query {
            Some(q) => format!("{}://{}:{}{}?{}", scheme, self.host, self.port, path, q),
            None => format!("{}://{}:{}{}", scheme, self.host, self.port, path),
        }
    }

    /// Returns the namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns true if this is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        url.starts_with("moq://") || url.starts_with("moqs://")
    }
}

// ============================================================================
// Frame information from decoder callback
// ============================================================================

/// Information about a decoded frame.
#[derive(Debug, Clone)]
pub struct WebMoqFrameInfo {
    /// Presentation timestamp in microseconds
    pub timestamp_us: i64,
    /// Display width
    pub width: u32,
    /// Display height
    pub height: u32,
    /// Duration in microseconds (may be 0)
    pub duration_us: i64,
}

// ============================================================================
// Decoder state shared between Rust and JS callbacks
// ============================================================================

/// Shared decoder state updated by JS callbacks.
struct DecoderSharedState {
    /// Latest frame info from decoder output callback
    latest_frame: Option<WebMoqFrameInfo>,
    /// Error message if decoder failed
    error_message: Option<String>,
    /// Whether a new frame is available since last poll
    frame_ready: bool,
    /// Total frames decoded
    frame_count: u64,
}

impl Default for DecoderSharedState {
    fn default() -> Self {
        Self {
            latest_frame: None,
            error_message: None,
            frame_ready: false,
            frame_count: 0,
        }
    }
}

// ============================================================================
// WebMoqDecoder implementation
// ============================================================================

/// Decoder state for MoQ streams on web.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebMoqDecoderState {
    /// Initial state, decoder not created
    Uninitialized,
    /// Decoder configured and ready
    Configured,
    /// Actively decoding frames
    Decoding,
    /// Decoder closed or ended
    Closed,
    /// Error occurred
    Error,
}

/// Web MoQ decoder using WebCodecs VideoDecoder.
///
/// This decoder receives encoded NAL units from MoQ transport (via JavaScript
/// moq-lite library) and decodes them using the browser's WebCodecs API.
/// Decoded frames can be efficiently copied to wgpu textures using
/// `copyExternalImageToTexture`.
pub struct WebMoqDecoder {
    /// Parsed MoQ URL
    url: WebMoqUrl,
    /// JavaScript decoder handle ID
    decoder_id: Option<u32>,
    /// Shared state with JS callbacks
    shared: Rc<RefCell<DecoderSharedState>>,
    /// Current decoder state
    state: WebMoqDecoderState,
    /// Video metadata
    metadata: VideoMetadata,
    /// Closure for frame callback (must be kept alive)
    _frame_callback: Option<Closure<dyn FnMut(f64, u32, u32, f64)>>,
    /// Closure for error callback (must be kept alive)
    _error_callback: Option<Closure<dyn FnMut(String)>>,
    /// Codec string for WebCodecs
    codec: String,
    /// Codec-specific description (SPS/PPS for H.264)
    description: Option<Vec<u8>>,
}

impl WebMoqDecoder {
    /// Creates a new Web MoQ decoder for the given URL.
    ///
    /// The decoder is created in an uninitialized state. Call `configure()`
    /// with codec information from the MoQ catalog before decoding frames.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Check WebCodecs support
        if !js_is_webcodecs_supported() {
            return Err(VideoError::DecoderInit(
                "WebCodecs VideoDecoder is not supported in this browser".to_string(),
            ));
        }

        let moq_url = WebMoqUrl::parse(url)?;

        Ok(Self {
            url: moq_url,
            decoder_id: None,
            shared: Rc::new(RefCell::new(DecoderSharedState::default())),
            state: WebMoqDecoderState::Uninitialized,
            metadata: VideoMetadata {
                width: 0,
                height: 0,
                duration: None, // Live streams have no duration
                frame_rate: 30.0,
                codec: String::new(),
                pixel_aspect_ratio: 1.0,
                start_time: None,
            },
            _frame_callback: None,
            _error_callback: None,
            codec: String::new(),
            description: None,
        })
    }

    /// Configures the decoder with codec information.
    ///
    /// This should be called with information from the MoQ catalog after
    /// connecting to the stream.
    ///
    /// # Arguments
    ///
    /// * `codec` - WebCodecs codec string (e.g., "avc1.42E01E" for H.264 baseline)
    /// * `width` - Video width in pixels
    /// * `height` - Video height in pixels
    /// * `frame_rate` - Frame rate (fps)
    /// * `description` - Optional codec-specific description (SPS/PPS for H.264)
    pub fn configure(
        &mut self,
        codec: &str,
        width: u32,
        height: u32,
        frame_rate: f32,
        description: Option<&[u8]>,
    ) -> Result<(), VideoError> {
        // Close existing decoder if any
        if let Some(id) = self.decoder_id.take() {
            js_close_decoder(id);
        }

        // Store codec info
        self.codec = codec.to_string();
        self.description = description.map(|d| d.to_vec());

        // Update metadata
        self.metadata.width = width;
        self.metadata.height = height;
        self.metadata.frame_rate = frame_rate;
        self.metadata.codec = codec.to_string();

        // Create callbacks
        let shared = self.shared.clone();
        let frame_callback = Closure::new(move |timestamp: f64, w: u32, h: u32, duration: f64| {
            let mut state = shared.borrow_mut();
            state.latest_frame = Some(WebMoqFrameInfo {
                timestamp_us: timestamp as i64,
                width: w,
                height: h,
                duration_us: duration as i64,
            });
            state.frame_ready = true;
            state.frame_count += 1;
        });

        let shared_err = self.shared.clone();
        let error_callback = Closure::new(move |error: String| {
            let mut state = shared_err.borrow_mut();
            state.error_message = Some(error);
        });

        // Convert description to Uint8Array if provided
        let desc_array = description.map(|d| {
            let arr = js_sys::Uint8Array::new_with_length(d.len() as u32);
            arr.copy_from(d);
            arr
        });

        // Create the decoder
        let decoder_id = js_create_moq_decoder(
            codec,
            width,
            height,
            desc_array,
            frame_callback.as_ref().unchecked_ref(),
            error_callback.as_ref().unchecked_ref(),
        )
        .map_err(|e| {
            VideoError::DecoderInit(format!("Failed to create WebCodecs decoder: {:?}", e))
        })?;

        self.decoder_id = Some(decoder_id);
        self._frame_callback = Some(frame_callback);
        self._error_callback = Some(error_callback);
        self.state = WebMoqDecoderState::Configured;

        tracing::info!(
            "WebMoqDecoder: Configured decoder {} with codec={}, {}x{}@{}fps",
            decoder_id,
            codec,
            width,
            height,
            frame_rate
        );

        Ok(())
    }

    /// Decodes an encoded video chunk (NAL unit from MoQ).
    ///
    /// This should be called with each frame received from the MoQ transport.
    /// The hang crate's frame format includes timestamp and keyframe information.
    ///
    /// # Arguments
    ///
    /// * `data` - Encoded frame data (H.264/H.265/AV1 NAL units)
    /// * `timestamp_us` - Presentation timestamp in microseconds (from hang::Timescale)
    /// * `is_keyframe` - Whether this is a keyframe (from hang::Frame)
    pub fn decode(
        &mut self,
        data: &[u8],
        timestamp_us: i64,
        is_keyframe: bool,
    ) -> Result<(), VideoError> {
        let decoder_id = self
            .decoder_id
            .ok_or_else(|| VideoError::DecodeFailed("Decoder not configured".to_string()))?;

        // Check for errors from previous operations
        if let Some(error) = self.shared.borrow().error_message.clone() {
            self.state = WebMoqDecoderState::Error;
            return Err(VideoError::DecodeFailed(error));
        }

        // Convert data to Uint8Array
        let data_array = js_sys::Uint8Array::new_with_length(data.len() as u32);
        data_array.copy_from(data);

        // Decode the chunk
        js_decode_chunk(decoder_id, &data_array, timestamp_us as f64, is_keyframe)
            .map_err(|e| VideoError::DecodeFailed(format!("Decode failed: {:?}", e)))?;

        self.state = WebMoqDecoderState::Decoding;
        Ok(())
    }

    /// Polls for a new decoded frame.
    ///
    /// Returns frame info if a new frame is available since the last poll.
    /// This is non-blocking.
    pub fn poll_frame(&mut self) -> Option<WebMoqFrameInfo> {
        let mut state = self.shared.borrow_mut();
        if state.frame_ready {
            state.frame_ready = false;
            state.latest_frame.clone()
        } else {
            None
        }
    }

    /// Gets the last decoded VideoFrame from JavaScript.
    ///
    /// The returned frame must be closed after use (call `close_frame()`).
    /// This is used for copying to wgpu texture.
    pub fn get_last_frame(&self) -> Option<VideoFrame> {
        self.decoder_id.and_then(js_get_last_frame)
    }

    /// Copies the current frame to a wgpu texture.
    ///
    /// This uses `copyExternalImageToTexture` for efficient GPU-to-GPU blit.
    /// Call this after `poll_frame()` returns Some to render the frame.
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu queue for the copy operation
    /// * `texture` - Target wgpu texture (must be RGBA8 format)
    pub fn copy_to_wgpu_texture(
        &self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(), VideoError> {
        let frame = self.get_last_frame().ok_or_else(|| {
            VideoError::DecodeFailed("No frame available for texture copy".to_string())
        })?;

        let width = frame.display_width();
        let height = frame.display_height();

        let source_info = wgpu::CopyExternalImageSourceInfo {
            source: wgpu::ExternalImageSource::VideoFrame(frame),
            origin: wgpu::Origin2d::ZERO,
            flip_y: false,
        };

        queue.copy_external_image_to_texture(
            &source_info,
            wgpu::CopyExternalImageDestInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        if let wgpu::ExternalImageSource::VideoFrame(vf) = source_info.source {
            vf.close();
        }

        Ok(())
    }

    /// Resets the decoder for seeking or error recovery.
    pub fn reset(&mut self) {
        if let Some(id) = self.decoder_id {
            js_reset_decoder(id);
            let mut state = self.shared.borrow_mut();
            state.latest_frame = None;
            state.frame_ready = false;
            state.error_message = None;
        }
    }

    /// Returns the current decoder state.
    pub fn decoder_state(&self) -> WebMoqDecoderState {
        self.state
    }

    /// Returns true if this is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        WebMoqUrl::is_moq_url(url)
    }

    /// Returns the error message if in error state.
    pub fn error_message(&self) -> Option<String> {
        self.shared.borrow().error_message.clone()
    }

    /// Returns the frame count.
    pub fn frame_count(&self) -> u64 {
        self.shared.borrow().frame_count
    }

    /// Returns the video metadata.
    pub fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    /// Returns the WebTransport URL for the MoQ connection.
    pub fn webtransport_url(&self) -> String {
        self.url.webtransport_url()
    }

    /// Returns the MoQ namespace.
    pub fn namespace(&self) -> &str {
        self.url.namespace()
    }

    /// Checks if a codec is supported by WebCodecs.
    pub async fn is_codec_supported(codec: &str, width: u32, height: u32) -> bool {
        match js_is_codec_supported(codec, width, height).await {
            Ok(supported) => supported.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    }
}

impl Drop for WebMoqDecoder {
    fn drop(&mut self) {
        if let Some(id) = self.decoder_id.take() {
            js_close_decoder(id);
        }
    }
}

// ============================================================================
// WebMoqTexture - GPU texture for zero-copy rendering
// ============================================================================

/// GPU texture for zero-copy MoQ video rendering on web.
///
/// This texture is designed for efficient frame upload via `copyExternalImageToTexture`.
/// It automatically handles texture recreation when dimensions change.
pub struct WebMoqTexture {
    /// The wgpu texture
    texture: wgpu::Texture,
    /// Texture view for rendering
    view: wgpu::TextureView,
    /// Current dimensions
    width: u32,
    height: u32,
}

impl WebMoqTexture {
    /// Creates a new texture for MoQ video frames.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("web_moq_texture"),
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

        Self {
            texture,
            view,
            width,
            height,
        }
    }

    /// Returns the texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Returns the texture view for rendering.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Returns current dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Checks if texture needs recreation for new dimensions.
    pub fn needs_resize(&self, width: u32, height: u32) -> bool {
        self.width != width || self.height != height
    }
}

// ============================================================================
// Codec string helpers
// ============================================================================

/// Common codec strings for WebCodecs.
pub mod codec_strings {
    /// H.264 Baseline Profile Level 3.0 (widely supported)
    pub const H264_BASELINE: &str = "avc1.42E01E";
    /// H.264 Baseline Profile Level 3.1
    pub const H264_BASELINE_31: &str = "avc1.42E01F";
    /// H.264 Main Profile Level 3.1
    pub const H264_MAIN: &str = "avc1.4D401F";
    /// H.264 High Profile Level 4.0
    pub const H264_HIGH: &str = "avc1.640028";
    /// H.264 High Profile Level 5.1
    pub const H264_HIGH_51: &str = "avc1.640033";

    /// H.265/HEVC Main Profile
    pub const H265_MAIN: &str = "hev1.1.6.L93.B0";
    /// H.265/HEVC Main 10 Profile
    pub const H265_MAIN_10: &str = "hev1.2.4.L93.B0";

    /// AV1 Main Profile Level 4.0
    pub const AV1_MAIN: &str = "av01.0.08M.08";
    /// AV1 Main Profile Level 5.1
    pub const AV1_MAIN_51: &str = "av01.0.13M.08";

    /// VP9 Profile 0
    pub const VP9_PROFILE_0: &str = "vp09.00.10.08";
    /// VP9 Profile 2 (10-bit)
    pub const VP9_PROFILE_2: &str = "vp09.02.10.10";
}

// ============================================================================
// wasm-bindgen extern declarations for moq-transport-bridge.js
// ============================================================================

#[wasm_bindgen(module = "/web/moq-transport-bridge.js")]
extern "C" {
    /// Connects to a MoQ relay and subscribes to a broadcast.
    #[wasm_bindgen(catch, js_name = "moqConnect")]
    async fn js_moq_connect(url: &str, namespace: &str) -> Result<JsValue, JsValue>;

    /// Disconnects a MoQ session.
    #[wasm_bindgen(js_name = "moqDisconnect")]
    fn js_moq_disconnect(session_id: u32);

    /// Gets the session state string.
    #[wasm_bindgen(js_name = "moqGetSessionState")]
    fn js_moq_get_session_state(session_id: u32) -> String;

    /// Gets the error message for a session.
    #[wasm_bindgen(js_name = "moqGetError")]
    fn js_moq_get_error(session_id: u32) -> Option<String>;

    /// Gets the parsed catalog as JSON string.
    #[wasm_bindgen(js_name = "moqGetCatalog")]
    fn js_moq_get_catalog(session_id: u32) -> Option<String>;

    /// Starts video decoding for a track.
    #[wasm_bindgen(catch, js_name = "moqStartVideo")]
    fn js_moq_start_video(
        session_id: u32,
        track_name: &str,
        codec: &str,
        width: u32,
        height: u32,
        container_kind: &str,
        timescale: u32,
        description: Option<String>,
        on_frame: &js_sys::Function,
        on_error: &js_sys::Function,
    ) -> Result<u32, JsValue>;

    /// Gets the latest decoded VideoFrame.
    #[wasm_bindgen(js_name = "moqGetVideoFrame")]
    fn js_moq_get_video_frame(decoder_id: u32) -> Option<VideoFrame>;

    /// Closes a video decoder.
    #[wasm_bindgen(js_name = "moqCloseVideo")]
    fn js_moq_close_video(decoder_id: u32);

    /// Starts audio decoding and playback.
    #[wasm_bindgen(catch, js_name = "moqStartAudio")]
    async fn js_moq_start_audio(
        session_id: u32,
        track_name: &str,
        codec: &str,
        sample_rate: u32,
        channels: u32,
        container_kind: &str,
        timescale: u32,
        description: Option<String>,
    ) -> Result<JsValue, JsValue>;

    /// Sets audio muted state.
    #[wasm_bindgen(js_name = "moqSetAudioMuted")]
    fn js_moq_set_audio_muted(audio_id: i32, muted: bool);

    /// Sets audio volume.
    #[wasm_bindgen(js_name = "moqSetAudioVolume")]
    fn js_moq_set_audio_volume(audio_id: i32, volume: f32);

    /// Gets audio state.
    #[wasm_bindgen(js_name = "moqGetAudioState")]
    fn js_moq_get_audio_state(audio_id: i32) -> JsValue;

    /// Closes an audio handle.
    #[wasm_bindgen(js_name = "moqCloseAudio")]
    fn js_moq_close_audio(audio_id: i32);

    /// Gets session stats.
    #[wasm_bindgen(js_name = "moqGetStats")]
    fn js_moq_get_stats(session_id: u32) -> JsValue;

    /// Gets extended stats for diagnostics overlay.
    #[wasm_bindgen(js_name = "moqGetExtendedStats")]
    fn js_moq_get_extended_stats(session_id: u32) -> JsValue;
}

// ============================================================================
// WebMoqSession — full MoQ lifecycle management
// ============================================================================

/// State of a MoQ session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebMoqSessionState {
    /// Waiting for connect() to be called
    Init,
    /// Connection in progress
    Connecting,
    /// Connected, waiting for catalog
    Connected,
    /// Catalog received, ready to start playback
    CatalogReady,
    /// Video (and optionally audio) are playing
    Playing,
    /// Error occurred
    Error(String),
    /// Session closed
    Closed,
}

/// Parsed catalog from a MoQ broadcast.
#[derive(Debug, Clone)]
pub struct WebMoqCatalog {
    /// Video renditions
    pub video: Vec<WebMoqVideoRendition>,
    /// Audio renditions
    pub audio: Vec<WebMoqAudioRendition>,
}

/// A video rendition from the catalog.
#[derive(Debug, Clone)]
pub struct WebMoqVideoRendition {
    /// Track name in the MoQ broadcast.
    pub name: String,
    /// WebCodecs codec string (e.g., `"avc1.42E01E"`).
    pub codec: String,
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Frame rate (fps), if specified in catalog.
    pub framerate: Option<f32>,
    /// Container format kind (`"legacy"` or `"cmaf"`).
    pub container_kind: String,
    /// Timescale for CMAF timestamps (e.g., 90000 for 90kHz).
    pub timescale: u32,
    /// Hex-encoded codec description (SPS/PPS for H.264).
    pub description: Option<String>,
}

/// An audio rendition from the catalog.
#[derive(Debug, Clone)]
pub struct WebMoqAudioRendition {
    /// Track name in the MoQ broadcast.
    pub name: String,
    /// WebCodecs codec string (e.g., `"opus"`).
    pub codec: String,
    /// Audio sample rate in Hz (e.g., 48000).
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u32,
    /// Container format kind (`"legacy"` or `"cmaf"`).
    pub container_kind: String,
    /// Timescale for CMAF timestamps.
    pub timescale: u32,
    /// Hex-encoded codec description.
    pub description: Option<String>,
}

/// Full MoQ session managing connection, catalog, video, and audio.
pub struct WebMoqSession {
    /// Parsed MoQ URL
    url: WebMoqUrl,
    /// JavaScript session ID
    session_id: Option<u32>,
    /// Current session state
    state: WebMoqSessionState,
    /// Parsed catalog
    catalog: Option<WebMoqCatalog>,
    /// Video decoder ID (from JS)
    video_decoder_id: Option<u32>,
    /// Audio handle ID (from JS, -1 means unsupported)
    audio_id: Option<i32>,
    /// Shared state for video frame callbacks
    video_shared: Rc<RefCell<DecoderSharedState>>,
    /// Frame callback closure (must keep alive)
    _video_frame_cb: Option<Closure<dyn FnMut(f64, u32, u32, f64)>>,
    /// Error callback closure (must keep alive)
    _video_error_cb: Option<Closure<dyn FnMut(String)>>,
    /// Video dimensions (updated from frame callback)
    video_width: u32,
    video_height: u32,
    /// Volume and mute state
    volume: f32,
    muted: bool,
}

impl WebMoqSession {
    /// Creates a new MoQ session for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        let moq_url = WebMoqUrl::parse(url)?;
        Ok(Self {
            url: moq_url,
            session_id: None,
            state: WebMoqSessionState::Init,
            catalog: None,
            video_decoder_id: None,
            audio_id: None,
            video_shared: Rc::new(RefCell::new(DecoderSharedState::default())),
            _video_frame_cb: None,
            _video_error_cb: None,
            video_width: 0,
            video_height: 0,
            volume: 1.0,
            muted: true, // Start muted per browser autoplay policy
        })
    }

    /// Initiates the async connection. Returns a future that resolves when connected.
    pub async fn connect(&mut self) -> Result<(), VideoError> {
        self.state = WebMoqSessionState::Connecting;
        let wt_url = self.url.webtransport_url();
        let namespace = self.url.namespace().to_string();

        match js_moq_connect(&wt_url, &namespace).await {
            Ok(id_val) => {
                let id = id_val
                    .as_f64()
                    .filter(|f| *f >= 0.0 && *f <= u32::MAX as f64)
                    .map(|f| f as u32)
                    .ok_or_else(|| {
                        VideoError::DecoderInit(format!("Invalid session ID from JS: {:?}", id_val))
                    })?;
                self.session_id = Some(id);
                self.state = WebMoqSessionState::Connected;
                Ok(())
            }
            Err(e) => {
                let msg = format!("{:?}", e);
                self.state = WebMoqSessionState::Error(msg.clone());
                Err(VideoError::OpenFailed(msg))
            }
        }
    }

    /// Polls the session state, checking for catalog availability.
    /// Call this each frame from the render loop.
    pub fn poll_state(&mut self) {
        let Some(session_id) = self.session_id else {
            return;
        };

        let js_state = js_moq_get_session_state(session_id);
        match js_state.as_str() {
            "catalog" | "playing" => {
                // Check if we need to parse the catalog
                if self.catalog.is_none() {
                    if let Some(catalog_json) = js_moq_get_catalog(session_id) {
                        self.parse_catalog(&catalog_json);
                    }
                }
                if self.catalog.is_some()
                    && !matches!(
                        self.state,
                        WebMoqSessionState::Playing | WebMoqSessionState::CatalogReady
                    )
                {
                    self.state = WebMoqSessionState::CatalogReady;
                }
            }
            "error" => {
                let err = js_moq_get_error(session_id).unwrap_or_default();
                self.state = WebMoqSessionState::Error(err);
            }
            "closed" => {
                self.state = WebMoqSessionState::Closed;
            }
            _ => {}
        }
    }

    fn parse_catalog(&mut self, json: &str) {
        // Parse the catalog JSON from moq-transport-bridge.js
        let Ok(parsed) = js_sys::JSON::parse(json) else {
            return;
        };

        let mut video = Vec::new();
        let mut audio = Vec::new();

        // Parse video renditions
        if let Ok(video_arr) = js_sys::Reflect::get(&parsed, &"video".into()) {
            if let Some(arr) = video_arr.dyn_ref::<js_sys::Array>() {
                for i in 0..arr.length() {
                    let item = arr.get(i);
                    let name = get_str(&item, "name").unwrap_or_default();
                    let codec = get_str(&item, "codec").unwrap_or_default();
                    let width = get_u32(&item, "width");
                    let height = get_u32(&item, "height");
                    let framerate = get_f32(&item, "framerate");
                    let description = get_str(&item, "description");
                    let (container_kind, timescale) = get_container(&item);

                    video.push(WebMoqVideoRendition {
                        name,
                        codec,
                        width,
                        height,
                        framerate,
                        container_kind,
                        timescale,
                        description,
                    });
                }
            }
        }

        // Parse audio renditions
        if let Ok(audio_arr) = js_sys::Reflect::get(&parsed, &"audio".into()) {
            if let Some(arr) = audio_arr.dyn_ref::<js_sys::Array>() {
                for i in 0..arr.length() {
                    let item = arr.get(i);
                    let name = get_str(&item, "name").unwrap_or_default();
                    let codec = get_str(&item, "codec").unwrap_or_default();
                    let sample_rate = get_u32(&item, "sampleRate");
                    let channels = get_u32(&item, "channels");
                    let description = get_str(&item, "description");
                    let (container_kind, timescale) = get_container(&item);

                    audio.push(WebMoqAudioRendition {
                        name,
                        codec,
                        sample_rate,
                        channels,
                        container_kind,
                        timescale,
                        description,
                    });
                }
            }
        }

        self.catalog = Some(WebMoqCatalog { video, audio });
    }

    /// Starts video and audio playback from the catalog.
    /// Call after poll_state() returns CatalogReady.
    pub async fn start_playback(&mut self) -> Result<(), VideoError> {
        let session_id = self
            .session_id
            .ok_or_else(|| VideoError::OpenFailed("Not connected".to_string()))?;

        let catalog = self
            .catalog
            .as_ref()
            .ok_or_else(|| VideoError::OpenFailed("No catalog".to_string()))?
            .clone();

        // Start first video rendition
        if let Some(v) = catalog.video.first() {
            self.start_video(session_id, v)?;
        }

        // Start first audio rendition (async for AudioContext init)
        if let Some(a) = catalog.audio.first() {
            self.start_audio(session_id, a).await;
        }

        self.state = WebMoqSessionState::Playing;
        Ok(())
    }

    fn start_video(
        &mut self,
        session_id: u32,
        rendition: &WebMoqVideoRendition,
    ) -> Result<(), VideoError> {
        // Create callbacks
        let shared = self.video_shared.clone();
        let frame_cb = Closure::new(move |timestamp: f64, w: u32, h: u32, duration: f64| {
            let mut state = shared.borrow_mut();
            state.latest_frame = Some(WebMoqFrameInfo {
                timestamp_us: timestamp as i64,
                width: w,
                height: h,
                duration_us: duration as i64,
            });
            state.frame_ready = true;
            state.frame_count += 1;
        });

        let shared_err = self.video_shared.clone();
        let error_cb = Closure::new(move |error: String| {
            let mut state = shared_err.borrow_mut();
            state.error_message = Some(error);
        });

        let decoder_id = js_moq_start_video(
            session_id,
            &rendition.name,
            &rendition.codec,
            rendition.width,
            rendition.height,
            &rendition.container_kind,
            rendition.timescale,
            rendition.description.clone(),
            frame_cb.as_ref().unchecked_ref(),
            error_cb.as_ref().unchecked_ref(),
        )
        .map_err(|e| VideoError::DecoderInit(format!("Failed to start video: {:?}", e)))?;

        self.video_decoder_id = Some(decoder_id);
        self.video_width = rendition.width;
        self.video_height = rendition.height;
        self._video_frame_cb = Some(frame_cb);
        self._video_error_cb = Some(error_cb);

        tracing::info!(
            "WebMoqSession: Started video decoder {} for {}",
            decoder_id,
            rendition.name
        );
        Ok(())
    }

    async fn start_audio(&mut self, session_id: u32, rendition: &WebMoqAudioRendition) {
        match js_moq_start_audio(
            session_id,
            &rendition.name,
            &rendition.codec,
            rendition.sample_rate,
            rendition.channels,
            &rendition.container_kind,
            rendition.timescale,
            rendition.description.clone(),
        )
        .await
        {
            Ok(id_val) => {
                let id = id_val.as_f64().unwrap_or(-1.0) as i32;
                if id >= 0 {
                    self.audio_id = Some(id);
                    // Apply current volume/mute state
                    js_moq_set_audio_muted(id, self.muted);
                    js_moq_set_audio_volume(id, self.volume);
                    tracing::info!("WebMoqSession: Started audio {} for {}", id, rendition.name);
                } else {
                    tracing::warn!(
                        "WebMoqSession: Audio codec {} not supported",
                        rendition.codec
                    );
                }
            }
            Err(e) => {
                tracing::warn!("WebMoqSession: Audio start failed: {:?}", e);
            }
        }
    }

    /// Polls for a new decoded video frame.
    pub fn poll_frame(&mut self) -> Option<WebMoqFrameInfo> {
        let mut state = self.video_shared.borrow_mut();
        if state.frame_ready {
            state.frame_ready = false;
            if let Some(ref frame) = state.latest_frame {
                self.video_width = frame.width;
                self.video_height = frame.height;
            }
            state.latest_frame.clone()
        } else {
            None
        }
    }

    /// Gets the latest decoded VideoFrame for GPU texture copy.
    pub fn get_video_frame(&self) -> Option<VideoFrame> {
        self.video_decoder_id.and_then(js_moq_get_video_frame)
    }

    /// Copies the current video frame to a wgpu texture.
    ///
    /// `moqGetVideoFrame` transfers ownership (nulls out JS handle), so we own it.
    /// After the GPU copy we extract and `.close()` the frame to avoid GC warnings.
    pub fn copy_to_wgpu_texture(
        &self,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(), VideoError> {
        let frame = self
            .get_video_frame()
            .ok_or_else(|| VideoError::DecodeFailed("No video frame available".to_string()))?;

        let width = frame.display_width();
        let height = frame.display_height();

        let source_info = wgpu::CopyExternalImageSourceInfo {
            source: wgpu::ExternalImageSource::VideoFrame(frame),
            origin: wgpu::Origin2d::ZERO,
            flip_y: false,
        };

        queue.copy_external_image_to_texture(
            &source_info,
            wgpu::CopyExternalImageDestInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Close the VideoFrame to release GPU resources (prevents GC warning)
        if let wgpu::ExternalImageSource::VideoFrame(vf) = source_info.source {
            vf.close();
        }

        Ok(())
    }

    /// Returns the video dimensions.
    pub fn video_dimensions(&self) -> (u32, u32) {
        (self.video_width, self.video_height)
    }

    /// Returns the current session state.
    pub fn state(&self) -> &WebMoqSessionState {
        &self.state
    }

    /// Returns the catalog if available.
    pub fn catalog(&self) -> Option<&WebMoqCatalog> {
        self.catalog.as_ref()
    }

    /// Returns the total video frame count.
    pub fn video_frame_count(&self) -> u64 {
        self.video_shared.borrow().frame_count
    }

    /// Sets the audio muted state.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        if let Some(id) = self.audio_id {
            js_moq_set_audio_muted(id, muted);
        }
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Sets the audio volume (0.0 to 1.0).
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
        if let Some(id) = self.audio_id {
            js_moq_set_audio_volume(id, self.volume);
        }
    }

    /// Returns the current volume.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Toggles mute state.
    pub fn toggle_mute(&mut self) {
        self.set_muted(!self.muted);
    }

    /// Returns the WebTransport URL.
    pub fn webtransport_url(&self) -> String {
        self.url.webtransport_url()
    }

    /// Returns the MoQ namespace.
    pub fn namespace(&self) -> &str {
        self.url.namespace()
    }

    /// Returns true if this URL is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        WebMoqUrl::is_moq_url(url)
    }

    /// Returns extended stats for diagnostics overlay.
    pub fn get_extended_stats(&self) -> Option<WebMoqStats> {
        let session_id = self.session_id?;
        let val = js_moq_get_extended_stats(session_id);
        if val.is_null() || val.is_undefined() {
            return None;
        }

        Some(WebMoqStats {
            connection_version: get_u32(&val, "connectionVersion"),
            video_decode_queue_size: get_u32(&val, "videoDecodeQueueSize"),
            audio_decode_queue_size: get_u32(&val, "audioDecodeQueueSize"),
            audio_context_state: get_str(&val, "audioContextState")
                .unwrap_or_else(|| "closed".to_string()),
            audio_stalled: js_sys::Reflect::get(&val, &"audioStalled".into())
                .ok()
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            audio_timestamp_ms: js_sys::Reflect::get(&val, &"audioTimestampMs".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            audio_frame_count: get_u32(&val, "audioFrameCount"),
            audio_underflow_samples: get_u32(&val, "audioUnderflowSamples"),
            audio_buffer_length: get_u32(&val, "audioBufferLength"),
            audio_buffer_capacity: get_u32(&val, "audioBufferCapacity"),
        })
    }
}

/// Extended stats for MoQ diagnostics overlay.
#[derive(Debug, Clone, Default)]
pub struct WebMoqStats {
    /// MoQ protocol version (e.g., 0xff070001)
    pub connection_version: u32,
    /// Number of encoded video chunks waiting to be decoded
    pub video_decode_queue_size: u32,
    /// Number of encoded audio chunks waiting to be decoded
    pub audio_decode_queue_size: u32,
    /// AudioContext state ("running", "suspended", "closed")
    pub audio_context_state: String,
    /// Whether the audio worklet ring buffer is stalled (buffering)
    pub audio_stalled: bool,
    /// Audio playback timestamp in ms
    pub audio_timestamp_ms: f64,
    /// Total decoded audio frames
    pub audio_frame_count: u32,
    /// Cumulative audio underflow samples
    pub audio_underflow_samples: u32,
    /// Current ring buffer fill level (samples)
    pub audio_buffer_length: u32,
    /// Ring buffer total capacity (samples)
    pub audio_buffer_capacity: u32,
}

impl Drop for WebMoqSession {
    fn drop(&mut self) {
        if let Some(id) = self.video_decoder_id.take() {
            js_moq_close_video(id);
        }
        if let Some(id) = self.audio_id.take() {
            js_moq_close_audio(id);
        }
        if let Some(id) = self.session_id.take() {
            js_moq_disconnect(id);
        }
    }
}

// ============================================================================
// JS interop helpers
// ============================================================================

fn get_str(obj: &JsValue, key: &str) -> Option<String> {
    js_sys::Reflect::get(obj, &key.into())
        .ok()
        .and_then(|v| v.as_string())
}

fn get_u32(obj: &JsValue, key: &str) -> u32 {
    js_sys::Reflect::get(obj, &key.into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as u32
}

fn get_f32(obj: &JsValue, key: &str) -> Option<f32> {
    js_sys::Reflect::get(obj, &key.into())
        .ok()
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
}

fn get_container(obj: &JsValue) -> (String, u32) {
    let container = js_sys::Reflect::get(obj, &"container".into()).ok();
    if let Some(ref c) = container {
        let kind = get_str(c, "kind").unwrap_or_else(|| "legacy".to_string());
        let timescale = get_u32(c, "timescale");
        (kind, timescale)
    } else {
        ("legacy".to_string(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moq_url_parse_with_auth_path() {
        let url = WebMoqUrl::parse("moq://localhost:4443/anon/bbb").unwrap();
        assert_eq!(url.host, "localhost");
        assert_eq!(url.port, 4443);
        assert!(!url.use_tls);
        assert_eq!(url.auth_path, Some("anon".to_string()));
        assert_eq!(url.namespace, "bbb");
        assert_eq!(url.webtransport_url(), "http://localhost:4443/anon");
    }

    #[test]
    fn test_moq_url_parse_no_auth_path() {
        let url = WebMoqUrl::parse("moqs://relay.example.com/live").unwrap();
        assert_eq!(url.host, "relay.example.com");
        assert_eq!(url.port, 443);
        assert!(url.use_tls);
        assert_eq!(url.auth_path, None);
        assert_eq!(url.namespace, "live");
        assert_eq!(url.webtransport_url(), "https://relay.example.com:443");
    }

    #[test]
    fn test_is_moq_url() {
        assert!(WebMoqUrl::is_moq_url("moq://localhost/test"));
        assert!(WebMoqUrl::is_moq_url("moqs://relay.example.com/live"));
        assert!(!WebMoqUrl::is_moq_url("https://example.com/video.mp4"));
    }
}
