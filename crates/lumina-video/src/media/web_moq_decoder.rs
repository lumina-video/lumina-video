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
#[derive(Debug, Clone)]
pub struct WebMoqUrl {
    /// Remote host
    host: String,
    /// Remote port
    port: u16,
    /// Whether to use TLS
    use_tls: bool,
    /// Namespace (first path component)
    namespace: String,
    /// Track name (reserved for track-specific subscriptions)
    #[allow(dead_code)]
    track: Option<String>,
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

        // Parse path
        let path = path.trim_matches('/');
        if path.is_empty() {
            return Err(VideoError::OpenFailed("Missing namespace".to_string()));
        }

        let (namespace, track) = match path.find('/') {
            Some(idx) => {
                let ns = &path[..idx];
                let track = &path[idx + 1..];
                (
                    ns.to_string(),
                    if track.is_empty() {
                        None
                    } else {
                        Some(track.to_string())
                    },
                )
            }
            None => (path.to_string(), None),
        };

        Ok(WebMoqUrl {
            host,
            port,
            use_tls,
            namespace,
            track,
            query,
            original,
        })
    }

    /// Returns the WebTransport URL for connection.
    pub fn webtransport_url(&self) -> String {
        let scheme = if self.use_tls { "https" } else { "http" };
        match &self.query {
            Some(q) => format!("{}://{}:{}?{}", scheme, self.host, self.port, q),
            None => format!("{}://{}:{}", scheme, self.host, self.port),
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

        // Get frame dimensions (web-sys uses snake_case for JavaScript camelCase)
        let width = frame.display_width();
        let height = frame.display_height();

        // Clone the frame for the copy operation (VideoFrame.clone() returns Result)
        let frame_clone = frame.clone().map_err(|e| {
            VideoError::DecodeFailed(format!("Failed to clone VideoFrame: {:?}", e))
        })?;

        // Use wgpu's copy_external_image_to_texture
        // Note: wgpu on web uses ExternalImageSource::VideoFrame which takes web_sys::VideoFrame
        queue.copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::VideoFrame(frame_clone),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
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

        // Close the original frame to release resources
        frame.close();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moq_url_parse() {
        let url = WebMoqUrl::parse("moqs://relay.example.com/live/stream").unwrap();
        assert_eq!(url.host, "relay.example.com");
        assert_eq!(url.port, 443);
        assert!(url.use_tls);
        assert_eq!(url.namespace, "live");
        assert_eq!(url.track, Some("stream".to_string()));
    }

    #[test]
    fn test_moq_url_no_tls() {
        let url = WebMoqUrl::parse("moq://localhost:4443/test/video").unwrap();
        assert_eq!(url.host, "localhost");
        assert_eq!(url.port, 4443);
        assert!(!url.use_tls);
        assert_eq!(url.webtransport_url(), "http://localhost:4443");
    }

    #[test]
    fn test_is_moq_url() {
        assert!(WebMoqUrl::is_moq_url("moq://localhost/test"));
        assert!(WebMoqUrl::is_moq_url("moqs://relay.example.com/live"));
        assert!(!WebMoqUrl::is_moq_url("https://example.com/video.mp4"));
    }
}
