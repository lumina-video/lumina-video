//! FFmpeg-based video decoder implementation.
//!
//! This module provides video decoding using FFmpeg (ffmpeg-next) with support
//! for hardware acceleration on multiple platforms:
//!
//! - **macOS**: VideoToolbox (H.264, HEVC, VP9, AV1 on Apple Silicon)
//! - **Linux**: VAAPI (Intel/AMD GPUs)
//! - **Windows**: D3D11VA
//!
//! # Feature Flag
//!
//! The FFmpeg integration requires the `ffmpeg` feature to be enabled:
//!
//! ```toml
//! [dependencies]
//! notedeck = { version = "0.7", features = ["ffmpeg"] }
//! ```
//!
//! Without this feature, a placeholder implementation is used that generates
//! gray test frames.
//!
//! # System Requirements
//!
//! FFmpeg must be installed on the system:
//! - **macOS**: `brew install ffmpeg`
//! - **Linux**: `apt install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev`
//! - **Windows**: Download from https://ffmpeg.org/download.html

use std::time::Duration;

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// Configuration for hardware acceleration.
#[derive(Debug, Clone)]
pub struct HwAccelConfig {
    /// The type of hardware acceleration to use
    pub hw_type: HwAccelType,
    /// Whether to fall back to software if hardware fails
    pub fallback_to_software: bool,
    /// Preferred output pixel format (None = let FFmpeg decide)
    pub preferred_output_format: Option<PixelFormat>,
}

impl Default for HwAccelConfig {
    fn default() -> Self {
        Self {
            hw_type: HwAccelType::platform_default(),
            fallback_to_software: true,
            preferred_output_format: None,
        }
    }
}

impl HwAccelConfig {
    /// Creates a config for software-only decoding.
    pub fn software_only() -> Self {
        Self {
            hw_type: HwAccelType::None,
            fallback_to_software: false,
            preferred_output_format: None,
        }
    }

    /// Creates a config for the specified hardware acceleration type.
    pub fn with_hw_type(hw_type: HwAccelType) -> Self {
        Self {
            hw_type,
            fallback_to_software: true,
            preferred_output_format: None,
        }
    }
}

// ============================================================================
// Real FFmpeg implementation (when feature is enabled)
// ============================================================================

#[cfg(target_os = "macos")]
mod real_impl {
    use super::*;
    use ffmpeg_next as ffmpeg;
    use ffmpeg_next::ffi;
    use std::ptr;

    /// Wrapper for hardware device context buffer reference.
    struct HwDeviceCtx {
        ptr: *mut ffi::AVBufferRef,
    }

    impl HwDeviceCtx {
        /// Creates a new hardware device context for the specified type.
        fn new(hw_type: ffi::AVHWDeviceType) -> Option<Self> {
            let mut hw_device_ctx: *mut ffi::AVBufferRef = ptr::null_mut();

            let ret = unsafe {
                ffi::av_hwdevice_ctx_create(
                    &mut hw_device_ctx,
                    hw_type,
                    ptr::null(),     // device (NULL = default)
                    ptr::null_mut(), // opts
                    0,               // flags
                )
            };

            if ret < 0 || hw_device_ctx.is_null() {
                tracing::warn!(
                    "Failed to create hardware device context for type {:?}, error: {}",
                    hw_type,
                    ret
                );
                None
            } else {
                tracing::info!("Created hardware device context for type {:?}", hw_type);
                Some(Self { ptr: hw_device_ctx })
            }
        }

        fn as_ptr(&self) -> *mut ffi::AVBufferRef {
            self.ptr
        }
    }

    impl Drop for HwDeviceCtx {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                unsafe {
                    ffi::av_buffer_unref(&mut self.ptr);
                }
            }
        }
    }

    // SAFETY: HwDeviceCtx is an RAII wrapper with exclusive ownership over the AVBufferRef.
    // The raw pointer is only accessed from a single decode thread (enforced through the
    // VideoPlayer architecture), ensuring no concurrent access. The Send trait is safe
    // here because exclusive ownership + single-threaded access prevents data races.
    unsafe impl Send for HwDeviceCtx {}

    /// FFmpeg-based video decoder with hardware acceleration support.
    pub struct FfmpegDecoder {
        /// Input format context
        input: ffmpeg::format::context::Input,
        /// Video stream index
        video_stream_index: usize,
        /// Video decoder
        decoder: ffmpeg::decoder::Video,
        /// Video scaler for format conversion
        scaler: Option<ffmpeg::software::scaling::Context>,
        /// Video metadata
        metadata: VideoMetadata,
        /// Stream time base (numerator, denominator)
        time_base: (i32, i32),
        /// Hardware acceleration configuration
        hw_config: HwAccelConfig,
        /// Active hardware acceleration type
        active_hw_type: HwAccelType,
        /// Hardware device context (kept alive for decoder lifetime)
        #[allow(dead_code)]
        hw_device_ctx: Option<HwDeviceCtx>,
        /// Whether EOF has been reached
        eof_reached: bool,
        /// Packet iterator state
        packet_iter_finished: bool,
    }

    impl FfmpegDecoder {
        /// Creates a new FFmpeg decoder for the given URL or file path.
        pub fn new(url: &str) -> Result<Self, VideoError> {
            Self::new_with_config(url, HwAccelConfig::default())
        }

        /// Creates a new FFmpeg decoder with explicit hardware acceleration configuration.
        pub fn new_with_config(url: &str, hw_config: HwAccelConfig) -> Result<Self, VideoError> {
            // ffmpeg::init() is safe to call multiple times (just registers codecs/formats)
            ffmpeg::init()
                .map_err(|e| VideoError::DecoderInit(format!("FFmpeg init failed: {e}")))?;

            // Open input file/stream
            let input = ffmpeg::format::input(&url)
                .map_err(|e| VideoError::OpenFailed(format!("Failed to open {url}: {e}")))?;

            // Find video stream
            let video_stream = input
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| VideoError::OpenFailed("No video stream found".to_string()))?;

            let video_stream_index = video_stream.index();
            let time_base = video_stream.time_base();

            // Get codec parameters
            let codec_params = video_stream.parameters();

            // Create decoder context from parameters
            let mut context = ffmpeg::codec::context::Context::from_parameters(codec_params)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to create codec context: {e}"))
                })?;

            // Try to initialize hardware acceleration
            let (hw_device_ctx, active_hw_type) = Self::try_init_hw_accel(&hw_config, &mut context);

            tracing::info!(
                "FfmpegDecoder: Opening {} with HW accel: {:?} (requested: {:?})",
                url,
                active_hw_type,
                hw_config.hw_type
            );

            // Open decoder
            let decoder = context
                .decoder()
                .video()
                .map_err(|e| VideoError::DecoderInit(format!("Failed to open decoder: {e}")))?;

            // Extract metadata
            let duration = if input.duration() > 0 {
                Some(Duration::from_micros(
                    (input.duration() as f64 * 1_000_000.0 / ffi::AV_TIME_BASE as f64) as u64,
                ))
            } else {
                None
            };

            let frame_rate = video_stream.avg_frame_rate().0 as f64
                / video_stream.avg_frame_rate().1.max(1) as f64;

            // Extract stream start time (convert from stream time_base to Duration)
            let start_time = {
                let st = video_stream.start_time();
                if st >= 0 && time_base.1 > 0 {
                    // start_time is in stream time_base units
                    let us = st as i128 * time_base.0 as i128 * 1_000_000 / time_base.1 as i128;
                    Some(Duration::from_micros(us.max(0) as u64))
                } else {
                    None
                }
            };

            let metadata = VideoMetadata {
                width: decoder.width(),
                height: decoder.height(),
                duration,
                frame_rate: if frame_rate.is_finite() && frame_rate > 0.0 {
                    frame_rate as f32
                } else {
                    30.0
                },
                codec: decoder
                    .codec()
                    .map(|c| c.name().to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                pixel_aspect_ratio: (decoder.aspect_ratio().0 as f32)
                    / (decoder.aspect_ratio().1.max(1) as f32),
                start_time,
            };

            tracing::info!(
                "Video: {}x{}, duration: {:?}, fps: {:.2}, codec: {}, hw_accel: {:?}",
                metadata.width,
                metadata.height,
                metadata.duration,
                metadata.frame_rate,
                metadata.codec,
                active_hw_type
            );

            Ok(Self {
                input,
                video_stream_index,
                decoder,
                scaler: None,
                metadata,
                time_base: (time_base.0, time_base.1),
                hw_config,
                active_hw_type,
                hw_device_ctx,
                eof_reached: false,
                packet_iter_finished: false,
            })
        }

        fn try_init_hw_accel(
            config: &HwAccelConfig,
            context: &mut ffmpeg::codec::context::Context,
        ) -> (Option<HwDeviceCtx>, HwAccelType) {
            if config.hw_type == HwAccelType::None {
                return (None, HwAccelType::None);
            }

            // Map our HwAccelType to FFmpeg's AVHWDeviceType
            let (ffmpeg_hw_type, our_type) = match config.hw_type {
                HwAccelType::VideoToolbox => {
                    #[cfg(target_os = "macos")]
                    {
                        (
                            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                            HwAccelType::VideoToolbox,
                        )
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        tracing::warn!("VideoToolbox is only available on macOS");
                        return (None, HwAccelType::None);
                    }
                }
                HwAccelType::Vaapi => {
                    #[cfg(target_os = "linux")]
                    {
                        (
                            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                            HwAccelType::Vaapi,
                        )
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        tracing::warn!("VAAPI is only available on Linux");
                        return (None, HwAccelType::None);
                    }
                }
                HwAccelType::Vdpau => {
                    #[cfg(target_os = "linux")]
                    {
                        (
                            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VDPAU,
                            HwAccelType::Vdpau,
                        )
                    }
                    #[cfg(not(target_os = "linux"))]
                    {
                        tracing::warn!("VDPAU is only available on Linux");
                        return (None, HwAccelType::None);
                    }
                }
                HwAccelType::D3d11va => {
                    #[cfg(target_os = "windows")]
                    {
                        (
                            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
                            HwAccelType::D3d11va,
                        )
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        tracing::warn!("D3D11VA is only available on Windows");
                        return (None, HwAccelType::None);
                    }
                }
                HwAccelType::Dxva2 => {
                    #[cfg(target_os = "windows")]
                    {
                        (
                            ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_DXVA2,
                            HwAccelType::Dxva2,
                        )
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        tracing::warn!("DXVA2 is only available on Windows");
                        return (None, HwAccelType::None);
                    }
                }
                HwAccelType::MediaCodec => {
                    // MediaCodec is only available on Android, but this FFmpeg decoder
                    // is compiled with #[cfg(not(target_os = "android"))], so this case
                    // is unreachable. Android uses ExoPlayer instead of FFmpeg.
                    tracing::warn!("MediaCodec is only available on Android");
                    return (None, HwAccelType::None);
                }
                HwAccelType::None => return (None, HwAccelType::None),
            };

            // Create hardware device context
            if let Some(hw_ctx) = HwDeviceCtx::new(ffmpeg_hw_type) {
                // Set the hardware device context on the decoder
                unsafe {
                    let ctx_ptr = context.as_mut_ptr();
                    // Create a new reference to the hw device ctx for the decoder
                    (*ctx_ptr).hw_device_ctx = ffi::av_buffer_ref(hw_ctx.as_ptr());
                }
                tracing::info!(
                    "Hardware acceleration {:?} initialized successfully",
                    our_type
                );
                (Some(hw_ctx), our_type)
            } else if config.fallback_to_software {
                tracing::warn!(
                    "Hardware acceleration {:?} failed to initialize, falling back to software",
                    config.hw_type
                );
                (None, HwAccelType::None)
            } else {
                tracing::error!(
                    "Hardware acceleration {:?} failed and fallback disabled",
                    config.hw_type
                );
                (None, HwAccelType::None)
            }
        }

        /// Returns the hardware acceleration configuration.
        pub fn hw_config(&self) -> &HwAccelConfig {
            &self.hw_config
        }

        /// Returns true if hardware acceleration is currently active.
        pub fn is_hw_accel_active(&self) -> bool {
            self.active_hw_type != HwAccelType::None
        }

        fn pts_to_duration(&self, pts: i64) -> Duration {
            if pts < 0 || self.time_base.1 == 0 {
                return Duration::ZERO;
            }
            let seconds = (pts as f64) * (self.time_base.0 as f64) / (self.time_base.1 as f64);
            Duration::from_secs_f64(seconds.max(0.0))
        }

        fn ensure_scaler(
            &mut self,
            width: u32,
            height: u32,
            src_format: ffmpeg::format::Pixel,
        ) -> Result<(), VideoError> {
            let dst_format = ffmpeg::format::Pixel::RGBA;

            // Recreate scaler if format OR dimensions changed
            let needs_recreate = self.scaler.as_ref().is_none_or(|s| {
                let input = s.input();
                input.format != src_format || input.width != width || input.height != height
            });

            if needs_recreate {
                let scaler = ffmpeg::software::scaling::Context::get(
                    src_format,
                    width,
                    height,
                    dst_format,
                    width,
                    height,
                    ffmpeg::software::scaling::Flags::BILINEAR,
                )
                .map_err(|e| VideoError::DecodeFailed(format!("Failed to create scaler: {e}")))?;

                self.scaler = Some(scaler);
            }

            Ok(())
        }

        /// Transfer hardware frame to CPU memory if needed.
        fn transfer_hw_frame(
            &self,
            frame: &ffmpeg::frame::Video,
        ) -> Result<ffmpeg::frame::Video, VideoError> {
            // Check if this is a hardware frame by looking at the pixel format
            let is_hw_frame = unsafe {
                let frame_ptr = frame.as_ptr();
                let format = (*frame_ptr).format;
                // Hardware pixel formats
                format == ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX as i32
                    || format == ffi::AVPixelFormat::AV_PIX_FMT_VAAPI as i32
                    || format == ffi::AVPixelFormat::AV_PIX_FMT_D3D11 as i32
                    || format == ffi::AVPixelFormat::AV_PIX_FMT_DXVA2_VLD as i32
            };

            if !is_hw_frame {
                // Not a hardware frame, return as-is (clone)
                return Ok(frame.clone());
            }

            // Create a new frame for the transferred data
            let mut sw_frame = ffmpeg::frame::Video::empty();

            let ret = unsafe {
                let frame_ptr = frame.as_ptr();
                ffi::av_hwframe_transfer_data(sw_frame.as_mut_ptr(), frame_ptr, 0)
            };

            if ret < 0 {
                return Err(VideoError::DecodeFailed(format!(
                    "Failed to transfer hardware frame to CPU: {ret}"
                )));
            }

            // Copy timing info
            unsafe {
                let frame_ptr = frame.as_ptr();
                (*sw_frame.as_mut_ptr()).pts = (*frame_ptr).pts;
            }

            tracing::trace!("Transferred hardware frame to CPU");
            Ok(sw_frame)
        }

        fn frame_to_cpu_frame(
            &mut self,
            frame: &ffmpeg::frame::Video,
        ) -> Result<CpuFrame, VideoError> {
            // Transfer from GPU if needed
            let cpu_frame = self.transfer_hw_frame(frame)?;

            // Get the actual pixel format after transfer
            let src_format = cpu_frame.format();
            let width = cpu_frame.width();
            let height = cpu_frame.height();

            // Ensure we have a scaler to convert to RGBA
            self.ensure_scaler(width, height, src_format)?;

            let Some(scaler) = self.scaler.as_mut() else {
                return Err(VideoError::DecodeFailed(
                    "Scaler not initialized".to_string(),
                ));
            };

            // Create output frame
            let mut rgba_frame = ffmpeg::frame::Video::empty();
            scaler
                .run(&cpu_frame, &mut rgba_frame)
                .map_err(|e| VideoError::DecodeFailed(format!("Scaling failed: {e}")))?;

            // Extract RGBA data
            let out_width = rgba_frame.width();
            let out_height = rgba_frame.height();
            let stride = rgba_frame.stride(0);
            let data = rgba_frame.data(0);

            // Copy data (may need to handle stride != width * 4)
            let mut pixels = Vec::with_capacity((out_width * out_height * 4) as usize);
            for y in 0..out_height as usize {
                let row_start = y * stride;
                let row_end = row_start + (out_width as usize * 4);
                pixels.extend_from_slice(&data[row_start..row_end]);
            }

            Ok(CpuFrame::new(
                PixelFormat::Rgba,
                out_width,
                out_height,
                vec![Plane {
                    data: pixels,
                    stride: out_width as usize * 4,
                }],
            ))
        }
    }

    // SAFETY: FfmpegDecoder is only accessed from a single thread (the decode thread).
    // The raw pointers are not inherently thread-safe, but we ensure single-threaded
    // access through our VideoPlayer architecture.
    unsafe impl Send for FfmpegDecoder {}

    impl VideoDecoderBackend for FfmpegDecoder {
        fn open(url: &str) -> Result<Self, VideoError>
        where
            Self: Sized,
        {
            Self::new(url)
        }

        fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
            if self.eof_reached {
                return Ok(None);
            }

            let mut decoded_frame = ffmpeg::frame::Video::empty();

            loop {
                // Try to receive a frame from the decoder
                match self.decoder.receive_frame(&mut decoded_frame) {
                    Ok(()) => {
                        // Process decoded frame inline (helper was incorrectly in trait impl)
                        let pts = decoded_frame.pts().unwrap_or(0);
                        let duration = self.pts_to_duration(pts);
                        let cpu_frame = self.frame_to_cpu_frame(&decoded_frame)?;
                        return Ok(Some(VideoFrame::new(
                            duration,
                            DecodedFrame::Cpu(cpu_frame),
                        )));
                    }
                    Err(ffmpeg::Error::Eof) => {
                        self.eof_reached = true;
                        return Ok(None);
                    }
                    Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                        // Feed next packet inline (helper was incorrectly in trait impl)
                        if self.packet_iter_finished {
                            self.decoder.send_eof().ok();
                            self.packet_iter_finished = false;
                            continue;
                        }

                        let mut found_packet = false;
                        for (stream, packet) in self.input.packets() {
                            if stream.index() != self.video_stream_index {
                                continue;
                            }
                            self.decoder.send_packet(&packet).map_err(|e| {
                                VideoError::DecodeFailed(format!("Send packet failed: {e}"))
                            })?;
                            found_packet = true;
                            break;
                        }

                        if !found_packet {
                            self.packet_iter_finished = true;
                        }
                    }
                    Err(e) => return Err(VideoError::DecodeFailed(format!("Decode error: {e}"))),
                }
            }
        }

        fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
            // input.seek() expects timestamps in AV_TIME_BASE (microseconds), not stream time_base
            let timestamp = position.as_micros() as i64;

            tracing::debug!(
                "FFmpeg seek: position={:?}, timestamp={} (AV_TIME_BASE)",
                position,
                timestamp
            );

            // Use RangeFull (`..`) to allow FFmpeg to seek to the nearest keyframe.
            self.input
                .seek(timestamp, ..)
                .map_err(|e| VideoError::SeekFailed(format!("Seek failed: {e}")))?;

            // Flush decoder
            self.decoder.flush();
            self.eof_reached = false;
            self.packet_iter_finished = false;

            Ok(())
        }

        fn metadata(&self) -> &VideoMetadata {
            &self.metadata
        }

        fn hw_accel_type(&self) -> HwAccelType {
            self.active_hw_type
        }

        fn is_eof(&self) -> bool {
            self.eof_reached
        }
    }
}

// ============================================================================
// Placeholder implementation (when feature is disabled or on Android)
// ============================================================================

#[cfg(not(target_os = "macos"))]
mod placeholder_impl {
    use super::*;

    /// Placeholder FFmpeg decoder (no actual FFmpeg integration).
    ///
    /// This is used when:
    /// - The `ffmpeg` feature is not enabled
    /// - On Android (which uses ExoPlayer instead)
    ///
    /// It generates gray test frames for UI development and testing.
    pub struct FfmpegDecoder {
        metadata: VideoMetadata,
        /// Source URL (stored for debugging and future diagnostic use,
        /// though not actively used in this placeholder implementation)
        #[allow(dead_code)]
        url: String,
        hw_config: HwAccelConfig,
        active_hw_type: HwAccelType,
        current_pts: Duration,
        eof_reached: bool,
    }

    impl FfmpegDecoder {
        pub fn new(url: &str) -> Result<Self, VideoError> {
            Self::new_with_config(url, HwAccelConfig::default())
        }

        pub fn new_with_config(url: &str, hw_config: HwAccelConfig) -> Result<Self, VideoError> {
            let active_hw_type = if hw_config.fallback_to_software {
                HwAccelType::None
            } else if hw_config.hw_type != HwAccelType::None {
                return Err(VideoError::DecoderInit(
                    "FFmpeg feature not enabled. Enable with: features = [\"ffmpeg\"]".to_string(),
                ));
            } else {
                HwAccelType::None
            };

            tracing::warn!(
                "FfmpegDecoder: Using placeholder implementation for {}. \
                 Enable 'ffmpeg' feature for real decoding.",
                url
            );

            let metadata = VideoMetadata {
                width: 1920,
                height: 1080,
                duration: Some(Duration::from_secs(60)),
                frame_rate: 30.0,
                codec: "placeholder".to_string(),
                pixel_aspect_ratio: 1.0,
                start_time: None, // Placeholder doesn't have real stream info
            };

            Ok(Self {
                metadata,
                url: url.to_string(),
                hw_config,
                active_hw_type,
                current_pts: Duration::ZERO,
                eof_reached: false,
            })
        }

        pub fn hw_config(&self) -> &HwAccelConfig {
            &self.hw_config
        }

        pub fn is_hw_accel_active(&self) -> bool {
            self.active_hw_type != HwAccelType::None
        }

        fn generate_test_frame(&self) -> CpuFrame {
            let width = self.metadata.width;
            let height = self.metadata.height;

            // Generate a simple gradient pattern
            let mut pixels = Vec::with_capacity((width * height * 4) as usize);
            let frame_num = (self.current_pts.as_secs_f32() * 30.0) as u8;

            for y in 0..height {
                for x in 0..width {
                    let r = ((x as f32 / width as f32) * 255.0) as u8;
                    let g = ((y as f32 / height as f32) * 255.0) as u8;
                    let b = frame_num.wrapping_mul(3);
                    pixels.extend_from_slice(&[r, g, b, 255]);
                }
            }

            CpuFrame::new(
                PixelFormat::Rgba,
                width,
                height,
                vec![Plane {
                    data: pixels,
                    stride: width as usize * 4,
                }],
            )
        }
    }

    impl VideoDecoderBackend for FfmpegDecoder {
        fn open(url: &str) -> Result<Self, VideoError>
        where
            Self: Sized,
        {
            Self::new(url)
        }

        fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
            if self.eof_reached {
                return Ok(None);
            }

            let frame_duration = self.metadata.frame_duration();

            if let Some(duration) = self.metadata.duration {
                if self.current_pts >= duration {
                    self.eof_reached = true;
                    return Ok(None);
                }
            }

            let pts = self.current_pts;
            self.current_pts += frame_duration;

            let cpu_frame = self.generate_test_frame();

            Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
        }

        fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
            self.current_pts = position;
            self.eof_reached = false;
            Ok(())
        }

        fn metadata(&self) -> &VideoMetadata {
            &self.metadata
        }

        fn hw_accel_type(&self) -> HwAccelType {
            self.active_hw_type
        }

        fn is_eof(&self) -> bool {
            self.eof_reached
        }
    }
}

// Re-export the appropriate implementation
#[cfg(target_os = "macos")]
pub use real_impl::FfmpegDecoder;

#[cfg(not(target_os = "macos"))]
pub use placeholder_impl::FfmpegDecoder;

/// Builder for FfmpegDecoder with configuration options.
pub struct FfmpegDecoderBuilder {
    url: String,
    hw_config: HwAccelConfig,
}

impl FfmpegDecoderBuilder {
    /// Creates a new builder for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            hw_config: HwAccelConfig::default(),
        }
    }

    /// Sets the hardware acceleration type to use.
    pub fn with_hw_accel(mut self, hw_type: HwAccelType) -> Self {
        self.hw_config.hw_type = hw_type;
        self
    }

    /// Disables hardware acceleration (software only).
    pub fn software_only(mut self) -> Self {
        self.hw_config = HwAccelConfig::software_only();
        self
    }

    /// Sets whether to fall back to software decoding if hardware fails.
    pub fn with_fallback(mut self, fallback: bool) -> Self {
        self.hw_config.fallback_to_software = fallback;
        self
    }

    /// Sets the preferred output pixel format.
    pub fn with_output_format(mut self, format: PixelFormat) -> Self {
        self.hw_config.preferred_output_format = Some(format);
        self
    }

    /// Sets the complete hardware acceleration configuration.
    pub fn with_hw_config(mut self, config: HwAccelConfig) -> Self {
        self.hw_config = config;
        self
    }

    /// Builds the decoder with the configured options.
    pub fn build(self) -> Result<FfmpegDecoder, VideoError> {
        FfmpegDecoder::new_with_config(&self.url, self.hw_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let _decoder = FfmpegDecoder::new("test.mp4");
        // With placeholder, this always succeeds
        // With real FFmpeg, it would fail if file doesn't exist
        #[cfg(not(target_os = "macos"))]
        assert!(_decoder.is_ok());
    }

    #[test]
    fn test_decoder_with_software_only() {
        let _decoder = FfmpegDecoder::new_with_config("test.mp4", HwAccelConfig::software_only());
        #[cfg(not(target_os = "macos"))]
        {
            let Ok(decoder) = _decoder else {
                panic!("Expected decoder to be created");
            };
            assert_eq!(decoder.hw_accel_type(), HwAccelType::None);
            assert!(!decoder.is_hw_accel_active());
        }
    }

    #[test]
    fn test_decoder_builder() {
        let _decoder = FfmpegDecoderBuilder::new("test.mp4")
            .software_only()
            .build();
        #[cfg(not(target_os = "macos"))]
        {
            let Ok(decoder) = _decoder else {
                panic!("Expected decoder to be created");
            };
            assert_eq!(decoder.hw_accel_type(), HwAccelType::None);
        }
    }

    #[test]
    fn test_hw_accel_config_default() {
        let config = HwAccelConfig::default();
        assert!(config.fallback_to_software);
        assert!(config.preferred_output_format.is_none());
    }

    #[test]
    fn test_metadata() {
        let _decoder = FfmpegDecoder::new("test.mp4");
        #[cfg(not(target_os = "macos"))]
        {
            let Ok(decoder) = _decoder else {
                panic!("Expected decoder to be created");
            };
            let metadata = decoder.metadata();
            assert_eq!(metadata.width, 1920);
            assert_eq!(metadata.height, 1080);
        }
    }
}
