// Linux GStreamer MoQ decoder — extracted from moq_decoder.rs

use std::sync::Arc;
use std::time::Duration;

use async_channel::{Receiver, Sender};
use tokio::runtime::Handle;

use super::{
    MoqDecoderConfig, MoqDecoderState, MoqSharedState, MoqVideoFrame,
};
use crate::media::moq::MoqUrl;
use crate::media::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

/// MoQ video decoder using GStreamer with appsrc for Linux zero-copy rendering.
///
/// This decoder:
/// 1. Receives NAL units from MoQ via the hang crate
/// 2. Pushes NAL units to GStreamer via appsrc element
/// 3. Decodes using VA-API hardware acceleration (vaH264Dec/vaH265Dec)
/// 4. Extracts DMABuf file descriptors from GstBuffer
/// 5. Imports DMABuf into Vulkan via wgpu for zero-copy rendering
///
/// # Pipeline
///
/// ```text
/// appsrc (stream-type=stream, format=time)
///    caps: video/x-h264,stream-format=byte-stream,alignment=au
///    → h264parse
///    → vaH264Dec OR avdec_h264 (fallback)
///    → video/x-raw(memory:DMABuf),format=NV12
///    → appsink (max-buffers=2, drop=true)
/// ```
pub struct MoqGStreamerDecoder {
    /// GStreamer pipeline
    pipeline: gstreamer::Pipeline,
    /// AppSrc element for pushing NAL units
    appsrc: gstreamer_app::AppSrc,
    /// AppSink element for pulled decoded frames
    appsink: gstreamer_app::AppSink,
    /// Parsed MoQ URL
    #[allow(dead_code)]
    url: MoqUrl,
    /// Configuration
    #[allow(dead_code)]
    config: MoqDecoderConfig,
    /// Shared state with async worker
    shared: Arc<MoqSharedState>,
    /// Receiver for encoded NAL units from async MoQ worker
    nal_rx: Receiver<MoqVideoFrame>,
    /// Active hardware acceleration type
    active_hw_type: HwAccelType,
    /// Owned tokio runtime (created if none exists)
    _owned_runtime: Option<tokio::runtime::Runtime>,
    /// Tokio runtime handle for async operations
    _runtime: Handle,
    /// Whether audio is muted
    audio_muted: bool,
    /// Audio volume (0.0 to 1.0)
    audio_volume: f32,
    /// Current playback position (from last decoded frame)
    position: Duration,
    /// Whether we've received the first keyframe (needed to start decoding)
    received_keyframe: bool,
    /// Codec detected from catalog (H264 or H265)
    #[allow(dead_code)]
    codec: MoqVideoCodec,
    /// Locally cached metadata (safe copy from shared state)
    cached_metadata: VideoMetadata,
}

/// Video codec type for MoQ streams on Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoqVideoCodec {
    H264,
    H265,
    Unknown,
}

impl MoqGStreamerDecoder {
    /// Creates a new MoQ GStreamer decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, MoqDecoderConfig::default())
    }

    /// Creates a new MoQ GStreamer decoder with explicit configuration.
    pub fn new_with_config(url: &str, mut config: MoqDecoderConfig) -> Result<Self, VideoError> {
        use gstreamer as gst;
        use gstreamer::prelude::*;
        use gstreamer_app as gst_app;

        // Initialize GStreamer (safe to call multiple times)
        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {e}")))?;

        // Parse the MoQ URL
        let moq_url = MoqUrl::parse(url).map_err(|e| VideoError::OpenFailed(e.to_string()))?;
        config.apply_localhost_tls_bypass(&moq_url);

        // Get existing runtime handle or create a new runtime
        let (owned_runtime, runtime) = match Handle::try_current() {
            Ok(handle) => (None, handle),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("moq-gst-runtime")
                    .build()
                    .map_err(|e| {
                        VideoError::OpenFailed(format!("Failed to create tokio runtime: {e}"))
                    })?;
                let handle = rt.handle().clone();
                (Some(rt), handle)
            }
        };

        // Create shared state
        let shared = Arc::new(MoqSharedState::new());

        // Create channel for NAL units (bounded to limit memory usage)
        let (nal_tx, nal_rx) = async_channel::bounded(60);

        // Start with H264 as default, will be updated from catalog
        let codec = MoqVideoCodec::H264;

        // Build the GStreamer pipeline
        let pipeline = gst::Pipeline::new();

        // AppSrc: Source element for pushing NAL units
        // stream-type=stream: Seekable stream (live, no duration)
        // format=time: We provide PTS timestamps
        // is-live=true: Real-time source, don't block
        let appsrc = gst_app::AppSrc::builder()
            .name("moq_appsrc")
            .stream_type(gst_app::AppStreamType::Stream)
            .format(gst::Format::Time)
            .is_live(true)
            .max_bytes(10 * 1024 * 1024) // 10MB buffer
            .build();

        // Set initial caps for H.264 byte-stream NAL units
        let h264_caps = gst::Caps::builder("video/x-h264")
            .field("stream-format", "byte-stream")
            .field("alignment", "au") // Access unit alignment (complete NALs)
            .build();
        appsrc.set_caps(Some(&h264_caps));

        // Parser element (h264parse)
        let parser = gst::ElementFactory::make("h264parse")
            .name("parser")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create h264parse: {e}")))?;

        // Decoder: Try VA-API first, fallback to software
        let decoder = Self::create_decoder(codec)?;

        // AppSink: Output element for pulling decoded frames
        // Request DMABuf memory type for zero-copy
        let appsink = gst_app::AppSink::builder()
            .name("moq_appsink")
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .features(["memory:DMABuf"])
                    .field("format", "NV12")
                    .build(),
            )
            .max_buffers(2)
            .drop(true) // Drop old frames when buffer is full (live streaming)
            .sync(false) // Don't sync to clock (we handle timing)
            .build();

        // Add elements to pipeline
        pipeline
            .add_many([appsrc.upcast_ref(), &parser, &decoder, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

        // Link elements: appsrc -> parser -> decoder -> appsink
        gst::Element::link_many([appsrc.upcast_ref(), &parser, &decoder, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link elements: {e}")))?;

        // Set pipeline to PLAYING state
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to start pipeline: {e:?}")))?;

        tracing::info!("MoQ GStreamer: Pipeline created with VA-API decoder");

        // Spawn the async MoQ worker
        let worker_shared = shared.clone();
        let worker_url = moq_url.clone();
        let worker_config = config.clone();

        runtime.spawn(async move {
            if let Err(e) =
                Self::run_moq_worker(worker_shared.clone(), worker_url, worker_config, nal_tx).await
            {
                worker_shared.set_error(format!("MoQ worker error: {e}"));
            }
        });

        let initial_volume = config.initial_volume;
        Ok(Self {
            pipeline,
            appsrc,
            appsink,
            url: moq_url,
            config,
            shared,
            nal_rx,
            active_hw_type: HwAccelType::Vaapi,
            _owned_runtime: owned_runtime,
            _runtime: runtime,
            audio_muted: false,
            audio_volume: initial_volume,
            position: Duration::ZERO,
            received_keyframe: false,
            codec,
            cached_metadata: VideoMetadata {
                codec: String::new(),
                width: 0,
                height: 0,
                frame_rate: 0.0,
                duration: None,
                pixel_aspect_ratio: 1.0,
                start_time: None,
            },
        })
    }

    /// Creates the decoder element, preferring VA-API hardware acceleration.
    fn create_decoder(codec: MoqVideoCodec) -> Result<gstreamer::Element, VideoError> {
        use gstreamer as gst;

        let (va_decoder, sw_decoder) = match codec {
            MoqVideoCodec::H264 => ("vaH264Dec", "avdec_h264"),
            MoqVideoCodec::H265 => ("vaH265Dec", "avdec_h265"),
            MoqVideoCodec::Unknown => ("vaH264Dec", "avdec_h264"),
        };

        // Try VA-API decoder first
        if let Ok(decoder) = gst::ElementFactory::make(va_decoder)
            .name("decoder")
            .build()
        {
            tracing::info!("MoQ GStreamer: Using VA-API decoder {}", va_decoder);
            return Ok(decoder);
        }

        // Fallback to software decoder
        tracing::warn!(
            "MoQ GStreamer: VA-API decoder {} not available, falling back to {}",
            va_decoder,
            sw_decoder
        );
        gst::ElementFactory::make(sw_decoder)
            .name("decoder")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create decoder: {e}")))
    }

    /// Async worker that handles MoQ connection, catalog fetching, and NAL unit receipt.
    async fn run_moq_worker(
        shared: Arc<MoqSharedState>,
        url: MoqUrl,
        config: MoqDecoderConfig,
        nal_tx: Sender<MoqVideoFrame>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        crate::media::moq::worker::run_moq_worker(shared, url, config, nal_tx, "GStreamer").await
    }

    /// Pushes a NAL unit to the GStreamer pipeline via appsrc.
    fn push_nal_unit(&mut self, moq_frame: &MoqVideoFrame) -> Result<(), VideoError> {
        use gstreamer as gst;

        // Skip non-keyframes until we receive a keyframe (decoder needs IDR to start)
        if !self.received_keyframe {
            if moq_frame.is_keyframe {
                self.received_keyframe = true;
                tracing::info!("MoQ GStreamer: Received first keyframe, starting decode");
            } else {
                tracing::debug!("MoQ GStreamer: Skipping non-keyframe (waiting for IDR)");
                return Ok(());
            }
        }

        // Create GstBuffer from NAL unit data
        let mut buffer = gst::Buffer::from_slice(moq_frame.data.clone());

        // Set PTS (presentation timestamp)
        {
            let buffer_ref = buffer.get_mut().ok_or_else(|| {
                VideoError::DecodeFailed("Failed to get mutable buffer reference".to_string())
            })?;
            buffer_ref.set_pts(gst::ClockTime::from_useconds(moq_frame.timestamp_us));

            // Mark non-keyframes with DELTA_UNIT (keyframes have no flag — absence of DELTA_UNIT implies sync point)
            if !moq_frame.is_keyframe {
                buffer_ref.set_flags(gst::BufferFlags::DELTA_UNIT);
            }
        }

        // Push buffer to appsrc
        match self.appsrc.push_buffer(buffer) {
            Ok(_) => Ok(()),
            Err(gst::FlowError::Flushing) => {
                tracing::debug!("MoQ GStreamer: Pipeline flushing, skipping buffer");
                Ok(())
            }
            Err(e) => Err(VideoError::DecodeFailed(format!(
                "Failed to push buffer to appsrc: {:?}",
                e
            ))),
        }
    }

    /// Pulls a decoded frame from appsink and converts it to VideoFrame.
    fn pull_decoded_frame(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        use gstreamer as gst;
        use gstreamer_video as gst_video;

        // Try to pull a sample (non-blocking)
        let sample = match self
            .appsink
            .try_pull_sample(gst::ClockTime::from_mseconds(0))
        {
            Some(s) => s,
            None => return Ok(None),
        };

        let buffer = sample
            .buffer()
            .ok_or_else(|| VideoError::DecodeFailed("Sample has no buffer".to_string()))?;

        let caps = sample
            .caps()
            .ok_or_else(|| VideoError::DecodeFailed("Sample has no caps".to_string()))?;

        let video_info = gst_video::VideoInfo::from_caps(caps)
            .map_err(|e| VideoError::DecodeFailed(format!("Invalid video caps: {e}")))?;

        let pts = buffer
            .pts()
            .map(|t| Duration::from_nanos(t.nseconds()))
            .unwrap_or(self.position);

        self.position = pts;

        let width = video_info.width();
        let height = video_info.height();

        // Try zero-copy DMABuf extraction
        if let Some(frame) =
            self.try_dmabuf_frame(buffer, &video_info, pts, width, height, &sample)?
        {
            return Ok(Some(frame));
        }

        // Fallback to CPU copy
        self.buffer_to_cpu_frame(buffer, &video_info, pts, width, height)
    }

    /// Attempts to extract DMABuf file descriptors for zero-copy rendering.
    fn try_dmabuf_frame(
        &self,
        buffer: &gstreamer::BufferRef,
        video_info: &gstreamer_video::VideoInfo,
        pts: Duration,
        width: u32,
        height: u32,
        sample: &gstreamer::Sample,
    ) -> Result<Option<VideoFrame>, VideoError> {
        use crate::media::video::{DmaBufPlane, LinuxGpuSurface};
        use gstreamer_allocators::DmaBufMemory;
        use std::os::fd::RawFd;

        let format = video_info.format();
        let num_planes = video_info.n_planes() as usize;

        // Map GStreamer format to our PixelFormat
        let pixel_format = match format {
            gstreamer_video::VideoFormat::Nv12 => PixelFormat::Nv12,
            gstreamer_video::VideoFormat::I420 => PixelFormat::Yuv420p,
            _ => {
                tracing::debug!(
                    "MoQ GStreamer: Unsupported format {:?} for zero-copy",
                    format
                );
                return Ok(None);
            }
        };

        // Check if memory is DMABuf
        let Some(memory) = buffer.memory(0) else {
            return Ok(None);
        };

        if !memory.is_memory_type::<DmaBufMemory>() {
            tracing::trace!("MoQ GStreamer: Buffer is not DMABuf, using CPU path");
            return Ok(None);
        }

        // Determine if single-FD or multi-FD layout
        let n_memory = buffer.n_memory();
        let multi_fd = n_memory >= num_planes && num_planes > 1;
        let is_single_fd = num_planes > 1 && !multi_fd;

        // Extract per-plane metadata
        let mut planes: Vec<DmaBufPlane> = Vec::with_capacity(num_planes);
        let mut primary_dup_fd: RawFd = -1;

        for plane_idx in 0..num_planes {
            let mem_idx = if multi_fd { plane_idx } else { 0 };
            let Some(plane_memory) = buffer.memory(mem_idx) else {
                tracing::warn!(
                    "MoQ GStreamer: Plane {} missing memory at index {}",
                    plane_idx,
                    mem_idx
                );
                return Ok(None);
            };

            if !plane_memory.is_memory_type::<DmaBufMemory>() {
                tracing::warn!("MoQ GStreamer: Plane {} is not DMABuf", plane_idx);
                return Ok(None);
            }

            let dmabuf_memory = plane_memory
                .downcast_memory_ref::<DmaBufMemory>()
                .ok_or_else(|| {
                    VideoError::DecodeFailed("Failed to downcast to DmaBufMemory".to_string())
                })?;

            let gst_fd: RawFd = dmabuf_memory.fd();
            if gst_fd < 0 {
                tracing::warn!("MoQ GStreamer: Plane {} has invalid fd", plane_idx);
                return Ok(None);
            }

            // Dup the FD (Vulkan takes ownership)
            let fd: RawFd = if is_single_fd {
                if plane_idx == 0 {
                    let dup_fd = unsafe { libc::dup(gst_fd) };
                    if dup_fd < 0 {
                        tracing::warn!("MoQ GStreamer: Failed to dup fd for plane {}", plane_idx);
                        return Ok(None);
                    }
                    primary_dup_fd = dup_fd;
                    dup_fd
                } else {
                    primary_dup_fd
                }
            } else {
                let dup_fd = unsafe { libc::dup(gst_fd) };
                if dup_fd < 0 {
                    // Clean up already-dup'd fds
                    for plane in &planes {
                        unsafe { libc::close(plane.fd) };
                    }
                    return Ok(None);
                }
                dup_fd
            };

            // Get stride and offset from VideoInfo
            let Some(&stride_i32) = video_info.stride().get(plane_idx) else {
                self.close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            };
            let Some(&offset_usize) = video_info.offset().get(plane_idx) else {
                self.close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            };

            if stride_i32 < 0 {
                self.close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            }

            planes.push(DmaBufPlane {
                fd,
                offset: offset_usize as u64,
                stride: stride_i32 as u32,
                size: plane_memory.size() as u64,
            });
        }

        // Parse DRM modifier from caps (GStreamer 1.24+)
        let modifier = Self::parse_drm_modifier_from_sample(sample).unwrap_or(0);

        // Extract CPU fallback data
        let cpu_fallback = self.extract_cpu_fallback(buffer, video_info, width, height);

        // Create LinuxGpuSurface
        // Keep sample alive via Arc
        let sample_owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(sample.clone());

        let surface = unsafe {
            LinuxGpuSurface::new(
                planes,
                width,
                height,
                pixel_format,
                modifier,
                is_single_fd,
                cpu_fallback,
                sample_owner,
            )
        };

        tracing::debug!(
            "MoQ GStreamer: Extracted DMABuf frame {}x{} {:?}",
            width,
            height,
            pixel_format
        );

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Linux(surface))))
    }

    /// Helper to close FDs on error during DMABuf extraction.
    fn close_fds_on_error(
        &self,
        planes: &[crate::media::video::DmaBufPlane],
        current_fd: std::os::fd::RawFd,
        is_single_fd: bool,
    ) {
        if is_single_fd {
            if current_fd >= 0 {
                unsafe { libc::close(current_fd) };
            }
        } else {
            for plane in planes {
                unsafe { libc::close(plane.fd) };
            }
            if current_fd >= 0 {
                unsafe { libc::close(current_fd) };
            }
        }
    }

    /// Parses DRM modifier from GStreamer caps (GStreamer 1.24+).
    fn parse_drm_modifier_from_sample(sample: &gstreamer::Sample) -> Option<u64> {
        let caps = sample.caps()?;
        let structure = caps.structure(0)?;

        let drm_format: String = structure.get("drm-format").ok()?;

        // Parse "NV12:0x0100000000000002" format
        let modifier_str = drm_format.split(':').nth(1)?;

        let modifier = if modifier_str.starts_with("0x") || modifier_str.starts_with("0X") {
            u64::from_str_radix(&modifier_str[2..], 16).ok()?
        } else {
            modifier_str.parse::<u64>().ok()?
        };

        tracing::debug!(
            "MoQ GStreamer: Parsed DRM modifier 0x{:016x} from '{}'",
            modifier,
            drm_format
        );

        Some(modifier)
    }

    /// Extracts CPU frame data for fallback when zero-copy fails.
    fn extract_cpu_fallback(
        &self,
        buffer: &gstreamer::BufferRef,
        video_info: &gstreamer_video::VideoInfo,
        width: u32,
        height: u32,
    ) -> Option<CpuFrame> {
        use gstreamer_video::VideoFormat;

        let map = buffer.map_readable().ok()?;
        let data = map.as_slice();
        let format = video_info.format();

        match format {
            VideoFormat::Nv12 => {
                let strides = video_info.stride();
                let offsets = video_info.offset();

                let y_stride = (*strides.first()?) as usize;
                let uv_stride = (*strides.get(1)?) as usize;
                let y_offset = *offsets.first()?;
                let uv_offset = *offsets.get(1)?;

                let y_size = y_stride * height as usize;
                let uv_size = uv_stride * (height as usize).div_ceil(2);

                if y_offset + y_size > data.len() || uv_offset + uv_size > data.len() {
                    return None;
                }

                let y_data = data[y_offset..y_offset + y_size].to_vec();
                let uv_data = data[uv_offset..uv_offset + uv_size].to_vec();

                Some(CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride: y_stride,
                        },
                        Plane {
                            data: uv_data,
                            stride: uv_stride,
                        },
                    ],
                ))
            }
            VideoFormat::I420 => {
                let strides = video_info.stride();
                let offsets = video_info.offset();

                let y_stride = (*strides.first()?) as usize;
                let u_stride = (*strides.get(1)?) as usize;
                let v_stride = (*strides.get(2)?) as usize;
                let y_offset = *offsets.first()?;
                let u_offset = *offsets.get(1)?;
                let v_offset = *offsets.get(2)?;

                let y_size = y_stride * height as usize;
                let uv_height = (height as usize).div_ceil(2);
                let u_size = u_stride * uv_height;
                let v_size = v_stride * uv_height;

                if y_offset + y_size > data.len()
                    || u_offset + u_size > data.len()
                    || v_offset + v_size > data.len()
                {
                    return None;
                }

                let y_data = data[y_offset..y_offset + y_size].to_vec();
                let u_data = data[u_offset..u_offset + u_size].to_vec();
                let v_data = data[v_offset..v_offset + v_size].to_vec();

                Some(CpuFrame::new(
                    PixelFormat::Yuv420p,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride: y_stride,
                        },
                        Plane {
                            data: u_data,
                            stride: u_stride,
                        },
                        Plane {
                            data: v_data,
                            stride: v_stride,
                        },
                    ],
                ))
            }
            _ => None,
        }
    }

    /// Converts a GStreamer buffer to a CPU frame (fallback path).
    fn buffer_to_cpu_frame(
        &self,
        buffer: &gstreamer::BufferRef,
        video_info: &gstreamer_video::VideoInfo,
        pts: Duration,
        width: u32,
        height: u32,
    ) -> Result<Option<VideoFrame>, VideoError> {
        let cpu_frame = self
            .extract_cpu_fallback(buffer, video_info, width, height)
            .ok_or_else(|| VideoError::DecodeFailed("Failed to extract CPU frame".to_string()))?;

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
    }

    /// Returns true if this URL is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        MoqUrl::is_moq_url(url)
    }

    /// Returns the current decoder state.
    pub fn decoder_state(&self) -> MoqDecoderState {
        *self.shared.state.lock()
    }

    /// Returns the error message if in error state.
    pub fn error_message(&self) -> Option<String> {
        self.shared.error_message.lock().clone()
    }
}

impl Drop for MoqGStreamerDecoder {
    fn drop(&mut self) {
        use gstreamer::prelude::*;

        tracing::debug!("MoQ: MoqGStreamerDecoder dropped (nal_rx closing)");

        // Signal EOS to appsrc
        let _ = self.appsrc.end_of_stream();

        // Stop pipeline
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}

impl VideoDecoderBackend for MoqGStreamerDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Sync cached metadata from shared state (safe copy under lock)
        {
            let shared_metadata = self.shared.metadata.lock();
            if shared_metadata.width != self.cached_metadata.width
                || shared_metadata.height != self.cached_metadata.height
            {
                self.cached_metadata = shared_metadata.clone();
            }
        }

        // Check for EOF
        if self.shared.eof_reached.load(std::sync::atomic::Ordering::Relaxed) {
            // Signal EOS to appsrc if not already done
            let _ = self.appsrc.end_of_stream();
            return Ok(None);
        }

        // Check for errors
        let state = *self.shared.state.lock();
        if state == MoqDecoderState::Error {
            return Err(VideoError::DecodeFailed(
                self.error_message()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        // Push any available NAL units to GStreamer
        while let Ok(moq_frame) = self.nal_rx.try_recv() {
            self.push_nal_unit(&moq_frame)?;
        }

        // Pull decoded frame from appsink
        self.pull_decoded_frame()
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        // Live streams don't support seeking
        Err(VideoError::SeekFailed(
            "Seeking is not supported on live MoQ streams".to_string(),
        ))
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.cached_metadata
    }

    fn duration(&self) -> Option<Duration> {
        // Live streams have no duration
        None
    }

    fn is_eof(&self) -> bool {
        self.shared.eof_reached.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn buffering_percent(&self) -> i32 {
        self.shared.buffering_percent.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.active_hw_type
    }

    fn handles_audio_internally(&self) -> bool {
        self.shared
            .audio
            .internal_audio_ready
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn audio_handle(&self) -> Option<crate::media::audio::AudioHandle> {
        self.shared.audio.moq_audio_handle.lock().clone()
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_muted = muted;
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        self.audio_volume = volume.clamp(0.0, 1.0);
        Ok(())
    }
}
