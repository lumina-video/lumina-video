//! GStreamer-based video decoder for Linux.
//!
//! This module provides hardware-accelerated video decoding using GStreamer,
//! which handles codec edge cases (frame_num gaps, broken streams) robustly.
//!
//! GStreamer automatically selects the best decoder (VA-API, software fallback)
//! and handles all the complexity of H.264/VP8/VP9/AV1 decoding.
//!
//! Audio is played directly by GStreamer via autoaudiosink, with volume control
//! exposed through the GStreamer volume element.
//!
//! ## Zero-Copy DMABuf Support
//!
//! When the `zero-copy` feature is enabled, this decoder can expose DMABuf file
//! descriptors from VA-API decoded frames. This allows GPU-to-GPU transfers without
//! copying data through the CPU.
//!
//! ## DRM Modifier Support (GStreamer 1.24+)
//!
//! With GStreamer 1.24+, the `va` plugin exposes DRM modifiers in caps via the
//! `drm-format` field (e.g., `NV12:0x0100000000000002` for Intel X-tile).
//! This module parses the modifier to ensure correct Vulkan import of tiled buffers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

use crate::media::video::{DmaBufPlane, LinuxGpuSurface};

/// Shared audio state for GStreamer audio control.
/// This is used to control volume/mute from the UI thread.
#[derive(Clone)]
pub struct GstAudioHandle {
    inner: Arc<GstAudioHandleInner>,
}

struct GstAudioHandleInner {
    /// Volume element for control (None if no audio)
    volume_element: Option<gst::Element>,
    /// Whether audio is available
    has_audio: AtomicBool,
    /// Whether audio is muted
    muted: AtomicBool,
    /// Volume level (0.0 - 1.0)
    volume: std::sync::atomic::AtomicU32, // stored as volume * 100
}

impl GstAudioHandle {
    fn new(volume_element: Option<gst::Element>) -> Self {
        // Start with has_audio=false; set to true when audio pad connects
        Self {
            inner: Arc::new(GstAudioHandleInner {
                volume_element,
                has_audio: AtomicBool::new(false),
                muted: AtomicBool::new(false),
                volume: std::sync::atomic::AtomicU32::new(100), // 100%
            }),
        }
    }

    /// Called when an audio pad successfully connects.
    fn set_audio_connected(&self) {
        self.inner.has_audio.store(true, Ordering::Relaxed);
    }

    /// Returns whether audio is available.
    pub fn has_audio(&self) -> bool {
        self.inner.has_audio.load(Ordering::Relaxed)
    }

    /// Returns whether audio is muted.
    pub fn is_muted(&self) -> bool {
        self.inner.muted.load(Ordering::Relaxed)
    }

    /// Sets the mute state.
    pub fn set_muted(&self, muted: bool) {
        self.inner.muted.store(muted, Ordering::Relaxed);
        self.apply_volume();
    }

    /// Toggles mute state.
    pub fn toggle_mute(&self) {
        // Use fetch_xor for atomic toggle to avoid TOCTOU race condition
        self.inner.muted.fetch_xor(true, Ordering::Relaxed);
        self.apply_volume();
    }

    /// Returns the current volume (0-100).
    pub fn volume(&self) -> u32 {
        self.inner.volume.load(Ordering::Relaxed)
    }

    /// Sets the volume (0-100).
    pub fn set_volume(&self, volume: u32) {
        self.inner.volume.store(volume.min(100), Ordering::Relaxed);
        self.apply_volume();
    }

    /// Applies the current volume/mute state to the GStreamer element.
    fn apply_volume(&self) {
        if let Some(ref vol_elem) = self.inner.volume_element {
            let effective_volume = if self.inner.muted.load(Ordering::Relaxed) {
                0.0
            } else {
                self.inner.volume.load(Ordering::Relaxed) as f64 / 100.0
            };
            vol_elem.set_property("volume", effective_volume);
        }
    }
}

/// Buffering thresholds for hysteresis to prevent rapid pause/resume oscillation.
/// - Low threshold: pause only when buffer drops critically low
/// - High threshold: resume only when buffer is sufficiently full
///
/// The gap between thresholds prevents rapid state changes on marginal connections.
const BUFFER_LOW_THRESHOLD: i32 = 10; // Pause when buffer drops below this %
const BUFFER_HIGH_THRESHOLD: i32 = 100; // Resume when buffer reaches this %

/// GStreamer-based video decoder for Linux.
///
/// Uses a GStreamer pipeline:
/// - Video: `uridecodebin ! videoconvert ! video/x-raw,format=NV12 ! appsink`
/// - Audio: `uridecodebin ! audioconvert ! audioresample ! volume ! autoaudiosink`
///
/// This handles:
/// - HTTP/HTTPS streaming
/// - All common codecs (H.264, VP8, VP9, AV1)
/// - Hardware acceleration via VA-API (automatic)
/// - Edge cases that break other decoders
/// - Audio playback with volume control
pub struct GStreamerDecoder {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    metadata: VideoMetadata,
    position: Duration,
    eof: bool,
    /// True if we just seeked and are waiting for first frame
    seeking: bool,
    /// Target position of the last seek (for stale frame detection)
    seek_target: Option<Duration>,
    /// True if the last seek was backward (target < position at seek time)
    last_seek_backward: bool,
    /// Cached preroll sample for first decode_next() call
    preroll_sample: Option<gst::Sample>,
    /// Buffering percentage (0-100), 100 means fully buffered
    buffering_percent: i32,
    /// True once we've reached 100% buffering at least once (for rebuffer detection)
    was_fully_buffered: bool,
    /// True if the user explicitly paused (prevents buffering auto-resume)
    user_paused: bool,
    /// Queued error from bus messages during seek (returned on next decode_next)
    pending_error: Option<VideoError>,
    /// Audio control handle
    audio_handle: GstAudioHandle,
}

impl GStreamerDecoder {
    /// Creates a new GStreamer decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Initialize vendored runtime environment before GStreamer init
        #[cfg(feature = "vendored-runtime")]
        {
            let runtime = crate::vendored_runtime::VendoredRuntime::new();
            if !runtime.init() {
                tracing::warn!("vendored-runtime: vendor directory not found; falling back to system libraries");
            }
        }

        // Initialize GStreamer (safe to call multiple times)
        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {e}")))?;

        // Build the pipeline
        let pipeline = gst::Pipeline::new();

        // Source element - handles HTTP, HTTPS, file://
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", url)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create uridecodebin: {e}")))?;

        // === Video elements ===
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create videoconvert: {e}")))?;

        // App sink to pull video frames - constrained buffering for better seek behavior
        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Nv12)
                    .build(),
            )
            .max_buffers(1)
            .drop(true)
            .build();

        // === Audio elements ===
        let audioconvert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create audioconvert: {e}")))?;

        let audioresample = gst::ElementFactory::make("audioresample")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create audioresample: {e}")))?;

        let volume = gst::ElementFactory::make("volume")
            .property("volume", 1.0f64)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create volume: {e}")))?;

        let audiosink = gst::ElementFactory::make("autoaudiosink")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create autoaudiosink: {e}")))?;

        // Add all elements to pipeline
        pipeline
            .add_many([
                &source,
                &videoconvert,
                appsink.upcast_ref(),
                &audioconvert,
                &audioresample,
                &volume,
                &audiosink,
            ])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

        // Link video chain: videoconvert -> appsink
        videoconvert
            .link(&appsink)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link video elements: {e}")))?;

        // Link audio chain: audioconvert -> audioresample -> volume -> audiosink
        gst::Element::link_many([&audioconvert, &audioresample, &volume, &audiosink])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link audio elements: {e}")))?;

        // Create audio handle with volume element (has_audio starts false until pad connects)
        let audio_handle = GstAudioHandle::new(Some(volume));

        // Handle dynamic pad creation from uridecodebin
        let videoconvert_weak = videoconvert.downgrade();
        let audioconvert_weak = audioconvert.downgrade();
        let audio_handle_clone = audio_handle.clone();
        source.connect_pad_added(move |_src, src_pad| {
            let caps = src_pad
                .current_caps()
                .unwrap_or_else(|| src_pad.query_caps(None));
            let Some(structure) = caps.structure(0) else {
                return;
            };
            let name = structure.name();

            if name.starts_with("video/") {
                if let Some(videoconvert) = videoconvert_weak.upgrade() {
                    let Some(sink_pad) = videoconvert.static_pad("sink") else {
                        tracing::warn!("videoconvert element has no sink pad");
                        return;
                    };
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            tracing::warn!("Failed to link video pad: {:?}", e);
                        } else {
                            tracing::info!("Linked video pad: {}", name);
                        }
                    }
                }
            } else if name.starts_with("audio/") {
                if let Some(audioconvert) = audioconvert_weak.upgrade() {
                    let Some(sink_pad) = audioconvert.static_pad("sink") else {
                        tracing::warn!("audioconvert element has no sink pad");
                        return;
                    };
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            tracing::warn!("Failed to link audio pad: {:?}", e);
                        } else {
                            tracing::info!("Linked audio pad: {}", name);
                            audio_handle_clone.set_audio_connected();
                        }
                    }
                }
            }
        });

        // Set pipeline to Paused to get metadata without starting playback
        // (Playing state would autoplay the video)
        pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to start pipeline: {e:?}")))?;

        // Wait for pipeline to reach paused state (preroll) or error
        let Some(bus) = pipeline.bus() else {
            let _ = pipeline.set_state(gst::State::Null);
            return Err(VideoError::DecoderInit("Pipeline has no bus".to_string()));
        };
        let mut width = 0u32;
        let mut height = 0u32;
        let mut duration = None;

        // Track buffering during init (in case 100% is reached before decode loop starts)
        let mut init_buffering_percent = 0i32;

        // Wait for async state change and get metadata
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
            match msg.view() {
                gst::MessageView::AsyncDone(_) => {
                    // Query duration
                    if let Some(dur) = pipeline.query_duration::<gst::ClockTime>() {
                        duration = Some(Duration::from_nanos(dur.nseconds()));
                    }
                    break;
                }
                gst::MessageView::Error(err) => {
                    // Clean up pipeline before returning error
                    let _ = pipeline.set_state(gst::State::Null);
                    let _ = pipeline.state(gst::ClockTime::from_seconds(2));
                    return Err(VideoError::DecoderInit(format!(
                        "Pipeline error: {} ({:?})",
                        err.error(),
                        err.debug()
                    )));
                }
                gst::MessageView::StateChanged(state) => {
                    if state
                        .src()
                        .map(|s| s == pipeline.upcast_ref::<gst::Object>())
                        .unwrap_or(false)
                    {
                        tracing::debug!(
                            "Pipeline state: {:?} -> {:?}",
                            state.old(),
                            state.current()
                        );
                    }
                }
                gst::MessageView::Buffering(buffering) => {
                    // Track buffering during init - important for fast streams
                    // that reach 100% before decode loop starts
                    init_buffering_percent = buffering.percent();
                    tracing::debug!("Init buffering: {}%", init_buffering_percent);
                }
                _ => {}
            }
        }

        // Get video dimensions and frame rate from appsink caps
        let mut frame_rate = 30.0f32; // Default fallback
        if let Some(caps) = appsink.sink_pads().first().and_then(|p| p.current_caps()) {
            if let Some(s) = caps.structure(0) {
                width = s.get::<i32>("width").unwrap_or(0) as u32;
                height = s.get::<i32>("height").unwrap_or(0) as u32;
                // Extract frame rate from caps (stored as fraction)
                if let Ok(fps) = s.get::<gst::Fraction>("framerate") {
                    if fps.denom() != 0 {
                        frame_rate = fps.numer() as f32 / fps.denom() as f32;
                        tracing::debug!("Detected frame rate: {:.2} fps", frame_rate);
                    }
                }
            }
        }

        // Try to pull preroll sample - this gives us dimensions AND the first frame
        // We cache this sample to return on the first decode_next() call
        // Use generous timeout for slow network streams
        let preroll_sample = appsink.try_pull_preroll(gst::ClockTime::from_seconds(10));

        // If we couldn't get dimensions/framerate from caps, try from preroll sample
        if width == 0 || height == 0 || frame_rate == 30.0 {
            if let Some(ref sample) = preroll_sample {
                if let Some(caps) = sample.caps() {
                    if let Some(s) = caps.structure(0) {
                        if width == 0 {
                            width = s.get::<i32>("width").unwrap_or(0) as u32;
                        }
                        if height == 0 {
                            height = s.get::<i32>("height").unwrap_or(0) as u32;
                        }
                        // Try to get frame rate from preroll sample caps
                        if frame_rate == 30.0 {
                            if let Ok(fps) = s.get::<gst::Fraction>("framerate") {
                                if fps.denom() != 0 {
                                    frame_rate = fps.numer() as f32 / fps.denom() as f32;
                                    tracing::debug!(
                                        "Detected frame rate from preroll: {:.2} fps",
                                        frame_rate
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        if width == 0 || height == 0 {
            // Clean up pipeline before returning error
            let _ = pipeline.set_state(gst::State::Null);
            let _ = pipeline.state(gst::ClockTime::from_seconds(2));
            return Err(VideoError::DecoderInit(
                "Could not determine video dimensions".to_string(),
            ));
        }

        tracing::info!(
            "GStreamer decoder initialized: {}x{}, duration: {:?}, audio: {}",
            width,
            height,
            duration,
            audio_handle.has_audio()
        );

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate, // Extracted from caps, defaults to 30fps if not found
            codec: "unknown".to_string(), // GStreamer handles codec internally
            pixel_aspect_ratio: 1.0,
            start_time: None, // GStreamer handles sync internally
        };

        // For network streams, use buffering tracked during init (may have reached 100% already)
        // For local files, assume 100%
        let initial_buffering = if url.starts_with("http://") || url.starts_with("https://") {
            // Use the buffering percentage observed during init
            // This handles fast streams that buffer completely during preroll
            init_buffering_percent
        } else {
            100 // Local files are immediately available
        };

        Ok(Self {
            pipeline,
            appsink,
            metadata,
            position: Duration::ZERO,
            eof: false,
            seeking: false,
            seek_target: None,
            last_seek_backward: false,
            preroll_sample,
            buffering_percent: initial_buffering,
            was_fully_buffered: initial_buffering >= 100,
            user_paused: false,
            pending_error: None,
            audio_handle,
        })
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &GstAudioHandle {
        &self.audio_handle
    }

    /// Converts a GStreamer sample to our VideoFrame format.
    ///
    /// When the `zero-copy` feature is enabled, this will attempt to extract
    /// a DMABuf file descriptor from the sample first. If DMABuf is not available
    /// (e.g., software decoder, or unsupported allocator), it falls back to
    /// CPU memory copy.
    fn sample_to_frame(&self, sample: gst::Sample) -> Result<VideoFrame, VideoError> {
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

        let width = video_info.width();
        let height = video_info.height();

        // Try zero-copy DMABuf path first (always enabled on Linux)
        {
            if let Some(frame) =
                self.try_dmabuf_frame(buffer, &video_info, pts, width, height, sample.clone())?
            {
                return Ok(frame);
            }
            // Fall through to CPU path if DMABuf not available
        }

        // CPU copy path (fallback)
        self.sample_to_cpu_frame(buffer, &video_info, pts, width, height)
    }

    /// Parses the DRM modifier from GStreamer 1.24+ `drm-format` caps field.
    ///
    /// The `drm-format` field contains a string like `NV12:0x0100000000000002` where:
    /// - `NV12` is the DRM fourcc format
    /// - `0x0100000000000002` is the DRM modifier (e.g., Intel X-tile)
    ///
    /// Returns the modifier if found and parseable, or `None` if:
    /// - Caps don't have `drm-format` field (GStreamer < 1.24)
    /// - The format is LINEAR (no modifier suffix)
    /// - Parsing fails
    fn parse_drm_modifier_from_caps(sample: &gst::Sample) -> Option<u64> {
        let caps = sample.caps()?;
        let structure = caps.structure(0)?;

        // Try to get the drm-format field (GStreamer 1.24+ with va plugin)
        let drm_format: String = structure.get("drm-format").ok()?;

        // Parse format like "NV12:0x0100000000000002"
        // If no colon, it's just the format without modifier (assume LINEAR)
        let modifier_str = drm_format.split(':').nth(1)?;

        // Parse the hex modifier value
        let modifier = if modifier_str.starts_with("0x") || modifier_str.starts_with("0X") {
            u64::from_str_radix(&modifier_str[2..], 16).ok()?
        } else {
            modifier_str.parse::<u64>().ok()?
        };

        tracing::debug!(
            "Parsed DRM modifier from caps: drm-format='{}' -> modifier=0x{:016x}",
            drm_format,
            modifier
        );

        Some(modifier)
    }

    /// Extracts DMABuf file descriptors and per-plane metadata from a GStreamer buffer.
    ///
    /// Returns `Ok(Some(frame))` if DMABuf extraction succeeded,
    /// `Ok(None)` if the memory is not a DMABuf (fall back to CPU),
    /// `Err` if there was an error during extraction.
    ///
    /// # Multi-Plane Support (lumina-video-s0e)
    ///
    /// This function extracts per-plane metadata for multi-plane formats (NV12, YUV420p).
    /// GStreamer can provide planes in two configurations:
    /// 1. **Single FD with offsets**: All planes share one fd, distinguished by offset
    /// 2. **Multiple FDs**: Each plane has its own fd (offset is typically 0)
    ///
    /// Both cases are handled by checking `buffer.n_memory()` and extracting the
    /// appropriate fd/offset/stride for each plane from `video_info`.
    ///
    /// Multi-plane YUV formats are parsed and returned as [`DmaBufFrame`] with
    /// [`PixelFormat::Nv12`] or [`PixelFormat::Yuv420p`]. The zero-copy import path
    /// in [`zero_copy::linux`] handles these via `VkImageDrmFormatModifierExplicitCreateInfoEXT`.
    /// If zero-copy import fails (e.g., driver doesn't support the modifier), the
    /// CPU fallback path in [`DmaBufFrame`] is used automatically.
    fn try_dmabuf_frame(
        &self,
        buffer: &gst::BufferRef,
        video_info: &gst_video::VideoInfo,
        pts: Duration,
        width: u32,
        height: u32,
        sample: gst::Sample,
    ) -> Result<Option<VideoFrame>, VideoError> {
        use std::os::fd::RawFd;

        let format = video_info.format();
        let num_planes = video_info.n_planes() as usize;

        // Map GStreamer format to our PixelFormat
        let pixel_format = match format {
            gst_video::VideoFormat::Bgra | gst_video::VideoFormat::Bgrx => PixelFormat::Bgra,
            gst_video::VideoFormat::Rgba | gst_video::VideoFormat::Rgbx => PixelFormat::Rgba,
            gst_video::VideoFormat::Nv12 => PixelFormat::Nv12,
            gst_video::VideoFormat::I420 => PixelFormat::Yuv420p,
            _ => {
                tracing::debug!(
                    "Linux zero-copy: unsupported format {:?}, using CPU path",
                    format
                );
                return Ok(None);
            }
        };

        // Check if the first memory block is a DMABuf
        let Some(memory) = buffer.memory(0) else {
            return Ok(None);
        };

        // Check if this is DMABuf memory
        if !memory.is_memory_type::<gstreamer_allocators::DmaBufMemory>() {
            tracing::trace!("Buffer memory is not DMABuf, using CPU copy path");
            return Ok(None);
        }

        // Determine if we have multiple FDs (one per plane) or single FD with offsets
        let n_memory = buffer.n_memory();
        let multi_fd = n_memory >= num_planes && num_planes > 1;

        // Detect single-FD multi-plane layouts (common with VA-API).
        // These require special Vulkan import using VkImageDrmFormatModifierExplicitCreateInfoEXT
        // with a pPlaneLayouts array specifying each plane's offset and stride.
        let is_single_fd = num_planes > 1 && !multi_fd;

        // Extract per-plane metadata
        let mut planes: Vec<DmaBufPlane> = Vec::with_capacity(num_planes);

        // For single-FD layouts, dup the primary FD once and share it across all planes.
        // This avoids leaking N-1 FDs per frame (lumina-video-dvh).
        // The import code only uses primary_fd() for single-FD layouts.
        let mut primary_dup_fd: RawFd = -1;

        for plane_idx in 0..num_planes {
            // Get the memory block for this plane
            // If multi_fd: each plane has its own GstMemory
            // If single_fd: all planes share the first GstMemory, differentiated by offset
            let mem_idx = if multi_fd { plane_idx as u32 } else { 0 };
            let Some(plane_memory) = buffer.memory(mem_idx as usize) else {
                tracing::warn!(
                    "DMABuf plane {} has no memory block (expected at index {})",
                    plane_idx,
                    mem_idx
                );
                return Ok(None);
            };

            // Verify it's DMABuf memory
            if !plane_memory.is_memory_type::<gstreamer_allocators::DmaBufMemory>() {
                tracing::warn!("DMABuf plane {} memory is not DMABuf type", plane_idx);
                return Ok(None);
            }

            // Downcast to DmaBufMemory to access fd() method
            let dmabuf_memory = plane_memory
                .downcast_memory_ref::<gstreamer_allocators::DmaBufMemory>()
                .ok_or_else(|| {
                    VideoError::DecodeFailed("Failed to downcast to DmaBufMemory".to_string())
                })?;

            // Extract the file descriptor
            let gst_fd: RawFd = dmabuf_memory.fd();
            if gst_fd < 0 {
                tracing::warn!("DMABuf plane {} has invalid fd: {}", plane_idx, gst_fd);
                return Ok(None);
            }

            // SAFETY: dup() the FD so Vulkan gets its own copy to take ownership of.
            // This avoids double-close: GStreamer closes its FD when GstMemory drops,
            // Vulkan closes the dup'd FD when vkFreeMemory is called.
            //
            // For single-FD layouts: only dup once (first plane), reuse for others.
            // The import code only uses primary_fd(), so other planes just need
            // offset/stride metadata - their fd field is set to the shared dup'd fd
            // but won't be used directly.
            let fd: RawFd = if is_single_fd {
                if plane_idx == 0 {
                    // First plane: dup and save for reuse
                    let dup_fd = unsafe { libc::dup(gst_fd) };
                    if dup_fd < 0 {
                        tracing::warn!(
                            "Failed to dup DMABuf fd {} for plane {}: {}",
                            gst_fd,
                            plane_idx,
                            std::io::Error::last_os_error()
                        );
                        return Ok(None);
                    }
                    primary_dup_fd = dup_fd;
                    dup_fd
                } else {
                    // Subsequent planes in single-FD: reuse the already dup'd fd
                    // This fd value is stored but not used directly - only offset/stride matter
                    primary_dup_fd
                }
            } else {
                // Multi-FD: each plane gets its own dup'd fd
                let dup_fd = unsafe { libc::dup(gst_fd) };
                if dup_fd < 0 {
                    tracing::warn!(
                        "Failed to dup DMABuf fd {} for plane {}: {}",
                        gst_fd,
                        plane_idx,
                        std::io::Error::last_os_error()
                    );
                    // Close any already-dup'd fds before returning
                    for plane in &planes {
                        unsafe { libc::close(plane.fd) };
                    }
                    return Ok(None);
                }
                dup_fd
            };

            // Get stride and offset from VideoInfo (use .get() to avoid panic on malformed caps)
            // Helper to close FDs on error - for single-FD we only have one unique FD to close
            let close_fds_on_error =
                |planes: &[DmaBufPlane], current_fd: RawFd, is_single: bool| {
                    if is_single {
                        // Single-FD: all planes share the same fd, close once
                        if current_fd >= 0 {
                            unsafe { libc::close(current_fd) };
                        }
                    } else {
                        // Multi-FD: close all unique plane fds plus current
                        for plane in planes {
                            unsafe { libc::close(plane.fd) };
                        }
                        if current_fd >= 0 {
                            unsafe { libc::close(current_fd) };
                        }
                    }
                };

            let Some(&stride_i32) = video_info.stride().get(plane_idx) else {
                tracing::warn!(
                    "DMABuf plane {} missing stride entry in VideoInfo",
                    plane_idx
                );
                close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            };
            let Some(&offset_usize) = video_info.offset().get(plane_idx) else {
                tracing::warn!(
                    "DMABuf plane {} missing offset entry in VideoInfo",
                    plane_idx
                );
                close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            };
            if stride_i32 < 0 {
                tracing::warn!(
                    "DMABuf plane {} has negative stride {}",
                    plane_idx,
                    stride_i32
                );
                close_fds_on_error(&planes, fd, is_single_fd);
                return Ok(None);
            }
            let stride = stride_i32 as u32;
            let offset = offset_usize as u64;

            // Calculate plane size (approximate - may not account for padding)
            let plane_size = plane_memory.size() as u64;

            planes.push(DmaBufPlane {
                fd,
                offset,
                stride,
                size: plane_size,
            });

            tracing::debug!(
                "Extracted DMABuf plane {}: fd={}, offset={}, stride={}, size={}",
                plane_idx,
                fd,
                offset,
                stride,
                plane_size
            );
        }

        // Get DRM format modifier from caps (GStreamer 1.24+ with va plugin)
        // The va plugin exposes the actual modifier in drm-format caps field.
        // Example: "NV12:0x0100000000000002" = Intel X-tile
        // Fall back to LINEAR (0) if not available (older GStreamer or vaapi plugin)
        let modifier: u64 = Self::parse_drm_modifier_from_caps(&sample).unwrap_or_else(|| {
            tracing::debug!(
                "No DRM modifier in caps (GStreamer < 1.24 or legacy vaapi plugin), assuming LINEAR"
            );
            0 // DRM_FORMAT_MOD_LINEAR
        });

        tracing::debug!(
            "Extracted DMABuf with {} planes: {}x{} {:?}, modifier=0x{:x}, multi_fd={}",
            planes.len(),
            width,
            height,
            pixel_format,
            modifier,
            multi_fd
        );

        // Extract CPU fallback data in case zero-copy import fails at render time.
        // This ensures graceful degradation rather than dropping frames.
        let cpu_fallback = self.extract_cpu_fallback(buffer, video_info, width, height);

        // Create the LinuxGpuSurface
        // The sample is kept alive by wrapping it in an Arc.
        // The FDs passed here are dup'd copies - Vulkan takes ownership and will close them.
        let sample_owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(sample);

        if is_single_fd {
            tracing::debug!(
                "Single-FD multi-plane DMABuf detected: {} planes share fd={}",
                planes.len(),
                planes.first().map(|p| p.fd).unwrap_or(-1)
            );
        }

        // SAFETY: We've verified all fds are valid (they're dup'd copies we own)
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

        Ok(Some(VideoFrame::new(pts, DecodedFrame::Linux(surface))))
    }

    /// Converts a GStreamer buffer to a CPU frame (fallback path).
    fn sample_to_cpu_frame(
        &self,
        buffer: &gst::BufferRef,
        video_info: &gst_video::VideoInfo,
        pts: Duration,
        width: u32,
        height: u32,
    ) -> Result<VideoFrame, VideoError> {
        // Map the buffer for reading
        let map = buffer
            .map_readable()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to map buffer: {e}")))?;

        let data = map.as_slice();
        let format = video_info.format();

        // Determine pixel format and extract planes accordingly
        let (pixel_format, planes) =
            match format {
                gst_video::VideoFormat::Nv12 => {
                    // NV12: Y plane followed by interleaved UV plane (2 planes)
                    let strides = video_info.stride();
                    let offsets = video_info.offset();
                    let y_stride = *strides.first().ok_or_else(|| {
                        VideoError::DecodeFailed("NV12: missing Y stride".to_string())
                    })? as usize;
                    let uv_stride = *strides.get(1).ok_or_else(|| {
                        VideoError::DecodeFailed("NV12: missing UV stride".to_string())
                    })? as usize;
                    let y_offset = *offsets.first().ok_or_else(|| {
                        VideoError::DecodeFailed("NV12: missing Y offset".to_string())
                    })?;
                    let uv_offset = *offsets.get(1).ok_or_else(|| {
                        VideoError::DecodeFailed("NV12: missing UV offset".to_string())
                    })?;

                    let y_size = y_stride * height as usize;
                    let uv_size = uv_stride * (height as usize).div_ceil(2);

                    // Extract Y plane
                    let y_data = if y_offset + y_size <= data.len() {
                        data[y_offset..y_offset + y_size].to_vec()
                    } else {
                        return Err(VideoError::DecodeFailed(
                            "Y plane out of bounds".to_string(),
                        ));
                    };

                    // Extract UV plane
                    let uv_data = if uv_offset + uv_size <= data.len() {
                        data[uv_offset..uv_offset + uv_size].to_vec()
                    } else {
                        return Err(VideoError::DecodeFailed(
                            "UV plane out of bounds".to_string(),
                        ));
                    };

                    let y_plane = Plane {
                        data: y_data,
                        stride: y_stride,
                    };

                    let uv_plane = Plane {
                        data: uv_data,
                        stride: uv_stride,
                    };

                    (PixelFormat::Nv12, vec![y_plane, uv_plane])
                }
                gst_video::VideoFormat::I420 => {
                    // I420/YUV420p: Y, U, V as separate planes (3 planes)
                    let strides = video_info.stride();
                    let offsets = video_info.offset();
                    let y_stride = *strides.first().ok_or_else(|| {
                        VideoError::DecodeFailed("I420: missing Y stride".to_string())
                    })? as usize;
                    let u_stride = *strides.get(1).ok_or_else(|| {
                        VideoError::DecodeFailed("I420: missing U stride".to_string())
                    })? as usize;
                    let v_stride = *strides.get(2).ok_or_else(|| {
                        VideoError::DecodeFailed("I420: missing V stride".to_string())
                    })? as usize;
                    let y_offset = *offsets.first().ok_or_else(|| {
                        VideoError::DecodeFailed("I420: missing Y offset".to_string())
                    })?;
                    let u_offset = *offsets.get(1).ok_or_else(|| {
                        VideoError::DecodeFailed("I420: missing U offset".to_string())
                    })?;
                    let v_offset = *offsets.get(2).ok_or_else(|| {
                        VideoError::DecodeFailed("I420: missing V offset".to_string())
                    })?;

                    let y_size = y_stride * height as usize;
                    // U and V planes are quarter size (half width, half height)
                    let uv_height = (height as usize).div_ceil(2);
                    let u_size = u_stride * uv_height;
                    let v_size = v_stride * uv_height;

                    // Extract Y plane
                    let y_data = if y_offset + y_size <= data.len() {
                        data[y_offset..y_offset + y_size].to_vec()
                    } else {
                        return Err(VideoError::DecodeFailed(
                            "Y plane out of bounds".to_string(),
                        ));
                    };

                    // Extract U plane
                    let u_data = if u_offset + u_size <= data.len() {
                        data[u_offset..u_offset + u_size].to_vec()
                    } else {
                        return Err(VideoError::DecodeFailed(
                            "U plane out of bounds".to_string(),
                        ));
                    };

                    // Extract V plane
                    let v_data = if v_offset + v_size <= data.len() {
                        data[v_offset..v_offset + v_size].to_vec()
                    } else {
                        return Err(VideoError::DecodeFailed(
                            "V plane out of bounds".to_string(),
                        ));
                    };

                    let y_plane = Plane {
                        data: y_data,
                        stride: y_stride,
                    };

                    let u_plane = Plane {
                        data: u_data,
                        stride: u_stride,
                    };

                    let v_plane = Plane {
                        data: v_data,
                        stride: v_stride,
                    };

                    (PixelFormat::Yuv420p, vec![y_plane, u_plane, v_plane])
                }
                _ => {
                    return Err(VideoError::DecodeFailed(format!(
                        "Unsupported pixel format for CPU path: {format:?}"
                    )));
                }
            };

        let cpu_frame = CpuFrame::new(pixel_format, width, height, planes);

        Ok(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame)))
    }

    /// Extracts CPU frame data from a GStreamer buffer for zero-copy fallback.
    ///
    /// This is called during DMABuf frame extraction to provide fallback data
    /// in case zero-copy import fails at render time. Returns `None` if extraction
    /// fails (e.g., buffer mapping error), in which case the frame may be dropped.
    ///
    /// Supports both NV12 (2 planes: Y, UV interleaved) and I420/YUV420p (3 planes: Y, U, V).
    fn extract_cpu_fallback(
        &self,
        buffer: &gst::BufferRef,
        video_info: &gst_video::VideoInfo,
        width: u32,
        height: u32,
    ) -> Option<CpuFrame> {
        // Map the buffer for reading
        let map = match buffer.map_readable() {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!("Failed to map buffer for CPU fallback: {e}");
                return None;
            }
        };

        let data = map.as_slice();
        let format = video_info.format();

        // Determine pixel format and extract planes accordingly
        match format {
            gst_video::VideoFormat::Nv12 => {
                // NV12: Y plane followed by interleaved UV plane (2 planes)
                let strides = video_info.stride();
                let offsets = video_info.offset();
                let y_stride = (*strides.first()?) as usize;
                let uv_stride = (*strides.get(1)?) as usize;
                let y_offset = *offsets.first()?;
                let uv_offset = *offsets.get(1)?;

                let y_size = y_stride * height as usize;
                let uv_size = uv_stride * (height as usize).div_ceil(2);

                // Extract Y plane
                let y_data = if y_offset + y_size <= data.len() {
                    data[y_offset..y_offset + y_size].to_vec()
                } else {
                    tracing::debug!("Y plane out of bounds for CPU fallback");
                    return None;
                };

                // Extract UV plane
                let uv_data = if uv_offset + uv_size <= data.len() {
                    data[uv_offset..uv_offset + uv_size].to_vec()
                } else {
                    tracing::debug!("UV plane out of bounds for CPU fallback");
                    return None;
                };

                let y_plane = Plane {
                    data: y_data,
                    stride: y_stride,
                };

                let uv_plane = Plane {
                    data: uv_data,
                    stride: uv_stride,
                };

                Some(CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![y_plane, uv_plane],
                ))
            }
            gst_video::VideoFormat::I420 => {
                // I420/YUV420p: Y, U, V as separate planes (3 planes)
                let strides = video_info.stride();
                let offsets = video_info.offset();
                let y_stride = (*strides.first()?) as usize;
                let u_stride = (*strides.get(1)?) as usize;
                let v_stride = (*strides.get(2)?) as usize;
                let y_offset = *offsets.first()?;
                let u_offset = *offsets.get(1)?;
                let v_offset = *offsets.get(2)?;

                let y_size = y_stride * height as usize;
                // U and V planes are quarter size (half width, half height)
                let uv_height = (height as usize).div_ceil(2);
                let u_size = u_stride * uv_height;
                let v_size = v_stride * uv_height;

                // Extract Y plane
                let y_data = if y_offset + y_size <= data.len() {
                    data[y_offset..y_offset + y_size].to_vec()
                } else {
                    tracing::debug!("Y plane out of bounds for CPU fallback");
                    return None;
                };

                // Extract U plane
                let u_data = if u_offset + u_size <= data.len() {
                    data[u_offset..u_offset + u_size].to_vec()
                } else {
                    tracing::debug!("U plane out of bounds for CPU fallback");
                    return None;
                };

                // Extract V plane
                let v_data = if v_offset + v_size <= data.len() {
                    data[v_offset..v_offset + v_size].to_vec()
                } else {
                    tracing::debug!("V plane out of bounds for CPU fallback");
                    return None;
                };

                let y_plane = Plane {
                    data: y_data,
                    stride: y_stride,
                };

                let u_plane = Plane {
                    data: u_data,
                    stride: u_stride,
                };

                let v_plane = Plane {
                    data: v_data,
                    stride: v_stride,
                };

                Some(CpuFrame::new(
                    PixelFormat::Yuv420p,
                    width,
                    height,
                    vec![y_plane, u_plane, v_plane],
                ))
            }
            _ => {
                tracing::debug!("Unsupported pixel format for CPU fallback: {:?}", format);
                None
            }
        }
    }

    /// Internal seek implementation (may be retried on transient errors).
    fn seek_internal(&mut self, position: Duration) -> Result<(), VideoError> {
        let position_ns = position.as_nanos() as u64;

        // Mark that we're seeking - decode_next will skip bus polling
        self.seeking = true;
        self.seek_target = Some(position);
        // Record seek direction BEFORE updating position (for stale frame detection)
        self.last_seek_backward = position < self.position;

        // Choose seek flags based on direction:
        // - Forward: KEY_UNIT for fast keyframe-based seeking
        // - Backward: ACCURATE for reliable frame-accurate seeking
        //   (KEY_UNIT + SNAP_BEFORE caused video freeze, see notedeck-vid-w4r)
        let flags = if self.last_seek_backward {
            gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE
        } else {
            gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT
        };

        if let Err(e) = self
            .pipeline
            .seek_simple(flags, gst::ClockTime::from_nseconds(position_ns))
        {
            // Clear seeking state on error to avoid getting stuck
            self.seeking = false;
            self.seek_target = None;
            return Err(VideoError::SeekFailed(format!("Seek failed: {e:?}")));
        }

        // Wait for seek completion using filtered pop - only consume ASYNC_DONE or ERROR
        // This prevents swallowing other messages that decode_next needs
        // Use generous timeout for slow network streams that need to rebuffer
        if let Some(bus) = self.pipeline.bus() {
            let msg = bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(10),
                &[gst::MessageType::AsyncDone, gst::MessageType::Error],
            );
            match msg {
                Some(msg) => match msg.view() {
                    gst::MessageView::AsyncDone(_) => {
                        let direction = if position < self.position {
                            "backward"
                        } else {
                            "forward"
                        };
                        tracing::debug!(
                            "Seek {} completed: {:?} -> {:?}",
                            direction,
                            self.position,
                            position
                        );
                    }
                    gst::MessageView::Error(err) => {
                        self.seeking = false;
                        self.seek_target = None;
                        return Err(VideoError::SeekFailed(format!(
                            "Seek error: {} ({:?})",
                            err.error(),
                            err.debug()
                        )));
                    }
                    _ => {}
                },
                None => {
                    // Timeout waiting for seek completion
                    self.seeking = false;
                    self.seek_target = None;
                    return Err(VideoError::SeekFailed("Seek timed out".into()));
                }
            }
        }

        self.position = position;
        self.eof = false;
        // Assume rebuffering will be needed after seek (HTTP streams)
        self.buffering_percent = 0;
        // Reset so we don't pause during post-seek buffering
        self.was_fully_buffered = false;

        Ok(())
    }

    /// Processes a bus message during decode_next.
    /// Returns Some(result) if decode_next should return early, None to continue.
    fn process_bus_message(
        &mut self,
        msg: &gst::Message,
    ) -> Option<Result<Option<VideoFrame>, VideoError>> {
        match msg.view() {
            gst::MessageView::Error(err) => {
                let error = VideoError::DecodeFailed(format!("Pipeline error: {}", err.error()));
                if self.seeking {
                    // Queue error to return on next decode_next() call
                    // Don't silently drop real pipeline failures during seek
                    self.pending_error = Some(error);
                    return None;
                }
                return Some(Err(error));
            }
            gst::MessageView::Eos(_) if !self.seeking => {
                self.eof = true;
                return Some(Ok(None));
            }
            gst::MessageView::Buffering(buffering) => {
                self.handle_buffering_message(buffering.percent());
            }
            _ => {}
        }
        None
    }

    /// Handles buffering percentage changes with hysteresis.
    fn handle_buffering_message(&mut self, percent: i32) {
        if percent == self.buffering_percent {
            return;
        }

        tracing::debug!("Buffering: {}%", percent);
        self.buffering_percent = percent;

        // Resume when buffer is full, but only if user hasn't explicitly paused
        if percent >= BUFFER_HIGH_THRESHOLD {
            self.was_fully_buffered = true;
            if !self.user_paused {
                let _ = self.pipeline.set_state(gst::State::Playing);
            }
            return;
        }

        // Pause only on rebuffer (after we've been at 100% once) when critically low
        if self.was_fully_buffered && percent < BUFFER_LOW_THRESHOLD {
            tracing::info!("Buffer critically low ({}%), pausing to refill", percent);
            let _ = self.pipeline.set_state(gst::State::Paused);
        }
    }

    /// Checks if a frame should be discarded as stale during seeking.
    /// Returns true if the frame is stale and should be skipped.
    fn is_stale_frame(&self, frame_pts: Duration, discarded: u32, max_stale: u32) -> bool {
        if !self.seeking || discarded >= max_stale {
            return false;
        }

        let Some(target) = self.seek_target else {
            return false;
        };

        // For backward seeks: discard frames far AFTER the target
        let too_far_after = frame_pts > target + Duration::from_secs(2);

        // For forward seeks: discard frames BEFORE the target
        let too_far_before =
            !self.last_seek_backward && frame_pts + Duration::from_millis(100) < target;

        if too_far_after || too_far_before {
            tracing::debug!(
                "Discarding stale frame at {:?} (seek target {:?}, {})",
                frame_pts,
                target,
                if too_far_before { "before" } else { "after" }
            );
            return true;
        }

        false
    }

    /// Handles the None case when pulling a sample from appsink.
    fn handle_no_sample(&mut self) {
        if self.seeking {
            tracing::debug!(
                "No frame after seek: eos={}, position={:?}",
                self.appsink.is_eos(),
                self.position
            );
        }

        if self.appsink.is_eos() {
            self.eof = true;
            self.seeking = false;
            self.seek_target = None;
        }
    }
}

impl Drop for GStreamerDecoder {
    fn drop(&mut self) {
        // Fire and forget - don't block the UI thread at all
        // GStreamer handles cleanup asynchronously
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

// Safety: GStreamerDecoder can be sent between threads because:
// - gst::Pipeline, gst::Element, gst_app::AppSink, and gst::Sample all implement Send
//   in gstreamer-rs (GStreamer objects are reference-counted and thread-safe)
// - All other fields (Duration, bool, i32, etc.) are Send
// - GstAudioHandle uses Arc for thread-safe sharing
// The compiler should derive Send automatically, but we verify it with a static assert:
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<gst::Pipeline>();
    assert_send::<gst_app::AppSink>();
    assert_send::<gst::Sample>();
    assert_send::<GstAudioHandle>();
};

impl VideoDecoderBackend for GStreamerDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Return any queued error from seek (errors during seek are queued, not dropped)
        if let Some(error) = self.pending_error.take() {
            // Clear seek state so the decoder doesn't use stale flags on next call
            self.seeking = false;
            self.seek_target = None;
            return Err(error);
        }

        if self.eof {
            return Ok(None);
        }

        // Return cached preroll sample on first call (consumed during init for dimensions)
        if let Some(sample) = self.preroll_sample.take() {
            let frame = self.sample_to_frame(sample)?;
            tracing::debug!("Returning cached preroll frame at {:?}", frame.pts);
            self.position = frame.pts;
            return Ok(Some(frame));
        }

        // Poll bus for messages - errors during seek are queued (not dropped) and returned above
        // EOS during seek is skipped; seek() handles AsyncDone
        if let Some(bus) = self.pipeline.bus() {
            while let Some(msg) = bus.pop() {
                if let Some(result) = self.process_bus_message(&msg) {
                    return result;
                }
            }
        }

        // Use longer timeout when buffering or after seek
        let timeout_ms = if self.seeking || self.buffering_percent < 100 {
            1000
        } else {
            100
        };

        // When seeking, we may need to discard stale frames
        let max_stale_frames: u32 = if self.seeking { 5 } else { 0 };
        let mut discarded: u32 = 0;

        loop {
            let Some(sample) = self
                .appsink
                .try_pull_sample(gst::ClockTime::from_mseconds(timeout_ms))
            else {
                self.handle_no_sample();
                return Ok(None);
            };

            let frame = self.sample_to_frame(sample)?;

            // Check for stale frames after seek
            if self.is_stale_frame(frame.pts, discarded, max_stale_frames) {
                discarded += 1;
                continue;
            }

            if self.seeking {
                tracing::debug!(
                    "First frame after seek at {:?} (expected ~{:?})",
                    frame.pts,
                    self.position
                );
            }

            self.position = frame.pts;
            self.seeking = false;
            self.seek_target = None;
            return Ok(Some(frame));
        }
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Retry seek up to 3 times for transient HTTP errors
        const MAX_RETRIES: u32 = 3;
        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            match self.seek_internal(position) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        tracing::warn!("Seek attempt {} failed, retrying: {}", attempt + 1, e);
                        // Capture user pause state before toggling pipeline states
                        let was_paused = self.user_paused;
                        // Reset pipeline state before retry - helps recover from HTTP errors
                        let _ = self.pipeline.set_state(gst::State::Paused);
                        let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
                        let _ = self.pipeline.set_state(gst::State::Playing);
                        let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
                        // Restore paused state if user had paused before seek
                        if was_paused {
                            let _ = self.pipeline.set_state(gst::State::Paused);
                            let _ = self.pipeline.state(gst::ClockTime::from_mseconds(100));
                        }
                        // Longer delay for HTTP reconnection
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    last_error = Some(e);
                }
            }
        }

        // last_error is always Some after the loop (MAX_RETRIES > 0 ensures at least one iteration)
        Err(last_error
            .unwrap_or_else(|| VideoError::SeekFailed("Seek failed with no error".into())))
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn pause(&mut self) -> Result<(), VideoError> {
        self.user_paused = true;
        self.pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| VideoError::Generic(format!("Pause failed: {e:?}")))?;
        Ok(())
    }

    fn resume(&mut self) -> Result<(), VideoError> {
        self.user_paused = false;
        tracing::debug!("GStreamer: resuming pipeline to Playing state");
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| VideoError::Generic(format!("Resume failed: {e:?}")))?;
        Ok(())
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_handle.set_muted(muted);
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        // Convert 0.0-1.0 to 0-100
        self.audio_handle.set_volume((volume * 100.0) as u32);
        Ok(())
    }

    fn is_eof(&self) -> bool {
        self.eof
    }

    fn buffering_percent(&self) -> i32 {
        self.buffering_percent
    }

    /// GStreamer handles audio internally - no separate FFmpeg audio thread needed.
    fn handles_audio_internally(&self) -> bool {
        true
    }

    fn hw_accel_type(&self) -> HwAccelType {
        // GStreamer handles HW accel internally via uridecodebin auto-selection.
        // We can't know at runtime which decoder (VA-API, software, etc.) is in use.
        HwAccelType::None
    }
}
