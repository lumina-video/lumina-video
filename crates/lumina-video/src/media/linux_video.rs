//! Zero-copy GPU video decoder for Linux using VA-API, DMABuf, and Vulkan.
//!
//! This module implements the optimal zero-copy path for video playback on Linux:
//!
//! ```text
//! VA-API Decoder -> DMABuf FD -> Vulkan Import -> wgpu Texture
//! ```
//!
//! # Zero-Copy Pipeline
//!
//! 1. GStreamer with `vapostproc` outputs frames as DMABuf file descriptors
//! 2. DMABuf FDs are imported directly into Vulkan via `VK_EXT_external_memory_dma_buf`
//! 3. Vulkan images are used by wgpu for rendering (no CPU copy)
//!
//! # Vulkan Specification References
//!
//! - [VK_EXT_external_memory_dma_buf](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html)
//! - [VK_EXT_image_drm_format_modifier](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_drm_format_modifier.html)
//!
//! See `docs/VULKAN-REFERENCE.md` for detailed extension usage and troubleshooting.
//!
//! # Fallback Policy
//!
//! Zero-copy is the only acceptable path for production use. Any fallback to CPU copy
//! is logged at WARN/ERROR level with diagnostic information (DRM modifier, driver, reason).
//! Fallbacks indicate a configuration or driver issue that requires engineering follow-up.
//!
//! # DRM Format Modifiers
//!
//! Modern GPUs use tiled memory layouts (Intel Y-tiled, AMD tiled, etc.). The DRM format
//! modifier encodes this layout and must be passed to Vulkan for correct import.

use std::os::fd::{AsRawFd, RawFd};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_allocators as gst_allocators;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::str::FromStr;

use crate::media::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

use crate::media::video::{DmaBufPlane, LinuxGpuSurface};

// =============================================================================
// DMABuf Information Types
// =============================================================================

/// Information about a DMABuf memory region.
///
/// This contains all the metadata needed to import a DMABuf into Vulkan:
/// - File descriptor for the DMABuf
/// - Stride (bytes per row) for each plane
/// - Offset within the DMABuf for each plane
/// - DRM format modifier (tiling layout)
#[derive(Debug, Clone)]
pub struct DmaBufInfo {
    /// File descriptor for the DMABuf (ownership may be transferred to Vulkan).
    pub fd: RawFd,
    /// Stride (bytes per row) for each plane.
    pub strides: Vec<u32>,
    /// Offset within the DMABuf for each plane.
    pub offsets: Vec<u32>,
    /// DRM format modifier (LINEAR, Y_TILED, etc.).
    /// Use `DRM_FORMAT_MOD_INVALID` (0xFFFFFFFFFFFFFFFF) if unknown.
    pub modifier: u64,
    /// Number of planes in this DMABuf.
    pub n_planes: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// DRM fourcc format code (e.g., DRM_FORMAT_NV12).
    pub drm_format: u32,
}

/// DRM format modifier for linear (non-tiled) layout.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// DRM format modifier indicating the modifier is invalid/unknown.
pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00FF_FFFF_FFFF_FFFF;

// =============================================================================
// Fallback Tracking
// =============================================================================

/// Metrics for tracking zero-copy fallback events.
///
/// When the zero-copy path fails and falls back to CPU copy, this struct
/// tracks the count and reason for regression monitoring.
#[derive(Debug, Default)]
pub struct ZeroCopyMetrics {
    /// Total number of frames processed via zero-copy path.
    pub zero_copy_frames: AtomicU64,
    /// Total number of frames that fell back to CPU copy.
    pub fallback_frames: AtomicU64,
    /// Whether we've already logged a warning about fallback (avoid spam).
    logged_fallback: AtomicBool,
}

impl ZeroCopyMetrics {
    /// Records a successful zero-copy frame and returns the new count.
    pub fn record_zero_copy(&self) -> u64 {
        self.zero_copy_frames.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Records a fallback to CPU copy with diagnostic information.
    #[allow(clippy::manual_is_multiple_of)] // Use modulo for stable Rust compatibility
    pub fn record_fallback(&self, reason: &str, modifier: u64, driver: &str) {
        self.fallback_frames.fetch_add(1, Ordering::Relaxed);

        // Log warning on first fallback, then every 100 frames
        let count = self.fallback_frames.load(Ordering::Relaxed);
        if !self.logged_fallback.swap(true, Ordering::Relaxed) || count % 100 == 0 {
            tracing::warn!(
                "Zero-copy fallback #{}: {} (modifier=0x{:016x}, driver={})",
                count,
                reason,
                modifier,
                driver
            );
        }
    }

    /// Returns the fallback rate as a percentage.
    pub fn fallback_rate(&self) -> f64 {
        let zero_copy = self.zero_copy_frames.load(Ordering::Relaxed) as f64;
        let fallback = self.fallback_frames.load(Ordering::Relaxed) as f64;
        let total = zero_copy + fallback;
        if total > 0.0 {
            (fallback / total) * 100.0
        } else {
            0.0
        }
    }

    /// Returns a snapshot of zero-copy metrics at this point in time.
    pub fn snapshot(&self) -> LinuxZeroCopyMetricsSnapshot {
        LinuxZeroCopyMetricsSnapshot {
            zero_copy_frames: self.zero_copy_frames.load(Ordering::Relaxed),
            fallback_frames: self.fallback_frames.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of Linux zero-copy metrics at a point in time.
///
/// This provides visibility into whether zero-copy DMABuf → Vulkan rendering
/// is working correctly, or if the system is falling back to CPU copy.
#[derive(Debug, Clone, Default)]
pub struct LinuxZeroCopyMetricsSnapshot {
    /// Total frames rendered via zero-copy DMABuf → Vulkan path.
    pub zero_copy_frames: u64,
    /// Total frames that fell back to CPU copy (VA-API unavailable or DMABuf import failed).
    pub fallback_frames: u64,
}

impl LinuxZeroCopyMetricsSnapshot {
    /// Returns the total number of frames processed.
    pub fn total_frames(&self) -> u64 {
        self.zero_copy_frames + self.fallback_frames
    }

    /// Returns the zero-copy success rate as a percentage (0-100).
    ///
    /// 100% means all frames used zero-copy, 0% means all used CPU fallback.
    pub fn zero_copy_percentage(&self) -> f64 {
        let total = self.total_frames() as f64;
        if total > 0.0 {
            (self.zero_copy_frames as f64 / total) * 100.0
        } else {
            0.0
        }
    }

    /// Returns the fallback rate as a percentage (0-100).
    ///
    /// This is the inverse of `zero_copy_percentage()`.
    /// Returns 0.0 when no frames have been processed.
    pub fn fallback_percentage(&self) -> f64 {
        let total = self.total_frames() as f64;
        if total > 0.0 {
            (self.fallback_frames as f64 / total) * 100.0
        } else {
            0.0
        }
    }

    /// Returns true if zero-copy is working (no fallbacks).
    pub fn is_zero_copy_active(&self) -> bool {
        self.zero_copy_frames > 0 && self.fallback_frames == 0
    }

    /// Returns a human-readable status string.
    pub fn status_string(&self) -> String {
        if self.total_frames() == 0 {
            "No frames rendered yet".to_string()
        } else if self.is_zero_copy_active() {
            format!(
                "✓ Zero-copy active ({} frames, 100%)",
                self.zero_copy_frames
            )
        } else {
            format!(
                "⚠ Partial fallback: {}/{} zero-copy ({:.1}%)",
                self.zero_copy_frames,
                self.total_frames(),
                self.zero_copy_percentage()
            )
        }
    }
}

// =============================================================================
// GStreamer Audio Handle
// =============================================================================

/// Shared audio state for GStreamer audio control.
/// This is used to control volume/mute from the UI thread.
#[derive(Clone)]
pub struct GstAudioHandle {
    inner: Arc<GstAudioHandleInner>,
}

/// Internal shared audio state for GStreamer audio control.
///
/// Contains the volume element and atomic state for thread-safe audio control.
struct GstAudioHandleInner {
    /// Volume element for control (None if no audio).
    volume_element: Option<gst::Element>,
    /// Audio sink element (for reset on seek to fix PulseAudio freeze bug).
    audio_sink: Option<gst::Element>,
    /// Whether audio is available.
    has_audio: AtomicBool,
    /// Whether audio is muted.
    muted: AtomicBool,
    /// Volume level (0.0 - 1.0), stored as volume * 100.
    volume: std::sync::atomic::AtomicU32,
}

impl GstAudioHandle {
    fn new(volume_element: Option<gst::Element>, audio_sink: Option<gst::Element>) -> Self {
        Self {
            inner: Arc::new(GstAudioHandleInner {
                volume_element,
                audio_sink,
                has_audio: AtomicBool::new(false),
                muted: AtomicBool::new(false),
                volume: std::sync::atomic::AtomicU32::new(100),
            }),
        }
    }

    /// Reset audio sink state after seek to fix PulseAudio/PipeWire freeze bug.
    ///
    /// NOTE: NULL → PLAYING approach breaks the pipeline (disconnects element from graph).
    /// This is currently a no-op. The workaround is to use alsasink (EGUI_VID_ALSA_AUDIO=1).
    ///
    /// TODO: Investigate alternative approaches:
    /// - Separate audio pipeline (isolates audio from video seek)
    /// - Manual audio scheduling with appsink + binary heap
    /// - pulsesink buffer-time/latency-time tuning
    pub fn reset_after_seek(&self) {
        // Disabled - NULL → PLAYING breaks the pipeline
        // The only reliable workaround is using alsasink instead of pulsesink
        if self.inner.audio_sink.is_some() {
            tracing::trace!(
                "Audio sink reset skipped (use EGUI_VID_ALSA_AUDIO=1 for reliable seek)"
            );
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
        let Some(ref vol_elem) = self.inner.volume_element else {
            return;
        };
        let effective_volume = if self.inner.muted.load(Ordering::Relaxed) {
            0.0
        } else {
            self.inner.volume.load(Ordering::Relaxed) as f64 / 100.0
        };
        vol_elem.set_property("volume", effective_volume);
    }
}

// =============================================================================
// Pipeline Mode
// =============================================================================

/// The current pipeline mode for frame output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    /// Zero-copy DMABuf mode (optimal).
    ZeroCopy,
    /// CPU copy fallback mode.
    CpuCopy,
}

// =============================================================================
// Zero-Copy GStreamer Decoder
// =============================================================================

/// Buffering threshold for resuming playback after buffering stall.
const BUFFER_HIGH_THRESHOLD: i32 = 100;

/// Zero-copy video decoder for Linux using VA-API and DMABuf.
///
/// This decoder attempts to use the optimal zero-copy path:
/// 1. VA-API hardware decoding via GStreamer
/// 2. DMABuf export from vapostproc
/// 3. Direct Vulkan import (no CPU copy)
///
/// If the zero-copy path is unavailable, it falls back to CPU copy with
/// visible warnings for engineering follow-up.
pub struct ZeroCopyGStreamerDecoder {
    /// GStreamer pipeline.
    pipeline: gst::Pipeline,
    /// Application sink for frame extraction.
    appsink: gst_app::AppSink,
    /// Cached video metadata.
    metadata: VideoMetadata,
    /// Current playback position.
    position: Duration,
    /// Whether end-of-stream has been reached.
    eof: bool,
    /// True if we just seeked and are waiting for first frame.
    seeking: bool,
    /// Target position of the last seek.
    seek_target: Option<Duration>,
    /// True if the last seek was backward.
    last_seek_backward: bool,
    /// Cached preroll sample for first decode_next() call.
    preroll_sample: Option<gst::Sample>,
    /// Buffering percentage (0-100).
    buffering_percent: i32,
    /// True once we've reached 100% buffering at least once.
    was_fully_buffered: bool,
    /// True if the user explicitly paused.
    user_paused: bool,
    /// Queued error from bus messages during seek.
    pending_error: Option<VideoError>,
    /// Audio control handle.
    audio_handle: GstAudioHandle,
    /// Current pipeline mode (zero-copy or fallback).
    mode: PipelineMode,
    /// Zero-copy metrics for fallback tracking.
    metrics: Arc<ZeroCopyMetrics>,
    /// Detected GPU driver name for diagnostics.
    driver_name: String,
    /// Wall-clock time when we resumed from buffering (for stuck detection).
    buffering_resume_time: Option<std::time::Instant>,
    /// Wall-clock time of last seek (for buffering grace period).
    last_seek_time: Option<std::time::Instant>,
    /// Count of consecutive no-sample events after buffering resume.
    no_sample_after_resume: u32,
    /// Count of recovery attempts since last successful sample (for backoff).
    recovery_attempts: u32,
    /// Count of consecutive samples received after recovery (for recovery completion).
    consecutive_samples_after_recovery: u32,
    /// Total samples received since last resume (for diagnostics).
    samples_since_resume: u32,
    /// True if playing a local file (not HTTP stream).
    /// Used to skip buffering waits since local files have use-buffering=false.
    is_local_file: bool,
}

impl ZeroCopyGStreamerDecoder {
    /// Creates a new zero-copy GStreamer decoder for the given URL.
    ///
    /// This attempts to set up the optimal pipeline using GStreamer 1.24+:
    /// ```text
    /// uridecodebin → vapostproc → video/x-raw(memory:DMABuf) → appsink
    /// ```
    ///
    /// The va decoder outputs VAMemory (VASurface), and vapostproc converts it
    /// to DMABuf with proper DRM modifiers (e.g., X-tile on Intel).
    ///
    /// If DMABuf output is not available, it falls back to:
    /// ```text
    /// uridecodebin → videoconvert → video/x-raw,format=NV12 → appsink
    /// ```
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Initialize vendored runtime environment before GStreamer init
        #[cfg(feature = "vendored-runtime")]
        {
            let runtime = crate::vendored_runtime::VendoredRuntime::new();
            if !runtime.init() {
                tracing::warn!("vendored-runtime: vendor directory not found; falling back to system libraries");
            }
        }

        gst::init().map_err(|e| VideoError::DecoderInit(format!("GStreamer init failed: {e}")))?;

        let is_local_file = url.starts_with("file://");

        // TEMPORARY: Force CPU-copy to test if seek/resume bug is DMABuf-specific
        let force_cpu_copy = std::env::var("EGUI_VID_FORCE_CPU_COPY").is_ok();
        if force_cpu_copy {
            tracing::warn!("EGUI_VID_FORCE_CPU_COPY set - using CPU copy pipeline for testing");
            return Self::cpu_copy_pipeline(url);
        }

        // TEMPORARY: Test direct DMABuf from decoder (skip vapostproc)
        let try_direct_dmabuf = std::env::var("EGUI_VID_DIRECT_DMABUF").is_ok();
        if try_direct_dmabuf {
            tracing::info!("EGUI_VID_DIRECT_DMABUF set - trying direct DMABuf from decoder");
            return Self::direct_dmabuf_pipeline(url);
        }

        // TEMPORARY: Use playbin3 with CPU copy for HTTP (for comparison testing only)
        let use_cpu_http = std::env::var("EGUI_VID_CPU_HTTP").is_ok();
        if use_cpu_http && !is_local_file {
            tracing::info!("EGUI_VID_CPU_HTTP set - using CPU copy for HTTP stream");
            return Self::playbin3_cpu_pipeline(url);
        }

        // Force zero-copy even if it would normally fall back (for testing)
        let force_zero_copy = std::env::var("EGUI_VID_FORCE_ZERO_COPY").is_ok();
        if force_zero_copy {
            tracing::info!("EGUI_VID_FORCE_ZERO_COPY set - forcing zero-copy pipeline");
            return Self::try_zero_copy_pipeline(url);
        }

        // Try zero-copy pipeline
        match Self::try_zero_copy_pipeline(url) {
            Ok(decoder) => {
                tracing::info!(
                    "Zero-copy GStreamer decoder initialized (DMABuf mode): {}x{}",
                    decoder.metadata.width,
                    decoder.metadata.height
                );
                Ok(decoder)
            }
            Err(zero_copy_err) => {
                tracing::warn!(
                    "Zero-copy pipeline failed, falling back to CPU copy: {}",
                    zero_copy_err
                );
                // Fall back to CPU copy pipeline
                Self::cpu_copy_pipeline(url).map_err(|cpu_err| {
                    VideoError::DecoderInit(format!(
                        "Both pipelines failed. Zero-copy: {zero_copy_err}. CPU copy: {cpu_err}"
                    ))
                })
            }
        }
    }

    /// Attempts to create a zero-copy DMABuf pipeline.
    ///
    /// The zero-copy pipeline is:
    /// uridecodebin -> vapostproc -> video/x-raw(memory:DMABuf) -> appsink
    ///
    /// Key insight: The va decoder outputs VAMemory (VASurface), and vapostproc
    /// converts it to DMABuf. Without vapostproc, we only get CPU-accessible NV12.
    ///
    /// On GStreamer 1.24+, vapostproc outputs format=DMA_DRM with drm-format field
    /// containing the actual format and DRM modifier (e.g., NV12:0x0100000000000002).
    fn try_zero_copy_pipeline(url: &str) -> Result<Self, VideoError> {
        let pipeline = gst::Pipeline::new();

        // Use uridecodebin which auto-detects container and codec
        // For local files, disable buffering (not needed, and our buffering handling has issues).
        // For HTTP streams, keep buffering enabled but we rely on the improved seek handling
        // to avoid the post-seek stall issue.
        let is_local_file = url.starts_with("file://");
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", url)
            .property("use-buffering", !is_local_file)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create uridecodebin: {e}")))?;

        tracing::info!(
            "uridecodebin: use-buffering={} (is_local_file={})",
            !is_local_file,
            is_local_file
        );

        // Add queue2 between uridecodebin and vapostproc to prevent buffer starvation
        // after seeks on HTTP streams. queue2 provides buffering that decouples the
        // network source from the decoder/sink timing.
        let video_queue = gst::ElementFactory::make("queue2")
            .name("video_queue")
            .property("max-size-buffers", 0u32) // Unlimited buffer count
            .property("max-size-bytes", 0u32) // Unlimited bytes
            .property("max-size-time", 3_000_000_000u64) // 3 seconds of data
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create video queue2: {e}")))?;

        // vapostproc converts VAMemory to DMABuf
        // This is essential for zero-copy: va decoder outputs VASurface,
        // vapostproc outputs DMABuf FD that we can import into Vulkan
        let vapostproc = gst::ElementFactory::make("vapostproc")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create vapostproc: {e}")))?;

        // DMABuf caps for appsink
        let dmabuf_caps = gst::Caps::from_str("video/x-raw(memory:DMABuf)")
            .map_err(|e| VideoError::DecoderInit(format!("Failed to parse DMABuf caps: {e}")))?;

        // Build appsink with DMABuf caps
        let appsink = gst_app::AppSink::builder()
            .caps(&dmabuf_caps)
            .max_buffers(2)
            .drop(true)
            .build();

        // Set up propose_allocation callback to advertise VideoMeta support.
        // This is required for vapostproc to output DMABuf - it refuses to negotiate
        // DMABuf caps without VideoMeta support in the downstream allocation query.
        let callbacks = gst_app::AppSinkCallbacks::builder()
            .propose_allocation(|_appsink, query| {
                // Add VideoMeta to the allocation query - this tells upstream (vapostproc)
                // that we support VideoMeta, which is mandatory for DMABuf output.
                // Without this, vapostproc fails with:
                // "DMABuf caps negotiated without the mandatory support of VideoMeta"
                query.add_allocation_meta::<gst_video::VideoMeta>(None);
                tracing::debug!("Added VideoMeta to allocation query for DMABuf support");
                true
            })
            .build();
        appsink.set_callbacks(callbacks);

        // Install event probe on appsink sink pad to diagnose seek issues
        Self::install_event_probe(&appsink, "zero-copy");

        // Check if audio should be disabled (for testing GitLab #3548 hypothesis)
        let disable_audio = std::env::var("EGUI_VID_NO_AUDIO").is_ok();
        if disable_audio {
            tracing::info!("EGUI_VID_NO_AUDIO set - audio disabled for seek testing");
        }

        if disable_audio {
            // Video-only pipeline
            pipeline
                .add_many([&source, &video_queue, &vapostproc, appsink.upcast_ref()])
                .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

            gst::Element::link_many([&video_queue, &vapostproc, appsink.upcast_ref()]).map_err(
                |e| VideoError::DecoderInit(format!("Failed to link video elements: {e}")),
            )?;

            let audio_handle = GstAudioHandle::new(None, None);

            // Connect pad-added - video only, ignore audio pads
            Self::connect_pad_added_video_only(&source, &video_queue);

            Self::init_pipeline(pipeline, appsink, audio_handle, url, PipelineMode::ZeroCopy)
        } else {
            // Full audio+video pipeline
            // Add audio queue to decouple audio/video sync (fixes seek freeze per GitLab #3548)
            let audio_queue = gst::ElementFactory::make("queue2")
                .name("audio_queue")
                .property("max-size-buffers", 0u32)
                .property("max-size-bytes", 0u32)
                .property("max-size-time", 2_000_000_000u64) // 2 seconds buffer
                .build()
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to create audio queue: {e}"))
                })?;

            let (audioconvert, audioresample, volume, audiosink) = Self::create_audio_elements()?;

            pipeline
                .add_many([
                    &source,
                    &video_queue,
                    &vapostproc,
                    appsink.upcast_ref(),
                    &audio_queue,
                    &audioconvert,
                    &audioresample,
                    &volume,
                    &audiosink,
                ])
                .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

            gst::Element::link_many([&video_queue, &vapostproc, appsink.upcast_ref()]).map_err(
                |e| VideoError::DecoderInit(format!("Failed to link video elements: {e}")),
            )?;

            // Audio path: queue -> audioconvert -> audioresample -> volume -> audiosink
            gst::Element::link_many([
                &audio_queue,
                &audioconvert,
                &audioresample,
                &volume,
                &audiosink,
            ])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link audio elements: {e}")))?;

            let audio_handle = GstAudioHandle::new(Some(volume), Some(audiosink));

            // Connect pad-added - video to video_queue, audio to audio_queue
            Self::connect_pad_added(&source, &video_queue, &audio_queue, audio_handle.clone());

            Self::init_pipeline(pipeline, appsink, audio_handle, url, PipelineMode::ZeroCopy)
        }
    }

    /// Creates a CPU copy fallback pipeline.
    fn cpu_copy_pipeline(url: &str) -> Result<Self, VideoError> {
        let pipeline = gst::Pipeline::new();

        let is_local_file = url.starts_with("file://");
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", url)
            .property("use-buffering", !is_local_file)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create uridecodebin: {e}")))?;

        // Add queue2 between uridecodebin and videoconvert to prevent buffer starvation
        let video_queue = gst::ElementFactory::make("queue2")
            .name("video_queue")
            .property("max-size-buffers", 0u32)
            .property("max-size-bytes", 0u32)
            .property("max-size-time", 3_000_000_000u64) // 3 seconds
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create video queue2: {e}")))?;

        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create videoconvert: {e}")))?;

        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Nv12)
                    .build(),
            )
            .max_buffers(2)
            .drop(true)
            .build();

        // Install event probe on appsink sink pad to diagnose seek issues
        Self::install_event_probe(&appsink, "cpu-copy");

        let (audioconvert, audioresample, volume, audiosink) = Self::create_audio_elements()?;

        pipeline
            .add_many([
                &source,
                &video_queue,
                &videoconvert,
                appsink.upcast_ref(),
                &audioconvert,
                &audioresample,
                &volume,
                &audiosink,
            ])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

        // Link queue2 -> videoconvert -> appsink
        gst::Element::link_many([&video_queue, &videoconvert, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link video elements: {e}")))?;

        gst::Element::link_many([&audioconvert, &audioresample, &volume, &audiosink])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link audio elements: {e}")))?;

        let audio_handle = GstAudioHandle::new(Some(volume), Some(audiosink));

        // Connect pad-added handler - video goes to queue2, audio goes to audioconvert
        Self::connect_pad_added(&source, &video_queue, &audioconvert, audio_handle.clone());

        Self::init_pipeline(pipeline, appsink, audio_handle, url, PipelineMode::CpuCopy)
    }

    /// Creates a playbin3-based pipeline with CPU copy for HTTP streams.
    /// This provides more robust seek handling than uridecodebin + vapostproc.
    fn playbin3_cpu_pipeline(url: &str) -> Result<Self, VideoError> {
        let playbin = gst::ElementFactory::make("playbin3")
            .property("uri", url)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create playbin3: {e}")))?;

        let video_sink_bin = gst::Bin::new();

        // Use videoconvert for CPU copy (more robust seeks than vapostproc)
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create videoconvert: {e}")))?;

        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Nv12)
                    .build(),
            )
            .max_buffers(2)
            .drop(true)
            .build();

        Self::install_event_probe(&appsink, "playbin3-cpu");

        video_sink_bin
            .add_many([&videoconvert, appsink.upcast_ref()])
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to add video elements to bin: {e}"))
            })?;

        gst::Element::link_many([&videoconvert, appsink.upcast_ref()])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link video elements: {e}")))?;

        let sink_pad = videoconvert.static_pad("sink").ok_or_else(|| {
            VideoError::DecoderInit("Failed to get videoconvert sink pad".to_string())
        })?;
        let ghost_pad = gst::GhostPad::with_target(&sink_pad)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create ghost pad: {e}")))?;
        video_sink_bin
            .add_pad(&ghost_pad)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add ghost pad to bin: {e}")))?;

        playbin.set_property("video-sink", &video_sink_bin);

        // Audio sink bin with volume control
        let audio_sink_bin = gst::Bin::new();
        let audioconvert = gst::ElementFactory::make("audioconvert")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create audioconvert: {e}")))?;
        let audioresample = gst::ElementFactory::make("audioresample")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create audioresample: {e}")))?;
        let volume = gst::ElementFactory::make("volume")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create volume: {e}")))?;
        let audiosink = gst::ElementFactory::make("autoaudiosink")
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create autoaudiosink: {e}")))?;

        audio_sink_bin
            .add_many([&audioconvert, &audioresample, &volume, &audiosink])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add audio elements: {e}")))?;

        gst::Element::link_many([&audioconvert, &audioresample, &volume, &audiosink])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link audio elements: {e}")))?;

        let audio_sink_pad = audioconvert.static_pad("sink").ok_or_else(|| {
            VideoError::DecoderInit("Failed to get audioconvert sink pad".to_string())
        })?;
        let audio_ghost_pad = gst::GhostPad::with_target(&audio_sink_pad).map_err(|e| {
            VideoError::DecoderInit(format!("Failed to create audio ghost pad: {e}"))
        })?;
        audio_sink_bin
            .add_pad(&audio_ghost_pad)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add audio ghost pad: {e}")))?;

        playbin.set_property("audio-sink", &audio_sink_bin);

        let audio_handle = GstAudioHandle::new(Some(volume), Some(audiosink));

        let pipeline: gst::Pipeline = playbin
            .downcast()
            .map_err(|_| VideoError::DecoderInit("playbin3 is not a Pipeline".to_string()))?;

        tracing::info!(
            "playbin3 CPU-copy pipeline created for HTTP stream: {}",
            url
        );

        Self::init_pipeline(pipeline, appsink, audio_handle, url, PipelineMode::CpuCopy)
    }

    /// Creates a pipeline that gets DMABuf directly from the decoder, skipping vapostproc.
    /// This tests whether vapostproc is the cause of seek issues on HTTP streams.
    fn direct_dmabuf_pipeline(url: &str) -> Result<Self, VideoError> {
        let pipeline = gst::Pipeline::new();

        let is_local_file = url.starts_with("file://");
        let source = gst::ElementFactory::make("uridecodebin")
            .property("uri", url)
            .property("use-buffering", !is_local_file)
            .build()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create uridecodebin: {e}")))?;

        // Request DMABuf directly from decoder - no vapostproc
        let dmabuf_caps = gst::Caps::from_str("video/x-raw(memory:DMABuf)")
            .map_err(|e| VideoError::DecoderInit(format!("Failed to parse DMABuf caps: {e}")))?;

        let appsink = gst_app::AppSink::builder()
            .caps(&dmabuf_caps)
            .max_buffers(2)
            .drop(true)
            .build();

        // VideoMeta support for DMABuf
        let callbacks = gst_app::AppSinkCallbacks::builder()
            .propose_allocation(|_appsink, query| {
                query.add_allocation_meta::<gst_video::VideoMeta>(None);
                tracing::debug!("Added VideoMeta to allocation query for direct DMABuf");
                true
            })
            .build();
        appsink.set_callbacks(callbacks);

        Self::install_event_probe(&appsink, "direct-dmabuf");

        // Audio elements
        let (audioconvert, audioresample, volume, audiosink) = Self::create_audio_elements()?;

        pipeline
            .add_many([
                &source,
                appsink.upcast_ref(),
                &audioconvert,
                &audioresample,
                &volume,
                &audiosink,
            ])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to add elements: {e}")))?;

        gst::Element::link_many([&audioconvert, &audioresample, &volume, &audiosink])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to link audio elements: {e}")))?;

        let audio_handle = GstAudioHandle::new(Some(volume), Some(audiosink));

        // Connect pad-added - video goes directly to appsink
        let appsink_weak = appsink.downgrade();
        let audioconvert_weak = audioconvert.downgrade();
        let audio_linked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let audio_linked_clone = audio_linked.clone();
        let audio_handle_clone = audio_handle.clone();

        source.connect_pad_added(move |_src, pad| {
            let caps = pad.current_caps().unwrap_or_else(|| pad.query_caps(None));
            let caps_str = caps.to_string();

            if caps_str.contains("video/") {
                if let Some(appsink) = appsink_weak.upgrade() {
                    if let Some(sink_pad) = appsink.static_pad("sink") {
                        if !sink_pad.is_linked() {
                            match pad.link(&sink_pad) {
                                Ok(_) => tracing::info!(
                                    "Linked video directly to appsink (no vapostproc)"
                                ),
                                Err(e) => {
                                    tracing::error!("Failed to link video to appsink: {:?}", e)
                                }
                            }
                        }
                    }
                }
            } else if caps_str.contains("audio/")
                && !audio_linked_clone.swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                if let Some(audioconvert) = audioconvert_weak.upgrade() {
                    if let Some(sink_pad) = audioconvert.static_pad("sink") {
                        if !sink_pad.is_linked() {
                            match pad.link(&sink_pad) {
                                Ok(_) => {
                                    tracing::info!("Linked audio pad");
                                    audio_handle_clone.set_audio_connected();
                                }
                                Err(e) => tracing::error!("Failed to link audio: {:?}", e),
                            }
                        }
                    }
                }
            }
        });

        tracing::info!(
            "Direct DMABuf pipeline (no vapostproc) created for: {}",
            url
        );

        Self::init_pipeline(pipeline, appsink, audio_handle, url, PipelineMode::ZeroCopy)
    }

    /// Installs an event probe on the appsink's sink pad to log critical events.
    /// This helps diagnose seek issues by logging SEGMENT, FLUSH, and EOS events.
    fn install_event_probe(appsink: &gst_app::AppSink, pipeline_name: &'static str) {
        let Some(sink_pad) = appsink.static_pad("sink") else {
            tracing::warn!("Could not get appsink sink pad for event probe");
            return;
        };

        sink_pad.add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
            if let Some(gst::PadProbeData::Event(ref event)) = info.data {
                match event.view() {
                    gst::EventView::Segment(seg) => {
                        let segment = seg.segment();
                        // Log segment details - this is critical for understanding data flow after seek
                        if let Some(time_seg) = segment.downcast_ref::<gst::format::Time>() {
                            tracing::info!(
                                "[{}] SEGMENT event: start={:?}, stop={:?}, position={:?}, rate={}",
                                pipeline_name,
                                time_seg.start(),
                                time_seg.stop(),
                                time_seg.position(),
                                time_seg.rate()
                            );
                        } else {
                            tracing::info!(
                                "[{}] SEGMENT event (non-time format): {:?}",
                                pipeline_name,
                                segment.format()
                            );
                        }
                    }
                    gst::EventView::FlushStart(_) => {
                        tracing::info!("[{}] FLUSH_START event received", pipeline_name);
                    }
                    gst::EventView::FlushStop(fs) => {
                        tracing::info!(
                            "[{}] FLUSH_STOP event: resets_time={}",
                            pipeline_name,
                            fs.resets_time()
                        );
                    }
                    gst::EventView::Eos(_) => {
                        tracing::info!("[{}] EOS event received", pipeline_name);
                    }
                    gst::EventView::StreamStart(_) => {
                        tracing::info!("[{}] STREAM_START event received", pipeline_name);
                    }
                    _ => {}
                }
            }
            gst::PadProbeReturn::Ok
        });

        tracing::debug!(
            "Installed event probe on appsink for {} pipeline",
            pipeline_name
        );
    }

    /// Creates audio processing elements.
    fn create_audio_elements(
    ) -> Result<(gst::Element, gst::Element, gst::Element, gst::Element), VideoError> {
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

        // Audio sink selection for Linux.
        //
        // DEFAULT: alsasink (ALSA direct)
        //
        // Why alsasink instead of pulsesink/autoaudiosink:
        // - pulsesink has a bug causing video freeze after 2-4 seeks on HTTP streams
        // - The freeze occurs because PulseAudio's internal buffer/clock state gets
        //   corrupted during flush-seek operations (GitLab #3548, MR !5344)
        // - alsasink bypasses PulseAudio and works reliably
        // - Audio sharing still works via ALSA's dmix plugin (multiple apps can play)
        //
        // Limitations of alsasink vs pulsesink:
        // - No per-app volume control in system tray
        // - No automatic audio device switching (Bluetooth, headphones)
        // - These are acceptable trade-offs for reliable video seeking
        //
        // Environment variables to override:
        // - EGUI_VID_PULSE_AUDIO: Force pulsesink (may freeze on HTTP seek)
        // - EGUI_VID_PIPEWIRE_AUDIO: Use pipewiresink (has glitches on backward seek)
        // - EGUI_VID_FAKE_AUDIO: Use fakesink (no audio output, for testing)
        let pulse_audio = std::env::var("EGUI_VID_PULSE_AUDIO").is_ok();
        let pipewire_audio = std::env::var("EGUI_VID_PIPEWIRE_AUDIO").is_ok();
        let fake_audio = std::env::var("EGUI_VID_FAKE_AUDIO").is_ok();

        let audiosink = if pulse_audio {
            // PulseAudio - has seek freeze bug on HTTP streams, use only if explicitly requested
            tracing::warn!(
                "EGUI_VID_PULSE_AUDIO: using pulsesink (may freeze after seek on HTTP streams)"
            );
            gst::ElementFactory::make("pulsesink")
                .build()
                .map_err(|e| VideoError::DecoderInit(format!("Failed to create pulsesink: {e}")))?
        } else if pipewire_audio {
            // PipeWire - has glitches on backward seek (GitLab #1245, #1980)
            tracing::info!(
                "EGUI_VID_PIPEWIRE_AUDIO: using pipewiresink (may glitch on backward seek)"
            );
            gst::ElementFactory::make("pipewiresink")
                .build()
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to create pipewiresink: {e}"))
                })?
        } else if fake_audio {
            tracing::info!("EGUI_VID_FAKE_AUDIO: using fakesink for audio (no output)");
            gst::ElementFactory::make("fakesink")
                .property("sync", false)
                .property("async", false)
                .build()
                .map_err(|e| VideoError::DecoderInit(format!("Failed to create fakesink: {e}")))?
        } else {
            // Default: alsasink - reliable seeking, audio sharing via dmix
            // Falls back to autoaudiosink if alsasink is unavailable (e.g., containerized environments)
            match gst::ElementFactory::make("alsasink").build() {
                Ok(sink) => {
                    tracing::info!("Using alsasink (reliable seek, audio sharing via dmix)");
                    sink
                }
                Err(e) => {
                    tracing::warn!(
                        "alsasink unavailable ({}), falling back to autoaudiosink (may freeze on HTTP seek)",
                        e
                    );
                    gst::ElementFactory::make("autoaudiosink")
                        .build()
                        .map_err(|e| {
                            VideoError::DecoderInit(format!("Failed to create autoaudiosink: {e}"))
                        })?
                }
            }
        };

        Ok((audioconvert, audioresample, volume, audiosink))
    }

    /// Connects the pad-added handler for uridecodebin dynamic pads.
    fn connect_pad_added(
        source: &gst::Element,
        video_sink: &gst::Element,
        audio_sink: &gst::Element,
        audio_handle: GstAudioHandle,
    ) {
        let video_sink_weak = video_sink.downgrade();
        let audio_sink_weak = audio_sink.downgrade();

        source.connect_pad_added(move |_src, src_pad| {
            let caps = src_pad
                .current_caps()
                .unwrap_or_else(|| src_pad.query_caps(None));

            // Log all pads for debugging
            tracing::debug!(
                "Pad added: name='{}', caps='{}'",
                src_pad.name(),
                caps.to_string()
            );

            let Some(structure) = caps.structure(0) else {
                tracing::warn!("Pad '{}' has no structure in caps", src_pad.name());
                return;
            };
            let name = structure.name();

            if name.starts_with("video/") {
                tracing::debug!("Detected video pad with caps name: {}", name);
                if let Some(video_sink) = video_sink_weak.upgrade() {
                    let Some(sink_pad) = video_sink.static_pad("sink") else {
                        tracing::warn!("Video sink element has no sink pad");
                        return;
                    };
                    if !sink_pad.is_linked() {
                        // Log the full caps for debugging
                        let full_caps = src_pad
                            .current_caps()
                            .unwrap_or_else(|| src_pad.query_caps(None));
                        tracing::debug!(
                            "Attempting to link video pad with caps: {}",
                            full_caps.to_string()
                        );

                        if let Err(e) = src_pad.link(&sink_pad) {
                            tracing::warn!(
                                "Failed to link video pad: {:?} (caps: {})",
                                e,
                                full_caps.to_string()
                            );
                        } else {
                            tracing::info!(
                                "Linked video pad: {} (caps: {})",
                                name,
                                full_caps.to_string()
                            );
                        }
                    } else {
                        tracing::debug!("Video sink pad already linked, skipping");
                    }
                } else {
                    tracing::warn!("Video sink element was dropped");
                }
            } else if name.starts_with("audio/") {
                if let Some(audio_sink) = audio_sink_weak.upgrade() {
                    let Some(sink_pad) = audio_sink.static_pad("sink") else {
                        tracing::warn!("Audio sink element has no sink pad");
                        return;
                    };
                    if !sink_pad.is_linked() {
                        if let Err(e) = src_pad.link(&sink_pad) {
                            tracing::warn!("Failed to link audio pad: {:?}", e);
                        } else {
                            tracing::info!("Linked audio pad: {}", name);
                            audio_handle.set_audio_connected();
                        }
                    }
                }
            }
        });
    }

    /// Connects only video pads, ignoring audio (for testing audio-related seek issues).
    fn connect_pad_added_video_only(source: &gst::Element, video_sink: &gst::Element) {
        let video_sink_weak = video_sink.downgrade();

        source.connect_pad_added(move |_src, src_pad| {
            let caps = src_pad
                .current_caps()
                .unwrap_or_else(|| src_pad.query_caps(None));

            let Some(structure) = caps.structure(0) else {
                return;
            };
            let name = structure.name();

            if name.starts_with("video/") {
                if let Some(video_sink) = video_sink_weak.upgrade() {
                    if let Some(sink_pad) = video_sink.static_pad("sink") {
                        if !sink_pad.is_linked() {
                            if let Err(e) = src_pad.link(&sink_pad) {
                                tracing::warn!("Failed to link video pad: {:?}", e);
                            } else {
                                tracing::info!("Linked video pad (audio disabled): {}", name);
                            }
                        }
                    }
                }
            } else if name.starts_with("audio/") {
                tracing::debug!("Ignoring audio pad (EGUI_VID_NO_AUDIO set): {}", name);
            }
        });
    }

    /// Initializes the pipeline and extracts metadata.
    fn init_pipeline(
        pipeline: gst::Pipeline,
        appsink: gst_app::AppSink,
        audio_handle: GstAudioHandle,
        url: &str,
        mode: PipelineMode,
    ) -> Result<Self, VideoError> {
        pipeline
            .set_state(gst::State::Paused)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to start pipeline: {e:?}")))?;

        let Some(bus) = pipeline.bus() else {
            let _ = pipeline.set_state(gst::State::Null);
            return Err(VideoError::DecoderInit("Pipeline has no bus".to_string()));
        };

        let mut width = 0u32;
        let mut height = 0u32;
        let mut duration = None;
        let mut init_buffering_percent = 0i32;

        // Wait for async state change
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
            match msg.view() {
                gst::MessageView::AsyncDone(_) => {
                    if let Some(dur) = pipeline.query_duration::<gst::ClockTime>() {
                        duration = Some(Duration::from_nanos(dur.nseconds()));
                    }
                    break;
                }
                gst::MessageView::Error(err) => {
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
                    init_buffering_percent = buffering.percent();
                    tracing::debug!("Init buffering: {}%", init_buffering_percent);
                }
                _ => {}
            }
        }

        // Log negotiated appsink caps once after PAUSED to confirm DMABuf path
        if let Some(caps) = appsink.sink_pads().first().and_then(|p| p.current_caps()) {
            let caps_str = caps.to_string();
            let dmabuf = caps_str.contains("memory:DMABuf");
            tracing::info!("Appsink negotiated caps: {} (dmabuf={})", caps_str, dmabuf);
        } else {
            tracing::warn!("Appsink has no negotiated caps after PAUSED");
        }

        // Extract dimensions from appsink caps
        let mut frame_rate = 30.0f32;
        if let Some(caps) = appsink.sink_pads().first().and_then(|p| p.current_caps()) {
            if let Some(s) = caps.structure(0) {
                width = s.get::<i32>("width").unwrap_or(0) as u32;
                height = s.get::<i32>("height").unwrap_or(0) as u32;
                if let Ok(fps) = s.get::<gst::Fraction>("framerate") {
                    if fps.denom() != 0 {
                        frame_rate = fps.numer() as f32 / fps.denom() as f32;
                    }
                }
            }
        }

        // Try preroll sample for dimensions
        let preroll_sample = appsink.try_pull_preroll(gst::ClockTime::from_seconds(10));
        if width == 0 || height == 0 {
            if let Some(ref sample) = preroll_sample {
                if let Some(caps) = sample.caps() {
                    if let Some(s) = caps.structure(0) {
                        if width == 0 {
                            width = s.get::<i32>("width").unwrap_or(0) as u32;
                        }
                        if height == 0 {
                            height = s.get::<i32>("height").unwrap_or(0) as u32;
                        }
                        if frame_rate == 30.0 {
                            if let Ok(fps) = s.get::<gst::Fraction>("framerate") {
                                if fps.denom() != 0 {
                                    frame_rate = fps.numer() as f32 / fps.denom() as f32;
                                }
                            }
                        }
                    }
                }
            }
        }

        if width == 0 || height == 0 {
            let _ = pipeline.set_state(gst::State::Null);
            let _ = pipeline.state(gst::ClockTime::from_seconds(2));
            return Err(VideoError::DecoderInit(
                "Could not determine video dimensions".to_string(),
            ));
        }

        // Extract codec info from pipeline elements or tags
        let codec = Self::extract_codec_name(&pipeline);

        let mode_str = match mode {
            PipelineMode::ZeroCopy => "zero-copy DMABuf",
            PipelineMode::CpuCopy => "CPU copy (fallback)",
        };
        tracing::info!(
            "GStreamer decoder initialized ({}): {}x{}, {:.2}fps, codec={}, duration: {:?}, audio: {}",
            mode_str,
            width,
            height,
            frame_rate,
            codec,
            duration,
            audio_handle.has_audio()
        );

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate,
            codec,
            pixel_aspect_ratio: 1.0,
            start_time: None, // VA-API doesn't expose stream start time
        };

        let is_local_file = url.starts_with("file://");
        let initial_buffering = if is_local_file {
            100
        } else {
            init_buffering_percent
        };

        let metrics = Arc::new(ZeroCopyMetrics::default());
        if mode == PipelineMode::CpuCopy {
            metrics.record_fallback(
                "Zero-copy pipeline creation failed",
                DRM_FORMAT_MOD_INVALID,
                "unknown",
            );
        }

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
            mode,
            metrics,
            driver_name: "unknown".to_string(),
            buffering_resume_time: None,
            no_sample_after_resume: 0,
            recovery_attempts: 0,
            consecutive_samples_after_recovery: 0,
            samples_since_resume: 0,
            last_seek_time: None,
            is_local_file,
        })
    }

    /// Returns the audio handle for volume/mute control.
    pub fn audio_handle(&self) -> &GstAudioHandle {
        &self.audio_handle
    }

    /// Returns the current pipeline mode.
    pub fn mode(&self) -> PipelineMode {
        self.mode
    }

    /// Returns the zero-copy metrics.
    pub fn metrics(&self) -> &Arc<ZeroCopyMetrics> {
        &self.metrics
    }

    /// Extracts codec name from pipeline elements or tags.
    ///
    /// Searches for video decoder elements and extracts codec info from:
    /// 1. Element factory name (e.g., "vah264dec" -> "H.264 (VA-API)")
    /// 2. Caps structure name (e.g., "video/x-h264")
    fn extract_codec_name(pipeline: &gst::Pipeline) -> String {
        // Try to find codec from element names
        let mut codec_name = String::new();

        // Iterate pipeline elements looking for decoder
        for element in pipeline.iterate_elements().into_iter().flatten() {
            let factory_name = element
                .factory()
                .map(|f| f.name().to_string())
                .unwrap_or_default();

            // Check for VA-API decoders
            if factory_name.contains("vah264") || factory_name.contains("vaapih264") {
                codec_name = "H.264 (VA-API)".to_string();
                break;
            } else if factory_name.contains("vah265") || factory_name.contains("vaapih265") {
                codec_name = "H.265/HEVC (VA-API)".to_string();
                break;
            } else if factory_name.contains("vavp9") || factory_name.contains("vaapivp9") {
                codec_name = "VP9 (VA-API)".to_string();
                break;
            } else if factory_name.contains("vavp8") || factory_name.contains("vaapivp8") {
                codec_name = "VP8 (VA-API)".to_string();
                break;
            } else if factory_name.contains("vaav1") || factory_name.contains("vaapiav1") {
                codec_name = "AV1 (VA-API)".to_string();
                break;
            } else if factory_name.contains("avdec_h264") {
                codec_name = "H.264 (FFmpeg)".to_string();
                break;
            } else if factory_name.contains("avdec_h265") || factory_name.contains("avdec_hevc") {
                codec_name = "H.265/HEVC (FFmpeg)".to_string();
                break;
            } else if factory_name.contains("avdec_vp9") {
                codec_name = "VP9 (FFmpeg)".to_string();
                break;
            } else if factory_name.contains("avdec_vp8") {
                codec_name = "VP8 (FFmpeg)".to_string();
                break;
            } else if factory_name.contains("avdec_av1") {
                codec_name = "AV1 (FFmpeg)".to_string();
                break;
            } else if factory_name.contains("openh264") {
                codec_name = "H.264 (OpenH264)".to_string();
                break;
            }
        }

        if codec_name.is_empty() {
            "H.264".to_string() // Default fallback for most web content
        } else {
            codec_name
        }
    }

    /// Extracts DMABuf information from a GStreamer buffer.
    ///
    /// Returns `Some(DmaBufInfo)` if the buffer contains DMABuf memory,
    /// `None` if it's regular system memory.
    fn extract_dmabuf_info(
        buffer: &gst::BufferRef,
        video_info: &gst_video::VideoInfo,
        sample: &gst::Sample,
    ) -> Option<DmaBufInfo> {
        // Check if the buffer has DMABuf memory (type-safe check)
        let memory = buffer.memory(0)?;

        // Verify it's actually DMABuf memory before trying to extract FD
        if !memory.is_memory_type::<gst_allocators::DmaBufMemory>() {
            return None;
        }

        // Downcast to DmaBufMemory to access fd() method (type-safe)
        let Some(dmabuf_memory) = memory.downcast_memory_ref::<gst_allocators::DmaBufMemory>()
        else {
            tracing::warn!("Failed to downcast to DmaBufMemory despite type check passing");
            return None;
        };

        // Extract the file descriptor using type-safe API
        let fd = dmabuf_memory.fd();
        if fd < 0 {
            tracing::warn!("DMABuf memory has invalid fd: {}", fd);
            return None;
        }

        let width = video_info.width();
        let height = video_info.height();

        // Get DRM format info from sample caps (GStreamer 1.24+ with va plugin)
        // Example: drm-format="NV12:0x0100000000000002" = Intel X-tile
        let (drm_fourcc_str, modifier) = Self::parse_drm_format_caps(sample).unwrap_or_else(|| {
            tracing::debug!("No drm-format in caps, falling back to video_info format");
            // Fall back to video_info format name
            let format_name = video_info.format().to_str().to_uppercase();
            (format_name, DRM_FORMAT_MOD_LINEAR)
        });

        // Determine n_planes and plane layout from the DRM format
        // For DMA_DRM format, video_info.n_planes() returns 0, so we derive it from the format
        let (n_planes, drm_format) = Self::drm_format_info(&drm_fourcc_str);
        if n_planes == 0 {
            tracing::warn!(
                "Unknown DRM format '{}', cannot determine plane count",
                drm_fourcc_str
            );
            return None;
        }

        // Check for multi-FD layouts after deriving actual plane count
        // (video_info.n_planes() can be 0 for DMA_DRM format, so we check after parsing)
        let n_memory = buffer.n_memory() as u32;
        if n_memory >= n_planes && n_planes > 1 {
            tracing::warn!(
                "Zero-copy: multi-FD DMABuf layout detected ({} memory blocks, {} planes). \
                 Phase 1 only supports single-FD layouts. Falling back to CPU copy.",
                n_memory,
                n_planes
            );
            return None;
        }

        // For DMA_DRM format with VideoMeta, get stride/offset from the meta
        // Otherwise calculate based on format and dimensions
        let (strides, offsets) = if let Some(video_meta) = buffer.meta::<gst_video::VideoMeta>() {
            // Use VideoMeta for stride/offset (more accurate)
            let mut strides = Vec::with_capacity(n_planes as usize);
            let mut offsets = Vec::with_capacity(n_planes as usize);
            for plane in 0..n_planes {
                let Some(&stride) = video_meta.stride().get(plane as usize) else {
                    tracing::warn!("Missing stride for plane {} in VideoMeta", plane);
                    return None;
                };
                let Some(&offset) = video_meta.offset().get(plane as usize) else {
                    tracing::warn!("Missing offset for plane {} in VideoMeta", plane);
                    return None;
                };
                strides.push(stride as u32);
                offsets.push(offset as u32);
            }
            tracing::debug!(
                "Using VideoMeta for plane info: strides={:?}, offsets={:?}",
                strides,
                offsets
            );
            (strides, offsets)
        } else if video_info.n_planes() > 0 {
            // Fall back to video_info if available
            let mut strides = Vec::with_capacity(n_planes as usize);
            let mut offsets = Vec::with_capacity(n_planes as usize);
            for plane in 0..n_planes {
                if let (Some(&stride), Some(&offset)) = (
                    video_info.stride().get(plane as usize),
                    video_info.offset().get(plane as usize),
                ) {
                    strides.push(stride as u32);
                    offsets.push(offset as u32);
                } else {
                    tracing::warn!("Missing stride/offset for plane {} in video_info", plane);
                    return None;
                }
            }
            (strides, offsets)
        } else if drm_fourcc_str == "NV12" && n_planes == 2 {
            // Calculate default layout for NV12 only
            // Y plane: width * height, stride = width
            // UV plane: width * height / 2, stride = width, offset = width * height
            let y_stride = width;
            let y_size = width * height;
            let uv_stride = width;
            let uv_offset = y_size;

            tracing::debug!(
                "Calculating default NV12 layout: Y stride={}, UV offset={}, UV stride={}",
                y_stride,
                uv_offset,
                uv_stride
            );
            (vec![y_stride, uv_stride], vec![0, uv_offset])
        } else {
            // For non-NV12 formats (I420, P010, etc.) without VideoMeta, fall back to CPU copy
            // We can't safely guess the plane layout for these formats
            tracing::warn!(
                "No VideoMeta for format '{}' with {} planes, falling back to CPU copy",
                drm_fourcc_str,
                n_planes
            );
            return None;
        };

        Some(DmaBufInfo {
            fd: fd.as_raw_fd(),
            strides,
            offsets,
            modifier,
            n_planes,
            width,
            height,
            drm_format,
        })
    }

    /// Parses the drm-format caps field to extract format name and modifier.
    /// Returns (fourcc_str, modifier) e.g., ("NV12", 0x0100000000000002)
    fn parse_drm_format_caps(sample: &gst::Sample) -> Option<(String, u64)> {
        let caps = sample.caps()?;
        let structure = caps.structure(0)?;

        // Try to get the drm-format field (GStreamer 1.24+ with va plugin)
        // Format is like "NV12:0x0100000000000002"
        let drm_format: String = structure.get("drm-format").ok()?;

        // Parse format and modifier
        let parts: Vec<&str> = drm_format.split(':').collect();
        let fourcc_str = parts.first()?.to_string();

        let modifier = if let Some(modifier_str) = parts.get(1) {
            if modifier_str.starts_with("0x") || modifier_str.starts_with("0X") {
                u64::from_str_radix(&modifier_str[2..], 16).unwrap_or(DRM_FORMAT_MOD_LINEAR)
            } else {
                modifier_str.parse::<u64>().unwrap_or(DRM_FORMAT_MOD_LINEAR)
            }
        } else {
            DRM_FORMAT_MOD_LINEAR
        };

        tracing::debug!(
            "Parsed drm-format caps: '{}' -> fourcc='{}', modifier=0x{:016x}",
            drm_format,
            fourcc_str,
            modifier
        );

        Some((fourcc_str, modifier))
    }

    /// Returns (n_planes, drm_fourcc) for a given DRM fourcc string.
    fn drm_format_info(fourcc_str: &str) -> (u32, u32) {
        match fourcc_str.to_uppercase().as_str() {
            "NV12" => (2, drm_fourcc::DrmFourcc::Nv12 as u32),
            "NV21" => (2, drm_fourcc::DrmFourcc::Nv21 as u32),
            "YU12" | "I420" => (3, drm_fourcc::DrmFourcc::Yuv420 as u32),
            "YV12" => (3, drm_fourcc::DrmFourcc::Yvu420 as u32),
            "YUYV" | "YUY2" => (1, drm_fourcc::DrmFourcc::Yuyv as u32),
            "UYVY" => (1, drm_fourcc::DrmFourcc::Uyvy as u32),
            "ARGB" | "AR24" => (1, drm_fourcc::DrmFourcc::Argb8888 as u32),
            "ABGR" | "AB24" => (1, drm_fourcc::DrmFourcc::Abgr8888 as u32),
            "XRGB" | "XR24" => (1, drm_fourcc::DrmFourcc::Xrgb8888 as u32),
            "XBGR" | "XB24" => (1, drm_fourcc::DrmFourcc::Xbgr8888 as u32),
            "RGBA" => (1, drm_fourcc::DrmFourcc::Rgba8888 as u32),
            "BGRA" => (1, drm_fourcc::DrmFourcc::Bgra8888 as u32),
            "P010" => (2, drm_fourcc::DrmFourcc::P010 as u32),
            _ => {
                tracing::warn!("Unknown DRM fourcc: '{}'", fourcc_str);
                (0, 0)
            }
        }
    }

    /// Converts DRM fourcc code to PixelFormat.
    /// Used when DMA_DRM format is negotiated and video_info.format() returns Unknown.
    fn drm_fourcc_to_pixel_format(drm_format: u32) -> Option<PixelFormat> {
        match drm_format {
            x if x == drm_fourcc::DrmFourcc::Nv12 as u32 => Some(PixelFormat::Nv12),
            // NV21 has swapped chroma order (VU instead of UV), not supported by our shaders
            x if x == drm_fourcc::DrmFourcc::Nv21 as u32 => None,
            x if x == drm_fourcc::DrmFourcc::Yuv420 as u32 => Some(PixelFormat::Yuv420p),
            x if x == drm_fourcc::DrmFourcc::Yvu420 as u32 => Some(PixelFormat::Yuv420p),
            x if x == drm_fourcc::DrmFourcc::Argb8888 as u32 => Some(PixelFormat::Bgra),
            x if x == drm_fourcc::DrmFourcc::Abgr8888 as u32 => Some(PixelFormat::Rgba),
            x if x == drm_fourcc::DrmFourcc::Xrgb8888 as u32 => Some(PixelFormat::Bgra),
            x if x == drm_fourcc::DrmFourcc::Xbgr8888 as u32 => Some(PixelFormat::Rgba),
            x if x == drm_fourcc::DrmFourcc::Bgra8888 as u32 => Some(PixelFormat::Bgra),
            x if x == drm_fourcc::DrmFourcc::Rgba8888 as u32 => Some(PixelFormat::Rgba),
            _ => None,
        }
    }

    /// Converts DmaBufInfo to LinuxGpuSurface with CPU fallback.
    ///
    /// This creates the LinuxGpuSurface structure that will be imported into Vulkan
    /// by the video_texture module. The GStreamer sample is kept alive as the owner
    /// to ensure the DMABuf FDs remain valid.
    fn dmabuf_info_to_surface(
        dmabuf_info: DmaBufInfo,
        buffer: &gst::BufferRef,
        sample: &gst::Sample,
        _video_info: &gst_video::VideoInfo,
    ) -> Result<LinuxGpuSurface, VideoError> {
        // Validate plane count before duping FD
        if dmabuf_info.n_planes == 0 {
            return Err(VideoError::DecodeFailed(
                "DMABuf has 0 planes, cannot import".to_string(),
            ));
        }

        // Convert DRM fourcc to PixelFormat
        // When using DMA_DRM format, video_info.format() returns Unknown,
        // so we use the drm_format from DmaBufInfo which was parsed from caps
        let format = Self::drm_fourcc_to_pixel_format(dmabuf_info.drm_format).ok_or_else(|| {
            VideoError::UnsupportedFormat(format!(
                "Unsupported DRM fourcc for zero-copy: 0x{:08x}",
                dmabuf_info.drm_format
            ))
        })?;

        // Duplicate the FD before passing to Vulkan
        // CRITICAL: Vulkan import takes ownership and will close the FD.
        // GStreamer also closes the FD when the sample drops.
        // We must dup() to avoid double-close.
        let dup_fd = unsafe { libc::dup(dmabuf_info.fd) };
        if dup_fd < 0 {
            return Err(VideoError::DecodeFailed(format!(
                "Failed to dup DMABuf fd {}: {}",
                dmabuf_info.fd,
                std::io::Error::last_os_error()
            )));
        }

        // Build DmaBufPlane structures for each plane
        let mut planes = Vec::with_capacity(dmabuf_info.n_planes as usize);

        // Validate that we have stride/offset data for all planes before constructing
        let n_planes = dmabuf_info.n_planes as usize;
        if dmabuf_info.strides.len() < n_planes {
            // Close the dup'd FD before returning error
            unsafe { libc::close(dup_fd) };
            return Err(VideoError::DecodeFailed(format!(
                "DMABuf has {} planes but only {} strides",
                n_planes,
                dmabuf_info.strides.len()
            )));
        }
        if dmabuf_info.offsets.len() < n_planes {
            // Close the dup'd FD before returning error
            unsafe { libc::close(dup_fd) };
            return Err(VideoError::DecodeFailed(format!(
                "DMABuf has {} planes but only {} offsets",
                n_planes,
                dmabuf_info.offsets.len()
            )));
        }

        for i in 0..n_planes {
            // Bounds-checked access per AGENTS.md
            let Some(&stride) = dmabuf_info.strides.get(i) else {
                unsafe { libc::close(dup_fd) };
                return Err(VideoError::DecodeFailed(format!(
                    "Missing stride for plane {} (strides.len()={})",
                    i,
                    dmabuf_info.strides.len()
                )));
            };
            let Some(&offset) = dmabuf_info.offsets.get(i) else {
                unsafe { libc::close(dup_fd) };
                return Err(VideoError::DecodeFailed(format!(
                    "Missing offset for plane {} (offsets.len()={})",
                    i,
                    dmabuf_info.offsets.len()
                )));
            };
            let offset = offset as u64;

            // Validate stride is non-zero (a zero stride would produce invalid planes)
            if stride == 0 {
                unsafe { libc::close(dup_fd) };
                return Err(VideoError::DecodeFailed(format!(
                    "DMABuf plane {} has zero stride",
                    i
                )));
            }

            // Calculate plane size based on format
            let plane_height = if i == 0 {
                // Y plane (or full RGBA/BGRA)
                dmabuf_info.height
            } else {
                // UV/U/V planes (half height for NV12/I420, ceiling div for odd heights)
                dmabuf_info.height.div_ceil(2)
            };

            let size = (stride * plane_height) as u64;

            planes.push(DmaBufPlane {
                fd: dup_fd, // Use dup'd FD that Vulkan can take ownership of
                offset,
                stride,
                size,
            });
        }

        // Check if all planes share the same FD (single-FD multi-plane layout)
        // This is typical for VA-API output (single GstMemory with multiple planes)
        let is_single_fd = dmabuf_info.n_planes > 1 && buffer.n_memory() == 1;

        // Keep the GStreamer sample alive to ensure DMABuf FDs remain valid
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(sample.clone());

        // Create the LinuxGpuSurface
        // Safety:
        // - All FDs in `planes` are valid DMABuf FDs obtained from gst_dmabuf_memory_get_fd()
        // - `owner` keeps the GStreamer sample alive, ensuring DMABuf FDs remain valid
        // - `width`, `height`, `format`, and `modifier` are validated from GStreamer video_info
        // - `is_single_fd` is correctly determined from buffer memory layout
        // - No CPU fallback is provided; DMABuf memory cannot be reliably CPU-mapped
        let surface = unsafe {
            LinuxGpuSurface::new(
                planes,
                dmabuf_info.width,
                dmabuf_info.height,
                format,
                dmabuf_info.modifier,
                is_single_fd,
                None, // No CPU fallback - DMABuf memory cannot be CPU-mapped
                owner,
            )
        };

        Ok(surface)
    }

    /// Converts a GStreamer sample to VideoFrame, using zero-copy if available.
    fn sample_to_frame(&mut self, sample: gst::Sample) -> Result<VideoFrame, VideoError> {
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

        // Try zero-copy path if in DMABuf mode
        if self.mode == PipelineMode::ZeroCopy {
            if let Some(dmabuf_info) = Self::extract_dmabuf_info(buffer, &video_info, &sample) {
                // Log at INFO level so we can see it's working
                if self.metrics.zero_copy_frames.load(Ordering::Relaxed) == 0 {
                    tracing::info!(
                        "🚀 Zero-copy DMABuf detected: fd={}, {}x{}, {} planes, modifier=0x{:016x}",
                        dmabuf_info.fd,
                        dmabuf_info.width,
                        dmabuf_info.height,
                        dmabuf_info.n_planes,
                        dmabuf_info.modifier
                    );
                }

                // Convert DmaBufInfo to LinuxGpuSurface
                match Self::dmabuf_info_to_surface(dmabuf_info, buffer, &sample, &video_info) {
                    Ok(surface) => {
                        let frame_count = self.metrics.record_zero_copy();

                        // Log first successful zero-copy frame
                        if frame_count == 1 {
                            tracing::info!(
                                "✓ Zero-copy active: DMABuf → Vulkan import ready ({}x{}, {} planes)",
                                surface.width,
                                surface.height,
                                surface.planes.len()
                            );
                        }

                        return Ok(VideoFrame::new(pts, DecodedFrame::Linux(surface)));
                    }
                    Err(e) => {
                        tracing::warn!("❌ Failed to create LinuxGpuSurface: {:?}", e);
                        self.metrics.record_fallback(
                            &format!("LinuxGpuSurface creation failed: {e}"),
                            0,
                            &self.driver_name,
                        );
                        // Fall through to CPU copy
                    }
                }
            } else {
                // Only log first occurrence to avoid spam
                if self.metrics.fallback_frames.load(Ordering::Relaxed) == 0 {
                    tracing::warn!(
                        "❌ Zero-copy: Buffer does not contain DMABuf memory (using CPU fallback)"
                    );
                }
                self.metrics.record_fallback(
                    "Buffer does not contain DMABuf memory",
                    DRM_FORMAT_MOD_INVALID,
                    &self.driver_name,
                );
            }
        }

        if self.mode == PipelineMode::CpuCopy {
            // In CpuCopy mode, increment fallback counter for each frame without logging
            // (logging is only done once at init to avoid spam)
            self.metrics.fallback_frames.fetch_add(1, Ordering::Relaxed);
        }

        // CPU copy fallback path - only supports NV12 format
        // Validate format before attempting CPU copy to avoid misinterpreting non-NV12 data
        let gst_format = video_info.format();
        if gst_format != gst_video::VideoFormat::Nv12 {
            // Check if the format string indicates NV12 (for DMA_DRM formats where format() returns Unknown)
            let format_name = gst_format.to_str();
            if format_name != "NV12" && format_name != "UNKNOWN" {
                return Err(VideoError::UnsupportedFormat(format!(
                    "CPU fallback only supports NV12, got: {} ({:?})",
                    format_name, gst_format
                )));
            }
            // For UNKNOWN format (DMA_DRM), check caps string for actual format
            if format_name == "UNKNOWN" {
                let caps_str = caps.to_string();
                if !caps_str.contains("NV12") && !caps_str.contains("nv12") {
                    return Err(VideoError::UnsupportedFormat(format!(
                        "CPU fallback only supports NV12, caps indicate: {}",
                        caps_str
                    )));
                }
            }
        }

        self.cpu_copy_frame(buffer, &video_info, pts)
    }

    /// Helper to extract CPU frame data without wrapping in VideoFrame.
    /// Used for CPU fallback in zero-copy path.
    fn cpu_copy_frame_boxed(
        buffer: &gst::BufferRef,
        video_info: &gst_video::VideoInfo,
    ) -> Result<CpuFrame, VideoError> {
        let map = buffer
            .map_readable()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to map buffer: {e}")))?;

        let width = video_info.width();
        let height = video_info.height();

        let y_stride = video_info
            .stride()
            .first()
            .copied()
            .ok_or_else(|| VideoError::DecodeFailed("Missing Y plane stride".into()))?
            as usize;
        let uv_stride = video_info
            .stride()
            .get(1)
            .copied()
            .ok_or_else(|| VideoError::DecodeFailed("Missing UV plane stride".into()))?
            as usize;
        let y_offset = video_info
            .offset()
            .first()
            .copied()
            .ok_or_else(|| VideoError::DecodeFailed("Missing Y plane offset".into()))?;
        let uv_offset = video_info
            .offset()
            .get(1)
            .copied()
            .ok_or_else(|| VideoError::DecodeFailed("Missing UV plane offset".into()))?;

        let y_size = y_stride * height as usize;
        let uv_size = uv_stride * (height as usize).div_ceil(2);

        let data = map.as_slice();

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

        Ok(CpuFrame::new(
            PixelFormat::Nv12,
            width,
            height,
            vec![y_plane, uv_plane],
        ))
    }

    /// Performs CPU copy of frame data (fallback path).
    fn cpu_copy_frame(
        &self,
        buffer: &gst::BufferRef,
        video_info: &gst_video::VideoInfo,
        pts: Duration,
    ) -> Result<VideoFrame, VideoError> {
        let cpu_frame = Self::cpu_copy_frame_boxed(buffer, video_info)?;
        Ok(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame)))
    }

    /// Internal seek implementation.
    fn seek_internal(&mut self, position: Duration) -> Result<(), VideoError> {
        let position_ns = position.as_nanos() as u64;

        self.seeking = true;
        self.seek_target = Some(position);
        self.last_seek_backward = position < self.position;

        let flags = if self.last_seek_backward {
            gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE
        } else {
            gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT
        };

        if let Err(e) = self
            .pipeline
            .seek_simple(flags, gst::ClockTime::from_nseconds(position_ns))
        {
            self.seeking = false;
            self.seek_target = None;
            return Err(VideoError::SeekFailed(format!("Seek failed: {e:?}")));
        }

        if let Some(bus) = self.pipeline.bus() {
            let msg = bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(10),
                &[gst::MessageType::AsyncDone, gst::MessageType::Error],
            );
            match msg {
                Some(msg) => match msg.view() {
                    gst::MessageView::AsyncDone(_) => {
                        tracing::debug!("Seek completed: {:?} -> {:?}", self.position, position);
                        self.seeking = false;
                        self.seek_target = None;
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
                    self.seeking = false;
                    self.seek_target = None;
                    return Err(VideoError::SeekFailed("Seek timed out".into()));
                }
            }
        }

        self.position = position;
        self.eof = false;

        // Reset stuck detection state after seek
        self.buffering_resume_time = None;
        self.no_sample_after_resume = 0;
        self.samples_since_resume = 0; // Reset sample counter for diagnostics

        // Reset audio sink to fix PulseAudio/PipeWire freeze bug after seek
        // Cycles audiosink NULL → PLAYING to clear internal buffer/clock state
        self.audio_handle.reset_after_seek();

        // Track seek time for buffering grace period
        // Don't pause for buffering immediately after seek - it breaks data flow
        self.last_seek_time = Some(std::time::Instant::now());

        // FLUSH seeks maintain the pipeline's previous state (Playing stays Playing).
        // Wait for the pipeline to fully return to Playing state before returning.
        // This gives the pipeline time to stabilize after the flush/segment sequence.
        if !self.user_paused {
            let (result, current, _pending) =
                self.pipeline.state(gst::ClockTime::from_mseconds(500));
            tracing::debug!(
                "Post-seek pipeline state: result={:?}, current={:?}",
                result,
                current
            );

            // If not already Playing, explicitly set it
            if current != gst::State::Playing {
                tracing::info!("Pipeline not Playing after seek, setting state explicitly");
                let _ = self.pipeline.set_state(gst::State::Playing);
                let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
            }

            // NOTE: Preroll pull after seek was removed - it didn't help and may interfere
            // with normal sample flow by consuming/blocking the first buffer.
        }

        Ok(())
    }

    /// Processes a bus message during decode_next.
    fn process_bus_message(
        &mut self,
        msg: &gst::Message,
    ) -> Option<Result<Option<VideoFrame>, VideoError>> {
        match msg.view() {
            gst::MessageView::Error(err) => {
                let error = VideoError::DecodeFailed(format!("Pipeline error: {}", err.error()));
                if self.seeking {
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
                // Log which element emits buffering messages for diagnostics
                let source_name = msg
                    .src()
                    .map(|s| s.name().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                tracing::debug!(
                    "BUFFERING message from '{}': {}%",
                    source_name,
                    buffering.percent()
                );
                self.handle_buffering_message(buffering.percent());
            }
            _ => {}
        }
        None
    }

    /// Handles buffering percentage changes.
    ///
    /// IMPORTANT: Buffering-triggered state changes are currently disabled.
    /// Testing showed that ANY pipeline state change (pause or resume) triggered by
    /// buffering messages after a seek corrupts the data flow, causing the
    /// "appsink produces 0-1 samples then stops" bug.
    ///
    /// GStreamer's internal buffering handling appears sufficient for HTTP streams.
    /// We only track the percentage for UI display.
    fn handle_buffering_message(&mut self, percent: i32) {
        if percent == self.buffering_percent {
            return;
        }

        tracing::debug!("Buffering: {}%", percent);
        self.buffering_percent = percent;

        if percent >= BUFFER_HIGH_THRESHOLD {
            self.was_fully_buffered = true;
        }

        // DISABLED: Buffering-triggered state changes break seek recovery.
        // GStreamer handles buffering internally via BUFFERING messages and
        // automatic pause/resume. Our manual intervention interferes with this.
        //
        // The old logic would:
        // - Call set_state(Playing) when buffer hits 100%
        // - Call set_state(Paused) when buffer drops below 10%
        //
        // But after seek, these state changes corrupt the pipeline's data flow,
        // causing samples_since_resume to stay at 0-1 forever.
    }

    /// Checks if a frame should be discarded as stale during seeking.
    fn is_stale_frame(&self, frame_pts: Duration, discarded: u32, max_stale: u32) -> bool {
        if !self.seeking || discarded >= max_stale {
            return false;
        }

        let Some(target) = self.seek_target else {
            return false;
        };

        let too_far_after = frame_pts > target + Duration::from_secs(2);
        let too_far_before =
            !self.last_seek_backward && frame_pts + Duration::from_millis(100) < target;

        if too_far_after || too_far_before {
            tracing::debug!(
                "Discarding stale frame at {:?} (seek target {:?})",
                frame_pts,
                target
            );
            return true;
        }

        false
    }

    /// Handles the None case when pulling a sample.
    #[allow(clippy::manual_is_multiple_of)]
    fn handle_no_sample(&mut self) {
        // Track consecutive no-sample events after buffering resume
        self.no_sample_after_resume = self.no_sample_after_resume.saturating_add(1);

        // Log periodically (every 10 no-sample events) to diagnose stuck pipeline
        if self.no_sample_after_resume % 10 == 0 {
            let (result, current, pending) = self.pipeline.state(gst::ClockTime::from_mseconds(10));
            tracing::warn!(
                "No sample #{}: pipeline={:?}/{:?} (result={:?}), eos={}, buffering={}%, seeking={}, samples_since_resume={}",
                self.no_sample_after_resume,
                current,
                pending,
                result,
                self.appsink.is_eos(),
                self.buffering_percent,
                self.seeking,
                self.samples_since_resume
            );
        }

        // Stuck detection: log when samples stop flowing
        // Recovery mechanisms (flush, micro-seek) have been tried but made things worse.
        // The root cause appears to be in GStreamer's internal state after multiple seeks on HTTP streams.
        if let Some(resume_time) = self.buffering_resume_time {
            let stuck_duration = resume_time.elapsed();
            if stuck_duration >= Duration::from_secs(2) && self.no_sample_after_resume == 20 {
                tracing::warn!(
                    "Appsink stuck for {:?} after resume (position={:?}). \
                     No automatic recovery - manual seek may help.",
                    stuck_duration,
                    self.position
                );
                // Clear tracking to avoid repeated warnings
                self.buffering_resume_time = None;
                self.no_sample_after_resume = 0;
            }
        }

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

impl Drop for ZeroCopyGStreamerDecoder {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gst::State::Null);

        // Log final metrics
        let zero_copy = self.metrics.zero_copy_frames.load(Ordering::Relaxed);
        let fallback = self.metrics.fallback_frames.load(Ordering::Relaxed);
        if fallback > 0 {
            tracing::warn!(
                "Zero-copy session ended: {} zero-copy, {} fallback ({:.1}% fallback rate)",
                zero_copy,
                fallback,
                self.metrics.fallback_rate()
            );
        }
    }
}

// Safety: Same reasoning as GStreamerDecoder in linux_video_gst.rs
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<gst::Pipeline>();
    assert_send::<gst_app::AppSink>();
    assert_send::<gst::Sample>();
    assert_send::<GstAudioHandle>();
};

impl VideoDecoderBackend for ZeroCopyGStreamerDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    #[allow(clippy::manual_is_multiple_of)]
    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        if let Some(error) = self.pending_error.take() {
            self.seeking = false;
            self.seek_target = None;
            return Err(error);
        }

        if self.eof {
            return Ok(None);
        }

        if let Some(sample) = self.preroll_sample.take() {
            let frame = self.sample_to_frame(sample)?;
            tracing::debug!("Returning cached preroll frame at {:?}", frame.pts);
            self.position = frame.pts;
            return Ok(Some(frame));
        }

        if let Some(bus) = self.pipeline.bus() {
            while let Some(msg) = bus.pop() {
                if let Some(result) = self.process_bus_message(&msg) {
                    return result;
                }
            }
        }

        let timeout_ms = if self.seeking || self.buffering_percent < 100 {
            1000
        } else {
            100
        };

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

            // Sample received - track for diagnostics
            self.samples_since_resume += 1;
            if self.samples_since_resume == 1 {
                tracing::info!("FIRST SAMPLE after resume received!");
            } else if self.samples_since_resume % 100 == 0 {
                tracing::info!("Samples since resume: {}", self.samples_since_resume);
            }

            // Clear stuck detection state
            self.buffering_resume_time = None;
            self.no_sample_after_resume = 0;
            // Track consecutive samples for recovery completion
            if self.recovery_attempts > 0 {
                self.consecutive_samples_after_recovery =
                    self.consecutive_samples_after_recovery.saturating_add(1);
                // Reset recovery counter after sustained playback (30+ frames)
                if self.consecutive_samples_after_recovery >= 30 {
                    tracing::info!(
                        "Recovery complete after {} consecutive samples",
                        self.consecutive_samples_after_recovery
                    );
                    self.recovery_attempts = 0;
                    self.consecutive_samples_after_recovery = 0;
                }
            }

            let frame = self.sample_to_frame(sample)?;

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
        // Prevent overlapping seeks - wait for previous seek to complete
        if self.seeking {
            tracing::info!("Seek already in progress, waiting for it to complete");
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(3);
            while self.seeking && start.elapsed() < timeout {
                if let Some(bus) = self.pipeline.bus() {
                    while let Some(msg) = bus.pop() {
                        let _ = self.process_bus_message(&msg);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if self.seeking {
                tracing::warn!("Previous seek still in progress after timeout, forcing new seek");
                self.seeking = false;
            }
        }

        // Wait for buffering to complete before seeking - seeking while buffering
        // can cause the pipeline to get stuck on HTTP streams.
        // Skip this wait for local files since they have use-buffering=false
        // and buffering messages may never arrive.
        if self.buffering_percent < 100 && !self.is_local_file {
            tracing::info!(
                "Waiting for buffering to complete before seek (currently {}%)",
                self.buffering_percent
            );
            // Poll for buffering completion with timeout
            let start = std::time::Instant::now();
            let timeout = std::time::Duration::from_secs(5);
            while self.buffering_percent < 100 && start.elapsed() < timeout {
                // Process bus messages to update buffering_percent
                if let Some(bus) = self.pipeline.bus() {
                    while let Some(msg) = bus.pop() {
                        let _ = self.process_bus_message(&msg);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if self.buffering_percent < 100 {
                tracing::warn!(
                    "Buffering still at {}% after timeout, proceeding with seek anyway",
                    self.buffering_percent
                );
            }
        }

        const MAX_RETRIES: u32 = 3;
        let mut last_error = None;

        for attempt in 0..=MAX_RETRIES {
            match self.seek_internal(position) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt < MAX_RETRIES {
                        tracing::warn!("Seek attempt {} failed, retrying: {}", attempt + 1, e);
                        let was_paused = self.user_paused;
                        let _ = self.pipeline.set_state(gst::State::Paused);
                        let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
                        let _ = self.pipeline.set_state(gst::State::Playing);
                        let _ = self.pipeline.state(gst::ClockTime::from_mseconds(500));
                        if was_paused {
                            let _ = self.pipeline.set_state(gst::State::Paused);
                            let _ = self.pipeline.state(gst::ClockTime::from_mseconds(100));
                        }
                        std::thread::sleep(std::time::Duration::from_millis(500));
                    }
                    last_error = Some(e);
                }
            }
        }

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
        tracing::info!("GStreamer: resuming pipeline to Playing state");

        // Track resume time for stuck detection (same as buffering resume)
        self.buffering_resume_time = Some(std::time::Instant::now());
        self.no_sample_after_resume = 0;
        self.samples_since_resume = 0; // Reset sample counter for diagnostics

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
        // Return VA-API since we're using VA-API for decoding
        HwAccelType::Vaapi
    }
}
