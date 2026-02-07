//! macOS hardware-accelerated video decoder using AVFoundation + VideoToolbox.
//!
//! # objc2 Version Coexistence
//!
//! This module uses objc2 0.6.x for AVFoundation bindings while winit uses objc2 0.5.x.
//! **Both versions coexist safely** because they interact with different Objective-C classes:
//! - winit 0.30.x uses objc2 0.5.x for NSWindow/NSView operations
//! - AVFoundation bindings use objc2 0.6.x for AVAsset/CVPixelBuffer handling
//!
//! Since they access disjoint sets of ObjC classes, the Objective-C runtime handles
//! method dispatch correctly for both versions linked in the same binary.
//!
//! # Native Frameworks
//!
//! This module provides zero-dependency video decoding on macOS using native Apple frameworks:
//! - **AVFoundation**: For streaming playback via AVPlayer
//! - **VideoToolbox**: For hardware-accelerated H.264/HEVC/VP9 decoding
//! - **CoreVideo**: For efficient pixel buffer handling
//! - **Metal**: For zero-copy GPU texture import via IOSurface
//!
//! VideoToolbox automatically uses the Apple GPU for decoding, providing excellent
//! performance and power efficiency on all Apple Silicon and Intel Macs.
//!
//! # Zero-Copy GPU Rendering
//!
//! This implementation attempts to use zero-copy GPU rendering via IOSurface when available:
//! 1. AVPlayerItemVideoOutput is configured with `kCVPixelBufferIOSurfacePropertiesKey`
//!    and `kCVPixelBufferMetalCompatibilityKey` to request IOSurface-backed pixel buffers.
//! 2. If `CVPixelBufferGetIOSurface()` returns a valid IOSurface, we confirm hardware decode
//!    is working and frames are GPU-resident.
//! 3. If IOSurface is not available (software decode, unsupported format), we fall back
//!    to CPU copy with WARN-level logging for visibility.
//!
//! Note: Zero-copy to wgpu is now available via `MacOSGpuSurface` using IOSurface
//! imported through the wgpu HAL. The IOSurface is obtained from CVPixelBuffer and
//! imported as a Metal texture for direct GPU rendering without CPU copies.
//!
//! # Streaming Support
//!
//! This implementation uses AVPlayer + AVPlayerItemVideoOutput instead of AVAssetReader.
//! AVPlayer handles buffering automatically and supports streaming from remote URLs.
//! AVAssetReader is designed for offline processing and fails with "Operation Stopped"
//! for remote assets that aren't fully downloaded.
//!
//! # Thread Safety
//!
//! AVPlayer must be created on the main thread. This decoder checks for main thread
//! during initialization and fails if called from a background thread. The video_player.rs
//! module handles this by initializing macOS decoders synchronously on the main thread.
//! Frame polling via AVPlayerItemVideoOutput is thread-safe and works from any thread.

use parking_lot::Mutex;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

// FFI declaration for mach_absolute_time (always available on macOS)
extern "C" {
    fn mach_absolute_time() -> u64;
}

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::MainThreadMarker;
use objc2_av_foundation::{
    AVMediaTypeVideo, AVPlayer, AVPlayerItem, AVPlayerItemOutput, AVPlayerItemOutputPullDelegate,
    AVPlayerItemStatus, AVPlayerItemVideoOutput,
};
use objc2_core_media::{CMTime, CMTimeFlags};
use objc2_core_video::{
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferMetalCompatibilityKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA, CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use objc2_foundation::{
    NSCopying, NSMutableDictionary, NSNumber, NSObjectProtocol, NSString, NSURL,
};

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

use super::video::MacOSGpuSurface;

// FFI declarations for IOSurface
extern "C" {
    /// Returns the IOSurface backing a CVPixelBuffer, or null if not IOSurface-backed.
    fn CVPixelBufferGetIOSurface(pixelBuffer: *const std::ffi::c_void) -> *mut std::ffi::c_void;
}

// ============================================================================
// AVPlayerItemOutputPullDelegate implementation
// ============================================================================
//
// This delegate receives callbacks when AVPlayerItemVideoOutput has new media
// data available after a seek or rebuffer. Without this, hasNewPixelBufferForItemTime
// may return false for an extended period after seeking.

use objc2::define_class;
use objc2::msg_send;
use objc2::AllocAnyThread;
use objc2::DefinedClass;

/// Shared state for the video output delegate.
/// Uses Arc to allow sharing between the delegate and decoder.
#[derive(Debug)]
struct VideoOutputDelegateState {
    /// True when we're waiting for AVPlayerItemVideoOutput to signal data availability
    awaiting_output: AtomicBool,
    /// True when the delegate has been notified that output data is available
    output_ready: AtomicBool,
    /// Timestamp (micros since epoch) when awaiting_output became true
    /// Used for adaptive tolerance expansion
    awaiting_since_us: AtomicU64,
}

impl VideoOutputDelegateState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            awaiting_output: AtomicBool::new(false),
            awaiting_since_us: AtomicU64::new(0),
            output_ready: AtomicBool::new(true), // Start ready (no seek yet)
        })
    }
}

/// Ivars for VideoOutputDelegate Objective-C class
struct VideoOutputDelegateIvars {
    state: Arc<VideoOutputDelegateState>,
}

define_class!(
    // SAFETY:
    // - The superclass NSObject does not have any subclassing requirements.
    // - VideoOutputDelegate does not implement Drop.
    // - This class is only accessed from the delegate queue and the decoder,
    //   both of which properly synchronize via atomic state flags.
    #[unsafe(super(objc2_foundation::NSObject))]
    #[name = "LuminaVideoOutputDelegate"]
    #[ivars = VideoOutputDelegateIvars]

    /// Objective-C class that implements AVPlayerItemOutputPullDelegate.
    /// Receives callbacks when video output has new media data available.
    struct VideoOutputDelegate;

    // Implement NSObjectProtocol (required for all ObjC classes)
    unsafe impl NSObjectProtocol for VideoOutputDelegate {}

    // Implement AVPlayerItemOutputPullDelegate protocol
    unsafe impl AVPlayerItemOutputPullDelegate for VideoOutputDelegate {
        /// Called by AVFoundation when media data becomes available.
        /// This signals that hasNewPixelBufferForItemTime should now return true.
        #[unsafe(method(outputMediaDataWillChange:))]
        fn output_media_data_will_change(&self, _sender: &AVPlayerItemOutput) {
            let state = &self.ivars().state;
            state.output_ready.store(true, Ordering::Release);
            state.awaiting_output.store(false, Ordering::Release);
            tracing::debug!(
                "VideoOutputDelegate: outputMediaDataWillChange - frames now available"
            );
        }
    }
);

impl VideoOutputDelegate {
    /// Creates a new delegate with the given shared state.
    fn new_with_state(state: Arc<VideoOutputDelegateState>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(VideoOutputDelegateIvars { state });
        unsafe { msg_send![super(this), init] }
    }
}

/// Thread-safe wrapper for CVPixelBuffer that can be used with zero-copy GPU surfaces.
///
/// CVPixelBuffer is a reference-counted immutable data container. Once created,
/// its pixel data and IOSurface backing are immutable. CoreFoundation's reference
/// counting is thread-safe, making this safe to share across threads.
///
/// This wrapper is used to keep the CVPixelBuffer alive while its IOSurface is
/// being used by a GPU texture.
#[allow(dead_code)] // Field is intentionally held for its Drop side effect
struct PixelBufferWrapper(Retained<objc2_core_video::CVPixelBuffer>);

impl std::fmt::Debug for PixelBufferWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelBufferWrapper").finish()
    }
}

// SAFETY: CVPixelBuffer is safe to send between threads because:
// - The pixel data is immutable after creation
// - CoreFoundation reference counting is thread-safe
// - The IOSurface backing (if any) is also thread-safe
unsafe impl Send for PixelBufferWrapper {}
unsafe impl Sync for PixelBufferWrapper {}

/// Extracts CPU frame data from a CVPixelBuffer.
///
/// Locks the pixel buffer, copies the BGRA data, and returns a CpuFrame.
/// This is used to provide CPU fallback data alongside zero-copy GPU surfaces.
fn extract_cpu_frame_from_pixel_buffer(
    pixel_buffer: &objc2_core_video::CVPixelBuffer,
    width: usize,
    height: usize,
) -> Result<CpuFrame, VideoError> {
    // Lock the pixel buffer for reading
    let lock_result =
        unsafe { CVPixelBufferLockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly) };
    if lock_result != 0 {
        return Err(VideoError::DecodeFailed(format!(
            "Failed to lock pixel buffer for CPU fallback: {}",
            lock_result
        )));
    }

    let bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
    let base_address = CVPixelBufferGetBaseAddress(pixel_buffer);

    if base_address.is_null() {
        unsafe {
            CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
        }
        return Err(VideoError::DecodeFailed(
            "Null pixel buffer base address".to_string(),
        ));
    }

    let Some(data_size) = bytes_per_row.checked_mul(height) else {
        unsafe {
            CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
        }
        return Err(VideoError::DecodeFailed("Data size overflow".to_string()));
    };
    // SAFETY:
    // - base_address is valid: obtained from CVPixelBufferGetBaseAddress after successful lock
    //   (lock_result == 0 checked above, null pointer checked above)
    // - size is correct: data_size computed from CVPixelBufferGetBytesPerRow * height,
    //   with overflow check via checked_mul
    // - buffer remains locked for the duration of this slice's use
    //   (CVPixelBufferLockBaseAddress succeeded, unlock happens after all slice accesses)
    // - the slice is not held beyond the unlock call at the end of this function
    let bgra_data = unsafe { std::slice::from_raw_parts(base_address as *const u8, data_size) };

    let Some(row_bytes) = width.checked_mul(4) else {
        unsafe {
            CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
        }
        return Err(VideoError::DecodeFailed("Width overflow".to_string()));
    };

    // Copy data, handling potential row padding from CVPixelBuffer
    const WGPU_ALIGNMENT: usize = 256;
    let (data, stride) = if bytes_per_row == row_bytes {
        // No padding - direct copy
        (bgra_data.to_vec(), row_bytes)
    } else if bytes_per_row.is_multiple_of(WGPU_ALIGNMENT) {
        // CVPixelBuffer padding is already 256-aligned - use directly
        (bgra_data.to_vec(), bytes_per_row)
    } else {
        // Has non-aligned padding - strip it
        if bytes_per_row < row_bytes {
            unsafe {
                CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
            }
            return Err(VideoError::DecodeFailed(format!(
                "Invalid bytes_per_row: {} < {}",
                bytes_per_row, row_bytes
            )));
        }
        let mut data = Vec::with_capacity(row_bytes * height);
        for y in 0..height {
            let row_start = y * bytes_per_row;
            let row_end = row_start + row_bytes;
            if row_end > bgra_data.len() {
                unsafe {
                    CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
                }
                return Err(VideoError::DecodeFailed("Row data truncated".to_string()));
            }
            data.extend_from_slice(&bgra_data[row_start..row_end]);
        }
        (data, row_bytes)
    };

    unsafe {
        CVPixelBufferUnlockBaseAddress(pixel_buffer, CVPixelBufferLockFlags::ReadOnly);
    }

    Ok(CpuFrame::new(
        PixelFormat::Bgra,
        width as u32,
        height as u32,
        vec![Plane { data, stride }],
    ))
}

/// Snapshot of zero-copy rendering statistics for diagnostics.
///
/// This is a public read-only view of internal decoding statistics.
/// Useful for performance monitoring and debugging fallback paths.
///
/// # Example
///
/// ```
/// # fn main() {
/// use lumina_video::media::macos_video::MacOSZeroCopyStatsSnapshot;
///
/// // Get stats from a decoder (typically via decoder.zero_copy_stats())
/// let stats = MacOSZeroCopyStatsSnapshot {
///     frames_total: 1000,
///     frames_iosurface_available: 980,
///     frames_cpu_fallback: 20,
/// };
///
/// // Check zero-copy success rate
/// let percentage = stats.iosurface_percentage();
/// assert!((percentage - 98.0).abs() < 0.01);
/// println!("IOSurface success rate: {:.1}%", percentage); // 98.0%
///
/// // Default returns a zeroed snapshot
/// let empty = MacOSZeroCopyStatsSnapshot::default();
/// assert_eq!(empty.iosurface_percentage(), 0.0);
/// # }
/// ```
#[derive(Debug, Clone, Default)]
pub struct MacOSZeroCopyStatsSnapshot {
    /// Total frames decoded
    pub frames_total: u64,
    /// Frames where IOSurface was available (hardware decode confirmed)
    pub frames_iosurface_available: u64,
    /// Frames using CPU fallback path (IOSurface not available)
    pub frames_cpu_fallback: u64,
}

impl MacOSZeroCopyStatsSnapshot {
    /// Returns the percentage of frames with IOSurface available (0.0 - 100.0).
    pub fn iosurface_percentage(&self) -> f64 {
        if self.frames_total == 0 {
            return 0.0;
        }
        (self.frames_iosurface_available as f64 / self.frames_total as f64) * 100.0
    }
}

/// Statistics for zero-copy rendering performance tracking (internal).
///
/// Used to track fallback occurrences for debugging and performance monitoring.
#[derive(Debug, Default)]
struct ZeroCopyStats {
    /// Total frames decoded
    frames_total: u64,
    /// Frames where IOSurface was available (hardware decode confirmed)
    frames_iosurface_available: u64,
    /// Frames using CPU fallback path (IOSurface not available)
    frames_cpu_fallback: u64,
    /// Whether we've logged the fallback warning (avoid spam)
    fallback_logged: bool,
}

/// macOS video decoder using AVPlayer and AVPlayerItemVideoOutput.
///
/// This decoder provides hardware-accelerated video decoding with streaming support
/// and automatic buffering for remote URLs. It checks for IOSurface availability
/// to confirm hardware decode is active, with visible fallback logging when not.
///
/// # Thread Requirements
///
/// - `new()` MUST be called from the main thread (will fail otherwise)
/// - `decode_next()` can be called from any thread (frame polling is thread-safe)
/// - `seek()` can be called from any thread
pub struct MacOSVideoDecoder {
    /// AVPlayer for playback control
    player: Retained<AVPlayer>,
    /// Player item (kept alive for output and status)
    player_item: Retained<AVPlayerItem>,
    /// Video output for frame extraction (thread-safe)
    video_output: Retained<AVPlayerItemVideoOutput>,
    /// Cached metadata (updated once when ready, then immutable)
    /// Using UnsafeCell because the trait requires &VideoMetadata return
    /// Safety: Only written once during init, reads are safe after metadata_ready is true
    metadata: UnsafeCell<VideoMetadata>,
    /// Duration in seconds (updated when ready)
    duration_secs: Mutex<f64>,
    /// Whether EOF has been reached
    eof_reached: AtomicBool,
    /// Whether we've successfully extracted metadata
    metadata_ready: AtomicBool,
    /// Whether preview extraction is done (first pause marks end of preview)
    /// After preview, resume() will unmute audio
    preview_done: AtomicBool,
    /// Whether we're seeking and waiting for completion (triggers buffering UI)
    /// Arc-wrapped for sharing with seek completion handler
    seeking: Arc<AtomicBool>,
    /// Seek generation counter for coalescing seeks
    /// Incremented on each seek, used to identify stale frames/completions
    seek_generation: Arc<AtomicU64>,
    /// Generation of the active seek target (0 = no active seek)
    /// Used with seek_target_pts_micros to reject stale frames
    /// Arc-wrapped for sharing with seek completion handler
    seek_target_gen: Arc<AtomicU64>,
    /// Target position in microseconds for current seek (valid when seek_target_gen > 0)
    /// Using AtomicU64 for lock-free access in hot decode loop
    seek_target_pts_micros: Arc<AtomicU64>,
    /// Last frame PTS to avoid returning duplicate frames
    /// Reset to None on seek to allow accepting the first frame at new position
    last_frame_pts: Mutex<Option<Duration>>,
    /// Zero-copy rendering statistics for performance tracking
    zero_copy_stats: Mutex<ZeroCopyStats>,
    /// Last log timestamp for throttling debug output (per-instance, not global)
    /// Used in decode_next(), stores seconds since epoch
    last_log: AtomicU64,
    /// Separate log throttle for is_seeking() logs (milliseconds since epoch)
    /// Kept separate from last_log to avoid unit confusion (seconds vs millis)
    last_seek_log_ms: AtomicU64,
    /// Poll rate tracking: total polls per second window
    poll_count: AtomicU64,
    /// Poll rate tracking: unique frames delivered per second window
    unique_frame_count: AtomicU64,
    /// Poll rate tracking: last stats log timestamp (seconds since epoch)
    last_poll_stats_log: AtomicU64,
    /// Post-seek warmup: count of unique frames seen before resuming normal flow
    /// This lets the decode buffer fill before the scheduler starts consuming
    warmup_frames: AtomicU32,
    /// Shared state for output delegate (awaiting_output, output_ready flags)
    output_state: Arc<VideoOutputDelegateState>,
    /// Delegate for receiving output data availability callbacks
    /// Must be kept alive for the delegate to receive callbacks
    #[allow(dead_code)] // Intentionally held for its delegate registration
    output_delegate: Retained<VideoOutputDelegate>,
    /// Flag set by seek completion handler to trigger output reset in decode_next
    /// This is the Apple bug workaround - remove/re-add output AFTER seek completes
    needs_output_reset: Arc<AtomicBool>,
}

// SAFETY: Thread-safety analysis for MacOSVideoDecoder:
//
// - AVPlayer/AVPlayerItem: Created on main thread (enforced by MainThreadMarker).
//   After creation, accessed from decode thread via:
//   - `decode_next()`: Uses only AVPlayerItemVideoOutput (documented thread-safe for frame polling)
//   - `seek()`: Calls cancelPendingSeeks/seekToTime (observed safe from background thread,
//     not explicitly documented - Apple's GCD-based implementation appears to handle this)
//   - `pause()/resume()/set_muted()`: Call player.pause/play/setMuted (observed safe,
//     commonly used from background threads in practice)
//   - `buffering_percent()`: Reads isPlaybackLikelyToKeepUp (KVO-backed property, observed safe)
//
// - AVPlayerItemVideoOutput: Documented thread-safe for copyPixelBuffer/hasNewPixelBuffer
//
// - Cross-thread state: All mutable state uses atomics or Mutex for synchronization
//
// - Completion handlers: Capture only Arc-wrapped atomics, run on arbitrary queues
//
// Note: Apple's AVPlayer documentation doesn't explicitly guarantee thread-safety for all
// methods, but GCD-based dispatch and common usage patterns suggest background-thread
// access is safe for the operations used here.
unsafe impl Send for MacOSVideoDecoder {}
unsafe impl Sync for MacOSVideoDecoder {}

impl MacOSVideoDecoder {
    /// Creates a new macOS video decoder for the given URL.
    ///
    /// Configures AVPlayerItemVideoOutput with IOSurface and Metal compatibility
    /// properties to enable zero-copy GPU rendering when available.
    ///
    /// # Thread Safety
    ///
    /// This method MUST be called from the main thread. It will return an error
    /// if called from a background thread. The video_player module handles this
    /// by initializing macOS decoders synchronously before spawning decode threads.
    ///
    /// # Non-blocking
    ///
    /// This method returns immediately without waiting for the video to be ready.
    /// The AVPlayer will buffer in the background. Frame polling in decode_next()
    /// will return None until frames are available.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        tracing::info!("MacOSVideoDecoder: Opening {}", url);

        // Check that we're on the main thread
        let mtm = MainThreadMarker::new().ok_or_else(|| {
            VideoError::DecoderInit(
                "MacOSVideoDecoder must be initialized on the main thread. \
                 This is required by AVPlayer. The video player should call \
                 this synchronously before spawning the decode thread."
                    .to_string(),
            )
        })?;

        Self::init_on_main_thread(url, mtm)
    }

    /// Initialize AVPlayer on the main thread (non-blocking).
    ///
    /// Creates AVPlayer, AVPlayerItem, and AVPlayerItemVideoOutput with BGRA pixel format
    /// and IOSurface/Metal compatibility for zero-copy GPU rendering.
    /// Returns immediately without waiting for the player to become ready - buffering
    /// happens in the background and metadata is extracted lazily via `try_update_metadata`.
    fn init_on_main_thread(url: &str, mtm: MainThreadMarker) -> Result<Self, VideoError> {
        // Create NSURL
        let ns_url: Retained<NSURL> = if url.starts_with("http://") || url.starts_with("https://") {
            let ns_string = NSString::from_str(url);
            NSURL::URLWithString(&ns_string)
                .ok_or_else(|| VideoError::DecoderInit(format!("Invalid URL: {}", url)))?
        } else {
            let path = url.strip_prefix("file://").unwrap_or(url);
            let ns_string = NSString::from_str(path);
            NSURL::fileURLWithPath(&ns_string)
        };

        // Create AVPlayerItem
        let player_item = unsafe { AVPlayerItem::playerItemWithURL(&ns_url, mtm) };

        // Create video output with BGRA settings + IOSurface/Metal compatibility
        let output_settings = Self::create_output_settings();
        let settings_ptr = Retained::as_ptr(&output_settings)
            as *const objc2_foundation::NSDictionary<NSString, AnyObject>;
        let settings: &objc2_foundation::NSDictionary<NSString, AnyObject> =
            unsafe { &*settings_ptr };

        let video_output = unsafe {
            use objc2::AllocAnyThread;
            AVPlayerItemVideoOutput::initWithPixelBufferAttributes(
                AVPlayerItemVideoOutput::alloc(),
                Some(settings),
            )
        };

        // Add output to player item
        unsafe { player_item.addOutput(&video_output) };

        // Create output delegate state and delegate for seek rebuffer notifications
        let output_state = VideoOutputDelegateState::new();
        let output_delegate = VideoOutputDelegate::new_with_state(Arc::clone(&output_state));

        // Set delegate on video output with main dispatch queue
        // The delegate will receive outputMediaDataWillChange: callbacks when
        // new frames become available after a seek or rebuffer
        unsafe {
            let delegate_proto = ProtocolObject::from_ref(&*output_delegate);
            // Use main queue for delegate callbacks (nil = main queue in Apple's API)
            video_output.setDelegate_queue(Some(delegate_proto), None);
        }

        // Request notification when media data becomes available.
        // This helps calibrate the itemTimeForMachAbsoluteTime timing from the start.
        // Without this, frame delivery can be very low (4-5 fps) until a seek triggers it.
        unsafe {
            video_output.requestNotificationOfMediaDataChangeWithAdvanceInterval(0.033);
        }
        tracing::debug!("MacOSVideoDecoder: requested media data change notification at init");

        // Create player
        let player = unsafe { AVPlayer::playerWithPlayerItem(Some(&player_item), mtm) };

        // Mute initially to prevent audio during preview extraction
        // Will be unmuted when user clicks play
        unsafe { player.setMuted(true) };

        // Use placeholder metadata - will be updated when video is ready
        // Note: AVPlayer status transitions happen via the run loop. Since eframe/winit
        // runs the main run loop, status will eventually transition to ReadyToPlay
        // through normal event processing. We don't pump manually to avoid conflicting
        // with winit's event handling.
        let metadata = VideoMetadata {
            width: 1920,
            height: 1080,
            duration: None,
            frame_rate: 30.0,
            codec: "videotoolbox".to_string(),
            pixel_aspect_ratio: 1.0,
            start_time: None, // AVPlayer doesn't expose stream start time
        };

        tracing::info!(
            "MacOSVideoDecoder: Created player with IOSurface/Metal compatibility (paused, waiting for play)"
        );

        Ok(Self {
            player,
            player_item,
            video_output,
            metadata: UnsafeCell::new(metadata),
            duration_secs: Mutex::new(0.0),
            eof_reached: AtomicBool::new(false),
            metadata_ready: AtomicBool::new(false),
            preview_done: AtomicBool::new(false),
            seeking: Arc::new(AtomicBool::new(false)),
            seek_generation: Arc::new(AtomicU64::new(0)),
            seek_target_gen: Arc::new(AtomicU64::new(0)),
            seek_target_pts_micros: Arc::new(AtomicU64::new(0)),
            last_frame_pts: Mutex::new(None),
            zero_copy_stats: Mutex::new(ZeroCopyStats::default()),
            last_log: AtomicU64::new(0),
            last_seek_log_ms: AtomicU64::new(0),
            poll_count: AtomicU64::new(0),
            unique_frame_count: AtomicU64::new(0),
            last_poll_stats_log: AtomicU64::new(0),
            warmup_frames: AtomicU32::new(0),
            output_state,
            output_delegate,
            needs_output_reset: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Try to extract metadata from the player item when it becomes ready.
    ///
    /// Called on each `decode_next()` until metadata is available. When the player
    /// reaches `ReadyToPlay` status, extracts video dimensions, frame rate, and duration.
    /// Uses atomic ordering to ensure metadata writes are visible to other threads.
    fn try_update_metadata(&self) {
        if self.metadata_ready.load(Ordering::Relaxed) {
            return;
        }

        let status = unsafe { self.player_item.status() };
        match status {
            AVPlayerItemStatus::ReadyToPlay => {
                // Extract real metadata now
                let duration_cm = unsafe { self.player_item.duration() };
                let duration = cmtime_to_duration(duration_cm);
                let duration_secs = cmtime_to_seconds(duration_cm);

                let asset = unsafe { self.player_item.asset() };
                let media_type = match unsafe { AVMediaTypeVideo } {
                    Some(mt) => mt,
                    None => return,
                };

                #[allow(deprecated)]
                let video_tracks = unsafe { asset.tracksWithMediaType(media_type) };

                if video_tracks.is_empty() {
                    return;
                }

                let video_track = video_tracks.objectAtIndex(0);
                let natural_size = unsafe { video_track.naturalSize() };
                let w = natural_size.width as u32;
                let h = natural_size.height as u32;

                if w == 0 || h == 0 {
                    return;
                }

                let fps = unsafe { video_track.nominalFrameRate() };
                let fps = if fps <= 0.0 { 30.0 } else { fps };

                // Safety: Only called once, and metadata_ready acts as a barrier
                // After this write completes and metadata_ready is set, no more writes occur
                unsafe {
                    let meta = &mut *self.metadata.get();
                    meta.width = w;
                    meta.height = h;
                    meta.duration = duration;
                    meta.frame_rate = fps;
                }

                *self.duration_secs.lock() = duration_secs;
                self.metadata_ready.store(true, Ordering::Release);

                tracing::info!(
                    "MacOSVideoDecoder: Video ready {}x{} @ {:.2}fps, duration: {:?}",
                    w,
                    h,
                    fps,
                    duration
                );
            }
            AVPlayerItemStatus::Failed => {
                let error = unsafe { self.player_item.error() };
                let error_msg = error
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| "Unknown error".to_string());
                tracing::error!("MacOSVideoDecoder: Player item failed: {}", error_msg);
            }
            _ => {
                // Still loading, do nothing
            }
        }
    }

    /// Create pixel buffer attribute dictionary for AVPlayerItemVideoOutput.
    ///
    /// Configures output to use 32-bit BGRA pixel format with IOSurface and Metal
    /// compatibility for zero-copy GPU rendering.
    fn create_output_settings() -> Retained<NSMutableDictionary<NSString, AnyObject>> {
        unsafe {
            let dict: Retained<NSMutableDictionary<NSString, AnyObject>> =
                NSMutableDictionary::new();

            // Set pixel format to BGRA
            let key_cfstring = kCVPixelBufferPixelFormatTypeKey;
            let pixel_format = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);

            let key_ptr = key_cfstring as *const _ as *const NSString;
            let key: &NSString = &*key_ptr;
            let key_copying: &ProtocolObject<dyn NSCopying> = ProtocolObject::from_ref(key);

            let value_ptr = Retained::as_ptr(&pixel_format) as *mut AnyObject;
            let value: &AnyObject = &*value_ptr;

            dict.setObject_forKey(value, key_copying);

            // Set IOSurface properties (empty dictionary enables IOSurface backing)
            // Using FFI constant kCVPixelBufferIOSurfacePropertiesKey from objc2_core_video
            let iosurface_key_cfstring = kCVPixelBufferIOSurfacePropertiesKey;
            let iosurface_key_ptr = iosurface_key_cfstring as *const _ as *const NSString;
            let iosurface_key: &NSString = &*iosurface_key_ptr;
            let iosurface_key_copying: &ProtocolObject<dyn NSCopying> =
                ProtocolObject::from_ref(iosurface_key);
            let iosurface_props: Retained<NSMutableDictionary<NSString, AnyObject>> =
                NSMutableDictionary::new();
            let iosurface_value_ptr = Retained::as_ptr(&iosurface_props) as *mut AnyObject;
            let iosurface_value: &AnyObject = &*iosurface_value_ptr;
            dict.setObject_forKey(iosurface_value, iosurface_key_copying);

            // Set Metal compatibility
            // Using FFI constant kCVPixelBufferMetalCompatibilityKey from objc2_core_video
            let metal_key_cfstring = kCVPixelBufferMetalCompatibilityKey;
            let metal_key_ptr = metal_key_cfstring as *const _ as *const NSString;
            let metal_key: &NSString = &*metal_key_ptr;
            let metal_key_copying: &ProtocolObject<dyn NSCopying> =
                ProtocolObject::from_ref(metal_key);
            let metal_value = NSNumber::numberWithBool(true);
            let metal_value_ptr = Retained::as_ptr(&metal_value) as *mut AnyObject;
            let metal_value: &AnyObject = &*metal_value_ptr;
            dict.setObject_forKey(metal_value, metal_key_copying);

            tracing::debug!(
                "MacOSVideoDecoder: Configured output with IOSurface + Metal compatibility"
            );

            dict
        }
    }

    /// Check if a CVPixelBuffer is IOSurface-backed.
    ///
    /// Returns true if the pixel buffer has an IOSurface backing (hardware decode),
    /// false otherwise (CPU-backed buffer, software decode).
    fn has_iosurface(pixel_buffer: &objc2_core_video::CVPixelBuffer) -> bool {
        // Get raw pointer to the CVPixelBuffer object (not the Retained wrapper)
        let pb_ptr = std::ptr::from_ref(pixel_buffer) as *const std::ffi::c_void;
        let io_surface = unsafe { CVPixelBufferGetIOSurface(pb_ptr) };
        !io_surface.is_null()
    }

    /// Update zero-copy stats and log fallback warning if needed.
    fn update_stats(
        &self,
        iosurface_available: bool,
        pixel_format: u32,
        width: usize,
        height: usize,
    ) {
        let mut stats = self.zero_copy_stats.lock();
        stats.frames_total += 1;

        if iosurface_available {
            stats.frames_iosurface_available += 1;
        } else {
            stats.frames_cpu_fallback += 1;

            // Log warning once per session (avoid spam)
            if !stats.fallback_logged {
                stats.fallback_logged = true;
                let pixel_format_str = pixel_format_to_string(pixel_format);

                // Get Metal device name if available
                let device_info = metal::Device::system_default()
                    .map(|d| format!(", metal_device={}", d.name()))
                    .unwrap_or_default();

                tracing::warn!(
                    "MacOSVideoDecoder: IOSurface not available, using CPU fallback. \
                     pixel_format={}, dimensions={}x{}, codec=videotoolbox{}. \
                     This may indicate software decode or unsupported format.",
                    pixel_format_str,
                    width,
                    height,
                    device_info
                );
            }
        }
    }

    /// Returns a snapshot of the current zero-copy rendering statistics.
    ///
    /// Useful for diagnostics and performance monitoring. The stats track
    /// IOSurface availability which indicates whether hardware decode is active.
    pub fn zero_copy_stats(&self) -> MacOSZeroCopyStatsSnapshot {
        let stats = self.zero_copy_stats.lock();
        MacOSZeroCopyStatsSnapshot {
            frames_total: stats.frames_total,
            frames_iosurface_available: stats.frames_iosurface_available,
            frames_cpu_fallback: stats.frames_cpu_fallback,
        }
    }
}

impl Drop for MacOSVideoDecoder {
    fn drop(&mut self) {
        // Pause playback before dropping
        unsafe { self.player.pause() };

        // Log final zero-copy stats
        let stats = self.zero_copy_stats.lock();
        if stats.frames_total > 0 {
            let iosurface_pct =
                (stats.frames_iosurface_available as f64 / stats.frames_total as f64) * 100.0;
            tracing::info!(
                "MacOSVideoDecoder: Decoded {} frames ({:.1}% IOSurface available, {} CPU fallback)",
                stats.frames_total,
                iosurface_pct,
                stats.frames_cpu_fallback
            );
        }
    }
}

impl VideoDecoderBackend for MacOSVideoDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    #[profiling::function]
    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        if self.eof_reached.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // WORKAROUND: Apple bug radar #24725691 - hasNewPixelBufferForItemTime returns NO after seek.
        // The fix is to remove and re-add the video output AFTER seek completes.
        // See: https://developer.apple.com/forums/thread/27589
        // The completion handler sets this flag, and we perform the reset here in decode_next.
        if self.needs_output_reset.swap(false, Ordering::AcqRel) {
            unsafe {
                self.player_item.removeOutput(&self.video_output);
                self.player_item.addOutput(&self.video_output);
            }
            tracing::info!(
                "MacOSVideoDecoder: output reset AFTER seek complete (Apple bug workaround)"
            );

            // Request notification when new frames are ready
            unsafe {
                self.video_output
                    .requestNotificationOfMediaDataChangeWithAdvanceInterval(0.033);
            }
        }

        // Try to update metadata if not ready yet
        self.try_update_metadata();

        // Gate polling when awaiting output after seek.
        // Don't hammer hasNewPixelBufferForItemTime while AVPlayer is rebuffering.
        // Wait for the delegate callback (outputMediaDataWillChange:) to signal readiness.
        let awaiting_output = self.output_state.awaiting_output.load(Ordering::Acquire);
        let output_ready = self.output_state.output_ready.load(Ordering::Acquire);
        if awaiting_output && !output_ready {
            // Check for timeout - delegate callback may not fire reliably on all streams.
            // After 1 second, give up waiting and resume polling to prevent indefinite stall.
            let awaiting_since = self.output_state.awaiting_since_us.load(Ordering::Relaxed);
            if awaiting_since > 0 {
                let now_us = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_micros() as u64;
                let waiting_ms = (now_us.saturating_sub(awaiting_since)) / 1000;
                if waiting_ms > 1000 {
                    // Timeout: delegate didn't fire, resume polling anyway
                    tracing::debug!(
                        "MacOSVideoDecoder: delegate timeout after {}ms, resuming poll",
                        waiting_ms
                    );
                    self.output_state
                        .awaiting_output
                        .store(false, Ordering::Release);
                } else {
                    // Still within timeout window - wait for delegate
                    return Ok(None);
                }
            } else {
                // No timestamp recorded, wait for delegate
                return Ok(None);
            }
        }

        // Track poll rate for debugging
        self.poll_count.fetch_add(1, Ordering::Relaxed);

        // Log poll stats every second
        if let Ok(dur) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            let now_secs = dur.as_secs();
            let last_stats = self.last_poll_stats_log.load(Ordering::Relaxed);
            if last_stats != now_secs
                && self
                    .last_poll_stats_log
                    .compare_exchange(last_stats, now_secs, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                let polls = self.poll_count.swap(0, Ordering::Relaxed);
                let frames = self.unique_frame_count.swap(0, Ordering::Relaxed);
                if polls > 0 {
                    tracing::debug!(
                        "MacOSVideoDecoder: poll rate = {}/s, unique frames = {}/s, efficiency = {:.1}%",
                        polls, frames, (frames as f64 / polls as f64) * 100.0
                    );
                }
            }
        }

        // A/B test: Use currentTime directly instead of mach_absolute_time conversion
        // Set to true to test if this fixes the timebase calibration issue
        const USE_CURRENT_TIME_TIMEBASE: bool = true;

        let item_time = if USE_CURRENT_TIME_TIMEBASE {
            // Use player.currentTime() directly - should be more accurate post-seek
            unsafe { self.player.currentTime() }
        } else {
            // Original: Use mach_absolute_time + itemTimeForMachAbsoluteTime
            let host_time = unsafe { mach_absolute_time() };
            unsafe {
                self.video_output
                    .itemTimeForMachAbsoluteTime(host_time as i64)
            }
        };

        // Timebase diagnostic: compare item_time vs current_time (throttled to 1/sec in main log)
        let current_time = unsafe { self.player.currentTime() };
        let item_time_dur = cmtime_to_duration(item_time);
        let current_time_dur = cmtime_to_duration(current_time);

        // Debug: check player state and output attachment
        let player_rate = unsafe { self.player.rate() };
        let item_status = unsafe { self.player_item.status() };
        let _time_control = unsafe { self.player.timeControlStatus() };
        let outputs = unsafe { self.player_item.outputs() };
        let _output_attached = !outputs.is_empty();

        // Check for errors
        let error = unsafe { self.player_item.error() };
        let error_msg = error.as_ref().map(|e| e.localizedDescription().to_string());

        // Also check player error
        let player_error = unsafe { self.player.error() };
        let player_error_msg = player_error
            .as_ref()
            .map(|e| e.localizedDescription().to_string());

        // Check buffering status
        let _buffer_empty = unsafe { self.player_item.isPlaybackBufferEmpty() };
        let _buffer_full = unsafe { self.player_item.isPlaybackBufferFull() };
        let _likely_keep_up = unsafe { self.player_item.isPlaybackLikelyToKeepUp() };

        // Log once per second to avoid spam (per-instance throttling)
        if let Ok(dur) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            let now = dur.as_secs();
            if self.last_log.swap(now, Ordering::Relaxed) != now {
                // Timebase diagnostic: show delta between item_time and current_time
                let delta_ms = match (item_time_dur, current_time_dur) {
                    (Some(it), Some(ct)) => {
                        let it_ms = it.as_millis() as i64;
                        let ct_ms = ct.as_millis() as i64;
                        Some(it_ms - ct_ms)
                    }
                    _ => None,
                };
                tracing::debug!(
                    "AVPlayer: rate={}, status={:?}, timebase_delta={:?}ms, item_time={:?}, current_time={:?}",
                    player_rate,
                    item_status,
                    delta_ms,
                    item_time_dur,
                    current_time_dur,
                );
            }
        }

        // Early error handling: surface AVPlayerItem failures instead of silently stalling.
        if item_status == AVPlayerItemStatus::Failed {
            let msg = error_msg
                .or(player_error_msg)
                .unwrap_or_else(|| "Unknown AVPlayerItem error".to_string());
            return Err(VideoError::DecodeFailed(format!(
                "AVPlayerItem failed: {msg}"
            )));
        }

        // If status is Unknown, log it - AVFoundation is still loading/buffering.
        // The run loop is pumped by eframe/winit, so status should eventually transition.
        if item_status == AVPlayerItemStatus(0) {
            // Log if there's an error
            if let Some(ref msg) = error_msg.or_else(|| player_error_msg.clone()) {
                tracing::warn!("AVPlayerItem status Unknown with error: {}", msg);
            }
        }

        // Check if there's a new frame available at this time
        let mut has_new = unsafe { self.video_output.hasNewPixelBufferForItemTime(item_time) };
        let is_seeking = self.seeking.load(Ordering::Relaxed);

        // After seeking, itemTimeForMachAbsoluteTime may return a time far from where AVPlayer
        // actually is. Fall back to player.currentTime() to get frames at the actual position.
        let item_time = if !has_new {
            let current_time = unsafe { self.player.currentTime() };
            has_new = unsafe { self.video_output.hasNewPixelBufferForItemTime(current_time) };
            if has_new {
                current_time
            } else {
                // Debug: log when no frame available during/after seek
                if is_seeking {
                    tracing::trace!(
                        "MacOSVideoDecoder: no frame available during seek (currentTime={:?})",
                        cmtime_to_duration(current_time)
                    );
                }
                // Check for EOF
                let duration_secs = *self.duration_secs.lock();
                let current_secs = cmtime_to_seconds(current_time);
                if duration_secs > 0.0 && current_secs >= duration_secs - 0.1 {
                    self.eof_reached.store(true, Ordering::Relaxed);
                }
                return Ok(None);
            }
        } else {
            item_time
        };

        // Copy pixel buffer (thread-safe operation)
        let mut actual_time = item_time;
        let pixel_buffer = unsafe {
            self.video_output
                .copyPixelBufferForItemTime_itemTimeForDisplay(
                    item_time,
                    &mut actual_time as *mut CMTime,
                )
        };

        let Some(pixel_buffer) = pixel_buffer else {
            return Ok(None);
        };

        // Get frame PTS and check for duplicates
        let pts = cmtime_to_duration(actual_time).unwrap_or(Duration::ZERO);

        // Stale frame filter: reject frames outside ±tolerance of seek target.
        // This prevents old frames (decoded before seek) from corrupting position tracking.
        // Bidirectional: handles both forward seeks (stale frames before target) and
        // backward seeks (stale frames after target).
        // Filter is active when seek_target_gen > 0, cleared when frame is accepted.
        let target_gen = self.seek_target_gen.load(Ordering::Acquire);
        if target_gen > 0 {
            let target_micros = self.seek_target_pts_micros.load(Ordering::Relaxed);
            let target = Duration::from_micros(target_micros);

            // Adaptive tolerance: start at 500ms, expand to 2000ms after 1 second of waiting.
            // This handles streams with large GOP/keyframe intervals (common on remote streams).
            let awaiting_since = self.output_state.awaiting_since_us.load(Ordering::Relaxed);
            let now_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_micros() as u64;
            let waiting_duration_ms = if awaiting_since > 0 {
                (now_us.saturating_sub(awaiting_since)) / 1000
            } else {
                0
            };
            let tolerance = if waiting_duration_ms > 1000 {
                // After 1 second, expand tolerance to 2 seconds for large GOP streams
                Duration::from_millis(2000)
            } else {
                // Initial tight tolerance for VOD - 500ms handles minor keyframe misalignment
                Duration::from_millis(500)
            };

            // Bidirectional: reject frames outside ±tolerance window
            let too_early = pts + tolerance < target;
            let too_late = pts > target + tolerance;

            if too_early || too_late {
                let delta = if too_early {
                    format!("too early by {:?}", target.saturating_sub(pts))
                } else {
                    format!("too late by {:?}", pts.saturating_sub(target))
                };
                tracing::debug!(
                    "MacOSVideoDecoder: rejecting stale frame at {:?} (target {:?} ±{:?}, {}, gen {})",
                    pts, target, tolerance, delta, target_gen
                );
                return Ok(None);
            }
            // Frame is within tolerance of target - acceptable, will clear filter below
        }

        let mut last_pts = self.last_frame_pts.lock();
        if *last_pts == Some(pts) {
            // Same frame as last time, skip to avoid duplicate conversion
            return Ok(None);
        }
        *last_pts = Some(pts);
        drop(last_pts);

        // Process pixel buffer - get dimensions and format first
        let pixel_format = CVPixelBufferGetPixelFormatType(&pixel_buffer);
        if pixel_format != kCVPixelFormatType_32BGRA {
            return Err(VideoError::DecodeFailed(format!(
                "Unexpected pixel format: {} (expected BGRA)",
                pixel_format_to_string(pixel_format)
            )));
        }

        let width = CVPixelBufferGetWidth(&pixel_buffer);
        let height = CVPixelBufferGetHeight(&pixel_buffer);

        // Check for IOSurface availability (confirms hardware decode is working)
        let iosurface_available = Self::has_iosurface(&pixel_buffer);

        // Update stats and log fallback if IOSurface not available
        self.update_stats(iosurface_available, pixel_format, width, height);

        // Post-seek warmup: let decode buffer fill before scheduler consumes.
        // TIME-BASED dual-gate approach matching is_seeking():
        // 1. Exit immediately if isPlaybackLikelyToKeepUp is true
        // 2. OR timeout fallback to avoid deadlock
        const MAX_WAIT_SECS: u64 = 3; // Timeout fallback

        let is_seeking = self.seeking.load(Ordering::Relaxed);
        if is_seeking {
            let warmup = self.warmup_frames.fetch_add(1, Ordering::Relaxed) + 1;
            let likely_to_keep_up = unsafe { self.player_item.isPlaybackLikelyToKeepUp() };

            // Check elapsed time since seek started
            let seek_started_us = self.output_state.awaiting_since_us.load(Ordering::Relaxed);
            let now_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_micros() as u64;
            let elapsed_ms = (now_us.saturating_sub(seek_started_us)) / 1_000;

            // Gate 1: Warmup complete if likely_to_keep_up is true
            // Gate 2: OR timeout fallback
            let warmup_complete = likely_to_keep_up || elapsed_ms >= MAX_WAIT_SECS * 1000;

            if warmup_complete {
                // Warmup complete - clear seeking state
                self.seeking.store(false, Ordering::Relaxed);
                self.seek_target_gen.store(0, Ordering::Release);
                self.output_state
                    .awaiting_output
                    .store(false, Ordering::Release);
                self.output_state
                    .awaiting_since_us
                    .store(0, Ordering::Relaxed);
                tracing::debug!(
                    "MacOSVideoDecoder: warmup complete at {:?} (buffered {} frames, likelyToKeepUp={}, elapsed={}ms)",
                    pts, warmup - 1, likely_to_keep_up, elapsed_ms
                );
            } else {
                // Still warming up
                tracing::debug!(
                    "MacOSVideoDecoder: warmup frame {} at {:?} (likelyToKeepUp={}, elapsed={}ms, timeout={}ms)",
                    warmup,
                    pts,
                    likely_to_keep_up,
                    elapsed_ms,
                    MAX_WAIT_SECS * 1000
                );
            }
        }

        // Track unique frame delivery for poll rate stats
        self.unique_frame_count.fetch_add(1, Ordering::Relaxed);

        // Zero-copy path: If IOSurface is available and zero-copy feature is enabled,
        // return a GPU surface that can be imported directly into wgpu/Metal.
        // CPU fallback is NOT extracted eagerly to preserve zero-copy benefits.
        // If zero-copy import fails at render time, the next frame decode will
        // fall back to the CPU path (iosurface_available check will fail).
        if iosurface_available {
            // Get raw pointer to the CVPixelBuffer object (Retained auto-derefs)
            let pb_ptr = std::ptr::from_ref(&*pixel_buffer) as *const std::ffi::c_void;
            let io_surface = unsafe { CVPixelBufferGetIOSurface(pb_ptr) };

            if !io_surface.is_null() {
                // Wrap the pixel_buffer in a thread-safe wrapper, then Arc it.
                // The IOSurface is owned by the CVPixelBuffer, so we need to keep the CVPixelBuffer alive.
                // PixelBufferWrapper implements Send+Sync for thread-safe sharing.
                let owner: Arc<dyn std::any::Any + Send + Sync> =
                    Arc::new(PixelBufferWrapper(pixel_buffer));

                // Note: cpu_fallback is None to avoid eager CPU copy. If zero-copy
                // import fails at render time, the rendering code will request
                // a new frame and hit the CPU fallback path below.
                let gpu_surface = unsafe {
                    MacOSGpuSurface::new(
                        io_surface,
                        width as u32,
                        height as u32,
                        PixelFormat::Bgra,
                        None, // No eager CPU fallback - preserves zero-copy benefits
                        owner,
                    )
                };

                tracing::trace!(
                    "MacOSVideoDecoder: returning zero-copy GPU surface {}x{}",
                    width,
                    height
                );

                return Ok(Some(VideoFrame::new(pts, DecodedFrame::MacOS(gpu_surface))));
            }
        }

        // CPU fallback path: Lock the pixel buffer and copy data
        let cpu_frame = extract_cpu_frame_from_pixel_buffer(&pixel_buffer, width, height)?;
        Ok(Some(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        // Cancel any pending seeks to prevent queue buildup
        unsafe { self.player_item.cancelPendingSeeks() };

        // Mark as seeking to trigger buffering UI until frames arrive
        self.seeking.store(true, Ordering::Relaxed);

        // Mark output as not ready - will be set true by delegate callback
        // This prevents hasNewPixelBuffer polling from spinning while AVPlayer rebuffers
        self.output_state
            .output_ready
            .store(false, Ordering::Release);
        self.output_state
            .awaiting_output
            .store(true, Ordering::Release);
        // Record when we started waiting (for adaptive tolerance)
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;
        self.output_state
            .awaiting_since_us
            .store(now_us, Ordering::Relaxed);

        // Reset warmup counter - require N unique frames before resuming normal flow
        // This lets the decode buffer fill before scheduler starts consuming
        self.warmup_frames.store(0, Ordering::Release);

        // Reset last frame PTS so the first frame at new position is accepted
        *self.last_frame_pts.lock() = None;

        let seek_time = duration_to_cmtime(position);
        // Use 0.1s tolerance for faster seeking during scrubbing
        // This allows AVPlayer to seek to nearby keyframes instead of exact position
        let tolerance = duration_to_cmtime(Duration::from_millis(100));

        // Increment generation counter - only the latest seek's completion clears seeking flag
        let gen = self.seek_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let seek_generation = Arc::clone(&self.seek_generation);
        let seeking = Arc::clone(&self.seeking);
        let seek_target_gen = Arc::clone(&self.seek_target_gen);
        let needs_output_reset = Arc::clone(&self.needs_output_reset);

        // Set seek target for frame filtering - reject stale frames from pre-seek position
        // Store target as microseconds for lock-free atomic access
        self.seek_target_pts_micros
            .store(position.as_micros() as u64, Ordering::Relaxed);
        self.seek_target_gen.store(gen, Ordering::Release);

        // Create completion handler for seek.
        // NOTE: We do NOT clear seeking/seek_target_gen here anymore.
        // Those are cleared when a valid frame is actually accepted in decode_next().
        // This ensures buffering UI stays visible until frames are truly available.
        let completion = RcBlock::new(move |finished: Bool| {
            let is_latest = seek_generation.load(Ordering::Relaxed) == gen;

            if !finished.as_bool() {
                tracing::debug!("MacOSVideoDecoder: seek {} was cancelled", gen);
                // Clear state on cancel if this is still the latest seek,
                // otherwise frame filtering stays active indefinitely
                if is_latest {
                    seeking.store(false, Ordering::Relaxed);
                    seek_target_gen.store(0, Ordering::Release);
                }
                return;
            }

            // Seek completed in AVPlayer, but frames may not be available yet.
            // Log completion but don't clear seeking - that happens when frame is accepted.
            if is_latest {
                tracing::debug!(
                    "MacOSVideoDecoder: seek {} completed (awaiting frames)",
                    gen
                );
                // WORKAROUND: Set flag to trigger output reset in decode_next.
                // This is the Apple bug fix - remove/re-add output AFTER seek completes.
                needs_output_reset.store(true, Ordering::Release);
            } else {
                tracing::debug!(
                    "MacOSVideoDecoder: ignoring stale seek {} completion (current: {})",
                    gen,
                    seek_generation.load(Ordering::Relaxed)
                );
            }
        });

        unsafe {
            self.player_item
                .seekToTime_toleranceBefore_toleranceAfter_completionHandler(
                    seek_time,
                    tolerance,
                    tolerance,
                    Some(&completion),
                )
        };

        // NOTE: Output reset workaround moved to decode_next() - it runs AFTER seek completes
        // via the needs_output_reset flag set in the completion handler.

        self.eof_reached.store(false, Ordering::Relaxed);
        tracing::debug!(
            "MacOSVideoDecoder: seeking to {:?} (gen {}, with tolerance)",
            position,
            gen
        );
        Ok(())
    }

    /// Pause AVPlayer playback.
    ///
    /// The first pause marks the end of preview extraction phase.
    fn pause(&mut self) -> Result<(), VideoError> {
        unsafe { self.player.pause() };
        // First pause marks end of preview - subsequent resumes will unmute
        self.preview_done.store(true, Ordering::Relaxed);
        tracing::debug!("MacOSVideoDecoder: paused");
        Ok(())
    }

    /// Resume AVPlayer playback.
    ///
    /// Mute state is preserved - callers should use set_muted() to control audio.
    fn resume(&mut self) -> Result<(), VideoError> {
        unsafe { self.player.play() };
        tracing::debug!("MacOSVideoDecoder: resumed/playing");
        Ok(())
    }

    /// Set muted state for AVPlayer audio.
    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        unsafe { self.player.setMuted(muted) };
        tracing::debug!("MacOSVideoDecoder: muted={}", muted);
        Ok(())
    }

    /// AVPlayer handles audio internally with perfect A/V sync.
    fn handles_audio_internally(&self) -> bool {
        true
    }

    /// Returns buffering percentage based on loadedTimeRanges.
    ///
    /// Calculates real percentage based on how much media is buffered ahead of
    /// the current playback position, targeting 10 seconds of buffer.
    /// See: https://developer.apple.com/documentation/avfoundation/avplayeritem/loadedtimeranges
    fn buffering_percent(&self) -> i32 {
        // Check if metadata is ready
        if !self.metadata_ready.load(Ordering::Relaxed) {
            return 0;
        }

        // Check if we're in active seeking state
        let is_seeking = self.seeking.load(Ordering::Relaxed);

        // If not seeking and AVPlayer says playback is likely to keep up, we're good
        // Don't trust isPlaybackLikelyToKeepUp during seeking - it can be premature
        if !is_seeking {
            let likely_to_keep_up = unsafe { self.player_item.isPlaybackLikelyToKeepUp() };
            if likely_to_keep_up {
                return 100;
            }
        }

        // Get position to check buffer for
        // During seek, use seek target; otherwise use player time
        let current_secs = if self.seek_target_gen.load(Ordering::Acquire) > 0 {
            self.seek_target_pts_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
        } else {
            let current_time = unsafe { self.player.currentTime() };
            if current_time.timescale > 0 {
                (current_time.value as f64 / current_time.timescale as f64).max(0.0)
            } else {
                0.0
            }
        };

        // Get loaded time ranges to calculate actual buffer amount
        // Use firstObject to safely get the first range (returns None if empty)
        // This avoids potential panics from objectAtIndex on a mutable array
        let loaded_ranges = unsafe { self.player_item.loadedTimeRanges() };

        let Some(range_value) = loaded_ranges.firstObject() else {
            return 0;
        };

        // NSValue containing CMTimeRange - extract using objc runtime
        let range: objc2_core_media::CMTimeRange =
            unsafe { objc2::msg_send![&*range_value, CMTimeRangeValue] };

        let start_secs = if range.start.timescale > 0 {
            range.start.value as f64 / range.start.timescale as f64
        } else {
            return 0;
        };

        let duration_secs = if range.duration.timescale > 0 {
            range.duration.value as f64 / range.duration.timescale as f64
        } else {
            return 0;
        };

        let end_secs = start_secs + duration_secs;

        // Calculate buffered seconds ahead of current position
        // Use small tolerance for start since position 0 should match a range starting at 0
        let buffered_ahead_secs = if current_secs >= (start_secs - 0.1) && current_secs <= end_secs
        {
            end_secs - current_secs
        } else {
            // Position not in first range - this happens after seek when AVPlayer
            // creates a new range around the seek target.
            0.0
        };

        // Calculate percentage based on target buffer (10 seconds = 100%)
        const TARGET_BUFFER_SECS: f64 = 10.0;
        ((buffered_ahead_secs / TARGET_BUFFER_SECS) * 100.0).min(100.0) as i32
    }

    fn is_seeking(&self) -> bool {
        // Returns true while in seeking/warmup state.
        // TIME-BASED dual-gate approach to avoid deadlock with queue capacity:
        // 1. Wait for MIN_WAIT_MS AND isPlaybackLikelyToKeepUp
        // 2. OR timeout fallback (MAX_WAIT_SECS) to avoid deadlock
        //
        // NOTE: Frame-count warmup doesn't work because:
        // - Queue capacity is 5 frames
        // - If we require 12 frames, push() blocks after 5
        // - But pop() returns None during seeking, so no space frees
        // - Deadlock! Using elapsed time avoids this.
        const MIN_WAIT_MS: u64 = 500; // Minimum warmup time (~12 frames at 24fps)
        const MAX_WAIT_SECS: u64 = 3; // Timeout fallback

        let seeking = self.seeking.load(Ordering::Relaxed);
        if !seeking {
            // Throttle this log - only every 100ms (use dedicated seek log throttle)
            let last = self.last_seek_log_ms.load(Ordering::Relaxed);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64;
            if now.saturating_sub(last) > 100 {
                self.last_seek_log_ms.store(now, Ordering::Relaxed);
                tracing::debug!("is_seeking() = false (seeking flag is false)");
            }
            return false;
        }

        let likely_to_keep_up = unsafe { self.player_item.isPlaybackLikelyToKeepUp() };

        // Check elapsed time since seek started
        let seek_started_us = self.output_state.awaiting_since_us.load(Ordering::Relaxed);
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;
        let elapsed_ms = (now_us.saturating_sub(seek_started_us)) / 1_000;
        let warmup = self.warmup_frames.load(Ordering::Relaxed);

        // Gate 1: Warmup complete if likely_to_keep_up is true
        // Don't wait for MIN_WAIT_MS - AVPlayer may stop providing frames before then
        if likely_to_keep_up {
            tracing::debug!(
                "is_seeking() = FALSE [gate1: likelyToKeepUp=true, elapsed={}ms, warmup={}]",
                elapsed_ms,
                warmup
            );
            return false;
        }

        // Gate 2: Timeout fallback - prevent deadlock if AVPlayer never signals ready
        if elapsed_ms >= MAX_WAIT_SECS * 1000 {
            tracing::debug!(
                "is_seeking() = FALSE [gate2/timeout: elapsed={}ms >= {}ms, likelyToKeepUp={}, warmup={}]",
                elapsed_ms, MAX_WAIT_SECS * 1000, likely_to_keep_up, warmup
            );
            return false;
        }

        // Still seeking - need more warmup time or waiting for AVPlayer
        // Throttle: log every 500ms (use dedicated seek log throttle)
        let last = self.last_seek_log_ms.load(Ordering::Relaxed);
        let now = now_us / 1000; // Convert to ms
        if now.saturating_sub(last) > 500 {
            self.last_seek_log_ms.store(now, Ordering::Relaxed);
            tracing::debug!(
                "is_seeking() = TRUE [elapsed={}ms/{}ms, likelyToKeepUp={}, warmup={}, seek_started_us={}]",
                elapsed_ms, MIN_WAIT_MS, likely_to_keep_up, warmup, seek_started_us
            );
        }
        true
    }

    fn metadata(&self) -> &VideoMetadata {
        // Safety: Caller must ensure metadata_ready is true before calling.
        // This is enforced by VideoPlayer which checks metadata_ready before
        // caching metadata. The Acquire load here synchronizes with the Release
        // store in try_update_metadata(), ensuring we see the complete write.
        // If called before metadata_ready is true, returns default/zeroed metadata.
        let _ = self.metadata_ready.load(Ordering::Acquire);
        unsafe { &*self.metadata.get() }
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::VideoToolbox
    }

    /// Returns the current playback position reported by AVPlayer.
    ///
    /// This queries AVPlayer's `currentTime()` and converts it to a Duration.
    /// Returns None if the time is invalid (e.g., before playback starts).
    fn current_time(&self) -> Option<Duration> {
        let time = unsafe { self.player.currentTime() };
        cmtime_to_duration(time)
    }
}

/// Convert a CVPixelFormat FourCC code to a human-readable string.
fn pixel_format_to_string(format: u32) -> String {
    // Common FourCC codes
    match format {
        0x42475241 => "BGRA".to_string(), // 'BGRA'
        0x52474241 => "RGBA".to_string(), // 'RGBA'
        0x32767579 => "2vuy".to_string(), // '2vuy' - UYVY
        0x79757632 => "yuvs".to_string(), // 'yuvs' - YUYV
        0x34323076 => "420v".to_string(), // '420v' - NV12 video range
        0x34323066 => "420f".to_string(), // '420f' - NV12 full range
        _ => {
            // Try to interpret as FourCC ASCII
            let bytes = format.to_be_bytes();
            if bytes.iter().all(|&b| b.is_ascii_graphic() || b == b' ') {
                format!("'{}'", String::from_utf8_lossy(&bytes))
            } else {
                format!("0x{:08X}", format)
            }
        }
    }
}

fn cmtime_to_duration(time: CMTime) -> Option<Duration> {
    if time.timescale <= 0 {
        return None;
    }
    let seconds = time.value as f64 / time.timescale as f64;
    if seconds < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(seconds))
}

fn cmtime_to_seconds(time: CMTime) -> f64 {
    if time.timescale <= 0 {
        return 0.0;
    }
    time.value as f64 / time.timescale as f64
}

fn duration_to_cmtime(duration: Duration) -> CMTime {
    let timescale: i32 = 600;
    let value = (duration.as_secs_f64() * timescale as f64) as i64;
    CMTime {
        value,
        timescale,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}
