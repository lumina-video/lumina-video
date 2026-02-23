//! Native Windows video decoder using Media Foundation with D3D11 hardware acceleration.
//!
//! This module provides hardware-accelerated video decoding on Windows using:
//! - Media Foundation (MF) for video demuxing and decoding
//! - DXVA2/D3D11VA for GPU-accelerated decode
//! - D3D11 textures for frame output
//!
//! # Architecture
//!
//! The decoder uses `IMFSourceReader` in synchronous mode to poll frames,
//! similar to how the macOS decoder uses `AVPlayerItemVideoOutput`. This avoids
//! the broken `IMFAsyncCallback` implementation in windows-rs.
//!
//! Frame flow:
//! ```text
//! IMFSourceReader::ReadSample()
//!     → IMFSample
//!     → IMFDXGIBuffer (if HW accel)
//!     → ID3D11Texture2D
//!     → Copy to CPU (NV12/BGRA)
//!     → VideoFrame
//! ```
//!
//! # Zero-Copy GPU Rendering (Future)
//!
//! True zero-copy from D3D11 to wgpu requires:
//! 1. Shared texture with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` flag
//! 2. Export NT handle via `IDXGIResource1::CreateSharedHandle`
//! 3. Import into D3D12 via `ID3D12Device::OpenSharedHandle`
//! 4. wgpu hal-level access (not yet public API)
//!
//! This implementation tracks CPU fallback occurrences for visibility.
//! When wgpu exposes ExternalTexture API, the zero-copy path can be enabled.
//!
//! # Hardware Acceleration
//!
//! When `MF_SOURCE_READER_D3D11_DEVICE` is set, Media Foundation automatically
//! uses DXVA2/D3D11VA for hardware decoding when available. The decoder falls
//! back to software decode if hardware acceleration fails.

#[cfg(feature = "zero-copy")]
use crate::video::WindowsGpuSurface;
use crate::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};
use crate::windows_audio::{AudioFormatInfo, AudioFrame};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;
use tracing::{debug, error, info, warn};

// ============================================================================
// Media Foundation Lifecycle Guard
// ============================================================================
//
// Media Foundation requires process-wide initialization via MFStartup/MFShutdown.
// These must be balanced: MFShutdown can only be called once for each MFStartup.
// If multiple decoders exist, calling MFShutdown when one drops would break others.
//
// This guard uses OnceLock<Mutex<Weak<MfGuard>>> to ensure:
// 1. MFStartup is called when the first decoder is created
// 2. MFShutdown is called when the last decoder drops (Weak allows refcount to reach 0)
// 3. If all decoders drop and a new one is created, MFStartup is called again
//
// NOTE: This is a necessary exception to the "no globals" rule because
// Media Foundation's API design requires process-wide state management.
// ============================================================================

/// Global Media Foundation guard slot. Stores a Weak reference so the guard
/// can actually be dropped when all decoders are gone.
static MF_GUARD: OnceLock<Mutex<Weak<MfGuard>>> = OnceLock::new();

/// RAII guard for Media Foundation lifecycle.
///
/// MFStartup is called when the guard is created.
/// MFShutdown is called when the guard is dropped (when last Arc reference drops).
struct MfGuard {
    /// Debug flag for logging.
    debug: bool,
}

impl MfGuard {
    /// Creates a new MF guard, calling MFStartup.
    fn new(debug: bool) -> Result<Self, VideoError> {
        if debug {
            info!("MFStartup: Initializing Media Foundation");
        }
        unsafe {
            MFStartup(MF_VERSION, MFSTARTUP_LITE)
                .map_err(|e| VideoError::DecoderInit(format!("MFStartup failed: {}", e)))?;
        }
        Ok(Self { debug })
    }
}

impl Drop for MfGuard {
    fn drop(&mut self) {
        if self.debug {
            info!("MFShutdown: Cleaning up Media Foundation");
        }
        unsafe {
            let _ = MFShutdown();
        }
    }
}

/// Gets or creates the global MF guard.
///
/// Returns a strong Arc to the guard. When all Arcs are dropped, the guard
/// is dropped and MFShutdown is called. A subsequent call will create a new guard.
fn get_mf_guard(debug: bool) -> Result<Arc<MfGuard>, VideoError> {
    let slot = MF_GUARD.get_or_init(|| Mutex::new(Weak::new()));
    let mut weak = slot.lock();

    // Try to upgrade existing weak reference
    if let Some(existing) = weak.upgrade() {
        return Ok(existing);
    }

    // No existing guard - create a new one
    let strong = Arc::new(MfGuard::new(debug)?);
    *weak = Arc::downgrade(&strong);
    Ok(strong)
}

// ============================================================================
// COM Lifecycle Guard
// ============================================================================
//
// COM requires CoInitializeEx/CoUninitialize to be balanced per-thread.
// This RAII guard ensures COM is properly uninitialized even on early returns.
// It must be the LAST field in WindowsVideoDecoder so it's dropped last
// (Rust drops struct fields in declaration order).
// ============================================================================

/// RAII guard for COM lifecycle.
///
/// CoInitializeEx is called when the guard is created.
/// CoUninitialize is called when the guard is dropped.
struct ComGuard {
    /// Debug flag for logging.
    debug: bool,
}

impl ComGuard {
    /// Creates a new COM guard, calling CoInitializeEx.
    fn new(debug: bool) -> Result<Self, VideoError> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).map_err(|e| {
                VideoError::DecoderInit(format!("COM initialization failed: {}", e))
            })?;
        }
        if debug {
            debug!("COM initialized for this thread");
        }
        Ok(Self { debug })
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        unsafe {
            CoUninitialize();
        }
        if self.debug {
            debug!("COM uninitialized for this thread");
        }
    }
}
use windows::{
    core::{Interface, HSTRING, PCWSTR},
    Win32::{
        Foundation::HANDLE,
        Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0},
        Graphics::Direct3D11::{
            D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Query, ID3D11Texture2D,
            D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_QUERY_DESC, D3D11_QUERY_EVENT,
            D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC,
            D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
        },
        Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12},
        Graphics::Dxgi::{IDXGIResource1, DXGI_SHARED_RESOURCE_READ},
        Media::MediaFoundation::{
            IMF2DBuffer2,
            IMFAttributes,
            IMFDXGIBuffer,
            IMFDXGIDeviceManager,
            IMFMediaBuffer,
            IMFMediaType,
            IMFSample,
            IMFSourceReader,
            MFAudioFormat_Float,
            MFAudioFormat_PCM,
            MFCreateAttributes,
            MFCreateDXGIDeviceManager,
            MFCreateMediaType,
            MFCreateSourceReaderFromURL,
            // Audio-related imports
            MFMediaType_Audio,
            MFMediaType_Video,
            MFShutdown,
            MFStartup,
            MFVideoFormat_NV12,
            MFVideoFormat_RGB32,
            MFSTARTUP_LITE,
            MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
            MF_MT_AUDIO_BITS_PER_SAMPLE,
            MF_MT_AUDIO_BLOCK_ALIGNMENT,
            MF_MT_AUDIO_NUM_CHANNELS,
            MF_MT_AUDIO_SAMPLES_PER_SECOND,
            MF_MT_FRAME_RATE,
            MF_MT_FRAME_SIZE,
            MF_MT_MAJOR_TYPE,
            MF_MT_PIXEL_ASPECT_RATIO,
            MF_MT_SUBTYPE,
            MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
            MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED,
            MF_SOURCE_READERF_ENDOFSTREAM,
            MF_SOURCE_READERF_NEWSTREAM,
            MF_SOURCE_READERF_STREAMTICK,
            MF_SOURCE_READER_D3D_MANAGER,
            MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING,
            MF_SOURCE_READER_FIRST_AUDIO_STREAM,
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            MF_SOURCE_READER_MEDIASOURCE,
        },
        Security::SECURITY_ATTRIBUTES,
        System::Com::StructuredStorage::{PropVariantClear, PROPVARIANT},
        System::Com::{CoInitializeEx, CoUninitialize, COINIT_MULTITHREADED},
    },
};

/// Media Foundation version constant.
const MF_VERSION: u32 = 0x0002_0070; // MF_VERSION from SDK

/// Output format for the video decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    /// NV12 (native hardware decoder format)
    Nv12,
    /// RGB32/BGRA (fallback format)
    Rgb32,
}

/// Windows Media Foundation video decoder.
///
/// Uses `IMFSourceReader` for synchronous frame polling with D3D11 hardware acceleration.
pub struct WindowsVideoDecoder {
    /// Media Foundation source reader for video decode.
    source_reader: IMFSourceReader,

    /// D3D11 device for hardware acceleration.
    device: ID3D11Device,

    /// D3D11 device context for GPU operations.
    context: ID3D11DeviceContext,

    /// DXGI device manager for MF↔D3D11 integration.
    dxgi_manager: IMFDXGIDeviceManager,

    /// Video metadata (dimensions, duration, codec, etc.).
    metadata: VideoMetadata,

    /// Current playback position.
    position: Duration,

    /// Whether end-of-stream has been reached.
    eof: AtomicBool,

    /// Current hardware acceleration type.
    hw_accel: HwAccelType,

    /// Staging texture for CPU readback (reused to avoid allocations).
    staging_texture: Option<ID3D11Texture2D>,

    /// Shared texture for zero-copy rendering via D3D11→D3D12 interop.
    /// Created with D3D11_RESOURCE_MISC_SHARED_NTHANDLE flag.
    #[cfg(feature = "zero-copy")]
    shared_texture: Option<ID3D11Texture2D>,

    /// Shared NT handle for the shared texture.
    /// Obtained via IDXGIResource1::CreateSharedHandle().
    #[cfg(feature = "zero-copy")]
    shared_handle: Option<HANDLE>,

    /// Whether zero-copy is enabled (feature flag + successful shared texture creation).
    #[cfg(feature = "zero-copy")]
    zero_copy_enabled: bool,

    /// Count of frames using CPU fallback (for zero-copy tracking).
    /// Incremented each frame since zero-copy to wgpu is not yet available.
    #[cfg(feature = "zero-copy")]
    cpu_fallback_count: AtomicU64,

    /// Whether the CPU fallback warning has been logged (avoid spam).
    #[cfg(feature = "zero-copy")]
    fallback_logged: AtomicBool,

    /// Debug logging enabled.
    debug_logging: bool,

    /// Media Foundation lifecycle guard.
    /// Ensures MFStartup is called once and MFShutdown only when last decoder drops.
    _mf_guard: Arc<MfGuard>,

    /// Output format (NV12 or RGB32).
    output_format: OutputFormat,

    // ========== Audio fields ==========
    /// Audio format info (None if no audio stream or audio disabled).
    audio_format: Option<AudioFormatInfo>,

    /// Whether audio is enabled and configured.
    audio_enabled: bool,

    /// Whether audio end-of-stream has been reached.
    audio_eof: AtomicBool,

    /// COM lifecycle guard. MUST be LAST field so it's dropped last.
    /// Rust drops struct fields in declaration order, so this ensures COM
    /// remains initialized while other COM objects are being dropped.
    _com_guard: ComGuard,
}

impl WindowsVideoDecoder {
    /// Creates a new Windows video decoder with debug logging control.
    ///
    /// # Arguments
    /// * `url` - URL or file path to the video
    /// * `debug_logging` - Enable verbose debug logging
    ///
    /// # Errors
    /// Returns `VideoError` if initialization fails.
    pub fn new(url: &str, debug_logging: bool) -> Result<Self, VideoError> {
        if debug_logging {
            info!("WindowsVideoDecoder::new() - Initializing for URL: {}", url);
        }

        // Initialize COM for this thread via RAII guard.
        // This ensures COM is properly uninitialized even if later initialization fails.
        let com_guard = ComGuard::new(debug_logging)?;

        // Initialize Media Foundation via global guard
        // This ensures MFStartup is called once and MFShutdown only when last decoder drops
        let mf_guard = get_mf_guard(debug_logging)?;

        // Create D3D11 device with video support
        let (device, context) = Self::create_d3d11_device(debug_logging)?;

        // Create DXGI device manager for hardware acceleration
        let dxgi_manager = Self::create_dxgi_manager(&device, debug_logging)?;

        // Create source reader with hardware acceleration
        let (source_reader, output_format) =
            Self::create_source_reader(url, &dxgi_manager, debug_logging)?;

        // Get video metadata
        let metadata = Self::extract_metadata(&source_reader, debug_logging)?;

        // Configure audio stream (if available)
        // We try to configure but don't fail if audio is unavailable
        let (audio_format, audio_enabled) =
            match Self::configure_audio_stream(&source_reader, 48000, debug_logging) {
                Ok(format) => {
                    if debug_logging {
                        info!(
                            "Audio configured: {}Hz, {} channels, {}-bit",
                            format.sample_rate, format.channels, format.bits_per_sample
                        );
                    }
                    (Some(format), true)
                }
                Err(e) => {
                    if debug_logging {
                        info!("No audio stream or audio configuration failed: {}", e);
                    }
                    (None, false)
                }
            };

        let hw_accel = HwAccelType::D3d11va;

        if debug_logging {
            info!(
                "WindowsVideoDecoder initialized: {}x{}, {:?}, hw_accel={:?}, output_format={:?}, audio={}",
                metadata.width, metadata.height, metadata.codec, hw_accel, output_format, audio_enabled
            );
        }

        Ok(Self {
            source_reader,
            device,
            context,
            dxgi_manager,
            metadata,
            position: Duration::ZERO,
            eof: AtomicBool::new(false),
            hw_accel,
            staging_texture: None,
            #[cfg(feature = "zero-copy")]
            shared_texture: None,
            #[cfg(feature = "zero-copy")]
            shared_handle: None,
            #[cfg(feature = "zero-copy")]
            zero_copy_enabled: true, // Will be disabled if shared texture creation fails
            #[cfg(feature = "zero-copy")]
            cpu_fallback_count: AtomicU64::new(0),
            #[cfg(feature = "zero-copy")]
            fallback_logged: AtomicBool::new(false),
            debug_logging,
            _mf_guard: mf_guard,
            output_format,
            audio_format,
            audio_enabled,
            audio_eof: AtomicBool::new(false),
            // _com_guard must be last so it's dropped last (after all COM objects)
            _com_guard: com_guard,
        })
    }

    /// Creates a D3D11 device with video support.
    fn create_d3d11_device(
        debug_logging: bool,
    ) -> Result<(ID3D11Device, ID3D11DeviceContext), VideoError> {
        if debug_logging {
            debug!("Creating D3D11 device with video support");
        }

        let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
        let feature_levels = [D3D_FEATURE_LEVEL_11_0];

        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;

        unsafe {
            D3D11CreateDevice(
                None, // Default adapter
                D3D_DRIVER_TYPE_HARDWARE,
                None, // No software rasterizer
                flags,
                Some(&feature_levels),
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .map_err(|e| VideoError::DecoderInit(format!("D3D11CreateDevice failed: {}", e)))?;
        }

        let device = device.ok_or_else(|| {
            VideoError::DecoderInit("D3D11CreateDevice returned null device".to_string())
        })?;
        let context = context.ok_or_else(|| {
            VideoError::DecoderInit("D3D11CreateDevice returned null context".to_string())
        })?;

        if debug_logging {
            debug!("D3D11 device created successfully");
        }

        Ok((device, context))
    }

    /// Creates a DXGI device manager for Media Foundation hardware acceleration.
    fn create_dxgi_manager(
        device: &ID3D11Device,
        debug_logging: bool,
    ) -> Result<IMFDXGIDeviceManager, VideoError> {
        if debug_logging {
            debug!("Creating DXGI device manager");
        }

        let mut reset_token: u32 = 0;
        let manager: IMFDXGIDeviceManager = unsafe {
            MFCreateDXGIDeviceManager(&mut reset_token).map_err(|e| {
                VideoError::DecoderInit(format!("MFCreateDXGIDeviceManager failed: {}", e))
            })?
        };

        unsafe {
            manager
                .ResetDevice(device, reset_token)
                .map_err(|e| VideoError::DecoderInit(format!("ResetDevice failed: {}", e)))?;
        }

        if debug_logging {
            debug!("DXGI device manager created with token {}", reset_token);
        }

        Ok(manager)
    }

    /// Creates a source reader with hardware acceleration enabled.
    ///
    /// Returns the source reader and the output format that was configured.
    fn create_source_reader(
        url: &str,
        dxgi_manager: &IMFDXGIDeviceManager,
        debug_logging: bool,
    ) -> Result<(IMFSourceReader, OutputFormat), VideoError> {
        if debug_logging {
            debug!("Creating source reader for: {}", url);
        }

        // Create attributes for source reader
        let attributes: IMFAttributes = unsafe {
            MFCreateAttributes(4)
                .map_err(|e| VideoError::DecoderInit(format!("MFCreateAttributes failed: {}", e)))?
        };

        // Enable hardware transforms
        unsafe {
            attributes
                .SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetUINT32 hardware transforms failed: {}", e))
                })?;
        }

        // Set DXGI device manager for D3D11 integration
        unsafe {
            attributes
                .SetUnknown(&MF_SOURCE_READER_D3D_MANAGER, dxgi_manager)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetUnknown D3D manager failed: {}", e))
                })?;
        }

        // Enable video processing for format conversion
        unsafe {
            attributes
                .SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetUINT32 video processing failed: {}", e))
                })?;
        }

        // Create the source reader
        let url_hstring = HSTRING::from(url);
        let reader: IMFSourceReader = unsafe {
            MFCreateSourceReaderFromURL(&url_hstring, &attributes).map_err(|e| {
                VideoError::OpenFailed(format!("MFCreateSourceReaderFromURL failed: {}", e))
            })?
        };

        // Configure output format (NV12 preferred, RGB32 fallback)
        let output_format = Self::configure_output_format(&reader, debug_logging)?;

        if debug_logging {
            debug!(
                "Source reader created successfully with format {:?}",
                output_format
            );
        }

        Ok((reader, output_format))
    }

    /// Configures the source reader output format.
    ///
    /// Tries NV12 first (native hardware decoder format), falls back to RGB32
    /// if the GPU doesn't support NV12 output.
    ///
    /// Returns the format that was successfully configured.
    fn configure_output_format(
        reader: &IMFSourceReader,
        debug_logging: bool,
    ) -> Result<OutputFormat, VideoError> {
        // Get the native media type to copy frame size
        let native_type: IMFMediaType = unsafe {
            reader
                .GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0)
                .map_err(|e| VideoError::DecoderInit(format!("GetNativeMediaType failed: {}", e)))?
        };

        let mut frame_size: u64 = 0;
        unsafe {
            native_type
                .GetUINT64(&MF_MT_FRAME_SIZE, &mut frame_size)
                .ok();
        }

        // Try NV12 first (native HW decoder format)
        if Self::try_set_output_format(reader, &MFVideoFormat_NV12, frame_size, debug_logging) {
            if debug_logging {
                debug!("Output format configured to NV12");
            }
            return Ok(OutputFormat::Nv12);
        }

        // Fall back to RGB32 if NV12 not supported
        if debug_logging {
            warn!("NV12 output not supported, falling back to RGB32");
        }

        if Self::try_set_output_format(reader, &MFVideoFormat_RGB32, frame_size, debug_logging) {
            if debug_logging {
                debug!("Output format configured to RGB32");
            }
            return Ok(OutputFormat::Rgb32);
        }

        Err(VideoError::DecoderInit(
            "Failed to configure output format (neither NV12 nor RGB32 supported)".to_string(),
        ))
    }

    /// Attempts to set the output format to a specific subtype.
    ///
    /// Returns true on success, false if the format is not supported.
    fn try_set_output_format(
        reader: &IMFSourceReader,
        subtype: &windows::core::GUID,
        frame_size: u64,
        debug_logging: bool,
    ) -> bool {
        let output_type: IMFMediaType = unsafe {
            match windows::Win32::Media::MediaFoundation::MFCreateMediaType() {
                Ok(t) => t,
                Err(e) => {
                    if debug_logging {
                        debug!("MFCreateMediaType failed: {}", e);
                    }
                    return false;
                }
            }
        };

        unsafe {
            // Set major type to video
            if output_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
                .is_err()
            {
                return false;
            }

            // Set requested subtype
            if output_type.SetGUID(&MF_MT_SUBTYPE, subtype).is_err() {
                return false;
            }

            // Set frame size if available
            if frame_size != 0
                && output_type
                    .SetUINT64(&MF_MT_FRAME_SIZE, frame_size)
                    .is_err()
            {
                return false;
            }

            // Try to set the output type
            reader
                .SetCurrentMediaType(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    None,
                    &output_type,
                )
                .is_ok()
        }
    }

    /// Extracts video metadata from the source reader.
    fn extract_metadata(
        reader: &IMFSourceReader,
        debug_logging: bool,
    ) -> Result<VideoMetadata, VideoError> {
        if debug_logging {
            debug!("Extracting video metadata");
        }

        let media_type: IMFMediaType = unsafe {
            reader
                .GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("GetCurrentMediaType failed: {}", e))
                })?
        };

        // Extract frame size
        let mut frame_size: u64 = 0;
        unsafe {
            media_type
                .GetUINT64(&MF_MT_FRAME_SIZE, &mut frame_size)
                .ok();
        }
        let width = (frame_size >> 32) as u32;
        let height = (frame_size & 0xFFFFFFFF) as u32;

        // Extract frame rate
        let mut frame_rate: u64 = 0;
        unsafe {
            media_type
                .GetUINT64(&MF_MT_FRAME_RATE, &mut frame_rate)
                .ok();
        }
        let fps_num = (frame_rate >> 32) as f32;
        let fps_den = (frame_rate & 0xFFFFFFFF) as f32;
        let frame_rate = if fps_den > 0.0 {
            fps_num / fps_den
        } else {
            30.0
        };

        // Extract pixel aspect ratio
        let mut par: u64 = 0;
        unsafe {
            media_type
                .GetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, &mut par)
                .ok();
        }
        let par_num = (par >> 32) as f32;
        let par_den = (par & 0xFFFFFFFF) as f32;
        let pixel_aspect_ratio = if par_den > 0.0 {
            par_num / par_den
        } else {
            1.0
        };

        // Get duration from presentation descriptor
        let duration = Self::get_duration(reader);

        let metadata = VideoMetadata {
            width,
            height,
            duration,
            frame_rate,
            codec: "h264".to_string(), // TODO: Extract actual codec
            pixel_aspect_ratio,
        };

        if debug_logging {
            info!(
                "Video metadata: {}x{} @ {:.2} fps, duration: {:?}",
                metadata.width, metadata.height, metadata.frame_rate, metadata.duration
            );
        }

        Ok(metadata)
    }

    /// Gets the video duration from the source reader.
    fn get_duration(reader: &IMFSourceReader) -> Option<Duration> {
        // MF_PD_DURATION GUID: {6C990D33-BB8E-477A-8598-0D5D96FCD88A}
        let mf_pd_duration = windows::core::GUID::from_u128(0x6c990d33_bb8e_477a_8598_0d5d96fcd88a);

        let mut var = PROPVARIANT::default();
        unsafe {
            // Use GetPresentationAttribute with MF_SOURCE_READER_MEDIASOURCE to get duration
            if reader
                .GetPresentationAttribute(
                    MF_SOURCE_READER_MEDIASOURCE.0 as u32,
                    &mf_pd_duration,
                    &mut var,
                )
                .is_ok()
            {
                // Duration is stored as a 64-bit value in 100-nanosecond units
                let duration_100ns = var.Anonymous.Anonymous.Anonymous.hVal.max(0) as u64;
                let _ = PropVariantClear(&mut var);
                return Some(Duration::from_nanos(duration_100ns * 100));
            }
        }
        None
    }

    /// Reads and decodes the next video frame.
    #[profiling::function]
    fn read_sample(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        let mut flags: u32 = 0;
        let mut timestamp: i64 = 0;
        let mut sample: Option<IMFSample> = None;

        unsafe {
            self.source_reader
                .ReadSample(
                    MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32,
                    0,
                    None,
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
                .map_err(|e| VideoError::DecodeFailed(format!("ReadSample failed: {}", e)))?;
        }

        // Check for end of stream (use imported constant)
        if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 != 0 {
            self.eof.store(true, Ordering::SeqCst);
            if self.debug_logging {
                debug!("End of stream reached");
            }
            return Ok(None);
        }

        // Check for stream tick (no data yet) (use imported constant)
        if flags & MF_SOURCE_READERF_STREAMTICK.0 != 0 {
            if self.debug_logging {
                debug!("Stream tick, no frame yet");
            }
            return Ok(None);
        }

        // Check for media type change (HLS/adaptive streams may change resolution)
        if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 != 0 {
            if self.debug_logging {
                info!("Media type changed, reconfiguring decoder");
            }
            // Re-extract metadata with new dimensions
            self.metadata = Self::extract_metadata(&self.source_reader, self.debug_logging)?;
            // Re-configure output format for new resolution
            self.output_format =
                Self::configure_output_format(&self.source_reader, self.debug_logging)?;
            // Clear staging texture so it's recreated with new dimensions
            self.staging_texture = None;
            if self.debug_logging {
                info!(
                    "Decoder reconfigured: {}x{}, format={:?}",
                    self.metadata.width, self.metadata.height, self.output_format
                );
            }
            // Continue to process the sample if present, or return None to fetch next
        }

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        // Convert timestamp to Duration (100ns units)
        // Clamp negative timestamps to 0 to avoid u64 wrap
        let pts = Duration::from_nanos(timestamp.max(0) as u64 * 100);
        self.position = pts;

        // Extract frame data from sample
        let frame = self.extract_frame(&sample)?;

        if self.debug_logging {
            debug!("Decoded frame at PTS {:?}", pts);
        }

        Ok(Some(VideoFrame::new(pts, frame)))
    }

    /// Extracts frame data from an IMFSample.
    ///
    /// Checks for DXGI buffers BEFORE calling ConvertToContiguousBuffer,
    /// since that operation may copy GPU data to system memory.
    #[profiling::function]
    fn extract_frame(&mut self, sample: &IMFSample) -> Result<DecodedFrame, VideoError> {
        // First, check original sample buffers for DXGI (hardware decode path)
        // We must do this BEFORE ConvertToContiguousBuffer which may copy to CPU memory.
        let buffer_count = unsafe {
            sample
                .GetBufferCount()
                .map_err(|e| VideoError::DecodeFailed(format!("GetBufferCount failed: {}", e)))?
        };

        for i in 0..buffer_count {
            let buffer: IMFMediaBuffer = unsafe {
                match sample.GetBufferByIndex(i) {
                    Ok(b) => b,
                    Err(_) => continue,
                }
            };

            // Try to cast to DXGI buffer for zero-copy hardware path
            if let Ok(dxgi_buffer) = buffer.cast::<IMFDXGIBuffer>() {
                if self.debug_logging {
                    debug!("Found DXGI buffer at index {}, using HW path", i);
                }
                return self.extract_frame_from_dxgi(&dxgi_buffer);
            }
        }

        // No DXGI buffer found - use CPU path
        // ConvertToContiguousBuffer is safe here since we're already on CPU path
        let buffer: IMFMediaBuffer = unsafe {
            sample.ConvertToContiguousBuffer().map_err(|e| {
                VideoError::DecodeFailed(format!("ConvertToContiguousBuffer failed: {}", e))
            })?
        };

        self.extract_frame_from_cpu(&buffer)
    }

    /// Extracts frame from DXGI buffer (hardware decode path).
    ///
    /// When zero-copy feature is enabled, this creates a D3D11 shared texture with
    /// D3D11_RESOURCE_MISC_SHARED_NTHANDLE flag and exports an NT handle via
    /// IDXGIResource1::CreateSharedHandle(). The handle can be imported into D3D12
    /// for wgpu rendering.
    ///
    /// Falls back to CPU readback via staging texture when zero-copy is disabled
    /// or shared texture creation fails.
    #[profiling::function]
    fn extract_frame_from_dxgi(
        &mut self,
        dxgi_buffer: &IMFDXGIBuffer,
    ) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!("Extracting frame from DXGI buffer (HW path)");
        }

        // Get the D3D11 texture and subresource index from DXGI buffer
        let (texture, subresource_index): (ID3D11Texture2D, u32) = unsafe {
            let mut resource: Option<ID3D11Texture2D> = None;
            let mut subresource: u32 = 0;
            dxgi_buffer
                .GetResource(&ID3D11Texture2D::IID, &mut resource as *mut _ as *mut _)
                .map_err(|e| VideoError::DecodeFailed(format!("GetResource failed: {}", e)))?;
            dxgi_buffer
                .GetSubresourceIndex(&mut subresource)
                .map_err(|e| {
                    VideoError::DecodeFailed(format!("GetSubresourceIndex failed: {}", e))
                })?;

            let tex = resource.ok_or_else(|| {
                VideoError::DecodeFailed("DXGI buffer resource is null".to_string())
            })?;
            (tex, subresource)
        };

        // Get texture description
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe {
            texture.GetDesc(&mut desc);
        }

        if self.debug_logging {
            debug!(
                "DXGI texture: {}x{}, format={:?}, subresource={}",
                desc.Width, desc.Height, desc.Format, subresource_index
            );
        }

        // Zero-copy path: Create shared texture and export NT handle
        #[cfg(feature = "zero-copy")]
        if self.zero_copy_enabled {
            // Only support BGRA format for zero-copy (NV12 requires YCbCr conversion)
            let is_bgra = desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM;

            if is_bgra {
                match self.get_or_create_shared_texture(desc.Width, desc.Height, desc.Format) {
                    Ok((shared_texture, shared_handle)) => {
                        // Copy from decoded texture to shared texture
                        unsafe {
                            self.context.CopySubresourceRegion(
                                &shared_texture,
                                0, // Destination subresource
                                0,
                                0,
                                0, // Destination x, y, z
                                &texture,
                                subresource_index, // Source subresource from DXGI buffer
                                None,              // Copy entire subresource
                            );

                            // Flush D3D11 command buffer and wait for GPU completion.
                            // This ensures the copy is fully complete before D3D12 reads.
                            // We use an event query to block (on decode thread, not UI thread)
                            // rather than just Flush() which only submits without waiting.
                            self.context.Flush();
                        }

                        // Wait for GPU to complete the copy before handing off to D3D12.
                        // If wait fails, the shared_texture may contain incomplete data,
                        // so we must NOT return it - fall through to CPU fallback instead.
                        match self.wait_for_gpu() {
                            Ok(()) => {
                                // GPU sync succeeded - safe to hand off to D3D12
                                let owner: Arc<dyn std::any::Any + Send + Sync> =
                                    Arc::new(shared_texture.clone());

                                let surface = unsafe {
                                    WindowsGpuSurface::new(
                                        shared_handle,
                                        desc.Width,
                                        desc.Height,
                                        PixelFormat::Bgra,
                                        None, // cpu_fallback: TODO(lumina-video) - map D3D11 texture and extract
                                        owner,
                                    )
                                };

                                if self.debug_logging {
                                    debug!(
                                        "Zero-copy: returning WindowsGpuSurface {}x{} handle={:?}",
                                        desc.Width, desc.Height, shared_handle
                                    );
                                }

                                return Ok(DecodedFrame::Windows(surface));
                            }
                            Err(e) => {
                                // GPU sync failed - shared_texture may contain incomplete data.
                                // Fall through to CPU fallback rather than returning corrupt frame.
                                warn!(
                                    "Windows zero-copy: wait_for_gpu failed, using CPU fallback: {}",
                                    e
                                );
                                // Don't disable zero_copy_enabled - this may be transient
                            }
                        }
                    }
                    Err(e) => {
                        // Disable zero-copy for future frames after first failure
                        self.zero_copy_enabled = false;
                        warn!(
                            "Windows zero-copy: shared texture creation failed, disabling: {}",
                            e
                        );
                    }
                }
            } else if self.debug_logging {
                debug!(
                    "Zero-copy: format {:?} not supported, using CPU fallback",
                    desc.Format
                );
            }
        }

        // CPU fallback path
        // Track CPU fallback for zero-copy visibility
        #[cfg(feature = "zero-copy")]
        {
            let _fallback_count = self.cpu_fallback_count.fetch_add(1, Ordering::Relaxed) + 1;
            if !self.fallback_logged.swap(true, Ordering::Relaxed) {
                warn!(
                    "Windows zero-copy: CPU fallback active. \
                     DXGI format={:?}, size={}x{}, MiscFlags={:?}",
                    desc.Format, desc.Width, desc.Height, desc.MiscFlags
                );
            }
        }

        // Create or reuse staging texture for CPU readback
        let staging = self.get_or_create_staging_texture(desc.Width, desc.Height, desc.Format)?;

        // Copy from the specific subresource of the GPU texture to subresource 0 of staging
        // This is important because DXVA decoders may use texture arrays where each
        // decoded frame is in a different array slice (subresource).
        unsafe {
            self.context.CopySubresourceRegion(
                &staging,
                0, // Destination subresource (staging texture has only one)
                0,
                0,
                0, // Destination x, y, z
                &texture,
                subresource_index, // Source subresource from DXGI buffer
                None,              // Copy entire subresource
            );
        }

        // Map staging texture and extract pixel data
        self.map_staging_texture(&staging, desc.Width, desc.Height, desc.Format)
    }

    /// Returns the number of frames that used CPU fallback instead of zero-copy.
    ///
    /// This is useful for performance monitoring and debugging. A high count
    /// indicates zero-copy is not available (expected until wgpu exposes the API).
    #[cfg(feature = "zero-copy")]
    pub fn cpu_fallback_count(&self) -> u64 {
        self.cpu_fallback_count.load(Ordering::Relaxed)
    }

    /// Waits for all GPU commands to complete using a D3D11 event query.
    ///
    /// This ensures the D3D11 copy operation is fully complete before the
    /// shared texture handle is passed to D3D12/wgpu. Called on the decode
    /// thread, so blocking here is acceptable per AGENTS.md guidelines
    /// (blocking is only forbidden on the UI/render thread).
    ///
    /// Uses `D3D11_QUERY_EVENT` which signals when all prior GPU work completes.
    #[cfg(feature = "zero-copy")]
    fn wait_for_gpu(&self) -> Result<(), VideoError> {
        // Create an event query
        let query_desc = D3D11_QUERY_DESC {
            Query: D3D11_QUERY_EVENT,
            MiscFlags: 0,
        };

        let query: ID3D11Query = unsafe {
            let mut query: Option<ID3D11Query> = None;
            self.device
                .CreateQuery(&query_desc, Some(&mut query))
                .map_err(|e| {
                    VideoError::DecodeFailed(format!("Failed to create D3D11 event query: {}", e))
                })?;
            query
                .ok_or_else(|| VideoError::DecodeFailed("CreateQuery returned null".to_string()))?
        };

        // End the query (this marks the point where we want to know completion)
        unsafe {
            self.context.End(&query);
        }

        // Poll until the query data is available (GPU work complete)
        // GetData returns S_OK when data is ready, S_FALSE when still pending
        let mut query_data: windows::Win32::Foundation::BOOL = windows::Win32::Foundation::BOOL(0);
        let data_size = std::mem::size_of::<windows::Win32::Foundation::BOOL>() as u32;

        // Bound the busy-wait to prevent infinite loops on GPU hangs
        // 5 seconds at ~1000 iterations/ms = 5_000_000 max iterations
        const MAX_WAIT_ITERATIONS: u32 = 5_000_000;
        let mut iterations: u32 = 0;

        loop {
            let result = unsafe {
                self.context.GetData(
                    &query,
                    Some(std::ptr::from_mut(&mut query_data).cast()),
                    data_size,
                    0, // No flags - block until ready
                )
            };

            match result {
                Ok(()) => {
                    // S_OK - data is ready, GPU work is complete
                    break;
                }
                Err(e) if e.code() == windows::Win32::Foundation::S_FALSE => {
                    // S_FALSE - still pending, yield and retry
                    iterations += 1;
                    if iterations >= MAX_WAIT_ITERATIONS {
                        return Err(VideoError::DecodeFailed(
                            "GPU sync timeout: wait_for_gpu exceeded 5 second limit".to_string(),
                        ));
                    }
                    std::thread::yield_now();
                }
                Err(e) => {
                    // Real error
                    return Err(VideoError::DecodeFailed(format!(
                        "GetData failed on event query: {}",
                        e
                    )));
                }
            }
        }

        Ok(())
    }

    /// Gets or creates a shared texture for zero-copy rendering.
    ///
    /// The shared texture is created with D3D11_RESOURCE_MISC_SHARED_NTHANDLE flag
    /// to enable D3D11→D3D12 interop via NT handles.
    #[cfg(feature = "zero-copy")]
    fn get_or_create_shared_texture(
        &mut self,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<(ID3D11Texture2D, HANDLE), VideoError> {
        // Check if existing shared texture is compatible
        if let (Some(ref shared), Some(handle)) = (&self.shared_texture, self.shared_handle) {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe {
                shared.GetDesc(&mut desc);
            }
            if desc.Width == width && desc.Height == height && desc.Format == format {
                return Ok((shared.clone(), handle));
            }
            // Close old handle before creating new one
            if !handle.is_invalid() {
                unsafe {
                    let _ = windows::Win32::Foundation::CloseHandle(handle);
                }
            }
            self.shared_texture = None;
            self.shared_handle = None;
        }

        // Create new shared texture with NT handle support
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE,
            CPUAccessFlags: windows::Win32::Graphics::Direct3D11::D3D11_CPU_ACCESS_FLAG(0),
            MiscFlags: D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
        };

        let shared_texture: ID3D11Texture2D = unsafe {
            let mut texture: Option<ID3D11Texture2D> = None;
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| {
                    VideoError::DecodeFailed(format!(
                        "CreateTexture2D with SHARED_NTHANDLE failed: {}",
                        e
                    ))
                })?;
            texture.ok_or_else(|| {
                VideoError::DecodeFailed(
                    "CreateTexture2D with SHARED_NTHANDLE returned null".to_string(),
                )
            })?
        };

        // Query IDXGIResource1 interface to get shared handle
        let dxgi_resource: IDXGIResource1 = shared_texture.cast().map_err(|e| {
            VideoError::DecodeFailed(format!("Failed to cast to IDXGIResource1: {}", e))
        })?;

        // Create shared NT handle
        let shared_handle = unsafe {
            dxgi_resource
                .CreateSharedHandle(
                    None::<*const SECURITY_ATTRIBUTES>,
                    DXGI_SHARED_RESOURCE_READ,
                    None,
                )
                .map_err(|e| {
                    VideoError::DecodeFailed(format!("CreateSharedHandle failed: {}", e))
                })?
        };

        if self.debug_logging {
            info!(
                "Created shared D3D11 texture: {}x{} format={:?}, handle={:?}",
                width, height, format, shared_handle
            );
        }

        self.shared_texture = Some(shared_texture.clone());
        self.shared_handle = Some(shared_handle);

        Ok((shared_texture, shared_handle))
    }

    /// Gets or creates a staging texture for CPU readback.
    fn get_or_create_staging_texture(
        &mut self,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<ID3D11Texture2D, VideoError> {
        // Check if existing staging texture is compatible
        if let Some(ref staging) = self.staging_texture {
            let mut desc = D3D11_TEXTURE2D_DESC::default();
            unsafe {
                staging.GetDesc(&mut desc);
            }
            if desc.Width == width && desc.Height == height && desc.Format == format {
                return Ok(staging.clone());
            }
        }

        // Create new staging texture
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            BindFlags: windows::Win32::Graphics::Direct3D11::D3D11_BIND_FLAG(0),
            CPUAccessFlags: D3D11_CPU_ACCESS_READ,
            MiscFlags: windows::Win32::Graphics::Direct3D11::D3D11_RESOURCE_MISC_FLAG(0),
        };

        let staging: ID3D11Texture2D = unsafe {
            let mut texture: Option<ID3D11Texture2D> = None;
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| VideoError::DecodeFailed(format!("CreateTexture2D failed: {}", e)))?;
            texture.ok_or_else(|| {
                VideoError::DecodeFailed("CreateTexture2D returned null".to_string())
            })?
        };

        self.staging_texture = Some(staging.clone());
        Ok(staging)
    }

    /// Maps a staging texture and extracts pixel data.
    fn map_staging_texture(
        &self,
        staging: &ID3D11Texture2D,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<DecodedFrame, VideoError> {
        use windows::Win32::Graphics::Direct3D11::{D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ};

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(|e| VideoError::DecodeFailed(format!("Map failed: {}", e)))?;
        }

        let result = self.copy_mapped_data(&mapped, width, height, format);

        unsafe {
            self.context.Unmap(staging, 0);
        }

        result
    }

    /// Copies mapped texture data to a CpuFrame.
    fn copy_mapped_data(
        &self,
        mapped: &windows::Win32::Graphics::Direct3D11::D3D11_MAPPED_SUBRESOURCE,
        width: u32,
        height: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<DecodedFrame, VideoError> {
        let stride = mapped.RowPitch as usize;
        let data_ptr = mapped.pData as *const u8;

        match format {
            f if f == DXGI_FORMAT_NV12 => {
                // NV12: Y plane followed by interleaved UV plane
                let y_size = stride * height as usize;
                let uv_height = (height as usize + 1) / 2;
                let uv_size = stride * uv_height;

                let y_data = unsafe { std::slice::from_raw_parts(data_ptr, y_size).to_vec() };
                let uv_data =
                    unsafe { std::slice::from_raw_parts(data_ptr.add(y_size), uv_size).to_vec() };

                let frame = CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride,
                        },
                        Plane {
                            data: uv_data,
                            stride,
                        },
                    ],
                );

                Ok(DecodedFrame::Cpu(frame))
            }
            f if f == DXGI_FORMAT_B8G8R8A8_UNORM => {
                // BGRA: single plane
                let size = stride * height as usize;
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size).to_vec() };

                let frame = CpuFrame::new(
                    PixelFormat::Bgra,
                    width,
                    height,
                    vec![Plane { data, stride }],
                );

                Ok(DecodedFrame::Cpu(frame))
            }
            _ => Err(VideoError::UnsupportedFormat(format!(
                "Unsupported DXGI format: {:?}",
                format
            ))),
        }
    }

    /// Extracts frame from CPU buffer (software decode fallback).
    ///
    /// Handles stride alignment properly using IMF2DBuffer2 when available,
    /// falling back to stride calculation from buffer size.
    fn extract_frame_from_cpu(&self, buffer: &IMFMediaBuffer) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!(
                "Extracting frame from CPU buffer (SW path), format={:?}",
                self.output_format
            );
        }

        let width = self.metadata.width;
        let height = self.metadata.height;

        // Try IMF2DBuffer2 first for proper stride handling
        if let Ok(buffer_2d) = buffer.cast::<IMF2DBuffer2>() {
            return self.extract_from_2d_buffer(&buffer_2d, width, height);
        }

        // Fallback: Lock buffer and calculate stride from buffer size
        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut max_length: u32 = 0;
        let mut current_length: u32 = 0;

        unsafe {
            buffer
                .Lock(
                    &mut data_ptr,
                    Some(&mut max_length),
                    Some(&mut current_length),
                )
                .map_err(|e| VideoError::DecodeFailed(format!("Lock failed: {}", e)))?;
        }

        // Guard against zero dimensions to prevent division by zero
        let height_usize = (height as usize).max(1);
        let width_usize = (width as usize).max(1);

        let frame = match self.output_format {
            OutputFormat::Nv12 => {
                // For NV12: total_size = stride * height * 1.5
                let uv_height = (height_usize + 1) / 2;
                let total_height = height_usize + uv_height; // Always >= 2 since height_usize >= 1
                let stride = (current_length as usize / total_height).max(width_usize);

                if self.debug_logging {
                    debug!(
                        "NV12 CPU buffer: {}x{}, stride={}, buffer_len={}",
                        width, height, stride, current_length
                    );
                }

                let y_size = stride * height_usize;
                let uv_size = stride * uv_height;
                let required_size = y_size + uv_size;

                // Validate buffer size before creating raw slices to prevent UB
                if (current_length as usize) < required_size {
                    unsafe {
                        buffer.Unlock().ok();
                    }
                    return Err(VideoError::DecodeFailed(format!(
                        "NV12 buffer too small: {} bytes, need {} ({}x{}, stride={})",
                        current_length, required_size, width, height, stride
                    )));
                }

                let y_data = unsafe { std::slice::from_raw_parts(data_ptr, y_size).to_vec() };
                let uv_data =
                    unsafe { std::slice::from_raw_parts(data_ptr.add(y_size), uv_size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride,
                        },
                        Plane {
                            data: uv_data,
                            stride,
                        },
                    ],
                )
            }
            OutputFormat::Rgb32 => {
                // For RGB32: total_size = stride * height, where stride >= width * 4
                let bytes_per_pixel = 4usize;
                let stride =
                    (current_length as usize / height_usize).max(width_usize * bytes_per_pixel);

                if self.debug_logging {
                    debug!(
                        "RGB32 CPU buffer: {}x{}, stride={}, buffer_len={}",
                        width, height, stride, current_length
                    );
                }

                let size = stride * height_usize;

                // Validate buffer size before creating raw slices to prevent UB
                if (current_length as usize) < size {
                    unsafe {
                        buffer.Unlock().ok();
                    }
                    return Err(VideoError::DecodeFailed(format!(
                        "RGB32 buffer too small: {} bytes, need {} ({}x{}, stride={})",
                        current_length, size, width, height, stride
                    )));
                }

                let data = unsafe { std::slice::from_raw_parts(data_ptr, size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Bgra,
                    width,
                    height,
                    vec![Plane { data, stride }],
                )
            }
        };

        unsafe {
            buffer.Unlock().ok();
        }

        Ok(DecodedFrame::Cpu(frame))
    }

    // ========================================================================
    // Audio Configuration and Reading
    // ========================================================================

    /// Configures the audio stream for PCM output.
    ///
    /// This method:
    /// 1. Selects the first audio stream
    /// 2. Configures it to output uncompressed PCM
    /// 3. Reads the resolved format attributes
    ///
    /// # Arguments
    /// * `reader` - The source reader to configure
    /// * `preferred_sample_rate` - Preferred output sample rate (e.g., 48000)
    /// * `debug_logging` - Enable verbose logging
    ///
    /// # Returns
    /// `Ok(AudioFormatInfo)` on success, `Err(VideoError)` if no audio stream
    /// or configuration fails.
    #[profiling::function]
    fn configure_audio_stream(
        reader: &IMFSourceReader,
        _preferred_sample_rate: u32,
        debug_logging: bool,
    ) -> Result<AudioFormatInfo, VideoError> {
        if debug_logging {
            debug!("Configuring audio stream for PCM output");
        }

        // Enable the audio stream
        unsafe {
            reader
                .SetStreamSelection(MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32, true)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to select audio stream: {}", e))
                })?;
        }

        // Create PCM output type
        let pcm_type: IMFMediaType = unsafe {
            MFCreateMediaType()
                .map_err(|e| VideoError::DecoderInit(format!("MFCreateMediaType failed: {}", e)))?
        };

        unsafe {
            // Set major type to audio
            pcm_type
                .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetGUID major type failed: {}", e))
                })?;

            // Set subtype to PCM (uncompressed)
            pcm_type
                .SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM)
                .map_err(|e| VideoError::DecoderInit(format!("SetGUID subtype failed: {}", e)))?;

            // Note: We don't set sample rate, channels, etc. here.
            // Media Foundation will negotiate based on the source and
            // we'll read the resolved values after SetCurrentMediaType.
        }

        // Set the PCM output type on the audio stream
        unsafe {
            reader
                .SetCurrentMediaType(
                    MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32,
                    None,
                    &pcm_type,
                )
                .map_err(|e| {
                    VideoError::DecoderInit(format!("SetCurrentMediaType for audio failed: {}", e))
                })?;
        }

        // Get the resolved media type to read actual format attributes
        let resolved_type: IMFMediaType = unsafe {
            reader
                .GetCurrentMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("GetCurrentMediaType for audio failed: {}", e))
                })?
        };

        // Read all required audio attributes
        let sample_rate = unsafe {
            let mut val: u32 = 0;
            resolved_type
                .GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, &mut val)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get sample rate: {}", e))
                })?;
            val
        };

        let channels = unsafe {
            let mut val: u32 = 0;
            resolved_type
                .GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, &mut val)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get channel count: {}", e))
                })?;
            val as u16
        };

        let bits_per_sample = unsafe {
            let mut val: u32 = 0;
            resolved_type
                .GetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, &mut val)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get bits per sample: {}", e))
                })?;
            val as u16
        };

        let block_align = unsafe {
            let mut val: u32 = 0;
            resolved_type
                .GetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, &mut val)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get block alignment: {}", e))
                })?;
            val as u16
        };

        let avg_bytes_per_sec = unsafe {
            let mut val: u32 = 0;
            resolved_type
                .GetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, &mut val)
                .map_err(|e| {
                    VideoError::DecoderInit(format!("Failed to get avg bytes per sec: {}", e))
                })?;
            val
        };

        // Check if the resolved format is float (MFAudioFormat_Float) vs integer PCM.
        // We request PCM, but check the resolved type to be defensive.
        let is_float = unsafe {
            let mut subtype = windows::core::GUID::default();
            if resolved_type.GetGUID(&MF_MT_SUBTYPE, &mut subtype).is_ok() {
                subtype == MFAudioFormat_Float
            } else {
                false
            }
        };

        let format = AudioFormatInfo {
            sample_rate,
            channels,
            bits_per_sample,
            block_align,
            avg_bytes_per_sec,
            is_float,
        };

        if debug_logging {
            info!(
                "Audio stream configured: {}Hz, {} channels, {}-bit, block_align={}, avg_bps={}",
                format.sample_rate,
                format.channels,
                format.bits_per_sample,
                format.block_align,
                format.avg_bytes_per_sec
            );
        }

        Ok(format)
    }

    /// Reads and decodes the next audio sample.
    ///
    /// Returns `Ok(Some(AudioFrame))` on success, `Ok(None)` if no sample available
    /// (e.g., stream tick or EOF), `Err` on error.
    ///
    /// # MF Reader Flags Handled
    /// - `MF_SOURCE_READERF_ENDOFSTREAM`: Sets audio_eof and returns None
    /// - `MF_SOURCE_READERF_STREAMTICK`: Returns None (gap in stream)
    /// - `MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED`: Logs warning, continues
    /// - `MF_SOURCE_READERF_NEWSTREAM`: Logs info, continues
    #[profiling::function]
    pub fn read_audio_sample(&mut self) -> Result<Option<AudioFrame>, VideoError> {
        if !self.audio_enabled {
            return Ok(None);
        }

        let audio_format = match &self.audio_format {
            Some(f) => f.clone(),
            None => return Ok(None),
        };

        let mut flags: u32 = 0;
        let mut timestamp: i64 = 0;
        let mut sample: Option<IMFSample> = None;

        unsafe {
            self.source_reader
                .ReadSample(
                    MF_SOURCE_READER_FIRST_AUDIO_STREAM.0 as u32,
                    0,
                    None,
                    Some(&mut flags),
                    Some(&mut timestamp),
                    Some(&mut sample),
                )
                .map_err(|e| VideoError::DecodeFailed(format!("Audio ReadSample failed: {}", e)))?;
        }

        // Handle MF reader flags
        if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 != 0 {
            self.audio_eof.store(true, Ordering::SeqCst);
            if self.debug_logging {
                debug!("Audio end of stream reached");
            }
            return Ok(None);
        }

        if flags & MF_SOURCE_READERF_STREAMTICK.0 != 0 {
            if self.debug_logging {
                debug!("Audio stream tick (gap in stream)");
            }
            return Ok(None);
        }

        if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 != 0 {
            warn!("Audio media type changed mid-stream");
            // Could re-read format here, but for now just log
        }

        if flags & MF_SOURCE_READERF_NEWSTREAM.0 != 0 {
            if self.debug_logging {
                info!("New audio stream detected");
            }
        }

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        // Convert timestamp to Duration (100ns units)
        // Clamp negative timestamps to 0 to avoid u64 wrap
        let pts = Duration::from_nanos(timestamp.max(0) as u64 * 100);

        // Get the buffer from the sample
        let buffer: IMFMediaBuffer = unsafe {
            sample.ConvertToContiguousBuffer().map_err(|e| {
                VideoError::DecodeFailed(format!("Audio ConvertToContiguousBuffer failed: {}", e))
            })?
        };

        // Lock buffer and extract PCM data
        let mut data_ptr: *mut u8 = std::ptr::null_mut();
        let mut current_length: u32 = 0;

        unsafe {
            buffer
                .Lock(&mut data_ptr, None, Some(&mut current_length))
                .map_err(|e| {
                    VideoError::DecodeFailed(format!("Audio buffer Lock failed: {}", e))
                })?;
        }

        // Convert raw bytes to i16 samples
        let byte_slice = unsafe { std::slice::from_raw_parts(data_ptr, current_length as usize) };

        let pcm_data: Vec<i16> = match audio_format.bits_per_sample {
            16 => {
                // PCM 16-bit: 2 bytes per sample, little-endian
                byte_slice
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect()
            }
            24 => {
                // PCM 24-bit: 3 bytes per sample, convert to 16-bit by taking top 16 bits
                byte_slice
                    .chunks_exact(3)
                    .map(|chunk| {
                        // 24-bit little-endian: [low, mid, high]
                        // Take top 16 bits: mid and high
                        i16::from_le_bytes([chunk[1], chunk[2]])
                    })
                    .collect()
            }
            32 => {
                // 32-bit: could be i32 PCM or f32 PCM
                byte_slice
                    .chunks_exact(4)
                    .map(|chunk| {
                        if audio_format.is_float {
                            // 32-bit float: convert f32 [-1.0, 1.0] to i16
                            let sample =
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            // Clamp to [-1.0, 1.0] and scale to i16 range
                            let clamped = sample.clamp(-1.0, 1.0);
                            (clamped * i16::MAX as f32) as i16
                        } else {
                            // 32-bit integer PCM: take top 16 bits
                            let sample =
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            (sample >> 16) as i16
                        }
                    })
                    .collect()
            }
            _ => {
                warn!(
                    "Unsupported audio bit depth: {}, returning silence",
                    audio_format.bits_per_sample
                );
                Vec::new()
            }
        };

        unsafe {
            buffer.Unlock().ok();
        }

        if self.debug_logging {
            debug!(
                "Audio frame: pts={:?}, samples={}, channels={}",
                pts,
                pcm_data.len() / audio_format.channels as usize,
                audio_format.channels
            );
        }

        Ok(Some(AudioFrame::new(
            pts,
            pcm_data,
            audio_format.channels,
            audio_format.sample_rate,
        )))
    }

    /// Returns whether audio is enabled and configured.
    pub fn has_audio(&self) -> bool {
        self.audio_enabled
    }

    /// Returns the audio format info if available.
    pub fn audio_format(&self) -> Option<&AudioFormatInfo> {
        self.audio_format.as_ref()
    }

    /// Returns whether audio end-of-stream has been reached.
    pub fn is_audio_eof(&self) -> bool {
        self.audio_eof.load(Ordering::SeqCst)
    }

    /// Resets audio EOF flag (used after seek).
    pub fn reset_audio_eof(&self) {
        self.audio_eof.store(false, Ordering::SeqCst);
    }

    /// Extracts frame from IMF2DBuffer2 with proper stride handling.
    fn extract_from_2d_buffer(
        &self,
        buffer_2d: &IMF2DBuffer2,
        width: u32,
        height: u32,
    ) -> Result<DecodedFrame, VideoError> {
        if self.debug_logging {
            debug!(
                "Using IMF2DBuffer2 for proper stride handling, format={:?}",
                self.output_format
            );
        }

        let mut scanline0: *mut u8 = std::ptr::null_mut();
        let mut pitch: i32 = 0;
        let mut buffer_start: *mut u8 = std::ptr::null_mut();
        let mut buffer_length: u32 = 0;

        unsafe {
            buffer_2d
                .Lock2DSize(
                    windows::Win32::Media::MediaFoundation::MF2DBuffer_LockFlags_Read,
                    &mut scanline0,
                    &mut pitch,
                    &mut buffer_start,
                    &mut buffer_length,
                )
                .map_err(|e| VideoError::DecodeFailed(format!("Lock2DSize failed: {}", e)))?;
        }

        let stride = pitch.unsigned_abs() as usize;
        let height_usize = height as usize;

        if self.debug_logging {
            debug!(
                "IMF2DBuffer2: {}x{}, pitch={}, buffer_len={}, format={:?}",
                width, height, pitch, buffer_length, self.output_format
            );
        }

        let frame = match self.output_format {
            OutputFormat::Nv12 => {
                let uv_height = (height_usize + 1) / 2;
                let y_size = stride * height_usize;
                let uv_size = stride * uv_height;
                let required_size = y_size + uv_size;

                // Validate buffer size before creating raw slices to prevent UB
                if (buffer_length as usize) < required_size {
                    unsafe {
                        buffer_2d.Unlock2D().ok();
                    }
                    return Err(VideoError::DecodeFailed(format!(
                        "NV12 2D buffer too small: {} bytes, need {} ({}x{}, stride={})",
                        buffer_length, required_size, width, height, stride
                    )));
                }

                let y_data = unsafe { std::slice::from_raw_parts(scanline0, y_size).to_vec() };
                let uv_data =
                    unsafe { std::slice::from_raw_parts(scanline0.add(y_size), uv_size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Nv12,
                    width,
                    height,
                    vec![
                        Plane {
                            data: y_data,
                            stride,
                        },
                        Plane {
                            data: uv_data,
                            stride,
                        },
                    ],
                )
            }
            OutputFormat::Rgb32 => {
                let size = stride * height_usize;

                // Validate buffer size before creating raw slices to prevent UB
                if (buffer_length as usize) < size {
                    unsafe {
                        buffer_2d.Unlock2D().ok();
                    }
                    return Err(VideoError::DecodeFailed(format!(
                        "RGB32 2D buffer too small: {} bytes, need {} ({}x{}, stride={})",
                        buffer_length, size, width, height, stride
                    )));
                }

                let data = unsafe { std::slice::from_raw_parts(scanline0, size).to_vec() };

                CpuFrame::new(
                    PixelFormat::Bgra,
                    width,
                    height,
                    vec![Plane { data, stride }],
                )
            }
        };

        unsafe {
            buffer_2d.Unlock2D().ok();
        }

        Ok(DecodedFrame::Cpu(frame))
    }
}

impl VideoDecoderBackend for WindowsVideoDecoder {
    #[profiling::function]
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        // Debug logging controlled via environment variable.
        // Default ON for testing phase. Set NOTEDECK_VIDEO_DEBUG=0 to disable.
        let debug_logging = std::env::var("NOTEDECK_VIDEO_DEBUG")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        Self::new(url, debug_logging)
    }

    #[profiling::function]
    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        self.read_sample()
    }

    #[profiling::function]
    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        if self.debug_logging {
            debug!("Seeking to {:?}", position);
        }

        // Convert Duration to 100ns units for Media Foundation
        let position_100ns = position.as_nanos() as i64 / 100;

        // Create PROPVARIANT for seek position.
        // Media Foundation expects VT_I8 (64-bit signed integer) in 100ns units.
        // The PROPVARIANT union layout is:
        //   - Anonymous.Anonymous.vt: variant type tag (VT_I8 = 20)
        //   - Anonymous.Anonymous.Anonymous.hVal: i64 value for VT_I8
        //
        // We use default() for safe initialization, then set fields explicitly
        // to avoid UB from uninitialized union fields.
        let mut prop_variant =
            windows::Win32::System::Com::StructuredStorage::PROPVARIANT::default();

        unsafe {
            // Access the inner union fields through the Anonymous chain.
            // This is the documented structure of PROPVARIANT:
            // PROPVARIANT { Anonymous: PROPVARIANT_0 { Anonymous: PROPVARIANT_0_0 { vt, ..., Anonymous: PROPVARIANT_0_0_0 { hVal, ... } } } }
            prop_variant.Anonymous.Anonymous.vt = windows::Win32::System::Variant::VT_I8;
            // hVal is the LARGE_INTEGER field for VT_I8, which is equivalent to i64
            prop_variant.Anonymous.Anonymous.Anonymous.hVal = position_100ns;
        }

        unsafe {
            self.source_reader
                .SetCurrentPosition(&windows::core::GUID::zeroed(), &prop_variant)
                .map_err(|e| VideoError::SeekFailed(format!("SetCurrentPosition failed: {}", e)))?;
        }

        self.position = position;
        self.eof.store(false, Ordering::SeqCst);
        // Reset audio EOF flag on seek
        self.audio_eof.store(false, Ordering::SeqCst);

        if self.debug_logging {
            debug!("Seek completed to {:?}", position);
        }

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.metadata
    }

    fn is_eof(&self) -> bool {
        self.eof.load(Ordering::SeqCst)
    }

    /// Windows Media Foundation handles audio internally - no separate FFmpeg audio thread needed.
    fn handles_audio_internally(&self) -> bool {
        true
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.hw_accel
    }

    /// Windows Media Foundation handles audio internally with its own A/V sync.
    fn handles_audio_internally(&self) -> bool {
        true
    }
}

impl Drop for WindowsVideoDecoder {
    fn drop(&mut self) {
        if self.debug_logging {
            debug!("WindowsVideoDecoder dropping, cleaning up");
        }

        // Log zero-copy stats
        #[cfg(feature = "zero-copy")]
        {
            let fallback_count = self.cpu_fallback_count.load(Ordering::Relaxed);
            if fallback_count > 0 {
                info!(
                    "WindowsVideoDecoder zero-copy stats: {} frames used CPU fallback \
                     (zero-copy awaits wgpu ExternalTexture API)",
                    fallback_count
                );
            }
        }

        // Release staging texture
        self.staging_texture = None;

        // Close shared handle to prevent NT handle leak
        #[cfg(feature = "zero-copy")]
        if let Some(handle) = self.shared_handle.take() {
            if !handle.is_invalid() {
                unsafe {
                    let _ = windows::Win32::Foundation::CloseHandle(handle);
                }
                if self.debug_logging {
                    debug!("Closed shared NT handle");
                }
            }
        }

        // Note: MFShutdown is handled by the _mf_guard Arc<MfGuard>.
        // It will only call MFShutdown when the last decoder is dropped.

        // Note: COM uninitialization is handled by the _com_guard.
        // Since it's declared last, it will be dropped last after all COM objects.

        if self.debug_logging {
            info!("WindowsVideoDecoder cleanup complete");
        }
    }
}

// Note: WindowsVideoDecoder is NOT Send because COM objects are thread-affine.
// The decoder must be created and used on the same thread (the decode thread).
// Use DecodeThread::new_from_factory to ensure proper thread confinement.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_type() {
        assert_eq!(HwAccelType::platform_default(), HwAccelType::D3d11va);
    }
}
