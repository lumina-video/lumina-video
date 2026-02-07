//! Zero-copy GPU texture import for video frames.
//!
//! This module enables importing video decoder output directly into wgpu textures
//! without CPU memory copies, using platform-specific GPU interop:
//!
//! - **macOS**: IOSurface → Metal texture → wgpu
//! - **Linux**: DMABuf → Vulkan image → wgpu
//! - **Android**: AHardwareBuffer → Vulkan image → wgpu
//! - **Windows**: D3D11 shared handle → D3D12 → wgpu
//!
//! # Current Implementation Status
//!
//! - **macOS**: Fully implemented. IOSurface import to Metal works for supported formats (BGRA, NV12).
//! - **Linux**: Partial. Only single-plane formats (RGBA/BGRA) work. NV12 falls back
//!   to CPU copy because `LinuxGpuSurface` only stores single-plane metadata (see
//!   lumina-video-4m8 for the multi-plane limitation).
//! - **Android**: Rust-side Vulkan import is ready. Waiting for Java/Kotlin ExoPlayer
//!   integration to expose AHardwareBuffer via ImageReader (see lumina-video-6dn).
//! - **Windows**: Implemented but missing proper D3D11→D3D12 fence synchronization.
//!   May cause visual artifacts or crashes under heavy load without sync primitives.
//!
//! # Platform Support
//!
//! | Platform | Backend | Import Method | Status |
//! |----------|---------|---------------|--------|
//! | macOS | Metal | IOSurface | Supported |
//! | Linux | Vulkan | DMABuf (VA-API, V4L2) | Partial (single-plane only) |
//! | Android | Vulkan | AHardwareBuffer (MediaCodec) | Rust ready, Java pending |
//! | Windows | D3D12 | D3D11 Shared Handle | Partial (missing fence sync) |
//! | iOS | Metal | IOSurface | Not yet implemented |
//! | Web/WASM | WebGPU | N/A | Not supported (no external memory) |
//!
//! # Feature Flag
//!
//! The `zero-copy` feature is enabled by default on supported platforms
//! (macOS, Linux, Android, Windows). No additional configuration is needed:
//!
//! ```toml
//! [dependencies]
//! lumina-video = { version = "..." }
//! ```
//!
//! To disable zero-copy (e.g., for unsupported platforms like iOS or WASM),
//! use `default-features = false`:
//!
//! ```toml
//! [dependencies]
//! lumina-video = { version = "...", default-features = false, features = ["macos-native-video"] }
//! ```
//!
//! **Note:** On unsupported platforms (iOS, WASM, etc.), compilation will fail
//! with a clear error message about platform support if zero-copy is enabled.
//!
//! # How It Works
//!
//! wgpu-hal v24+ provides `Device::texture_from_raw()` on each backend:
//!
//! 1. Get the raw HAL device via `device.as_hal::<Metal>()`
//! 2. Create a native texture from the external resource
//! 3. Wrap with `Device::texture_from_raw()` to get a HAL Texture
//! 4. Call `device.create_texture_from_hal()` to get a wgpu::Texture
//!
//! # References
//!
//! - wgpu #2320: Texture memory import API
//! - wgpu #4067: Proposal for API interoperability
//! - wgpu PR #6161: D3D11 shared handle import (merged)

use std::fmt;

// =============================================================================
// Platform Support Verification
// =============================================================================
//
// The zero-copy feature requires platform-specific GPU interop APIs that are
// only available on certain platforms. This section provides compile-time
// verification with clear error messages for unsupported configurations.

/// Compile-time check for iOS - not yet implemented
#[cfg(target_os = "ios")]
compile_error!(
    "The `zero-copy` feature is not yet supported on iOS. \
     While IOSurface import is technically possible via Metal, the implementation \
     has not been completed. Please disable the `zero-copy` feature on iOS, \
     or contribute an implementation! See the macOS module for reference."
);

/// Compile-time check for WebAssembly - not supported
#[cfg(target_family = "wasm")]
compile_error!(
    "The `zero-copy` feature is not supported on WebAssembly/WASM. \
     WebGPU does not provide external memory import APIs, so zero-copy \
     texture import is not possible. Please disable the `zero-copy` feature \
     when targeting WASM."
);

/// Compile-time check for other unsupported Unix platforms (FreeBSD, OpenBSD, etc.)
#[cfg(all(
    target_family = "unix",
    not(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        target_os = "ios"
    ))
))]
compile_error!(
    "Zero-copy video is not supported on this Unix platform. \
     Currently supported platforms: macOS (Metal/IOSurface), Linux (Vulkan/DMABuf), \
     Android (Vulkan/AHardwareBuffer). Please contribute an implementation for your platform."
);

/// Returns whether zero-copy import is supported on the current platform.
///
/// This is a compile-time constant that can be used for conditional logic.
/// On unsupported platforms with the `zero-copy` feature enabled, compilation
/// will fail with a clear error message before this function is ever called.
///
/// # Supported Platforms
///
/// - macOS: Metal backend with IOSurface
/// - Linux: Vulkan backend with DMABuf
/// - Android: Vulkan backend with AHardwareBuffer
/// - Windows: D3D12 backend with D3D11 shared handles
pub const fn is_platform_supported() -> bool {
    cfg!(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        target_os = "windows"
    ))
}

/// Error type for zero-copy texture import operations.
///
/// This error is returned when zero-copy texture import fails. Each variant
/// provides specific information about what went wrong.
///
/// # Example
///
/// ```ignore
/// use lumina_video::media::zero_copy::ZeroCopyError;
///
/// fn handle_import_error(err: ZeroCopyError) {
///     match err {
///         ZeroCopyError::NotAvailable(reason) => println!("Fallback needed: {}", reason),
///         ZeroCopyError::TextureCreationFailed(msg) => println!("GPU error: {}", msg),
///         ZeroCopyError::UnsupportedBackend(msg) => println!("Wrong backend: {}", msg),
///         ZeroCopyError::HalAccessFailed(msg) => println!("HAL error: {}", msg),
///         ZeroCopyError::InvalidResource(msg) => println!("Invalid resource: {}", msg),
///         ZeroCopyError::FormatMismatch(msg) => println!("Format error: {}", msg),
///     }
/// }
/// ```
#[derive(Debug)]

pub enum ZeroCopyError {
    /// The wgpu device doesn't support the required backend
    UnsupportedBackend(String),
    /// Failed to access the HAL device
    HalAccessFailed(String),
    /// The external resource is invalid or incompatible
    InvalidResource(String),
    /// Texture creation failed
    TextureCreationFailed(String),
    /// Feature not available on this platform
    NotAvailable(String),
    /// Resource format doesn't match expected format
    FormatMismatch(String),
    /// Import operation failed
    ImportFailed(String),
    /// GPU device is busy (lock contention) - caller should retry or use cached frame
    DeviceBusy,
}

impl fmt::Display for ZeroCopyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeroCopyError::UnsupportedBackend(msg) => write!(f, "Unsupported backend: {}", msg),
            ZeroCopyError::HalAccessFailed(msg) => write!(f, "HAL access failed: {}", msg),
            ZeroCopyError::InvalidResource(msg) => write!(f, "Invalid resource: {}", msg),
            ZeroCopyError::TextureCreationFailed(msg) => {
                write!(f, "Texture creation failed: {}", msg)
            }
            ZeroCopyError::NotAvailable(msg) => write!(f, "Not available: {}", msg),
            ZeroCopyError::FormatMismatch(msg) => write!(f, "Format mismatch: {}", msg),
            ZeroCopyError::ImportFailed(msg) => write!(f, "Import failed: {}", msg),
            ZeroCopyError::DeviceBusy => write!(f, "GPU device busy (lock contention)"),
        }
    }
}

impl std::error::Error for ZeroCopyError {}

/// Statistics for zero-copy operations.
///
/// Tracks the number of frames processed via zero-copy vs fallback paths,
/// useful for performance monitoring and debugging.
///
/// # Example
///
/// ```ignore
/// use lumina_video::media::zero_copy::ZeroCopyStats;
///
/// let stats = ZeroCopyStats {
///     total_frames: 1000,
///     zero_copy_frames: 985,
///     fallback_frames: 15,
/// };
///
/// println!("Zero-copy efficiency: {:.1}%", stats.zero_copy_percentage());
/// // Output: "Zero-copy efficiency: 98.5%"
/// ```
#[derive(Debug, Clone, Default)]

pub struct ZeroCopyStats {
    /// Total frames processed
    pub total_frames: u64,
    /// Frames imported via zero-copy path
    pub zero_copy_frames: u64,
    /// Frames that fell back to CPU copy
    pub fallback_frames: u64,
}

impl ZeroCopyStats {
    /// Returns the percentage of frames imported via zero-copy path (0.0 - 100.0).
    ///
    /// This is useful for monitoring zero-copy effectiveness and identifying
    /// when fallback paths are being used. A low percentage may indicate:
    /// - Unsupported pixel format from the decoder
    /// - Missing GPU extensions (e.g., DMABuf import on Linux)
    /// - Driver limitations
    ///
    /// Returns `0.0` if no frames have been processed yet.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stats = decoder.zero_copy_stats();
    /// println!("Zero-copy usage: {:.1}%", stats.zero_copy_percentage());
    /// // Output: "Zero-copy usage: 98.5%"
    /// ```
    pub fn zero_copy_percentage(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.zero_copy_frames as f64 / self.total_frames as f64) * 100.0
    }
}

// =============================================================================
// macOS: IOSurface → Metal → wgpu
// =============================================================================

/// macOS-specific zero-copy import via IOSurface and Metal.
///
/// This module provides functions to import IOSurface-backed video frames
/// directly into wgpu textures without CPU memory copies.
///
/// # Requirements
///
/// - macOS 10.11+ (for Metal support)
/// - wgpu using Metal backend
/// - Video frames backed by IOSurface (e.g., from VideoToolbox)
///
/// # Example
///
/// ```ignore
/// use lumina_video::media::zero_copy::macos;
///
/// // Check if Metal backend is available
/// if macos::is_metal_backend(&device) {
///     // Import IOSurface from VideoToolbox decoder
///     let texture = unsafe {
///         macos::import_iosurface(&device, io_surface, width, height, format)?
///     };
/// }
/// ```
#[cfg(target_os = "macos")]
pub mod macos {
    use super::ZeroCopyError;
    use metal::foreign_types::ForeignType;
    use metal::objc::runtime::Object;
    use metal::objc::*;
    use std::ffi::c_void;
    use tracing::{debug, info, warn};

    /// Opaque handle to an IOSurface (from CoreVideo).
    /// This is obtained from CVPixelBufferGetIOSurface().
    pub type IOSurfaceRef = *mut c_void;

    /// Checks if the current wgpu device supports Metal backend.
    ///
    /// This verifies that the HAL APIs are accessible for zero-copy import.
    pub fn is_metal_backend(device: &wgpu::Device) -> bool {
        unsafe {
            device.as_hal::<wgpu::hal::api::Metal, _, bool>(|hal_device| hal_device.is_some())
        }
    }

    /// Gets information about the Metal device for diagnostics.
    ///
    /// Returns the device name if Metal backend is available, None otherwise.
    pub fn get_metal_device_info(device: &wgpu::Device) -> Option<String> {
        unsafe {
            device.as_hal::<wgpu::hal::api::Metal, _, Option<String>>(|hal_device| {
                hal_device.map(|d| {
                    let metal_device = d.raw_device();
                    let guard = metal_device.lock();
                    guard.name().to_string()
                })
            })
        }
    }

    /// Imports an IOSurface into wgpu as a texture (zero-copy).
    ///
    /// This function creates a wgpu::Texture that directly references the IOSurface's
    /// GPU memory, enabling zero-copy video frame display.
    ///
    /// # Safety
    ///
    /// - `io_surface` must be a valid IOSurfaceRef obtained from CVPixelBufferGetIOSurface()
    /// - The IOSurface must remain valid for the lifetime of the returned texture
    /// - The caller must retain the CVPixelBuffer that owns the IOSurface
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device (must be Metal backend)
    /// * `io_surface` - A valid IOSurfaceRef
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `format` - The wgpu texture format (should match IOSurface pixel format)
    ///
    /// # Returns
    ///
    /// A wgpu::Texture that references the IOSurface memory directly.
    pub unsafe fn import_iosurface(
        device: &wgpu::Device,
        io_surface: IOSurfaceRef,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::Texture, ZeroCopyError> {
        if io_surface.is_null() {
            return Err(ZeroCopyError::InvalidResource(
                "IOSurface is null".to_string(),
            ));
        }

        // Access the Metal HAL device and create the texture
        let hal_texture_result = device
            .as_hal::<wgpu::hal::api::Metal, _, Result<wgpu::hal::metal::Texture, ZeroCopyError>>(
                |hal_device| {
                    let Some(hal_device) = hal_device else {
                        warn!("Failed to get Metal HAL device");
                        return Err(ZeroCopyError::HalAccessFailed(
                            "wgpu not using Metal backend".to_string(),
                        ));
                    };

                    // Get the raw Metal device
                    let metal_device = hal_device.raw_device();
                    let metal_device_guard = metal_device.lock();

                    debug!(
                        "Creating Metal texture from IOSurface via objc2 ({}x{} {:?}) on {}",
                        width,
                        height,
                        format,
                        metal_device_guard.name()
                    );

                    // Create Metal texture descriptor
                    let descriptor = metal::TextureDescriptor::new();
                    descriptor.set_texture_type(metal::MTLTextureType::D2);
                    let metal_format = wgpu_format_to_metal(format)?;
                    descriptor.set_pixel_format(metal_format);
                    descriptor.set_width(width as u64);
                    descriptor.set_height(height as u64);
                    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
                    // Note: For IOSurface-backed textures created via newTextureWithDescriptor:iosurface:plane:,
                    // Metal ignores the storage mode in the descriptor - the IOSurface dictates memory layout.
                    // We set Managed here as the nominal value for macOS texture compatibility.
                    descriptor.set_storage_mode(metal::MTLStorageMode::Managed);

                    // Get raw pointers for objc msg_send
                    let device_ptr = metal_device_guard.as_ptr();
                    let descriptor_ptr = descriptor.as_ptr();

                    // Call [MTLDevice newTextureWithDescriptor:iosurface:plane:]
                    // This is the key method that metal-rs doesn't expose
                    // Using objc crate's msg_send! (same as metal-rs uses internally)
                    let texture_ptr: *mut Object = msg_send![
                        device_ptr,
                        newTextureWithDescriptor: descriptor_ptr
                        iosurface: io_surface
                        plane: 0usize  // plane 0 for single-plane formats like BGRA
                    ];

                    if texture_ptr.is_null() {
                        warn!("Metal newTextureWithDescriptor:iosurface:plane: returned null");
                        return Err(ZeroCopyError::TextureCreationFailed(
                            "Metal failed to create texture from IOSurface".to_string(),
                        ));
                    }

                    // Wrap the raw pointer as a metal::Texture
                    // The texture is retained by msg_send (returns +1 retain count)
                    let metal_texture =
                        metal::Texture::from_ptr(texture_ptr as *mut metal::MTLTexture);

                    info!(
                        "Created Metal texture from IOSurface: {}x{} {:?}",
                        width, height, format
                    );

                    // Wrap as wgpu_hal::metal::Texture using the existing API
                    let hal_texture = wgpu::hal::metal::Device::texture_from_raw(
                        metal_texture,
                        format,
                        metal::MTLTextureType::D2,
                        1, // array_layers
                        1, // mip_levels
                        wgpu::hal::CopyExtent {
                            width,
                            height,
                            depth: 1,
                        },
                    );

                    Ok(hal_texture)
                },
            );

        // Get the HAL texture from the closure result
        let hal_texture = hal_texture_result?;

        // Create wgpu texture descriptor
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("zero-copy IOSurface texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // Wrap the HAL texture as a wgpu::Texture
        let wgpu_texture =
            device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &texture_desc);

        info!("Successfully imported IOSurface as wgpu texture (zero-copy)");

        Ok(wgpu_texture)
    }

    /// Converts wgpu TextureFormat to Metal MTLPixelFormat.
    ///
    /// Returns an error for unsupported formats instead of silently defaulting.
    pub fn wgpu_format_to_metal(
        format: wgpu::TextureFormat,
    ) -> Result<metal::MTLPixelFormat, ZeroCopyError> {
        match format {
            wgpu::TextureFormat::Bgra8Unorm => Ok(metal::MTLPixelFormat::BGRA8Unorm),
            wgpu::TextureFormat::Rgba8Unorm => Ok(metal::MTLPixelFormat::RGBA8Unorm),
            wgpu::TextureFormat::R8Unorm => Ok(metal::MTLPixelFormat::R8Unorm),
            wgpu::TextureFormat::Rg8Unorm => Ok(metal::MTLPixelFormat::RG8Unorm),
            wgpu::TextureFormat::Bgra8UnormSrgb => Ok(metal::MTLPixelFormat::BGRA8Unorm_sRGB),
            wgpu::TextureFormat::Rgba8UnormSrgb => Ok(metal::MTLPixelFormat::RGBA8Unorm_sRGB),
            _ => {
                warn!("Unsupported texture format {:?}; refusing import", format);
                Err(ZeroCopyError::InvalidResource(format!(
                    "Unsupported texture format {:?}",
                    format
                )))
            }
        }
    }
}

// =============================================================================
// Linux: DMABuf → Vulkan → wgpu
// =============================================================================

/// Linux-specific zero-copy import via DMABuf and Vulkan.
///
/// This module provides functions to import DMABuf file descriptors
/// directly into wgpu textures without CPU memory copies.
///
/// # Requirements
///
/// - Linux kernel with DMABuf support
/// - wgpu using Vulkan backend
/// - Vulkan extensions: `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`
/// - Video frames as DMABuf fd (e.g., from VA-API, V4L2, GStreamer)
///
/// # Supported DRM Modifiers
///
/// The module supports both linear and tiled memory layouts via DRM format modifiers:
/// - `DRM_FORMAT_MOD_LINEAR` - Universal, works everywhere
/// - Intel: X-tiled, Y-tiled, Tile4
/// - AMD: GFX9 64KB tiled
/// - NVIDIA: Block linear
///
/// # Example
///
/// ```ignore
/// use lumina_video::media::zero_copy::linux::{self, DmaBufHandle, drm_modifiers};
///
/// let dmabuf = DmaBufHandle {
///     fd: va_surface_fd,
///     size: width * height * 4,
///     offset: 0,
///     stride: width * 4,
///     modifier: drm_modifiers::DRM_FORMAT_MOD_LINEAR,
/// };
///
/// let texture = unsafe {
///     linux::import_dmabuf(&device, dmabuf, width, height, format)?
/// };
/// ```
#[cfg(target_os = "linux")]
pub mod linux {
    use super::ZeroCopyError;
    use ash::vk;
    use std::ffi::CStr;
    use std::os::fd::RawFd;
    use tracing::{debug, info, warn};

    /// Per-plane DMABuf metadata for multi-plane import.
    ///
    /// Multi-plane formats like NV12 require separate metadata for each plane.
    /// This struct mirrors `DmaBufPlane` from video.rs for use in the import API.
    #[derive(Debug, Clone, Copy)]
    pub struct DmaBufPlaneHandle {
        /// The file descriptor for this plane.
        /// Multiple planes may share the same fd (with different offsets) or have separate fds.
        pub fd: RawFd,
        /// Offset within the buffer where this plane starts (bytes)
        pub offset: u64,
        /// Row pitch/stride in bytes for this plane
        pub stride: u32,
        /// Size of this plane's data in bytes (may be 0 if unknown)
        pub size: u64,
    }

    /// DMABuf handle for zero-copy import with multi-plane support.
    ///
    /// DMABuf (DMA Buffer Sharing) is a Linux kernel mechanism for sharing
    /// memory buffers between devices. Video decoders like VA-API and V4L2
    /// export decoded frames as DMABuf file descriptors.
    ///
    /// # Multi-Plane Support (lumina-video-s0e)
    ///
    /// This struct now supports multi-plane formats like NV12 and YUV420p:
    /// - **Single-plane (RGBA/BGRA)**: Use `DmaBufHandle::single_plane()`
    /// - **Multi-plane (NV12)**: Use `DmaBufHandle::new()` with plane metadata
    ///
    /// Note: Actual Vulkan multi-plane import requires VkSamplerYcbcrConversion
    /// which is not yet implemented. The current import_dmabuf() only handles
    /// single-plane formats. Multi-plane metadata is preserved for future use.
    #[derive(Debug, Clone)]
    pub struct DmaBufHandle {
        /// Per-plane metadata. For single-plane formats, this has length 1.
        pub planes: Vec<DmaBufPlaneHandle>,
        /// DRM format modifier (e.g., I915_FORMAT_MOD_Y_TILED)
        /// Use DRM_FORMAT_MOD_LINEAR (0) for linear layout
        pub modifier: u64,
    }

    impl DmaBufHandle {
        /// Creates a multi-plane DmaBufHandle.
        pub fn new(planes: Vec<DmaBufPlaneHandle>, modifier: u64) -> Self {
            Self { planes, modifier }
        }

        /// Creates a single-plane DmaBufHandle (convenience for RGBA/BGRA).
        pub fn single_plane(fd: RawFd, size: u64, offset: u64, stride: u32, modifier: u64) -> Self {
            Self {
                planes: vec![DmaBufPlaneHandle {
                    fd,
                    offset,
                    stride,
                    size,
                }],
                modifier,
            }
        }

        /// Returns the primary file descriptor (plane 0).
        pub fn fd(&self) -> RawFd {
            self.planes.first().map(|p| p.fd).unwrap_or(-1)
        }

        /// Returns the primary plane's size.
        pub fn size(&self) -> u64 {
            self.planes.first().map(|p| p.size).unwrap_or(0)
        }

        /// Returns the primary plane's offset.
        pub fn offset(&self) -> u64 {
            self.planes.first().map(|p| p.offset).unwrap_or(0)
        }

        /// Returns the primary plane's stride.
        pub fn stride(&self) -> u32 {
            self.planes.first().map(|p| p.stride).unwrap_or(0)
        }

        /// Returns the number of planes.
        pub fn num_planes(&self) -> usize {
            self.planes.len()
        }

        /// Returns true if this is a multi-plane format.
        pub fn is_multi_plane(&self) -> bool {
            self.planes.len() > 1
        }
    }

    /// Extension names required for DMABuf import
    const EXT_EXTERNAL_MEMORY_DMA_BUF: &CStr = c"VK_EXT_external_memory_dma_buf";
    const KHR_EXTERNAL_MEMORY_FD: &CStr = c"VK_KHR_external_memory_fd";
    const EXT_IMAGE_DRM_FORMAT_MODIFIER: &CStr = c"VK_EXT_image_drm_format_modifier";

    /// Checks if the wgpu device is using the Vulkan backend.
    ///
    /// DMABuf zero-copy import requires Vulkan. This function checks whether
    /// the wgpu device was created with the Vulkan backend, which is necessary
    /// for using `VK_EXT_external_memory_dma_buf` and related extensions.
    ///
    /// Returns `false` if using OpenGL, software rendering, or another backend.
    pub fn is_vulkan_backend(device: &wgpu::Device) -> bool {
        unsafe {
            device.as_hal::<wgpu::hal::api::Vulkan, _, bool>(|hal_device| hal_device.is_some())
        }
    }

    /// Gets information about the Vulkan device for diagnostics.
    ///
    /// Returns the device name if Vulkan backend is available, None otherwise.
    pub fn get_vulkan_device_info(device: &wgpu::Device) -> Option<String> {
        unsafe {
            device.as_hal::<wgpu::hal::api::Vulkan, _, Option<String>>(|hal_device| {
                hal_device.map(|d| {
                    let instance = d.shared_instance();
                    let raw_instance = instance.raw_instance();
                    let physical_device = d.raw_physical_device();

                    let properties = raw_instance.get_physical_device_properties(physical_device);
                    let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned();
                    format!(
                        "{} (Vulkan {}.{}.{})",
                        device_name,
                        vk::api_version_major(properties.api_version),
                        vk::api_version_minor(properties.api_version),
                        vk::api_version_patch(properties.api_version)
                    )
                })
            })
        }
    }

    /// Checks if Vulkan DMABuf import extensions are available.
    ///
    /// This checks for VK_EXT_external_memory_dma_buf extension support.
    /// The extension allows importing Linux DMABuf file descriptors as
    /// Vulkan external memory.
    ///
    /// # Required Extensions
    ///
    /// - VK_KHR_external_memory (Vulkan 1.1 core)
    /// - VK_KHR_external_memory_fd
    /// - VK_EXT_external_memory_dma_buf
    /// - VK_EXT_image_drm_format_modifier (for tiled formats)
    pub fn is_dmabuf_import_available(device: &wgpu::Device) -> bool {
        unsafe {
            device.as_hal::<wgpu::hal::api::Vulkan, _, bool>(|hal_device| {
                let Some(hal_device) = hal_device else {
                    return false;
                };

                let extensions = hal_device.enabled_device_extensions();
                let has_dma_buf = extensions.contains(&EXT_EXTERNAL_MEMORY_DMA_BUF);
                let has_fd = extensions.contains(&KHR_EXTERNAL_MEMORY_FD);

                has_dma_buf && has_fd
            })
        }
    }

    /// Finds a suitable memory type index for the given requirements.
    fn find_memory_type_index(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        type_bits_req: u32,
        flags_req: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        let mem_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        for i in 0..mem_properties.memory_type_count {
            let type_bits = 1 << i;
            let is_required_type = type_bits_req & type_bits != 0;
            // Use safe .get() to avoid potential panic if memory_type_count mismatches array
            let has_required_flags = mem_properties
                .memory_types
                .get(i as usize)
                .map(|mt| mt.property_flags & flags_req == flags_req)
                .unwrap_or(false);

            if is_required_type && has_required_flags {
                return Some(i);
            }
        }
        None
    }

    /// Executes a one-shot command buffer to transition an image layout and perform
    /// queue family ownership transfer for externally imported memory.
    ///
    /// This is required for external memory imports (DMABuf) to ensure proper
    /// synchronization and layout before the image can be sampled in shaders.
    ///
    /// # Safety
    ///
    /// - `vk_device` must be a valid Vulkan device handle
    /// - `vk_queue` must be a valid queue from the same device
    /// - `vk_image` must be a valid image created on this device
    /// - `queue_family_index` must be the queue family index of `vk_queue`
    unsafe fn transition_image_layout_external(
        vk_device: &ash::Device,
        vk_queue: vk::Queue,
        queue_family_index: u32,
        vk_image: vk::Image,
    ) -> Result<(), ZeroCopyError> {
        // Create a one-shot command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        let command_pool = vk_device
            .create_command_pool(&pool_info, None)
            .map_err(|e| {
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to create command pool for layout transition: {:?}",
                    e
                ))
            })?;

        // Allocate a command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = vk_device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| {
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to allocate command buffer for layout transition: {:?}",
                    e
                ))
            })?;
        let cmd_buf = *command_buffers.first().ok_or_else(|| {
            vk_device.destroy_command_pool(command_pool, None);
            ZeroCopyError::TextureCreationFailed(
                "No command buffer returned from allocation".to_string(),
            )
        })?;

        // Begin the command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        vk_device
            .begin_command_buffer(cmd_buf, &begin_info)
            .map_err(|e| {
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to begin command buffer: {:?}",
                    e
                ))
            })?;

        // Record the image memory barrier for layout transition and queue ownership transfer
        // VK_QUEUE_FAMILY_EXTERNAL is defined as (~1U) = 0xFFFFFFFE
        const VK_QUEUE_FAMILY_EXTERNAL: u32 = !1u32;

        let image_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ) // External memory may have been read
            .dst_access_mask(vk::AccessFlags::SHADER_READ) // Will be sampled in fragment shader
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(VK_QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(queue_family_index)
            .image(vk_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        vk_device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::TOP_OF_PIPE, // No prior stage within this queue
            vk::PipelineStageFlags::FRAGMENT_SHADER, // Will be used in fragment shader
            vk::DependencyFlags::empty(),
            &[], // No memory barriers
            &[], // No buffer barriers
            &[image_barrier],
        );

        // End and submit the command buffer
        vk_device.end_command_buffer(cmd_buf).map_err(|e| {
            vk_device.destroy_command_pool(command_pool, None);
            ZeroCopyError::TextureCreationFailed(format!("Failed to end command buffer: {:?}", e))
        })?;

        let cmd_bufs = [cmd_buf];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

        // Create a fence to wait for completion
        let fence_info = vk::FenceCreateInfo::default();
        let fence = vk_device.create_fence(&fence_info, None).map_err(|e| {
            vk_device.destroy_command_pool(command_pool, None);
            ZeroCopyError::TextureCreationFailed(format!(
                "Failed to create fence for layout transition: {:?}",
                e
            ))
        })?;

        vk_device
            .queue_submit(vk_queue, &[submit_info], fence)
            .map_err(|e| {
                vk_device.destroy_fence(fence, None);
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to submit layout transition command: {:?}",
                    e
                ))
            })?;

        // Wait for the command to complete (with 1 second timeout)
        vk_device
            .wait_for_fences(&[fence], true, 1_000_000_000)
            .map_err(|e| {
                vk_device.destroy_fence(fence, None);
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Timeout waiting for layout transition: {:?}",
                    e
                ))
            })?;

        // Cleanup
        vk_device.destroy_fence(fence, None);
        vk_device.destroy_command_pool(command_pool, None);

        debug!("Successfully transitioned image layout to SHADER_READ_ONLY_OPTIMAL with queue ownership transfer");
        Ok(())
    }

    /// Imports a DMABuf file descriptor into wgpu as a texture (zero-copy).
    ///
    /// This function creates a wgpu::Texture that directly references the DMABuf's
    /// GPU memory, enabling zero-copy video frame display from hardware decoders
    /// like VA-API, V4L2, or GStreamer with dmabuf export.
    ///
    /// # Implementation Notes
    ///
    /// The Vulkan import process:
    /// 1. Create VkImage with VkExternalMemoryImageCreateInfo specifying DMA_BUF handle type
    /// 2. Query memory requirements for the image
    /// 3. Import external memory via VkImportMemoryFdInfoKHR with the DMABuf fd
    /// 4. Bind the imported memory to the image
    /// 5. Wrap with wgpu-hal's texture_from_raw and create_texture_from_hal
    ///
    /// # Safety
    ///
    /// - `dmabuf` must contain a valid DMABuf file descriptor
    /// - The DMABuf must remain valid for the lifetime of the returned texture
    /// - The caller must ensure the DMABuf fd is not closed while the texture is in use
    /// - The fd ownership IS transferred to Vulkan - do NOT close it after this call
    /// - The DMABuf content must match the specified width, height, and format
    ///
    /// # Required Vulkan Extensions
    ///
    /// - VK_KHR_external_memory_fd
    /// - VK_EXT_external_memory_dma_buf
    /// - VK_EXT_image_drm_format_modifier (for non-linear formats)
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device (must be Vulkan backend)
    /// * `dmabuf` - DMABuf handle containing fd, size, offset, stride, and modifier
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `format` - The wgpu texture format (should match DMABuf pixel format)
    ///
    /// # Returns
    ///
    /// A wgpu::Texture that references the DMABuf memory directly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // From VA-API: vaSyncSurface + vaExportSurfaceHandle
    /// // From GStreamer: gst_buffer_get_dmabuf_memory
    /// let dmabuf = DmaBufHandle::single_plane(
    ///     va_surface_fd,
    ///     width * height * 4,  // size
    ///     0,                   // offset
    ///     width * 4,           // stride
    ///     drm_modifiers::DRM_FORMAT_MOD_LINEAR,
    /// );
    ///
    /// let texture = unsafe {
    ///     import_dmabuf(&device, dmabuf, 1920, 1080, wgpu::TextureFormat::Bgra8Unorm)?
    /// };
    /// ```
    ///
    /// # Multi-Plane Formats (Not Yet Supported)
    ///
    /// Multi-plane formats like NV12 require VkSamplerYcbcrConversion which is not
    /// yet implemented. For now, multi-plane handles will be rejected with an error.
    /// The infrastructure is in place for future implementation.
    pub unsafe fn import_dmabuf(
        device: &wgpu::Device,
        dmabuf: DmaBufHandle,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::Texture, ZeroCopyError> {
        // Validate the handle
        if dmabuf.fd() < 0 {
            return Err(ZeroCopyError::InvalidResource(
                "DMABuf fd is invalid (negative)".to_string(),
            ));
        }

        // Multi-plane formats require VkSamplerYcbcrConversion - not yet implemented
        if dmabuf.is_multi_plane() {
            return Err(ZeroCopyError::NotAvailable(
                "Multi-plane DMABuf import not yet implemented. \
                 Requires VkSamplerYcbcrConversion for NV12/YUV420p formats."
                    .to_string(),
            ));
        }

        // Access the Vulkan HAL device and create the texture
        let hal_texture_result = device
            .as_hal::<wgpu::hal::api::Vulkan, _, Result<wgpu::hal::vulkan::Texture, ZeroCopyError>>(
                |hal_device| {
                    let Some(hal_device) = hal_device else {
                        warn!("Failed to get Vulkan HAL device");
                        return Err(ZeroCopyError::HalAccessFailed(
                            "wgpu not using Vulkan backend".to_string(),
                        ));
                    };

                    // Check for required extensions
                    let extensions = hal_device.enabled_device_extensions();
                    let has_dma_buf = extensions.contains(&EXT_EXTERNAL_MEMORY_DMA_BUF);
                    let has_fd = extensions.contains(&KHR_EXTERNAL_MEMORY_FD);
                    let has_drm_modifier = extensions.contains(&EXT_IMAGE_DRM_FORMAT_MODIFIER);

                    if !has_dma_buf || !has_fd {
                        return Err(ZeroCopyError::NotAvailable(
                            "VK_EXT_external_memory_dma_buf or VK_KHR_external_memory_fd not available".to_string(),
                        ));
                    }

                    // Use DRM modifier extension when:
                    // 1. Non-linear modifier (always needs the extension), OR
                    // 2. Non-zero offset (linear tiling without extension can't specify offsets)
                    //
                    // For single-FD multi-plane layouts, each plane has a different offset.
                    // Without explicit plane layout, Vulkan binds memory at offset 0, causing
                    // planes to read wrong data (e.g., UV plane reading Y data → color corruption).
                    let has_nonzero_offset = dmabuf.offset() != 0;
                    let use_drm_modifier = has_drm_modifier
                        && (dmabuf.modifier != drm_modifiers::DRM_FORMAT_MOD_LINEAR
                            || has_nonzero_offset);

                    debug!(
                        "Importing DMABuf fd={} ({}x{} {:?}, modifier=0x{:x}, offset={}, use_drm={}) into Vulkan",
                        dmabuf.fd(), width, height, format, dmabuf.modifier, dmabuf.offset(), use_drm_modifier
                    );

                    let vk_device = hal_device.raw_device();
                    let physical_device = hal_device.raw_physical_device();
                    let instance = hal_device.shared_instance().raw_instance();
                    let vk_queue = hal_device.raw_queue();
                    let queue_family_index = hal_device.queue_family_index();
                    let vk_format = wgpu_format_to_vulkan(format)?;

                    // Step 1: Create VkImage with external memory info
                    let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
                        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

                    // For tiled formats, specify the DRM format modifier
                    let plane_layout;
                    let mut drm_modifier_info;
                    let mut drm_modifier_list_info;
                    let modifiers;

                    let mut image_create_info = vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk_format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .push_next(&mut external_memory_info);

                    if use_drm_modifier {
                        // Use explicit DRM format modifier
                        plane_layout = vk::SubresourceLayout {
                            offset: dmabuf.offset(),
                            size: dmabuf.size(),
                            row_pitch: dmabuf.stride() as u64,
                            array_pitch: 0,
                            depth_pitch: 0,
                        };

                        drm_modifier_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                            .drm_format_modifier(dmabuf.modifier)
                            .plane_layouts(std::slice::from_ref(&plane_layout));

                        image_create_info = image_create_info
                            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                            .push_next(&mut drm_modifier_info);
                    } else if dmabuf.modifier == drm_modifiers::DRM_FORMAT_MOD_LINEAR {
                        // Linear format - use LINEAR tiling
                        // IMPORTANT: Linear tiling without DRM modifier extension cannot honor
                        // non-zero offsets. If we have an offset, we reach this branch because
                        // has_drm_modifier is false, and we must fail fast to avoid color corruption.
                        if has_nonzero_offset {
                            return Err(ZeroCopyError::NotAvailable(format!(
                                "DMABuf with non-zero offset ({}) requires VK_EXT_image_drm_format_modifier extension \
                                 for correct plane binding. Without it, planes would read from wrong memory locations.",
                                dmabuf.offset()
                            )));
                        }
                        image_create_info = image_create_info.tiling(vk::ImageTiling::LINEAR);
                    } else {
                        // Non-linear modifier requires DRM modifier extension
                        // Falling back to OPTIMAL tiling would silently corrupt sampling
                        if !has_drm_modifier {
                            return Err(ZeroCopyError::NotAvailable(format!(
                                "Non-linear DMABuf modifier 0x{:x} requires VK_EXT_image_drm_format_modifier extension",
                                dmabuf.modifier
                            )));
                        }

                        // Store modifier in outer-scoped variable so the slice lives until create_image
                        modifiers = [dmabuf.modifier];
                        drm_modifier_list_info = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                            .drm_format_modifiers(&modifiers);

                        image_create_info = image_create_info
                            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                            .push_next(&mut drm_modifier_list_info);
                    }

                    let vk_image = vk_device
                        .create_image(&image_create_info, None)
                        .map_err(|e| {
                            ZeroCopyError::TextureCreationFailed(format!(
                                "vkCreateImage failed: {:?}",
                                e
                            ))
                        })?;

                    // Step 2: Get memory requirements
                    let mem_requirements = vk_device.get_image_memory_requirements(vk_image);

                    // Step 3: Import external memory from DMABuf fd
                    let mut import_memory_info = vk::ImportMemoryFdInfoKHR::default()
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                        .fd(dmabuf.fd());

                    // Find suitable memory type (device local preferred)
                    let memory_type_index = find_memory_type_index(
                        instance,
                        physical_device,
                        mem_requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .or_else(|| {
                        // Fallback: try without device local requirement
                        find_memory_type_index(
                            instance,
                            physical_device,
                            mem_requirements.memory_type_bits,
                            vk::MemoryPropertyFlags::empty(),
                        )
                    })
                    .ok_or_else(|| {
                        // Clean up the image before returning error
                        vk_device.destroy_image(vk_image, None);
                        ZeroCopyError::TextureCreationFailed(
                            "No suitable memory type for DMABuf import".to_string(),
                        )
                    })?;

                    let memory_allocate_info = vk::MemoryAllocateInfo::default()
                        .allocation_size(mem_requirements.size)
                        .memory_type_index(memory_type_index)
                        .push_next(&mut import_memory_info);

                    let device_memory = vk_device
                        .allocate_memory(&memory_allocate_info, None)
                        .map_err(|e| {
                            vk_device.destroy_image(vk_image, None);
                            ZeroCopyError::TextureCreationFailed(format!(
                                "vkAllocateMemory (DMABuf import) failed: {:?}",
                                e
                            ))
                        })?;

                    // Step 4: Bind memory to image
                    vk_device
                        .bind_image_memory(vk_image, device_memory, 0)
                        .map_err(|e| {
                            vk_device.free_memory(device_memory, None);
                            vk_device.destroy_image(vk_image, None);
                            ZeroCopyError::TextureCreationFailed(format!(
                                "vkBindImageMemory failed: {:?}",
                                e
                            ))
                        })?;

                    info!(
                        "Successfully created Vulkan image from DMABuf fd={} ({}x{} {:?})",
                        dmabuf.fd(), width, height, format
                    );

                    // Step 5: Transition image layout and acquire queue ownership
                    // External memory requires explicit layout transition from UNDEFINED
                    // to SHADER_READ_ONLY_OPTIMAL and queue family ownership transfer
                    // from VK_QUEUE_FAMILY_EXTERNAL to our graphics queue family.
                    transition_image_layout_external(
                        vk_device,
                        vk_queue,
                        queue_family_index,
                        vk_image,
                    )
                    .inspect_err(|_| {
                        vk_device.free_memory(device_memory, None);
                        vk_device.destroy_image(vk_image, None);
                    })?;

                    // Step 6: Wrap as wgpu-hal Texture
                    // Create a TextureDescriptor for texture_from_raw
                    let texture_desc = wgpu::hal::TextureDescriptor {
                        label: Some("zero-copy DMABuf texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::hal::TextureUses::RESOURCE,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    };

                    // Create a drop callback to free imported Vulkan resources when
                    // the texture is destroyed. Clone the device handle for the callback.
                    let device_clone = vk_device.clone();
                    let drop_callback = Box::new(move || {
                        debug!("Freeing imported DMABuf Vulkan resources");
                        // SAFETY: vk_image and device_memory were allocated by us and are valid
                        // until this callback is invoked when the texture is dropped.
                        // destroy_image must be called before free_memory.
                        unsafe {
                            device_clone.destroy_image(vk_image, None);
                            device_clone.free_memory(device_memory, None);
                        }
                    });

                    // drop_callback is called when wgpu is done with the texture,
                    // allowing us to free the externally managed VkDeviceMemory
                    let hal_texture = wgpu::hal::vulkan::Device::texture_from_raw(
                        vk_image,
                        &texture_desc,
                        Some(drop_callback),
                    );

                    Ok(hal_texture)
                },
            );

        // Get the HAL texture from the closure result
        let hal_texture = hal_texture_result?;

        // Create wgpu texture descriptor
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("zero-copy DMABuf texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // Wrap the HAL texture as a wgpu::Texture
        let wgpu_texture =
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &texture_desc);

        info!("Successfully imported DMABuf as wgpu texture (zero-copy)");

        Ok(wgpu_texture)
    }

    /// Imports a multi-plane DMABuf (NV12/YUV420p) into wgpu as separate textures.
    ///
    /// This function creates separate wgpu::Textures for each YUV plane, enabling
    /// zero-copy video frame display from hardware decoders that output multi-plane
    /// formats like NV12 (2 planes: Y + interleaved UV) or YUV420p (3 planes: Y + U + V).
    ///
    /// # Implementation Notes
    ///
    /// Instead of using VkSamplerYcbcrConversion (which requires immutable samplers
    /// incompatible with wgpu), this function:
    /// 1. Imports each plane as a separate single-channel VkImage
    /// 2. Returns a vector of textures that can be bound to shader slots
    /// 3. Lets the existing WGSL shaders perform YUV→RGB conversion
    ///
    /// # Safety
    ///
    /// - `dmabuf` must contain valid DMABuf file descriptors for all planes
    /// - The DMABufs must remain valid for the lifetime of the returned textures
    /// - The fd ownership IS transferred to Vulkan - do NOT close them after this call
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device (must be Vulkan backend)
    /// * `dmabuf` - Multi-plane DMABuf handle with per-plane metadata
    /// * `width` - Full frame width in pixels
    /// * `height` - Full frame height in pixels
    /// * `format` - The pixel format (NV12, YUV420p, etc.)
    ///
    /// # Returns
    ///
    /// A vector of wgpu::Textures:
    /// - NV12: `[Y (R8, full size), UV (RG8, half size)]`
    /// - YUV420p: `[Y (R8, full size), U (R8, half size), V (R8, half size)]`
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lumina_video::media::zero_copy::linux::{self, DmaBufHandle, DmaBufPlaneHandle};
    /// use lumina_video::media::video::PixelFormat;
    ///
    /// let planes = vec![
    ///     DmaBufPlaneHandle { fd: y_fd, offset: 0, stride: 1920, size: 1920 * 1080 },
    ///     DmaBufPlaneHandle { fd: uv_fd, offset: 0, stride: 1920, size: 960 * 540 * 2 },
    /// ];
    /// let dmabuf = DmaBufHandle::new(planes, drm_modifiers::DRM_FORMAT_MOD_LINEAR);
    ///
    /// let textures = unsafe {
    ///     import_dmabuf_multi_plane(&device, dmabuf, 1920, 1080, PixelFormat::Nv12)?
    /// };
    /// // textures[0] = Y plane, textures[1] = UV plane
    /// ```
    pub unsafe fn import_dmabuf_multi_plane(
        device: &wgpu::Device,
        dmabuf: DmaBufHandle,
        width: u32,
        height: u32,
        format: super::super::video::PixelFormat,
    ) -> Result<Vec<wgpu::Texture>, ZeroCopyError> {
        use super::super::video::PixelFormat;

        // Validate plane count matches format
        let expected_planes = match format {
            PixelFormat::Nv12 => 2,
            PixelFormat::Yuv420p => 3,
            _ => {
                return Err(ZeroCopyError::InvalidResource(format!(
                    "Format {:?} is not a multi-plane YUV format",
                    format
                )));
            }
        };

        if dmabuf.num_planes() != expected_planes {
            return Err(ZeroCopyError::InvalidResource(format!(
                "Expected {} planes for {:?}, got {}",
                expected_planes,
                format,
                dmabuf.num_planes()
            )));
        }

        // Validate all FDs
        for (i, plane) in dmabuf.planes.iter().enumerate() {
            if plane.fd < 0 {
                return Err(ZeroCopyError::InvalidResource(format!(
                    "Plane {} has invalid fd (negative)",
                    i
                )));
            }
        }

        // Check for single-FD multi-plane layout (not supported due to offset handling)
        if dmabuf.planes.len() > 1 {
            // Check if all planes share the same FD (single-FD layout)
            let first_fd = dmabuf.planes[0].fd;
            let is_single_fd = dmabuf.planes.iter().all(|p| p.fd == first_fd);
            if is_single_fd {
                warn!(
                    "Multi-plane single-FD DMABuf layout detected (all {} planes share fd={}). \
                     Per-plane offsets not honored by vkBindImageMemory. Falling back to CPU path.",
                    dmabuf.planes.len(),
                    first_fd
                );
                return Err(ZeroCopyError::NotAvailable(
                    "Multi-plane single-FD DMABuf layout not supported (per-plane offsets not honored). \
                     Use multi-FD layout or CPU fallback.".to_string()
                ));
            }
        }

        info!(
            "Importing multi-plane DMABuf: {:?} ({}x{}, {} planes)",
            format,
            width,
            height,
            dmabuf.num_planes()
        );

        // Build per-plane import specifications
        let plane_specs: Vec<(u32, u32, wgpu::TextureFormat)> = match format {
            PixelFormat::Nv12 => vec![
                // Plane 0: Y (luma) - full resolution, R8
                (width, height, wgpu::TextureFormat::R8Unorm),
                // Plane 1: UV (chroma) - half resolution, RG8 (interleaved)
                (width / 2, height / 2, wgpu::TextureFormat::Rg8Unorm),
            ],
            PixelFormat::Yuv420p => vec![
                // Plane 0: Y (luma) - full resolution, R8
                (width, height, wgpu::TextureFormat::R8Unorm),
                // Plane 1: U (Cb) - half resolution, R8
                (width / 2, height / 2, wgpu::TextureFormat::R8Unorm),
                // Plane 2: V (Cr) - half resolution, R8
                (width / 2, height / 2, wgpu::TextureFormat::R8Unorm),
            ],
            _ => unreachable!(),
        };

        // Import each plane as a separate texture
        let mut textures = Vec::with_capacity(plane_specs.len());

        for (i, ((plane_width, plane_height, wgpu_format), plane)) in
            plane_specs.iter().zip(dmabuf.planes.iter()).enumerate()
        {
            debug!(
                "Importing plane {}: {}x{} {:?} (fd={}, offset={}, stride={})",
                i, plane_width, plane_height, wgpu_format, plane.fd, plane.offset, plane.stride
            );

            // Create a single-plane handle for this plane
            let single_plane_handle = DmaBufHandle::single_plane(
                plane.fd,
                plane.size,
                plane.offset,
                plane.stride,
                dmabuf.modifier,
            );

            // Import using the existing single-plane function
            match import_dmabuf(
                device,
                single_plane_handle,
                *plane_width,
                *plane_height,
                *wgpu_format,
            ) {
                Ok(texture) => {
                    textures.push(texture);
                }
                Err(e) => {
                    // Clean up already-imported textures on failure
                    // (wgpu::Texture drops automatically when Vec is dropped)
                    warn!("Failed to import plane {}: {:?}", i, e);
                    return Err(ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to import plane {}: {}",
                        i, e
                    )));
                }
            }
        }

        info!(
            "Successfully imported {} planes as wgpu textures (zero-copy multi-plane)",
            textures.len()
        );

        Ok(textures)
    }

    /// Imports a single-FD multi-plane DMABuf (NV12/YUV420p) into wgpu as separate textures.
    ///
    /// This function handles the case where all planes share a single file descriptor but
    /// have different offsets within that buffer. This is common with VA-API which outputs
    /// single-FD multi-plane layouts.
    ///
    /// # Implementation Notes
    ///
    /// For single-FD multi-plane layouts, the Vulkan import uses:
    /// 1. VkImageDrmFormatModifierExplicitCreateInfoEXT with pPlaneLayouts array
    /// 2. VkSubresourceLayout per plane specifying offset, size, and rowPitch
    /// 3. VK_FORMAT_G8_B8R8_2PLANE_420_UNORM for NV12 (uses VkSamplerYcbcrConversion)
    ///
    /// However, VkSamplerYcbcrConversion requires immutable samplers which are incompatible
    /// with wgpu's sampler model. Instead, this implementation imports each plane as a
    /// separate single-channel texture:
    /// - For the Y plane: Create image from FD at plane 0's offset
    /// - For the UV plane: Create separate image from same FD at plane 1's offset
    ///
    /// # Safety
    ///
    /// - `planes` must contain valid DMABuf plane handles with the same FD
    /// - The DMABuf must remain valid for the lifetime of the returned textures
    /// - The fd ownership IS transferred to Vulkan - do NOT close it after this call
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device (must be Vulkan backend)
    /// * `fd` - The shared DMABuf file descriptor
    /// * `planes` - Per-plane metadata (offset, stride, size) - all must reference same fd
    /// * `width` - Full frame width in pixels
    /// * `height` - Full frame height in pixels
    /// * `format` - The pixel format (NV12, YUV420p, etc.)
    /// * `modifier` - DRM format modifier
    ///
    /// # Returns
    ///
    /// A vector of wgpu::Textures:
    /// - NV12: `[Y (R8, full size), UV (RG8, half size)]`
    /// - YUV420p: `[Y (R8, full size), U (R8, half size), V (R8, half size)]`
    pub unsafe fn import_dmabuf_single_fd_multi_plane(
        device: &wgpu::Device,
        fd: std::os::fd::RawFd,
        planes: &[DmaBufPlaneHandle],
        width: u32,
        height: u32,
        format: super::super::video::PixelFormat,
        modifier: u64,
    ) -> Result<Vec<wgpu::Texture>, ZeroCopyError> {
        use super::super::video::PixelFormat;

        // Validate FD
        if fd < 0 {
            return Err(ZeroCopyError::InvalidResource(
                "DMABuf fd is invalid (negative)".to_string(),
            ));
        }

        // Validate plane count matches format
        let expected_planes = match format {
            PixelFormat::Nv12 => 2,
            PixelFormat::Yuv420p => 3,
            _ => {
                return Err(ZeroCopyError::InvalidResource(format!(
                    "Format {:?} is not a multi-plane YUV format",
                    format
                )));
            }
        };

        if planes.len() != expected_planes {
            return Err(ZeroCopyError::InvalidResource(format!(
                "Expected {} planes for {:?}, got {}",
                expected_planes,
                format,
                planes.len()
            )));
        }

        info!(
            "Importing single-FD multi-plane DMABuf: {:?} ({}x{}, {} planes, fd={})",
            format,
            width,
            height,
            planes.len(),
            fd
        );

        // Build per-plane import specifications
        let plane_specs: Vec<(u32, u32, wgpu::TextureFormat)> = match format {
            PixelFormat::Nv12 => vec![
                // Plane 0: Y (luma) - full resolution, R8
                (width, height, wgpu::TextureFormat::R8Unorm),
                // Plane 1: UV (chroma) - half resolution, RG8 (interleaved)
                (width / 2, height / 2, wgpu::TextureFormat::Rg8Unorm),
            ],
            PixelFormat::Yuv420p => vec![
                // Plane 0: Y (luma) - full resolution, R8
                (width, height, wgpu::TextureFormat::R8Unorm),
                // Plane 1: U (Cb) - half resolution, R8
                (width / 2, height / 2, wgpu::TextureFormat::R8Unorm),
                // Plane 2: V (Cr) - half resolution, R8
                (width / 2, height / 2, wgpu::TextureFormat::R8Unorm),
            ],
            _ => unreachable!(),
        };

        // Import each plane as a separate texture using dup'd FD
        let mut textures = Vec::with_capacity(plane_specs.len());

        for (i, ((plane_width, plane_height, wgpu_format), plane)) in
            plane_specs.iter().zip(planes.iter()).enumerate()
        {
            debug!(
                "Importing plane {}: {}x{} {:?} (offset={}, stride={}, size={})",
                i, plane_width, plane_height, wgpu_format, plane.offset, plane.stride, plane.size
            );

            // Each plane import needs its own FD since Vulkan takes ownership
            // For the first plane, we use the original fd
            // For subsequent planes, we dup the fd
            let plane_fd = if i == 0 {
                fd
            } else {
                let dup_fd = libc::dup(fd);
                if dup_fd < 0 {
                    warn!(
                        "Failed to dup DMABuf fd {} for plane {}: {}",
                        fd,
                        i,
                        std::io::Error::last_os_error()
                    );
                    return Err(ZeroCopyError::InvalidResource(format!(
                        "Failed to dup DMABuf fd for plane {}: {}",
                        i,
                        std::io::Error::last_os_error()
                    )));
                }
                dup_fd
            };

            // Create a single-plane handle for this plane with the correct offset
            let single_plane_handle = DmaBufHandle::single_plane(
                plane_fd,
                plane.size,
                plane.offset,
                plane.stride,
                modifier,
            );

            // Import using the existing single-plane function
            match import_dmabuf(
                device,
                single_plane_handle,
                *plane_width,
                *plane_height,
                *wgpu_format,
            ) {
                Ok(texture) => {
                    textures.push(texture);
                }
                Err(e) => {
                    // Close the dup'd FD for this plane since import_dmabuf failed
                    // (import_dmabuf takes ownership on success, but on failure we must clean up)
                    // Note: plane 0 uses the original fd which caller owns, so only close if i > 0
                    if i > 0 {
                        unsafe {
                            libc::close(plane_fd);
                        }
                    }
                    warn!(
                        "Failed to import plane {} from single-FD layout: {:?}",
                        i, e
                    );
                    return Err(ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to import plane {} from single-FD layout: {}",
                        i, e
                    )));
                }
            }
        }

        info!(
            "Successfully imported {} planes from single-FD DMABuf as wgpu textures (zero-copy)",
            textures.len()
        );

        Ok(textures)
    }

    /// Converts wgpu TextureFormat to Vulkan VkFormat.
    ///
    /// Returns an error for unsupported formats instead of silently defaulting.
    /// NV12 is explicitly rejected because it is a multi-plane format requiring
    /// VK_KHR_sampler_ycbcr_conversion which is not yet implemented.
    pub fn wgpu_format_to_vulkan(format: wgpu::TextureFormat) -> Result<vk::Format, ZeroCopyError> {
        match format {
            wgpu::TextureFormat::Bgra8Unorm => Ok(vk::Format::B8G8R8A8_UNORM),
            wgpu::TextureFormat::Rgba8Unorm => Ok(vk::Format::R8G8B8A8_UNORM),
            wgpu::TextureFormat::R8Unorm => Ok(vk::Format::R8_UNORM),
            wgpu::TextureFormat::Rg8Unorm => Ok(vk::Format::R8G8_UNORM),
            wgpu::TextureFormat::Bgra8UnormSrgb => Ok(vk::Format::B8G8R8A8_SRGB),
            wgpu::TextureFormat::Rgba8UnormSrgb => Ok(vk::Format::R8G8B8A8_SRGB),
            // NV12 is a multi-plane format (Y plane + interleaved UV plane) that requires
            // VK_KHR_sampler_ycbcr_conversion for proper handling. The current single-plane
            // DMABuf import cannot handle NV12 correctly. Proper support would require:
            // - Creating a multi-plane VkImage with VK_IMAGE_CREATE_DISJOINT_BIT
            // - Setting up VkSamplerYcbcrConversion for Y'CbCr color space conversion
            // - Using separate plane layouts for Y and UV data
            // - Binding memory to each plane separately
            wgpu::TextureFormat::NV12 => {
                warn!("NV12 format requested but multi-plane import not yet supported");
                Err(ZeroCopyError::InvalidResource(
                    "NV12 multi-plane format not yet supported for zero-copy import. \
                     Proper support requires VK_KHR_sampler_ycbcr_conversion."
                        .to_string(),
                ))
            }
            _ => {
                warn!("Unsupported texture format {:?}; refusing import", format);
                Err(ZeroCopyError::InvalidResource(format!(
                    "Unsupported texture format {:?}",
                    format
                )))
            }
        }
    }

    /// Common DRM format modifiers for reference.
    ///
    /// These are from drm_fourcc.h and are used with VK_EXT_image_drm_format_modifier.
    pub mod drm_modifiers {
        /// Linear layout (no tiling)
        pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

        /// Invalid/unspecified modifier
        pub const DRM_FORMAT_MOD_INVALID: u64 = 0x00ffffffffffffff;

        // Intel modifiers (I915_FORMAT_MOD_*)
        /// Intel X-tiled
        pub const I915_FORMAT_MOD_X_TILED: u64 = 0x0100000000000001;
        /// Intel Y-tiled
        pub const I915_FORMAT_MOD_Y_TILED: u64 = 0x0100000000000002;
        /// Intel Yf-tiled
        pub const I915_FORMAT_MOD_YF_TILED: u64 = 0x0100000000000003;
        /// Intel Y-tiled with CCS (Tile4)
        pub const I915_FORMAT_MOD_Y_TILED_CCS: u64 = 0x0100000000000004;
        /// Intel Tile4 (DG2+)
        pub const I915_FORMAT_MOD_4_TILED: u64 = 0x0100000000000009;

        // AMD modifiers (AMD_FMT_MOD)
        /// AMD GFX9 64KB tiled
        pub const AMD_FMT_MOD_TILE_GFX9_64K_S: u64 = 0x0200000000000001;

        // NVIDIA modifiers
        /// NVIDIA block linear (16Bx2)
        pub const NVIDIA_FORMAT_MOD_BLOCK_LINEAR_2D: u64 = 0x0300000000000010;
    }
}

// =============================================================================
// Android: AHardwareBuffer → Vulkan → wgpu
// =============================================================================

/// Android-specific zero-copy import via AHardwareBuffer and Vulkan.
///
/// This module provides functions to import Android hardware buffers
/// directly into wgpu textures without CPU memory copies.
///
/// # Requirements
///
/// - Android API level 26+ (Android 8.0 Oreo)
/// - wgpu using Vulkan backend
/// - Vulkan extension: `VK_ANDROID_external_memory_android_hardware_buffer`
/// - AHardwareBuffer with `AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE` usage
///
/// # Limitations
///
/// - YCbCr/YUV formats (NV12, etc.) require `VkSamplerYcbcrConversion` which
///   is not yet implemented. These formats are common from MediaCodec video
///   output. For now, request RGBA output from MediaCodec if possible.
///
/// # Example
///
/// ```ignore
/// use lumina_video::media::zero_copy::android;
///
/// // From MediaCodec: AImage_getHardwareBuffer()
/// let ahb = get_hardware_buffer_from_media_codec();
///
/// if android::is_ahardwarebuffer_import_available(&device) {
///     let texture = unsafe {
///         android::import_ahardwarebuffer(&device, ahb, width, height, format)?
///     };
/// }
/// ```
#[cfg(target_os = "android")]
pub mod android {
    use super::ZeroCopyError;
    use ash::vk::{self, Handle};
    use std::ffi::CStr;
    use tracing::{debug, info, warn};

    // For closing sync fence FD after semaphore import
    extern "C" {
        fn close(fd: std::ffi::c_int) -> std::ffi::c_int;
    }
    // Alias the libc close function
    mod libc {
        pub unsafe fn close(fd: i32) -> i32 {
            super::close(fd)
        }
    }

    /// Opaque handle to an AHardwareBuffer (from Android NDK).
    /// This is obtained from MediaCodec's output Image or from ANativeWindow.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid AHardwareBuffer obtained from the Android NDK.
    /// The caller is responsible for managing the AHardwareBuffer's lifetime
    /// (acquire/release reference counting).
    pub type AHardwareBufferPtr = *mut std::ffi::c_void;

    /// VK_ANDROID_external_memory_android_hardware_buffer extension name
    const VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME: &CStr = unsafe {
        CStr::from_bytes_with_nul_unchecked(b"VK_ANDROID_external_memory_android_hardware_buffer\0")
    };

    /// Checks if the wgpu device is using the Vulkan backend.
    ///
    /// AHardwareBuffer zero-copy import requires Vulkan. This function checks whether
    /// the wgpu device was created with the Vulkan backend, which is necessary
    /// for using `VK_ANDROID_external_memory_android_hardware_buffer`.
    ///
    /// On Android, Vulkan is typically available on devices with API level 24+
    /// (Android 7.0 Nougat) and a compatible GPU. Returns `false` if using
    /// OpenGL ES or software rendering.
    pub fn is_vulkan_backend(device: &wgpu::Device) -> bool {
        unsafe {
            device.as_hal::<wgpu::hal::api::Vulkan, _, bool>(|hal_device| hal_device.is_some())
        }
    }

    /// Checks if Vulkan AHardwareBuffer import extension is available.
    ///
    /// This checks for VK_ANDROID_external_memory_android_hardware_buffer extension
    /// which is required for zero-copy AHardwareBuffer import.
    pub fn is_ahardwarebuffer_import_available(device: &wgpu::Device) -> bool {
        unsafe {
            device.as_hal::<wgpu::hal::api::Vulkan, _, bool>(|hal_device| {
                let Some(hal_device) = hal_device else {
                    return false;
                };

                let enabled_extensions = hal_device.enabled_device_extensions();
                enabled_extensions
                    .contains(&VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME)
            })
        }
    }

    /// Gets information about the Vulkan device for diagnostics.
    ///
    /// Returns device name and driver version if Vulkan backend is available.
    pub fn get_vulkan_device_info(device: &wgpu::Device) -> Option<String> {
        unsafe {
            device.as_hal::<wgpu::hal::api::Vulkan, _, Option<String>>(|hal_device| {
                hal_device.map(|d| {
                    let instance = d.shared_instance();
                    let raw_instance = instance.raw_instance();
                    let physical_device = d.raw_physical_device();

                    let properties = raw_instance.get_physical_device_properties(physical_device);
                    let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned();
                    format!(
                        "{} (Vulkan {}.{}.{})",
                        device_name,
                        vk::api_version_major(properties.api_version),
                        vk::api_version_minor(properties.api_version),
                        vk::api_version_patch(properties.api_version)
                    )
                })
            })
        }
    }

    /// Imports an AHardwareBuffer into wgpu as a texture (zero-copy).
    ///
    /// This function creates a wgpu::Texture that directly references the AHardwareBuffer's
    /// GPU memory, enabling zero-copy video frame display from MediaCodec.
    ///
    /// # Safety
    ///
    /// - `ahardware_buffer` must be a valid AHardwareBuffer pointer obtained from
    ///   AImage_getHardwareBuffer() or similar NDK functions
    /// - The AHardwareBuffer must remain valid for the lifetime of the returned texture
    /// - The caller must maintain a reference to the AHardwareBuffer (via AHardwareBuffer_acquire)
    /// - The AHardwareBuffer must have been created with AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device (must be Vulkan backend)
    /// * `ahardware_buffer` - A valid AHardwareBuffer pointer
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `format` - The wgpu texture format (should match AHardwareBuffer format)
    ///
    /// # Returns
    ///
    /// A wgpu::Texture that references the AHardwareBuffer memory directly.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The wgpu device is not using Vulkan backend
    /// - VK_ANDROID_external_memory_android_hardware_buffer extension is not available
    /// - The AHardwareBuffer is invalid or incompatible
    /// - Vulkan image/memory creation fails
    ///
    /// # Implementation Notes
    ///
    /// This uses the following Vulkan extensions:
    /// - VK_KHR_external_memory (Vulkan 1.1 core)
    /// - VK_ANDROID_external_memory_android_hardware_buffer
    ///
    /// The import process:
    /// 1. Query AHardwareBuffer properties via vkGetAndroidHardwareBufferPropertiesANDROID
    /// 2. Create VkImage with VkExternalMemoryImageCreateInfo
    /// 3. Import memory via VkImportAndroidHardwareBufferInfoANDROID
    /// 4. Bind memory to image with VkMemoryDedicatedAllocateInfo
    /// 5. Wrap as wgpu-hal::vulkan::Texture via texture_from_raw
    /// 6. Create wgpu::Texture via create_texture_from_hal
    pub unsafe fn import_ahardwarebuffer(
        device: &wgpu::Device,
        ahardware_buffer: AHardwareBufferPtr,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::Texture, ZeroCopyError> {
        if ahardware_buffer.is_null() {
            return Err(ZeroCopyError::InvalidResource(
                "AHardwareBuffer is null".to_string(),
            ));
        }

        // Access the Vulkan HAL device and create the texture
        let hal_texture_result = device
            .as_hal::<wgpu::hal::api::Vulkan, _, Result<wgpu::hal::vulkan::Texture, ZeroCopyError>>(
                |hal_device| {
                    let Some(hal_device) = hal_device else {
                        warn!("Failed to get Vulkan HAL device");
                        return Err(ZeroCopyError::HalAccessFailed(
                            "wgpu not using Vulkan backend".to_string(),
                        ));
                    };

                    // Check for required extension
                    let enabled_extensions = hal_device.enabled_device_extensions();
                    let has_ahb_extension = enabled_extensions
                        .contains(&VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);

                    if !has_ahb_extension {
                        warn!("VK_ANDROID_external_memory_android_hardware_buffer not available");
                        return Err(ZeroCopyError::NotAvailable(
                            "VK_ANDROID_external_memory_android_hardware_buffer extension not enabled".to_string(),
                        ));
                    }

                    let raw_device = hal_device.raw_device();
                    let instance = hal_device.shared_instance();
                    let raw_instance = instance.raw_instance();
                    let physical_device = hal_device.raw_physical_device();
                    let vk_queue = hal_device.raw_queue();
                    let queue_family_index = hal_device.queue_family_index();

                    debug!(
                        "Creating Vulkan image from AHardwareBuffer ({}x{} {:?})",
                        width, height, format
                    );

                    // Step 1: Get AHardwareBuffer format properties
                    // Query VkAndroidHardwareBufferFormatPropertiesANDROID and extract all needed
                    // values. The push_next pattern creates a mutable borrow chain, so we must
                    // scope ahb_props to drop it before accessing ahb_format_props fields.
                    let mut ahb_format_props = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
                    let (ahb_allocation_size, ahb_memory_type_bits) = {
                        let mut ahb_props = vk::AndroidHardwareBufferPropertiesANDROID::default()
                            .push_next(&mut ahb_format_props);

                        // Get the AHardwareBuffer properties using the extension function
                        // We need to load the extension function manually since ash doesn't
                        // have a convenient wrapper for this extension
                        type GetAndroidHardwareBufferPropertiesFn = unsafe extern "system" fn(
                            device: vk::Device,
                            buffer: *const std::ffi::c_void,
                            properties: *mut vk::AndroidHardwareBufferPropertiesANDROID,
                        ) -> vk::Result;

                        let get_ahb_props_fn: GetAndroidHardwareBufferPropertiesFn = {
                            let fn_name = CStr::from_bytes_with_nul_unchecked(
                                b"vkGetAndroidHardwareBufferPropertiesANDROID\0"
                            );
                            let fn_ptr = raw_instance
                                .get_device_proc_addr(raw_device.handle(), fn_name.as_ptr());
                            if fn_ptr.is_none() {
                                return Err(ZeroCopyError::HalAccessFailed(
                                    "vkGetAndroidHardwareBufferPropertiesANDROID not found"
                                        .to_string(),
                                ));
                            }
                            // SAFETY: fn_ptr is verified non-null above. The function pointer
                            // is obtained from vkGetDeviceProcAddr for a valid extension function
                            // name, and we transmute it to a matching function signature as
                            // defined by the Vulkan spec for VK_ANDROID_external_memory_android_hardware_buffer.
                            std::mem::transmute(fn_ptr)
                        };

                        let result = get_ahb_props_fn(
                            raw_device.handle(),
                            ahardware_buffer,
                            &mut ahb_props,
                        );

                        if result != vk::Result::SUCCESS {
                            warn!("vkGetAndroidHardwareBufferPropertiesANDROID failed: {:?}", result);
                            return Err(ZeroCopyError::InvalidResource(
                                format!("Failed to get AHardwareBuffer properties: {:?}", result),
                            ));
                        }

                        // Extract ahb_props values before it drops (releasing borrow on ahb_format_props)
                        (ahb_props.allocation_size, ahb_props.memory_type_bits)
                    };
                    // Now ahb_props is dropped, we can access ahb_format_props
                    let ahb_format = ahb_format_props.format;
                    let ahb_external_format = ahb_format_props.external_format;

                    debug!(
                        "AHardwareBuffer properties: size={}, memory_type_bits={:#x}, format={:?}",
                        ahb_allocation_size,
                        ahb_memory_type_bits,
                        ahb_format
                    );

                    // Check for YCbCr/YUV formats that require VkSamplerYcbcrConversion.
                    // These formats are common from Android camera/video sources but require
                    // additional Vulkan setup that is not yet implemented:
                    // - VkSamplerYcbcrConversion object
                    // - VkSamplerYcbcrConversionInfo in image view and sampler creation
                    // - Immutable samplers in descriptor set layout
                    // Reject them explicitly with a clear error until this is implemented.
                    if is_ycbcr_format(ahb_format) {
                        warn!(
                            "AHardwareBuffer has YCbCr/YUV format {:?} which is not supported",
                            ahb_format
                        );
                        return Err(ZeroCopyError::InvalidResource(
                            "YCbCr/YUV formats require VkSamplerYcbcrConversion which is not yet implemented".to_string(),
                        ));
                    }

                    // Step 2: Create VkImage with external memory info
                    let vk_format = wgpu_format_to_vulkan(format)?;

                    // Check for external-format AHardwareBuffers.
                    // When format == VK_FORMAT_UNDEFINED but external_format != 0, the buffer
                    // uses an implementation-specific format that requires VkExternalFormatANDROID
                    // and VkSamplerYcbcrConversion. This path is complex and not yet supported.
                    if ahb_format == vk::Format::UNDEFINED && ahb_external_format != 0 {
                        warn!(
                            "AHardwareBuffer has external format {:#x} which requires VkExternalFormatANDROID (not supported)",
                            ahb_external_format
                        );
                        return Err(ZeroCopyError::NotAvailable(
                            "External-format AHardwareBuffers require VkExternalFormatANDROID which is not yet implemented".to_string(),
                        ));
                    }

                    // Validate AHardwareBuffer format matches expected format
                    if ahb_format != vk::Format::UNDEFINED && ahb_format != vk_format {
                        warn!(
                            "AHardwareBuffer format {:?} doesn't match expected wgpu format {:?} (Vulkan: {:?})",
                            ahb_format, format, vk_format
                        );
                        return Err(ZeroCopyError::FormatMismatch(format!(
                            "AHardwareBuffer has format {:?} but expected {:?}",
                            ahb_format, vk_format
                        )));
                    }

                    // Specify that this image will use external Android hardware buffer memory
                    let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
                        .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);

                    // Use the format from AHardwareBuffer if it's valid, otherwise use converted format
                    let image_format = if ahb_format != vk::Format::UNDEFINED {
                        debug!(
                            "Using AHardwareBuffer's native format {:?} (requested: {:?})",
                            ahb_format, vk_format
                        );
                        ahb_format
                    } else {
                        debug!(
                            "AHardwareBuffer format is UNDEFINED, using converted format {:?}",
                            vk_format
                        );
                        vk_format
                    };

                    let image_create_info = vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(image_format)
                        .extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .push_next(&mut external_memory_info);

                    let vk_image = raw_device.create_image(&image_create_info, None)
                        .map_err(|e| {
                            warn!("Failed to create VkImage: {:?}", e);
                            ZeroCopyError::TextureCreationFailed(
                                format!("Vulkan image creation failed: {:?}", e),
                            )
                        })?;

                    // Step 3: Allocate and bind memory from AHardwareBuffer
                    // VkImportAndroidHardwareBufferInfoANDROID
                    let mut import_ahb_info = vk::ImportAndroidHardwareBufferInfoANDROID::default()
                        .buffer(ahardware_buffer);

                    // Find suitable memory type
                    let mem_properties = raw_instance
                        .get_physical_device_memory_properties(physical_device);

                    let memory_type_index = find_memory_type_index(
                        ahb_memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        &mem_properties,
                    ).ok_or_else(|| {
                        raw_device.destroy_image(vk_image, None);
                        ZeroCopyError::TextureCreationFailed(
                            "No suitable memory type found for AHardwareBuffer".to_string(),
                        )
                    })?;

                    // VkMemoryDedicatedAllocateInfo - required for AHardwareBuffer
                    let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfo::default()
                        .image(vk_image);

                    let memory_allocate_info = vk::MemoryAllocateInfo::default()
                        .allocation_size(ahb_allocation_size)
                        .memory_type_index(memory_type_index)
                        .push_next(&mut import_ahb_info)
                        .push_next(&mut dedicated_alloc_info);

                    let vk_memory = raw_device.allocate_memory(&memory_allocate_info, None)
                        .map_err(|e| {
                            raw_device.destroy_image(vk_image, None);
                            warn!("Failed to allocate memory from AHardwareBuffer: {:?}", e);
                            ZeroCopyError::TextureCreationFailed(
                                format!("Memory allocation failed: {:?}", e),
                            )
                        })?;

                    // Bind memory to image
                    raw_device.bind_image_memory(vk_image, vk_memory, 0)
                        .map_err(|e| {
                            raw_device.free_memory(vk_memory, None);
                            raw_device.destroy_image(vk_image, None);
                            warn!("Failed to bind image memory: {:?}", e);
                            ZeroCopyError::TextureCreationFailed(
                                format!("Memory binding failed: {:?}", e),
                            )
                        })?;

                    info!(
                        "Created Vulkan image from AHardwareBuffer: {}x{} {:?}",
                        width, height, format
                    );

                    // Step 4: Transition image layout and acquire queue ownership
                    // External memory requires explicit layout transition from UNDEFINED
                    // to SHADER_READ_ONLY_OPTIMAL and queue family ownership transfer
                    // from VK_QUEUE_FAMILY_EXTERNAL to our graphics queue family.
                    transition_image_layout_external(
                        raw_device,
                        vk_queue,
                        queue_family_index,
                        vk_image,
                    )
                    .map_err(|e| {
                        raw_device.free_memory(vk_memory, None);
                        raw_device.destroy_image(vk_image, None);
                        e
                    })?;

                    // Step 5: Create wgpu-hal Texture descriptor
                    let texture_desc = wgpu::hal::TextureDescriptor {
                        label: Some("zero-copy AHardwareBuffer texture"),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::hal::TextureUses::RESOURCE,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    };

                    // Create a drop callback to free imported Vulkan resources when
                    // the texture is destroyed. Clone the device handle for the callback.
                    let device_clone = raw_device.clone();
                    let drop_callback = Box::new(move || {
                        debug!("Freeing imported AHardwareBuffer Vulkan resources");
                        // SAFETY: vk_image and vk_memory were allocated by us and are valid
                        // until this callback is invoked when the texture is dropped.
                        // The AHardwareBuffer itself remains valid (caller's responsibility).
                        // destroy_image must be called before free_memory.
                        unsafe {
                            device_clone.destroy_image(vk_image, None);
                            device_clone.free_memory(vk_memory, None);
                        }
                    });

                    // drop_callback is called when wgpu is done with the texture,
                    // allowing us to free the externally managed VkDeviceMemory
                    let hal_texture = wgpu::hal::vulkan::Device::texture_from_raw(
                        vk_image,
                        &texture_desc,
                        Some(drop_callback),
                    );

                    Ok(hal_texture)
                },
            );

        // Get the HAL texture from the closure result
        let hal_texture = hal_texture_result?;

        // Create wgpu texture descriptor
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("zero-copy AHardwareBuffer texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // Wrap the HAL texture as a wgpu::Texture
        let wgpu_texture =
            device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &texture_desc);

        info!("Successfully imported AHardwareBuffer as wgpu texture (zero-copy)");

        Ok(wgpu_texture)
    }

    /// Thread-local cache for the VulkanYuvPipeline, keyed by device handle.
    ///
    /// This is created once per device when first needed and reused for all
    /// subsequent YUV→RGBA conversions. Using thread_local to avoid synchronization.
    ///
    /// The cache stores (device_handle, pipeline) to detect device changes.
    /// If the wgpu device is dropped and recreated, the old pipeline becomes invalid
    /// and must be replaced. We detect this by comparing raw device handles.
    use std::cell::RefCell;
    thread_local! {
        /// Cache stores (device_handle, pipeline) - cleared on device mismatch
        static YUV_PIPELINE_CACHE: RefCell<Option<(u64, VulkanYuvPipeline)>> = const { RefCell::new(None) };
    }

    /// Imports a YUV AHardwareBuffer as a single RGBA texture (true zero-copy).
    ///
    /// This uses raw Vulkan to bypass wgpu's TextureAspect limitation and perform
    /// true zero-copy YUV→RGBA conversion on the GPU. The AHardwareBuffer memory
    /// is directly imported into Vulkan without any CPU copies.
    ///
    /// # Advantages over `import_ahardwarebuffer_yuv`
    ///
    /// - **True zero-copy**: No CPU memory reads/writes
    /// - **~186 MB/s bandwidth saved** at 1080p60 (1.5x pixel data not copied)
    /// - **Lower latency**: GPU does YUV→RGB conversion directly
    ///
    /// # Implementation
    ///
    /// 1. Creates a disjoint VkImage with `VK_IMAGE_CREATE_DISJOINT_BIT`
    /// 2. Imports AHardwareBuffer memory to each plane via `VkBindImagePlaneMemoryInfo`
    /// 3. Creates plane-specific VkImageViews (`PLANE_0`/`PLANE_1` aspects)
    /// 4. Runs a custom Vulkan render pass for YUV→RGB conversion
    /// 5. Returns the RGBA result as a wgpu texture
    ///
    /// # Safety
    ///
    /// - `ahardware_buffer` must be a valid YUV AHardwareBuffer
    /// - The AHardwareBuffer must remain valid until conversion completes
    ///
    /// # Returns
    ///
    /// A single RGBA texture containing the converted video frame.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not using Vulkan backend
    /// - The AHardwareBuffer is not a YUV format
    /// - Vulkan resource creation fails
    /// - The conversion render pass fails
    /// Imports a YUV AHardwareBuffer with true zero-copy GPU rendering.
    ///
    /// # Arguments
    /// * `device` - The wgpu device
    /// * `ahardware_buffer` - Raw AHardwareBuffer pointer from MediaCodec
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `color_space` - Optional YUV color space hint
    /// * `fence_fd` - Sync fence FD from producer (-1 if none/already signaled).
    ///   When >= 0, the fence is imported as a VkSemaphore and waited on before
    ///   sampling the AHB. This is critical for correct synchronization with
    ///   hardware video decoders. The FD is closed after import.
    pub unsafe fn import_ahardwarebuffer_yuv_zero_copy(
        device: &wgpu::Device,
        ahardware_buffer: AHardwareBufferPtr,
        width: u32,
        height: u32,
        color_space: Option<YuvColorSpace>,
        fence_fd: i32,
    ) -> Result<wgpu::Texture, ZeroCopyError> {
        if ahardware_buffer.is_null() {
            return Err(ZeroCopyError::InvalidResource(
                "AHardwareBuffer is null".to_string(),
            ));
        }

        // Access Vulkan HAL device
        let result = device
            .as_hal::<wgpu::hal::api::Vulkan, _, Result<wgpu::Texture, ZeroCopyError>>(
                |hal_device| {
                    let Some(hal_device) = hal_device else {
                        warn!("Failed to get Vulkan HAL device");
                        return Err(ZeroCopyError::HalAccessFailed(
                            "wgpu not using Vulkan backend".to_string(),
                        ));
                    };

                    let raw_device = hal_device.raw_device();
                    let instance = hal_device.shared_instance();
                    let raw_instance = instance.raw_instance();
                    let physical_device = hal_device.raw_physical_device();
                    let vk_queue = hal_device.raw_queue();
                    let queue_family_index = hal_device.queue_family_index();

                    // Get or create the YUV pipeline (cached per thread, keyed by device)
                    let current_device_handle = raw_device.handle().as_raw();
                    YUV_PIPELINE_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();

                        // Check if cached pipeline is for a different device (stale)
                        if let Some((cached_handle, _)) = cache.as_ref() {
                            if *cached_handle != current_device_handle {
                                debug!(
                                    "Device changed (0x{:x} -> 0x{:x}), clearing stale pipeline cache",
                                    cached_handle, current_device_handle
                                );
                                *cache = None;
                            }
                        }

                        // Create pipeline if not cached or was cleared
                        if cache.is_none() {
                            debug!("Creating VulkanYuvPipeline for device 0x{:x}", current_device_handle);
                            match VulkanYuvPipeline::new(raw_device.clone(), raw_instance, physical_device, queue_family_index) {
                                Ok(pipeline) => {
                                    *cache = Some((current_device_handle, pipeline));
                                }
                                Err(e) => {
                                    warn!("Failed to create VulkanYuvPipeline: {:?}", e);
                                    return Err(e);
                                }
                            }
                        }

                        let Some((_, pipeline)) = cache.as_ref() else {
                            // This should never happen - cache was just populated above
                            return Err(ZeroCopyError::ImportFailed(
                                "VulkanYuvPipeline cache unexpectedly empty".into(),
                            ));
                        };

                        // Perform the conversion
                        pipeline.convert_yuv_ahardwarebuffer(
                            device,
                            ahardware_buffer,
                            raw_instance,
                            physical_device,
                            vk_queue,
                            width,
                            height,
                            color_space,
                            fence_fd,
                        )
                    })
                },
            );

        result
    }

    /// Imports a YUV AHardwareBuffer as separate plane textures for shader-based conversion.
    ///
    /// This function handles both NV12 and YV12 formats:
    ///
    /// **NV12 (2-plane)** - the most common from MediaCodec:
    /// - Plane 0: Y (luma) - full resolution, R8Unorm
    /// - Plane 1: UV (chroma) - half resolution, RG8Unorm (interleaved)
    /// - Returns: `[Y, UV]`
    ///
    /// **YV12 (3-plane)** - planar format with V before U:
    /// - Plane 0: Y (luma) - full resolution, R8Unorm
    /// - Plane 1: V (chroma) - quarter resolution, R8Unorm
    /// - Plane 2: U (chroma) - quarter resolution, R8Unorm
    /// - Returns: `[Y, U, V]` (planes swapped to match Yuv420p shader order)
    ///
    /// # Implementation Strategy
    ///
    /// Instead of using VkSamplerYcbcrConversion (requires immutable samplers incompatible
    /// with wgpu's dynamic binding model), we create separate VkImageViews for each plane
    /// that can be sampled independently by the YUV->RGB shader.
    ///
    /// # Safety
    ///
    /// - `ahardware_buffer` must be a valid YUV AHardwareBuffer pointer
    /// - The AHardwareBuffer must remain valid for the lifetime of returned textures
    /// - The caller must maintain a reference (via AHardwareBuffer_acquire)
    ///
    /// # Returns
    ///
    /// Vector of plane textures:
    /// - NV12: `[Y (R8Unorm), UV (RG8Unorm)]`
    /// - YV12: `[Y (R8Unorm), U (R8Unorm), V (R8Unorm)]`
    pub unsafe fn import_ahardwarebuffer_yuv(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ahardware_buffer: AHardwareBufferPtr,
        width: u32,
        height: u32,
        hw_buffer_format: u32,
    ) -> Result<Vec<wgpu::Texture>, ZeroCopyError> {
        use crate::media::android_video::AHARDWAREBUFFER_FORMAT_YV12;
        let is_yv12 = hw_buffer_format == AHARDWAREBUFFER_FORMAT_YV12;
        if ahardware_buffer.is_null() {
            return Err(ZeroCopyError::InvalidResource(
                "AHardwareBuffer is null".to_string(),
            ));
        }

        // Implementation Strategy:
        //
        // True zero-copy YUV import would require:
        // 1. VkImage with VK_IMAGE_CREATE_DISJOINT_BIT and G8_B8R8_2PLANE_420_UNORM
        // 2. VkBindImagePlaneMemoryInfo for per-plane memory binding
        // 3. VkImageView with VK_IMAGE_ASPECT_PLANE_0_BIT / PLANE_1_BIT
        //
        // However, wgpu's TextureAspect enum doesn't include plane aspects, so we can't
        // create plane-specific texture views through wgpu. This is a wgpu limitation.
        //
        // As a workaround, we use AHardwareBuffer_lockPlanes to get CPU-accessible plane
        // pointers, copy the data, and create standard wgpu textures. This is more efficient
        // than full RGBA conversion because:
        // 1. We only copy Y and UV planes (1.5x pixel data vs 4x for RGBA)
        // 2. The YUV->RGB conversion happens on GPU via the NV12 shader
        //
        // Future: When wgpu adds multi-planar texture support with plane aspects,
        // this can be upgraded to true zero-copy using the Vulkan path above.

        // NDK AHardwareBuffer plane locking structures
        #[repr(C)]
        struct ARect {
            left: i32,
            top: i32,
            right: i32,
            bottom: i32,
        }

        #[repr(C)]
        struct AHardwareBufferPlane {
            data: *mut u8,
            pixel_stride: u32,
            row_stride: u32,
        }

        #[repr(C)]
        struct AHardwareBufferPlanes {
            plane_count: u32,
            planes: [AHardwareBufferPlane; 4],
        }

        // NDK function declarations
        extern "C" {
            fn AHardwareBuffer_lockPlanes(
                buffer: *mut std::ffi::c_void,
                usage: u64,
                fence: i32,
                rect: *const ARect,
                out_planes: *mut AHardwareBufferPlanes,
            ) -> i32;

            fn AHardwareBuffer_unlock(buffer: *mut std::ffi::c_void, fence: *mut i32) -> i32;
        }

        // AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN = 0x00000003
        const AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN: u64 = 3;

        let rect = ARect {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };

        let mut planes_info = std::mem::zeroed::<AHardwareBufferPlanes>();

        let result = AHardwareBuffer_lockPlanes(
            ahardware_buffer,
            AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
            -1, // no fence
            &rect,
            &mut planes_info,
        );

        if result != 0 {
            // Lock failed - buffer might not support CPU access
            // This is expected for some hardware-only buffers
            debug!(
                "AHardwareBuffer_lockPlanes failed with code {}. \
                 Buffer may not support CPU access. Using CPU fallback.",
                result
            );
            return Err(ZeroCopyError::NotAvailable(format!(
                "AHardwareBuffer_lockPlanes failed: {}. Buffer does not support CPU read access.",
                result
            )));
        }

        // Verify we have the expected number of planes
        let expected_planes = if is_yv12 { 3 } else { 2 };
        if planes_info.plane_count < expected_planes {
            AHardwareBuffer_unlock(ahardware_buffer, std::ptr::null_mut());
            let format_name = if is_yv12 { "YV12" } else { "NV12" };
            warn!(
                "YUV buffer has {} planes, expected at least {} for {}",
                planes_info.plane_count, expected_planes, format_name
            );
            return Err(ZeroCopyError::InvalidResource(format!(
                "YUV buffer has {} planes, expected at least {} for {}",
                planes_info.plane_count, expected_planes, format_name
            )));
        }

        let y_plane = &planes_info.planes[0];

        // Validate Y plane pointer
        if y_plane.data.is_null() {
            AHardwareBuffer_unlock(ahardware_buffer, std::ptr::null_mut());
            return Err(ZeroCopyError::InvalidResource(
                "Y plane data pointer is null".to_string(),
            ));
        }

        // Validate chroma plane pointers based on format
        if is_yv12 {
            // YV12: planes 1 (V) and 2 (U) must be valid
            if planes_info.planes[1].data.is_null() || planes_info.planes[2].data.is_null() {
                AHardwareBuffer_unlock(ahardware_buffer, std::ptr::null_mut());
                return Err(ZeroCopyError::InvalidResource(
                    "YV12 chroma plane data pointer is null".to_string(),
                ));
            }
        } else {
            // NV12: plane 1 (UV interleaved) must be valid
            if planes_info.planes[1].data.is_null() {
                AHardwareBuffer_unlock(ahardware_buffer, std::ptr::null_mut());
                return Err(ZeroCopyError::InvalidResource(
                    "NV12 UV plane data pointer is null".to_string(),
                ));
            }
        }

        // Copy Y plane data (full resolution, R8)
        let y_row_stride = y_plane.row_stride as usize;
        let y_height = height as usize;
        let y_copy_width = width as usize; // actual pixel width to copy

        let mut y_data = Vec::with_capacity(y_row_stride * y_height);
        for row in 0..y_height {
            let src_ptr = y_plane.data.add(row * y_row_stride);
            let row_slice = std::slice::from_raw_parts(src_ptr, y_copy_width);
            y_data.extend_from_slice(row_slice);
            // Pad to match wgpu's expected row alignment if needed
            // wgpu requires COPY_BYTES_PER_ROW_ALIGNMENT (256) alignment for buffer copies
            // but write_texture handles this internally
        }

        // Chroma plane dimensions (half resolution for 4:2:0)
        let chroma_width = (width as usize + 1) / 2;
        let chroma_height = (height as usize + 1) / 2;

        // Extract chroma data based on format
        let (u_data, v_data, uv_data) = if is_yv12 {
            // YV12: 3-plane format with separate V and U planes
            // Plane order in buffer: Y, V, U
            // We extract and will return as [Y, U, V] to match Yuv420p shader expectations
            let v_plane = &planes_info.planes[1]; // V comes first in YV12
            let u_plane = &planes_info.planes[2]; // U comes second

            let v_row_stride = v_plane.row_stride as usize;
            let u_row_stride = u_plane.row_stride as usize;

            let mut v_data = Vec::with_capacity(chroma_width * chroma_height);
            for row in 0..chroma_height {
                let src_ptr = v_plane.data.add(row * v_row_stride);
                let row_slice = std::slice::from_raw_parts(src_ptr, chroma_width);
                v_data.extend_from_slice(row_slice);
            }

            let mut u_data = Vec::with_capacity(chroma_width * chroma_height);
            for row in 0..chroma_height {
                let src_ptr = u_plane.data.add(row * u_row_stride);
                let row_slice = std::slice::from_raw_parts(src_ptr, chroma_width);
                u_data.extend_from_slice(row_slice);
            }

            (Some(u_data), Some(v_data), None)
        } else {
            // NV12: 2-plane format with interleaved UV
            let uv_plane = &planes_info.planes[1];
            let uv_row_stride = uv_plane.row_stride as usize;
            let uv_copy_width = width as usize; // UV plane has width/2 pixels, but each is 2 bytes (RG)

            let mut uv_data = Vec::with_capacity(uv_row_stride * chroma_height);
            for row in 0..chroma_height {
                let src_ptr = uv_plane.data.add(row * uv_row_stride);
                let row_slice = std::slice::from_raw_parts(src_ptr, uv_copy_width);
                uv_data.extend_from_slice(row_slice);
            }

            (None, None, Some(uv_data))
        };

        // Unlock the buffer - we've copied the data
        let unlock_result = AHardwareBuffer_unlock(ahardware_buffer, std::ptr::null_mut());
        if unlock_result != 0 {
            warn!("AHardwareBuffer_unlock returned {}", unlock_result);
            // Continue anyway - we have the data
        }

        // Create Y texture (R8Unorm, full resolution)
        let y_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Android YUV Y plane"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload Y plane data
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &y_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &y_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width), // R8 = 1 byte per pixel
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        let chroma_width_u32 = chroma_width as u32;
        let chroma_height_u32 = chroma_height as u32;

        if is_yv12 {
            // YV12: Create separate U and V textures (R8Unorm each)
            // Safety: u_data and v_data are guaranteed Some when is_yv12 is true (set above)
            let (Some(u_data), Some(v_data)) = (u_data, v_data) else {
                return Err(ZeroCopyError::ImportFailed(
                    "YV12 plane data missing (internal error)".into(),
                ));
            };

            let u_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Android YV12 U plane"),
                size: wgpu::Extent3d {
                    width: chroma_width_u32,
                    height: chroma_height_u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &u_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &u_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(chroma_width_u32),
                    rows_per_image: Some(chroma_height_u32),
                },
                wgpu::Extent3d {
                    width: chroma_width_u32,
                    height: chroma_height_u32,
                    depth_or_array_layers: 1,
                },
            );

            let v_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Android YV12 V plane"),
                size: wgpu::Extent3d {
                    width: chroma_width_u32,
                    height: chroma_height_u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &v_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &v_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(chroma_width_u32),
                    rows_per_image: Some(chroma_height_u32),
                },
                wgpu::Extent3d {
                    width: chroma_width_u32,
                    height: chroma_height_u32,
                    depth_or_array_layers: 1,
                },
            );

            debug!(
                "Extracted YV12 planes from AHardwareBuffer: Y={}x{}, U={}x{}, V={}x{}",
                width, height, chroma_width, chroma_height, chroma_width, chroma_height
            );

            info!(
                "Imported Android YV12 AHardwareBuffer as plane textures: Y={}x{} R8, U={}x{} R8, V={}x{} R8",
                width, height, chroma_width_u32, chroma_height_u32, chroma_width_u32, chroma_height_u32
            );

            // Return [Y, U, V] plane textures for the Yuv420p shader
            // Note: We swapped U and V from the YV12 buffer order (Y, V, U) to match
            // the shader's expected order (Y, U, V)
            Ok(vec![y_texture, u_texture, v_texture])
        } else {
            // NV12: Create single UV texture (Rg8Unorm, interleaved)
            // Safety: uv_data is guaranteed Some when is_yv12 is false (set above)
            let Some(uv_data) = uv_data else {
                return Err(ZeroCopyError::ImportFailed(
                    "NV12 UV plane data missing (internal error)".into(),
                ));
            };

            let uv_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Android NV12 UV plane"),
                size: wgpu::Extent3d {
                    width: chroma_width_u32,
                    height: chroma_height_u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &uv_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &uv_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(chroma_width_u32 * 2), // Rg8 = 2 bytes per pixel
                    rows_per_image: Some(chroma_height_u32),
                },
                wgpu::Extent3d {
                    width: chroma_width_u32,
                    height: chroma_height_u32,
                    depth_or_array_layers: 1,
                },
            );

            debug!(
                "Extracted NV12 planes from AHardwareBuffer: Y={}x{} stride={}, UV={}x{}",
                width, height, y_row_stride, chroma_width, chroma_height
            );

            info!(
                "Imported Android NV12 AHardwareBuffer as plane textures: Y={}x{} R8, UV={}x{} RG8",
                width, height, chroma_width_u32, chroma_height_u32
            );

            // Return [Y, UV] plane textures for the NV12 shader
            Ok(vec![y_texture, uv_texture])
        }
    }

    /// Finds a suitable memory type index for the given requirements.
    fn find_memory_type_index(
        type_bits: u32,
        required_flags: vk::MemoryPropertyFlags,
        mem_properties: &vk::PhysicalDeviceMemoryProperties,
    ) -> Option<u32> {
        for i in 0..mem_properties.memory_type_count {
            let type_bit = 1 << i;
            let is_required_type = (type_bits & type_bit) != 0;
            // Use safe .get() to avoid potential panic if memory_type_count mismatches array
            let has_required_flags = mem_properties
                .memory_types
                .get(i as usize)
                .map(|mt| (mt.property_flags & required_flags) == required_flags)
                .unwrap_or(false);

            if is_required_type && has_required_flags {
                return Some(i);
            }
        }
        None
    }

    /// Checks if a Vulkan format is a YCbCr/YUV multi-planar format.
    ///
    /// These formats require VkSamplerYcbcrConversion for proper sampling, which involves:
    /// - Creating a VkSamplerYcbcrConversion object with color space and range parameters
    /// - Including VkSamplerYcbcrConversionInfo in VkImageViewCreateInfo
    /// - Including VkSamplerYcbcrConversionInfo in VkSamplerCreateInfo
    /// - Using immutable samplers in the descriptor set layout
    ///
    /// Common YCbCr formats from Android AHardwareBuffer:
    /// - G8_B8R8_2PLANE_420_UNORM: NV12 format (most common from video/camera)
    /// - G8_B8_R8_3PLANE_420_UNORM: I420/YV12 format
    /// - G8_B8R8_2PLANE_422_UNORM: NV16 format
    /// - G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: 10-bit HDR video
    fn is_ycbcr_format(format: vk::Format) -> bool {
        matches!(
            format,
            // 2-plane 4:2:0 formats (NV12, P010, etc.)
            vk::Format::G8_B8R8_2PLANE_420_UNORM
                | vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                | vk::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16
                | vk::Format::G16_B16R16_2PLANE_420_UNORM
                // 3-plane 4:2:0 formats (I420, YV12, etc.)
                | vk::Format::G8_B8_R8_3PLANE_420_UNORM
                | vk::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16
                | vk::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16
                | vk::Format::G16_B16_R16_3PLANE_420_UNORM
                // 2-plane 4:2:2 formats (NV16, P210, etc.)
                | vk::Format::G8_B8R8_2PLANE_422_UNORM
                | vk::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16
                | vk::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16
                | vk::Format::G16_B16R16_2PLANE_422_UNORM
                // 3-plane 4:2:2 formats
                | vk::Format::G8_B8_R8_3PLANE_422_UNORM
                | vk::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16
                | vk::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16
                | vk::Format::G16_B16_R16_3PLANE_422_UNORM
                // 2-plane 4:4:4 formats
                | vk::Format::G8_B8R8_2PLANE_444_UNORM
                | vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
                | vk::Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16
                | vk::Format::G16_B16R16_2PLANE_444_UNORM
                // 3-plane 4:4:4 formats
                | vk::Format::G8_B8_R8_3PLANE_444_UNORM
                | vk::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16
                | vk::Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16
                | vk::Format::G16_B16_R16_3PLANE_444_UNORM
        )
    }

    /// Executes a one-shot command buffer to transition an image layout and perform
    /// queue family ownership transfer for externally imported memory.
    ///
    /// This is required for external memory imports (AHardwareBuffer) to ensure proper
    /// synchronization and layout before the image can be sampled in shaders.
    ///
    /// # Safety
    ///
    /// - `vk_device` must be a valid Vulkan device handle
    /// - `vk_queue` must be a valid queue from the same device
    /// - `vk_image` must be a valid image created on this device
    /// - `queue_family_index` must be the queue family index of `vk_queue`
    unsafe fn transition_image_layout_external(
        vk_device: &ash::Device,
        vk_queue: vk::Queue,
        queue_family_index: u32,
        vk_image: vk::Image,
    ) -> Result<(), ZeroCopyError> {
        // Create a one-shot command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);

        let command_pool = vk_device
            .create_command_pool(&pool_info, None)
            .map_err(|e| {
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to create command pool for layout transition: {:?}",
                    e
                ))
            })?;

        // Allocate a command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = vk_device
            .allocate_command_buffers(&alloc_info)
            .map_err(|e| {
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to allocate command buffer for layout transition: {:?}",
                    e
                ))
            })?;
        let cmd_buf = *command_buffers.first().ok_or_else(|| {
            vk_device.destroy_command_pool(command_pool, None);
            ZeroCopyError::TextureCreationFailed(
                "No command buffer returned from allocation".to_string(),
            )
        })?;

        // Begin the command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        vk_device
            .begin_command_buffer(cmd_buf, &begin_info)
            .map_err(|e| {
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to begin command buffer: {:?}",
                    e
                ))
            })?;

        // Record the image memory barrier for layout transition and queue ownership transfer
        // VK_QUEUE_FAMILY_EXTERNAL is defined as (~1U) = 0xFFFFFFFE
        const VK_QUEUE_FAMILY_EXTERNAL: u32 = !1u32;

        let image_barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ) // External memory may have been read
            .dst_access_mask(vk::AccessFlags::SHADER_READ) // Will be sampled in fragment shader
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(VK_QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(queue_family_index)
            .image(vk_image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        vk_device.cmd_pipeline_barrier(
            cmd_buf,
            vk::PipelineStageFlags::TOP_OF_PIPE, // No prior stage within this queue
            vk::PipelineStageFlags::FRAGMENT_SHADER, // Will be used in fragment shader
            vk::DependencyFlags::empty(),
            &[], // No memory barriers
            &[], // No buffer barriers
            &[image_barrier],
        );

        // End and submit the command buffer
        vk_device.end_command_buffer(cmd_buf).map_err(|e| {
            vk_device.destroy_command_pool(command_pool, None);
            ZeroCopyError::TextureCreationFailed(format!("Failed to end command buffer: {:?}", e))
        })?;

        let cmd_bufs = [cmd_buf];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs);

        // Create a fence to wait for completion
        let fence_info = vk::FenceCreateInfo::default();
        let fence = vk_device.create_fence(&fence_info, None).map_err(|e| {
            vk_device.destroy_command_pool(command_pool, None);
            ZeroCopyError::TextureCreationFailed(format!(
                "Failed to create fence for layout transition: {:?}",
                e
            ))
        })?;

        vk_device
            .queue_submit(vk_queue, &[submit_info], fence)
            .map_err(|e| {
                vk_device.destroy_fence(fence, None);
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to submit layout transition command: {:?}",
                    e
                ))
            })?;

        // Wait for the command to complete (with 1 second timeout)
        vk_device
            .wait_for_fences(&[fence], true, 1_000_000_000)
            .map_err(|e| {
                vk_device.destroy_fence(fence, None);
                vk_device.destroy_command_pool(command_pool, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Timeout waiting for layout transition: {:?}",
                    e
                ))
            })?;

        // Cleanup
        vk_device.destroy_fence(fence, None);
        vk_device.destroy_command_pool(command_pool, None);

        debug!("Successfully transitioned image layout to SHADER_READ_ONLY_OPTIMAL with queue ownership transfer");
        Ok(())
    }

    /// Converts wgpu TextureFormat to Vulkan VkFormat.
    ///
    /// Returns an error for unsupported formats instead of silently defaulting.
    /// Android MediaCodec typically outputs BGRA8 or RGBA8 for video frames
    /// when using AHardwareBuffer, so these are the primary supported formats.
    ///
    /// # Errors
    ///
    /// Returns `ZeroCopyError::InvalidResource` for unsupported texture formats.
    pub fn wgpu_format_to_vulkan(format: wgpu::TextureFormat) -> Result<vk::Format, ZeroCopyError> {
        match format {
            wgpu::TextureFormat::Bgra8Unorm => Ok(vk::Format::B8G8R8A8_UNORM),
            wgpu::TextureFormat::Rgba8Unorm => Ok(vk::Format::R8G8B8A8_UNORM),
            wgpu::TextureFormat::R8Unorm => Ok(vk::Format::R8_UNORM),
            wgpu::TextureFormat::Rg8Unorm => Ok(vk::Format::R8G8_UNORM),
            _ => {
                warn!("Unsupported texture format {:?}; refusing import", format);
                Err(ZeroCopyError::InvalidResource(format!(
                    "Unsupported texture format {:?}",
                    format
                )))
            }
        }
    }

    // =========================================================================
    // VulkanYuvPipeline: True zero-copy YUV→RGB conversion via raw Vulkan
    // =========================================================================

    /// Vulkan YUV processing pipeline for true zero-copy NV12 import.
    ///
    /// This bypasses wgpu's `TextureAspect` limitation (which lacks `Plane0`/`Plane1`
    /// variants) by using raw Vulkan (ash) to:
    /// 1. Create multi-plane VkImage with `VK_IMAGE_CREATE_DISJOINT_BIT`
    /// 2. Create plane-specific VkImageViews with `PLANE_0`/`PLANE_1` aspects
    /// 3. Perform YUV→RGB conversion via a custom render pass
    /// 4. Return the RGB output as a wgpu texture
    ///
    /// The pipeline is created once per device and reused for all YUV frames.
    pub struct VulkanYuvPipeline {
        /// Vertex shader module (fullscreen triangle)
        vert_shader: vk::ShaderModule,
        /// Fragment shader module (YUV→RGB conversion)
        frag_shader: vk::ShaderModule,
        /// Descriptor set layout for Y/UV texture samplers + YUV params uniform
        descriptor_set_layout: vk::DescriptorSetLayout,
        /// Pipeline layout
        pipeline_layout: vk::PipelineLayout,
        /// Render pass for converting to RGBA output
        render_pass: vk::RenderPass,
        /// Graphics pipeline
        pipeline: vk::Pipeline,
        /// Descriptor pool for allocating descriptor sets per-frame
        descriptor_pool: vk::DescriptorPool,
        /// Shared sampler for Y and UV planes
        sampler: vk::Sampler,
        /// Uniform buffer for YUV→RGB conversion parameters
        yuv_params_buffer: vk::Buffer,
        /// Memory backing the uniform buffer
        yuv_params_memory: vk::DeviceMemory,
        /// Command pool for conversion commands
        command_pool: vk::CommandPool,
        /// Queue family index for command submission
        queue_family_index: u32,
        /// Device handle for cleanup and commands
        device: ash::Device,
        /// Instance handle for loading device extension functions
        instance: ash::Instance,
    }

    /// Color space and range for YUV→RGB conversion.
    ///
    /// Most HD content uses BT.709 limited range. SD content (480i/576i) typically
    /// uses BT.601. Full range is used when video is explicitly marked as such.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub enum YuvColorSpace {
        /// BT.709 limited range (16-235 Y, 16-240 UV) - default for HD content
        #[default]
        Bt709Limited,
        /// BT.709 full range (0-255 for all components)
        Bt709Full,
        /// BT.601 limited range (16-235 Y, 16-240 UV) - for SD content
        Bt601Limited,
        /// BT.601 full range (0-255 for all components)
        Bt601Full,
    }

    /// YUV→RGB conversion parameters passed to the shader.
    /// Layout matches the uniform buffer expected by the fragment shader.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct YuvParams {
        /// 3x3 YUV to RGB conversion matrix (column-major for GLSL mat3)
        /// Padded to vec4 per column for std140 layout
        yuv_to_rgb: [[f32; 4]; 3],
        /// Offset to subtract from normalized [0,1] YUV values
        yuv_offset: [f32; 3],
        /// Padding for std140 alignment
        _padding: f32,
    }

    impl YuvParams {
        /// BT.709 limited range (16-235 Y, 16-240 UV)
        /// Used for most HD content
        const BT709_LIMITED: Self = Self {
            // Matrix converts from offset-adjusted YUV to RGB
            // Column 0: Y coefficients for R, G, B
            // Column 1: U coefficients for R, G, B
            // Column 2: V coefficients for R, G, B
            yuv_to_rgb: [
                [1.164, 1.164, 1.164, 0.0],  // Y contribution
                [0.0, -0.1873, 1.8556, 0.0], // U contribution
                [1.5748, -0.4681, 0.0, 0.0], // V contribution
            ],
            // Offsets: Y scaled from 16-235, UV centered at 128
            yuv_offset: [16.0 / 255.0, 0.5, 0.5],
            _padding: 0.0,
        };

        /// BT.709 full range (0-255 for all components)
        /// Used when video is explicitly marked as full range
        const BT709_FULL: Self = Self {
            yuv_to_rgb: [
                [1.0, 1.0, 1.0, 0.0],        // Y contribution (no scaling)
                [0.0, -0.1873, 1.8556, 0.0], // U contribution
                [1.5748, -0.4681, 0.0, 0.0], // V contribution
            ],
            // Offsets: Y at 0, UV centered at 128
            yuv_offset: [0.0, 0.5, 0.5],
            _padding: 0.0,
        };

        /// BT.601 limited range (16-235 Y, 16-240 UV)
        /// Used for SD content (480i/576i)
        const BT601_LIMITED: Self = Self {
            // BT.601 coefficients differ from BT.709 in chroma contribution
            yuv_to_rgb: [
                [1.164, 1.164, 1.164, 0.0], // Y contribution (same scaling as BT.709 limited)
                [0.0, -0.3917, 2.0172, 0.0], // U contribution (different from BT.709)
                [1.5960, -0.8130, 0.0, 0.0], // V contribution (different from BT.709)
            ],
            // Offsets: Y scaled from 16-235, UV centered at 128
            yuv_offset: [16.0 / 255.0, 0.5, 0.5],
            _padding: 0.0,
        };

        /// BT.601 full range (0-255 for all components)
        const BT601_FULL: Self = Self {
            yuv_to_rgb: [
                [1.0, 1.0, 1.0, 0.0],        // Y contribution (no scaling)
                [0.0, -0.3917, 2.0172, 0.0], // U contribution
                [1.5960, -0.8130, 0.0, 0.0], // V contribution
            ],
            // Offsets: Y at 0, UV centered at 128
            yuv_offset: [0.0, 0.5, 0.5],
            _padding: 0.0,
        };

        /// Get YuvParams for a given color space
        pub fn for_color_space(color_space: YuvColorSpace) -> Self {
            match color_space {
                YuvColorSpace::Bt709Limited => Self::BT709_LIMITED,
                YuvColorSpace::Bt709Full => Self::BT709_FULL,
                YuvColorSpace::Bt601Limited => Self::BT601_LIMITED,
                YuvColorSpace::Bt601Full => Self::BT601_FULL,
            }
        }
    }

    impl VulkanYuvPipeline {
        /// Creates a new VulkanYuvPipeline.
        ///
        /// # Safety
        ///
        /// - `device` must be a valid Vulkan device
        /// - `raw_instance` must be the instance that created the device
        /// - `physical_device` must be the physical device backing `device`
        /// - `queue_family_index` must be a valid graphics queue family
        pub unsafe fn new(
            device: ash::Device,
            raw_instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            queue_family_index: u32,
        ) -> Result<Self, ZeroCopyError> {
            debug!("Creating VulkanYuvPipeline for zero-copy YUV conversion");

            // Query device properties to check Vulkan/SPIR-V version support
            let device_props = raw_instance.get_physical_device_properties(physical_device);
            let api_version = device_props.api_version;
            let api_major = vk::api_version_major(api_version);
            let api_minor = vk::api_version_minor(api_version);
            let api_patch = vk::api_version_patch(api_version);

            tracing::info!(
                "VulkanYuvPipeline: Device '{}' supports Vulkan {}.{}.{}",
                std::ffi::CStr::from_ptr(device_props.device_name.as_ptr()).to_string_lossy(),
                api_major,
                api_minor,
                api_patch
            );

            // SPIR-V version requirements:
            // - Vulkan 1.0 supports SPIR-V 1.0
            // - Vulkan 1.1 supports SPIR-V 1.0-1.3
            // - Vulkan 1.2 supports SPIR-V 1.0-1.5
            // Our shaders are compiled for SPIR-V 1.5 which requires Vulkan 1.2
            if api_major == 1 && api_minor < 2 {
                tracing::warn!(
                    "Device only supports Vulkan {}.{}, but SPIR-V 1.5 shaders require Vulkan 1.2+. \
                     Pipeline creation may fail.",
                    api_major,
                    api_minor
                );
            }

            // Query memory properties for proper uniform buffer allocation
            let mem_properties =
                raw_instance.get_physical_device_memory_properties(physical_device);

            // Create vertex shader module (fullscreen triangle)
            let vert_shader = Self::create_shader_module(&device, Self::VERT_SHADER_SPIRV)?;

            // Create fragment shader module (YUV→RGB conversion)
            let frag_shader = Self::create_shader_module(&device, Self::FRAG_SHADER_SPIRV)?;

            // Create sampler for texture sampling
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .min_lod(0.0)
                .max_lod(0.0);

            let sampler = device.create_sampler(&sampler_info, None).map_err(|e| {
                device.destroy_shader_module(vert_shader, None);
                device.destroy_shader_module(frag_shader, None);
                ZeroCopyError::TextureCreationFailed(format!("Failed to create sampler: {:?}", e))
            })?;

            // Create descriptor set layout with bindings for Y/UV textures and YUV params
            let bindings = [
                // Binding 0: Y plane (R8)
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                // Binding 1: UV plane (RG8)
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                // Binding 2: YUV conversion parameters (mat3 + vec3 offset)
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

            let descriptor_set_layout = device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| {
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create descriptor set layout: {:?}",
                        e
                    ))
                })?;

            // Create pipeline layout
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| {
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create pipeline layout: {:?}",
                        e
                    ))
                })?;

            // Create render pass (single color attachment for RGBA output)
            let color_attachment = vk::AttachmentDescription::default()
                .format(vk::Format::R8G8B8A8_UNORM)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let color_attachment_ref = vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::slice::from_ref(&color_attachment_ref));

            let render_pass_info = vk::RenderPassCreateInfo::default()
                .attachments(std::slice::from_ref(&color_attachment))
                .subpasses(std::slice::from_ref(&subpass));

            let render_pass = device
                .create_render_pass(&render_pass_info, None)
                .map_err(|e| {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create render pass: {:?}",
                        e
                    ))
                })?;

            // Create graphics pipeline
            let pipeline = Self::create_pipeline(
                &device,
                vert_shader,
                frag_shader,
                pipeline_layout,
                render_pass,
            )
            .map_err(|e| {
                device.destroy_render_pass(render_pass, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_sampler(sampler, None);
                device.destroy_shader_module(vert_shader, None);
                device.destroy_shader_module(frag_shader, None);
                e
            })?;

            // Create descriptor pool (allow up to 8 concurrent conversions)
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(16), // 2 textures * 8 frames
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(8), // 1 uniform buffer * 8 frames
            ];

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(8)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

            let descriptor_pool = device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_render_pass(render_pass, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create descriptor pool: {:?}",
                        e
                    ))
                })?;

            // Create command pool
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .map_err(|e| {
                    device.destroy_descriptor_pool(descriptor_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_render_pass(render_pass, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create command pool: {:?}",
                        e
                    ))
                })?;

            // Create uniform buffer for YUV params (sized for YuvParams struct)
            let yuv_params_size = std::mem::size_of::<YuvParams>() as vk::DeviceSize;
            let buffer_info = vk::BufferCreateInfo::default()
                .size(yuv_params_size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let yuv_params_buffer = device.create_buffer(&buffer_info, None).map_err(|e| {
                device.destroy_command_pool(command_pool, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_render_pass(render_pass, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_sampler(sampler, None);
                device.destroy_shader_module(vert_shader, None);
                device.destroy_shader_module(frag_shader, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to create YUV params buffer: {:?}",
                    e
                ))
            })?;

            // Get memory requirements and allocate
            let mem_requirements = device.get_buffer_memory_requirements(yuv_params_buffer);

            // Find a host-visible, host-coherent memory type for the uniform buffer
            // HOST_VISIBLE allows CPU mapping, HOST_COHERENT avoids explicit flush/invalidate
            let memory_type_index = find_memory_type_index(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                &mem_properties,
            )
            .ok_or_else(|| {
                device.destroy_buffer(yuv_params_buffer, None);
                device.destroy_command_pool(command_pool, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_render_pass(render_pass, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_sampler(sampler, None);
                device.destroy_shader_module(vert_shader, None);
                device.destroy_shader_module(frag_shader, None);
                ZeroCopyError::TextureCreationFailed(
                    "No HOST_VISIBLE | HOST_COHERENT memory type for uniform buffer".to_string(),
                )
            })?;

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_requirements.size)
                .memory_type_index(memory_type_index);

            let yuv_params_memory = device.allocate_memory(&alloc_info, None).map_err(|e| {
                device.destroy_buffer(yuv_params_buffer, None);
                device.destroy_command_pool(command_pool, None);
                device.destroy_descriptor_pool(descriptor_pool, None);
                device.destroy_pipeline(pipeline, None);
                device.destroy_render_pass(render_pass, None);
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                device.destroy_sampler(sampler, None);
                device.destroy_shader_module(vert_shader, None);
                device.destroy_shader_module(frag_shader, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to allocate YUV params memory: {:?}",
                    e
                ))
            })?;

            // Bind buffer to memory
            device
                .bind_buffer_memory(yuv_params_buffer, yuv_params_memory, 0)
                .map_err(|e| {
                    device.free_memory(yuv_params_memory, None);
                    device.destroy_buffer(yuv_params_buffer, None);
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_descriptor_pool(descriptor_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_render_pass(render_pass, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to bind YUV params buffer memory: {:?}",
                        e
                    ))
                })?;

            // Initialize with BT.709 limited range params (most common for HD content)
            let params_ptr = device
                .map_memory(
                    yuv_params_memory,
                    0,
                    yuv_params_size,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| {
                    device.free_memory(yuv_params_memory, None);
                    device.destroy_buffer(yuv_params_buffer, None);
                    device.destroy_command_pool(command_pool, None);
                    device.destroy_descriptor_pool(descriptor_pool, None);
                    device.destroy_pipeline(pipeline, None);
                    device.destroy_render_pass(render_pass, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                    device.destroy_shader_module(vert_shader, None);
                    device.destroy_shader_module(frag_shader, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to map YUV params memory: {:?}",
                        e
                    ))
                })?;

            std::ptr::copy_nonoverlapping(
                &YuvParams::BT709_LIMITED as *const YuvParams,
                params_ptr as *mut YuvParams,
                1,
            );
            device.unmap_memory(yuv_params_memory);

            info!("VulkanYuvPipeline created successfully");

            Ok(Self {
                vert_shader,
                frag_shader,
                descriptor_set_layout,
                pipeline_layout,
                render_pass,
                pipeline,
                descriptor_pool,
                sampler,
                yuv_params_buffer,
                yuv_params_memory,
                command_pool,
                queue_family_index,
                device,
                instance: raw_instance.clone(),
            })
        }

        /// Updates the uniform buffer with parameters for the specified color space.
        ///
        /// Returns `true` if the update succeeded, `false` if memory mapping failed.
        /// On failure, the previous color space parameters remain in effect.
        ///
        /// # Safety
        ///
        /// The device and yuv_params_memory must still be valid.
        pub unsafe fn update_color_space(&self, color_space: YuvColorSpace) -> bool {
            let params = YuvParams::for_color_space(color_space);
            let params_size = std::mem::size_of::<YuvParams>() as vk::DeviceSize;

            match self.device.map_memory(
                self.yuv_params_memory,
                0,
                params_size,
                vk::MemoryMapFlags::empty(),
            ) {
                Ok(params_ptr) => {
                    std::ptr::copy_nonoverlapping(
                        &params as *const YuvParams,
                        params_ptr as *mut YuvParams,
                        1,
                    );
                    self.device.unmap_memory(self.yuv_params_memory);
                    true
                }
                Err(e) => {
                    warn!(
                        "Failed to map YUV params memory for color space {:?}: {:?}. \
                         Using previous color space parameters.",
                        color_space, e
                    );
                    false
                }
            }
        }

        /// Creates a shader module from SPIR-V bytes.
        unsafe fn create_shader_module(
            device: &ash::Device,
            spirv: &[u8],
        ) -> Result<vk::ShaderModule, ZeroCopyError> {
            // SPIR-V must have length divisible by 4
            if spirv.len() % 4 != 0 {
                return Err(ZeroCopyError::TextureCreationFailed(
                    "SPIR-V code length must be divisible by 4".to_string(),
                ));
            }

            // Copy bytes into a properly aligned Vec<u32> to avoid alignment issues.
            // The embedded SPIR-V as &[u8] isn't guaranteed to be 4-byte aligned.
            let mut code: Vec<u32> = vec![0u32; spirv.len() / 4];
            for (i, chunk) in spirv.chunks_exact(4).enumerate() {
                code[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }

            let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);

            device
                .create_shader_module(&shader_info, None)
                .map_err(|e| {
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create shader module: {:?}",
                        e
                    ))
                })
        }

        /// Creates the graphics pipeline for YUV→RGB conversion.
        unsafe fn create_pipeline(
            device: &ash::Device,
            vert_shader: vk::ShaderModule,
            frag_shader: vk::ShaderModule,
            pipeline_layout: vk::PipelineLayout,
            render_pass: vk::RenderPass,
        ) -> Result<vk::Pipeline, ZeroCopyError> {
            let entry_point = CStr::from_bytes_with_nul_unchecked(b"main\0");

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_shader)
                    .name(entry_point),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_shader)
                    .name(entry_point),
            ];

            // No vertex input - fullscreen triangle generated in vertex shader
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            // Dynamic viewport and scissor (set at draw time)
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);

            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);

            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&color_blend_attachment));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_state)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multisample_state)
                .color_blend_state(&color_blend_state)
                .dynamic_state(&dynamic_state)
                .layout(pipeline_layout)
                .render_pass(render_pass)
                .subpass(0);

            let pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| {
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create graphics pipeline: {:?}",
                        e
                    ))
                })?;

            pipelines.into_iter().next().ok_or_else(|| {
                ZeroCopyError::TextureCreationFailed(
                    "create_graphics_pipelines returned empty vector".to_string(),
                )
            })
        }

        // =====================================================================
        // Embedded SPIR-V shaders (pre-compiled)
        // =====================================================================
        //
        // These are pre-compiled SPIR-V bytecode for the YUV→RGB conversion.
        // Source files kept in src/media/shaders/ for reference.
        //
        // SPIR-V Version Choice: 1.0 (Vulkan 1.0 compatible)
        // --------------------------------------------------
        // We use SPIR-V 1.0 for maximum Android device compatibility:
        // - SPIR-V 1.0 → Vulkan 1.0+ (all Vulkan-capable devices)
        // - SPIR-V 1.5 → Vulkan 1.2+ only (excludes many mid-range devices)
        //
        // Performance impact: Negligible for our use case.
        // Our YUV→RGB shader is ~20 instructions of pure arithmetic (sample textures,
        // matrix multiply, clamp). The bottleneck is memory bandwidth (texture reads,
        // framebuffer writes), not shader ALU. SPIR-V 1.5 optimizations primarily
        // benefit complex compute shaders with subgroup operations, not simple
        // per-pixel fragment shaders like ours. Frame render time is identical.
        //
        // Compile command: glslc --target-env=vulkan1.0 -O -fshader-stage=<stage> <file>

        /// Fullscreen triangle vertex shader (SPIR-V).
        ///
        /// Generates a fullscreen triangle from vertex ID without vertex buffers:
        /// - Vertex 0: (-1, -1), UV (0, 0)
        /// - Vertex 1: (3, -1), UV (2, 0)
        /// - Vertex 2: (-1, 3), UV (0, 2)
        ///
        /// GLSL source: `src/media/shaders/yuv_fullscreen.vert`
        #[rustfmt::skip]
        const VERT_SHADER_SPIRV: &[u8] = &[
            // SPIR-V 1.0 (Vulkan 1.0 compatible)
            // Compiled with: glslc --target-env=vulkan1.0 -O -fshader-stage=vert yuv_fullscreen.vert -o -
            0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x0b, 0x00, 0x0d, 0x00,
            0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x06, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x47, 0x4c, 0x53, 0x4c, 0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30,
            0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
            0x09, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00,
            0x47, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00,
            0x0b, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00,
            0x1b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
            0x1b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x1b, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x48, 0x00, 0x05, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
            0x1b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x16, 0x00, 0x03, 0x00, 0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x17, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
            0x08, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x15, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
            0x0b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x2b, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
            0x17, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
            0x15, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00,
            0x19, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x04, 0x00,
            0x1a, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
            0x1e, 0x00, 0x06, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x17, 0x00, 0x00, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x1b, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x1c, 0x00, 0x00, 0x00,
            0x1d, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
            0x0a, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x2b, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x40, 0x2b, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x2b, 0x00, 0x04, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x04, 0x00, 0x29, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x17, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00,
            0x2b, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
            0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00,
            0x05, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
            0x0d, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0xc4, 0x00, 0x05, 0x00,
            0x0a, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
            0x0e, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00, 0x0a, 0x00, 0x00, 0x00,
            0x11, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
            0x6f, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
            0x11, 0x00, 0x00, 0x00, 0xc7, 0x00, 0x05, 0x00, 0x0a, 0x00, 0x00, 0x00,
            0x14, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
            0x6f, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
            0x14, 0x00, 0x00, 0x00, 0x50, 0x00, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00,
            0x16, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00,
            0x3e, 0x00, 0x03, 0x00, 0x09, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
            0x3d, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00,
            0x09, 0x00, 0x00, 0x00, 0x8e, 0x00, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00,
            0x21, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
            0x83, 0x00, 0x05, 0x00, 0x07, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
            0x21, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x27, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x50, 0x00, 0x07, 0x00, 0x17, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
            0x26, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00,
            0x22, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00, 0x29, 0x00, 0x00, 0x00,
            0x2a, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x3e, 0x00, 0x03, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
            0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00,
        ];

        /// YUV→RGB fragment shader (SPIR-V).
        ///
        /// Samples Y and UV planes and converts to RGB using uniform matrix/offset.
        /// Supports configurable color space (BT.709/BT.601) and range (limited/full).
        ///
        /// Bindings:
        ///   0: Y plane sampler (R8)
        ///   1: UV plane sampler (RG8)
        ///   2: YuvParams uniform buffer (mat3 yuv_to_rgb, vec3 yuv_offset)
        ///
        /// GLSL source: `src/media/shaders/yuv_to_rgb.frag`
        #[rustfmt::skip]
        const FRAG_SHADER_SPIRV: &[u8] = &[
            // SPIR-V 1.0 (Vulkan 1.0 compatible)
            // Compiled with: glslc -O -fshader-stage=frag yuv_to_rgb.frag -o -
            // New version with uniform buffer for configurable YUV params
            0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x0b, 0x00, 0x0d, 0x00,
            0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x06, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x47, 0x4c, 0x53, 0x4c, 0x2e, 0x73, 0x74, 0x64, 0x2e, 0x34, 0x35, 0x30,
            0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x07, 0x00, 0x04, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00, 0x10, 0x00, 0x03, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
            0x0c, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x47, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x10, 0x00, 0x00, 0x00,
            0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
            0x19, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x47, 0x00, 0x04, 0x00, 0x19, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x03, 0x00, 0x29, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x48, 0x00, 0x04, 0x00, 0x29, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00,
            0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x48, 0x00, 0x05, 0x00, 0x29, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x48, 0x00, 0x05, 0x00, 0x29, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x23, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
            0x2b, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x47, 0x00, 0x04, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x22, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x3a, 0x00, 0x00, 0x00,
            0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x00, 0x00, 0x19, 0x00, 0x09, 0x00, 0x09, 0x00, 0x00, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x0a, 0x00, 0x00, 0x00,
            0x09, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
            0x0b, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x17, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
            0x0f, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x17, 0x00, 0x04, 0x00, 0x12, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x0b, 0x00, 0x00, 0x00,
            0x19, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
            0x1e, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x18, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x04, 0x00, 0x29, 0x00, 0x00, 0x00,
            0x28, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
            0x2a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00,
            0x3b, 0x00, 0x04, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00, 0x15, 0x00, 0x04, 0x00, 0x2c, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
            0x2c, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x04, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x1e, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x2c, 0x00, 0x00, 0x00,
            0x33, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
            0x34, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
            0x20, 0x00, 0x04, 0x00, 0x39, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0x12, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x39, 0x00, 0x00, 0x00,
            0x3a, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x2b, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x3f, 0x2c, 0x00, 0x06, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x47, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
            0x3c, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x48, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x00, 0x00,
            0x3d, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
            0xf8, 0x00, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
            0x0a, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
            0x3d, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
            0x10, 0x00, 0x00, 0x00, 0x57, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00,
            0x13, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00,
            0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
            0x13, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
            0x0a, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00,
            0x57, 0x00, 0x05, 0x00, 0x12, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
            0x1a, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x26, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x50, 0x00, 0x06, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00,
            0x16, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00,
            0x41, 0x00, 0x05, 0x00, 0x2e, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
            0x2b, 0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
            0x1e, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00,
            0x83, 0x00, 0x05, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x31, 0x00, 0x00, 0x00,
            0x27, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00,
            0x34, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00,
            0x33, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x28, 0x00, 0x00, 0x00,
            0x36, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0x91, 0x00, 0x05, 0x00,
            0x1e, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x36, 0x00, 0x00, 0x00,
            0x31, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x08, 0x00, 0x1e, 0x00, 0x00, 0x00,
            0x40, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00,
            0x38, 0x00, 0x00, 0x00, 0x47, 0x00, 0x00, 0x00, 0x48, 0x00, 0x00, 0x00,
            0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00,
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
            0x06, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x06, 0x00, 0x00, 0x00,
            0x43, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
            0x50, 0x00, 0x07, 0x00, 0x12, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00,
            0x41, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00,
            0x3d, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00, 0x3a, 0x00, 0x00, 0x00,
            0x44, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00,
        ];

        /// Imports a YUV AHardwareBuffer and converts it to RGBA via GPU.
        ///
        /// This performs true zero-copy import by:
        /// 1. Creating a disjoint VkImage with the NV12 format
        /// 2. Importing AHardwareBuffer memory to each plane
        /// 3. Creating plane-specific VkImageViews
        /// 4. Running the YUV→RGB render pass
        /// 5. Returning the RGBA result as a wgpu texture
        ///
        /// # Safety
        ///
        /// - `ahardware_buffer` must be a valid YUV AHardwareBuffer
        /// - The buffer must remain valid until the returned texture is dropped
        /// - `vk_queue` must be from the same queue family as the pipeline
        pub unsafe fn convert_yuv_ahardwarebuffer(
            &self,
            wgpu_device: &wgpu::Device,
            ahardware_buffer: AHardwareBufferPtr,
            raw_instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            vk_queue: vk::Queue,
            width: u32,
            height: u32,
            color_space: Option<YuvColorSpace>,
            fence_fd: i32,
        ) -> Result<wgpu::Texture, ZeroCopyError> {
            if ahardware_buffer.is_null() {
                return Err(ZeroCopyError::InvalidResource(
                    "AHardwareBuffer is null".to_string(),
                ));
            }

            // RAII guard to ensure fence_fd is closed on any exit path (early return, error, or success)
            // For NV12 path: we do a CPU wait then close. For external format path: fence_guard is
            // passed through and that function manages its own FenceFdGuard.
            struct FenceFdGuard(i32);
            impl Drop for FenceFdGuard {
                fn drop(&mut self) {
                    if self.0 >= 0 {
                        unsafe {
                            libc::close(self.0);
                        }
                    }
                }
            }
            impl FenceFdGuard {
                /// Mark the FD as consumed (e.g., when passing ownership to another function)
                fn take(&mut self) -> i32 {
                    let fd = self.0;
                    self.0 = -1;
                    fd
                }
            }

            let mut fence_guard = FenceFdGuard(fence_fd);

            debug!(
                "Converting YUV AHardwareBuffer to RGBA: {}x{}",
                width, height
            );

            // Step 1: Query AHardwareBuffer properties
            // The push_next pattern creates a mutable borrow chain, so we must
            // scope ahb_props to drop it before accessing ahb_format_props fields.
            let mut ahb_format_props = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
            let (allocation_size, memory_type_bits) = {
                let mut ahb_props = vk::AndroidHardwareBufferPropertiesANDROID::default()
                    .push_next(&mut ahb_format_props);

                type GetAndroidHardwareBufferPropertiesFn = unsafe extern "system" fn(
                    device: vk::Device,
                    buffer: *const std::ffi::c_void,
                    properties: *mut vk::AndroidHardwareBufferPropertiesANDROID,
                )
                    -> vk::Result;

                let get_ahb_props_fn: GetAndroidHardwareBufferPropertiesFn = {
                    let fn_name = CStr::from_bytes_with_nul_unchecked(
                        b"vkGetAndroidHardwareBufferPropertiesANDROID\0",
                    );
                    let fn_ptr = self
                        .instance
                        .get_device_proc_addr(self.device.handle(), fn_name.as_ptr());
                    if fn_ptr.is_none() {
                        return Err(ZeroCopyError::HalAccessFailed(
                            "vkGetAndroidHardwareBufferPropertiesANDROID not found".to_string(),
                        ));
                    }
                    // SAFETY: fn_ptr is verified non-null above. The function pointer
                    // is obtained from vkGetDeviceProcAddr for a valid extension function
                    // name, and we transmute it to a matching function signature as
                    // defined by the Vulkan spec for VK_ANDROID_external_memory_android_hardware_buffer.
                    std::mem::transmute(fn_ptr)
                };

                let result =
                    get_ahb_props_fn(self.device.handle(), ahardware_buffer, &mut ahb_props);

                if result != vk::Result::SUCCESS {
                    return Err(ZeroCopyError::InvalidResource(format!(
                        "Failed to get AHardwareBuffer properties: {:?}",
                        result
                    )));
                }

                // Extract ahb_props values before it drops (releasing borrow on ahb_format_props)
                (ahb_props.allocation_size, ahb_props.memory_type_bits)
            };
            // Now ahb_props is dropped, we can access ahb_format_props
            let ahb_format = ahb_format_props.format;
            let ahb_external_format = ahb_format_props.external_format;

            // Extract suggested YCbCr conversion parameters from the AHB format properties
            // These are driver-provided recommendations for correct color conversion
            let suggested_ycbcr_model = ahb_format_props.suggested_ycbcr_model;
            let suggested_ycbcr_range = ahb_format_props.suggested_ycbcr_range;
            let suggested_x_chroma_offset = ahb_format_props.suggested_x_chroma_offset;
            let suggested_y_chroma_offset = ahb_format_props.suggested_y_chroma_offset;
            let sampler_ycbcr_components = ahb_format_props.sampler_ycbcr_conversion_components;

            // Check for YUV format - either standard Vulkan YCbCr format or external format
            // External formats have format=UNDEFINED but external_format!=0
            let is_external_format =
                ahb_format == vk::Format::UNDEFINED && ahb_external_format != 0;
            let is_standard_ycbcr = is_ycbcr_format(ahb_format);

            if !is_external_format && !is_standard_ycbcr {
                return Err(ZeroCopyError::FormatMismatch(format!(
                    "Expected YUV format, got {:?} (external_format={})",
                    ahb_format, ahb_external_format
                )));
            }

            // For now, we support external formats (vendor YUV) and standard NV12
            // External formats require VkExternalFormatANDROID and YCbCr conversion
            if is_external_format {
                debug!(
                    "AHardwareBuffer external YUV format: external_format={}, suggested_model={:?}, suggested_range={:?}",
                    ahb_external_format, suggested_ycbcr_model, suggested_ycbcr_range
                );
                // External formats use a dedicated path with VkSamplerYcbcrConversion
                // for hardware YCbCr→RGB conversion
                // Take ownership of fence_fd - import_ahb_yuv_external_format has its own guard
                let external_fence_fd = fence_guard.take();
                return self.import_ahb_yuv_external_format(
                    wgpu_device,
                    raw_instance,
                    physical_device,
                    vk_queue,
                    ahardware_buffer,
                    ahb_external_format,
                    allocation_size,
                    memory_type_bits,
                    width,
                    height,
                    suggested_ycbcr_model,
                    suggested_ycbcr_range,
                    suggested_x_chroma_offset,
                    suggested_y_chroma_offset,
                    sampler_ycbcr_components,
                    external_fence_fd,
                );
            } else {
                // Gate standard formats to NV12 only - our pipeline hardcodes 2-plane binding
                const NV12_FORMAT: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
                if ahb_format != NV12_FORMAT {
                    debug!(
                        "YUV format {:?} not supported for zero-copy (only NV12), falling back to CPU",
                        ahb_format
                    );
                    return Err(ZeroCopyError::NotAvailable(format!(
                        "YUV zero-copy only supports NV12 (G8_B8R8_2PLANE_420_UNORM), got {:?}",
                        ahb_format
                    )));
                }
                debug!(
                    "AHardwareBuffer YUV format: {:?} (NV12), size: {}",
                    ahb_format, allocation_size
                );
                // NV12 path: fence_fd will be passed to run_yuv_to_rgba_pass for GPU sync
            }

            // Step 2: Create VkImage for YUV
            // External formats (VK_FORMAT_UNDEFINED with external_format != 0) require:
            // - VkExternalFormatANDROID in pNext chain
            // - NO DISJOINT flag (external format images are single-plane from API perspective)
            // - VkSamplerYcbcrConversion for sampling
            // Standard YCbCr formats (like NV12) can use DISJOINT + plane views
            let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);

            // For external formats, we need VkExternalFormatANDROID
            let mut external_format_info =
                vk::ExternalFormatANDROID::default().external_format(ahb_external_format);

            let image_create_info = if is_external_format {
                // External format path: no DISJOINT, use VkExternalFormatANDROID
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::UNDEFINED) // Must be UNDEFINED for external formats
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    // No DISJOINT flag for external formats
                    .push_next(&mut external_format_info)
                    .push_next(&mut external_memory_info)
            } else {
                // Standard YCbCr format path (e.g., NV12): use DISJOINT for plane access
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(ahb_format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .flags(vk::ImageCreateFlags::DISJOINT)
                    .push_next(&mut external_memory_info)
            };

            let yuv_image = self
                .device
                .create_image(&image_create_info, None)
                .map_err(|e| {
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create YUV image (external_format={}): {:?}",
                        is_external_format, e
                    ))
                })?;

            // Step 3: Import AHardwareBuffer memory and bind to image
            let mut import_ahb_info =
                vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(ahardware_buffer);

            // Find suitable memory type
            let mem_properties =
                raw_instance.get_physical_device_memory_properties(physical_device);

            let memory_type_index_val = find_memory_type_index(
                memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &mem_properties,
            )
            .ok_or_else(|| {
                self.device.destroy_image(yuv_image, None);
                ZeroCopyError::TextureCreationFailed(
                    "No suitable memory type found for AHardwareBuffer".to_string(),
                )
            })?;

            let mut dedicated_alloc_info =
                vk::MemoryDedicatedAllocateInfo::default().image(yuv_image);

            let memory_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(memory_type_index_val)
                .push_next(&mut import_ahb_info)
                .push_next(&mut dedicated_alloc_info);

            let yuv_memory = self
                .device
                .allocate_memory(&memory_allocate_info, None)
                .map_err(|e| {
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate memory from AHardwareBuffer: {:?}",
                        e
                    ))
                })?;

            // Bind memory to image
            // External formats: simple bind (image is not disjoint)
            // Standard YCbCr (DISJOINT): bind to each plane separately
            if is_external_format {
                // Simple bind for external format images
                let result = self.device.bind_image_memory(yuv_image, yuv_memory, 0);
                if let Err(e) = result {
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    return Err(ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to bind image memory (external format): {:?}",
                        e
                    )));
                }
            } else {
                // Disjoint binding for standard YCbCr formats
                // Bind memory to plane 0 (Y)
                let mut plane0_memory_info = vk::BindImagePlaneMemoryInfo::default()
                    .plane_aspect(vk::ImageAspectFlags::PLANE_0);

                let bind_info_plane0 = vk::BindImageMemoryInfo::default()
                    .image(yuv_image)
                    .memory(yuv_memory)
                    .memory_offset(0)
                    .push_next(&mut plane0_memory_info);

                // Bind memory to plane 1 (UV)
                let mut plane1_memory_info = vk::BindImagePlaneMemoryInfo::default()
                    .plane_aspect(vk::ImageAspectFlags::PLANE_1);

                let bind_info_plane1 = vk::BindImageMemoryInfo::default()
                    .image(yuv_image)
                    .memory(yuv_memory)
                    .memory_offset(0)
                    .push_next(&mut plane1_memory_info);

                // vkBindImageMemory2 for both planes
                type BindImageMemory2Fn = unsafe extern "system" fn(
                    device: vk::Device,
                    bind_info_count: u32,
                    bind_infos: *const vk::BindImageMemoryInfo,
                ) -> vk::Result;

                let bind_image_memory2_fn: BindImageMemory2Fn = {
                    let fn_name = CStr::from_bytes_with_nul_unchecked(b"vkBindImageMemory2\0");
                    let fn_ptr = self
                        .instance
                        .get_device_proc_addr(self.device.handle(), fn_name.as_ptr());
                    if fn_ptr.is_none() {
                        self.device.free_memory(yuv_memory, None);
                        self.device.destroy_image(yuv_image, None);
                        return Err(ZeroCopyError::HalAccessFailed(
                            "vkBindImageMemory2 not found".to_string(),
                        ));
                    }
                    // SAFETY: fn_ptr is verified non-null above. The function pointer
                    // is obtained from vkGetDeviceProcAddr for vkBindImageMemory2 (core Vulkan 1.1),
                    // and we transmute it to the matching function signature as defined by the Vulkan spec.
                    std::mem::transmute(fn_ptr)
                };

                let bind_infos = [bind_info_plane0, bind_info_plane1];
                let result = bind_image_memory2_fn(self.device.handle(), 2, bind_infos.as_ptr());

                if result != vk::Result::SUCCESS {
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    return Err(ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to bind image memory (disjoint): {:?}",
                        result
                    )));
                }
            }

            // Step 4: Create plane-specific VkImageViews
            let y_view_info = vk::ImageViewCreateInfo::default()
                .image(yuv_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let y_view = self
                .device
                .create_image_view(&y_view_info, None)
                .map_err(|e| {
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create Y plane view: {:?}",
                        e
                    ))
                })?;

            let uv_view_info = vk::ImageViewCreateInfo::default()
                .image(yuv_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let uv_view = self
                .device
                .create_image_view(&uv_view_info, None)
                .map_err(|e| {
                    self.device.destroy_image_view(y_view, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create UV plane view: {:?}",
                        e
                    ))
                })?;

            // Step 5: Create RGBA output image
            let rgba_image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let rgba_image = self
                .device
                .create_image(&rgba_image_info, None)
                .map_err(|e| {
                    self.device.destroy_image_view(uv_view, None);
                    self.device.destroy_image_view(y_view, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create RGBA output image: {:?}",
                        e
                    ))
                })?;

            // Allocate memory for RGBA image
            let rgba_mem_reqs = self.device.get_image_memory_requirements(rgba_image);

            let rgba_memory_type_index = find_memory_type_index(
                rgba_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &mem_properties,
            )
            .ok_or_else(|| {
                self.device.destroy_image(rgba_image, None);
                self.device.destroy_image_view(uv_view, None);
                self.device.destroy_image_view(y_view, None);
                self.device.free_memory(yuv_memory, None);
                self.device.destroy_image(yuv_image, None);
                ZeroCopyError::TextureCreationFailed(
                    "No suitable memory type for RGBA image".to_string(),
                )
            })?;

            let rgba_alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(rgba_mem_reqs.size)
                .memory_type_index(rgba_memory_type_index);

            let rgba_memory = self
                .device
                .allocate_memory(&rgba_alloc_info, None)
                .map_err(|e| {
                    self.device.destroy_image(rgba_image, None);
                    self.device.destroy_image_view(uv_view, None);
                    self.device.destroy_image_view(y_view, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate RGBA memory: {:?}",
                        e
                    ))
                })?;

            self.device
                .bind_image_memory(rgba_image, rgba_memory, 0)
                .map_err(|e| {
                    self.device.free_memory(rgba_memory, None);
                    self.device.destroy_image(rgba_image, None);
                    self.device.destroy_image_view(uv_view, None);
                    self.device.destroy_image_view(y_view, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to bind RGBA memory: {:?}",
                        e
                    ))
                })?;

            // Create RGBA view for framebuffer
            let rgba_view_info = vk::ImageViewCreateInfo::default()
                .image(rgba_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let rgba_view = self
                .device
                .create_image_view(&rgba_view_info, None)
                .map_err(|e| {
                    self.device.free_memory(rgba_memory, None);
                    self.device.destroy_image(rgba_image, None);
                    self.device.destroy_image_view(uv_view, None);
                    self.device.destroy_image_view(y_view, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create RGBA view: {:?}",
                        e
                    ))
                })?;

            // Step 6: Run the YUV→RGB conversion render pass
            // Take ownership of fence_fd - run_yuv_to_rgba_pass will handle closing it
            let nv12_fence_fd = fence_guard.take();
            let convert_result = self.run_yuv_to_rgba_pass(
                vk_queue,
                raw_instance,
                yuv_image,
                y_view,
                uv_view,
                rgba_image,
                rgba_view,
                width,
                height,
                color_space.unwrap_or_default(),
                nv12_fence_fd,
            );

            // Cleanup YUV resources (no longer needed after conversion)
            self.device.destroy_image_view(uv_view, None);
            self.device.destroy_image_view(y_view, None);
            self.device.free_memory(yuv_memory, None);
            self.device.destroy_image(yuv_image, None);

            if let Err(e) = convert_result {
                self.device.destroy_image_view(rgba_view, None);
                self.device.free_memory(rgba_memory, None);
                self.device.destroy_image(rgba_image, None);
                return Err(e);
            }

            // Cleanup intermediate RGBA view (texture will have its own)
            self.device.destroy_image_view(rgba_view, None);

            // Step 7: Wrap RGBA image as wgpu texture
            let texture_desc = wgpu::hal::TextureDescriptor {
                label: Some("YUV→RGBA converted texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::hal::TextureUses::RESOURCE,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            };

            let device_clone = self.device.clone();
            let drop_callback = Box::new(move || {
                debug!("Freeing YUV→RGBA converted texture resources");
                // SAFETY: rgba_image and rgba_memory were allocated by us and are valid
                // until this callback is invoked when the texture is dropped.
                // destroy_image must be called before free_memory.
                unsafe {
                    device_clone.destroy_image(rgba_image, None);
                    device_clone.free_memory(rgba_memory, None);
                }
            });

            let hal_texture = wgpu::hal::vulkan::Device::texture_from_raw(
                rgba_image,
                &texture_desc,
                Some(drop_callback),
            );

            let wgpu_desc = wgpu::TextureDescriptor {
                label: Some("YUV→RGBA converted texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };

            let wgpu_texture = wgpu_device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc);

            info!(
                "Successfully converted YUV AHardwareBuffer to RGBA ({}x{})",
                width, height
            );

            Ok(wgpu_texture)
        }

        /// Import an AHardwareBuffer with external YUV format using VkSamplerYcbcrConversion.
        ///
        /// External formats (VK_FORMAT_UNDEFINED with external_format != 0) are vendor-specific
        /// YUV formats that require hardware YCbCr conversion for sampling. This function:
        /// 1. Creates VkSamplerYcbcrConversion for the external format
        /// 2. Creates VkImage with VkExternalFormatANDROID
        /// 3. Imports AHardwareBuffer memory
        /// 4. Creates image view and sampler with YCbCr conversion
        /// 5. Runs a simple blit pass to RGBA (hardware does YCbCr→RGB during sampling)
        ///
        /// This is true zero-copy: the GPU's dedicated YCbCr sampling hardware converts
        /// YUV→RGB during texture fetch, with no CPU involvement.
        #[allow(clippy::too_many_arguments)]
        unsafe fn import_ahb_yuv_external_format(
            &self,
            wgpu_device: &wgpu::Device,
            raw_instance: &ash::Instance,
            physical_device: vk::PhysicalDevice,
            vk_queue: vk::Queue,
            ahardware_buffer: *mut vk::AHardwareBuffer,
            external_format: u64,
            allocation_size: vk::DeviceSize,
            memory_type_bits: u32,
            width: u32,
            height: u32,
            // Driver-suggested YCbCr conversion parameters from VkAndroidHardwareBufferFormatPropertiesANDROID
            suggested_ycbcr_model: vk::SamplerYcbcrModelConversion,
            suggested_ycbcr_range: vk::SamplerYcbcrRange,
            suggested_x_chroma_offset: vk::ChromaLocation,
            suggested_y_chroma_offset: vk::ChromaLocation,
            sampler_ycbcr_components: vk::ComponentMapping,
            // Sync fence FD from producer (-1 if none)
            fence_fd: i32,
        ) -> Result<wgpu::Texture, ZeroCopyError> {
            use std::ffi::CStr;

            // RAII guard to ensure fence_fd is closed on early return
            // The guard is consumed (via take()) when we pass fence_fd to run_ycbcr_blit_pass
            struct FenceFdGuard(i32);
            impl Drop for FenceFdGuard {
                fn drop(&mut self) {
                    if self.0 >= 0 {
                        unsafe {
                            libc::close(self.0);
                        }
                    }
                }
            }
            impl FenceFdGuard {
                /// Take ownership of the FD, preventing the guard from closing it
                fn take(&mut self) -> i32 {
                    let fd = self.0;
                    self.0 = -1;
                    fd
                }
            }

            let mut fence_guard = FenceFdGuard(fence_fd);

            info!(
                "Importing AHB with external format {} ({}x{}) using YCbCr conversion",
                external_format, width, height
            );
            debug!(
                "YCbCr params: model={:?}, range={:?}, x_chroma={:?}, y_chroma={:?}",
                suggested_ycbcr_model,
                suggested_ycbcr_range,
                suggested_x_chroma_offset,
                suggested_y_chroma_offset
            );

            // Step 1: Create VkSamplerYcbcrConversion for external format
            // For external formats, we use VkExternalFormatANDROID in the pNext chain
            // Use driver-suggested parameters for correct color conversion
            let mut external_format_info =
                vk::ExternalFormatANDROID::default().external_format(external_format);

            let ycbcr_create_info = vk::SamplerYcbcrConversionCreateInfo::default()
                .format(vk::Format::UNDEFINED) // Must be UNDEFINED for external formats
                .ycbcr_model(suggested_ycbcr_model)
                .ycbcr_range(suggested_ycbcr_range)
                .components(sampler_ycbcr_components)
                .x_chroma_offset(suggested_x_chroma_offset)
                .y_chroma_offset(suggested_y_chroma_offset)
                .chroma_filter(vk::Filter::LINEAR)
                .force_explicit_reconstruction(false)
                .push_next(&mut external_format_info);

            // Load vkCreateSamplerYcbcrConversion
            type CreateSamplerYcbcrConversionFn = unsafe extern "system" fn(
                device: vk::Device,
                create_info: *const vk::SamplerYcbcrConversionCreateInfo,
                allocator: *const vk::AllocationCallbacks,
                ycbcr_conversion: *mut vk::SamplerYcbcrConversion,
            )
                -> vk::Result;

            // Try core Vulkan 1.1 function name first, then KHR extension fallback
            // Some drivers (especially older ones) may only expose the KHR variant
            let create_ycbcr_fn: CreateSamplerYcbcrConversionFn = {
                let core_fn_name =
                    CStr::from_bytes_with_nul_unchecked(b"vkCreateSamplerYcbcrConversion\0");
                let mut fn_ptr =
                    raw_instance.get_device_proc_addr(self.device.handle(), core_fn_name.as_ptr());

                // Fallback to KHR extension name if core not available
                if fn_ptr.is_none() {
                    let khr_fn_name =
                        CStr::from_bytes_with_nul_unchecked(b"vkCreateSamplerYcbcrConversionKHR\0");
                    fn_ptr = raw_instance
                        .get_device_proc_addr(self.device.handle(), khr_fn_name.as_ptr());
                    if fn_ptr.is_some() {
                        debug!("Using KHR fallback for vkCreateSamplerYcbcrConversion");
                    }
                }

                if fn_ptr.is_none() {
                    return Err(ZeroCopyError::HalAccessFailed(
                        "vkCreateSamplerYcbcrConversion[KHR] not found - YCbCr extension not enabled?"
                            .to_string(),
                    ));
                }
                std::mem::transmute(fn_ptr)
            };

            let mut ycbcr_conversion = vk::SamplerYcbcrConversion::null();
            let result = create_ycbcr_fn(
                self.device.handle(),
                &ycbcr_create_info,
                std::ptr::null(),
                &mut ycbcr_conversion,
            );

            if result != vk::Result::SUCCESS {
                return Err(ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to create YCbCr conversion: {:?}",
                    result
                )));
            }

            debug!("Created VkSamplerYcbcrConversion for external format");

            // Step 2: Create VkImage with external format
            let mut ext_mem_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);

            let mut ext_format_image =
                vk::ExternalFormatANDROID::default().external_format(external_format);

            let image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::UNDEFINED)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_format_image)
                .push_next(&mut ext_mem_info);

            let yuv_image = self
                .device
                .create_image(&image_create_info, None)
                .map_err(|e| {
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create external format image: {:?}",
                        e
                    ))
                })?;

            // Step 3: Import AHardwareBuffer memory
            let mut import_ahb_info =
                vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(ahardware_buffer);

            let mem_properties =
                raw_instance.get_physical_device_memory_properties(physical_device);

            let memory_type_index = find_memory_type_index(
                memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &mem_properties,
            )
            .ok_or_else(|| {
                self.device.destroy_image(yuv_image, None);
                self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                ZeroCopyError::TextureCreationFailed(
                    "No suitable memory type for AHardwareBuffer".to_string(),
                )
            })?;

            let mut dedicated_alloc = vk::MemoryDedicatedAllocateInfo::default().image(yuv_image);

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(allocation_size)
                .memory_type_index(memory_type_index)
                .push_next(&mut import_ahb_info)
                .push_next(&mut dedicated_alloc);

            let yuv_memory = self
                .device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| {
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate AHB memory: {:?}",
                        e
                    ))
                })?;

            // Bind memory (simple bind for external format, not per-plane)
            self.device
                .bind_image_memory(yuv_image, yuv_memory, 0)
                .map_err(|e| {
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to bind AHB memory: {:?}",
                        e
                    ))
                })?;

            // Step 4: Create sampler with YCbCr conversion
            let mut ycbcr_sampler_info =
                vk::SamplerYcbcrConversionInfo::default().conversion(ycbcr_conversion);

            let sampler_create_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .push_next(&mut ycbcr_sampler_info);

            let ycbcr_sampler = self
                .device
                .create_sampler(&sampler_create_info, None)
                .map_err(|e| {
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create YCbCr sampler: {:?}",
                        e
                    ))
                })?;

            // Step 5: Create image view with YCbCr conversion
            // For external formats, only VkSamplerYcbcrConversionInfo is needed in image view
            // The external format is already baked into the image from VkExternalFormatANDROID
            let mut ycbcr_view_info =
                vk::SamplerYcbcrConversionInfo::default().conversion(ycbcr_conversion);

            let view_create_info = vk::ImageViewCreateInfo::default()
                .image(yuv_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::UNDEFINED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .push_next(&mut ycbcr_view_info);

            let yuv_view = self
                .device
                .create_image_view(&view_create_info, None)
                .map_err(|e| {
                    self.device.destroy_sampler(ycbcr_sampler, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create YCbCr image view: {:?}",
                        e
                    ))
                })?;

            debug!("Created YCbCr sampler and image view for external format");

            // Step 6: Create RGBA output image for the blit
            let rgba_image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let rgba_image = self
                .device
                .create_image(&rgba_image_info, None)
                .map_err(|e| {
                    self.device.destroy_image_view(yuv_view, None);
                    self.device.destroy_sampler(ycbcr_sampler, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create RGBA image: {:?}",
                        e
                    ))
                })?;

            let rgba_mem_reqs = self.device.get_image_memory_requirements(rgba_image);
            let rgba_mem_type = find_memory_type_index(
                rgba_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &mem_properties,
            )
            .ok_or_else(|| {
                self.device.destroy_image(rgba_image, None);
                self.device.destroy_image_view(yuv_view, None);
                self.device.destroy_sampler(ycbcr_sampler, None);
                self.device.free_memory(yuv_memory, None);
                self.device.destroy_image(yuv_image, None);
                self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                ZeroCopyError::TextureCreationFailed("No memory type for RGBA".to_string())
            })?;

            let rgba_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(rgba_mem_reqs.size)
                .memory_type_index(rgba_mem_type);

            let rgba_memory = self
                .device
                .allocate_memory(&rgba_alloc, None)
                .map_err(|e| {
                    self.device.destroy_image(rgba_image, None);
                    self.device.destroy_image_view(yuv_view, None);
                    self.device.destroy_sampler(ycbcr_sampler, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate RGBA memory: {:?}",
                        e
                    ))
                })?;

            self.device
                .bind_image_memory(rgba_image, rgba_memory, 0)
                .map_err(|e| {
                    self.device.free_memory(rgba_memory, None);
                    self.device.destroy_image(rgba_image, None);
                    self.device.destroy_image_view(yuv_view, None);
                    self.device.destroy_sampler(ycbcr_sampler, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to bind RGBA memory: {:?}",
                        e
                    ))
                })?;

            let rgba_view_info = vk::ImageViewCreateInfo::default()
                .image(rgba_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let rgba_view = self
                .device
                .create_image_view(&rgba_view_info, None)
                .map_err(|e| {
                    self.device.free_memory(rgba_memory, None);
                    self.device.destroy_image(rgba_image, None);
                    self.device.destroy_image_view(yuv_view, None);
                    self.device.destroy_sampler(ycbcr_sampler, None);
                    self.device.free_memory(yuv_memory, None);
                    self.device.destroy_image(yuv_image, None);
                    self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create RGBA view: {:?}",
                        e
                    ))
                })?;

            // Step 7: Run the YCbCr blit pass (hardware YCbCr→RGB during sampling)
            // Take fence_fd from guard - run_ycbcr_blit_pass will handle closing it
            let blit_result = self.run_ycbcr_blit_pass(
                raw_instance,
                vk_queue,
                yuv_image,
                yuv_view,
                ycbcr_sampler,
                ycbcr_conversion,
                rgba_image,
                rgba_view,
                width,
                height,
                fence_guard.take(),
            );

            // Cleanup YUV resources (blit is complete)
            self.device.destroy_image_view(yuv_view, None);
            self.device.destroy_sampler(ycbcr_sampler, None);
            self.device.free_memory(yuv_memory, None);
            self.device.destroy_image(yuv_image, None);
            self.destroy_ycbcr_conversion(raw_instance, ycbcr_conversion);

            if let Err(e) = blit_result {
                self.device.destroy_image_view(rgba_view, None);
                self.device.free_memory(rgba_memory, None);
                self.device.destroy_image(rgba_image, None);
                return Err(e);
            }

            self.device.destroy_image_view(rgba_view, None);

            // Step 8: Wrap RGBA as wgpu texture
            let texture_desc = wgpu::hal::TextureDescriptor {
                label: Some("YCbCr→RGBA zero-copy texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::hal::TextureUses::RESOURCE,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            };

            let device_clone = self.device.clone();
            let drop_callback = Box::new(move || {
                debug!("Freeing YCbCr→RGBA texture resources");
                unsafe {
                    device_clone.destroy_image(rgba_image, None);
                    device_clone.free_memory(rgba_memory, None);
                }
            });

            let hal_texture = wgpu::hal::vulkan::Device::texture_from_raw(
                rgba_image,
                &texture_desc,
                Some(drop_callback),
            );

            let wgpu_desc = wgpu::TextureDescriptor {
                label: Some("YCbCr→RGBA zero-copy texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };

            let wgpu_texture = wgpu_device
                .create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal_texture, &wgpu_desc);

            info!(
                "Successfully imported AHB via YCbCr conversion ({}x{})",
                width, height
            );

            Ok(wgpu_texture)
        }

        /// Destroy a VkSamplerYcbcrConversion object.
        unsafe fn destroy_ycbcr_conversion(
            &self,
            raw_instance: &ash::Instance,
            ycbcr_conversion: vk::SamplerYcbcrConversion,
        ) {
            type DestroySamplerYcbcrConversionFn = unsafe extern "system" fn(
                device: vk::Device,
                ycbcr_conversion: vk::SamplerYcbcrConversion,
                allocator: *const vk::AllocationCallbacks,
            );

            // Try core Vulkan 1.1 function name first, then KHR extension fallback
            let core_fn_name =
                std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroySamplerYcbcrConversion\0");
            let mut fn_ptr =
                raw_instance.get_device_proc_addr(self.device.handle(), core_fn_name.as_ptr());

            // Fallback to KHR extension name if core not available
            if fn_ptr.is_none() {
                let khr_fn_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
                    b"vkDestroySamplerYcbcrConversionKHR\0",
                );
                fn_ptr =
                    raw_instance.get_device_proc_addr(self.device.handle(), khr_fn_name.as_ptr());
            }

            if let Some(fp) = fn_ptr {
                let destroy_fn: DestroySamplerYcbcrConversionFn = std::mem::transmute(fp);
                destroy_fn(self.device.handle(), ycbcr_conversion, std::ptr::null());
            }
        }

        /// Run a blit pass that samples from YCbCr image (hardware conversion) to RGBA.
        ///
        /// The YCbCr sampler performs YUV→RGB conversion in hardware during the texture
        /// fetch, so the shader just copies the RGB result to RGBA output.
        #[allow(clippy::too_many_arguments)]
        unsafe fn run_ycbcr_blit_pass(
            &self,
            raw_instance: &ash::Instance,
            vk_queue: vk::Queue,
            yuv_image: vk::Image,
            yuv_view: vk::ImageView,
            ycbcr_sampler: vk::Sampler,
            ycbcr_conversion: vk::SamplerYcbcrConversion,
            rgba_image: vk::Image,
            rgba_view: vk::ImageView,
            width: u32,
            height: u32,
            fence_fd: i32,
        ) -> Result<(), ZeroCopyError> {
            // For YCbCr sampling, we need a descriptor set layout with immutable sampler
            // This is required because YCbCr samplers must be immutable
            let _ = ycbcr_conversion; // Used via sampler, but keep param for future use

            let immutable_samplers = [ycbcr_sampler];
            let binding = vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&immutable_samplers);

            let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(std::slice::from_ref(&binding));

            let ycbcr_desc_layout = self
                .device
                .create_descriptor_set_layout(&layout_info, None)
                .map_err(|e| {
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create YCbCr descriptor layout: {:?}",
                        e
                    ))
                })?;

            // Allocate descriptor set
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            }];

            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);

            let ycbcr_pool = self
                .device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|e| {
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create YCbCr descriptor pool: {:?}",
                        e
                    ))
                })?;

            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(ycbcr_pool)
                .set_layouts(std::slice::from_ref(&ycbcr_desc_layout));

            let desc_sets = self
                .device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|e| {
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate YCbCr descriptor set: {:?}",
                        e
                    ))
                })?;

            let desc_set = *desc_sets.first().ok_or_else(|| {
                self.device.destroy_descriptor_pool(ycbcr_pool, None);
                self.device
                    .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                ZeroCopyError::TextureCreationFailed(
                    "allocate_descriptor_sets returned empty vector".to_string(),
                )
            })?;

            // Update descriptor with YCbCr image view
            // Note: sampler is already bound via immutable sampler in layout
            let image_info = vk::DescriptorImageInfo::default()
                .sampler(ycbcr_sampler) // Ignored for immutable sampler, but required
                .image_view(yuv_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            self.device.update_descriptor_sets(&[write], &[]);

            // Create pipeline layout
            let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&ycbcr_desc_layout));

            let pipeline_layout = self
                .device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| {
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create pipeline layout: {:?}",
                        e
                    ))
                })?;

            // Use the existing blit shaders (passthrough vertex + simple fragment)
            // The fragment shader just samples and outputs - YCbCr conversion is in hardware
            let ycbcr_pipeline = self.create_ycbcr_blit_pipeline(pipeline_layout)?;

            // Create framebuffer
            let fb_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(std::slice::from_ref(&rgba_view))
                .width(width)
                .height(height)
                .layers(1);

            let framebuffer = self
                .device
                .create_framebuffer(&fb_info, None)
                .map_err(|e| {
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create framebuffer: {:?}",
                        e
                    ))
                })?;

            // Record and submit command buffer
            let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_buffers = self
                .device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| {
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate command buffer: {:?}",
                        e
                    ))
                })?;

            let cmd = *cmd_buffers.first().ok_or_else(|| {
                self.device.destroy_framebuffer(framebuffer, None);
                self.device.destroy_pipeline(ycbcr_pipeline, None);
                self.device.destroy_pipeline_layout(pipeline_layout, None);
                self.device.destroy_descriptor_pool(ycbcr_pool, None);
                self.device
                    .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                ZeroCopyError::TextureCreationFailed(
                    "allocate_command_buffers returned empty vector".to_string(),
                )
            })?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|e| {
                    self.device
                        .free_command_buffers(self.command_pool, &cmd_buffers);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to begin command buffer: {:?}",
                        e
                    ))
                })?;

            // Queue ownership transfer from external (AHardwareBuffer producer) to our queue
            // This is required for proper synchronization with the MediaCodec producer.
            // VK_QUEUE_FAMILY_EXTERNAL indicates the image comes from an external source.
            let yuv_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .dst_queue_family_index(self.queue_family_index)
                .image(yuv_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[yuv_barrier],
            );

            // Begin render pass
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];

            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .clear_values(&clear_values);

            self.device
                .cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, ycbcr_pipeline);

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[desc_set],
                &[],
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            };
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);

            // Draw fullscreen triangle
            self.device.cmd_draw(cmd, 3, 1, 0, 0);

            self.device.cmd_end_render_pass(cmd);

            // Memory barrier to ensure RGBA writes are visible to wgpu
            // Note: Layout transition is handled by render pass (final_layout = SHADER_READ_ONLY_OPTIMAL)
            // We only need to synchronize memory access, not transition layouts.
            let rgba_barrier = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(rgba_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[rgba_barrier],
            );

            self.device.end_command_buffer(cmd).map_err(|e| {
                self.device
                    .free_command_buffers(self.command_pool, &cmd_buffers);
                self.device.destroy_framebuffer(framebuffer, None);
                self.device.destroy_pipeline(ycbcr_pipeline, None);
                self.device.destroy_pipeline_layout(pipeline_layout, None);
                self.device.destroy_descriptor_pool(ycbcr_pool, None);
                self.device
                    .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to end command buffer: {:?}",
                    e
                ))
            })?;

            // Submit with per-frame fence (more efficient than queue_wait_idle)
            // queue_wait_idle blocks ALL queue operations; a fence waits only for this submit.
            let cmd_buffers_submit = [cmd];

            // Import sync fence from producer (MediaCodec) if available.
            // The fence FD indicates when the producer finished writing to the AHardwareBuffer.
            // We must wait on this fence before sampling to avoid reading incomplete data.
            let wait_semaphore = if fence_fd >= 0 {
                // Create a semaphore to import the fence FD into
                let sem_info = vk::SemaphoreCreateInfo::default();
                let semaphore = self.device.create_semaphore(&sem_info, None).map_err(|e| {
                    self.device
                        .free_command_buffers(self.command_pool, &cmd_buffers);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create semaphore: {:?}",
                        e
                    ))
                })?;

                // Import fence FD via VK_KHR_external_semaphore_fd
                // VK_SEMAPHORE_IMPORT_TEMPORARY_BIT means Vulkan takes ownership of the FD
                // and the semaphore reverts to its prior state after one wait operation.
                let import_info = vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(semaphore)
                    .flags(vk::SemaphoreImportFlags::TEMPORARY)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                    .fd(fence_fd);

                // Load vkImportSemaphoreFdKHR function pointer
                type ImportSemaphoreFdFn = unsafe extern "system" fn(
                    device: vk::Device,
                    p_import_semaphore_fd_info: *const vk::ImportSemaphoreFdInfoKHR,
                ) -> vk::Result;

                let fn_name =
                    std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkImportSemaphoreFdKHR\0");
                let fn_ptr =
                    raw_instance.get_device_proc_addr(self.device.handle(), fn_name.as_ptr());

                if let Some(fp) = fn_ptr {
                    let import_fn: ImportSemaphoreFdFn = std::mem::transmute(fp);
                    let result = import_fn(self.device.handle(), &import_info);

                    if result != vk::Result::SUCCESS {
                        self.device.destroy_semaphore(semaphore, None);
                        libc::close(fence_fd);
                        self.device
                            .free_command_buffers(self.command_pool, &cmd_buffers);
                        self.device.destroy_framebuffer(framebuffer, None);
                        self.device.destroy_pipeline(ycbcr_pipeline, None);
                        self.device.destroy_pipeline_layout(pipeline_layout, None);
                        self.device.destroy_descriptor_pool(ycbcr_pool, None);
                        self.device
                            .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                        return Err(ZeroCopyError::TextureCreationFailed(format!(
                            "vkImportSemaphoreFdKHR failed: {:?} - dropping frame to avoid unsync read",
                            result
                        )));
                    }
                    debug!("Imported sync fence FD {} as VkSemaphore", fence_fd);
                    // FD ownership transferred to Vulkan with TEMPORARY flag
                    Some(semaphore)
                } else {
                    self.device.destroy_semaphore(semaphore, None);
                    libc::close(fence_fd);
                    self.device
                        .free_command_buffers(self.command_pool, &cmd_buffers);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    return Err(ZeroCopyError::TextureCreationFailed(
                        "vkImportSemaphoreFdKHR not available - cannot safely import without fence sync"
                            .to_string(),
                    ));
                }
            } else {
                None
            };

            // Build submit info with optional wait semaphore
            let wait_semaphores;
            let wait_stages;
            let submit_info = if let Some(sem) = wait_semaphore {
                wait_semaphores = [sem];
                // Gate at TOP_OF_PIPE so the ownership barrier waits for the producer fence.
                // This ensures the AHardwareBuffer is fully written before we read from it.
                wait_stages = [vk::PipelineStageFlags::TOP_OF_PIPE];
                vk::SubmitInfo::default()
                    .command_buffers(&cmd_buffers_submit)
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
            } else {
                vk::SubmitInfo::default().command_buffers(&cmd_buffers_submit)
            };

            let fence_info = vk::FenceCreateInfo::default();
            let fence = self.device.create_fence(&fence_info, None).map_err(|e| {
                if let Some(sem) = wait_semaphore {
                    self.device.destroy_semaphore(sem, None);
                }
                self.device
                    .free_command_buffers(self.command_pool, &cmd_buffers);
                self.device.destroy_framebuffer(framebuffer, None);
                self.device.destroy_pipeline(ycbcr_pipeline, None);
                self.device.destroy_pipeline_layout(pipeline_layout, None);
                self.device.destroy_descriptor_pool(ycbcr_pool, None);
                self.device
                    .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                ZeroCopyError::TextureCreationFailed(format!("Failed to create fence: {:?}", e))
            })?;

            self.device
                .queue_submit(vk_queue, &[submit_info], fence)
                .map_err(|e| {
                    if let Some(sem) = wait_semaphore {
                        self.device.destroy_semaphore(sem, None);
                    }
                    self.device.destroy_fence(fence, None);
                    self.device
                        .free_command_buffers(self.command_pool, &cmd_buffers);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!("Failed to submit: {:?}", e))
                })?;

            // Wait for this specific command buffer to complete (1 second timeout)
            self.device
                .wait_for_fences(&[fence], true, 1_000_000_000)
                .map_err(|e| {
                    if let Some(sem) = wait_semaphore {
                        self.device.destroy_semaphore(sem, None);
                    }
                    self.device.destroy_fence(fence, None);
                    self.device
                        .free_command_buffers(self.command_pool, &cmd_buffers);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device.destroy_pipeline(ycbcr_pipeline, None);
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_pool(ycbcr_pool, None);
                    self.device
                        .destroy_descriptor_set_layout(ycbcr_desc_layout, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to wait for fence: {:?}",
                        e
                    ))
                })?;

            // Clean up sync resources
            if let Some(sem) = wait_semaphore {
                self.device.destroy_semaphore(sem, None);
            }
            self.device.destroy_fence(fence, None);

            // Cleanup
            self.device
                .free_command_buffers(self.command_pool, &cmd_buffers);
            self.device.destroy_framebuffer(framebuffer, None);
            self.device.destroy_pipeline(ycbcr_pipeline, None);
            self.device.destroy_pipeline_layout(pipeline_layout, None);
            self.device.destroy_descriptor_pool(ycbcr_pool, None);
            self.device
                .destroy_descriptor_set_layout(ycbcr_desc_layout, None);

            debug!("YCbCr blit pass completed successfully");
            Ok(())
        }

        /// Create a simple blit pipeline for YCbCr sampling.
        /// Uses passthrough shaders - the YCbCr sampler does the color conversion.
        unsafe fn create_ycbcr_blit_pipeline(
            &self,
            layout: vk::PipelineLayout,
        ) -> Result<vk::Pipeline, ZeroCopyError> {
            // Simple passthrough vertex shader (fullscreen triangle)
            // gl_Position covers [-1,1] screen space, TexCoord is [0,1]
            const VERT_SPIRV: &[u8] = include_bytes!("shaders/blit.vert.spv");

            // Simple passthrough fragment shader
            // Just samples the YCbCr texture (hardware does conversion) and outputs RGBA
            const FRAG_SPIRV: &[u8] = include_bytes!("shaders/ycbcr_blit.frag.spv");

            let vert_code: Vec<u32> = VERT_SPIRV
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let frag_code: Vec<u32> = FRAG_SPIRV
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let vert_info = vk::ShaderModuleCreateInfo::default().code(&vert_code);
            let frag_info = vk::ShaderModuleCreateInfo::default().code(&frag_code);

            let vert_module = self
                .device
                .create_shader_module(&vert_info, None)
                .map_err(|e| {
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create vert shader: {:?}",
                        e
                    ))
                })?;

            let frag_module = self
                .device
                .create_shader_module(&frag_info, None)
                .map_err(|e| {
                    self.device.destroy_shader_module(vert_module, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create frag shader: {:?}",
                        e
                    ))
                })?;

            let entry_name = std::ffi::CStr::from_bytes_with_nul_unchecked(b"main\0");

            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vert_module)
                    .name(entry_name),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(frag_module)
                    .name(entry_name),
            ];

            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);

            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA);

            let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(std::slice::from_ref(&blend_attachment));

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state =
                vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .dynamic_state(&dynamic_state)
                .layout(layout)
                .render_pass(self.render_pass)
                .subpass(0);

            let pipelines = self
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| {
                    self.device.destroy_shader_module(frag_module, None);
                    self.device.destroy_shader_module(vert_module, None);
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create pipeline: {:?}",
                        e
                    ))
                })?;

            self.device.destroy_shader_module(frag_module, None);
            self.device.destroy_shader_module(vert_module, None);

            pipelines.into_iter().next().ok_or_else(|| {
                ZeroCopyError::TextureCreationFailed(
                    "create_graphics_pipelines returned empty vector".to_string(),
                )
            })
        }

        /// Runs the YUV→RGB conversion render pass.
        ///
        /// This function:
        /// 1. Updates the uniform buffer with the specified color space params
        /// 2. Transitions the YUV image planes to SHADER_READ_ONLY_OPTIMAL
        /// 3. Executes the YUV→RGB shader
        /// 4. Outputs to the RGBA image
        ///
        /// # Sync fence handling
        /// If `fence_fd >= 0`, imports it as a VkSemaphore and waits on it before
        /// executing the command buffer. This ensures the producer (MediaCodec)
        /// has finished writing to the AHardwareBuffer before we read from it.
        #[allow(clippy::too_many_arguments)]
        unsafe fn run_yuv_to_rgba_pass(
            &self,
            vk_queue: vk::Queue,
            raw_instance: &ash::Instance,
            yuv_image: vk::Image,
            y_view: vk::ImageView,
            uv_view: vk::ImageView,
            _rgba_image: vk::Image,
            rgba_view: vk::ImageView,
            width: u32,
            height: u32,
            color_space: YuvColorSpace,
            fence_fd: i32,
        ) -> Result<(), ZeroCopyError> {
            // Update uniform buffer with color space parameters
            if !self.update_color_space(color_space) {
                debug!("Color space update failed, using previous params");
            }

            // Allocate descriptor set
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));

            let descriptor_sets =
                self.device
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(|e| {
                        ZeroCopyError::TextureCreationFailed(format!(
                            "Failed to allocate descriptor set: {:?}",
                            e
                        ))
                    })?;

            let descriptor_set = descriptor_sets.into_iter().next().ok_or_else(|| {
                ZeroCopyError::TextureCreationFailed(
                    "allocate_descriptor_sets returned empty vector".to_string(),
                )
            })?;

            // Update descriptor set with Y and UV views
            let y_image_info = vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(y_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let uv_image_info = vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(uv_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            // Uniform buffer for YUV params
            let yuv_params_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(self.yuv_params_buffer)
                .offset(0)
                .range(std::mem::size_of::<YuvParams>() as vk::DeviceSize);

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&y_image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&uv_image_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&yuv_params_buffer_info)),
            ];

            self.device.update_descriptor_sets(&descriptor_writes, &[]);

            // Create framebuffer
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(self.render_pass)
                .attachments(std::slice::from_ref(&rgba_view))
                .width(width)
                .height(height)
                .layers(1);

            let framebuffer = self
                .device
                .create_framebuffer(&framebuffer_info, None)
                .map_err(|e| {
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create framebuffer: {:?}",
                        e
                    ))
                })?;

            // Allocate command buffer
            let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let cmd_buffers = self
                .device
                .allocate_command_buffers(&cmd_alloc_info)
                .map_err(|e| {
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to allocate command buffer: {:?}",
                        e
                    ))
                })?;

            let cmd_buf = cmd_buffers.into_iter().next().ok_or_else(|| {
                self.device.destroy_framebuffer(framebuffer, None);
                self.device
                    .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                    .ok();
                ZeroCopyError::TextureCreationFailed(
                    "allocate_command_buffers returned empty vector".to_string(),
                )
            })?;

            // Record command buffer
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device
                .begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|e| {
                    self.device
                        .free_command_buffers(self.command_pool, &[cmd_buf]);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to begin command buffer: {:?}",
                        e
                    ))
                })?;

            // Transition YUV image planes from UNDEFINED to SHADER_READ_ONLY_OPTIMAL
            // For disjoint NV12, we need barriers for both PLANE_0 (Y) and PLANE_1 (UV)
            // VK_QUEUE_FAMILY_EXTERNAL is (~1U) = 0xFFFFFFFE for external memory ownership transfer
            const VK_QUEUE_FAMILY_EXTERNAL: u32 = !1u32;

            let yuv_barriers = [
                // Y plane (PLANE_0) barrier
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::NONE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(VK_QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(yuv_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::PLANE_0)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                // UV plane (PLANE_1) barrier
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::NONE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(VK_QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(self.queue_family_index)
                    .image(yuv_image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::PLANE_1)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
            ];

            self.device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &yuv_barriers,
            );

            // Begin render pass
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];

            let render_pass_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(
                cmd_buf,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );

            // Set viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(cmd_buf, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            };
            self.device.cmd_set_scissor(cmd_buf, 0, &[scissor]);

            // Bind pipeline and descriptor set
            self.device
                .cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            self.device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            // Draw fullscreen triangle (3 vertices)
            self.device.cmd_draw(cmd_buf, 3, 1, 0, 0);

            self.device.cmd_end_render_pass(cmd_buf);

            self.device.end_command_buffer(cmd_buf).map_err(|e| {
                self.device
                    .free_command_buffers(self.command_pool, &[cmd_buf]);
                self.device.destroy_framebuffer(framebuffer, None);
                self.device
                    .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                    .ok();
                ZeroCopyError::TextureCreationFailed(format!(
                    "Failed to end command buffer: {:?}",
                    e
                ))
            })?;

            // Import sync fence from producer (MediaCodec) if available.
            // The fence FD indicates when the producer finished writing to the AHardwareBuffer.
            // We must wait on this fence before sampling to avoid reading incomplete data.
            let cmd_bufs = [cmd_buf];
            let wait_semaphore = if fence_fd >= 0 {
                // Create a semaphore to import the fence FD into
                let sem_info = vk::SemaphoreCreateInfo::default();
                let semaphore = self.device.create_semaphore(&sem_info, None).map_err(|e| {
                    self.device
                        .free_command_buffers(self.command_pool, &[cmd_buf]);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to create semaphore: {:?}",
                        e
                    ))
                })?;

                // Import fence FD via VK_KHR_external_semaphore_fd
                // VK_SEMAPHORE_IMPORT_TEMPORARY_BIT means Vulkan takes ownership of the FD
                // and the semaphore reverts to its prior state after one wait operation.
                let import_info = vk::ImportSemaphoreFdInfoKHR::default()
                    .semaphore(semaphore)
                    .flags(vk::SemaphoreImportFlags::TEMPORARY)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                    .fd(fence_fd);

                // Load vkImportSemaphoreFdKHR function pointer
                type ImportSemaphoreFdFn = unsafe extern "system" fn(
                    device: vk::Device,
                    p_import_semaphore_fd_info: *const vk::ImportSemaphoreFdInfoKHR,
                ) -> vk::Result;

                let fn_name =
                    std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkImportSemaphoreFdKHR\0");
                let fn_ptr =
                    raw_instance.get_device_proc_addr(self.device.handle(), fn_name.as_ptr());

                if let Some(fp) = fn_ptr {
                    let import_fn: ImportSemaphoreFdFn = std::mem::transmute(fp);
                    let result = import_fn(self.device.handle(), &import_info);

                    if result != vk::Result::SUCCESS {
                        self.device.destroy_semaphore(semaphore, None);
                        libc::close(fence_fd);
                        self.device
                            .free_command_buffers(self.command_pool, &[cmd_buf]);
                        self.device.destroy_framebuffer(framebuffer, None);
                        self.device
                            .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                            .ok();
                        return Err(ZeroCopyError::TextureCreationFailed(format!(
                            "vkImportSemaphoreFdKHR failed: {:?} - dropping frame to avoid unsync read",
                            result
                        )));
                    }
                    debug!("NV12: Imported sync fence FD {} as VkSemaphore", fence_fd);
                    // FD ownership transferred to Vulkan with TEMPORARY flag
                    Some(semaphore)
                } else {
                    self.device.destroy_semaphore(semaphore, None);
                    libc::close(fence_fd);
                    self.device
                        .free_command_buffers(self.command_pool, &[cmd_buf]);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    return Err(ZeroCopyError::TextureCreationFailed(
                        "vkImportSemaphoreFdKHR not available - cannot safely import without fence sync"
                            .to_string(),
                    ));
                }
            } else {
                None
            };

            // Build submit info with optional wait semaphore
            let wait_semaphores;
            let wait_stages;
            let submit_info = if let Some(sem) = wait_semaphore {
                wait_semaphores = [sem];
                // Gate at TOP_OF_PIPE so the ownership barrier waits for the producer fence.
                // This ensures the AHardwareBuffer is fully written before we read from it.
                wait_stages = [vk::PipelineStageFlags::TOP_OF_PIPE];
                vk::SubmitInfo::default()
                    .command_buffers(&cmd_bufs)
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
            } else {
                vk::SubmitInfo::default().command_buffers(&cmd_bufs)
            };

            let fence_info = vk::FenceCreateInfo::default();
            let fence = self.device.create_fence(&fence_info, None).map_err(|e| {
                if let Some(sem) = wait_semaphore {
                    self.device.destroy_semaphore(sem, None);
                }
                self.device
                    .free_command_buffers(self.command_pool, &[cmd_buf]);
                self.device.destroy_framebuffer(framebuffer, None);
                self.device
                    .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                    .ok();
                ZeroCopyError::TextureCreationFailed(format!("Failed to create fence: {:?}", e))
            })?;

            self.device
                .queue_submit(vk_queue, &[submit_info], fence)
                .map_err(|e| {
                    if let Some(sem) = wait_semaphore {
                        self.device.destroy_semaphore(sem, None);
                    }
                    self.device
                        .free_command_buffers(self.command_pool, &[cmd_buf]);
                    self.device.destroy_fence(fence, None);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Failed to submit command buffer: {:?}",
                        e
                    ))
                })?;

            // Wait for completion (1 second timeout)
            self.device
                .wait_for_fences(&[fence], true, 1_000_000_000)
                .map_err(|e| {
                    if let Some(sem) = wait_semaphore {
                        self.device.destroy_semaphore(sem, None);
                    }
                    self.device
                        .free_command_buffers(self.command_pool, &[cmd_buf]);
                    self.device.destroy_fence(fence, None);
                    self.device.destroy_framebuffer(framebuffer, None);
                    self.device
                        .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                        .ok();
                    ZeroCopyError::TextureCreationFailed(format!(
                        "Timeout waiting for YUV conversion: {:?}",
                        e
                    ))
                })?;

            // Cleanup
            if let Some(sem) = wait_semaphore {
                self.device.destroy_semaphore(sem, None);
            }
            self.device.destroy_fence(fence, None);
            self.device.destroy_framebuffer(framebuffer, None);
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[descriptor_set])
                .ok();
            // Free command buffer to prevent leak on repeated conversions
            self.device
                .free_command_buffers(self.command_pool, &[cmd_buf]);

            debug!("YUV→RGBA conversion completed successfully");
            Ok(())
        }
    }

    impl Drop for VulkanYuvPipeline {
        fn drop(&mut self) {
            unsafe {
                debug!("Destroying VulkanYuvPipeline");
                self.device.destroy_command_pool(self.command_pool, None);
                self.device.free_memory(self.yuv_params_memory, None);
                self.device.destroy_buffer(self.yuv_params_buffer, None);
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
                self.device.destroy_pipeline(self.pipeline, None);
                self.device.destroy_render_pass(self.render_pass, None);
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
                self.device.destroy_sampler(self.sampler, None);
                self.device.destroy_shader_module(self.frag_shader, None);
                self.device.destroy_shader_module(self.vert_shader, None);
            }
        }
    }
}

// =============================================================================
// Windows: D3D11 shared handle → D3D12 → wgpu
// =============================================================================

/// Windows-specific zero-copy import via D3D11/D3D12 shared handles.
///
/// This module provides functions to import D3D11 shared textures
/// directly into wgpu (via D3D12) without CPU memory copies.
///
/// # Requirements
///
/// - Windows 10 or later
/// - wgpu using D3D12 backend
/// - D3D11 texture created with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`
///
/// # D3D11/D3D12 Interop
///
/// This uses the standard Windows cross-API texture sharing:
/// 1. D3D11 creates a texture with `SHARED_NTHANDLE` flag
/// 2. D3D11 calls `CreateSharedHandle()` to get an NT handle
/// 3. D3D12 calls `OpenSharedHandle()` to get `ID3D12Resource`
/// 4. wgpu wraps the resource via `texture_from_raw()`
///
/// # Synchronization
///
/// D3D11/D3D12 cross-API synchronization is required to ensure D3D11 decode
/// completes before D3D12/wgpu reads the texture.
///
/// ## Current Implementation: Query Polling (Safe but Suboptimal)
///
/// The current implementation in `windows_video.rs` uses:
/// - `ID3D11DeviceContext::Flush()` to submit all pending D3D11 work
/// - `D3D11_QUERY_EVENT` to poll for completion on the decode thread
/// - CPU blocks until the query signals, then passes handle to D3D12
///
/// This is safe and prevents visual corruption, but adds ~0.1-0.5ms latency
/// per frame due to CPU polling.
///
/// ## Future Improvement: True GPU Fence
///
/// For optimal performance, cross-API fence sync could be implemented:
/// - Create `ID3D11Fence` from `ID3D11Device5` (Windows 10+)
/// - Share fence via HANDLE between D3D11 and D3D12
/// - D3D11 signals fence after decode completes
/// - D3D12 waits on fence before reading shared texture
///
/// This would eliminate CPU blocking and enable true GPU-GPU synchronization.
/// See Microsoft docs on "Sharing Surfaces Between Windows Graphics APIs".
///
/// # Example
///
/// ```ignore
/// use lumina_video::media::zero_copy::windows;
///
/// // D3D11 side: Create shared texture
/// let shared_handle = device11.CreateSharedHandle(&d3d11_texture, ...)?;
///
/// // wgpu/D3D12 side: Import the texture
/// let texture = unsafe {
///     windows::import_d3d11_shared_handle(&device, shared_handle, width, height, format)?
/// };
/// ```
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub mod windows {
    use super::ZeroCopyError;
    use tracing::{debug, info, warn};
    use windows::Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D12::{ID3D12Device, ID3D12Resource},
            Dxgi::Common as DxgiCommon,
        },
    };

    /// Opaque handle to a D3D11 shared texture (HANDLE from CreateSharedHandle).
    /// This is obtained from ID3D11Device1::CreateSharedHandle() on a D3D11 texture
    /// created with D3D11_RESOURCE_MISC_SHARED_NTHANDLE.
    pub type SharedHandle = HANDLE;

    /// Checks if the current wgpu device supports D3D12 backend.
    pub fn is_d3d12_backend(device: &wgpu::Device) -> bool {
        unsafe { device.as_hal::<wgpu::hal::api::Dx12, _, bool>(|hal_device| hal_device.is_some()) }
    }

    /// Gets information about the D3D12 device for diagnostics.
    ///
    /// Returns a description if D3D12 backend is available, None otherwise.
    pub fn get_d3d12_device_info(device: &wgpu::Device) -> Option<String> {
        unsafe {
            device.as_hal::<wgpu::hal::api::Dx12, _, Option<String>>(|hal_device| {
                hal_device.map(|_| "D3D12 Device".to_string())
            })
        }
    }

    /// Checks if D3D11/D3D12 shared handle import is available.
    ///
    /// This always returns true on Windows when using D3D12 backend,
    /// as shared handle import is a core D3D12 feature (no extensions needed).
    pub fn is_shared_handle_import_available(device: &wgpu::Device) -> bool {
        is_d3d12_backend(device)
    }

    /// Imports a D3D11 shared handle into wgpu as a texture (zero-copy).
    ///
    /// This function creates a wgpu::Texture that directly references the D3D11 texture's
    /// GPU memory via D3D12 interop, enabling zero-copy video frame display.
    ///
    /// # D3D11/D3D12 Interop Flow
    ///
    /// 1. D3D11 decoder creates a texture with SHARED_NTHANDLE flag
    /// 2. D3D11 calls CreateSharedHandle() to get an NT handle
    /// 3. D3D12 calls OpenSharedHandle() to get ID3D12Resource
    /// 4. wgpu wraps the resource via texture_from_raw()
    ///
    /// # Handle Ownership
    ///
    /// **The caller retains ownership of the shared handle.** This function does not
    /// close the handle; `OpenSharedHandle` only creates a reference to the underlying
    /// D3D11 resource without transferring handle ownership.
    ///
    /// The caller must close the handle via `CloseHandle()` when done, but only AFTER:
    /// - The returned wgpu::Texture has been dropped, AND
    /// - All GPU operations using the texture have completed
    ///
    /// Closing the handle prematurely while the texture is in use results in undefined
    /// behavior (typically GPU hangs or access violations).
    ///
    /// # Safety
    ///
    /// - `shared_handle` must be a valid HANDLE from ID3D11Device1::CreateSharedHandle()
    /// - The D3D11 texture must have been created with D3D11_RESOURCE_MISC_SHARED_NTHANDLE
    /// - The D3D11 texture must remain valid for the lifetime of the returned texture
    /// - The caller is responsible for synchronization between D3D11 and D3D12 access
    ///   (use ID3D11Fence for cross-API synchronization)
    /// - The handle must not be closed until the returned texture is dropped and GPU idle
    ///
    /// # Errors
    ///
    /// Returns [`ZeroCopyError::InvalidResource`] if the handle is invalid (null or closed).
    /// Returns [`ZeroCopyError::TextureCreationFailed`] if `OpenSharedHandle` fails, which
    /// can occur if the handle was closed or refers to an incompatible resource.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu Device (must be D3D12 backend)
    /// * `shared_handle` - A valid HANDLE from CreateSharedHandle()
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `format` - The wgpu texture format (should match D3D11 texture format)
    ///
    /// # Returns
    ///
    /// A wgpu::Texture that references the D3D11 texture memory via D3D12 interop.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // D3D11 side: Create shared texture
    /// let d3d11_texture = device11.CreateTexture2D(&desc, None)?;
    /// let shared_handle = device11.CreateSharedHandle(&d3d11_texture, None, GENERIC_ALL, None)?;
    ///
    /// // wgpu/D3D12 side: Import the texture
    /// let wgpu_texture = unsafe {
    ///     import_d3d11_shared_handle(&wgpu_device, shared_handle, width, height, format)?
    /// };
    /// ```
    pub unsafe fn import_d3d11_shared_handle(
        device: &wgpu::Device,
        shared_handle: HANDLE,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<wgpu::Texture, ZeroCopyError> {
        use windows::core::Interface;

        if shared_handle.is_invalid() {
            return Err(ZeroCopyError::InvalidResource(
                "Shared handle is invalid".to_string(),
            ));
        }

        // Access the D3D12 HAL device and open the shared handle
        let hal_texture_result = device
            .as_hal::<wgpu::hal::api::Dx12, _, Result<wgpu::hal::dx12::Texture, ZeroCopyError>>(
                |hal_device| {
                    let Some(hal_device) = hal_device else {
                        warn!("Failed to get D3D12 HAL device");
                        return Err(ZeroCopyError::HalAccessFailed(
                            "wgpu not using D3D12 backend".to_string(),
                        ));
                    };

                    // Get the raw D3D12 device
                    let d3d12_device: &ID3D12Device = hal_device.raw_device();

                    debug!(
                        "Opening D3D11 shared handle via D3D12 ({}x{} {:?})",
                        width, height, format
                    );

                    // Open the shared handle as a D3D12 resource
                    // This is the key interop call: D3D12 can open NT handles from D3D11
                    let d3d12_resource: ID3D12Resource =
                        d3d12_device.OpenSharedHandle(shared_handle).map_err(|e| {
                            warn!("D3D12 OpenSharedHandle failed: {:?}", e);
                            ZeroCopyError::TextureCreationFailed(format!(
                                "D3D12 OpenSharedHandle failed: {:?}",
                                e
                            ))
                        })?;

                    info!(
                        "Opened D3D11 shared handle as D3D12 resource: {}x{} {:?}",
                        width, height, format
                    );

                    // Wrap as wgpu_hal::dx12::Texture using the existing API
                    // Note: texture_from_raw is unsafe because it trusts the caller
                    // to provide valid parameters matching the actual resource
                    let hal_texture = wgpu::hal::dx12::Device::texture_from_raw(
                        d3d12_resource,
                        format,
                        wgpu::TextureDimension::D2,
                        wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        1, // mip_level_count
                        1, // sample_count
                    );

                    Ok(hal_texture)
                },
            );

        // Get the HAL texture from the closure result
        let hal_texture = hal_texture_result?;

        // Create wgpu texture descriptor
        let texture_desc = wgpu::TextureDescriptor {
            label: Some("zero-copy D3D11 shared texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // Wrap the HAL texture as a wgpu::Texture
        let wgpu_texture =
            device.create_texture_from_hal::<wgpu::hal::api::Dx12>(hal_texture, &texture_desc);

        info!("Successfully imported D3D11 shared handle as wgpu texture (zero-copy)");

        Ok(wgpu_texture)
    }

    /// Converts wgpu TextureFormat to DXGI_FORMAT.
    ///
    /// Used for validation and debugging when creating shared textures.
    pub fn wgpu_format_to_dxgi(
        format: wgpu::TextureFormat,
    ) -> Result<DxgiCommon::DXGI_FORMAT, ZeroCopyError> {
        match format {
            wgpu::TextureFormat::Bgra8Unorm => Ok(DxgiCommon::DXGI_FORMAT_B8G8R8A8_UNORM),
            wgpu::TextureFormat::Rgba8Unorm => Ok(DxgiCommon::DXGI_FORMAT_R8G8B8A8_UNORM),
            wgpu::TextureFormat::R8Unorm => Ok(DxgiCommon::DXGI_FORMAT_R8_UNORM),
            wgpu::TextureFormat::Rg8Unorm => Ok(DxgiCommon::DXGI_FORMAT_R8G8_UNORM),
            wgpu::TextureFormat::Bgra8UnormSrgb => Ok(DxgiCommon::DXGI_FORMAT_B8G8R8A8_UNORM_SRGB),
            wgpu::TextureFormat::Rgba8UnormSrgb => Ok(DxgiCommon::DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
            _ => {
                warn!("Unsupported texture format {:?}; refusing import", format);
                Err(ZeroCopyError::InvalidResource(format!(
                    "Unsupported texture format {:?}",
                    format
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_stats() {
        let mut stats = ZeroCopyStats::default();
        assert_eq!(stats.zero_copy_percentage(), 0.0);

        stats.total_frames = 100;
        stats.zero_copy_frames = 75;
        stats.fallback_frames = 25;
        assert!((stats.zero_copy_percentage() - 75.0).abs() < 0.01);
    }
}
