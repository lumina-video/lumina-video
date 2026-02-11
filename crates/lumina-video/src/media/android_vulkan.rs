//! Vulkan import for Android AHardwareBuffer.
//!
//! This module handles importing AHardwareBuffer into Vulkan for zero-copy
//! video rendering. It uses the VK_ANDROID_external_memory_android_hardware_buffer
//! extension to create VkImage backed by hardware buffers from MediaCodec.
//!
//! # Key Components
//!
//! - Query hardware buffer format via vkGetAndroidHardwareBufferPropertiesANDROID
//! - Create VkImage with external format for vendor-specific YUV
//! - Use dedicated allocation for imported memory
//! - Set up VkSamplerYcbcrConversion for YUV to RGB conversion
//!
//! # Vulkan Specification References
//!
//! - [VK_ANDROID_external_memory_android_hardware_buffer](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_ANDROID_external_memory_android_hardware_buffer.html)
//! - [VK_KHR_sampler_ycbcr_conversion](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_sampler_ycbcr_conversion.html)
//! - [VK_KHR_external_memory](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory.html)
//! - [VK_EXT_queue_family_foreign](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_queue_family_foreign.html)
//!
//! See `docs/VULKAN-REFERENCE.md` for detailed extension usage and troubleshooting.
//!
//! # Threading
//!
//! All Vulkan calls must happen on the render thread, not the AImageReader callback thread.
//! Use channels or atomics to pass buffer handles to the render thread.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use ash::vk;

use super::video::VideoError;

/// RAII wrapper for Android hardware buffer to ensure proper cleanup.
///
/// Holds a raw pointer to an AHardwareBuffer and calls `AHardwareBuffer_release`
/// on drop to decrement the reference count.
pub struct HardwareBufferHandle {
    buffer: *mut c_void,
}

impl HardwareBufferHandle {
    /// Creates a new handle from a raw hardware buffer pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure the buffer pointer is valid and that the reference
    /// count has been incremented (via AHardwareBuffer_acquire) if ownership
    /// is being transferred.
    pub unsafe fn new(buffer: *mut c_void) -> Self {
        Self { buffer }
    }

    /// Returns the raw buffer pointer.
    pub fn as_ptr(&self) -> *mut c_void {
        self.buffer
    }
}

impl Drop for HardwareBufferHandle {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            // Safety: We own this buffer reference and must release it
            unsafe {
                ndk_sys::AHardwareBuffer_release(self.buffer as *mut ndk_sys::AHardwareBuffer);
            }
        }
    }
}

// Safety: The buffer handle is just a pointer that can be sent between threads
// The actual AHardwareBuffer is thread-safe when accessed through the NDK API
unsafe impl Send for HardwareBufferHandle {}
unsafe impl Sync for HardwareBufferHandle {}

/// Extension name constants for compile-time checking
const VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME: &std::ffi::CStr = unsafe {
    std::ffi::CStr::from_bytes_with_nul_unchecked(
        b"VK_ANDROID_external_memory_android_hardware_buffer\0",
    )
};
const VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_sampler_ycbcr_conversion\0") };
const VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory\0") };
const VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME: &std::ffi::CStr =
    unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_EXT_queue_family_foreign\0") };

/// Result type for Vulkan operations.
pub type VulkanResult<T> = Result<T, VulkanError>;

/// Errors that can occur during Vulkan import.
#[derive(Debug, Clone)]
pub enum VulkanError {
    /// Extension not available
    ExtensionNotAvailable(String),
    /// Failed to query buffer properties
    QueryPropertiesFailed(String),
    /// Failed to create image
    ImageCreationFailed(String),
    /// Failed to allocate memory
    AllocationFailed(String),
    /// Failed to create sampler YCbCr conversion
    YcbcrConversionFailed(String),
    /// Invalid buffer format
    InvalidFormat(String),
    /// Device lost or other fatal error
    DeviceLost(String),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::ExtensionNotAvailable(ext) => {
                write!(f, "Vulkan extension not available: {}", ext)
            }
            VulkanError::QueryPropertiesFailed(msg) => {
                write!(f, "Failed to query buffer properties: {}", msg)
            }
            VulkanError::ImageCreationFailed(msg) => write!(f, "Failed to create image: {}", msg),
            VulkanError::AllocationFailed(msg) => write!(f, "Failed to allocate memory: {}", msg),
            VulkanError::YcbcrConversionFailed(msg) => {
                write!(f, "Failed to create YCbCr conversion: {}", msg)
            }
            VulkanError::InvalidFormat(msg) => write!(f, "Invalid buffer format: {}", msg),
            VulkanError::DeviceLost(msg) => write!(f, "Device lost: {}", msg),
        }
    }
}

impl std::error::Error for VulkanError {}

impl From<VulkanError> for VideoError {
    fn from(e: VulkanError) -> Self {
        VideoError::DecodeFailed(e.to_string())
    }
}

/// Properties of an imported Android hardware buffer.
#[derive(Debug, Clone)]
pub struct HardwareBufferProperties {
    /// Allocation size in bytes
    pub allocation_size: u64,
    /// Memory type bits for allocation
    pub memory_type_bits: u32,
    /// Vulkan format (may be VK_FORMAT_UNDEFINED for external formats)
    pub format: vk::Format,
    /// External format for vendor-specific YUV formats
    pub external_format: u64,
    /// Suggested YCbCr model
    pub suggested_ycbcr_model: vk::SamplerYcbcrModelConversion,
    /// Suggested YCbCr range
    pub suggested_ycbcr_range: vk::SamplerYcbcrRange,
    /// Suggested X chroma offset
    pub suggested_x_chroma_offset: vk::ChromaLocation,
    /// Suggested Y chroma offset
    pub suggested_y_chroma_offset: vk::ChromaLocation,
}

/// Imported Vulkan image from an Android hardware buffer.
///
/// Manages the lifecycle of VkImage, VkDeviceMemory, and VkSamplerYcbcrConversion
/// created from an AHardwareBuffer.
pub struct ImportedHardwareBuffer {
    /// The imported Vulkan image
    pub image: vk::Image,
    /// Image view for sampling
    pub image_view: vk::ImageView,
    /// Allocated device memory
    pub memory: vk::DeviceMemory,
    /// YCbCr sampler conversion (for YUV formats)
    pub ycbcr_conversion: Option<vk::SamplerYcbcrConversion>,
    /// Sampler with YCbCr conversion
    pub sampler: vk::Sampler,
    /// Image dimensions
    pub width: u32,
    /// Image height
    pub height: u32,
    /// The hardware buffer handle (keeps buffer alive)
    _buffer_handle: Arc<HardwareBufferHandle>,
}

/// Function pointer for vkGetAndroidHardwareBufferPropertiesANDROID.
type PfnGetAndroidHardwareBufferPropertiesANDROID = unsafe extern "system" fn(
    device: vk::Device,
    buffer: *const ndk_sys::AHardwareBuffer,
    p_properties: *mut vk::AndroidHardwareBufferPropertiesANDROID,
) -> vk::Result;

/// Function pointer for vkCreateSamplerYcbcrConversion.
type PfnCreateSamplerYcbcrConversion = unsafe extern "system" fn(
    device: vk::Device,
    p_create_info: *const vk::SamplerYcbcrConversionCreateInfo,
    p_allocator: *const vk::AllocationCallbacks,
    p_ycbcr_conversion: *mut vk::SamplerYcbcrConversion,
) -> vk::Result;

/// Function pointer for vkDestroySamplerYcbcrConversion.
type PfnDestroySamplerYcbcrConversion = unsafe extern "system" fn(
    device: vk::Device,
    ycbcr_conversion: vk::SamplerYcbcrConversion,
    p_allocator: *const vk::AllocationCallbacks,
);

/// Context for Vulkan hardware buffer operations.
///
/// Holds references to Vulkan device and instance for performing
/// hardware buffer imports. Uses dynamically loaded function pointers
/// for Android-specific extensions.
pub struct VulkanHardwareBufferContext {
    /// Vulkan device handle
    device: ash::Device,
    /// Vulkan physical device
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    /// Function pointer for getting hardware buffer properties
    get_ahb_properties: PfnGetAndroidHardwareBufferPropertiesANDROID,
    /// Function pointer for creating YCbCr conversion
    create_ycbcr_conversion: PfnCreateSamplerYcbcrConversion,
    /// Function pointer for destroying YCbCr conversion
    destroy_ycbcr_conversion: PfnDestroySamplerYcbcrConversion,
}

impl VulkanHardwareBufferContext {
    /// Creates a new context for hardware buffer operations.
    ///
    /// # Safety
    ///
    /// The device and instance must be valid and the required extensions must be enabled.
    pub unsafe fn new(
        instance: &ash::Instance,
        device: ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> VulkanResult<Self> {
        // Load Android hardware buffer extension function
        let get_ahb_properties_name = std::ffi::CStr::from_bytes_with_nul_unchecked(
            b"vkGetAndroidHardwareBufferPropertiesANDROID\0",
        );
        let get_ahb_properties_ptr =
            instance.get_device_proc_addr(device.handle(), get_ahb_properties_name.as_ptr());
        let get_ahb_properties: PfnGetAndroidHardwareBufferPropertiesANDROID =
            if get_ahb_properties_ptr.is_none() {
                return Err(VulkanError::ExtensionNotAvailable(
                    "vkGetAndroidHardwareBufferPropertiesANDROID".to_string(),
                ));
            } else {
                // SAFETY: Function pointer type matches the Vulkan specification for
                // vkGetAndroidHardwareBufferPropertiesANDROID. The extension was verified
                // to be available via get_device_proc_addr returning Some.
                std::mem::transmute(get_ahb_properties_ptr)
            };

        // Load YCbCr conversion functions (try KHR first, then core 1.1)
        let create_ycbcr_name =
            std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateSamplerYcbcrConversionKHR\0");
        let mut create_ycbcr_ptr =
            instance.get_device_proc_addr(device.handle(), create_ycbcr_name.as_ptr());
        if create_ycbcr_ptr.is_none() {
            let core_name =
                std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkCreateSamplerYcbcrConversion\0");
            create_ycbcr_ptr = instance.get_device_proc_addr(device.handle(), core_name.as_ptr());
        }
        let create_ycbcr_conversion: PfnCreateSamplerYcbcrConversion = if create_ycbcr_ptr.is_none()
        {
            return Err(VulkanError::ExtensionNotAvailable(
                "vkCreateSamplerYcbcrConversion".to_string(),
            ));
        } else {
            // SAFETY: Function pointer type matches the Vulkan specification for
            // vkCreateSamplerYcbcrConversion. The extension was verified to be
            // available via get_device_proc_addr returning Some.
            std::mem::transmute(create_ycbcr_ptr)
        };

        let destroy_ycbcr_name =
            std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroySamplerYcbcrConversionKHR\0");
        let mut destroy_ycbcr_ptr =
            instance.get_device_proc_addr(device.handle(), destroy_ycbcr_name.as_ptr());
        if destroy_ycbcr_ptr.is_none() {
            let core_name =
                std::ffi::CStr::from_bytes_with_nul_unchecked(b"vkDestroySamplerYcbcrConversion\0");
            destroy_ycbcr_ptr = instance.get_device_proc_addr(device.handle(), core_name.as_ptr());
        }
        let destroy_ycbcr_conversion: PfnDestroySamplerYcbcrConversion =
            if destroy_ycbcr_ptr.is_none() {
                return Err(VulkanError::ExtensionNotAvailable(
                    "vkDestroySamplerYcbcrConversion".to_string(),
                ));
            } else {
                // SAFETY: Function pointer type matches the Vulkan specification for
                // vkDestroySamplerYcbcrConversion. The extension was verified to be
                // available via get_device_proc_addr returning Some.
                std::mem::transmute(destroy_ycbcr_ptr)
            };

        Ok(Self {
            device,
            physical_device,
            get_ahb_properties,
            create_ycbcr_conversion,
            destroy_ycbcr_conversion,
        })
    }

    /// Queries the properties of an Android hardware buffer.
    ///
    /// This must be called before creating a VkImage to determine the correct
    /// format and allocation parameters.
    ///
    /// # Safety
    ///
    /// The buffer pointer must be a valid AHardwareBuffer.
    pub unsafe fn query_buffer_properties(
        &self,
        buffer: *mut c_void,
    ) -> VulkanResult<HardwareBufferProperties> {
        // Set up format properties struct in pNext chain
        let mut format_props = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();

        let mut props =
            vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut format_props);

        // Call vkGetAndroidHardwareBufferPropertiesANDROID
        let buffer_ptr = buffer as *const ndk_sys::AHardwareBuffer;
        let result = (self.get_ahb_properties)(self.device.handle(), buffer_ptr, &mut props);

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::QueryPropertiesFailed(format!(
                "vkGetAndroidHardwareBufferPropertiesANDROID failed: {:?}",
                result
            )));
        }

        Ok(HardwareBufferProperties {
            allocation_size: props.allocation_size,
            memory_type_bits: props.memory_type_bits,
            format: format_props.format,
            external_format: format_props.external_format,
            suggested_ycbcr_model: format_props.suggested_ycbcr_model,
            suggested_ycbcr_range: format_props.suggested_ycbcr_range,
            suggested_x_chroma_offset: format_props.suggested_x_chroma_offset,
            suggested_y_chroma_offset: format_props.suggested_y_chroma_offset,
        })
    }

    /// Imports an Android hardware buffer as a Vulkan image.
    ///
    /// Creates a VkImage, allocates memory, and sets up YCbCr conversion
    /// for the imported hardware buffer.
    ///
    /// # Safety
    ///
    /// The buffer handle must be valid and the reference count must have been
    /// incremented (via AHardwareBuffer_acquire).
    pub unsafe fn import_buffer(
        &self,
        buffer_handle: Arc<HardwareBufferHandle>,
        width: u32,
        height: u32,
    ) -> VulkanResult<ImportedHardwareBuffer> {
        let buffer = buffer_handle.as_ptr();

        // Query buffer properties first
        let props = self.query_buffer_properties(buffer)?;

        tracing::debug!(
            "Importing AHardwareBuffer: {}x{}, format={:?}, external_format={}, \
             allocation_size={}, memory_type_bits={:#x}",
            width,
            height,
            props.format,
            props.external_format,
            props.allocation_size,
            props.memory_type_bits
        );

        // Use external format if Vulkan format is undefined (vendor-specific YUV)
        let use_external_format = props.format == vk::Format::UNDEFINED;

        // Create YCbCr conversion if using external format
        let ycbcr_conversion = if use_external_format {
            Some(self.create_ycbcr_conversion(&props)?)
        } else {
            None
        };

        // Create the VkImage
        let image = self.create_image(&props, width, height)?;

        // Allocate and bind memory
        let memory = self.allocate_and_bind_memory(image, buffer, &props)?;

        // Create image view
        let image_view = self.create_image_view(image, &props, ycbcr_conversion)?;

        // Create sampler with YCbCr conversion
        let sampler = self.create_sampler(ycbcr_conversion)?;

        Ok(ImportedHardwareBuffer {
            image,
            image_view,
            memory,
            ycbcr_conversion,
            sampler,
            width,
            height,
            _buffer_handle: buffer_handle,
        })
    }

    /// Creates a VkSamplerYcbcrConversion for YUV to RGB conversion.
    unsafe fn create_ycbcr_conversion(
        &self,
        props: &HardwareBufferProperties,
    ) -> VulkanResult<vk::SamplerYcbcrConversion> {
        // External format must be specified for vendor-specific formats
        let mut external_format =
            vk::ExternalFormatANDROID::default().external_format(props.external_format);

        let create_info = vk::SamplerYcbcrConversionCreateInfo::default()
            .format(vk::Format::UNDEFINED) // Use external format
            .ycbcr_model(props.suggested_ycbcr_model)
            .ycbcr_range(props.suggested_ycbcr_range)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .x_chroma_offset(props.suggested_x_chroma_offset)
            .y_chroma_offset(props.suggested_y_chroma_offset)
            .chroma_filter(vk::Filter::LINEAR)
            .force_explicit_reconstruction(false)
            .push_next(&mut external_format);

        let mut conversion = vk::SamplerYcbcrConversion::null();
        let result = (self.create_ycbcr_conversion)(
            self.device.handle(),
            &create_info,
            ptr::null(),
            &mut conversion,
        );

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::YcbcrConversionFailed(format!(
                "vkCreateSamplerYcbcrConversion failed: {:?}",
                result
            )));
        }

        tracing::debug!(
            "Created YCbCr conversion: model={:?}, range={:?}",
            props.suggested_ycbcr_model,
            props.suggested_ycbcr_range
        );

        Ok(conversion)
    }

    /// Creates a VkImage for the imported hardware buffer.
    unsafe fn create_image(
        &self,
        props: &HardwareBufferProperties,
        width: u32,
        height: u32,
    ) -> VulkanResult<vk::Image> {
        // External format for vendor-specific YUV
        let mut external_format =
            vk::ExternalFormatANDROID::default().external_format(props.external_format);

        // External memory info
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);

        // Build base image create info
        let mut image_create_info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::empty())
            .image_type(vk::ImageType::TYPE_2D)
            .format(props.format) // May be UNDEFINED for external format
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
            .push_next(&mut external_memory_info);

        // Chain external format if using undefined format (for vendor-specific YUV)
        if props.format == vk::Format::UNDEFINED {
            image_create_info = image_create_info.push_next(&mut external_format);
        }

        let image = self
            .device
            .create_image(&image_create_info, None)
            .map_err(|e| VulkanError::ImageCreationFailed(format!("{:?}", e)))?;

        Ok(image)
    }

    /// Allocates device memory and binds it to the image.
    unsafe fn allocate_and_bind_memory(
        &self,
        image: vk::Image,
        buffer: *mut c_void,
        props: &HardwareBufferProperties,
    ) -> VulkanResult<vk::DeviceMemory> {
        // Import info with the hardware buffer (cast to ash's AHardwareBuffer pointer type)
        let mut import_info =
            vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(buffer.cast());

        // Dedicated allocation is required for hardware buffer import
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);

        // Find suitable memory type
        let memory_type_index = self.find_memory_type_index(props.memory_type_bits)?;

        // Chain: MemoryAllocateInfo → MemoryDedicatedAllocateInfo → ImportAndroidHardwareBufferInfoANDROID
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(props.allocation_size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info)
            .push_next(&mut dedicated_info);

        let memory = self
            .device
            .allocate_memory(&alloc_info, None)
            .map_err(|e| VulkanError::AllocationFailed(format!("{:?}", e)))?;

        // Bind memory to image
        self.device
            .bind_image_memory(image, memory, 0)
            .map_err(|e| VulkanError::AllocationFailed(format!("bind_image_memory: {:?}", e)))?;

        Ok(memory)
    }

    /// Creates an image view for sampling.
    unsafe fn create_image_view(
        &self,
        image: vk::Image,
        props: &HardwareBufferProperties,
        ycbcr_conversion: Option<vk::SamplerYcbcrConversion>,
    ) -> VulkanResult<vk::ImageView> {
        // YCbCr conversion info if using external format
        let mut ycbcr_info =
            ycbcr_conversion.map(|conv| vk::SamplerYcbcrConversionInfo::default().conversion(conv));

        let mut view_create_info = vk::ImageViewCreateInfo::default()
            .flags(vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(props.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        // Chain YCbCr conversion info if present
        if let Some(ref mut info) = ycbcr_info {
            view_create_info = view_create_info.push_next(info);
        }

        let view = self
            .device
            .create_image_view(&view_create_info, None)
            .map_err(|e| VulkanError::ImageCreationFailed(format!("create_image_view: {:?}", e)))?;

        Ok(view)
    }

    /// Creates a sampler with YCbCr conversion.
    unsafe fn create_sampler(
        &self,
        ycbcr_conversion: Option<vk::SamplerYcbcrConversion>,
    ) -> VulkanResult<vk::Sampler> {
        let mut ycbcr_info =
            ycbcr_conversion.map(|conv| vk::SamplerYcbcrConversionInfo::default().conversion(conv));

        let mut sampler_create_info = vk::SamplerCreateInfo::default()
            .flags(vk::SamplerCreateFlags::empty())
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .mip_lod_bias(0.0)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .min_lod(0.0)
            .max_lod(0.0)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);

        // Chain YCbCr conversion info if present
        if let Some(ref mut info) = ycbcr_info {
            sampler_create_info = sampler_create_info.push_next(info);
        }

        let sampler = self
            .device
            .create_sampler(&sampler_create_info, None)
            .map_err(|e| VulkanError::ImageCreationFailed(format!("create_sampler: {:?}", e)))?;

        Ok(sampler)
    }

    /// Finds a suitable memory type index from the available bits.
    unsafe fn find_memory_type_index(&self, type_bits: u32) -> VulkanResult<u32> {
        // Get the first set bit - for hardware buffer import, any suitable type works
        let index = type_bits.trailing_zeros();
        if index >= 32 || (type_bits & (1 << index)) == 0 {
            return Err(VulkanError::AllocationFailed(
                "No suitable memory type found".to_string(),
            ));
        }
        Ok(index)
    }

    /// Destroys an imported hardware buffer's Vulkan resources.
    ///
    /// # Safety
    ///
    /// The imported buffer must not be in use by any pending GPU operations.
    pub unsafe fn destroy_imported_buffer(&self, buffer: ImportedHardwareBuffer) {
        self.device.destroy_sampler(buffer.sampler, None);
        self.device.destroy_image_view(buffer.image_view, None);

        if let Some(conv) = buffer.ycbcr_conversion {
            (self.destroy_ycbcr_conversion)(self.device.handle(), conv, ptr::null());
        }

        self.device.destroy_image(buffer.image, None);
        self.device.free_memory(buffer.memory, None);
        // _buffer_handle is dropped here, releasing the AHardwareBuffer reference
    }
}

/// Checks if the device supports Android hardware buffer import.
///
/// # Safety
///
/// Instance and physical_device must be valid.
pub unsafe fn check_hardware_buffer_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    // Query extension properties
    let extensions = match instance.enumerate_device_extension_properties(physical_device) {
        Ok(exts) => exts,
        Err(_) => return false,
    };

    let required_extensions: [&std::ffi::CStr; 4] = [
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
        VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
    ];

    for required in &required_extensions {
        let found = extensions.iter().any(|ext| {
            let name = std::ffi::CStr::from_ptr(ext.extension_name.as_ptr());
            name == *required
        });
        if !found {
            tracing::warn!("Required Vulkan extension not available: {:?}", required);
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_error_display() {
        let err = VulkanError::ExtensionNotAvailable("VK_ANDROID_foo".to_string());
        assert!(err.to_string().contains("VK_ANDROID_foo"));
    }

    #[test]
    fn test_properties_default() {
        let props = HardwareBufferProperties {
            allocation_size: 1024,
            memory_type_bits: 0x3,
            format: vk::Format::UNDEFINED,
            external_format: 0x12345,
            suggested_ycbcr_model: vk::SamplerYcbcrModelConversion::YCBCR_709,
            suggested_ycbcr_range: vk::SamplerYcbcrRange::ITU_NARROW,
            suggested_x_chroma_offset: vk::ChromaLocation::MIDPOINT,
            suggested_y_chroma_offset: vk::ChromaLocation::MIDPOINT,
        };
        assert_eq!(props.allocation_size, 1024);
        assert_eq!(props.external_format, 0x12345);
    }
}
