# Vulkan Extensions Reference

Quick reference for Vulkan extensions used in lumina-video's zero-copy video paths.

## Overview

lumina-video uses **direct Vulkan APIs** (via the `ash` crate) to bypass wgpu limitations for hardware buffer import on Android and Linux. This document links to the official Vulkan specifications for these extensions.

**Full Vulkan Documentation**: [Vulkan-Docs Repository](https://github.com/KhronosGroup/Vulkan-Docs)

---

## Android Zero-Copy Extensions

### VK_ANDROID_external_memory_android_hardware_buffer

**Purpose**: Import AHardwareBuffer from MediaCodec into Vulkan for zero-copy rendering.

**Official Spec**: [VK_ANDROID_external_memory_android_hardware_buffer](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_ANDROID_external_memory_android_hardware_buffer.html)

**Implementation**: `crates/lumina-video/src/media/android_vulkan.rs`

**Key Functions**:
- [`vkGetAndroidHardwareBufferPropertiesANDROID`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetAndroidHardwareBufferPropertiesANDROID.html) - Query buffer format, memory requirements, and YCbCr properties
- `VkExternalFormatANDROID` - Handle vendor-specific YUV formats
- `VkImportAndroidHardwareBufferInfoANDROID` - Import the buffer into Vulkan memory

**Usage Pattern**:
```rust
// 1. Query buffer properties
vkGetAndroidHardwareBufferPropertiesANDROID(device, buffer, &mut props);

// 2. Create VkImage with external format
VkImageCreateInfo { external_format: props.external_format, ... }

// 3. Allocate and bind memory with dedicated allocation
VkMemoryAllocateInfo {
    pNext: VkMemoryDedicatedAllocateInfo,
    pNext: VkImportAndroidHardwareBufferInfoANDROID
}
```

**Troubleshooting**:
- **External format undefined**: Use `VkSamplerYcbcrConversion` for vendor-specific YUV
- **Allocation failed**: Ensure dedicated allocation is used
- **Query failed**: Check that buffer is valid and has correct usage flags

---

### VK_KHR_sampler_ycbcr_conversion

**Purpose**: GPU-side YUV to RGB color space conversion (MediaCodec outputs YUV, egui needs RGB).

**Official Spec**: [VK_KHR_sampler_ycbcr_conversion](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_sampler_ycbcr_conversion.html)

**Implementation**: `crates/lumina-video/src/media/android_vulkan.rs:404-451`

**Key Types**:
- `VkSamplerYcbcrConversion` - Conversion object
- `VkSamplerYcbcrConversionInfo` - Attached to sampler/image view
- `VkSamplerYcbcrModelConversion` - REC601, REC709, REC2020
- `VkSamplerYcbcrRange` - ITU_FULL or ITU_NARROW

**Usage Pattern**:
```rust
// Create YCbCr conversion from hardware buffer properties
VkSamplerYcbcrConversionCreateInfo {
    format: VK_FORMAT_UNDEFINED,  // Use external format
    ycbcr_model: props.suggested_ycbcr_model,
    ycbcr_range: props.suggested_ycbcr_range,
    pNext: VkExternalFormatANDROID
}

// Attach to image view and sampler
VkImageViewCreateInfo { pNext: VkSamplerYcbcrConversionInfo { conversion } }
VkSamplerCreateInfo { pNext: VkSamplerYcbcrConversionInfo { conversion } }
```

---

### VK_KHR_external_memory

**Purpose**: Foundation for importing external memory (required by Android extension).

**Official Spec**: [VK_KHR_external_memory](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory.html)

**Key Structs**:
- `VkExternalMemoryImageCreateInfo` - Tag image for external memory
- `VkMemoryDedicatedAllocateInfo` - Required for hardware buffer import

---

### VK_EXT_queue_family_foreign

**Purpose**: Allow queues from external APIs (MediaCodec) to access Vulkan images.

**Official Spec**: [VK_EXT_queue_family_foreign](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_queue_family_foreign.html)

**Usage**: Set `sharingMode = CONCURRENT` with `VK_QUEUE_FAMILY_FOREIGN_EXT` in queue family indices.

---

## Linux Zero-Copy Extensions

### VK_EXT_external_memory_dma_buf

**Purpose**: Import VA-API DMABuf file descriptors directly into Vulkan.

**Official Spec**: [VK_EXT_external_memory_dma_buf](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html)

**Implementation**: `crates/lumina-video/src/media/linux_video.rs`

**Key Types**:
- `VkImportMemoryFdInfoKHR` - Import DMABuf FD
- `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` - Handle type

**Usage Pattern**:
```rust
// Import DMABuf FD as Vulkan memory
VkImportMemoryFdInfoKHR {
    handleType: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    fd: dmabuf_fd
}
```

**Troubleshooting**:
- **Import failed**: Check that FD is valid and has correct permissions
- **Color corruption**: May need DRM format modifier (see below)

---

### VK_EXT_image_drm_format_modifier

**Purpose**: Handle tiled memory layouts (Intel Y-tiled, AMD tiled, etc.) for single-FD multi-plane imports.

**Official Spec**: [VK_EXT_image_drm_format_modifier](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_drm_format_modifier.html)

**Implementation**: `crates/lumina-video/src/media/linux_video.rs:77-81`

**Why Critical**: VA-API outputs single-FD DMABuf with plane offsets encoded in DRM modifier. Without this extension, multi-plane imports fail.

**Known Issues**:
- **Intel ANV (Mesa ≤25.1)**: UV plane offset handling bug causes color corruption on NV12
- **AMD RADV**: Works correctly

**Usage Pattern**:
```rust
VkImageDrmFormatModifierExplicitCreateInfoEXT {
    drmFormatModifier: dmabuf_info.modifier,
    drmFormatModifierPlaneCount: n_planes,
    pPlaneLayouts: &subresource_layouts  // Per-plane offset/stride
}
```

---

## Debugging Tools

### Vulkan Validation Layers
**Purpose**: Catch API misuse, memory leaks, and synchronization errors.

**Installation**:
```bash
# Ubuntu/Debian
sudo apt install vulkan-validationlayers

# macOS (via MoltenVK)
brew install molten-vk

# Android
Included in NDK
```

**Enable**:
```bash
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
```

**Common Warnings**:
- `VUID-VkMemoryAllocateInfo-pNext-00639`: Missing dedicated allocation
- `VUID-vkBindImageMemory-memory-01047`: Incompatible memory type

---

### RenderDoc
**Purpose**: GPU frame capture and Vulkan state inspection.

**Download**: [renderdoc.org](https://renderdoc.org/)

**Features**:
- Frame capture with Vulkan API calls
- Texture inspection (check YUV→RGB conversion)
- Memory allocation tracking
- Pipeline state debugging

---

### Android GPU Inspector
**Purpose**: Real-time profiling on Android devices.

**Download**: [gpuinspector.dev](https://gpuinspector.dev/)

**Features**:
- Frame timing analysis
- Memory bandwidth profiling
- Vulkan API trace

---

## Extension Support Matrix

| Extension | Android Min API | Linux Kernel | Notes |
|-----------|----------------|--------------|-------|
| `VK_ANDROID_external_memory_android_hardware_buffer` | 26 (Oreo) | N/A | Android only |
| `VK_KHR_sampler_ycbcr_conversion` | 26 (Oreo) | 4.13+ | Required for Android YUV |
| `VK_EXT_external_memory_dma_buf` | N/A | 4.11+ | Linux only |
| `VK_EXT_image_drm_format_modifier` | N/A | 5.2+ | Required for tiled formats |

---

## Further Reading

- **Vulkan Tutorial**: [vulkan-tutorial.com](https://vulkan-tutorial.com/)
- **Vulkan Guide**: [github.com/KhronosGroup/Vulkan-Guide](https://github.com/KhronosGroup/Vulkan-Guide)
- **Android Hardware Buffer**: [developer.android.com/ndk/reference/group/a-hardware-buffer](https://developer.android.com/ndk/reference/group/a-hardware-buffer)
- **DMABuf Overview**: [kernel.org/doc/html/latest/driver-api/dma-buf.html](https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html)

---

**Questions?** See `docs/ZERO-COPY.md` for high-level architecture or dive into the source:
- Android: `crates/lumina-video/src/media/android_vulkan.rs`
- Linux: `crates/lumina-video/src/media/linux_video.rs`
