# Zero-Copy GPU Rendering

Technical deep-dive on lumina-video's zero-copy rendering pipeline.

## Overview

Zero-copy rendering eliminates the CPU→GPU memory copy by importing hardware decoder output directly as GPU textures. This is **enabled by default** on supported platforms.

```toml
[dependencies]
# Zero-copy enabled by default
lumina-video = { git = "https://github.com/lumina-video/lumina-video" }
```

To disable (e.g., for iOS or WASM):
```toml
lumina-video = { git = "https://github.com/lumina-video/lumina-video", default-features = false, features = ["macos-native-video"] }
```

## Platform Implementation

| Platform | Source | Backend | Extension/API |
|----------|--------|---------|---------------|
| macOS | IOSurface | Metal | `newTextureWithDescriptor:iosurface:plane:` |
| Linux | DMABuf | Vulkan | `VK_EXT_external_memory_dma_buf` |
| Android | AHardwareBuffer | Vulkan | `VkSamplerYcbcrConversion` + raw Vulkan import |
| Windows | D3D11 shared handle | D3D12 | `ID3D12Device::OpenSharedHandle()` |

## Performance Benefits

| Metric | CPU Copy | Zero-Copy | Improvement |
|--------|----------|-----------|-------------|
| Memory bandwidth | 2x (decode + upload) | 1x (decode only) | 50% reduction |
| Latency | ~2-5ms per frame | <1ms per frame | 2-5x faster |
| CPU usage | Higher (memcpy) | Lower (GPU only) | Reduced |
| Power consumption | Higher | Lower | Better battery life |

## Automatic Fallback

Zero-copy can fail for various reasons. The library provides graceful degradation:

1. **Try zero-copy first**: If hardware decoder provides a GPU-accessible surface
2. **Fall back to CPU copy**: If zero-copy fails or isn't available
3. **Log once**: A warning is logged on first fallback (no spam)
4. **Track statistics**: Session summary shows zero-copy effectiveness

```text
WARN  MacOSVideoDecoder: IOSurface not available, using CPU fallback.
      pixel_format=420v, dimensions=1920x1080, codec=videotoolbox.

INFO  MacOSVideoDecoder: Decoded 1847 frames (98.2% zero-copy, 33 CPU fallback)
```

## When Zero-Copy May Not Be Available

- Software decode fallback (no hardware decoder)
- Unsupported pixel format (e.g., some YUV variants)
- Missing Vulkan extensions (Linux/Android)
- Driver limitations
- Protected content (DRM)

## Multi-Plane YUV Import (Linux/Android)

For YUV formats like NV12 and YUV420p, zero-copy uses multi-plane import with shader-based color conversion:

```text
NV12:     Y plane (R8, full res) + UV plane (RG8, half res)
YUV420p:  Y plane (R8) + U plane (R8) + V plane (R8)
```

Each plane is imported as a separate wgpu texture, then combined in the fragment shader for YUV→RGB conversion. This avoids the complexity of `VkSamplerYcbcrConversion` while maintaining zero-copy benefits.

## Known Limitations

| Platform | Limitation | Reason | Status |
|----------|------------|--------|--------|
| **Linux** | Single-FD multi-plane requires DRM modifier extension | VA-API outputs single-FD with plane offsets | Fails fast if extension unavailable |
| **Linux** | Intel ANV driver (Mesa ≤25.1) may show color corruption | UV plane offset handling differs from spec | Works on AMD RADV; Intel fix pending |
| **Linux** | PulseAudio causes video freeze after 2-4 seeks on HTTP | pulsesink buffer/clock state corruption | Workaround: alsasink (default) |
| **Windows** | NV12 format not yet supported | DXVA decoders may output NV12 instead of BGRA | CPU fallback for non-BGRA |

## Android: VkSamplerYcbcrConversion Approach

Android achieves 1 GPU hop (zero CPU copies) by bypassing wgpu and using raw Vulkan:

```
MediaCodec (YUV) → HardwareBuffer → Vulkan Import → YCbCr Blit Pass → RGBA Texture → wgpu → egui
                                                    ↑
                                              (1 GPU hop)
```

**Why this works:** While wgpu doesn't expose `VK_ANDROID_external_memory_android_hardware_buffer`, we can use raw Vulkan to import the AHardwareBuffer, perform YUV→RGBA conversion on GPU via `VkSamplerYcbcrConversion`, then hand the resulting RGBA texture to wgpu.

**Performance comparison:**

| Approach | CPU Copies | GPU Copies | Memory Bandwidth |
|----------|------------|------------|------------------|
| VkSamplerYcbcrConversion (PR #22) | 0 | 1 | ~1x frame size |
| CPU-assisted fallback | 1+ | 1 | ~2x frame size |
| FFmpeg (CPU path) | 2+ | 1 | ~3x frame size |

See [PR #22](https://github.com/lumina-video/lumina-video/pull/22) for implementation details.

### Linux Details

- Single-FD multi-plane DMABuf layouts (common with VA-API) require `VK_EXT_image_drm_format_modifier` to specify per-plane offsets via `VkSubresourceLayout`
- Without the extension, single-FD multi-plane imports fail fast to avoid color corruption

### Android YUV Import Flow (PR #22)

1. Import AHardwareBuffer via raw Vulkan (`VK_ANDROID_external_memory_android_hardware_buffer`)
2. Create VkSamplerYcbcrConversion for GPU-side YUV→RGBA
3. Blit to RGBA texture (1 GPU hop)
4. Hand RGBA texture to wgpu for egui rendering
5. If VkSamplerYcbcrConversion unavailable, falls back to CPU-assisted path

## Testing Status

| Platform | Status |
|----------|--------|
| macOS | Fully tested and working |
| Linux | Fully tested and working (DMABuf → Vulkan) |
| Web | 1 GPU hop via copyExternalImageToTexture |
| Android | 1 GPU hop via VkSamplerYcbcrConversion (PR #22) |
| Windows | Implemented, needs device testing |
