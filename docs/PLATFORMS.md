# Platform Support

Detailed platform requirements, supported codecs, and hardware specifications.

## Support Matrix

| Platform | Decoder | HW Decode | Rendering | Status |
|----------|---------|-----------|-----------|--------|
| macOS | VideoToolbox | Yes | Zero-copy (MP4); CPU for MKV | **Tested** |
| Linux | GStreamer + VA-API | Yes | Zero-copy (DMABuf → Vulkan) | **Tested** |
| Android | ExoPlayer + MediaCodec | Yes | 1 GPU hop* | **Tested** |
| Web | HTMLVideoElement + hls.js | Yes (browser) | GPU-to-GPU (WebGPU) | **Tested** |
| Windows | Media Foundation | Yes | Pending testing | **WIP/Untested** |

\* Android uses `VkSamplerYcbcrConversion` for GPU-side YUV→RGBA conversion with zero CPU copies. See [#22](https://github.com/lumina-video/lumina-video/pull/22).

## macOS

- **OS**: macOS 10.13 (High Sierra) or later
- **Hardware**: Any Mac with Metal support
  - Apple Silicon (M1, M2, M3, M4) — native hardware decode
  - Intel Macs with QuickSync (2012 and later)
- **Codecs**: H.264, HEVC, VP9, AV1 (M3+ for AV1)
- **Containers**: MP4, MOV, HLS native; MKV/WebM require `ffmpeg` feature

## Windows

- **OS**: Windows 10 version 1803 or later (Windows 11 recommended)
- **Hardware**: Any GPU with DirectX 11 support
  - NVIDIA: Kepler (GTX 600) or newer
  - AMD: GCN 1.0 (HD 7000) or newer
  - Intel: Haswell (4th gen) or newer
- **Codecs**: H.264, HEVC (may require HEVC Video Extensions on Windows 10; included in Windows 11), AV1 (with AV1 Extensions)

## Linux

- **OS**: Ubuntu 22.04+, Fedora 38+, or any distro with GStreamer 1.24+
- **Hardware**: VA-API compatible GPU with appropriate drivers
  - Intel: Haswell (4th gen) or newer with `intel-media-va-driver`
  - AMD: GCN 1.0 or newer with Mesa VA-API
  - NVIDIA: With proprietary drivers and NVDEC via `nvcodec` plugin
- **Runtime packages**: `gstreamer1.0-plugins-good`, `gstreamer1.0-plugins-bad`, `gstreamer1.0-libav`
- **GStreamer requirement**: 1.24+ required for zero-copy (explicit DRM modifier support via `drm-format` caps field)

> **Note**: The modern `va` plugin (in `gstreamer1.0-plugins-bad`) replaces the deprecated `gstreamer-vaapi` plugin. The `va` decoders have higher rank and support explicit DRM modifiers for zero-copy.

### Linux Audio Configuration

By default, lumina-video uses **alsasink** for audio output on Linux. This bypasses PulseAudio to avoid a known bug where video freezes after 2-4 seeks on HTTP streams.

**Trade-offs:**
- ✅ Reliable video seeking on HTTP streams
- ✅ Audio sharing works via ALSA's dmix plugin
- ⚠️ No per-app volume control in system tray
- ⚠️ No automatic audio device switching (Bluetooth, headphones)

**Environment variables to override:**

| Variable | Audio Sink | Notes |
|----------|------------|-------|
| *(default)* | alsasink | Reliable seeking |
| `LUMINA_VIDEO_PULSE_AUDIO=1` | pulsesink | May freeze after seek on HTTP |
| `LUMINA_VIDEO_PIPEWIRE_AUDIO=1` | pipewiresink | May glitch on backward seek |
| `LUMINA_VIDEO_FAKE_AUDIO=1` | fakesink | No audio output |
| `LUMINA_VIDEO_NO_AUDIO=1` | *(disabled)* | Video only |

## Android

- **OS**: Android 5.0 (API 21) or later
- **Tested devices**: Pixel 4a/6/7/8, Samsung Galaxy S10+/S21/S23, OnePlus 8T/9 Pro
- **Hardware**: Any device with hardware MediaCodec support (virtually all Android 5.0+ devices)
- **Codecs**: H.264, HEVC, VP8, VP9, AV1 (Pixel 8+, recent Samsung flagships)

> **Note**: Hardware acceleration availability varies by device manufacturer and Android version. H.264 is universally supported; HEVC/VP9 support is widespread on 2018+ devices.

## Web

- **Browsers**:
  - Chrome 113+ / Edge 113+ (WebGPU stable)
  - Firefox 141+ (WebGPU stable, WebGL2 51+)
  - Safari 26.0+ (WebGPU stable, WebGL2 15+)
- **HLS Streaming**: Native support in Safari; hls.js for Chrome/Firefox/Edge
- **Codecs**: H.264 (universal), VP9 (Chrome/Firefox/Edge), HEVC (Safari), AV1 (Chrome 94+, Firefox 98+)
- **Requirements**:
  - Modern browser with WebAssembly support
  - `requestVideoFrameCallback` for frame-accurate sync (Chrome 83+, Safari 15.4+, Firefox 132+)

## Supported Formats

| Format | macOS | Linux | Windows | Android |
|--------|-------|-------|---------|---------|
| **Video** |||||
| H.264/AVC | Yes | Yes | Yes | Yes |
| H.265/HEVC | Yes | Yes | Yes* | Yes |
| VP8 | Yes | Yes | No | Yes |
| VP9 | Yes | Yes | No | Yes |
| AV1 | Yes (M3+) | Yes** | Yes** | Yes (newer) |
| **Audio** |||||
| AAC | Yes | Yes | Yes | Yes |
| MP3 | Yes | Yes | Yes | Yes |
| Opus | Yes | Yes | Yes | Yes |
| **Containers** |||||
| MP4/M4V | Yes | Yes | Yes | Yes |
| WebM | FFmpeg | Yes | No | Yes |
| MKV | FFmpeg | Yes | Partial | Yes |
| HLS (m3u8) | Yes | Yes | Yes | Yes |

\* Windows HEVC requires HEVC Video Extensions from Microsoft Store (included in Windows 11)
\*\* AV1 requires appropriate system codecs/plugins

> **Tip**: H.264 + AAC in MP4 container has the broadest compatibility across all platforms.

## Why Native Decoders Over FFmpeg?

| Aspect | Native Decoder | FFmpeg |
|--------|---------------|--------|
| **HW Integration** | Direct API access | Abstraction layer overhead |
| **Memory Efficiency** | Optimal memory locations | Extra copy through libav buffers |
| **Power Consumption** | OS-optimized for battery | Higher power draw |
| **Binary Size** | Uses system libraries (0 MB) | +15-30 MB for FFmpeg libs |
| **Codec Updates** | Automatic via OS updates | Must rebuild/redeploy |

## FFmpeg Fallback

When the `ffmpeg` feature is enabled, lumina-video automatically falls back to FFmpeg for unsupported containers:

| Platform | FFmpeg Fallback When |
|----------|---------------------|
| **macOS** | MKV, WebM, OGG containers; unsupported codecs |
| **Linux** | GStreamer init fails; missing plugins |
| **Windows** | *(FFmpeg primary currently)* |
| **Android** | *(No FFmpeg on Android)* |

Enable with:
```toml
lumina-video = { git = "https://github.com/lumina-video/lumina-video", features = ["ffmpeg"] }
```

## Packaging Recommendations

**End users should never need to install separate video dependencies.**

| Platform | Recommendation |
|----------|---------------|
| **macOS** | VideoToolbox is a system framework — ship your `.app` as-is |
| **Windows** | Media Foundation is built-in. Document HEVC Extensions if needed |
| **Linux** | Declare GStreamer plugins as package dependencies in `.deb`/`.rpm` |
| **Android** | MediaCodec is part of Android. ExoPlayer is bundled in your APK |
