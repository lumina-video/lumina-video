# lumina-video

*It just works.*

[![CI](https://github.com/lumina-video/lumina-video/actions/workflows/ci.yml/badge.svg)](https://github.com/lumina-video/lumina-video/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/lumina-video/lumina-video)


> ⚠️ **Experimental** 👷 - macOS, Linux, and Web are tested and working with zero-copy GPU rendering. Android achieves 1 GPU hop. Windows is untested.

Hardware-accelerated embedded video player for [egui](https://github.com/emilk/egui) with zero-copy GPU rendering.

**Goal:** Be the most performant embedded video player for apps built on egui.

## Quick Start

```toml
[dependencies]
lumina-video = { git = "https://github.com/lumina-video/lumina-video" }
# That's it! Native decoders + zero-copy GPU rendering are always-on.
# No feature flags needed for macOS, Linux, Android, or Web.
```

```rust
use lumina_video::VideoPlayer;

// In your egui app:
egui::CentralPanel::default().show(ctx, |ui| {
    if self.player.is_none() {
        self.player = Some(VideoPlayer::new("https://example.com/video.mp4"));
    }
    if let Some(player) = &mut self.player {
        player.show(ui, [640.0, 360.0].into());
    }
});
```

> **egui version:** Targets egui 0.31. For 0.33 support, [open an issue](https://github.com/lumina-video/lumina-video/issues).

## Running the Demo

### macOS

**Prerequisites:** macOS 10.13+, [Rust toolchain](https://rustup.rs/)

```bash
# One-time setup for MKV/WebM support (FFmpeg)
./scripts/setup-macos-ffmpeg.sh

# Set SDKROOT and run demo
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
cargo run --package lumina-video-demo
```

> **Note:** FFmpeg is required on macOS for MKV/WebM container support. MP4 files work without it via VideoToolbox.

<details>
<summary><b>Linux</b></summary>

#### Ubuntu 24.04+ (Recommended)

**Zero-dependency option** — GStreamer libraries are bundled automatically:

```bash
cargo add lumina-video --features vendored-runtime
cargo build  # Downloads GStreamer automatically
```

**Or install system packages:**

```bash
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gstreamer1.0-vaapi

cargo add lumina-video
cargo build
```

#### Ubuntu 22.04

**Option 1: PPA for GStreamer 1.24 (recommended)** — zero-copy works:

```bash
# Add savoury1 PPA for GStreamer 1.24+
sudo add-apt-repository ppa:savoury1/multimedia
sudo apt update

# Install GStreamer
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    gstreamer1.0-vaapi

cargo build
```

**Option 2: Nix** — handles all dependencies:

```bash
nix develop github:lumina-video/lumina-video
cargo build
```

**Option 3: Flatpak** — for distributing apps:

```bash
flatpak install --user ./lumina-video-demo-*.flatpak
```

> **Note:** The `vendored-runtime` feature requires Ubuntu 24.04+ (glibc 2.39). Ubuntu 22.04 users should use the PPA, Nix, or Flatpak.

#### Fedora 38+

```bash
sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel \
    gstreamer1-plugins-good gstreamer1-plugins-bad-free gstreamer1-libav \
    gstreamer1-vaapi
cargo build
```

#### NixOS / Nix

No manual dependency installation needed:

```bash
nix run github:lumina-video/lumina-video
```

> **Note:** First run downloads and builds dependencies (may take several minutes). Subsequent runs are instant.

Or enter a development shell with all dependencies:

```bash
nix develop github:lumina-video/lumina-video
cargo run --package lumina-video-demo
```

#### Pre-built Demo Packages

Download from [GitHub Releases](https://github.com/lumina-video/lumina-video/releases):

```bash
# Debian/Ubuntu 24.04+
sudo dpkg -i lumina-video-demo_*_amd64.deb

# Fedora/RHEL
sudo rpm -i lumina-video-demo-*.rpm

# Any distro (including Ubuntu 22.04)
flatpak install --user ./lumina-video-demo-*.flatpak
```

</details>

<details>
<summary><b>Web</b></summary>

**Prerequisites:** [Rust toolchain](https://rustup.rs/), modern browser with WebGPU (Chrome 113+, Firefox 141+, Safari 26+)

```bash
# Add WASM target and install trunk
rustup target add wasm32-unknown-unknown
cargo install trunk

# Run the web demo
cd crates/lumina-video-web-demo
trunk serve --open
```

</details>

<details>
<summary><b>Android</b></summary>

**Prerequisites:** [Rust toolchain](https://rustup.rs/), Android SDK (API 35), Android NDK 26.x, connected device or emulator

```bash
# Add Android target and install cargo-ndk
rustup target add aarch64-linux-android
cargo install cargo-ndk

# Build native library
cargo ndk -t arm64-v8a build --package lumina-video-android --release

# Copy to jniLibs
mkdir -p android/app/src/main/jniLibs/arm64-v8a
cp target/aarch64-linux-android/release/liblumina_video_android.so \
   android/app/src/main/jniLibs/arm64-v8a/

# Build and install APK
cd android && ./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

**[Full Android guide →](docs/ANDROID.md)**

</details>

<details>
<summary><b>Windows</b></summary>

**Prerequisites:** Windows 10+, [Rust toolchain](https://rustup.rs/), LLVM (for FFmpeg audio)

```bash
# Install LLVM for FFmpeg compilation
choco install llvm

# Run demo with native video feature
cargo run --package lumina-video-demo --features windows-native-video
```

> **Note:** Windows native video is opt-in until the zero-copy implementation is validated. See [PR #23](https://github.com/lumina-video/lumina-video/pull/23).

</details>

## Features

- **Hardware acceleration** via native decoders (VideoToolbox, MediaCodec, Media Foundation, GStreamer)
- **Zero-copy GPU rendering** where supported — frames go directly from decoder to GPU texture
- **Minimal dependencies** — Linux/Android/Web need no FFmpeg; macOS uses FFmpeg for MKV/WebM containers
- **Subtitles** — SRT and WebVTT with customizable styling
- **HLS streaming** with adaptive bitrate
- **A/V sync** — callback-based audio clock (drift TBC)

## Platform Support

| Platform | Decoder | Zero-Copy | Status |
|----------|---------|-----------|--------|
| macOS | VideoToolbox + FFmpeg | Yes (MP4) | **Tested** |
| Linux | GStreamer + VA-API | Yes (DMABuf) | **Tested** |
| Android | ExoPlayer | 1 GPU hop* | **Tested** |
| Web | HTMLVideoElement | GPU-to-GPU | **Tested** |
| Windows | Media Foundation | Pending testing | WIP |

\* Uses `VkSamplerYcbcrConversion` for GPU-side YUV→RGBA (no CPU copies).

**[Detailed platform requirements →](docs/PLATFORMS.md)**

### Pending Improvements

| PR | Platform | Enhancement |
|----|----------|-------------|
| [#23](https://github.com/lumina-video/lumina-video/pull/23) | Windows | Zero-copy diagnostics |
| [#24](https://github.com/lumina-video/lumina-video/pull/24) | Windows | A/V sync bridge |

> **Testers wanted!** Windows zero-copy needs validation. Try the PR branches and report results.

## Configuration

**Zero-copy + native decoders are always-on** for supported platforms. No feature flags needed!

| Platform | What's Included | Feature Flag |
|----------|-----------------|--------------|
| macOS | VideoToolbox + FFmpeg (MKV/WebM) | None (always-on) |
| Linux | GStreamer + Vulkan DMABuf | None (always-on) |
| Android | MediaCodec + Vulkan AHardwareBuffer | None (always-on) |
| Web | HTMLVideoElement + WebGPU | None (always-on) |
| Windows | Media Foundation + D3D12 | `windows-native-video` |

**Optional features:**

| Feature | Description |
|---------|-------------|
| `windows-native-video` | Enable Windows native video (opt-in until validated) |
| `vendored-runtime` | Bundle GStreamer libraries with binary (Linux only, no system deps) |

<details>
<summary><b>Vendored Runtime (Linux)</b></summary>

The `vendored-runtime` feature bundles GStreamer libraries with your binary, eliminating the need for users to install system packages. Useful for AppImage, standalone binaries, or environments where GStreamer installation is difficult.

```toml
[dependencies]
lumina-video = { git = "...", features = ["vendored-runtime"] }
```

**Setup:** Run `./scripts/fetch-vendor-libs.sh linux-x86_64` to populate the vendor directory, then build with the feature enabled.

**LGPL Compliance:** GStreamer is LGPL-2.1+. When distributing vendored binaries, you must provide source availability for relinking. See `vendor/README.md` for details.

</details>

## API Overview

```rust
impl VideoPlayer {
    // Construction
    fn new(url: impl Into<String>) -> Self;
    fn with_controls(self, show: bool) -> Self;
    fn with_subtitles(self, enabled: bool) -> Self;
    fn with_subtitle_style(self, style: SubtitleStyle) -> Self;

    // Playback
    fn play(&mut self);
    fn pause(&mut self);
    fn toggle_playback(&mut self);
    fn seek(&mut self, position: Duration);
    fn set_muted(&mut self, muted: bool);

    // Query
    fn duration(&self) -> Option<Duration>;
    fn position(&self) -> Option<Duration>;

    // Rendering
    fn show(&mut self, ui: &mut Ui, size: Vec2) -> VideoPlayerResponse;

    // Subtitles
    fn load_subtitles_srt(&mut self, content: &str) -> Result<()>;
    fn load_subtitles_vtt(&mut self, content: &str) -> Result<()>;
}
```

<details>
<summary><b>Subtitles example</b></summary>

```rust
use lumina_video::{VideoPlayer, SubtitleStyle};

let style = SubtitleStyle {
    font_size: 24.0,
    text_color: [255, 255, 255, 255],
    background_color: [0, 0, 0, 180],
    ..Default::default()
};

let player = VideoPlayer::new(url)
    .with_subtitle_style(style)
    .with_subtitles(true);

player.load_subtitles_srt(srt_content)?;
```

Full Unicode support including CJK, Cyrillic, Thai, Arabic. See font configuration in [docs/PLATFORMS.md](docs/PLATFORMS.md).

</details>

<details>
<summary><b>Custom controls example</b></summary>

```rust
let mut player = VideoPlayer::new(url).with_controls(false);
player.show(ui, size);

if ui.button("Play/Pause").clicked() {
    player.toggle_playback();
}

if let Some(duration) = player.duration() {
    let pos = player.position().unwrap_or_default();
    ui.label(format!("{:.1}s / {:.1}s", pos.as_secs_f32(), duration.as_secs_f32()));
}
```

</details>

## Building

```bash
# macOS (one-time FFmpeg setup, then set SDKROOT for each terminal session)
./scripts/setup-macos-ffmpeg.sh
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
cargo build

# Linux (install GStreamer dev libs + VA-API for zero-copy)
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-vaapi
cargo build

# Linux (alternative: vendored GStreamer, no system deps)
cargo build --features vendored-runtime

# Windows (native video is opt-in)
cargo build --features windows-native-video
```

**[Android build guide →](docs/ANDROID.md)**

## Architecture

```text
lumina-video
├── video_player.rs    # Main VideoPlayer widget
├── video_texture.rs   # GPU texture management, YUV→RGB shaders
├── frame_queue.rs     # Thread-safe frame buffer
├── macos_video.rs     # VideoToolbox (macOS)
├── linux_video.rs     # GStreamer (Linux)
├── windows_video.rs   # Media Foundation (Windows)
├── android_video.rs   # ExoPlayer/MediaCodec (Android)
└── audio.rs           # Audio playback, A/V sync
```

**[Zero-copy internals →](docs/ZERO-COPY.md)** | **[A/V sync details →](docs/AV-SYNC.md)**

## Alternatives

| Feature | lumina-video | [re_video](https://github.com/rerun-io/rerun) | [egui-video](https://github.com/n00kii/egui-video) | [rvp](https://github.com/v0l/rvp) |
|---------|----------|----------|-------------|-----|
| **HW Decode** | Yes (native) | Web only | No | No |
| **Rendering** | Zero-copy GPU | CPU decode, GPU convert | CPU upload | CPU upload |
| **Android/Web** | Yes / Yes | No / Yes | No | No |
| **External deps** | macOS: FFmpeg; others: none | FFmpeg | FFmpeg+SDL2 | FFmpeg |
| **License** | MIT/Apache-2.0 | MIT/Apache | MIT | MIT |

**lumina-video**: Hardware-accelerated decoding + zero-copy GPU rendering. Android, Web, and desktop (macOS/Linux tested; Windows WIP).

**re_video**: Part of Rerun SDK. Native uses FFmpeg software decode + GPU conversion; Web uses browser WebCodecs (HW accelerated). No native Android.

**egui-video/rvp**: FFmpeg software decode + CPU upload. Desktop only. Requires release builds (egui-video).

## License

Dual-licensed under MIT or Apache 2.0, at your option.

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

## Credits

lumina-video was created as a performant video solution for [Damus Notedeck](https://github.com/damus-io/notedeck). lumina-video and Damus Notedeck are built on [egui](https://github.com/emilk/egui).
