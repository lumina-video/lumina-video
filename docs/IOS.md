# iOS Integration Guide

Complete guide for building and integrating lumina-video on iOS.

## Prerequisites

- macOS with Xcode 15+ and Command Line Tools
- [Rust toolchain](https://rustup.rs/) with iOS targets
- [xcodegen](https://github.com/yonaskolb/XcodeGen) (for test harness only)

```bash
# Install Rust iOS targets
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Verify Xcode
xcodebuild -version   # Xcode 15.0+
xcrun --sdk iphoneos --show-sdk-path
```

## Architecture

```text
┌─────────────────────────────────┐
│     Swift App (SwiftUI/UIKit)   │
├─────────────────────────────────┤
│     LuminaVideoBridge (Swift)   │  ios/lumina-video-bridge/
│  LuminaVideoPlayer, delegate,  │
│  LuminaVideoFrame (IOSurface)  │
├─────────────────────────────────┤
│     C FFI (lumina-video-ios)    │  crates/lumina-video-ios/
│  lumina_player_create/destroy,  │
│  poll_frame, play/pause/seek    │
├─────────────────────────────────┤
│     lumina-video-core (Rust)    │  crates/lumina-video-core/
│  MacOSVideoDecoder (AVPlayer),  │
│  VideoToolbox HW decode         │
└─────────────────────────────────┘
```

- **Decoding**: AVPlayer + VideoToolbox (hardware-accelerated, same path as macOS)
- **Audio**: Native AVFoundation (no FFmpeg dependency)
- **Rendering**: Zero-copy via IOSurface → MTLTexture (no CPU readback)
- **FFI**: Poll-based C-ABI, no callbacks. See [FFI contract](ios-ffi-contract.md).

## Building the Static Library

```bash
# Build for both device and simulator (release mode)
./scripts/build-ios.sh

# Or debug mode
./scripts/build-ios.sh --debug
```

This script:
1. Cross-compiles `lumina-video-ios` for `aarch64-apple-ios` and `aarch64-apple-ios-sim`
2. Verifies all expected FFI symbols are exported
3. Runs a Swift link smoke test for both targets

Output:
- `target/aarch64-apple-ios/release/liblumina_video_ios.a` (device)
- `target/aarch64-apple-ios-sim/release/liblumina_video_ios.a` (simulator)

## Integrating into Your App

### 1. Add LuminaVideoBridge as a local SPM dependency

In your Xcode project or `Package.swift`:

```swift
.package(path: "path/to/lumina-video/ios/lumina-video-bridge")
```

Or in xcodegen `project.yml`:

```yaml
packages:
  LuminaVideoBridge:
    path: path/to/lumina-video/ios/lumina-video-bridge

targets:
  YourApp:
    dependencies:
      - package: LuminaVideoBridge
    settings:
      base:
        LIBRARY_SEARCH_PATHS:
          - "$(inherited)"
          - "path/to/lumina-video/target/aarch64-apple-ios-sim/release"
          - "path/to/lumina-video/target/aarch64-apple-ios/release"
        OTHER_LDFLAGS:
          - "$(inherited)"
          - "-lz"
          - "-liconv"
          - "-lbz2"
          - "-lc++"
```

### 2. Create a player and render frames

```swift
import LuminaVideoBridge

let player = try LuminaVideoPlayer(url: "https://example.com/video.m3u8")
player.delegate = self
player.play()

// In delegate callback:
func luminaPlayer(_ player: LuminaVideoPlayer, didReceiveFrame frame: LuminaVideoFrame) {
    // frame.ioSurface → MTLTexture via device.makeTexture(descriptor:iosurface:plane:)
    // Zero-copy: IOSurface is shared GPU memory, no CPU copies needed
}
```

### 3. Metal rendering (zero-copy)

```swift
// Create texture from IOSurface (zero-copy)
let texture = device.makeTexture(
    descriptor: texDesc,
    iosurface: frame.ioSurface! as IOSurfaceRef,
    plane: 0
)
// Render texture with your Metal pipeline
```

## Test Harness

A complete SwiftUI test app is provided at `ios/test-harness/`:

```bash
# Install xcodegen (one-time)
brew install xcodegen

# Generate Xcode project
cd ios/test-harness
xcodegen generate

# Build for simulator
xcodebuild -project LuminaTestHarness.xcodeproj \
  -scheme LuminaTestHarness \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
  build

# Or open in Xcode
open LuminaTestHarness.xcodeproj
```

The test harness validates: player lifecycle, Metal zero-copy rendering, playback controls (play/pause, seek, volume), and diagnostics (codec info, FPS, render path).

## Requirements

| Component | Version |
|-----------|---------|
| macOS | 13+ (build host) |
| Xcode | 15.0+ |
| iOS Deployment Target | 16.0+ |
| Device | arm64 (iPhone/iPad) |
| Simulator | arm64 Apple Silicon Mac |

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Undefined symbols: _lumina_player_*` | Missing static library | Run `./scripts/build-ios.sh` first, check `LIBRARY_SEARCH_PATHS` |
| `No such module 'LuminaVideoBridge'` | SPM dependency not resolved | Add LuminaVideoBridge package path to your project |
| `building for 'iOS-simulator', but linking in object file built for 'iOS'` | Wrong library arch | Ensure both `aarch64-apple-ios` and `aarch64-apple-ios-sim` are built |
| `x86_64` linker errors on simulator | Intel simulator slice | Add `EXCLUDED_ARCHS[sdk=iphonesimulator*]: x86_64` to build settings |
| Black screen (no video) | IOSurface nil on simulator | CPU-only decode on some simulators; test on device for zero-copy |
| `-lz` / `-liconv` / `-lbz2` linker errors | Missing system libraries | Add to `OTHER_LDFLAGS`: `-lz -liconv -lbz2 -lc++` |

## Related Documentation

- [FFI contract](ios-ffi-contract.md) — C-ABI function signatures, threading model, memory ownership
- [Zero-copy rendering](ZERO-COPY.md) — IOSurface architecture across platforms
- [Platform support](PLATFORMS.md) — Cross-platform comparison
