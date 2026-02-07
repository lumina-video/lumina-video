# Android Integration Guide

Complete guide for building and integrating lumina-video on Android.

## Prerequisites

- Android SDK with API 35 (compileSdk)
- Android NDK 26.x
- `ANDROID_NDK_HOME` environment variable pointing to the NDK (cargo-ndk requires this)
- Rust with Android targets: `rustup target add aarch64-linux-android`
- `cargo-ndk`: `cargo install cargo-ndk`

```bash
# Verify your environment
echo "SDK: $ANDROID_HOME"        # e.g. /opt/homebrew/share/android-commandlinetools
echo "NDK: $ANDROID_NDK_HOME"    # e.g. $ANDROID_HOME/ndk/26.1.10909125
ls "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/"  # should list darwin-x86_64 or linux-x86_64
```

## Building the Native Library

```bash
# Build native library for arm64
# -P 26 sets the minimum API level (required — default API 21 is missing libaaudio)
cargo ndk -t arm64-v8a -P 26 build --package lumina-video-android --release

# Copy to jniLibs
mkdir -p android/app/src/main/jniLibs/arm64-v8a
cp target/aarch64-linux-android/release/liblumina_video_android.so \
   android/app/src/main/jniLibs/arm64-v8a/
```

## Building the APK

```bash
# Create local.properties pointing to your SDK
# NOTE: The SDK path varies by install method. Use $ANDROID_HOME if set,
# or check: ~/Library/Android/sdk, /opt/homebrew/share/android-commandlinetools
echo "sdk.dir=$ANDROID_HOME" > android/local.properties

# Build APK
cd android && ./gradlew assembleDebug

# Install on connected device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch the app
adb shell am start -n com.luminavideo.demo/.MainActivity
```

## Requirements

| Component | Version |
|-----------|---------|
| Android SDK | API 35 (compileSdk) |
| Android NDK | 26.x |
| Android Gradle Plugin | 8.7.0+ |
| Gradle | 8.9+ |
| Target device | Android 8.0+ (API 26), arm64-v8a |

## Verifying the Build

```bash
# Check if app is running
adb shell pidof com.luminavideo.demo

# View app logs
adb logcat -s "lumina-video:*" "LuminaVideoDemo:*" "ExoPlayerBridge:*"

# Expected log output on successful start:
# I LuminaVideoDemo: Loaded liblumina_video_android.so
# I lumina-video: android_main: Starting lumina-video demo
# I lumina-video: AHardwareBuffer zero-copy available (API XX)
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `dlopen failed: library "liblumina_video_android.so" not found` | Missing native library | Ensure `liblumina_video_android.so` is in `jniLibs/arm64-v8a/` |
| `assertion failed: previous.is_none()` | ndk-context double init | Use latest code (fixed in commit 7359f445) |
| `ClassNotFoundException: ExoPlayerBridge` | Wrong JNI package | Use latest code (fixed in commit 7359f445) |
| `NoSuchMethodError: play(String)` | Missing Kotlin methods | Use latest Kotlin bridge code |
| `AHardwareBuffer format errors` | Non-fatal wgpu probing | Ignore - failures handled gracefully |
| `unable to find library -laaudio` | API level too low | Pass `-P 26` (or higher) to `cargo ndk` |
| `sdk.dir` not set / wrong path | `local.properties` misconfigured | Run `echo "sdk.dir=$ANDROID_HOME" > android/local.properties` |

## ExoPlayer Bridge Architecture

The Kotlin bridge connects ExoPlayer to the Rust rendering pipeline:

```text
ExoPlayer → ImageReader → HardwareBuffer → JNI → Rust Queue → Vulkan
```

Key requirements:
- API 26+ (HardwareBuffer support)
- `VK_ANDROID_external_memory_android_hardware_buffer` extension
- ImageReader configured with `ImageFormat.PRIVATE` and GPU usage flags

See `android/README.md` for detailed integration instructions.

## Testing Checklist

| Component | Validation |
|-----------|------------|
| ExoPlayerBridge.kt | Real device with ExoPlayer integration |
| JNI entry point | Frame submission from Kotlin to Rust |
| HardwareBuffer queue | Throughput at 30/60fps |
| AHardwareBuffer → Vulkan (RGBA) | Various device GPU drivers |
| AHardwareBuffer YUV extraction | NV12 buffers from MediaCodec |
| Reference counting | Memory leak testing under load |

## Test Plan

1. Build Android app with ExoPlayer and lumina-video-bridge dependency
2. Play various video formats (H.264, HEVC, VP9)
3. Verify frames render correctly (no black frames, correct colors)
4. Monitor memory usage during extended playback
5. Test device rotation and lifecycle events
