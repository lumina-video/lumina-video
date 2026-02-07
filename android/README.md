# lumina-video Android Bridge

Zero-copy video rendering for Android using ExoPlayer and lumina-video.

## Overview

This module provides a Kotlin bridge between ExoPlayer and lumina-video's Rust rendering pipeline, enabling zero-copy GPU frame sharing via Android's HardwareBuffer API.

```
ExoPlayer → ImageReader → HardwareBuffer → JNI → Vulkan → wgpu → egui
```

## Requirements

- Android API 26+ (required for HardwareBuffer)
- ExoPlayer / Media3
- lumina-video with `zero-copy` feature enabled
- Vulkan-capable device

## Installation

### Gradle

Add the dependency to your app's `build.gradle.kts`:

```kotlin
dependencies {
    implementation(project(":lumina-video-bridge"))
    // Or if published to Maven:
    // implementation("com.luminavideo:lumina-video-bridge:1.0.0")
}
```

### Native Library

Ensure `liblumina_video_android.so` is included in your APK's `jniLibs/arm64-v8a/` directory.

## Usage

### Basic Setup

```kotlin
import com.luminavideo.bridge.ExoPlayerBridge
import androidx.media3.exoplayer.ExoPlayer

class VideoPlayerActivity : AppCompatActivity() {
    private lateinit var player: ExoPlayer
    private lateinit var bridge: ExoPlayerBridge

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create ExoPlayer
        player = ExoPlayer.Builder(this).build()

        // Create and attach the bridge
        bridge = ExoPlayerBridge()
        bridge.attachToPlayer(player)

        // Set up your media source
        val mediaItem = MediaItem.fromUri("https://example.com/video.mp4")
        player.setMediaItem(mediaItem)
        player.prepare()
        player.play()
    }

    override fun onDestroy() {
        super.onDestroy()
        bridge.release()
        player.release()
    }
}
```

### How It Works

1. **Video Size Detection**: The bridge listens for `onVideoSizeChanged` to get the correct dimensions before creating the ImageReader.

2. **ImageReader Configuration**: Creates an ImageReader with:
   - `ImageFormat.PRIVATE` (required for HardwareBuffer)
   - `USAGE_GPU_SAMPLED_IMAGE | USAGE_GPU_COLOR_OUTPUT`
   - Buffer count of 3 for smooth playback

3. **Frame Extraction**: For each decoded frame:
   - Acquires the latest image from ImageReader
   - Extracts the HardwareBuffer
   - Submits to Rust via JNI

4. **Zero-Copy Import**: Rust imports the HardwareBuffer as a Vulkan texture without any CPU copies.

## API Reference

### ExoPlayerBridge

```kotlin
class ExoPlayerBridge {
    /**
     * Attaches this bridge to an ExoPlayer instance.
     * Call this after creating the player but before starting playback.
     */
    fun attachToPlayer(exoPlayer: ExoPlayer)

    /**
     * Returns the number of frames successfully submitted to Rust.
     */
    fun getFrameCount(): Int

    /**
     * Releases all resources. Must be called when done with video playback.
     */
    fun release()
}
```

## Troubleshooting

### BufferQueue Abandoned Error

If you see `BufferQueueProducer: queueBuffer: BufferQueue has been abandoned`:
- Ensure the ImageReader dimensions match the video resolution
- Don't create the ImageReader before receiving `onVideoSizeChanged`

### Black or Corrupt Frames

- Verify your device supports `VK_ANDROID_external_memory_android_hardware_buffer`
- Check that the `zero-copy` feature is enabled in lumina-video
- Ensure API level is 26+

### Performance Issues

- The bridge runs ImageReader callbacks on a background thread
- Frame drops may occur if Rust rendering can't keep up
- Consider reducing video resolution or frame rate

## Technical Details

### JNI Interface

The bridge uses a single JNI method:

```kotlin
external fun nativeSubmitHardwareBuffer(
    buffer: HardwareBuffer,
    timestampNs: Long,
    width: Int,
    height: Int
)
```

### Rust Entry Point

```rust
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeSubmitHardwareBuffer(
    env: JNIEnv,
    _class: JClass,
    buffer: JObject,
    timestamp_ns: jlong,
    width: jint,
    height: jint,
)
```

### Memory Management

- HardwareBuffer ownership is transferred to Rust via `AHardwareBuffer_acquire`
- Rust releases the buffer when the frame is consumed via `AHardwareBuffer_release`
- Java can safely close the Image immediately after the JNI call

## License

Same as lumina-video (MIT or Apache 2.0).
