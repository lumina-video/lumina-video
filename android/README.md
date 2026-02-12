# lumina-video Android Bridge

Zero-copy video rendering for Android using ExoPlayer and lumina-video's Rust/Vulkan pipeline.

## Overview

```
LuminaVideo.init(activity)           [one-time, in onCreate()]
    |
    +--> Rust: AndroidVideoDecoder::new()
             |
             +--> JNI: LuminaVideo.createPlayer(nativeHandle)
                      |
                      +--> ExoPlayer [on HandlerThread]
                               |
                               +--> ImageReader (PRIVATE) -> HardwareBuffer
                                        |
                                        +--> JNI: nativeSubmitHardwareBuffer()
                                                 |
                                                 +--> Vulkan import -> wgpu -> egui
```

## Requirements

- Android API 26+ (required for HardwareBuffer)
- Media3 / ExoPlayer
- Vulkan-capable device with `VK_ANDROID_external_memory_android_hardware_buffer`

Zero-copy is always-on for Android -- no feature flags needed.

## Integration

### 1. Add the bridge dependency

```kotlin
dependencies {
    implementation(project(":lumina-video-bridge"))
}
```

### 2. Initialize in your Activity

```kotlin
import com.luminavideo.bridge.LuminaVideo

class MainActivity : GameActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        // Must be called BEFORE super.onCreate() -- GameActivity's native
        // startup may create AndroidVideoDecoder, which calls createPlayer() via JNI.
        LuminaVideo.init(this)
        super.onCreate(savedInstanceState)
    }
}
```

That's it. Rust calls `VideoPlayer::with_wgpu(url, render_state)` and the bridge handles
ExoPlayer creation, frame extraction, and HardwareBuffer submission automatically.

### Optional: Custom ExoPlayer configuration

```kotlin
val builder = ExoPlayer.Builder(this)
    .setBandwidthMeter(myBandwidthMeter)
LuminaVideo.init(this, builder)
```

## How It Works

1. **`LuminaVideo.init(activity)`** stores the application context and registers a lifecycle observer for cleanup on Activity destroy.

2. **Rust creates a player** via `AndroidVideoDecoder::new()`, which calls `LuminaVideo.createPlayer(nativeHandle)` over JNI.

3. **`createPlayer()`** creates a dedicated `HandlerThread` with a Looper, builds ExoPlayer on that thread, and sets up an `ImageReader` with `ImageFormat.PRIVATE` and GPU usage flags.

4. **Frame extraction**: On each decoded frame, `onImageAvailable` extracts the `HardwareBuffer` and calls `nativeSubmitHardwareBuffer()` to submit it to Rust's per-player frame queue.

5. **Vulkan import**: Rust imports the HardwareBuffer as a Vulkan texture. The path depends on the buffer format:
   - **RGBA buffers**: direct single-plane Vulkan import (true zero-copy)
   - **YUV buffers** (NV12/Y8Cb8Cr8_420): `VkSamplerYcbcrConversion` for GPU-side YUV-to-RGBA (true zero-copy, 1 GPU hop)
   - **Fallback**: CPU-assisted `lockPlanes` + memcpy when Vulkan YUV import is unavailable

## API Reference

### LuminaVideo

```kotlin
object LuminaVideo {
    /**
     * Initialize lumina-video. Call once in onCreate() before super.onCreate().
     * Only applicationContext is retained (no Activity leak).
     */
    fun init(activity: Activity, builder: ExoPlayer.Builder? = null)

    /**
     * Called from Rust JNI -- do not call directly.
     * Creates ExoPlayer on a dedicated HandlerThread, blocks until ready.
     */
    fun createPlayer(nativeHandle: Long): ExoPlayerBridge?
}
```

### ExoPlayerBridge

Created internally by `LuminaVideo.createPlayer()` -- do not construct directly.

```kotlin
class ExoPlayerBridge internal constructor(nativeHandle: Long) {
    fun play(url: String)             // Load and play a video URL
    fun pause()                       // Pause playback
    fun resume()                      // Resume playback
    fun seek(positionMs: Long)        // Seek to position
    fun setVolume(volume: Float)      // Set volume (0.0 - 1.0)
    fun setMuted(muted: Boolean)      // Mute/unmute
    fun getCurrentPosition(): Long    // Current position in ms (cached, any-thread safe)
    fun getDuration(): Long           // Duration in ms, or -1 if unknown
    fun getPlayerId(): Long           // Per-player queue ID
    fun getFrameCount(): Int          // Frames submitted to Rust
    fun release()                     // Release all resources (idempotent)
}
```

### JNI Native Methods

All private to `ExoPlayerBridge`:

| Method | Direction | Purpose |
|--------|-----------|---------|
| `nativeGeneratePlayerId()` | Kotlin -> Rust | Get unique per-player queue ID |
| `nativeReleasePlayer(playerId)` | Kotlin -> Rust | Release player's frame queue and stats |
| `nativeSubmitHardwareBuffer(buffer, timestampNs, width, height, playerId, fenceFd)` | Kotlin -> Rust | Submit decoded frame |
| `nativeOnVideoSizeChanged(handle, width, height)` | Kotlin -> Rust | Notify dimension change |
| `nativeOnPlaybackStateChanged(handle, state)` | Kotlin -> Rust | Notify state transition |
| `nativeOnError(handle, message)` | Kotlin -> Rust | Notify playback error |
| `nativeOnDurationChanged(handle, durationMs)` | Kotlin -> Rust | Notify duration update |

### MoqMediaCodecBridge

For MoQ live streaming, a separate `MoqMediaCodecBridge` receives raw NAL units from Rust
and decodes them directly via MediaCodec (bypassing ExoPlayer for lower latency).

## Multi-Player Support

Each player gets a unique ID via `nativeGeneratePlayerId()`. Frames are routed to per-player
queues (`PlayerState` in Rust), so multiple simultaneous players never interfere with each other.
Stats (zero-copy vs. CPU-assisted vs. failed frame counts) are co-located with each player's queue.

## Memory Management

- HardwareBuffer ownership transfers to Rust via `AHardwareBuffer_acquire` in JNI
- Rust releases the buffer via `AHardwareBuffer_release` when the frame is consumed
- Java safely closes the Image immediately after the JNI call
- `release_player_queue()` cleans up both frames and stats when a player is destroyed

## Troubleshooting

### BufferQueue Abandoned Error

`BufferQueueProducer: queueBuffer: BufferQueue has been abandoned`:
- ImageReader dimensions must match video resolution
- The bridge handles this automatically via `onVideoSizeChanged`

### Black or Corrupt Frames

- Verify device supports `VK_ANDROID_external_memory_android_hardware_buffer`
- Check logcat for `ExoPlayerBridge` / `lumina-video` tags
- The zero-copy debug panel (in the demo app) shows per-frame import results

### Performance

- ImageReader callbacks run on a dedicated background HandlerThread
- Frame drops may occur if the Rust rendering can't keep up (check queue depth)
- Per-player queue max size is 8 frames; oldest frames are evicted when full

## License

Same as lumina-video (MIT or Apache 2.0).
