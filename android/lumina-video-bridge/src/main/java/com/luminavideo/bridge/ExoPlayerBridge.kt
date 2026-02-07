/**
 * ExoPlayer Bridge for lumina-video zero-copy video rendering.
 *
 * This class bridges ExoPlayer's video output to lumina-video's Rust rendering pipeline
 * using Android's HardwareBuffer API for zero-copy GPU frame sharing.
 *
 * # Usage
 *
 * ```kotlin
 * val bridge = ExoPlayerBridge()
 * bridge.attachToPlayer(exoPlayer)
 * // Frames are automatically submitted to Rust via JNI
 * bridge.release()
 * ```
 *
 * # Requirements
 *
 * - Android API 26+ (HardwareBuffer)
 * - ExoPlayer 2.x or Media3
 * - lumina-video native library loaded
 *
 * # Known Limitation: ImageFormat.PRIVATE yields YUV, not RGBA
 *
 * This implementation uses [ImageFormat.PRIVATE] which is required for HardwareBuffer
 * extraction, but ExoPlayer's video decoder typically produces YUV data (usually NV12),
 * not RGBA. The HardwareBuffer format will be something like Y8Cb8Cr8_420 rather than
 * R8G8B8A8_UNORM.
 *
 * The Rust side correctly validates the buffer format and rejects non-RGBA data,
 * causing a fallback to the CPU conversion path. This means zero-copy GPU import
 * will rarely succeed with the current implementation.
 *
 * Future work: Implement YUV multi-plane import on the Rust/Vulkan side using
 * VkSamplerYcbcrConversion to handle NV12/YUV420p HardwareBuffers directly.
 *
 * Tracking: lumina-video-5hd
 */
package com.luminavideo.bridge

import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.exoplayer.ExoPlayer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Bridge between ExoPlayer and lumina-video's Rust zero-copy rendering.
 *
 * Extracts HardwareBuffer from decoded video frames via ImageReader
 * and submits them to Rust via JNI for GPU-accelerated display.
 */
class ExoPlayerBridge() {
    /**
     * Secondary constructor for JNI creation from Rust.
     * The context and nativeHandle are stored for potential future use.
     *
     * @param context Android application context
     * @param nativeHandle Native pointer for Rust-side state (currently unused)
     */
    @Suppress("UNUSED_PARAMETER")
    constructor(context: android.content.Context, nativeHandle: Long) : this() {
        Log.d(TAG, "Created via JNI with nativeHandle=$nativeHandle")
        this.nativeHandle = nativeHandle
    }

    /** Native handle passed from Rust (if constructed via JNI) */
    private var nativeHandle: Long = 0L

    companion object {
        private const val TAG = "ExoPlayerBridge"

        /** Maximum number of images to hold in the ImageReader buffer */
        private const val MAX_IMAGES = 3

        /** Whether the native library was successfully loaded */
        private var nativeLibraryLoaded = false

        init {
            try {
                // Load the Android native library (contains android_main + JNI functions)
                System.loadLibrary("lumina_video_android")
                nativeLibraryLoaded = true
                Log.i(TAG, "Loaded liblumina_video_android.so")
            } catch (e: UnsatisfiedLinkError) {
                nativeLibraryLoaded = false
                Log.e(TAG, "Failed to load liblumina_video_android.so: ${e.message}")
            }
        }
    }

    // ImageReader for capturing decoded frames
    private var imageReader: ImageReader? = null

    // Surface provided to ExoPlayer for rendering
    private var surface: Surface? = null

    // Background thread for ImageReader callbacks
    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null

    // Volume before muting (to restore on unmute)
    private var volumeBeforeMute: Float = 1.0f

    // Player reference for lifecycle management
    private var player: ExoPlayer? = null

    // State tracking
    private val isReleased = AtomicBoolean(false)
    private val frameCount = AtomicInteger(0)

    // Player listener for cleanup on release
    private var playerListener: Player.Listener? = null

    // Player ID for frame queue routing.
    // Use 0 (legacy/shared queue) for compatibility with Rust VideoPlayer.
    // Multi-player isolation would use nativeGeneratePlayerId() but requires
    // coordinating player IDs between Kotlin and Rust sides.
    private val playerId: Long = 0L

    // Current video dimensions (updated on video size change)
    private var videoWidth = 0
    private var videoHeight = 0

    // True when Surface is provided by native NDK ImageReader (via setVideoSurfaceFromNative)
    // In NDK mode, we don't use Java ImageReader
    private var ndkMode = false

    /**
     * Initializes the bridge for use.
     * Called by Rust code after JNI construction.
     * This is a no-op when ExoPlayer isn't attached yet.
     */
    fun initialize() {
        Log.d(TAG, "initialize() called (playerId=$playerId)")
        // Start background thread for ImageReader callbacks if not already running
        if (handlerThread == null) {
            handlerThread = HandlerThread("LuminaVideoBridge").apply { start() }
            handler = Handler(handlerThread!!.looper)
        }
    }

    /**
     * Loads and plays a video from the given URL.
     * Called by Rust code to start video playback.
     *
     * @param url The video URL to play
     */
    fun play(url: String) {
        Log.d(TAG, "play($url)")
        val p = player
        if (p == null) {
            Log.w(TAG, "play() called but no player attached - creating new player")
            // For JNI-created bridge, we need to create our own player
            // This is handled by the host activity, so just log a warning
            return
        }

        // If player exists and has different content, load new URL
        val mediaItem = androidx.media3.common.MediaItem.fromUri(url)
        p.setMediaItem(mediaItem)
        p.prepare()
        p.play()
    }

    /**
     * Pauses video playback.
     * Called by Rust code.
     */
    fun pause() {
        Log.d(TAG, "pause()")
        player?.pause()
    }

    /**
     * Resumes video playback.
     * Called by Rust code.
     */
    fun resume() {
        Log.d(TAG, "resume()")
        player?.play()
    }

    /**
     * Seeks to the specified position.
     * Called by Rust code.
     *
     * @param positionMs Position in milliseconds
     */
    fun seek(positionMs: Long) {
        Log.d(TAG, "seek($positionMs)")
        player?.seekTo(positionMs)
    }

    /**
     * Sets the playback volume.
     * Called by Rust code.
     *
     * @param volume Volume level (0.0 to 1.0)
     */
    fun setVolume(volume: Float) {
        Log.d(TAG, "setVolume($volume)")
        player?.volume = volume.coerceIn(0f, 1f)
    }

    /**
     * Sets the muted state.
     * Called by Rust code.
     *
     * @param muted True to mute, false to unmute
     */
    fun setMuted(muted: Boolean) {
        Log.d(TAG, "setMuted($muted)")
        player?.let { p ->
            if (muted) {
                // Save current volume before muting
                volumeBeforeMute = p.volume
                p.volume = 0f
            } else {
                // Restore volume from before mute
                p.volume = volumeBeforeMute
            }
        }
    }

    /**
     * Gets the current playback position in milliseconds.
     * Called by Rust code.
     *
     * @return Current position in ms, or 0 if not playing
     */
    fun getCurrentPosition(): Long {
        return player?.currentPosition ?: 0L
    }

    /**
     * Gets the total duration of the video in milliseconds.
     * Called by Rust code.
     *
     * @return Duration in ms, or -1 if unknown
     */
    fun getDuration(): Long {
        val duration = player?.duration ?: -1L
        return if (duration == androidx.media3.common.C.TIME_UNSET) -1L else duration
    }

    /**
     * Extracts the current frame as a byte array (CPU fallback path).
     * Not typically used when zero-copy HardwareBuffer path is working.
     *
     * @return Frame bytes in RGBA format, or null if not available
     */
    fun extractCurrentFrame(): ByteArray? {
        // CPU fallback - not implemented, zero-copy path preferred
        Log.d(TAG, "extractCurrentFrame() called - CPU fallback not implemented")
        return null
    }

    /**
     * Attaches this bridge to an ExoPlayer instance.
     *
     * This configures the player to render to an ImageReader surface,
     * enabling HardwareBuffer extraction for zero-copy rendering.
     *
     * Two modes are supported:
     * 1. NDK ImageReader mode (preferred): Rust creates AImageReader via NDK and passes
     *    Surface via setVideoSurfaceFromNative(). This provides sync fence FD access.
     * 2. Java ImageReader mode (legacy): This method creates Java ImageReader. Used when
     *    NDK mode is not available.
     *
     * @param exoPlayer The ExoPlayer instance to attach to
     */
    fun attachToPlayer(exoPlayer: ExoPlayer) {
        if (isReleased.get()) {
            Log.w(TAG, "Cannot attach to player - bridge is released")
            return
        }

        if (!nativeLibraryLoaded) {
            Log.e(TAG, "Cannot attach to player - native library not loaded")
            return
        }

        player = exoPlayer

        // Start background thread for ImageReader callbacks
        handlerThread = HandlerThread("LuminaVideoBridge").apply { start() }
        handler = Handler(handlerThread!!.looper)

        // Listen for video size changes to notify Rust (for NDK ImageReader recreation)
        // Save listener reference for cleanup on release
        playerListener = object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: VideoSize) {
                Log.d(TAG, "onVideoSizeChanged: ${videoSize.width}x${videoSize.height}")
                if (videoSize.width > 0 && videoSize.height > 0) {
                    videoWidth = videoSize.width
                    videoHeight = videoSize.height
                    // If using Java ImageReader (legacy mode), recreate it
                    // Skip if NDK mode is active (Rust handles ImageReader)
                    if (!ndkMode && imageReader != null) {
                        setupImageReader(videoSize.width, videoSize.height)
                    }
                    // Notify Rust of size change (for NDK ImageReader mode)
                    if (nativeHandle != 0L) {
                        nativeOnVideoSizeChanged(nativeHandle, videoSize.width, videoSize.height)
                    }
                }
            }
        }
        exoPlayer.addListener(playerListener!!)

        // Create initial ImageReader with default size so ExoPlayer has a surface to render to.
        // This is needed because onVideoSizeChanged only fires when there's a surface.
        // The ImageReader will be recreated with correct size once we know the video dimensions.
        // Skip if NDK mode is active (Rust provides its own surface via setVideoSurfaceFromNative).
        if (!ndkMode && imageReader == null && surface == null) {
            Log.i(TAG, "Creating initial ImageReader with default 1920x1080 size")
            setupImageReader(1920, 1080)
        } else {
            Log.i(TAG, "NDK ImageReader mode active; skipping Java ImageReader setup")
        }

        Log.i(TAG, "Attached to ExoPlayer, waiting for video size")
    }

    /**
     * Sets the video surface from native code (NDK ImageReader mode).
     *
     * Called by Rust when using NDK AImageReader for zero-copy with sync fence support.
     * This replaces the Java ImageReader path, allowing access to the raw fence FD
     * via AImageReader_acquireLatestImageAsync().
     *
     * @param surface The Surface from ANativeWindow_toSurface (backed by NDK AImageReader)
     */
    fun setVideoSurfaceFromNative(surface: Surface) {
        Log.d(TAG, "setVideoSurfaceFromNative called - switching to NDK ImageReader mode")

        // Mark NDK mode as active - this prevents Java ImageReader from being created/recreated
        ndkMode = true

        // Release any existing Java ImageReader (we're switching to NDK mode)
        imageReader?.close()
        imageReader = null
        this.surface?.release()
        this.surface = surface

        // Set the surface on the player
        player?.setVideoSurface(surface)

        Log.i(TAG, "Video surface set from native (NDK ImageReader mode active)")
    }

    /**
     * Creates or recreates the ImageReader with the correct dimensions.
     *
     * IMPORTANT: ImageReader dimensions MUST match video resolution,
     * otherwise BufferQueue errors will occur.
     */
    private fun setupImageReader(width: Int, height: Int) {
        if (width == videoWidth && height == videoHeight && imageReader != null) {
            return // Already set up with correct size
        }

        // Release old resources before attempting new creation
        imageReader?.close()
        imageReader = null
        surface?.release()
        surface = null

        Log.i(TAG, "Setting up ImageReader: ${width}x${height}")

        try {
            // Create ImageReader for HardwareBuffer extraction
            // ImageFormat.PRIVATE is required for hardware decoder output
            // Note: Using basic constructor without explicit usage flags
            // as some devices don't support GPU usage combinations
            val newReader = ImageReader.newInstance(
                width,
                height,
                ImageFormat.PRIVATE,
                MAX_IMAGES
            ).apply {
                setOnImageAvailableListener(
                    { reader -> onImageAvailable(reader) },
                    handler
                )
            }

            val newSurface = newReader.surface

            // Give the new surface to ExoPlayer
            player?.setVideoSurface(newSurface)

            // Only update state after successful creation
            imageReader = newReader
            surface = newSurface
            videoWidth = width
            videoHeight = height

            Log.i(TAG, "ImageReader created and attached to player")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ImageReader: ${e.message}")
            // Dimensions NOT updated on failure, allowing retry
        }
    }

    /**
     * Called when a new decoded frame is available in the ImageReader.
     */
    private fun onImageAvailable(reader: ImageReader) {
        if (isReleased.get()) return
        if (!nativeLibraryLoaded) return

        val image = try {
            reader.acquireLatestImage()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to acquire image: ${e.message}")
            return
        } ?: return

        var hardwareBuffer: HardwareBuffer? = null
        try {
            hardwareBuffer = image.hardwareBuffer
            if (hardwareBuffer != null) {
                // Get sync fence from producer (MediaCodec) if available
                // The fence indicates when the producer finished writing to the buffer.
                // Vulkan must wait on this fence before reading from the AHardwareBuffer.
                //
                // TODO: The Java SyncFence API (API 33+) doesn't expose the raw FD.
                // For proper fence passing, we need to use NDK's AImageReader with
                // AImageReader_acquireLatestImageAsync() which returns the fence FD directly.
                // For now, we pass -1 (no fence) and rely on the VK_QUEUE_FAMILY_EXTERNAL
                // queue ownership transfer barrier for synchronization.
                val fenceFd = -1

                // Submit to Rust via JNI
                nativeSubmitHardwareBuffer(
                    hardwareBuffer,
                    image.timestamp,
                    image.width,
                    image.height,
                    playerId,
                    fenceFd
                )

                frameCount.incrementAndGet()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame: ${e.message}")
        } finally {
            // Always close the HardwareBuffer, even if JNI threw
            // Rust has acquired its own reference via AHardwareBuffer_acquire
            hardwareBuffer?.close()
            // Always close the image to return it to the pool
            image.close()
        }
    }

    /**
     * Gets the number of frames submitted to Rust.
     */
    fun getFrameCount(): Int = frameCount.get()

    /**
     * Releases all resources held by this bridge.
     *
     * Must be called when done with video playback.
     */
    fun release() {
        if (isReleased.getAndSet(true)) {
            return // Already released
        }

        Log.i(TAG, "Releasing bridge (submitted $frameCount frames)")

        // Remove listener before releasing player to prevent callbacks after release
        playerListener?.let { listener ->
            player?.removeListener(listener)
        }
        playerListener = null

        player?.setVideoSurface(null)
        player = null

        surface?.release()
        surface = null

        imageReader?.close()
        imageReader = null

        handlerThread?.quitSafely()
        handlerThread = null
        handler = null

        // Release this player's frame queue in Rust to prevent memory leaks
        // in long-running apps with many player create/destroy cycles
        if (nativeLibraryLoaded && playerId != 0L) {
            nativeReleasePlayer(playerId)
        }
    }

    // ========================================================================
    // JNI Native Methods
    // ========================================================================

    /**
     * Generates a unique player ID for multi-player isolation.
     *
     * Each ExoPlayerBridge instance gets a unique ID to ensure frames
     * from different players are not mixed in the rendering queue.
     *
     * @return Unique player ID (never 0)
     */
    private external fun nativeGeneratePlayerId(): Long

    /**
     * Releases the player's frame queue in Rust.
     *
     * Called during release() to clean up any pending frames and remove
     * the player's queue from memory. This prevents memory leaks in
     * long-running apps that create/destroy many players.
     *
     * @param playerId The player ID returned by nativeGeneratePlayerId
     */
    private external fun nativeReleasePlayer(playerId: Long)

    /**
     * Submits a HardwareBuffer to the Rust rendering pipeline.
     *
     * The HardwareBuffer is imported into Vulkan via
     * VK_ANDROID_external_memory_android_hardware_buffer for zero-copy rendering.
     *
     * @param buffer The HardwareBuffer containing the decoded video frame
     * @param timestampNs Presentation timestamp in nanoseconds
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @param playerId Unique player ID from nativeGeneratePlayerId
     * @param fenceFd Sync fence FD from producer (-1 if none/already signaled)
     */
    private external fun nativeSubmitHardwareBuffer(
        buffer: HardwareBuffer,
        timestampNs: Long,
        width: Int,
        height: Int,
        playerId: Long,
        fenceFd: Int
    )

    /**
     * Notifies Rust of a video size change.
     *
     * Called when ExoPlayer reports new video dimensions. Rust uses this to
     * recreate the NDK AImageReader with the correct size.
     *
     * @param nativeHandle The native handle from constructor
     * @param width New video width in pixels
     * @param height New video height in pixels
     */
    private external fun nativeOnVideoSizeChanged(nativeHandle: Long, width: Int, height: Int)
}
