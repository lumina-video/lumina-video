/**
 * ExoPlayer Bridge for lumina-video zero-copy video rendering.
 *
 * This class bridges ExoPlayer's video output to lumina-video's Rust rendering pipeline
 * using Android's HardwareBuffer API for zero-copy GPU frame sharing.
 *
 * # Usage
 *
 * Created internally by [LuminaVideo.createPlayer]. Do not construct directly.
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

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.exoplayer.ExoPlayer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/**
 * Bridge between ExoPlayer and lumina-video's Rust zero-copy rendering.
 *
 * Extracts HardwareBuffer from decoded video frames via ImageReader
 * and submits them to Rust via JNI for GPU-accelerated display.
 *
 * Instances are created by [LuminaVideo.createPlayer] — do not construct directly.
 */
class ExoPlayerBridge internal constructor(private val nativeHandle: Long = 0L) {

    companion object {
        private const val TAG = "ExoPlayerBridge"

        /** Maximum number of images to hold in the ImageReader buffer */
        private const val MAX_IMAGES = 3

        /** Timeout for ExoPlayer creation on the HandlerThread */
        private const val INIT_TIMEOUT_SECONDS = 5L

        /** Whether the native library was successfully loaded */
        private var nativeLibraryLoaded = false

        init {
            try {
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

    // Dedicated Looper thread for ExoPlayer (ExoPlayer requires a Looper thread)
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
    private val playerId: Long = 0L

    // Current video dimensions (updated on video size change)
    private var videoWidth = 0
    private var videoHeight = 0

    // True when Surface is provided by native NDK ImageReader (via setVideoSurfaceFromNative)
    private var ndkMode = false

    /**
     * Creates ExoPlayer on a dedicated HandlerThread and sets up ImageReader.
     *
     * This method blocks the calling thread (via CountDownLatch) until ExoPlayer
     * is fully initialized, or until [INIT_TIMEOUT_SECONDS] elapses.
     *
     * Called by [LuminaVideo.createPlayer] from a Rust background thread.
     *
     * @param context Application context for ExoPlayer creation
     * @param builder Optional custom ExoPlayer.Builder. If null, default is used.
     * @return true if ExoPlayer was created successfully, false on timeout or error
     */
    internal fun initializeWithPlayer(context: Context, builder: ExoPlayer.Builder? = null): Boolean {
        if (isReleased.get()) {
            Log.w(TAG, "Cannot initialize - bridge is released")
            return false
        }

        if (!nativeLibraryLoaded) {
            Log.e(TAG, "Cannot initialize - native library not loaded")
            return false
        }

        // Create dedicated Looper thread for this player instance
        handlerThread = HandlerThread("LuminaVideoPlayer").apply { start() }
        handler = Handler(handlerThread!!.looper)

        val latch = CountDownLatch(1)
        var success = false

        handler!!.post {
            try {
                val exoPlayer = (builder ?: ExoPlayer.Builder(context))
                    .setLooper(handlerThread!!.looper)
                    .build()
                    .apply {
                        setVideoScalingMode(androidx.media3.common.C.VIDEO_SCALING_MODE_SCALE_TO_FIT)
                    }

                player = exoPlayer
                setupListenersAndSurface(exoPlayer)
                success = true
                Log.i(TAG, "ExoPlayer created on ${Thread.currentThread().name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to create ExoPlayer: ${e.message}", e)
            }
            latch.countDown()
        }

        val completed = latch.await(INIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        if (!completed) {
            Log.e(TAG, "ExoPlayer creation timed out after ${INIT_TIMEOUT_SECONDS}s")
            return false
        }

        return success
    }

    /**
     * Sets up video size listener and initial ImageReader surface on the player.
     *
     * Must be called on the player's Looper thread.
     */
    private fun setupListenersAndSurface(exoPlayer: ExoPlayer) {
        // Listen for video size changes
        playerListener = object : Player.Listener {
            override fun onVideoSizeChanged(videoSize: VideoSize) {
                Log.d(TAG, "onVideoSizeChanged: ${videoSize.width}x${videoSize.height}")
                if (videoSize.width > 0 && videoSize.height > 0) {
                    videoWidth = videoSize.width
                    videoHeight = videoSize.height
                    if (!ndkMode && imageReader != null) {
                        setupImageReader(videoSize.width, videoSize.height)
                    }
                    if (nativeHandle != 0L) {
                        nativeOnVideoSizeChanged(nativeHandle, videoSize.width, videoSize.height)
                    }
                }
            }
        }
        exoPlayer.addListener(playerListener!!)

        // Create initial ImageReader (unless NDK mode is active)
        if (!ndkMode && imageReader == null && surface == null) {
            Log.i(TAG, "Creating initial ImageReader with default 1920x1080 size")
            setupImageReader(1920, 1080)
        }
    }

    /**
     * Loads and plays a video from the given URL.
     *
     * Called by Rust code to start video playback. Posts to the player's
     * HandlerThread to satisfy ExoPlayer's threading requirements.
     */
    fun play(url: String) {
        handler?.post {
            val p = player
            if (p == null) {
                Log.w(TAG, "play() called but player not initialized")
                return@post
            }
            Log.d(TAG, "play($url)")
            val mediaItem = MediaItem.fromUri(url)
            p.setMediaItem(mediaItem)
            p.prepare()
            p.play()
        } ?: Log.w(TAG, "play() called before initializeWithPlayer()")
    }

    /**
     * Pauses video playback.
     */
    fun pause() {
        handler?.post { player?.pause() }
    }

    /**
     * Resumes video playback.
     */
    fun resume() {
        handler?.post { player?.play() }
    }

    /**
     * Seeks to the specified position.
     *
     * @param positionMs Position in milliseconds
     */
    fun seek(positionMs: Long) {
        handler?.post { player?.seekTo(positionMs) }
    }

    /**
     * Sets the playback volume.
     *
     * @param volume Volume level (0.0 to 1.0)
     */
    fun setVolume(volume: Float) {
        handler?.post { player?.let { it.volume = volume.coerceIn(0f, 1f) } }
    }

    /**
     * Sets the muted state.
     *
     * @param muted True to mute, false to unmute
     */
    fun setMuted(muted: Boolean) {
        handler?.post {
            player?.let { p ->
                if (muted) {
                    volumeBeforeMute = p.volume
                    p.volume = 0f
                } else {
                    p.volume = volumeBeforeMute
                }
            }
        }
    }

    /**
     * Gets the current playback position in milliseconds.
     *
     * Synchronous getter — ExoPlayer position getters are safe for stale reads
     * from any thread.
     *
     * @return Current position in ms, or 0 if not playing
     */
    fun getCurrentPosition(): Long {
        return player?.currentPosition ?: 0L
    }

    /**
     * Gets the total duration of the video in milliseconds.
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
        Log.d(TAG, "extractCurrentFrame() called - CPU fallback not implemented")
        return null
    }

    /**
     * Sets the video surface from native code (NDK ImageReader mode).
     *
     * Called by Rust when using NDK AImageReader for zero-copy with sync fence support.
     *
     * @param surface The Surface from ANativeWindow_toSurface (backed by NDK AImageReader)
     */
    fun setVideoSurfaceFromNative(surface: Surface) {
        Log.d(TAG, "setVideoSurfaceFromNative called - switching to NDK ImageReader mode")
        ndkMode = true

        imageReader?.close()
        imageReader = null
        this.surface?.release()
        this.surface = surface

        handler?.post { player?.setVideoSurface(surface) }

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
            return
        }

        imageReader?.close()
        imageReader = null
        surface?.release()
        surface = null

        Log.i(TAG, "Setting up ImageReader: ${width}x${height}")

        try {
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
            player?.setVideoSurface(newSurface)

            imageReader = newReader
            surface = newSurface
            videoWidth = width
            videoHeight = height

            Log.i(TAG, "ImageReader created and attached to player")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create ImageReader: ${e.message}")
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
                val fenceFd = -1

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
            hardwareBuffer?.close()
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
     * Safe to call multiple times (idempotent).
     */
    fun release() {
        if (isReleased.getAndSet(true)) {
            return
        }

        Log.i(TAG, "Releasing bridge (submitted $frameCount frames)")

        // Post cleanup to the player's thread
        handler?.post {
            playerListener?.let { listener ->
                player?.removeListener(listener)
            }
            playerListener = null

            player?.setVideoSurface(null)
            player?.release()
            player = null

            surface?.release()
            surface = null

            imageReader?.close()
            imageReader = null
        }

        // Give the handler a moment to process the cleanup post
        handlerThread?.quitSafely()
        handlerThread = null
        handler = null

        if (nativeLibraryLoaded && playerId != 0L) {
            nativeReleasePlayer(playerId)
        }
    }

    // ========================================================================
    // JNI Native Methods
    // ========================================================================

    private external fun nativeGeneratePlayerId(): Long

    private external fun nativeReleasePlayer(playerId: Long)

    private external fun nativeSubmitHardwareBuffer(
        buffer: HardwareBuffer,
        timestampNs: Long,
        width: Int,
        height: Int,
        playerId: Long,
        fenceFd: Int
    )

    private external fun nativeOnVideoSizeChanged(nativeHandle: Long, width: Int, height: Int)
}
