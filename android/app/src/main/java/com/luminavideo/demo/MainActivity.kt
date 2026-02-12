/**
 * lumina-video Demo Activity
 *
 * Main activity for the lumina-video demo application. This activity uses GameActivity
 * from the Android Game SDK to host the native Rust/egui application.
 *
 * The GameActivity pattern is required by egui's android-game-activity feature,
 * which expects a NativeActivity-style interface for window and input handling.
 *
 * # Architecture
 *
 * ```
 * MainActivity (Kotlin)
 *     |
 *     +--> GameActivity (androidx.games)
 *             |
 *             +--> liblumina_video_android.so (Rust + egui + wgpu)
 *                     |
 *                     +--> ExoPlayerBridge (via JNI callback)
 *                             |
 *                             +--> ExoPlayer (video decoding)
 *                                     |
 *                                     +--> HardwareBuffer (zero-copy frames)
 * ```
 *
 * # Video Playback Flow
 *
 * 1. Rust calls into Kotlin to initialize ExoPlayer via JNI
 * 2. ExoPlayerBridge receives decoded frames as HardwareBuffers
 * 3. Bridge submits HardwareBuffers to Rust via nativeSubmitHardwareBuffer
 * 4. Rust imports HardwareBuffer into Vulkan via VK_ANDROID_external_memory
 * 5. egui displays the texture with zero CPU copies
 */
package com.luminavideo.demo

import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.androidgamesdk.GameActivity
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.common.MediaItem
import com.luminavideo.bridge.ExoPlayerBridge

/**
 * Main activity hosting the lumina-video native application.
 *
 * Extends GameActivity which provides the native window surface
 * and input handling required by egui's android-game-activity feature.
 */
class MainActivity : GameActivity() {

    companion object {
        private const val TAG = "LuminaVideoDemo"

        // Sample video URL for testing
        // Big Buck Bunny - Creative Commons licensed, H.264 720p
        private const val SAMPLE_VIDEO_URL =
            "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

        init {
            // Load the native library
            // This must happen before any native methods are called
            try {
                // Load the Android native library (built from lumina-video-android crate)
                // Contains android_main for GameActivity + JNI functions for ExoPlayerBridge
                System.loadLibrary("lumina_video_android")
                Log.i(TAG, "Loaded liblumina_video_android.so")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load liblumina_video_android.so: ${e.message}")
            }
        }
    }

    // ExoPlayer instance for video decoding
    private var exoPlayer: ExoPlayer? = null

    // Bridge for zero-copy frame transfer to Rust
    private var bridge: ExoPlayerBridge? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.i(TAG, "onCreate: Initializing lumina-video demo")

        // Enable immersive fullscreen mode
        enableImmersiveMode()

        // Keep screen on during video playback
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Initialize video playback
        initializePlayer()
    }

    /**
     * Enables immersive fullscreen mode, hiding system bars.
     */
    private fun enableImmersiveMode() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.hide(WindowInsetsCompat.Type.systemBars())
        controller.systemBarsBehavior =
            WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
    }

    /**
     * Initializes ExoPlayer and the bridge for zero-copy rendering.
     *
     * This sets up the video playback pipeline:
     * 1. Create ExoPlayer instance
     * 2. Create ExoPlayerBridge to capture HardwareBuffers
     * 3. Attach bridge to player
     * 4. Set media source and start playback
     */
    private fun initializePlayer() {
        Log.i(TAG, "Initializing ExoPlayer")

        // Create ExoPlayer
        exoPlayer = ExoPlayer.Builder(this).build().apply {
            // Enable video output for HardwareBuffer capture
            setVideoScalingMode(androidx.media3.common.C.VIDEO_SCALING_MODE_SCALE_TO_FIT)
        }

        // Create bridge for zero-copy frame transfer
        bridge = ExoPlayerBridge().apply {
            attachToPlayer(exoPlayer!!)
        }

        // Get video URL from intent or use sample
        val videoUrl = intent?.data?.toString() ?: SAMPLE_VIDEO_URL

        Log.i(TAG, "Playing video: $videoUrl")

        // Set up media item and start playback
        val mediaItem = MediaItem.fromUri(videoUrl)
        exoPlayer?.apply {
            setMediaItem(mediaItem)
            prepare()
            playWhenReady = true
        }
    }

    override fun onStart() {
        super.onStart()
        Log.d(TAG, "onStart")
        exoPlayer?.playWhenReady = true
    }

    override fun onStop() {
        super.onStop()
        Log.d(TAG, "onStop")
        exoPlayer?.playWhenReady = false
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy: Cleaning up resources")

        // Release bridge first (stops frame capture)
        bridge?.release()
        bridge = null

        // Then release player
        exoPlayer?.release()
        exoPlayer = null

        super.onDestroy()
    }

    /**
     * Called from native code to get the current player bridge.
     *
     * This allows Rust to query the bridge for frame data or control playback.
     */
    @Suppress("unused") // Called via JNI
    fun getBridge(): ExoPlayerBridge? = bridge

    /**
     * Called from native code to get the player instance.
     *
     * Enables native control of playback (pause, seek, etc.)
     */
    @Suppress("unused") // Called via JNI
    fun getPlayer(): ExoPlayer? = exoPlayer

    /**
     * Called from native code to play a new video URL.
     *
     * @param url The video URL to play
     */
    @Suppress("unused") // Called via JNI
    fun playVideo(url: String) {
        Log.i(TAG, "playVideo called from native: $url")
        runOnUiThread {
            exoPlayer?.apply {
                setMediaItem(MediaItem.fromUri(url))
                prepare()
                playWhenReady = true
            }
        }
    }

    /**
     * Called from native code to pause playback.
     */
    @Suppress("unused") // Called via JNI
    fun pauseVideo() {
        Log.d(TAG, "pauseVideo called from native")
        runOnUiThread {
            exoPlayer?.playWhenReady = false
        }
    }

    /**
     * Called from native code to resume playback.
     */
    @Suppress("unused") // Called via JNI
    fun resumeVideo() {
        Log.d(TAG, "resumeVideo called from native")
        runOnUiThread {
            exoPlayer?.playWhenReady = true
        }
    }

    /**
     * Called from native code to seek to a position.
     *
     * @param positionMs Position in milliseconds
     */
    @Suppress("unused") // Called via JNI
    fun seekTo(positionMs: Long) {
        Log.d(TAG, "seekTo called from native: $positionMs ms")
        runOnUiThread {
            exoPlayer?.seekTo(positionMs)
        }
    }

    /**
     * Called from native code to get current playback position.
     *
     * @return Current position in milliseconds
     */
    @Suppress("unused") // Called via JNI
    fun getCurrentPosition(): Long = exoPlayer?.currentPosition ?: 0L

    /**
     * Called from native code to get video duration.
     *
     * @return Duration in milliseconds, or -1 if unknown
     */
    @Suppress("unused") // Called via JNI
    fun getDuration(): Long = exoPlayer?.duration ?: -1L

    /**
     * Called from native code to check if video is playing.
     *
     * @return true if playing, false otherwise
     */
    @Suppress("unused") // Called via JNI
    fun isPlaying(): Boolean = exoPlayer?.isPlaying ?: false
}
