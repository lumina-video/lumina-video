/**
 * lumina-video Demo Activity
 *
 * Main activity for the lumina-video demo application. This activity uses GameActivity
 * from the Android Game SDK to host the native Rust/egui application.
 *
 * # Architecture
 *
 * ```
 * MainActivity (Kotlin)
 *     |
 *     +--> LuminaVideo.init(this)  [one-time, stores applicationContext]
 *     |
 *     +--> GameActivity (androidx.games)
 *             |
 *             +--> liblumina_video_android.so (Rust + egui + wgpu)
 *                     |
 *                     +--> LuminaVideo.createPlayer(nativeHandle) [via JNI]
 *                             |
 *                             +--> ExoPlayerBridge + ExoPlayer [on HandlerThread]
 *                                     |
 *                                     +--> HardwareBuffer (zero-copy frames)
 * ```
 *
 * # Video Playback Flow
 *
 * 1. `LuminaVideo.init(this)` stores the application context
 * 2. Rust's `AndroidVideoDecoder::new()` calls `LuminaVideo.createPlayer()` via JNI
 * 3. `createPlayer()` creates ExoPlayer on a dedicated HandlerThread
 * 4. ExoPlayerBridge receives decoded frames as HardwareBuffers
 * 5. Bridge submits HardwareBuffers to Rust via nativeSubmitHardwareBuffer
 * 6. Rust imports HardwareBuffer into Vulkan via VK_ANDROID_external_memory
 * 7. egui displays the texture with zero CPU copies
 */
package com.luminavideo.demo

import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.androidgamesdk.GameActivity
import com.luminavideo.bridge.LuminaVideo

/**
 * Main activity hosting the lumina-video native application.
 *
 * Extends GameActivity which provides the native window surface
 * and input handling required by egui's android-game-activity feature.
 */
class MainActivity : GameActivity() {

    companion object {
        private const val TAG = "LuminaVideoDemo"

        init {
            try {
                System.loadLibrary("lumina_video_android")
                Log.i(TAG, "Loaded liblumina_video_android.so")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load liblumina_video_android.so: ${e.message}")
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // Initialize lumina-video BEFORE super.onCreate() — GameActivity's super
        // may trigger native startup which creates AndroidVideoDecoder, and that
        // calls LuminaVideo.createPlayer() via JNI. Init must be complete first.
        LuminaVideo.init(this)

        super.onCreate(savedInstanceState)

        Log.i(TAG, "onCreate: lumina-video demo initialized")

        // Enable immersive fullscreen mode
        enableImmersiveMode()

        // Keep screen on during video playback
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
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
}
