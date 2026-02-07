/**
 * MediaCodec Bridge for lumina-video MoQ zero-copy video rendering.
 *
 * This class bridges MediaCodec directly (not via ExoPlayer) to lumina-video's Rust
 * rendering pipeline using Android's HardwareBuffer API for zero-copy GPU frame sharing.
 *
 * Unlike ExoPlayerBridge which handles URL-based playback internally, MoqMediaCodecBridge
 * receives raw NAL units from Rust and decodes them with MediaCodec, providing lower
 * latency for MoQ live streaming.
 *
 * # Architecture
 *
 * ```
 * Rust (MoQ NAL units) -> submitNalUnit() -> MediaCodec
 *                                         -> ImageReader (GPU_SAMPLED_IMAGE)
 *                                         -> HardwareBuffer
 *                                         -> nativeSubmitHardwareBuffer()
 *                                         -> Vulkan/wgpu rendering
 * ```
 *
 * # Requirements
 *
 * - Android API 29+ (ImageReader.newInstance 5-arg overload with usage flags)
 * - lumina-video native library loaded
 *
 * # Zero-Copy Pipeline
 *
 * 1. Rust MoQ worker receives NAL units from hang crate
 * 2. Rust calls submitNalUnit() to queue NAL data in MediaCodec input buffers
 * 3. MediaCodec decodes to ImageReader configured with GPU_SAMPLED_IMAGE usage
 * 4. ImageReader.OnImageAvailableListener extracts HardwareBuffer
 * 5. HardwareBuffer submitted to Rust via nativeSubmitHardwareBuffer()
 * 6. Rust imports via VK_ANDROID_external_memory_android_hardware_buffer
 * 7. VkSamplerYcbcrConversion handles YUV to RGB conversion on GPU
 */
package com.luminavideo.bridge

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.HardwareBuffer
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaFormat
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.LinkedBlockingDeque

/**
 * Bridge between Rust MoQ decoder and Android MediaCodec for zero-copy rendering.
 *
 * @param context Android application context
 * @param mimeType MediaCodec MIME type ("video/avc" for H.264, "video/hevc" for H.265)
 * @param width Video width in pixels
 * @param height Video height in pixels
 * @param playerId Unique player ID for frame queue routing
 */
class MoqMediaCodecBridge(
    private val context: Context,
    private val mimeType: String,
    width: Int,
    height: Int,
    private val playerId: Long
) {
    companion object {
        private const val TAG = "MoqMediaCodecBridge"

        /** Maximum number of images in the ImageReader buffer */
        private const val MAX_IMAGES = 5

        /** Input buffer timeout in microseconds (10ms for low latency) */
        private const val INPUT_TIMEOUT_US = 10_000L

        /** Output buffer timeout in microseconds (10ms for low latency) */
        private const val OUTPUT_TIMEOUT_US = 10_000L

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

    // MediaCodec for H.264/H.265 decoding
    private var codec: MediaCodec? = null

    // ImageReader for extracting HardwareBuffer from decoded frames
    private var imageReader: ImageReader? = null

    // Surface from ImageReader (passed to MediaCodec as output)
    private var outputSurface: Surface? = null

    // Background thread for codec and ImageReader callbacks
    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null

    // Codec input thread for async NAL submission
    private var inputThread: Thread? = null
    private val nalQueue = LinkedBlockingDeque<NalUnit>(100)

    // Thread-safe video dimensions (written by input thread in processOutputBuffers)
    private val width = AtomicInteger(width)
    private val height = AtomicInteger(height)

    // State tracking
    private val isReleased = AtomicBoolean(false)
    private val isRunning = AtomicBoolean(false)
    private val frameCount = AtomicInteger(0)
    private val nalCount = AtomicInteger(0)

    // Codec configuration state
    private var codecConfigured = false

    /**
     * NAL unit data with timestamp for queue.
     */
    private data class NalUnit(val data: ByteArray, val timestampUs: Long)

    /**
     * Starts the decoder.
     *
     * Creates MediaCodec, ImageReader, and begins async processing.
     * Must be called before submitNalUnit().
     */
    fun start() {
        if (isReleased.get() || isRunning.get()) {
            Log.w(TAG, "start() called but already running or released")
            return
        }

        if (!nativeLibraryLoaded) {
            Log.e(TAG, "Cannot start MoqMediaCodecBridge (playerId=$playerId): native library not loaded")
            return
        }

        Log.i(TAG, "Starting MoqMediaCodecBridge: $mimeType ${width.get()}x${height.get()} playerId=$playerId")

        // Start background thread
        handlerThread = HandlerThread("MoqMediaCodec-$playerId").apply { start() }
        val looper = handlerThread?.looper
        if (looper == null) {
            handlerThread?.quitSafely()
            handlerThread = null
            nativeOnError(playerId, "Failed to start MoQ decoder thread")
            return
        }
        handler = Handler(looper)

        try {
            // Create ImageReader for HardwareBuffer extraction with GPU usage
            // 5-arg ImageReader.newInstance requires API 29+
            imageReader = ImageReader.newInstance(
                width.get(),
                height.get(),
                ImageFormat.PRIVATE,  // Required for hardware decoder output
                MAX_IMAGES,
                HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE
            ).apply {
                setOnImageAvailableListener(
                    { reader -> onImageAvailable(reader) },
                    handler
                )
            }

            outputSurface = imageReader!!.surface

            // Create and configure MediaCodec
            codec = MediaCodec.createDecoderByType(mimeType).apply {
                val format = MediaFormat.createVideoFormat(mimeType, width.get(), height.get()).apply {
                    // Request low latency decoding (KEY_LOW_LATENCY requires API 30+)
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                        setInteger(MediaFormat.KEY_LOW_LATENCY, 1)
                    }
                    // Use hardware decoder
                    setInteger(MediaFormat.KEY_PRIORITY, 0) // Realtime priority
                }

                configure(format, outputSurface, null, 0)
            }

            // Start codec
            codec!!.start()
            codecConfigured = true
            isRunning.set(true)

            // Start input thread for async NAL processing
            startInputThread()

            Log.i(TAG, "MediaCodec started successfully")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to start MediaCodec: ${e.message}", e)
            nativeOnError(playerId, "Failed to start MediaCodec: ${e.message}")
            release()
        }
    }

    /**
     * Submits a NAL unit for decoding.
     *
     * Called by Rust when a NAL unit is received from MoQ.
     * NAL units are queued and processed asynchronously by the input thread.
     *
     * @param data NAL unit data (Annex B or raw NAL format)
     * @param timestampUs Presentation timestamp in microseconds
     */
    fun submitNalUnit(data: ByteArray, timestampUs: Long) {
        if (isReleased.get() || !isRunning.get()) {
            return
        }

        // Queue the NAL unit for async processing
        val nal = NalUnit(data.copyOf(), timestampUs)
        if (!nalQueue.offer(nal)) {
            Log.w(TAG, "NAL queue full, dropping frame")
        } else {
            nalCount.incrementAndGet()
        }
    }

    /**
     * Starts the input thread that processes NAL units from the queue.
     */
    private fun startInputThread() {
        inputThread = Thread({
            while (isRunning.get() && !Thread.interrupted()) {
                try {
                    // Take from queue with timeout to allow shutdown
                    val nal = nalQueue.poll(50, java.util.concurrent.TimeUnit.MILLISECONDS)
                    if (nal != null) {
                        processNalUnit(nal)
                    }

                    // Process any available output buffers
                    processOutputBuffers()

                } catch (e: InterruptedException) {
                    break
                } catch (e: Exception) {
                    Log.e(TAG, "Input thread error: ${e.message}", e)
                }
            }
            Log.d(TAG, "Input thread exiting")
        }, "MoqCodecInput-$playerId").apply {
            start()
        }
    }

    /**
     * Processes a single NAL unit by queuing it in MediaCodec's input buffer.
     */
    private fun processNalUnit(nal: NalUnit) {
        val codec = this.codec ?: return

        try {
            // Get an input buffer
            val inputIndex = codec.dequeueInputBuffer(INPUT_TIMEOUT_US)
            if (inputIndex < 0) {
                Log.v(TAG, "No input buffer available, retrying")
                // Push back to front to preserve decode order
                nalQueue.offerFirst(nal)
                return
            }

            // Copy NAL data to input buffer
            val inputBuffer = codec.getInputBuffer(inputIndex)
            if (inputBuffer == null) {
                // Return the slot to avoid exhausting MediaCodec buffers
                codec.queueInputBuffer(inputIndex, 0, 0, 0L, 0)
                return
            }
            inputBuffer.clear()

            if (nal.data.size > inputBuffer.capacity()) {
                Log.w(TAG, "Dropping oversized NAL: ${nal.data.size} > ${inputBuffer.capacity()}")
                // Return the slot to avoid exhausting MediaCodec buffers
                codec.queueInputBuffer(inputIndex, 0, 0, 0L, 0)
                return
            }

            inputBuffer.put(nal.data)

            // Queue the buffer for decoding
            codec.queueInputBuffer(
                inputIndex,
                0,
                nal.data.size,
                nal.timestampUs,
                0 // flags
            )

        } catch (e: Exception) {
            Log.e(TAG, "Error processing NAL: ${e.message}")
        }
    }

    /**
     * Processes any available output buffers from MediaCodec.
     */
    private fun processOutputBuffers() {
        val codec = this.codec ?: return

        try {
            val bufferInfo = MediaCodec.BufferInfo()
            val outputIndex = codec.dequeueOutputBuffer(bufferInfo, OUTPUT_TIMEOUT_US)

            when {
                outputIndex >= 0 -> {
                    // Frame decoded - release to surface (ImageReader will capture it)
                    codec.releaseOutputBuffer(outputIndex, true) // render = true
                }
                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val newFormat = codec.outputFormat
                    Log.i(TAG, "Output format changed: $newFormat")
                    val newWidth = newFormat.getInteger(MediaFormat.KEY_WIDTH)
                    val newHeight = newFormat.getInteger(MediaFormat.KEY_HEIGHT)
                    if (newWidth != width.get() || newHeight != height.get()) {
                        width.set(newWidth)
                        height.set(newHeight)
                        nativeOnVideoSizeChanged(playerId, newWidth, newHeight)
                    }
                }
                outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    // No output available yet
                }
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing output: ${e.message}")
        }
    }

    /**
     * Called when a decoded frame is available in the ImageReader.
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
                // Get sync fence FD if available (API 33+)
                // For now, pass -1 (no fence) and rely on Vulkan queue ownership transfer
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
            hardwareBuffer?.close()
            image.close()
        }
    }

    /**
     * Gets the number of frames submitted to Rust.
     */
    fun getFrameCount(): Int = frameCount.get()

    /**
     * Gets the number of NAL units received.
     */
    fun getNalCount(): Int = nalCount.get()

    /**
     * Releases all resources held by this bridge.
     */
    fun release() {
        if (isReleased.getAndSet(true)) {
            return // Already released
        }

        Log.i(TAG, "Releasing MoqMediaCodecBridge (frames=$frameCount, nals=$nalCount)")

        isRunning.set(false)

        // Stop input thread
        inputThread?.interrupt()
        inputThread?.join(1000)
        inputThread = null

        // Clear queue
        nalQueue.clear()

        // Stop and release codec
        try {
            codec?.stop()
            codec?.release()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing codec: ${e.message}")
        }
        codec = null

        // Stop handler thread first to prevent onImageAvailable racing with close
        handlerThread?.quitSafely()
        handlerThread = null
        handler = null

        // Now safe to close ImageReader and Surface
        imageReader?.close()
        imageReader = null

        outputSurface?.release()
        outputSurface = null

        Log.i(TAG, "MoqMediaCodecBridge released")
    }

    // ========================================================================
    // JNI Native Methods
    // ========================================================================

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
     * @param playerId Unique player ID for frame queue routing
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
     * Notifies Rust of an error.
     *
     * @param playerId The player ID
     * @param message Error message
     */
    private external fun nativeOnError(playerId: Long, message: String)

    /**
     * Notifies Rust of a video size change.
     *
     * @param playerId The player ID
     * @param width New width in pixels
     * @param height New height in pixels
     */
    private external fun nativeOnVideoSizeChanged(playerId: Long, width: Int, height: Int)
}
