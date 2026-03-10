use super::*;
use crate::media::android_video::{
    generate_player_id, try_receive_hardware_buffer_for_player, AndroidVideoFrame,
};
use crate::media::video::AndroidGpuSurface;
use jni::objects::{GlobalRef, JClass, JObject, JValue};
use jni::sys::{jint, jlong};
use jni::JNIEnv;
use std::collections::VecDeque;

/// Android MoQ decoder using MediaCodec with zero-copy HardwareBuffer output.
///
/// This decoder receives NAL units from MoQ and decodes them using Android's
/// MediaCodec API directly, outputting to an ImageReader configured with
/// `AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE` for zero-copy Vulkan import.
///
/// # Zero-Copy Pipeline
///
/// ```text
/// MoQ NAL units -> MediaCodec -> ImageReader -> HardwareBuffer
///              -> JNI nativeSubmitHardwareBuffer()
///              -> import_ahardwarebuffer_yuv_zero_copy()
///              -> wgpu::Texture (GPU-side YUV to RGB)
/// ```
pub struct MoqAndroidDecoder {
    /// Parsed MoQ URL
    #[allow(dead_code)]
    url: MoqUrl,
    /// Configuration
    #[allow(dead_code)]
    config: MoqDecoderConfig,
    /// Shared state with async worker
    shared: Arc<MoqSharedState>,
    /// Receiver for encoded NAL units from MoQ worker
    nal_rx: Receiver<MoqVideoFrame>,
    /// Owned tokio runtime (created if none exists)
    _owned_runtime: Option<tokio::runtime::Runtime>,
    /// Tokio runtime handle
    _runtime: Handle,
    /// Whether audio is muted
    audio_muted: bool,
    /// Audio volume (0.0 to 1.0)
    audio_volume: f32,
    /// JNI reference to MoqMediaCodecBridge
    bridge: Option<GlobalRef>,
    /// Unique player ID for frame queue isolation
    player_id: u64,
    /// Pending decoded frames from HardwareBuffer queue
    pending_frames: VecDeque<AndroidVideoFrame>,
    /// Whether MediaCodec has been configured
    codec_configured: bool,
    /// Codec type detected from catalog
    codec_type: Option<CodecType>,
    /// Locally cached metadata (safe copy from shared state)
    cached_metadata: VideoMetadata,
}

/// Supported video codec types for MediaCodec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecType {
    /// H.264/AVC
    H264,
    /// H.265/HEVC
    H265,
}

impl CodecType {
    /// Returns the MediaCodec MIME type string.
    pub fn mime_type(&self) -> &'static str {
        match self {
            CodecType::H264 => "video/avc",
            CodecType::H265 => "video/hevc",
        }
    }

    /// Parses codec type from hang catalog codec string.
    pub fn from_catalog_codec(codec: &str) -> Option<Self> {
        let lower = codec.to_lowercase();
        if lower.contains("avc") || lower.contains("h264") || lower.contains("h.264") {
            Some(CodecType::H264)
        } else if lower.contains("hevc")
            || lower.contains("hvc1")
            || lower.contains("h265")
            || lower.contains("h.265")
        {
            Some(CodecType::H265)
        } else {
            None
        }
    }
}

impl MoqAndroidDecoder {
    /// Creates a new Android MoQ decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, MoqDecoderConfig::default())
    }

    /// Creates a new Android MoQ decoder with explicit configuration.
    pub fn new_with_config(
        url: &str,
        mut config: MoqDecoderConfig,
    ) -> Result<Self, VideoError> {
        let moq_url = MoqUrl::parse(url).map_err(|e| VideoError::OpenFailed(e.to_string()))?;
        config.apply_localhost_tls_bypass(&moq_url);

        // Get existing runtime handle or create a new runtime
        let (owned_runtime, runtime) = match Handle::try_current() {
            Ok(handle) => (None, handle),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("moq-android-runtime")
                    .build()
                    .map_err(|e| {
                        VideoError::OpenFailed(format!("Failed to create tokio runtime: {e}"))
                    })?;
                let handle = rt.handle().clone();
                (Some(rt), handle)
            }
        };

        let shared = Arc::new(MoqSharedState::new());
        let (nal_tx, nal_rx) = async_channel::bounded(60); // Larger buffer for NAL units

        // Generate unique player ID for frame queue isolation
        let player_id = generate_player_id();
        tracing::info!("MoqAndroidDecoder: Created with player_id={}", player_id);

        // Spawn MoQ worker (reuses same logic as MoqDecoder)
        let worker_shared = shared.clone();
        let worker_url = moq_url.clone();
        let worker_config = config.clone();

        runtime.spawn(async move {
            if let Err(e) =
                Self::run_moq_worker(worker_shared.clone(), worker_url, worker_config, nal_tx)
                    .await
            {
                worker_shared.set_error(format!("MoQ worker error: {e}"));
            }
        });

        Ok(Self {
            url: moq_url,
            config,
            shared,
            nal_rx,
            _owned_runtime: owned_runtime,
            _runtime: runtime,
            audio_muted: false,
            audio_volume: config.initial_volume,
            bridge: None,
            player_id,
            pending_frames: VecDeque::new(),
            codec_configured: false,
            codec_type: None,
            cached_metadata: VideoMetadata {
                codec: String::new(),
                width: 0,
                height: 0,
                frame_rate: 0.0,
                duration: None,
                pixel_aspect_ratio: 1.0,
                start_time: None,
            },
        })
    }

    /// Async worker that handles MoQ connection and NAL unit receipt.
    async fn run_moq_worker(
        shared: Arc<MoqSharedState>,
        url: MoqUrl,
        config: MoqDecoderConfig,
        nal_tx: Sender<MoqVideoFrame>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        crate::media::moq::worker::run_moq_worker(shared, url, config, nal_tx, "Android").await
    }

    /// Initializes the MediaCodec decoder via JNI.
    ///
    /// Creates the MoqMediaCodecBridge Java object which:
    /// 1. Creates MediaCodec for H.264/H.265
    /// 2. Configures ImageReader with GPU_SAMPLED_IMAGE usage
    /// 3. Sets up the output surface for zero-copy frame extraction
    fn initialize_codec(&mut self) -> Result<(), VideoError> {
        if self.codec_configured {
            return Ok(());
        }

        let metadata = self.shared.metadata.lock();
        let width = metadata.width;
        let height = metadata.height;
        let codec_str = metadata.codec.clone();
        drop(metadata);

        // Determine codec type from catalog metadata
        let codec_type = CodecType::from_catalog_codec(&codec_str).ok_or_else(|| {
            VideoError::UnsupportedFormat(format!(
                "Unsupported codec for Android MediaCodec: {}",
                codec_str
            ))
        })?;
        self.codec_type = Some(codec_type);

        // Get JVM and create bridge via JNI
        let vm = Self::get_jvm()?;
        let mut env = vm.attach_current_thread().map_err(|e| {
            VideoError::DecoderInit(format!("Failed to attach JNI thread: {}", e))
        })?;

        // Get Android context
        let context =
            unsafe { JObject::from_raw(ndk_context::android_context().context().cast()) };

        // Load MoqMediaCodecBridge class via class loader
        let class_loader = env
            .call_method(&context, "getClassLoader", "()Ljava/lang/ClassLoader;", &[])
            .map_err(|e| VideoError::DecoderInit(format!("Failed to get class loader: {}", e)))?
            .l()
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to get class loader object: {}", e))
            })?;

        let class_name = env
            .new_string("com.luminavideo.bridge.MoqMediaCodecBridge")
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to create class name string: {}", e))
            })?;

        let bridge_class = env
            .call_method(
                &class_loader,
                "loadClass",
                "(Ljava/lang/String;)Ljava/lang/Class;",
                &[JValue::Object(&class_name)],
            )
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to load MoqMediaCodecBridge: {}", e))
            })?
            .l()
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to get bridge class: {}", e))
            })?;

        let bridge_class = jni::objects::JClass::from(bridge_class);

        // Create MIME type string for MediaCodec
        let mime_type = env.new_string(codec_type.mime_type()).map_err(|e| {
            VideoError::DecoderInit(format!("Failed to create MIME type string: {}", e))
        })?;

        // Create bridge: MoqMediaCodecBridge(Context, String mimeType, int width, int height, long playerId)
        let bridge = env
            .new_object(
                bridge_class,
                "(Landroid/content/Context;Ljava/lang/String;IIJ)V",
                &[
                    JValue::Object(&context),
                    JValue::Object(&mime_type),
                    JValue::Int(width as i32),
                    JValue::Int(height as i32),
                    JValue::Long(self.player_id as i64),
                ],
            )
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to create MoqMediaCodecBridge: {}", e))
            })?;

        let bridge_ref = env.new_global_ref(bridge).map_err(|e| {
            VideoError::DecoderInit(format!("Failed to create global ref: {}", e))
        })?;

        // Start the decoder
        env.call_method(&bridge_ref, "start", "()V", &[])
            .map_err(|e| {
                VideoError::DecoderInit(format!("Failed to start MediaCodec: {}", e))
            })?;

        self.bridge = Some(bridge_ref);
        self.codec_configured = true;

        tracing::info!(
            "MoqAndroidDecoder: MediaCodec initialized for {} ({}x{})",
            codec_type.mime_type(),
            width,
            height
        );

        Ok(())
    }

    /// Submits a NAL unit to MediaCodec for decoding.
    ///
    /// The NAL unit is passed to the Java bridge which queues it in
    /// MediaCodec's input buffer. Decoded frames appear asynchronously
    /// in the HardwareBuffer queue.
    fn submit_nal_unit(&self, nal_data: &[u8], timestamp_us: u64) -> Result<(), VideoError> {
        let Some(bridge) = &self.bridge else {
            return Err(VideoError::DecodeFailed(
                "MediaCodec not initialized".to_string(),
            ));
        };

        let vm = Self::get_jvm()?;
        let mut env = vm.attach_current_thread().map_err(|e| {
            VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e))
        })?;

        // Create byte array from NAL data
        let byte_array = env.new_byte_array(nal_data.len() as i32).map_err(|e| {
            VideoError::DecodeFailed(format!("Failed to create byte array: {}", e))
        })?;

        // Convert u8 slice to i8 slice for JNI
        let nal_data_i8: Vec<i8> = nal_data.iter().map(|&b| b as i8).collect();
        env.set_byte_array_region(&byte_array, 0, &nal_data_i8)
            .map_err(|e| {
                VideoError::DecodeFailed(format!("Failed to set byte array data: {}", e))
            })?;

        // Submit to MediaCodec: submitNalUnit(byte[] data, long timestampUs)
        env.call_method(
            bridge,
            "submitNalUnit",
            "([BJ)V",
            &[
                JValue::Object(&byte_array),
                JValue::Long(timestamp_us as i64),
            ],
        )
        .map_err(|e| VideoError::DecodeFailed(format!("Failed to submit NAL unit: {}", e)))?;

        Ok(())
    }

    /// Polls for decoded frames from the HardwareBuffer queue.
    ///
    /// MediaCodec outputs to ImageReader, which extracts HardwareBuffers
    /// and submits them via JNI to the per-player queue.
    fn poll_decoded_frames(&mut self) {
        while let Some(frame) = try_receive_hardware_buffer_for_player(self.player_id) {
            self.pending_frames.push_back(frame);
        }
    }

    /// Converts an AndroidVideoFrame to a VideoFrame with zero-copy GPU surface.
    ///
    /// The HardwareBuffer is wrapped in an AndroidGpuSurface which can be
    /// imported into Vulkan via import_ahardwarebuffer_yuv_zero_copy().
    fn convert_to_video_frame(&self, frame: AndroidVideoFrame) -> VideoFrame {
        let pts = Duration::from_nanos(frame.timestamp_ns as u64);

        // Create owner to track HardwareBuffer lifetime
        struct HardwareBufferOwner {
            #[allow(dead_code)]
            buffer: *mut std::ffi::c_void,
        }

        // SAFETY: AHardwareBuffer is thread-safe per Android NDK docs
        unsafe impl Send for HardwareBufferOwner {}
        unsafe impl Sync for HardwareBufferOwner {}

        impl Drop for HardwareBufferOwner {
            fn drop(&mut self) {
                // Don't release here - AndroidVideoFrame::drop handles it
                // This owner is just for lifetime tracking
            }
        }

        let owner = Arc::new(HardwareBufferOwner {
            buffer: frame.buffer,
        });

        // Determine pixel format from AHardwareBuffer format
        let pixel_format =
            if crate::media::android_video::is_yuv_hardware_buffer_format(frame.format) {
                PixelFormat::Nv12 // Most common YUV format from MediaCodec
            } else {
                PixelFormat::Rgba
            };

        let surface = unsafe {
            AndroidGpuSurface::new(
                frame.buffer,
                frame.width,
                frame.height,
                pixel_format,
                None, // No CPU fallback for zero-copy frames
                owner,
            )
        };

        // Transfer ownership - prevent AndroidVideoFrame from releasing the buffer
        // since the AndroidGpuSurface now owns the reference
        std::mem::forget(frame);

        VideoFrame::new(pts, DecodedFrame::Android(surface))
    }

    /// Gets the Java VM from NDK context.
    fn get_jvm() -> Result<jni::JavaVM, VideoError> {
        unsafe { jni::JavaVM::from_raw(ndk_context::android_context().vm().cast()) }
            .map_err(|e| VideoError::DecoderInit(format!("Failed to get JavaVM: {}", e)))
    }

    /// Returns the current decoder state.
    pub fn decoder_state(&self) -> MoqDecoderState {
        *self.shared.state.lock()
    }

    /// Returns the error message if in error state.
    pub fn error_message(&self) -> Option<String> {
        self.shared.error_message.lock().clone()
    }

    /// Returns the player ID for this decoder instance.
    pub fn player_id(&self) -> u64 {
        self.player_id
    }
}

impl Drop for MoqAndroidDecoder {
    fn drop(&mut self) {
        // Release MediaCodec resources via JNI
        if let Some(bridge) = self.bridge.take() {
            if let Ok(vm) = Self::get_jvm() {
                if let Ok(mut env) = vm.attach_current_thread() {
                    let _ = env.call_method(&bridge, "release", "()V", &[]);
                }
            }
        }

        // Release player's frame queue
        crate::media::android_video::release_player_queue(self.player_id);

        tracing::info!("MoqAndroidDecoder: Released player_id={}", self.player_id);
    }
}

impl VideoDecoderBackend for MoqAndroidDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Sync cached metadata from shared state (safe copy under lock)
        {
            let shared_metadata = self.shared.metadata.lock();
            if shared_metadata.width != self.cached_metadata.width
                || shared_metadata.height != self.cached_metadata.height
            {
                self.cached_metadata = shared_metadata.clone();
            }
        }

        // Check EOF
        if self.shared.eof_reached.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // Check for errors
        let state = *self.shared.state.lock();
        if state == MoqDecoderState::Error {
            return Err(VideoError::DecodeFailed(
                self.error_message()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        // Wait for metadata before initializing codec
        if state != MoqDecoderState::Streaming && !self.codec_configured {
            // Try to receive NAL units to trigger state updates
            let _ = self.nal_rx.try_recv();
            return Ok(None);
        }

        // Initialize codec once we have metadata
        if !self.codec_configured {
            self.initialize_codec()?;
        }

        // Submit any pending NAL units to MediaCodec
        while let Ok(nal_frame) = self.nal_rx.try_recv() {
            self.submit_nal_unit(&nal_frame.data, nal_frame.timestamp_us)?;
        }

        // Poll for decoded frames from HardwareBuffer queue
        self.poll_decoded_frames();

        // Return next decoded frame if available
        if let Some(frame) = self.pending_frames.pop_front() {
            return Ok(Some(self.convert_to_video_frame(frame)));
        }

        Ok(None)
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        Err(VideoError::SeekFailed(
            "Seeking is not supported on live MoQ streams".to_string(),
        ))
    }

    fn metadata(&self) -> &VideoMetadata {
        &self.cached_metadata
    }

    fn duration(&self) -> Option<Duration> {
        None // Live streams have no duration
    }

    fn is_eof(&self) -> bool {
        self.shared.eof_reached.load(Ordering::Relaxed)
    }

    fn buffering_percent(&self) -> i32 {
        self.shared.buffering_percent.load(Ordering::Relaxed)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::MediaCodec
    }

    fn handles_audio_internally(&self) -> bool {
        self.shared
            .audio
            .internal_audio_ready
            .load(Ordering::Relaxed)
    }

    fn audio_handle(&self) -> Option<super::audio::AudioHandle> {
        self.shared.audio.moq_audio_handle.lock().clone()
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_muted = muted;
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        self.audio_volume = volume.clamp(0.0, 1.0);
        Ok(())
    }
}

// ========================================================================
// JNI Entry Points for MoqMediaCodecBridge
// ========================================================================
//
// These functions are called from com.luminavideo.bridge.MoqMediaCodecBridge
// when decoded frames are available from MediaCodec.

/// JNI callback when a decoded frame is available from MediaCodec.
///
/// Called by MoqMediaCodecBridge.onOutputBufferAvailable() after acquiring
/// the HardwareBuffer from ImageReader.
///
/// This reuses the existing nativeSubmitHardwareBuffer infrastructure from
/// ExoPlayerBridge - the frame is queued by player_id for isolation.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_MoqMediaCodecBridge_nativeSubmitHardwareBuffer(
    env: JNIEnv,
    class: JClass,
    buffer: JObject,
    timestamp_ns: jlong,
    width: jint,
    height: jint,
    player_id: jlong,
    fence_fd: jint,
) {
    // Delegate to the existing ExoPlayerBridge implementation
    // This reuses all the HardwareBuffer acquisition and queue logic
    crate::media::android_video::Java_com_luminavideo_bridge_ExoPlayerBridge_nativeSubmitHardwareBuffer(
        env,
        class,
        buffer,
        timestamp_ns,
        width,
        height,
        player_id,
        fence_fd,
    );
}

/// JNI callback for codec errors.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_MoqMediaCodecBridge_nativeOnError(
    mut env: JNIEnv,
    _class: JClass,
    player_id: jlong,
    error_message: jni::objects::JString,
) {
    let error: String = env
        .get_string(&error_message)
        .map(|s| s.into())
        .unwrap_or_else(|_| "Unknown MediaCodec error".to_string());

    tracing::error!(
        "MoqMediaCodecBridge error (player_id={}): {}",
        player_id,
        error
    );
}

/// JNI callback when video dimensions change (e.g., adaptive bitrate switch).
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_MoqMediaCodecBridge_nativeOnVideoSizeChanged(
    _env: JNIEnv,
    _class: JClass,
    player_id: jlong,
    width: jint,
    height: jint,
) {
    tracing::info!(
        "MoqMediaCodecBridge video size changed (player_id={}): {}x{}",
        player_id,
        width,
        height
    );
}
