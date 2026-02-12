//! Android video decoder using ExoPlayer via JNI.
//!
//! This module provides video decoding on Android using ExoPlayer,
//! which automatically handles hardware acceleration via MediaCodec.
//!
//! # Self-Contained API
//!
//! Call `LuminaVideo.init(activity)` once in your Activity's `onCreate()`, then
//! `VideoPlayer::with_wgpu(url)` works self-contained — no Kotlin ExoPlayer setup needed.
//!
//! `AndroidVideoDecoder::new()` calls `LuminaVideo.createPlayer(nativeHandle)` via JNI,
//! which creates ExoPlayer on a dedicated HandlerThread with a Looper.
//!
//! # Zero-Copy GPU Rendering
//!
//! ```text
//! LuminaVideo.createPlayer() → ExoPlayer → ImageReader → HardwareBuffer → JNI → Vulkan Import
//! ```
//!
//! With the `zero-copy` feature enabled:
//! - YUV HardwareBuffers are imported directly into Vulkan via `VulkanYuvPipeline`
//! - GPU-side YUV→RGB conversion eliminates CPU copies
//!
//! Without zero-copy or when import fails, frames use CPU fallback (ByteBuffer extraction).
//!
//! ## Requirements
//!
//! - Android API 26+ (HardwareBuffer)
//! - `zero-copy` feature enabled
//! - Vulkan backend with `VK_ANDROID_external_memory_android_hardware_buffer`
//!
//! ## Implementation Files
//!
//! - `android/lumina-video-bridge/LuminaVideo.kt` - Static init + createPlayer()
//! - `android/lumina-video-bridge/ExoPlayerBridge.kt` - Kotlin bridge
//! - `android_video.rs` - JNI entry points and frame queue
//! - `zero_copy.rs` - Vulkan AHardwareBuffer import
//!
//! Tracking: lumina-video-5hd

use crossbeam_channel::{Receiver, Sender};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tracing::{info, warn};

use parking_lot::Mutex;

use jni::objects::{GlobalRef, JByteArray, JClass, JObject, JValue};
use jni::sys::{jint, jlong};
use jni::{JNIEnv, JavaVM};

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

use super::video::AndroidGpuSurface;

/// Gets the Java VM from the Android context.
///
/// # Safety
///
/// This function uses ndk_context which must be initialized by the Android activity
/// before calling this function.
fn get_jvm() -> Result<JavaVM, VideoError> {
    // Safety: ndk_context::android_context() returns a valid pointer when called
    // from an Android app that has been properly initialized by android-activity.
    unsafe { JavaVM::from_raw(ndk_context::android_context().vm().cast()) }
        .map_err(|e| VideoError::DecoderInit(format!("Failed to get JavaVM: {}", e)))
}

/// Gets the Android application context as a JNI object.
///
/// # Safety
///
/// Must be called from a thread attached to the JVM.
#[allow(dead_code)]
unsafe fn get_android_context() -> JObject<'static> {
    JObject::from_raw(ndk_context::android_context().context().cast())
}

/// Cached Android device information for diagnostics.
#[derive(Debug, Clone)]
struct AndroidDeviceInfo {
    /// API level (SDK_INT), e.g., 29 for Android 10
    sdk_int: i32,
    /// Device model (Build.MODEL), e.g., "Pixel 6"
    model: String,
    /// Device manufacturer (Build.MANUFACTURER), e.g., "Google"
    manufacturer: String,
}

/// Fetches Android device info via JNI.
///
/// Queries JNI for:
/// - `android.os.Build.VERSION.SDK_INT`
/// - `android.os.Build.MODEL`
/// - `android.os.Build.MANUFACTURER`
fn fetch_android_device_info() -> Option<AndroidDeviceInfo> {
    let vm = get_jvm().ok()?;

    let mut env = vm.attach_current_thread().ok()?;

    // Get SDK_INT from android.os.Build.VERSION
    let version_class = env.find_class("android/os/Build$VERSION").ok()?;

    let sdk_int = env
        .get_static_field(&version_class, "SDK_INT", "I")
        .ok()?
        .i()
        .ok()?;

    // Get MODEL and MANUFACTURER from android.os.Build
    let build_class = env.find_class("android/os/Build").ok()?;

    let model_obj = env
        .get_static_field(&build_class, "MODEL", "Ljava/lang/String;")
        .ok()?
        .l()
        .ok()?;

    let model: String = env
        .get_string((&model_obj).into())
        .map(|s| s.into())
        .unwrap_or_else(|_| "unknown".to_string());

    let manufacturer_obj = env
        .get_static_field(&build_class, "MANUFACTURER", "Ljava/lang/String;")
        .ok()?
        .l()
        .ok()?;

    let manufacturer: String = env
        .get_string((&manufacturer_obj).into())
        .map(|s| s.into())
        .unwrap_or_else(|_| "unknown".to_string());

    tracing::debug!(
        "Android device info: API {}, {} {}",
        sdk_int,
        manufacturer,
        model
    );

    Some(AndroidDeviceInfo {
        sdk_int,
        model,
        manufacturer,
    })
}

/// State shared between Rust and JNI callbacks.
#[allow(dead_code)]
struct SharedState {
    /// Channel for receiving frames from JNI callbacks (wired to JNI on frame submit)
    frame_sender: Sender<AndroidFrame>,
    /// Current video width
    width: u32,
    /// Current video height
    height: u32,
    /// Video duration in milliseconds
    duration_ms: i64,
    /// Current playback state from ExoPlayer
    playback_state: i32,
    /// Last error message
    last_error: Option<String>,
    /// Whether a new frame is available
    frame_available: bool,
}

/// Frame data received from Android.
pub struct AndroidFrame {
    /// RGBA pixel data
    pub pixels: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Timestamp in nanoseconds
    pub timestamp_ns: i64,
}

/// Android video decoder using ExoPlayer.
///
/// # Current State
///
/// **All frames currently use CPU fallback (ByteBuffer extraction).**
///
/// MediaCodec performs hardware-accelerated decoding, but decoded frames are copied
/// to CPU memory via ExoPlayer's `extractCurrentFrame()` before upload to wgpu.
/// This works but involves a GPU→CPU→GPU copy path.
///
/// # Zero-Copy Infrastructure (Ready, Blocked on Java)
///
/// The Rust-side infrastructure for zero-copy is complete:
/// - `AndroidGpuSurface` type exists for AHardwareBuffer references
/// - Vulkan import path is implemented
/// - `try_extract_ahardwarebuffer()` method is stubbed and ready
///
/// **Blocked on:** Java/Kotlin ExoPlayerBridge (not in this repo) implementing:
/// 1. AImageReader as output surface with `AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE`
/// 2. `getHardwareBuffer()` JNI method to expose native AHardwareBuffer pointer
/// 3. AImage lifecycle management
///
/// Tracking: lumina-video-5hd
/// See `zero-copy-android.md` for the full implementation plan.
pub struct AndroidVideoDecoder {
    /// JNI reference to ExoPlayerBridge instance
    bridge: GlobalRef,
    /// Shared state between Rust and JNI
    state: Arc<Mutex<SharedState>>,
    /// Frame receiver channel (reserved for future async frame delivery).
    /// Currently unused as frame extraction uses synchronous JNI calls and callback-based
    /// state updates. Kept for potential future migration to event-driven frame reception
    /// from the ExoPlayer bridge. Tracking: lumina-video-5hd
    #[allow(dead_code)]
    frame_receiver: Receiver<AndroidFrame>,
    /// Video metadata
    metadata: VideoMetadata,
    /// Whether the decoder is initialized (reserved for future state tracking).
    /// Currently always set to true and never read. Kept as placeholder for potential
    /// future use in state machine or lifecycle tracking. Tracking: lumina-video-5hd
    #[allow(dead_code)]
    initialized: bool,
    /// Native handle for JNI callback lookup
    native_handle: i64,
    /// Last known playback position (for placeholder frames)
    last_position: Duration,
    /// Video URL (for deferred playback start)
    url: String,
    /// Whether playback has been started
    started: bool,
    /// Count of frames using CPU fallback (ByteBuffer extraction).
    /// Incremented each frame since zero-copy to wgpu is not yet available.
    cpu_fallback_count: AtomicU64,
    /// Whether the CPU fallback warning has been logged (avoid spam).
    fallback_logged: AtomicBool,
    /// Cached Android device info for diagnostics (fetched once per decoder instance).
    device_info: Option<AndroidDeviceInfo>,
    /// Whether AHardwareBuffer zero-copy is available (API 29+ and ExoPlayer configured for it).
    /// This is checked once during initialization.
    ahardwarebuffer_available: bool,
}

/// Converts an Arc<Mutex<SharedState>> into a raw pointer handle for JNI.
/// The Arc's reference count is incremented, so the caller must call
/// `release_native_handle` to avoid leaking memory.
fn create_native_handle(state: Arc<Mutex<SharedState>>) -> i64 {
    // Clone to increment refcount, then convert to raw pointer
    let ptr = Arc::into_raw(state);
    ptr as i64
}

/// Releases a native handle, decrementing the Arc's reference count.
/// # Safety
/// The handle must have been created by `create_native_handle` and must not
/// have been released before.
fn release_native_handle(handle: i64) {
    if handle == 0 {
        return;
    }
    // Convert back to Arc and let it drop (decrements refcount)
    let ptr = handle as *const Mutex<SharedState>;
    unsafe {
        let _ = Arc::from_raw(ptr);
    }
}

/// Gets a clone of the SharedState Arc from a native handle.
/// Returns None if the handle is null (0).
/// # Safety
/// The handle must be valid (created by `create_native_handle` and not yet released).
fn get_native_state(handle: i64) -> Option<Arc<Mutex<SharedState>>> {
    if handle == 0 {
        return None;
    }
    let ptr = handle as *const Mutex<SharedState>;
    // Reconstruct Arc, clone it, then forget the original to avoid double-free
    let arc = unsafe { Arc::from_raw(ptr) };
    let cloned = Arc::clone(&arc);
    std::mem::forget(arc);
    Some(cloned)
}

impl AndroidVideoDecoder {
    /// Creates a new Android video decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        // Get JNI environment
        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to attach JNI thread: {}", e)))?;

        // Get Android context
        let context = unsafe { JObject::from_raw(ndk_context::android_context().context().cast()) };

        // Create frame channel (crossbeam for better performance)
        let (frame_sender, frame_receiver) = crossbeam_channel::unbounded();

        // Create shared state
        let state = Arc::new(Mutex::new(SharedState {
            frame_sender,
            width: 0,
            height: 0,
            duration_ms: 0,
            playback_state: 0,
            last_error: None,
            frame_available: false,
        }));

        // Create native handle (stores raw pointer to Arc for JNI callbacks)
        let native_handle = create_native_handle(Arc::clone(&state));

        // Helper to release handle on error - prevents Arc leak if initialization fails
        let release_on_error = |e: VideoError| {
            release_native_handle(native_handle);
            e
        };

        // Use LuminaVideo.createPlayer() for self-contained ExoPlayer creation.
        // This creates a dedicated HandlerThread, builds ExoPlayer on it, and sets up
        // ImageReader — all blocking until ready via CountDownLatch.

        // Get the app's class loader (native threads can't use find_class for app classes)
        let class_loader = env
            .call_method(&context, "getClassLoader", "()Ljava/lang/ClassLoader;", &[])
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get class loader: {}",
                    e
                )))
            })?
            .l()
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get class loader object: {}",
                    e
                )))
            })?;

        // Load LuminaVideo class via app classloader
        let class_name = env
            .new_string("com.luminavideo.bridge.LuminaVideo")
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to create class name string: {}",
                    e
                )))
            })?;

        let lumina_class = env
            .call_method(
                &class_loader,
                "loadClass",
                "(Ljava/lang/String;)Ljava/lang/Class;",
                &[JValue::Object(&class_name)],
            )
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to load LuminaVideo class: {}",
                    e
                )))
            })?
            .l()
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get LuminaVideo class: {}",
                    e
                )))
            })?;

        // Call LuminaVideo.createPlayer(nativeHandle) — blocks until ExoPlayer is ready
        let bridge_obj = env
            .call_static_method(
                JClass::from(lumina_class),
                "createPlayer",
                "(J)Lcom/luminavideo/bridge/ExoPlayerBridge;",
                &[JValue::Long(native_handle)],
            )
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "LuminaVideo.createPlayer() failed: {}",
                    e
                )))
            })?
            .l()
            .map_err(|e| {
                release_on_error(VideoError::DecoderInit(format!(
                    "Failed to get bridge object: {}",
                    e
                )))
            })?;

        if bridge_obj.is_null() {
            return Err(release_on_error(VideoError::DecoderInit(
                "LuminaVideo not initialized. Call LuminaVideo.init(activity) in your Activity.onCreate().".into()
            )));
        }

        // Create global reference
        let bridge_ref = env.new_global_ref(bridge_obj).map_err(|e| {
            release_on_error(VideoError::DecoderInit(format!(
                "Failed to create global ref: {}",
                e
            )))
        })?;

        // Don't call play() here - playback will start when decode_next() is first called
        // This prevents auto-play on app start

        // Initial metadata (will be updated by callbacks)
        let metadata = VideoMetadata {
            width: 1920,
            height: 1080,
            duration: None,
            frame_rate: 30.0,
            codec: "mediacodec".to_string(),
            pixel_aspect_ratio: 1.0,
            start_time: None, // MediaCodec doesn't expose stream start time
        };

        // Fetch device info once per decoder instance for diagnostics
        let device_info = fetch_android_device_info();

        // Check if AHardwareBuffer zero-copy is available (API 29+ required)
        let ahardwarebuffer_available = {
            let api_level = device_info.as_ref().map(|i| i.sdk_int).unwrap_or(0);
            let available = api_level >= 29;
            if available {
                info!(
                    "AHardwareBuffer zero-copy available (API {}). \
                     Note: Java/Kotlin ExoPlayerBridge must be configured to expose AHardwareBuffer.",
                    api_level
                );
            } else {
                info!(
                    "AHardwareBuffer zero-copy not available (API {} < 29). Using CPU fallback.",
                    api_level
                );
            }
            // ExoPlayerBridge.kt submits HardwareBuffers via JNI to HARDWARE_BUFFER_QUEUE.
            // The VulkanYuvPipeline handles YUV→RGB conversion on GPU.
            available
        };

        Ok(Self {
            bridge: bridge_ref,
            state,
            frame_receiver,
            metadata,
            initialized: true,
            native_handle,
            last_position: Duration::ZERO,
            url: url.to_string(),
            started: false,
            cpu_fallback_count: AtomicU64::new(0),
            fallback_logged: AtomicBool::new(false),
            device_info,
            ahardwarebuffer_available,
        })
    }

    /// Extracts the current frame from ExoPlayer.
    fn extract_frame(&self) -> Result<Option<AndroidFrame>, VideoError> {
        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        // Call extractCurrentFrame
        let result = env
            .call_method(&self.bridge, "extractCurrentFrame", "()[B", &[])
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to extract frame: {}", e)))?;

        let bytes_obj = result
            .l()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get frame bytes: {}", e)))?;

        if bytes_obj.is_null() {
            tracing::debug!("extractCurrentFrame returned null");
            return Ok(None);
        }

        let bytes_array = JByteArray::from(bytes_obj);
        let len = env
            .get_array_length(&bytes_array)
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get array length: {}", e)))?
            as usize;

        let mut pixels: Vec<i8> = vec![0; len];
        env.get_byte_array_region(&bytes_array, 0, &mut pixels)
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to get array data: {}", e)))?;

        // Convert i8 to u8
        let pixels: Vec<u8> = pixels.into_iter().map(|b| b as u8).collect();

        // Get current dimensions from state
        let state = self.state.lock();
        let width = state.width;
        let height = state.height;

        if width == 0 || height == 0 {
            tracing::debug!("Frame dimensions are 0x0");
            return Ok(None);
        }

        // Debug: Check if we got actual pixel data
        let non_zero_count = pixels.iter().filter(|&&b| b != 0).count();
        tracing::trace!(
            "Extracted frame: {}x{}, {} bytes, {} non-zero bytes",
            width,
            height,
            len,
            non_zero_count
        );

        // Use last_position for timestamp - don't call getCurrentPosition() here
        // because ExoPlayer requires main thread access and we're on a decode thread
        let position_ms = self.last_position.as_millis() as i64;

        Ok(Some(AndroidFrame {
            pixels,
            width,
            height,
            timestamp_ns: position_ms * 1_000_000, // ms to ns
        }))
    }

    /// Creates a minimal placeholder frame with the last known playback position.
    /// This keeps the decode loop alive without resetting playback position.
    fn create_placeholder_frame(&self) -> VideoFrame {
        let placeholder = CpuFrame::new(
            PixelFormat::Rgba,
            1,
            1,
            vec![Plane {
                data: vec![0, 0, 0, 255],
                stride: 4,
            }],
        );
        VideoFrame::new(self.last_position, DecodedFrame::Cpu(placeholder))
    }

    /// Starts playback if not already started.
    fn start_playback(&mut self) -> Result<(), VideoError> {
        if self.started {
            return Ok(());
        }

        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecoderInit(format!("Failed to attach JNI thread: {}", e)))?;

        let url_jstring = env
            .new_string(&self.url)
            .map_err(|e| VideoError::DecoderInit(format!("Failed to create URL string: {}", e)))?;

        env.call_method(
            &self.bridge,
            "play",
            "(Ljava/lang/String;)V",
            &[JValue::Object(&url_jstring)],
        )
        .map_err(|e| VideoError::DecoderInit(format!("Failed to start playback: {}", e)))?;

        self.started = true;
        tracing::info!("Started ExoPlayer playback for {}", self.url);
        Ok(())
    }

    /// Checks shared state for errors or EOS.
    /// Returns Some(result) if decode_next should return early, None to continue.
    fn check_state_for_early_return(&self) -> Option<Result<Option<VideoFrame>, VideoError>> {
        const STATE_ENDED: i32 = 4;

        let state = self.state.lock();

        if let Some(ref error) = state.last_error {
            return Some(Err(VideoError::DecodeFailed(error.clone())));
        }

        if state.playback_state == STATE_ENDED {
            return Some(Ok(None));
        }

        None
    }

    /// Waits for a frame to be available with timeout.
    /// Returns true if a frame is ready, false on timeout.
    fn wait_for_frame(&self, max_wait_ms: u64) -> Result<bool, VideoError> {
        const STATE_ENDED: i32 = 4;

        let start = std::time::Instant::now();

        while start.elapsed().as_millis() < max_wait_ms as u128 {
            {
                let mut state = self.state.lock();

                if state.frame_available {
                    state.frame_available = false;
                    return Ok(true);
                }

                if let Some(ref error) = state.last_error {
                    return Err(VideoError::DecodeFailed(error.clone()));
                }

                if state.playback_state == STATE_ENDED {
                    return Ok(false);
                }
            }

            std::thread::sleep(Duration::from_millis(5));
        }

        Ok(false)
    }

    /// Converts an AndroidFrame to a VideoFrame, updating last_position.
    ///
    /// This tracks CPU fallback since we're using ByteBuffer extraction instead
    /// of zero-copy AHardwareBuffer → Vulkan.
    fn android_frame_to_video_frame(&mut self, android_frame: AndroidFrame) -> VideoFrame {
        // Track CPU fallback for zero-copy visibility
        // TODO(e): When AHardwareBuffer → Vulkan integration is complete,
        // this counter should only increment on actual fallbacks, not every frame.
        let fallback_count = self.cpu_fallback_count.fetch_add(1, Ordering::Relaxed) + 1;
        if !self.fallback_logged.swap(true, Ordering::Relaxed) {
            // Use cached device info for diagnostics
            let (api_level, device_info_str) = match &self.device_info {
                Some(info) => (
                    format!("{}", info.sdk_int),
                    format!("{} {}", info.manufacturer, info.model),
                ),
                None => ("unknown".to_string(), "unknown".to_string()),
            };

            warn!(
                "Android zero-copy: CPU fallback active (ByteBuffer extraction). \
                 AHardwareBuffer → Vulkan → wgpu not yet integrated. \
                 Frame: {}x{}, format: RGBA, device: {}, API level: {} (requires 29+)",
                android_frame.width, android_frame.height, device_info_str, api_level
            );
        }

        if fallback_count % 1000 == 0 {
            tracing::debug!(
                "Android CPU fallback frame #{} ({}x{})",
                fallback_count,
                android_frame.width,
                android_frame.height
            );
        }

        let cpu_frame = CpuFrame::new(
            PixelFormat::Rgba,
            android_frame.width,
            android_frame.height,
            vec![Plane {
                data: android_frame.pixels,
                stride: android_frame.width as usize * 4,
            }],
        );

        let pts = Duration::from_nanos(android_frame.timestamp_ns as u64);
        self.last_position = pts;

        VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame))
    }

    /// Checks if playback has ended.
    fn is_playback_ended(&self) -> bool {
        const STATE_ENDED: i32 = 4;
        let state = self.state.lock();
        state.playback_state == STATE_ENDED
    }
}

impl Drop for AndroidVideoDecoder {
    fn drop(&mut self) {
        // Log zero-copy stats before cleanup
        let fallback_count = self.cpu_fallback_count.load(Ordering::Relaxed);
        if fallback_count > 0 {
            info!(
                "AndroidVideoDecoder zero-copy stats: {} frames used CPU fallback \
                 (ByteBuffer extraction). Zero-copy via AHardwareBuffer → Vulkan → wgpu \
                 awaits integration.",
                fallback_count
            );
        }

        // Release ExoPlayer resources FIRST to stop all callbacks
        // before invalidating the native handle
        if let Ok(vm) = get_jvm() {
            if let Ok(mut env) = vm.attach_current_thread() {
                let _ = env.call_method(&self.bridge, "release", "()V", &[]);
            }
        }

        // Now safe to release the native handle (decrements Arc refcount)
        release_native_handle(self.native_handle);
    }
}

impl VideoDecoderBackend for AndroidVideoDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn pause(&mut self) -> Result<(), VideoError> {
        if !self.started {
            return Ok(());
        }

        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::Generic(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(&self.bridge, "pause", "()V", &[])
            .map_err(|e| VideoError::Generic(format!("Failed to pause ExoPlayer: {}", e)))?;

        tracing::info!("Paused ExoPlayer playback");
        Ok(())
    }

    fn resume(&mut self) -> Result<(), VideoError> {
        if !self.started {
            return Ok(());
        }

        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::Generic(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(&self.bridge, "resume", "()V", &[])
            .map_err(|e| VideoError::Generic(format!("Failed to resume ExoPlayer: {}", e)))?;

        tracing::info!("Resumed ExoPlayer playback");
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Start playback on first decode_next call
        self.start_playback()?;
        tracing::debug!("decode_next called, started={}", self.started);

        // Check for errors or EOS from callbacks
        if let Some(result) = self.check_state_for_early_return() {
            return result;
        }

        // Wait for a frame to be available (with timeout)
        let frame_ready = self.wait_for_frame(100)?;

        if !frame_ready {
            // EOS check is handled in wait_for_frame, this is just timeout
            tracing::debug!("No frame ready, returning placeholder");
            return Ok(Some(self.create_placeholder_frame()));
        }

        tracing::debug!("Frame ready, extracting...");
        let frame = self.extract_frame()?;

        let Some(android_frame) = frame else {
            // Frame extraction failed - check if EOS or return placeholder
            if self.is_playback_ended() {
                return Ok(None);
            }
            return Ok(Some(self.create_placeholder_frame()));
        };

        Ok(Some(self.android_frame_to_video_frame(android_frame)))
    }

    fn seek(&mut self, position: Duration) -> Result<(), VideoError> {
        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::SeekFailed(format!("Failed to attach JNI thread: {}", e)))?;

        let position_ms = position.as_millis() as i64;

        env.call_method(&self.bridge, "seek", "(J)V", &[JValue::Long(position_ms)])
            .map_err(|e| VideoError::SeekFailed(format!("Seek failed: {}", e)))?;

        Ok(())
    }

    fn metadata(&self) -> &VideoMetadata {
        // Update duration from shared state if available
        let state = self.state.lock();
        if state.duration_ms > 0 {
            // We need to return updated metadata, but can't mutate self here
            // This is a limitation - duration will be returned via get_duration() instead
        }
        drop(state);
        &self.metadata
    }

    fn hw_accel_type(&self) -> HwAccelType {
        HwAccelType::MediaCodec
    }

    fn is_eof(&self) -> bool {
        // ExoPlayer playback states (from Player.java)
        const STATE_ENDED: i32 = 4;
        let state = self.state.lock();
        state.playback_state == STATE_ENDED
    }

    /// Android ExoPlayer handles audio internally - no separate FFmpeg audio thread needed.
    fn handles_audio_internally(&self) -> bool {
        true
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        // Call the inherent method
        AndroidVideoDecoder::set_muted(self, muted)
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        // Call the inherent method
        AndroidVideoDecoder::set_volume(self, volume)
    }

    fn duration(&self) -> Option<Duration> {
        // Use the dynamic get_duration() which reads from shared state (updated by callbacks)
        self.get_duration()
    }

    fn dimensions(&self) -> (u32, u32) {
        // Read dimensions from SharedState (updated by JNI callbacks)
        let state = self.state.lock();
        if state.width > 0 && state.height > 0 {
            (state.width, state.height)
        } else {
            // Fall back to placeholder if not yet known
            (self.metadata.width, self.metadata.height)
        }
    }
}

impl AndroidVideoDecoder {
    /// Sets the muted state for audio playback.
    pub fn set_muted(&self, muted: bool) -> Result<(), VideoError> {
        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(
            &self.bridge,
            "setMuted",
            "(Z)V",
            &[JValue::Bool(muted as u8)],
        )
        .map_err(|e| VideoError::DecodeFailed(format!("setMuted failed: {}", e)))?;

        tracing::debug!("Set muted: {}", muted);
        Ok(())
    }

    /// Sets the volume for audio playback.
    pub fn set_volume(&self, volume: f32) -> Result<(), VideoError> {
        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        env.call_method(&self.bridge, "setVolume", "(F)V", &[JValue::Float(volume)])
            .map_err(|e| VideoError::DecodeFailed(format!("setVolume failed: {}", e)))?;

        tracing::debug!("Set volume: {}", volume);
        Ok(())
    }

    /// Gets the current playback position.
    pub fn get_position(&self) -> Result<Duration, VideoError> {
        let vm = get_jvm()?;
        let mut env = vm
            .attach_current_thread()
            .map_err(|e| VideoError::DecodeFailed(format!("Failed to attach JNI thread: {}", e)))?;

        let result = env
            .call_method(&self.bridge, "getCurrentPosition", "()J", &[])
            .map_err(|e| VideoError::DecodeFailed(format!("getCurrentPosition failed: {}", e)))?;

        let position_ms = result.j().unwrap_or(0);
        Ok(Duration::from_millis(position_ms as u64))
    }

    /// Gets the video duration.
    pub fn get_duration(&self) -> Option<Duration> {
        // First check the shared state (updated by callbacks)
        let state = self.state.lock();
        if state.duration_ms > 0 {
            return Some(Duration::from_millis(state.duration_ms as u64));
        }
        drop(state);

        // Fall back to querying ExoPlayer directly
        let vm = get_jvm().ok()?;
        let mut env = match vm.attach_current_thread() {
            Ok(env) => env,
            Err(_) => return None,
        };

        let result = match env.call_method(&self.bridge, "getDuration", "()J", &[]) {
            Ok(r) => r,
            Err(_) => return None,
        };

        let duration_ms = result.j().unwrap_or(0);
        if duration_ms > 0 {
            Some(Duration::from_millis(duration_ms as u64))
        } else {
            None
        }
    }

    /// Returns the number of frames that used CPU fallback instead of zero-copy.
    ///
    /// Currently, ALL frames use CPU fallback (ByteBuffer extraction) since
    /// AHardwareBuffer → Vulkan → wgpu integration is not yet complete.
    pub fn cpu_fallback_count(&self) -> u64 {
        self.cpu_fallback_count.load(Ordering::Relaxed)
    }

    /// Returns true if AHardwareBuffer zero-copy is available.
    ///
    /// Requirements:
    /// - Android API 29+ (Android 10)
    /// - `zero-copy` and `android-zero-copy` features enabled
    /// - ExoPlayerBridge Java side configured to expose AHardwareBuffer
    ///
    /// Returns the cached availability status set during initialization.
    /// When all requirements are met, this returns true and zero-copy
    /// frame submission via [`submit_hardware_buffer_frame`] is available.
    pub fn is_ahardwarebuffer_available(&self) -> bool {
        self.ahardwarebuffer_available
    }

    /// Attempts to extract an AHardwareBuffer from ExoPlayer for zero-copy rendering.
    ///
    /// # Current Status
    ///
    /// **Rust-side infrastructure: READY** - This method is a stub awaiting Java/Kotlin
    /// implementation. The AndroidGpuSurface type and Vulkan import path are complete.
    ///
    /// **Blocked on Java/Kotlin side** - The ExoPlayerBridge class must expose AHardwareBuffer
    /// before this method can be activated. See `zero-copy-android.md` and tracking issue
    /// lumina-video-5hd for the full implementation plan.
    ///
    /// # Returns
    /// - `Some(AndroidGpuSurface)` if AHardwareBuffer was successfully obtained
    /// - `None` if zero-copy is not available (falls back to CPU extraction)
    ///
    /// # Java Side Requirements
    ///
    /// For this to work, the Java/Kotlin ExoPlayerBridge needs to:
    /// 1. Configure MediaCodec to output to an AImageReader surface
    /// 2. Create AImageReader with flags: `AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE`
    /// 3. Provide a JNI method `getHardwareBuffer()` that returns the AHardwareBuffer pointer
    ///
    /// Example Java implementation needed in ExoPlayerBridge.kt:
    /// ```java
    /// // In ExoPlayerBridge.kt
    /// private var imageReader: ImageReader? = null
    ///
    /// fun setupImageReader(width: Int, height: Int) {
    ///     imageReader = ImageReader.newInstance(
    ///         width, height,
    ///         ImageFormat.PRIVATE,  // or RGBA_8888
    ///         3,  // maxImages
    ///         HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE
    ///     )
    ///     player.setVideoSurface(imageReader!!.surface)
    /// }
    ///
    /// fun getHardwareBuffer(): Long {
    ///     val image = imageReader?.acquireLatestImage() ?: return 0
    ///     val buffer = image.hardwareBuffer ?: return 0
    ///     // Return the native AHardwareBuffer pointer
    ///     // Note: Need to use NDK to get the pointer from HardwareBuffer
    ///     return nativeGetHardwareBufferPointer(buffer)
    /// }
    /// ```
    #[allow(dead_code)]
    fn try_extract_ahardwarebuffer(&self) -> Option<AndroidGpuSurface> {
        if !self.ahardwarebuffer_available {
            return None;
        }

        // TODO(lumina-video-5hd): Android zero-copy AHardwareBuffer integration
        //
        // STATUS: Rust-side infrastructure is READY. The AndroidGpuSurface type and
        // wgpu-hal Vulkan import path are implemented. Blocked on Java/Kotlin side.
        //
        // Required Java/Kotlin work (in ExoPlayerBridge, not in this repo):
        // 1. Configure ExoPlayer to output to ImageReader with USAGE_GPU_SAMPLED_IMAGE
        // 2. Implement getHardwareBuffer() JNI method returning native AHardwareBuffer ptr
        // 3. Use NDK's AHardwareBuffer_fromHardwareBuffer() to get native pointer
        // 4. Handle AImage lifecycle (release after GPU is done with the buffer)
        //
        // Once Java side is ready, uncomment and adapt the JNI call below:
        //
        // let vm = get_jvm().ok()?;
        // let mut env = vm.attach_current_thread().ok()?;
        //
        // // Call Java method to get AHardwareBuffer pointer
        // let result = env.call_method(&self.bridge, "getHardwareBuffer", "()J", &[]).ok()?;
        // let ahb_ptr = result.j().ok()?;
        //
        // if ahb_ptr == 0 {
        //     return None;
        // }
        //
        // // Get dimensions from shared state
        // let (width, height) = {
        //     let state = self.state.lock();
        //     (state.width, state.height)
        // };
        //
        // // Create an owner to keep the AImage alive
        // // (In real impl, this would be a wrapper around the AImage)
        // struct AImageOwner { /* ... */ }
        // let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new(AImageOwner { /* ... */ });
        //
        // Some(unsafe {
        //     AndroidGpuSurface::new(
        //         ahb_ptr as *mut std::ffi::c_void,
        //         width,
        //         height,
        //         PixelFormat::Rgba, // or Nv12 depending on decoder output
        //         owner,
        //     )
        // })

        None // Not implemented yet
    }
}

// JNI callback implementations
// These are called from Java when events occur

#[no_mangle]
pub extern "C" fn Java_io_lumina_1video_video_ExoPlayerBridge_nativeOnFrameAvailable(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    width: jint,
    height: jint,
    _timestamp_ns: jlong,
) {
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock();
        state.width = width as u32;
        state.height = height as u32;
        state.frame_available = true;
    }
}

#[no_mangle]
pub extern "C" fn Java_io_lumina_1video_video_ExoPlayerBridge_nativeOnPlaybackStateChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    state_value: jint,
) {
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock();
        state.playback_state = state_value;
    }
}

#[no_mangle]
pub extern "C" fn Java_io_lumina_1video_video_ExoPlayerBridge_nativeOnError(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
    error_message: jni::objects::JString,
) {
    if let Some(state) = get_native_state(handle) {
        let error: String = env
            .get_string(&error_message)
            .map(|s| s.into())
            .unwrap_or_else(|_| "Unknown error".to_string());

        let mut state = state.lock();
        state.last_error = Some(error);
    }
}

#[no_mangle]
pub extern "C" fn Java_io_lumina_1video_video_ExoPlayerBridge_nativeOnVideoSizeChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    width: jint,
    height: jint,
) {
    tracing::info!(
        "nativeOnVideoSizeChanged: handle={}, {}x{}",
        handle,
        width,
        height
    );
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock();
        state.width = width as u32;
        state.height = height as u32;
        tracing::info!("Video size updated in SharedState: {}x{}", width, height);
    } else {
        tracing::warn!("nativeOnVideoSizeChanged: handle {} not found!", handle);
    }
}

#[no_mangle]
pub extern "C" fn Java_io_lumina_1video_video_ExoPlayerBridge_nativeOnDurationChanged(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
    duration_ms: jlong,
) {
    tracing::info!(
        "nativeOnDurationChanged: handle={}, duration_ms={}",
        handle,
        duration_ms
    );
    if let Some(state) = get_native_state(handle) {
        let mut state = state.lock();
        state.duration_ms = duration_ms;
        tracing::info!("Duration updated in SharedState: {} ms", duration_ms);
    } else {
        tracing::warn!("nativeOnDurationChanged: handle {} not found!", handle);
    }
}

// ============================================================================
// lumina-video Bridge JNI Entry Points
// ============================================================================
//
// These JNI functions are called from com.luminavideo.bridge.ExoPlayerBridge
// for zero-copy HardwareBuffer submission.
//
// Tracking: lumina-video-5hd

use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;

/// Per-player frame queues for HardwareBuffer submissions.
/// Each player has its own queue to avoid frame stealing between players.
/// Max queue size per player to prevent unbounded memory growth.
const MAX_QUEUE_SIZE_PER_PLAYER: usize = 8;

/// Per-player queues for multi-player isolation.
/// Uses parking_lot::RwLock for efficient concurrent access.
static PLAYER_QUEUES: OnceLock<parking_lot::RwLock<HashMap<u64, VecDeque<AndroidVideoFrame>>>> =
    OnceLock::new();

/// Legacy single-player queue for backward compatibility (player_id = 0).
/// The rendering thread reads from this queue to get zero-copy frames.
static HARDWARE_BUFFER_QUEUE: OnceLock<crossbeam_channel::Sender<AndroidVideoFrame>> =
    OnceLock::new();
static HARDWARE_BUFFER_RECEIVER: OnceLock<crossbeam_channel::Receiver<AndroidVideoFrame>> =
    OnceLock::new();

// NOTE: NDK ImageReader integration is disabled for now.
//
// The ndk crate's ImageReader is !Send because it wraps a raw pointer to AImageReader.
// This means we can't store it in a static HashMap. The ImageReader must stay on the
// thread that created it (for callback delivery).
//
// Future work: Either:
// 1. Create the NDK ImageReader on the JNI callback thread and keep it there
// 2. Use raw ndk-sys calls with manual lifetime management
// 3. Accept that Java ImageReader is sufficient (fence_fd=-1, queue barrier provides sync)
//
// For now, the Java ImageReader path works correctly. The queue ownership transfer
// barrier (VK_QUEUE_FAMILY_EXTERNAL) provides sufficient synchronization in most cases.
// The fence FD would provide tighter sync for edge cases with very fast decode.
//
// Tracking: Consider creating a beads issue for this future work.

/// Initializes the per-player queues map.
fn init_player_queues() {
    let _ = PLAYER_QUEUES.get_or_init(|| parking_lot::RwLock::new(HashMap::new()));
}

/// AHardwareBuffer format constant for RGBA8 (matches Android's AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM)
pub const AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM: u32 = 1;

/// AHardwareBuffer format constant for NV12 (Y8Cb8Cr8_420).
/// This is the most common YUV format from MediaCodec video decoders.
/// AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420 = 0x23 = 35
pub const AHARDWAREBUFFER_FORMAT_Y8CB8CR8_420: u32 = 35;

/// AHardwareBuffer format constant for YV12 (planar YUV 4:2:0).
/// Y plane followed by V plane then U plane.
/// AHARDWAREBUFFER_FORMAT_YV12 = 0x32315659 = 842094169
pub const AHARDWAREBUFFER_FORMAT_YV12: u32 = 0x32315659;

/// Returns true if the AHardwareBuffer format is a YUV format requiring multi-plane import.
pub fn is_yuv_hardware_buffer_format(format: u32) -> bool {
    matches!(
        format,
        AHARDWAREBUFFER_FORMAT_Y8CB8CR8_420 | AHARDWAREBUFFER_FORMAT_YV12
    )
}

/// Returns true if the AHardwareBuffer format is YV12 (3-plane: Y, V, U).
///
/// YV12 is a planar format where V comes before U, which is opposite to I420/YUV420p.
/// When importing YV12 buffers, the U and V planes must be swapped to match the
/// shader's expected order (Y, U, V).
pub fn is_yv12_format(format: u32) -> bool {
    format == AHARDWAREBUFFER_FORMAT_YV12
}

/// Counter for generating unique player IDs
static NEXT_PLAYER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Generates a unique player ID for multi-player isolation.
/// Each ExoPlayerBridge instance should call this once and use the returned ID
/// for all frame submissions.
pub fn generate_player_id() -> u64 {
    NEXT_PLAYER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Represents a video frame backed by an AHardwareBuffer.
/// Used for zero-copy rendering via Vulkan external memory.
pub struct AndroidVideoFrame {
    /// Raw AHardwareBuffer pointer (owned, must call AHardwareBuffer_release on drop)
    pub buffer: *mut std::ffi::c_void,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Presentation timestamp in nanoseconds
    pub timestamp_ns: i64,
    /// AHardwareBuffer format (from AHardwareBuffer_describe)
    /// Use AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM (1) for RGBA, other values typically indicate YUV
    pub format: u32,
    /// Player ID for multi-player isolation (0 = legacy/unknown)
    pub player_id: u64,
    /// Sync fence file descriptor from the producer (MediaCodec).
    /// -1 means no fence (already signaled or not available).
    /// The consumer (Vulkan) must wait on this fence before reading the buffer.
    /// This is critical for correct synchronization with hardware video decoders.
    pub fence_fd: i32,
}

// SAFETY: AndroidVideoFrame can be sent and shared between threads because:
// - The raw AHardwareBuffer pointer is an opaque handle that Android guarantees
//   is safe to use from any thread (per NDK documentation).
// - AHardwareBuffer_release() is explicitly documented as thread-safe.
// - All other fields (width, height, timestamp_ns, format, player_id) are Copy types.
// - The pointer is not mutated after creation; only Drop reads it.
unsafe impl Send for AndroidVideoFrame {}
unsafe impl Sync for AndroidVideoFrame {}

impl Drop for AndroidVideoFrame {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            // AHardwareBuffer_release is safe to call from any thread
            extern "C" {
                fn AHardwareBuffer_release(buffer: *mut std::ffi::c_void);
            }
            unsafe {
                AHardwareBuffer_release(self.buffer);
            }
        }

        // Close the sync fence FD if we own it (not yet consumed by Vulkan)
        // -1 means no fence or already signaled
        if self.fence_fd >= 0 {
            extern "C" {
                fn close(fd: i32) -> i32;
            }
            unsafe {
                close(self.fence_fd);
            }
        }
    }
}

/// Initializes the HardwareBuffer queue.
/// Called automatically on first use.
fn init_hardware_buffer_queue() {
    let (sender, receiver) = crossbeam_channel::bounded(8);
    let _ = HARDWARE_BUFFER_QUEUE.set(sender);
    let _ = HARDWARE_BUFFER_RECEIVER.set(receiver);
}

/// Gets the next available HardwareBuffer frame for a specific player, if any.
/// Called by the rendering thread to get zero-copy frames.
///
/// Each player has its own queue, so frames from one player never affect another.
/// Use player_id=0 to accept any frame (legacy/single-player mode via shared queue).
pub fn try_receive_hardware_buffer_for_player(player_id: u64) -> Option<AndroidVideoFrame> {
    // Legacy mode (player_id=0): use the shared queue for backward compatibility
    if player_id == 0 {
        let receiver = HARDWARE_BUFFER_RECEIVER.get()?;
        return receiver.try_recv().ok();
    }

    // Per-player mode: pop from this player's dedicated queue
    let queues = PLAYER_QUEUES.get()?;
    let mut queues_guard = queues.write();
    queues_guard.get_mut(&player_id)?.pop_front()
}

/// Gets the next available HardwareBuffer frame, if any (legacy single-player mode).
/// Called by the rendering thread to get zero-copy frames.
pub fn try_receive_hardware_buffer() -> Option<AndroidVideoFrame> {
    try_receive_hardware_buffer_for_player(0)
}

/// Releases a player's frame queue, freeing any pending frames.
///
/// Call this when an ExoPlayerBridge is released to prevent memory leaks
/// in long-running apps that create/destroy many players.
///
/// For legacy mode (player_id=0), this is a no-op since the shared queue
/// is meant to persist for the app lifetime.
pub fn release_player_queue(player_id: u64) {
    if player_id == 0 {
        return; // Legacy mode uses shared queue, don't clear
    }
    if let Some(queues) = PLAYER_QUEUES.get() {
        let mut queues_guard = queues.write();
        if let Some(removed_queue) = queues_guard.remove(&player_id) {
            let frame_count = removed_queue.len();
            if frame_count > 0 {
                tracing::debug!(
                    "Released player {} queue with {} pending frames",
                    player_id,
                    frame_count
                );
            }
            // Frames are dropped here, releasing their AHardwareBuffers
        }
    }
}

/// JNI entry point for generating a unique player ID.
///
/// Called by ExoPlayerBridge constructor to get a unique ID for multi-player isolation.
/// Each ExoPlayerBridge instance should call this once and use the returned ID
/// for all frame submissions.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeGeneratePlayerId(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    generate_player_id() as jlong
}

/// JNI entry point for releasing a player's frame queue.
///
/// Called by ExoPlayerBridge.release() to clean up the player's queue
/// and free any pending frames. This prevents memory leaks in long-running
/// apps that create/destroy many players.
///
/// # Arguments
///
/// - `player_id`: The player ID returned by nativeGeneratePlayerId
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeReleasePlayer(
    _env: JNIEnv,
    _class: JClass,
    player_id: jlong,
) {
    release_player_queue(player_id as u64);

    // NOTE: NDK ImageReader cleanup was removed - using Java ImageReader path
    // which handles its own lifecycle in Kotlin.
}

/// JNI entry point for video size change notification.
///
/// Called by ExoPlayerBridge when ExoPlayer reports a new video size.
///
/// NOTE: NDK ImageReader integration is currently disabled due to threading constraints
/// (ImageReader is !Send). The Java ImageReader path continues to work, passing fence_fd=-1.
/// The queue ownership transfer barrier provides synchronization in most cases.
///
/// Future work: Implement proper thread-local NDK ImageReader management.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeOnVideoSizeChanged(
    _env: JNIEnv,
    _this: JObject,
    native_handle: jlong,
    width: jint,
    height: jint,
) {
    tracing::info!(
        "nativeOnVideoSizeChanged: handle={}, {}x{} (using Java ImageReader, fence_fd=-1)",
        native_handle,
        width,
        height
    );
    // Update SharedState with new dimensions so AndroidVideoDecoder::dimensions() returns correct values
    if let Some(state) = get_native_state(native_handle) {
        let mut state = state.lock();
        state.width = width as u32;
        state.height = height as u32;
        tracing::info!("Video size updated in SharedState: {}x{}", width, height);
    } else {
        tracing::warn!(
            "nativeOnVideoSizeChanged: handle {} not found!",
            native_handle
        );
    }
}

/// JNI entry point for HardwareBuffer submission from Kotlin.
///
/// Called by ExoPlayerBridge.nativeSubmitHardwareBuffer().
/// The HardwareBuffer is converted to an AHardwareBuffer* and queued for rendering.
///
/// # Safety
///
/// - `buffer` must be a valid HardwareBuffer JNI object
/// - The HardwareBuffer is retained by this function (AHardwareBuffer_acquire called)
/// - The caller (Java) can safely close their Image after this returns
/// - `player_id` should be the value returned by nativeGeneratePlayerId
/// - `fence_fd` is the sync fence from the producer (-1 if none/already signaled)
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeSubmitHardwareBuffer(
    env: JNIEnv,
    _class: JClass,
    buffer: JObject,
    timestamp_ns: jlong,
    width: jint,
    height: jint,
    player_id: jlong,
    fence_fd: jint,
) {
    // Ensure queues are initialized
    if HARDWARE_BUFFER_QUEUE.get().is_none() {
        init_hardware_buffer_queue();
    }
    init_player_queues();

    // Convert Java HardwareBuffer to native AHardwareBuffer*
    extern "C" {
        fn AHardwareBuffer_fromHardwareBuffer(
            env: *mut std::ffi::c_void,
            hardware_buffer: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void;
        fn AHardwareBuffer_acquire(buffer: *mut std::ffi::c_void);
        fn AHardwareBuffer_describe(buffer: *mut std::ffi::c_void, desc: *mut AHardwareBufferDesc);
    }

    /// AHardwareBuffer descriptor for querying buffer properties
    #[repr(C)]
    struct AHardwareBufferDesc {
        width: u32,
        height: u32,
        layers: u32,
        format: u32,
        usage: u64,
        stride: u32,
        rfu0: u32,
        rfu1: u64,
    }

    let ahb = unsafe {
        let env_ptr = env.get_raw() as *mut std::ffi::c_void;
        let buffer_ptr = buffer.as_raw() as *mut std::ffi::c_void;
        AHardwareBuffer_fromHardwareBuffer(env_ptr, buffer_ptr)
    };

    if ahb.is_null() {
        tracing::warn!(
            "nativeSubmitHardwareBuffer: AHardwareBuffer_fromHardwareBuffer returned null"
        );
        return;
    }

    // Query the buffer format using AHardwareBuffer_describe
    let format = unsafe {
        let mut desc = std::mem::zeroed::<AHardwareBufferDesc>();
        AHardwareBuffer_describe(ahb, &mut desc);
        desc.format
    };

    // Acquire a reference to keep the buffer alive
    // Rust now owns this reference and will release it when AndroidVideoFrame is dropped
    unsafe {
        AHardwareBuffer_acquire(ahb);
    }

    let player_id_u64 = player_id as u64;

    let frame = AndroidVideoFrame {
        buffer: ahb,
        width: width as u32,
        height: height as u32,
        timestamp_ns,
        format,
        player_id: player_id_u64,
        fence_fd,
    };

    // Route to the appropriate queue based on player_id
    if player_id_u64 == 0 {
        // Legacy mode: use shared queue for backward compatibility
        if let Some(sender) = HARDWARE_BUFFER_QUEUE.get() {
            if let Err(e) = sender.try_send(frame) {
                tracing::debug!("HardwareBuffer queue full, dropping frame: {:?}", e);
            }
        }
    } else {
        // Per-player mode: route to player's dedicated queue
        if let Some(queues) = PLAYER_QUEUES.get() {
            let mut queues_guard = queues.write();
            let queue = queues_guard
                .entry(player_id_u64)
                .or_insert_with(VecDeque::new);

            // Enforce max queue size per player
            if queue.len() >= MAX_QUEUE_SIZE_PER_PLAYER {
                // Drop oldest frame to make room
                if let Some(old_frame) = queue.pop_front() {
                    tracing::debug!(
                        "Player {} queue full, dropping oldest frame (ts={})",
                        player_id_u64,
                        old_frame.timestamp_ns
                    );
                    // old_frame is dropped here, releasing AHardwareBuffer
                }
            }
            queue.push_back(frame);
        }
    }
}
