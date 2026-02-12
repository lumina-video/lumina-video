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

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tracing::info;

use parking_lot::Mutex;

use jni::objects::{GlobalRef, JClass, JObject, JValue};
use jni::sys::{jint, jlong};
use jni::{JNIEnv, JavaVM};

use super::video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata,
};

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

/// Fetches the Android SDK API level via JNI and logs device info for diagnostics.
fn fetch_android_api_level() -> Option<i32> {
    let vm = get_jvm().ok()?;
    let mut env = vm.attach_current_thread().ok()?;

    let version_class = env.find_class("android/os/Build$VERSION").ok()?;
    let sdk_int = env
        .get_static_field(&version_class, "SDK_INT", "I")
        .ok()?
        .i()
        .ok()?;

    // Log device model/manufacturer for diagnostics
    let build_class = env.find_class("android/os/Build").ok()?;

    let model_obj = env
        .get_static_field(&build_class, "MODEL", "Ljava/lang/String;")
        .ok()
        .and_then(|v| v.l().ok());
    let model: String = if let Some(ref obj) = model_obj {
        env.get_string(obj.into())
            .map(|s| s.into())
            .unwrap_or_else(|_| "unknown".to_string())
    } else {
        "unknown".to_string()
    };

    let mfr_obj = env
        .get_static_field(&build_class, "MANUFACTURER", "Ljava/lang/String;")
        .ok()
        .and_then(|v| v.l().ok());
    let manufacturer: String = if let Some(ref obj) = mfr_obj {
        env.get_string(obj.into())
            .map(|s| s.into())
            .unwrap_or_else(|_| "unknown".to_string())
    } else {
        "unknown".to_string()
    };

    tracing::debug!("Android device: API {}, {} {}", sdk_int, manufacturer, model);
    Some(sdk_int)
}

/// State shared between Rust and JNI callbacks.
struct SharedState {
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
}

/// Android video decoder using ExoPlayer via JNI.
///
/// # Architecture
///
/// `LuminaVideo.init(activity)` in Kotlin stores the application context. When Rust
/// calls `AndroidVideoDecoder::new(url)`, it invokes `LuminaVideo.createPlayer()` via
/// JNI, which creates ExoPlayer on a dedicated HandlerThread with a Looper. ExoPlayer
/// decodes to an ImageReader, and decoded frames arrive as HardwareBuffers via JNI
/// callbacks (`nativeSubmitHardwareBuffer`).
///
/// # Zero-Copy Rendering
///
/// HardwareBuffers are imported into Vulkan via `VK_ANDROID_external_memory_android_hardware_buffer`
/// with GPU-side YCbCr→RGB conversion (`VkSamplerYcbcrConversion`). No CPU pixel copies
/// in steady-state playback. See `video_texture.rs` for the import path and `zero_copy.rs`
/// for the Vulkan YUV pipeline.
///
/// Tracking: lumina-video-5hd
pub struct AndroidVideoDecoder {
    /// JNI reference to ExoPlayerBridge instance
    bridge: GlobalRef,
    /// Shared state between Rust and JNI
    state: Arc<Mutex<SharedState>>,
    /// Video metadata
    metadata: VideoMetadata,
    /// Native handle for JNI callback lookup
    native_handle: i64,
    /// Last known playback position (for placeholder frames)
    last_position: Duration,
    /// Video URL (for deferred playback start)
    url: String,
    /// Whether playback has been started
    started: bool,
    /// Whether AHardwareBuffer zero-copy is available (Android API 29+).
    /// Checked once during initialization.
    ahardwarebuffer_available: bool,
    /// Player ID for per-player frame queue routing.
    /// Generated by `nativeGeneratePlayerId()` on the Kotlin side.
    player_id: u64,
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

        // Create shared state for JNI callbacks
        let state = Arc::new(Mutex::new(SharedState {
            width: 0,
            height: 0,
            duration_ms: 0,
            playback_state: 0,
            last_error: None,
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

        // Query player ID from bridge (generated by nativeGeneratePlayerId in Kotlin)
        let player_id = env
            .call_method(&bridge_ref, "getPlayerId", "()J", &[])
            .ok()
            .and_then(|v| v.j().ok())
            .unwrap_or(0) as u64;

        info!("AndroidVideoDecoder: player_id={}", player_id);

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

        // Check if AHardwareBuffer zero-copy is available (API 29+ required)
        let ahardwarebuffer_available = {
            let api_level = fetch_android_api_level().unwrap_or(0);
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
            metadata,
            native_handle,
            last_position: Duration::ZERO,
            url: url.to_string(),
            started: false,
            ahardwarebuffer_available,
            player_id,
        })
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
    ///
    /// Checks the HardwareBuffer queue directly — frames arrive via
    /// `nativeSubmitHardwareBuffer` and are consumed by the render thread
    /// in `video_texture.rs::prepare()`.
    fn wait_for_frame(&self, max_wait_ms: u64) -> Result<bool, VideoError> {
        const STATE_ENDED: i32 = 4;

        let start = std::time::Instant::now();

        while start.elapsed().as_millis() < max_wait_ms as u128 {
            {
                let state = self.state.lock();

                if let Some(ref error) = state.last_error {
                    return Err(VideoError::DecodeFailed(error.clone()));
                }

                if state.playback_state == STATE_ENDED {
                    return Ok(false);
                }
            }

            // Check HardwareBuffer queue (frames arrive from ExoPlayerBridge)
            if has_pending_hardware_buffer(self.player_id) {
                return Ok(true);
            }

            std::thread::sleep(Duration::from_millis(5));
        }

        Ok(false)
    }

}

impl Drop for AndroidVideoDecoder {
    fn drop(&mut self) {
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

    fn android_player_id(&self) -> u64 {
        self.player_id
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

        // Check for errors or EOS from callbacks
        if let Some(result) = self.check_state_for_early_return() {
            return result;
        }

        // Wait for a HardwareBuffer frame or error/EOS.
        // Actual frame delivery to the render thread happens independently via
        // try_receive_hardware_buffer_for_player() in video_texture.rs::prepare().
        // This loop serves as a heartbeat for error/EOS detection and position updates.
        let frame_ready = self.wait_for_frame(100)?;

        if !frame_ready {
            return Ok(Some(self.create_placeholder_frame()));
        }

        // Update position from cached bridge state
        let vm = get_jvm();
        if let Ok(vm) = vm {
            if let Ok(mut env) = vm.attach_current_thread() {
                if let Ok(pos) = env.call_method(&self.bridge, "getCurrentPosition", "()J", &[]) {
                    if let Ok(pos_ms) = pos.j() {
                        if pos_ms >= 0 {
                            self.last_position = Duration::from_millis(pos_ms as u64);
                        }
                    }
                }
            }
        }

        // Return placeholder — the real frame is delivered via HardwareBuffer queue
        Ok(Some(self.create_placeholder_frame()))
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

    /// Returns true if AHardwareBuffer zero-copy is available (Android API 29+).
    pub fn is_ahardwarebuffer_available(&self) -> bool {
        self.ahardwarebuffer_available
    }
}

// JNI callback implementations
// Called from com.luminavideo.bridge.ExoPlayerBridge

// ============================================================================
// Android Zero-Copy Stats (Per-Player)
// ============================================================================

/// Per-player zero-copy rendering counters.
/// Stored in the per-player state map and cleaned up when the player is released.
/// Last frame import result, stored as atomic for lock-free reads.
/// 0 = no frames yet, 1 = true zero-copy, 2 = cpu-assisted, 3 = failed
type AtomicFrameResult = AtomicU8;
const FRAME_RESULT_NONE: u8 = 0;
const FRAME_RESULT_ZERO_COPY: u8 = 1;
const FRAME_RESULT_CPU_ASSISTED: u8 = 2;
const FRAME_RESULT_FAILED: u8 = 3;

struct PerPlayerStats {
    /// Frames imported via true Vulkan zero-copy (0 CPU copies)
    true_zero_copy: AtomicU64,
    /// Frames imported via CPU-assisted path (lockPlanes → memcpy → GPU)
    cpu_assisted: AtomicU64,
    /// Frames that failed import entirely
    failed: AtomicU64,
    /// Result of the most recent frame (for non-sticky UI status)
    last_result: AtomicFrameResult,
}

impl PerPlayerStats {
    fn new() -> Self {
        Self {
            true_zero_copy: AtomicU64::new(0),
            cpu_assisted: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            last_result: AtomicFrameResult::new(FRAME_RESULT_NONE),
        }
    }
}

/// Per-player stats map, keyed by player_id.
/// Cleaned up alongside frame queues in `release_player_queue()`.
static PLAYER_STATS: OnceLock<parking_lot::RwLock<HashMap<u64, PerPlayerStats>>> = OnceLock::new();

/// Initializes the per-player stats map.
fn init_player_stats() {
    let _ = PLAYER_STATS.get_or_init(|| parking_lot::RwLock::new(HashMap::new()));
}

/// Gets or creates per-player stats entry, then applies the given function.
fn with_player_stats(player_id: u64, f: impl FnOnce(&PerPlayerStats)) {
    init_player_stats();
    let Some(stats_map) = PLAYER_STATS.get() else {
        return;
    };
    // Try read lock first (common path)
    {
        let guard = stats_map.read();
        if let Some(stats) = guard.get(&player_id) {
            f(stats);
            return;
        }
    }
    // Create entry under write lock
    let mut guard = stats_map.write();
    let stats = guard.entry(player_id).or_insert_with(PerPlayerStats::new);
    f(stats);
}

/// Current zero-copy status based on the most recent frame.
/// Variants ordered by severity (higher = more concerning) for aggregate comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ZeroCopyStatus {
    /// No frames processed yet
    Waiting = 0,
    /// Most recent frame used true zero-copy
    ZeroCopy = 1,
    /// Most recent frame used CPU-assisted path
    CpuAssisted = 2,
    /// Most recent frame failed import
    Failed = 3,
}

/// Snapshot of Android zero-copy rendering statistics.
#[derive(Debug, Clone)]
pub struct AndroidZeroCopySnapshot {
    /// Frames imported via true Vulkan zero-copy (0 CPU copies)
    pub true_zero_copy_frames: u64,
    /// Frames imported via CPU-assisted path (lockPlanes → memcpy → GPU)
    pub cpu_assisted_frames: u64,
    /// Frames that failed import entirely
    pub failed_frames: u64,
    /// Status of the most recent frame (non-sticky)
    pub current_status: ZeroCopyStatus,
}

impl AndroidZeroCopySnapshot {
    /// Total frames processed
    pub fn total(&self) -> u64 {
        self.true_zero_copy_frames + self.cpu_assisted_frames + self.failed_frames
    }

    /// True if most recent frame used true zero-copy
    pub fn is_true_zero_copy(&self) -> bool {
        self.current_status == ZeroCopyStatus::ZeroCopy
    }

    /// Returns a human-readable status label based on most recent frame
    pub fn status_label(&self) -> &'static str {
        match self.current_status {
            ZeroCopyStatus::Waiting => "Waiting",
            ZeroCopyStatus::ZeroCopy => "Zero-Copy",
            ZeroCopyStatus::CpuAssisted => "CPU Fallback",
            ZeroCopyStatus::Failed => "Failed"
        }
    }
}

fn last_result_to_status(v: u8) -> ZeroCopyStatus {
    match v {
        FRAME_RESULT_ZERO_COPY => ZeroCopyStatus::ZeroCopy,
        FRAME_RESULT_CPU_ASSISTED => ZeroCopyStatus::CpuAssisted,
        FRAME_RESULT_FAILED => ZeroCopyStatus::Failed,
        _ => ZeroCopyStatus::Waiting,
    }
}

/// Returns a snapshot of zero-copy stats for the given player.
///
/// Use `player_id=0` for aggregate stats across all players (demo/single-player mode).
pub fn android_zero_copy_snapshot(player_id: u64) -> AndroidZeroCopySnapshot {
    init_player_stats();
    let Some(stats_map) = PLAYER_STATS.get() else {
        return AndroidZeroCopySnapshot {
            true_zero_copy_frames: 0,
            cpu_assisted_frames: 0,
            failed_frames: 0,
            current_status: ZeroCopyStatus::Waiting,
        };
    };
    let guard = stats_map.read();

    if player_id == 0 {
        // Aggregate across all players
        let mut snap = AndroidZeroCopySnapshot {
            true_zero_copy_frames: 0,
            cpu_assisted_frames: 0,
            failed_frames: 0,
            current_status: ZeroCopyStatus::Waiting,
        };
        for stats in guard.values() {
            snap.true_zero_copy_frames += stats.true_zero_copy.load(Ordering::Relaxed);
            snap.cpu_assisted_frames += stats.cpu_assisted.load(Ordering::Relaxed);
            snap.failed_frames += stats.failed.load(Ordering::Relaxed);
            // Pick the most concerning status across all players:
            // Failed > CpuAssisted > ZeroCopy > Waiting
            let status = last_result_to_status(stats.last_result.load(Ordering::Relaxed));
            if status > snap.current_status {
                snap.current_status = status;
            }
        }
        snap
    } else if let Some(stats) = guard.get(&player_id) {
        AndroidZeroCopySnapshot {
            true_zero_copy_frames: stats.true_zero_copy.load(Ordering::Relaxed),
            cpu_assisted_frames: stats.cpu_assisted.load(Ordering::Relaxed),
            failed_frames: stats.failed.load(Ordering::Relaxed),
            current_status: last_result_to_status(stats.last_result.load(Ordering::Relaxed)),
        }
    } else {
        AndroidZeroCopySnapshot {
            true_zero_copy_frames: 0,
            cpu_assisted_frames: 0,
            failed_frames: 0,
            current_status: ZeroCopyStatus::Waiting,
        }
    }
}

/// Increment true zero-copy frame count for a player (called from video_texture.rs)
pub fn record_true_zero_copy(player_id: u64) {
    with_player_stats(player_id, |s| {
        s.true_zero_copy.fetch_add(1, Ordering::Relaxed);
        s.last_result.store(FRAME_RESULT_ZERO_COPY, Ordering::Relaxed);
    });
}

/// Increment CPU-assisted frame count for a player (called from video_texture.rs)
pub fn record_cpu_assisted(player_id: u64) {
    with_player_stats(player_id, |s| {
        s.cpu_assisted.fetch_add(1, Ordering::Relaxed);
        s.last_result.store(FRAME_RESULT_CPU_ASSISTED, Ordering::Relaxed);
    });
}

/// Increment failed frame count for a player (called from video_texture.rs)
pub fn record_import_failed(player_id: u64) {
    with_player_stats(player_id, |s| {
        s.failed.fetch_add(1, Ordering::Relaxed);
        s.last_result.store(FRAME_RESULT_FAILED, Ordering::Relaxed);
    });
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

/// Returns true if a HardwareBuffer frame is pending for the given player.
/// Non-consuming peek used by wait_for_frame to avoid 100ms timeout spin.
pub fn has_pending_hardware_buffer(player_id: u64) -> bool {
    if player_id == 0 {
        HARDWARE_BUFFER_RECEIVER
            .get()
            .map(|r| !r.is_empty())
            .unwrap_or(false)
    } else {
        PLAYER_QUEUES
            .get()
            .map(|queues| {
                queues
                    .read()
                    .get(&player_id)
                    .map(|q| !q.is_empty())
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }
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
    // Clean up per-player stats
    if let Some(stats_map) = PLAYER_STATS.get() {
        let mut guard = stats_map.write();
        guard.remove(&player_id);
    }
}

/// JNI entry point for generating a unique player ID.
///
/// Called by ExoPlayerBridge.initializeWithPlayer() to get a per-player ID.
/// IDs start from 1; 0 is reserved for legacy shared queue fallback.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeGeneratePlayerId(
    _env: JNIEnv,
    _this: JObject,
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
    _this: JObject,
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

/// JNI entry point for playback state change notification.
///
/// Called by ExoPlayerBridge when ExoPlayer's playback state changes.
/// Maps to ExoPlayer's Player.STATE_* constants (1=IDLE, 2=BUFFERING, 3=READY, 4=ENDED).
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeOnPlaybackStateChanged(
    _env: JNIEnv,
    _this: JObject,
    native_handle: jlong,
    state_value: jint,
) {
    tracing::debug!(
        "nativeOnPlaybackStateChanged: handle={}, state={}",
        native_handle,
        state_value
    );
    if let Some(state) = get_native_state(native_handle) {
        let mut state = state.lock();
        state.playback_state = state_value;
    }
}

/// JNI entry point for ExoPlayer error notification.
///
/// Called by ExoPlayerBridge when ExoPlayer encounters a playback error.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeOnError(
    mut env: JNIEnv,
    _this: JObject,
    native_handle: jlong,
    error_message: jni::objects::JString,
) {
    let error: String = env
        .get_string(&error_message)
        .map(|s| s.into())
        .unwrap_or_else(|_| "Unknown error".to_string());

    tracing::error!("nativeOnError: handle={}, error={}", native_handle, error);
    if let Some(state) = get_native_state(native_handle) {
        let mut state = state.lock();
        state.last_error = Some(error);
    }
}

/// JNI entry point for duration change notification.
///
/// Called by ExoPlayerBridge when the video duration becomes known.
#[no_mangle]
pub extern "C" fn Java_com_luminavideo_bridge_ExoPlayerBridge_nativeOnDurationChanged(
    _env: JNIEnv,
    _this: JObject,
    native_handle: jlong,
    duration_ms: jlong,
) {
    tracing::info!(
        "nativeOnDurationChanged: handle={}, duration_ms={}",
        native_handle,
        duration_ms
    );
    if let Some(state) = get_native_state(native_handle) {
        let mut state = state.lock();
        state.duration_ms = duration_ms;
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
    _this: JObject,
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
