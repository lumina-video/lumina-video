//! NDK AImageReader wrapper for Android video frame capture with sync fence support.
//!
//! This module provides a native (Rust) wrapper around Android's AImageReader API,
//! enabling access to `AImageReader_acquireLatestImageAsync()` which returns the
//! sync fence FD from the producer (MediaCodec). The fence FD is critical for
//! proper Vulkan synchronization when importing AHardwareBuffer.
//!
//! ## Why NDK instead of Java ImageReader?
//!
//! Java's `ImageReader.Image.getFence()` (API 33+) returns a `SyncFence` object
//! but does not expose the raw file descriptor. The NDK's `AImageReader` API
//! provides direct access to the fence FD via `acquire_latest_image_async()`.
//!
//! ## Usage
//!
//! ```ignore
//! // Create reader with GPU usage flags
//! let reader = NdkImageReaderBridge::new(width, height, max_images)?;
//!
//! // Get Java Surface to pass to ExoPlayer
//! let surface = reader.to_java_surface(&mut env)?;
//! env.call_method(&exoplayer, "setVideoSurface", "(Landroid/view/Surface;)V", &[surface.into()])?;
//!
//! // Set callback for frame arrival
//! reader.set_frame_callback(|frame| {
//!     // frame.fence_fd contains the sync fence from MediaCodec
//!     submit_to_vulkan(frame);
//! });
//! ```

#![cfg(all(target_os = "android", feature = "android-zero-copy"))]

use crate::media::android_video::AndroidVideoFrame;
use jni::objects::JObject;
use jni::JNIEnv;
use ndk::hardware_buffer::{HardwareBuffer, HardwareBufferUsage};
use ndk::media::image_reader::{AcquireResult, Image, ImageFormat, ImageReader};
use std::sync::mpsc::Sender;
use tracing::{debug, warn};

/// Bridge between NDK AImageReader and the lumina-video frame queue.
///
/// Manages an AImageReader configured for MediaCodec output with GPU usage flags,
/// and provides a Java Surface for ExoPlayer to render to.
pub struct NdkImageReaderBridge {
    /// The underlying NDK ImageReader
    reader: ImageReader,
    /// Player ID for frame routing
    player_id: u64,
}

impl NdkImageReaderBridge {
    /// Create a new NDK ImageReader configured for video decode output.
    ///
    /// # Arguments
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `max_images` - Maximum number of images in the buffer queue (typically 3)
    ///
    /// # Returns
    /// A new `NdkImageReaderBridge` or an error if creation fails.
    pub fn new(width: i32, height: i32, max_images: i32) -> Result<Self, NdkImageReaderError> {
        // Use PRIVATE format for MediaCodec output (YUV internally, opaque to app)
        // GPU_SAMPLED_IMAGE allows Vulkan to sample from the buffer
        let usage = HardwareBufferUsage::GPU_SAMPLED_IMAGE;

        let reader =
            ImageReader::new_with_usage(width, height, ImageFormat::PRIVATE, usage, max_images)
                .map_err(|e| NdkImageReaderError::CreationFailed(format!("{:?}", e)))?;

        debug!(
            "Created NDK ImageReader: {}x{} format=PRIVATE max_images={}",
            width, height, max_images
        );

        Ok(Self {
            reader,
            player_id: 0,
        })
    }

    /// Set the player ID for frame routing in multi-player scenarios.
    pub fn set_player_id(&mut self, player_id: u64) {
        self.player_id = player_id;
    }

    /// Convert the ImageReader's ANativeWindow to a Java Surface object.
    ///
    /// The returned Surface can be passed to ExoPlayer via `setVideoSurface()`.
    /// The Surface maintains a reference to the underlying ANativeWindow.
    ///
    /// # Safety
    /// Requires a valid JNI environment. The returned JObject is a local reference
    /// that must be used before the JNI call returns or converted to a global ref.
    pub fn to_java_surface<'local>(
        &self,
        env: &mut JNIEnv<'local>,
    ) -> Result<JObject<'local>, NdkImageReaderError> {
        let window = self
            .reader
            .window()
            .map_err(|e| NdkImageReaderError::WindowFailed(format!("{:?}", e)))?;

        // Convert ANativeWindow to Java Surface using NDK function
        // SAFETY:
        // 1. `env.get_raw()` returns a valid JNIEnv pointer from the jni crate
        // 2. `window.ptr()` returns a NonNull<ANativeWindow> from the ndk crate's
        //    NativeWindow wrapper, which is valid for the lifetime of `window`
        // 3. `ANativeWindow_toSurface` is an NDK function that creates a Java Surface
        //    object from the native window, returning a local JNI reference
        let surface_jobject = unsafe {
            let env_ptr = env.get_raw() as *mut jni::sys::JNIEnv;
            let window_ptr = window.ptr().as_ptr();
            ndk_sys::ANativeWindow_toSurface(env_ptr, window_ptr)
        };

        if surface_jobject.is_null() {
            return Err(NdkImageReaderError::SurfaceConversionFailed);
        }

        // Wrap in JObject - this is a local reference
        // SAFETY: surface_jobject is a valid non-null jobject from ANativeWindow_toSurface,
        // which returns a local reference valid until the JNI method returns
        Ok(unsafe { JObject::from_raw(surface_jobject) })
    }

    /// Set up the image available callback that sends frames to the channel.
    ///
    /// When a new frame is available from MediaCodec, this callback:
    /// 1. Acquires the image with `acquire_latest_image_async()` to get fence FD
    /// 2. Extracts the HardwareBuffer
    /// 3. Creates an AndroidVideoFrame with the fence FD
    /// 4. Sends it to the render thread via the channel
    pub fn set_frame_sender(
        &mut self,
        sender: Sender<AndroidVideoFrame>,
    ) -> Result<(), NdkImageReaderError> {
        let player_id = self.player_id;

        // The callback closure needs to be Send since it may be called from
        // the AImageReader's internal thread
        let callback = move |reader: &ImageReader| {
            Self::on_image_available(reader, &sender, player_id);
        };

        self.reader
            .set_image_listener(Box::new(callback))
            .map_err(|e| NdkImageReaderError::CallbackFailed(format!("{:?}", e)))?;

        debug!("Set NDK ImageReader callback for player {}", player_id);
        Ok(())
    }

    /// Internal callback invoked when a new image is available.
    fn on_image_available(
        reader: &ImageReader,
        sender: &Sender<AndroidVideoFrame>,
        player_id: u64,
    ) {
        // Use acquire_latest_image_async to get the fence FD
        // SAFETY: We must wait on the fence before accessing the image data
        let (image, fence_fd): (Image, i32) = match unsafe { reader.acquire_latest_image_async() } {
            Ok(AcquireResult::Image((img, Some(fence)))) => {
                // Take raw FD from OwnedFd (consumes ownership, Vulkan will close it)
                use std::os::fd::IntoRawFd;
                (img, fence.into_raw_fd())
            }
            Ok(AcquireResult::Image((img, None))) => {
                (img, -1) // No fence, already signaled
            }
            Ok(AcquireResult::NoBufferAvailable) => {
                return; // No frame ready, ignore
            }
            Ok(AcquireResult::MaxImagesAcquired) => {
                warn!("NDK ImageReader: max images acquired, dropping callback");
                return;
            }
            Err(e) => {
                warn!("NDK ImageReader acquire failed: {:?}", e);
                return;
            }
        };

        // Get HardwareBuffer from the image
        let hardware_buffer: HardwareBuffer = match image.hardware_buffer() {
            Ok(hb) => hb,
            Err(e) => {
                warn!("Failed to get HardwareBuffer from AImage: {:?}", e);
                return;
            }
        };

        // Get timestamp
        let timestamp_ns = image.timestamp().unwrap_or(0);

        // Get dimensions
        let width = image.width().unwrap_or(0) as u32;
        let height = image.height().unwrap_or(0) as u32;

        // Acquire our own reference to the HardwareBuffer
        // Use the ndk crate's as_ptr() to get the raw AHardwareBuffer pointer
        let ahb_ptr = unsafe {
            let ptr = hardware_buffer.as_ptr();
            // Acquire an additional reference since we're passing this to another consumer
            ndk_sys::AHardwareBuffer_acquire(ptr);
            ptr as *mut std::ffi::c_void
        };

        // Query format from the buffer descriptor
        let format = {
            let mut desc: ndk_sys::AHardwareBuffer_Desc = unsafe { std::mem::zeroed() };
            unsafe {
                ndk_sys::AHardwareBuffer_describe(
                    ahb_ptr as *const ndk_sys::AHardwareBuffer,
                    &mut desc,
                );
            }
            desc.format
        };

        // Create frame with fence FD - this is the key improvement!
        let frame = AndroidVideoFrame {
            buffer: ahb_ptr,
            width,
            height,
            timestamp_ns,
            format,
            player_id,
            fence_fd, // Now we have the actual fence from the producer!
        };

        // Send to render thread
        if let Err(e) = sender.send(frame) {
            debug!("Frame channel closed: {:?}", e);
        }

        // The AImage is dropped here, but we've acquired our own HardwareBuffer reference
    }

    /// Get the width of the ImageReader.
    pub fn width(&self) -> Result<i32, NdkImageReaderError> {
        self.reader
            .width()
            .map_err(|e| NdkImageReaderError::QueryFailed(format!("{:?}", e)))
    }

    /// Get the height of the ImageReader.
    pub fn height(&self) -> Result<i32, NdkImageReaderError> {
        self.reader
            .height()
            .map_err(|e| NdkImageReaderError::QueryFailed(format!("{:?}", e)))
    }
}

// NOTE: We don't implement Drop because the ndk crate's ImageReader handles cleanup.
// The ImageReader::drop calls AImageReader_delete which unregisters listeners.
// The ndk crate stores the callback in an Option that's dropped with the ImageReader,
// so the boxed closure lifetime is tied to the ImageReader's lifetime.

/// Errors that can occur with NDK ImageReader operations.
#[derive(Debug, Clone)]
pub enum NdkImageReaderError {
    /// Failed to create the ImageReader
    CreationFailed(String),
    /// Failed to get the ANativeWindow
    WindowFailed(String),
    /// Failed to convert ANativeWindow to Java Surface
    SurfaceConversionFailed,
    /// Failed to set the image callback
    CallbackFailed(String),
    /// Failed to query ImageReader properties
    QueryFailed(String),
}

impl std::fmt::Display for NdkImageReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreationFailed(s) => write!(f, "ImageReader creation failed: {}", s),
            Self::WindowFailed(s) => write!(f, "Failed to get window: {}", s),
            Self::SurfaceConversionFailed => write!(f, "Failed to convert to Java Surface"),
            Self::CallbackFailed(s) => write!(f, "Failed to set callback: {}", s),
            Self::QueryFailed(s) => write!(f, "Query failed: {}", s),
        }
    }
}

impl std::error::Error for NdkImageReaderError {}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require Android environment and cannot run on host
    // They serve as documentation for expected behavior
}
