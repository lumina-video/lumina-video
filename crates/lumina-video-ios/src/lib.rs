//! C FFI layer for lumina-video-core (iOS/Swift integration).
//!
//! Provides `#[no_mangle] pub extern "C"` entry points matching
//! `include/LuminaVideo.h`. All functions are thread-safe.
//!
//! See `docs/ios-ffi-contract.md` for the full contract.

// FFI functions intentionally take raw pointers without `unsafe` on the fn signature.
// Safety is enforced inside each function body via null checks + ffi_boundary().
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::macro_metavars_in_unsafe)]

pub mod error;
pub mod handle;
pub mod safety;

use std::ffi::CStr;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use lumina_video_core::player::CorePlayer;
use lumina_video_core::video::VideoState;

use crate::error::LuminaError;
use crate::handle::{LuminaFrame, LuminaPlayer};
use crate::safety::ffi_boundary;

// =========================================================================
// Player lifecycle
// =========================================================================

/// Creates a new video player for the given URL.
///
/// # Safety
/// - `url` must be a valid null-terminated UTF-8 C string.
/// - `out_player` must be a valid non-null pointer to a `*mut LuminaPlayer`.
#[no_mangle]
pub extern "C" fn lumina_player_create(
    url: *const std::os::raw::c_char,
    out_player: *mut *mut LuminaPlayer,
) -> i32 {
    ffi_boundary(|| {
        if url.is_null() || out_player.is_null() {
            return Err(LuminaError::NullPtr);
        }

        let url_str = unsafe { CStr::from_ptr(url) }
            .to_str()
            .map_err(|_| LuminaError::InvalidUrl)?;

        if url_str.is_empty() {
            return Err(LuminaError::InvalidUrl);
        }

        let mut core = CorePlayer::new(url_str);
        core.init_decoder();

        let player = Box::new(LuminaPlayer {
            core: Arc::new(Mutex::new(core)),
        });

        unsafe {
            *out_player = Box::into_raw(player);
        }
        Ok(())
    })
}

/// Destroys a video player and frees all resources.
///
/// # Safety
/// - `player` must be a valid non-null pointer to a `*mut LuminaPlayer`.
/// - After return, `*player` is NULL.
#[no_mangle]
pub extern "C" fn lumina_player_destroy(player: *mut *mut LuminaPlayer) -> i32 {
    ffi_boundary(|| {
        if player.is_null() {
            return Err(LuminaError::NullPtr);
        }

        let player_ptr = unsafe { *player };
        if player_ptr.is_null() {
            return Ok(()); // Already destroyed — no-op
        }

        // Null out the caller's pointer first
        unsafe {
            *player = std::ptr::null_mut();
        }

        // Reconstruct and drop
        let _player = unsafe { Box::from_raw(player_ptr) };
        // Drop: Arc<Mutex<CorePlayer>> runs CorePlayer::drop() which stops threads

        Ok(())
    })
}

// =========================================================================
// Playback control
// =========================================================================

/// Starts or resumes playback.
///
/// # Safety
/// `player` must be a valid non-null `LuminaPlayer` pointer.
#[no_mangle]
pub extern "C" fn lumina_player_play(player: *mut LuminaPlayer) -> i32 {
    ffi_boundary(|| {
        let player = check_not_null!(player);
        player.with_core(|core| {
            // Complete init if still pending
            core.check_init_complete();
            core.play();
        });
        Ok(())
    })
}

/// Pauses playback.
///
/// # Safety
/// `player` must be a valid non-null `LuminaPlayer` pointer.
#[no_mangle]
pub extern "C" fn lumina_player_pause(player: *mut LuminaPlayer) -> i32 {
    ffi_boundary(|| {
        let player = check_not_null!(player);
        player.with_core(|core| core.pause());
        Ok(())
    })
}

/// Seeks to a position in seconds.
///
/// # Safety
/// `player` must be a valid non-null `LuminaPlayer` pointer.
#[no_mangle]
pub extern "C" fn lumina_player_seek(player: *mut LuminaPlayer, position_secs: f64) -> i32 {
    ffi_boundary(|| {
        let player = check_not_null!(player);
        let position = Duration::from_secs_f64(position_secs.max(0.0));
        player.with_core(|core| core.seek(position));
        Ok(())
    })
}

// =========================================================================
// State queries
// =========================================================================

/// Returns the current playback state.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_state(player: *const LuminaPlayer) -> i32 {
    if player.is_null() {
        return 5; // LUMINA_STATE_ERROR
    }
    let player = unsafe { &*player };
    let state = player.core.lock().state().clone();
    match state {
        VideoState::Loading => 0,
        VideoState::Ready => 1,
        VideoState::Playing { .. } => 2,
        VideoState::Paused { .. } => 3,
        VideoState::Ended => 4,
        VideoState::Error(_) => 5,
        VideoState::Buffering { .. } => 2, // Treat buffering as playing
    }
}

/// Returns the current playback position in seconds.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_position(player: *const LuminaPlayer) -> f64 {
    if player.is_null() {
        return 0.0;
    }
    let player = unsafe { &*player };
    player.core.lock().position().as_secs_f64()
}

/// Returns the video duration in seconds, or -1.0 if unknown.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_duration(player: *const LuminaPlayer) -> f64 {
    if player.is_null() {
        return -1.0;
    }
    let player = unsafe { &*player };
    player
        .core
        .lock()
        .duration()
        .map(|d| d.as_secs_f64())
        .unwrap_or(-1.0)
}

// =========================================================================
// Frame retrieval
// =========================================================================

/// Polls for the next decoded video frame.
///
/// Returns NULL if no frame is ready. Caller owns the returned frame
/// and must call `lumina_frame_release()`.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_poll_frame(player: *mut LuminaPlayer) -> *mut LuminaFrame {
    if player.is_null() {
        return std::ptr::null_mut();
    }
    let player = unsafe { &*player };
    let mut core = player.core.lock();

    // Drive init forward if needed
    core.check_init_complete();

    match core.poll_frame() {
        Some(frame) => Box::into_raw(Box::new(LuminaFrame { frame })),
        None => std::ptr::null_mut(),
    }
}

// =========================================================================
// Frame accessors
// =========================================================================

/// Returns the frame width in pixels.
///
/// # Safety
/// `frame` must be a valid `LuminaFrame` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_frame_width(frame: *const LuminaFrame) -> u32 {
    if frame.is_null() {
        return 0;
    }
    let frame = unsafe { &*frame };
    let (w, _) = frame.frame.dimensions();
    w
}

/// Returns the frame height in pixels.
///
/// # Safety
/// `frame` must be a valid `LuminaFrame` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_frame_height(frame: *const LuminaFrame) -> u32 {
    if frame.is_null() {
        return 0;
    }
    let frame = unsafe { &*frame };
    let (_, h) = frame.frame.dimensions();
    h
}

/// Returns the IOSurface for zero-copy Metal rendering.
///
/// Returns NULL if the frame is CPU-only or if on a non-Apple platform.
///
/// # Safety
/// `frame` must be a valid `LuminaFrame` pointer (or NULL).
/// The returned IOSurface is valid only while the LuminaFrame is alive.
#[no_mangle]
pub extern "C" fn lumina_frame_iosurface(frame: *const LuminaFrame) -> *mut std::ffi::c_void {
    if frame.is_null() {
        return std::ptr::null_mut();
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let frame = unsafe { &*frame };
        if let Some(surface) = frame.frame.frame.as_macos_surface() {
            return surface.io_surface;
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let _ = frame;

    std::ptr::null_mut()
}

/// Releases a decoded video frame and frees its resources.
///
/// No-op if frame is NULL.
///
/// # Safety
/// `frame` must be a valid `LuminaFrame` pointer (or NULL).
/// After this call, the pointer is invalid.
#[no_mangle]
pub extern "C" fn lumina_frame_release(frame: *mut LuminaFrame) {
    if frame.is_null() {
        return;
    }
    let _frame = unsafe { Box::from_raw(frame) };
    // Drop frees the frame and its GPU surfaces
}
