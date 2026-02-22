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

pub mod diagnostics;
pub mod error;
pub mod handle;
pub mod safety;

use std::ffi::CStr;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use lumina_video_core::player::CorePlayer;
use lumina_video_core::video::VideoState;

use crate::error::LuminaError;
use crate::handle::{LuminaFrame, LuminaPlayer};
use crate::safety::{ffi_boundary, ffi_boundary_or};

// =========================================================================
// State constants (matching include/LuminaVideo.h)
// =========================================================================

const LUMINA_STATE_LOADING: i32 = 0;
const LUMINA_STATE_READY: i32 = 1;
const LUMINA_STATE_PLAYING: i32 = 2;
const LUMINA_STATE_PAUSED: i32 = 3;
const LUMINA_STATE_ENDED: i32 = 4;
const LUMINA_STATE_ERROR: i32 = 5;

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

        // Null out first so callers always see NULL on failure
        unsafe {
            *out_player = std::ptr::null_mut();
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

        let raw = Box::into_raw(player);
        diagnostics::register_player(raw as *const u8);
        #[cfg(debug_assertions)]
        diagnostics::record_player_created();

        unsafe {
            *out_player = raw;
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

        // Atomic swap: read the pointer and null it in one operation.
        // This prevents TOCTOU races where two threads could both read
        // a non-null pointer and attempt Box::from_raw on it.
        let atomic = unsafe { AtomicPtr::from_ptr(player) };
        let player_ptr = atomic.swap(std::ptr::null_mut(), Ordering::AcqRel);

        if player_ptr.is_null() {
            return Ok(()); // Already destroyed — no-op
        }

        // Check registry: unknown pointer → double-free attempt
        if !diagnostics::unregister_player(player_ptr as *const u8) {
            tracing::error!(
                "FFI: lumina_player_destroy called with unknown pointer {:?} (possible double-free)",
                player_ptr
            );
            return Err(LuminaError::Internal);
        }

        // Reconstruct and drop
        let _player = unsafe { Box::from_raw(player_ptr) };
        // Drop: Arc<Mutex<CorePlayer>> runs CorePlayer::drop() which stops threads

        #[cfg(debug_assertions)]
        diagnostics::record_player_destroyed();

        Ok(())
    })
}

// =========================================================================
// Playback control
// =========================================================================

/// Starts or resumes playback.
///
/// Preserves the current mute state — `CorePlayer::play()` would force unmute,
/// so we use `play_with_muted()` to keep whatever the caller set.
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
            // Preserve mute state: play() forces unmute, play_with_muted() respects it
            let muted = core.audio_handle().is_muted();
            core.play_with_muted(muted);
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
        if !position_secs.is_finite() || position_secs > u64::MAX as f64 {
            return Err(LuminaError::InvalidArgument);
        }
        let position = Duration::from_secs_f64(position_secs.max(0.0));
        player.with_core(|core| core.seek(position));
        Ok(())
    })
}

// =========================================================================
// Audio control
// =========================================================================

/// Sets the muted state.
///
/// # Safety
/// `player` must be a valid non-null `LuminaPlayer` pointer.
#[no_mangle]
pub extern "C" fn lumina_player_set_muted(player: *mut LuminaPlayer, muted: bool) -> i32 {
    ffi_boundary(|| {
        let player = check_not_null!(player);
        player.with_core(|core| {
            core.set_muted(muted);
        });
        Ok(())
    })
}

/// Returns the current muted state.
///
/// Returns `true` (safe default: muted) if player is NULL.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_is_muted(player: *const LuminaPlayer) -> bool {
    ffi_boundary_or(true, || {
        if player.is_null() {
            return true;
        }
        let player = unsafe { &*player };
        player.core.lock().audio_handle().is_muted()
    })
}

/// Sets the volume level (0-100).
///
/// Values outside 0-100 are clamped.
///
/// # Safety
/// `player` must be a valid non-null `LuminaPlayer` pointer.
#[no_mangle]
pub extern "C" fn lumina_player_set_volume(player: *mut LuminaPlayer, volume: i32) -> i32 {
    ffi_boundary(|| {
        let player = check_not_null!(player);
        // Clamp in i32 domain first to avoid negative → large unsigned wrap
        let clamped = volume.clamp(0, 100) as u32;
        player.with_core(|core| {
            core.set_volume(clamped);
        });
        Ok(())
    })
}

/// Returns the current volume level (0-100).
///
/// Returns 0 (silent) if player is NULL.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_volume(player: *const LuminaPlayer) -> i32 {
    ffi_boundary_or(0, || {
        if player.is_null() {
            return 0;
        }
        let player = unsafe { &*player };
        player.core.lock().audio_handle().volume() as i32
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
    ffi_boundary_or(LUMINA_STATE_ERROR, || {
        if player.is_null() {
            return LUMINA_STATE_ERROR;
        }
        let player = unsafe { &*player };
        let state = player.core.lock().state().clone();
        match state {
            VideoState::Loading => LUMINA_STATE_LOADING,
            VideoState::Ready => LUMINA_STATE_READY,
            VideoState::Playing { .. } => LUMINA_STATE_PLAYING,
            VideoState::Paused { .. } => LUMINA_STATE_PAUSED,
            VideoState::Ended => LUMINA_STATE_ENDED,
            VideoState::Error(_) => LUMINA_STATE_ERROR,
            VideoState::Buffering { .. } => LUMINA_STATE_PLAYING,
        }
    })
}

/// Returns the current playback position in seconds.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_position(player: *const LuminaPlayer) -> f64 {
    ffi_boundary_or(0.0, || {
        if player.is_null() {
            return 0.0;
        }
        let player = unsafe { &*player };
        player.core.lock().position().as_secs_f64()
    })
}

/// Returns the video duration in seconds, or -1.0 if unknown.
///
/// # Safety
/// `player` must be a valid `LuminaPlayer` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_player_duration(player: *const LuminaPlayer) -> f64 {
    ffi_boundary_or(-1.0, || {
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
    })
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
    ffi_boundary_or(std::ptr::null_mut(), || {
        if player.is_null() {
            return std::ptr::null_mut();
        }
        let player = unsafe { &*player };
        let frame = {
            let mut core = player.core.lock();
            // Drive init forward if needed
            core.check_init_complete();
            core.poll_frame()
        };
        // Lock dropped — boxing and diagnostics outside critical section
        match frame {
            Some(frame) => {
                let raw = Box::into_raw(Box::new(LuminaFrame { frame }));
                #[cfg(debug_assertions)]
                {
                    diagnostics::register_frame(raw as *const u8);
                    diagnostics::record_frame_created();
                }
                raw
            }
            None => std::ptr::null_mut(),
        }
    })
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
    ffi_boundary_or(0, || {
        if frame.is_null() {
            return 0;
        }
        let frame = unsafe { &*frame };
        let (w, _) = frame.frame.dimensions();
        w
    })
}

/// Returns the frame height in pixels.
///
/// # Safety
/// `frame` must be a valid `LuminaFrame` pointer (or NULL).
#[no_mangle]
pub extern "C" fn lumina_frame_height(frame: *const LuminaFrame) -> u32 {
    ffi_boundary_or(0, || {
        if frame.is_null() {
            return 0;
        }
        let frame = unsafe { &*frame };
        let (_, h) = frame.frame.dimensions();
        h
    })
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
    ffi_boundary_or(std::ptr::null_mut(), || {
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
    })
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
    ffi_boundary_or((), || {
        if frame.is_null() {
            return;
        }

        #[cfg(debug_assertions)]
        {
            if !diagnostics::unregister_frame(frame as *const u8) {
                tracing::error!(
                    "FFI: lumina_frame_release called with unknown pointer {:?} (possible double-free)",
                    frame
                );
                return; // Skip Box::from_raw on unknown pointer
            }
            diagnostics::record_frame_destroyed();
        }

        let _frame = unsafe { Box::from_raw(frame) };
        // Drop frees the frame and its GPU surfaces
    });
}

// =========================================================================
// Diagnostics
// =========================================================================

/// FFI diagnostics snapshot (matches `LuminaDiagnostics` in the C header).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct LuminaDiagnostics {
    /// Total players created over the process lifetime.
    pub players_created: u64,
    /// Total players destroyed over the process lifetime.
    pub players_destroyed: u64,
    /// Peak concurrent players.
    pub players_peak: u64,
    /// Current number of live players.
    pub players_live: u64,
    /// Total frames created over the process lifetime.
    pub frames_created: u64,
    /// Total frames released over the process lifetime.
    pub frames_destroyed: u64,
    /// Peak concurrent frames.
    pub frames_peak: u64,
    /// Current number of live frames.
    pub frames_live: u64,
    /// Total FFI calls made.
    pub ffi_calls: u64,
}

/// Fills a diagnostics snapshot.
///
/// All fields are zero in release builds (debug_assertions disabled).
///
/// # Safety
/// `out` must be a valid non-null pointer to `LuminaDiagnostics`.
#[no_mangle]
pub extern "C" fn lumina_diagnostics_snapshot(out: *mut LuminaDiagnostics) -> i32 {
    ffi_boundary(|| {
        if out.is_null() {
            return Err(LuminaError::NullPtr);
        }

        #[cfg(debug_assertions)]
        {
            let snap = diagnostics::snapshot();
            unsafe {
                *out = LuminaDiagnostics {
                    players_created: snap.players_created,
                    players_destroyed: snap.players_destroyed,
                    players_peak: snap.players_peak,
                    players_live: snap.players_live,
                    frames_created: snap.frames_created,
                    frames_destroyed: snap.frames_destroyed,
                    frames_peak: snap.frames_peak,
                    frames_live: snap.frames_live,
                    ffi_calls: snap.ffi_calls,
                };
            }
        }

        #[cfg(not(debug_assertions))]
        unsafe {
            *out = LuminaDiagnostics::default();
        }

        Ok(())
    })
}
