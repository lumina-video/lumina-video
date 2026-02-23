//! FFI conformance tests for lumina-video-ios.
//!
//! Tests are gated to macOS/iOS since the underlying decoders require
//! those platforms. On Linux, the crate compiles but tests are skipped.

#![allow(unused_unsafe)]

use lumina_video_ios::error::LuminaError;
use lumina_video_ios::handle::{LuminaFrame, LuminaPlayer};
use lumina_video_ios::LuminaDiagnostics;
use std::ptr;

// Import FFI functions
extern "C" {
    fn lumina_player_create(url: *const i8, out_player: *mut *mut LuminaPlayer) -> i32;
    fn lumina_player_destroy(player: *mut *mut LuminaPlayer) -> i32;
    fn lumina_player_play(player: *mut LuminaPlayer) -> i32;
    fn lumina_player_pause(player: *mut LuminaPlayer) -> i32;
    fn lumina_player_seek(player: *mut LuminaPlayer, position_secs: f64) -> i32;
    fn lumina_player_state(player: *const LuminaPlayer) -> i32;
    fn lumina_player_position(player: *const LuminaPlayer) -> f64;
    fn lumina_player_duration(player: *const LuminaPlayer) -> f64;
    fn lumina_player_poll_frame(player: *mut LuminaPlayer) -> *mut LuminaFrame;
    fn lumina_frame_width(frame: *const LuminaFrame) -> u32;
    fn lumina_frame_height(frame: *const LuminaFrame) -> u32;
    fn lumina_frame_iosurface(frame: *const LuminaFrame) -> *mut std::ffi::c_void;
    fn lumina_frame_release(frame: *mut LuminaFrame);
    fn lumina_player_set_muted(player: *mut LuminaPlayer, muted: bool) -> i32;
    fn lumina_player_is_muted(player: *const LuminaPlayer) -> bool;
    fn lumina_player_set_volume(player: *mut LuminaPlayer, volume: i32) -> i32;
    fn lumina_player_volume(player: *const LuminaPlayer) -> i32;
    fn lumina_diagnostics_snapshot(out: *mut LuminaDiagnostics) -> i32;
}

const LUMINA_OK: i32 = LuminaError::Ok as i32;
const LUMINA_ERROR_NULL_PTR: i32 = LuminaError::NullPtr as i32;
const LUMINA_ERROR_INVALID_URL: i32 = LuminaError::InvalidUrl as i32;
const LUMINA_STATE_LOADING: i32 = 0;
const LUMINA_STATE_ERROR: i32 = 5;

// =========================================================================
// Lifecycle tests
// =========================================================================

#[test]
fn create_and_destroy() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();

        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);
        assert!(!player.is_null());

        let err = lumina_player_destroy(&mut player);
        assert_eq!(err, LUMINA_OK);
        assert!(player.is_null());
    }
}

#[test]
fn destroy_null_pointer_to_pointer() {
    unsafe {
        // Passing NULL for the pointer-to-pointer should error
        let err = lumina_player_destroy(ptr::null_mut());
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn destroy_already_null_is_noop() {
    unsafe {
        // *player is NULL — should be a no-op
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_destroy(&mut player);
        assert_eq!(err, LUMINA_OK);
    }
}

#[test]
fn double_destroy_is_safe() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();

        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        // First destroy
        let err = lumina_player_destroy(&mut player);
        assert_eq!(err, LUMINA_OK);
        assert!(player.is_null());

        // Second destroy — *player is now NULL, should be no-op
        let err = lumina_player_destroy(&mut player);
        assert_eq!(err, LUMINA_OK);
    }
}

// =========================================================================
// NULL safety tests
// =========================================================================

#[test]
fn create_null_url() {
    unsafe {
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(ptr::null(), &mut player);
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
        assert!(player.is_null());
    }
}

#[test]
fn create_null_out_player() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let err = lumina_player_create(url, ptr::null_mut());
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn create_empty_url() {
    unsafe {
        let url = b"\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_ERROR_INVALID_URL);
        assert!(player.is_null());
    }
}

#[test]
fn play_null_player() {
    unsafe {
        let err = lumina_player_play(ptr::null_mut());
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn pause_null_player() {
    unsafe {
        let err = lumina_player_pause(ptr::null_mut());
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn seek_null_player() {
    unsafe {
        let err = lumina_player_seek(ptr::null_mut(), 1.0);
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn seek_nan_and_infinity_return_invalid_arg() {
    let err_invalid_arg: i32 = LuminaError::InvalidArgument as i32;
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        let err = lumina_player_seek(player, f64::NAN);
        assert_eq!(err, err_invalid_arg, "NAN seek should return INVALID_ARG");

        let err = lumina_player_seek(player, f64::INFINITY);
        assert_eq!(
            err, err_invalid_arg,
            "INFINITY seek should return INVALID_ARG"
        );

        let err = lumina_player_seek(player, f64::NEG_INFINITY);
        assert_eq!(
            err, err_invalid_arg,
            "NEG_INFINITY seek should return INVALID_ARG"
        );

        lumina_player_destroy(&mut player);
    }
}

#[test]
fn state_null_returns_error() {
    unsafe {
        let state = lumina_player_state(ptr::null());
        assert_eq!(state, LUMINA_STATE_ERROR);
    }
}

#[test]
fn position_null_returns_zero() {
    unsafe {
        let pos = lumina_player_position(ptr::null());
        assert_eq!(pos, 0.0);
    }
}

#[test]
fn duration_null_returns_negative() {
    unsafe {
        let dur = lumina_player_duration(ptr::null());
        assert_eq!(dur, -1.0);
    }
}

#[test]
fn poll_frame_null_returns_null() {
    unsafe {
        let frame = lumina_player_poll_frame(ptr::null_mut());
        assert!(frame.is_null());
    }
}

// =========================================================================
// Frame accessor NULL safety
// =========================================================================

#[test]
fn frame_width_null() {
    unsafe {
        assert_eq!(lumina_frame_width(ptr::null()), 0);
    }
}

#[test]
fn frame_height_null() {
    unsafe {
        assert_eq!(lumina_frame_height(ptr::null()), 0);
    }
}

#[test]
fn frame_iosurface_null() {
    unsafe {
        assert!(lumina_frame_iosurface(ptr::null()).is_null());
    }
}

#[test]
fn frame_release_null_is_noop() {
    unsafe {
        lumina_frame_release(ptr::null_mut());
        // No crash = success
    }
}

// =========================================================================
// State query tests
// =========================================================================

#[test]
fn initial_state_is_loading() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();

        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        let state = lumina_player_state(player);
        assert_eq!(state, LUMINA_STATE_LOADING);

        lumina_player_destroy(&mut player);
    }
}

#[test]
fn initial_position_is_zero() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();

        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        let pos = lumina_player_position(player);
        assert_eq!(pos, 0.0);

        lumina_player_destroy(&mut player);
    }
}

// =========================================================================
// Stress tests
// =========================================================================

#[test]
fn rapid_create_destroy_100x() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        for _ in 0..100 {
            let mut player: *mut LuminaPlayer = ptr::null_mut();
            let err = lumina_player_create(url, &mut player);
            assert_eq!(err, LUMINA_OK);
            let err = lumina_player_destroy(&mut player);
            assert_eq!(err, LUMINA_OK);
        }
    }
}

#[test]
fn play_pause_seek_without_init() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        // These should not crash even before init completes
        let _ = lumina_player_play(player);
        let _ = lumina_player_pause(player);
        let _ = lumina_player_seek(player, 5.0);
        let _ = lumina_player_state(player);
        let _ = lumina_player_position(player);
        let _ = lumina_player_duration(player);
        let _ = lumina_player_poll_frame(player);

        lumina_player_destroy(&mut player);
    }
}

#[test]
fn thread_safety_create_destroy_different_thread() {
    use std::thread;

    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        // Send to another thread for destroy
        let player_ptr = player as usize;
        let handle = thread::spawn(move || {
            let mut player = player_ptr as *mut LuminaPlayer;
            let err = lumina_player_destroy(&mut player);
            assert_eq!(err, LUMINA_OK);
        });
        handle.join().unwrap();
    }
}

// =========================================================================
// Audio control — NULL safety
// =========================================================================

#[test]
fn set_muted_null_player() {
    unsafe {
        let err = lumina_player_set_muted(ptr::null_mut(), true);
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn is_muted_null_returns_true() {
    unsafe {
        // Safe default: muted
        assert!(lumina_player_is_muted(ptr::null()));
    }
}

#[test]
fn set_volume_null_player() {
    unsafe {
        let err = lumina_player_set_volume(ptr::null_mut(), 50);
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[test]
fn volume_null_returns_zero() {
    unsafe {
        assert_eq!(lumina_player_volume(ptr::null()), 0);
    }
}

// =========================================================================
// Audio control — functional round-trips
// =========================================================================

#[test]
fn mute_toggle_round_trip() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        // Initially not muted (AudioHandle default)
        assert!(!lumina_player_is_muted(player));

        // Mute
        let err = lumina_player_set_muted(player, true);
        assert_eq!(err, LUMINA_OK);
        assert!(lumina_player_is_muted(player));

        // Unmute
        let err = lumina_player_set_muted(player, false);
        assert_eq!(err, LUMINA_OK);
        assert!(!lumina_player_is_muted(player));

        lumina_player_destroy(&mut player);
    }
}

#[test]
fn volume_set_get_round_trip() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        let err = lumina_player_set_volume(player, 75);
        assert_eq!(err, LUMINA_OK);
        assert_eq!(lumina_player_volume(player), 75);

        let err = lumina_player_set_volume(player, 0);
        assert_eq!(err, LUMINA_OK);
        assert_eq!(lumina_player_volume(player), 0);

        lumina_player_destroy(&mut player);
    }
}

#[test]
fn volume_clamping() {
    unsafe {
        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        // Above 100 → clamp to 100
        let err = lumina_player_set_volume(player, 200);
        assert_eq!(err, LUMINA_OK);
        assert_eq!(lumina_player_volume(player), 100);

        // Negative → clamp to 0 (critical: must clamp in i32 domain before cast to u32)
        let err = lumina_player_set_volume(player, -50);
        assert_eq!(err, LUMINA_OK);
        assert_eq!(lumina_player_volume(player), 0);

        lumina_player_destroy(&mut player);
    }
}

// =========================================================================
// Diagnostics
// =========================================================================

#[test]
fn diagnostics_snapshot_null() {
    unsafe {
        let err = lumina_diagnostics_snapshot(ptr::null_mut());
        assert_eq!(err, LUMINA_ERROR_NULL_PTR);
    }
}

#[cfg(debug_assertions)]
#[test]
fn diagnostics_tracks_player_lifecycle() {
    unsafe {
        let mut before = std::mem::zeroed::<LuminaDiagnostics>();
        lumina_diagnostics_snapshot(&mut before);

        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        let mut player: *mut LuminaPlayer = ptr::null_mut();
        let err = lumina_player_create(url, &mut player);
        assert_eq!(err, LUMINA_OK);

        let mut during = std::mem::zeroed::<LuminaDiagnostics>();
        lumina_diagnostics_snapshot(&mut during);
        assert!(during.players_created > before.players_created);
        assert!(during.players_live > before.players_live);

        lumina_player_destroy(&mut player);

        let mut after = std::mem::zeroed::<LuminaDiagnostics>();
        lumina_diagnostics_snapshot(&mut after);
        assert!(after.players_destroyed > before.players_destroyed);
    }
}

#[cfg(debug_assertions)]
#[test]
fn diagnostics_no_leak_after_100_cycles() {
    unsafe {
        let mut before = std::mem::zeroed::<LuminaDiagnostics>();
        lumina_diagnostics_snapshot(&mut before);
        let initial_live = before.players_live;

        let url = b"https://example.com/test.mp4\0".as_ptr() as *const i8;
        for _ in 0..100 {
            let mut player: *mut LuminaPlayer = ptr::null_mut();
            let err = lumina_player_create(url, &mut player);
            assert_eq!(err, LUMINA_OK);
            let err = lumina_player_destroy(&mut player);
            assert_eq!(err, LUMINA_OK);
        }

        let mut after = std::mem::zeroed::<LuminaDiagnostics>();
        lumina_diagnostics_snapshot(&mut after);
        // No leaked players
        assert_eq!(after.players_live, initial_live);
    }
}
