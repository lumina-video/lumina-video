//! FFI boundary safety utilities.
//!
//! All FFI entry points wrap their body in [`ffi_boundary`] which:
//! 1. Catches panics via `std::panic::catch_unwind()`
//! 2. Converts `Result<(), LuminaError>` to raw `i32`

use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::error::LuminaError;

/// Wraps an FFI entry point body with panic catching.
///
/// Returns `LUMINA_ERROR_INTERNAL` if the closure panics.
pub fn ffi_boundary<F>(f: F) -> i32
where
    F: FnOnce() -> Result<(), LuminaError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => LuminaError::Ok.as_raw(),
        Ok(Err(e)) => e.as_raw(),
        Err(_panic) => {
            tracing::error!("FFI: caught Rust panic at FFI boundary");
            LuminaError::Internal.as_raw()
        }
    }
}

/// Wraps a non-Result FFI entry point with panic catching.
///
/// Returns `default` if the closure panics.
pub fn ffi_boundary_or<T, F>(default: T, f: F) -> T
where
    F: FnOnce() -> T,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(val) => val,
        Err(_panic) => {
            tracing::error!("FFI: caught Rust panic at FFI boundary");
            default
        }
    }
}

/// Validates a pointer is non-null, returning a reference.
///
/// # Safety
/// The pointer must be valid and properly aligned if non-null.
#[macro_export]
macro_rules! check_not_null {
    ($ptr:expr) => {
        if $ptr.is_null() {
            return Err($crate::error::LuminaError::NullPtr);
        } else {
            unsafe { &*$ptr }
        }
    };
}

/// Validates a pointer is non-null, returning a mutable reference.
///
/// # Safety
/// The pointer must be valid and properly aligned if non-null.
#[macro_export]
macro_rules! check_not_null_mut {
    ($ptr:expr) => {
        if $ptr.is_null() {
            return Err($crate::error::LuminaError::NullPtr);
        } else {
            unsafe { &mut *$ptr }
        }
    };
}
