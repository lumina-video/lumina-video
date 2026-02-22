//! FFI handle types.
//!
//! [`LuminaPlayer`] wraps `CorePlayer` with a `Mutex` for thread-safe FFI access.
//! [`LuminaFrame`] wraps a decoded `VideoFrame` (immutable after creation).

use std::sync::Arc;

use parking_lot::Mutex;

use lumina_video_core::player::CorePlayer;
use lumina_video_core::video::VideoFrame;

/// Opaque player handle exposed via FFI.
///
/// All access is serialized by the internal mutex. Each FFI call acquires the
/// lock exactly once, performs the operation, and releases. No nested locking.
pub struct LuminaPlayer {
    pub(crate) core: Arc<Mutex<CorePlayer>>,
}

/// Opaque frame handle exposed via FFI.
///
/// Immutable after creation -- no synchronization needed.
/// Caller must free via `lumina_frame_release()`.
pub struct LuminaFrame {
    pub(crate) frame: VideoFrame,
}

impl LuminaPlayer {
    /// Runs a closure with exclusive access to the CorePlayer.
    pub fn with_core<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut CorePlayer) -> R,
    {
        let mut core = self.core.lock();
        f(&mut core)
    }
}
