//! FFI handle types.
//!
//! [`LuminaPlayer`] wraps `CorePlayer` with a `Mutex` for thread-safe FFI access.
//! [`LuminaFrame`] wraps a decoded `VideoFrame` (immutable after creation).

use std::sync::Arc;

use parking_lot::Mutex;

use lumina_video_core::player::CorePlayer;
use lumina_video_core::video::VideoFrame;
#[cfg(feature = "moq")]
use lumina_video_core::video::{VideoDecoderBackend, VideoError};

/// State for async MoQ decoder initialization.
///
/// `MoqDecoder::new_with_config()` blocks while connecting to the relay, so we
/// run it on a background thread and poll for completion from FFI entry points.
#[cfg(feature = "moq")]
pub(crate) struct MoqInitState {
    pub rx: std::sync::mpsc::Receiver<Result<Box<dyn VideoDecoderBackend + Send>, VideoError>>,
    pub url: String,
}

/// Opaque player handle exposed via FFI.
///
/// All access is serialized by the internal mutex. Each FFI call acquires the
/// lock exactly once, performs the operation, and releases. No nested locking.
pub struct LuminaPlayer {
    pub(crate) core: Arc<Mutex<CorePlayer>>,
    /// Pending MoQ decoder init (video-only, no audio).
    #[cfg(feature = "moq")]
    pub(crate) moq_init: Mutex<Option<MoqInitState>>,
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

    /// Polls MoQ decoder initialization to completion.
    ///
    /// Called from FFI entry points before accessing CorePlayer. When the
    /// background thread finishes, replaces the empty CorePlayer shell with
    /// one wrapping the live MoQ decoder.
    #[cfg(feature = "moq")]
    pub(crate) fn try_complete_moq_init(&self) {
        use lumina_video_core::video::VideoState;

        let mut init = self.moq_init.lock();
        let Some(ref state) = *init else { return };

        match state.rx.try_recv() {
            Ok(Ok(decoder)) => {
                let url = state.url.clone();
                *init = None;
                drop(init);
                let new_core = CorePlayer::with_decoder(url, decoder);
                *self.core.lock() = new_core;
                tracing::info!("MoQ decoder initialized successfully (video-only)");
            }
            Ok(Err(e)) => {
                tracing::error!("MoQ decoder init failed: {e}");
                *init = None;
                drop(init);
                self.core
                    .lock()
                    .set_state(VideoState::Error(e));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Still initializing — no-op
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                tracing::error!("MoQ init thread crashed (channel disconnected)");
                *init = None;
                drop(init);
                self.core.lock().set_state(VideoState::Error(
                    VideoError::Generic("MoQ init thread crashed".into()),
                ));
            }
        }
    }
}
