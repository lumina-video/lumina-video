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

/// Payload sent from the MoQ init thread when decoder creation completes.
#[cfg(feature = "moq")]
pub(crate) struct MoqInitResult {
    pub decoder: Box<dyn VideoDecoderBackend + Send>,
    pub stats: lumina_video::media::MoqStatsHandle,
}

/// State for async MoQ decoder initialization.
///
/// `MoqDecoder::new_with_config()` blocks while connecting to the relay, so we
/// run it on a background thread and poll for completion from FFI entry points.
#[cfg(feature = "moq")]
pub(crate) struct MoqInitState {
    pub rx: std::sync::mpsc::Receiver<Result<MoqInitResult, VideoError>>,
    pub url: String,
}

/// Opaque player handle exposed via FFI.
///
/// All access is serialized by the internal mutex. Each FFI call acquires the
/// lock exactly once, performs the operation, and releases. No nested locking.
pub struct LuminaPlayer {
    pub(crate) core: Arc<Mutex<CorePlayer>>,
    /// Pending MoQ decoder init.
    #[cfg(feature = "moq")]
    pub(crate) moq_init: Mutex<Option<MoqInitState>>,
    /// MoQ stats handle for audio late-binding (set after init completes).
    #[cfg(feature = "moq")]
    pub(crate) moq_stats: Mutex<Option<lumina_video::media::MoqStatsHandle>>,
    /// Whether the MoQ audio handle has been late-bound to the CorePlayer.
    #[cfg(feature = "moq")]
    pub(crate) moq_audio_bound: std::sync::atomic::AtomicBool,
    /// Whether the MoQ audio handle has been promoted to sync master
    /// (only after observing samples_played > 0, proving audio is advancing).
    #[cfg(feature = "moq")]
    pub(crate) moq_audio_sync_master: std::sync::atomic::AtomicBool,
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
            Ok(Ok(result)) => {
                let url = state.url.clone();
                *init = None;
                drop(init);

                // Store stats handle for audio late-binding
                *self.moq_stats.lock() = Some(result.stats);

                let mut new_core = CorePlayer::with_decoder(url, result.decoder);
                // No frame-rate pacing for MoQ — the deferred epoch rebase
                // handles burst consumption. A fixed pacing rate (e.g. 30fps)
                // causes drift when content fps differs (24fps BBB → 8ms/frame
                // → ~250ms/sec A/V drift). Wall-clock acceptance naturally
                // matches real-time for live frames after the rebase.
                //
                // Auto-play: MoQ live streams should start immediately.
                // Swift called play() on the old shell CorePlayer before init
                // completed — that play state is lost when we swap. Force-start
                // the new CorePlayer so the decode thread isn't stuck in PAUSED.
                let muted = new_core.audio_handle().is_muted();
                new_core.play_with_muted(muted);
                *self.core.lock() = new_core;
                tracing::info!("MoQ decoder initialized successfully (auto-playing)");
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

    /// Polls the MoQ audio handle and late-binds it to the CorePlayer.
    ///
    /// Two-phase bind:
    /// 1. **Metrics-only**: On first detection, start the epoch and bind for
    ///    metrics/mute/volume, but wall-clock still drives frame pacing.
    /// 2. **Sync master**: Once `samples_played > 0` (proving cpal passed the
    ///    ring buffer prefill gate and is actually consuming audio), promote to
    ///    audio-as-sync-master and re-anchor the scheduler clock to eliminate
    ///    the wall-clock→audio discontinuity.
    #[cfg(feature = "moq")]
    pub(crate) fn poll_moq_audio_handle(&self) {
        use std::sync::atomic::Ordering;

        let stats = self.moq_stats.lock();
        let Some(ref stats) = *stats else { return };

        if let Some(moq_ah) = stats.audio_handle() {
            let bound = self.moq_audio_bound.load(Ordering::Relaxed);
            let is_sync_master = self.moq_audio_sync_master.load(Ordering::Relaxed);

            if !bound {
                // Phase 1: First detection — bind as metrics-only.
                // Start the epoch so cpal begins consuming the ring buffer
                // (it needs to pass the 500ms prefill gate before samples_played
                // increments). Wall-clock still drives video pacing.
                let mut core = self.core.lock();
                let muted = core.audio_handle().is_muted();
                let volume = core.audio_handle().volume();
                moq_ah.set_muted(muted);
                moq_ah.set_volume(volume);
                moq_ah.set_available(true);
                moq_ah.start_playback_epoch();

                // Metrics-only: drift is measured but wall-clock drives pacing
                core.scheduler_mut()
                    .set_audio_handle_metrics_only(moq_ah.clone());

                // Replace the player-level handle (for UI mute/volume controls)
                core.set_audio_handle(moq_ah);
                drop(core);

                self.moq_audio_bound.store(true, Ordering::Relaxed);
                tracing::info!(
                    "MoQ audio: late-bound audio handle (metrics-only, epoch started)"
                );
            } else if !is_sync_master {
                // Phase 2: Wait for audio progress, then promote to sync master.
                // samples_played > 0 proves: epoch is set, ring buffer prefilled,
                // cpal callback is reading samples. Safe to pace video off audio.
                let samples = moq_ah.samples_played();
                if samples > 0 {
                    let audio_pos = moq_ah.position_for_sync();
                    if audio_pos > std::time::Duration::ZERO {
                        let mut core = self.core.lock();
                        // Promote: flip wall-clock → audio-as-sync-master
                        core.scheduler_mut()
                            .set_audio_handle(moq_ah.clone());
                        // Re-anchor the scheduler clock to the current audio
                        // position. This eliminates the discontinuity between
                        // where wall-clock had advanced to and where audio
                        // actually is.
                        core.scheduler_mut()
                            .resync_clock_to_audio();
                        drop(core);

                        self.moq_audio_sync_master.store(true, Ordering::Relaxed);
                        tracing::info!(
                            "MoQ audio: promoted to sync master (samples_played={}, audio_pos={:?})",
                            samples,
                            audio_pos,
                        );
                    }
                }
            }
        } else if self.moq_audio_bound.load(Ordering::Relaxed) {
            // Audio thread torn down — unbind
            let mut core = self.core.lock();
            let placeholder = lumina_video_core::audio::AudioHandle::new();
            placeholder.set_muted(core.audio_handle().is_muted());
            placeholder.set_volume(core.audio_handle().volume());
            core.set_audio_handle(placeholder);
            core.scheduler_mut().clear_audio_handle();
            drop(core);

            self.moq_audio_bound.store(false, Ordering::Relaxed);
            self.moq_audio_sync_master.store(false, Ordering::Relaxed);
            tracing::info!("MoQ audio: unbound stale audio handle");
        }
    }
}
