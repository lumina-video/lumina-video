//! MoQ audio pipeline: crossbeam handoff → symphonia AAC decode → rodio playback.
//!
//! This module is gated on `cfg(all(feature = "moq", any(target_os = "macos", target_os = "linux", target_os = "android")))`.
//! Android uses cpal/Oboe (same pipeline as desktop). Windows has no MoQ decoder yet.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;

use super::audio::{AudioConfig, AudioHandle, AudioPlayer, AudioSamples};
use super::moq_decoder::{MoqAudioShared, MoqAudioStatus};

/// A raw encoded audio frame received from MoQ transport, ready for AAC decoding.
pub(crate) struct MoqAudioFrame {
    /// Presentation timestamp in microseconds.
    pub timestamp_us: u64,
    /// Encoded AAC frame data.
    pub data: Bytes,
}

/// Sentinel error indicating the crossbeam channel is permanently closed.
pub(crate) struct ChannelClosed;

/// Result of a [`LiveEdgeSender::send`] attempt (excluding disconnect).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendResult {
    /// Frame was enqueued successfully.
    Sent,
    /// An older frame was evicted (or the new frame was dropped) to stay near live edge.
    Dropped,
}

/// Bounded crossbeam channel wrapper with best-effort live-edge policy.
///
/// When the channel is full, attempts to drain the oldest item and retry.
/// Under contention with the consumer, the drain may race — the policy goal
/// is "stay near live edge", not strict FIFO eviction.
pub(crate) struct LiveEdgeSender<T> {
    tx: crossbeam_channel::Sender<T>,
    rx_drain: crossbeam_channel::Receiver<T>,
}

impl<T> LiveEdgeSender<T> {
    /// Creates a new sender. Both `tx` and `rx_drain` must be from the same channel.
    pub fn new(tx: crossbeam_channel::Sender<T>, rx_drain: crossbeam_channel::Receiver<T>) -> Self {
        Self { tx, rx_drain }
    }

    /// Sends with best-effort live-edge policy. Never blocks.
    ///
    /// Returns `Ok(SendResult::Sent)` on direct enqueue,
    /// `Ok(SendResult::Dropped)` when the full-path was hit (an older frame
    /// was evicted or the new frame was dropped in a benign race),
    /// `Err(ChannelClosed)` if the channel is permanently disconnected.
    pub fn send(&self, item: T) -> Result<SendResult, ChannelClosed> {
        match self.tx.try_send(item) {
            Ok(()) => Ok(SendResult::Sent),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(ChannelClosed),
            Err(crossbeam_channel::TrySendError::Full(item)) => {
                let _ = self.rx_drain.try_recv(); // best-effort drain (race ok)
                match self.tx.try_send(item) {
                    Ok(()) => Ok(SendResult::Dropped), // evicted old frame to make room
                    Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(ChannelClosed),
                    Err(crossbeam_channel::TrySendError::Full(_)) => Ok(SendResult::Dropped), // benign race-drop
                }
            }
        }
    }
}

/// AAC-LC decoder using symphonia, producing interleaved f32 `AudioSamples`.
struct SymphoniaAacDecoder {
    decoder: Box<dyn symphonia_core::codecs::Decoder>,
    sample_rate: u32,
    channels: u16,
}

impl SymphoniaAacDecoder {
    /// Creates a new AAC-LC decoder with the given parameters.
    ///
    /// `description` is the AudioSpecificConfig bytes from the catalog, if present.
    fn new(sample_rate: u32, channels: u32, description: Option<&Bytes>) -> Result<Self, String> {
        use symphonia_core::codecs::{
            CodecParameters, Decoder as _, DecoderOptions, CODEC_TYPE_AAC,
        };

        let mut params = CodecParameters::new();
        params
            .for_codec(CODEC_TYPE_AAC)
            .with_sample_rate(sample_rate)
            .with_channels(
                symphonia_core::audio::Channels::from_bits(
                    // stereo = front-left + front-right
                    if channels >= 2 { 0x3 } else { 0x4 }, // 0x4 = front-centre (mono)
                )
                .unwrap_or(
                    symphonia_core::audio::Channels::FRONT_LEFT
                        | symphonia_core::audio::Channels::FRONT_RIGHT,
                ),
            );

        if let Some(desc) = description {
            params.with_extra_data(desc.to_vec().into_boxed_slice());
        }

        let decoder = symphonia_codec_aac::AacDecoder::try_new(&params, &DecoderOptions::default())
            .map_err(|e| format!("Failed to create AAC decoder: {e}"))?;

        Ok(Self {
            decoder: Box::new(decoder),
            sample_rate,
            channels: channels.min(2) as u16,
        })
    }

    /// Decodes a single AAC frame into interleaved f32 samples.
    fn decode_frame(&mut self, data: &[u8], timestamp_us: u64) -> Result<AudioSamples, String> {
        use symphonia_core::formats::Packet;

        let packet = Packet::new_from_slice(0, 0, 0, data);
        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|e| format!("AAC decode error: {e}"))?;

        let spec = *decoded.spec();
        let num_channels = spec.channels.count();
        let num_frames = decoded.frames();

        if num_frames == 0 || num_channels == 0 {
            return Err("Empty decoded buffer".to_string());
        }

        let mut sample_buf =
            symphonia_core::audio::SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let interleaved = sample_buf.samples().to_vec();

        Ok(AudioSamples {
            data: interleaved,
            sample_rate: self.sample_rate,
            channels: self.channels,
            pts: Duration::from_micros(timestamp_us),
        })
    }
}

/// Owns the audio decode/playback thread. Signals stop on drop and joins.
pub(crate) struct MoqAudioThread {
    handle: Option<std::thread::JoinHandle<()>>,
    stop_flag: Arc<AtomicBool>,
    audio_shared: Arc<MoqAudioShared>,
}

impl MoqAudioThread {
    /// Spawns the audio thread. Returns `Err` if the OS refuses thread creation.
    pub fn spawn(
        audio_rx: crossbeam_channel::Receiver<MoqAudioFrame>,
        sample_rate: u32,
        channels: u32,
        description: Option<Bytes>,
        audio_handle: AudioHandle,
        audio_shared: Arc<MoqAudioShared>,
    ) -> Result<Self, String> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();
        let audio_handle_clone = audio_handle.clone();
        let audio_shared_clone = audio_shared.clone();

        let handle = std::thread::Builder::new()
            .name("moq-audio".into())
            .spawn(move || {
                moq_audio_thread_main(
                    audio_rx,
                    sample_rate,
                    channels,
                    description,
                    audio_handle_clone,
                    stop_flag_clone,
                    audio_shared_clone,
                );
            })
            .map_err(|e| format!("Failed to spawn audio thread: {e}"))?;

        Ok(Self {
            handle: Some(handle),
            stop_flag,
            audio_shared,
        })
    }
}

impl Drop for MoqAudioThread {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            if let Err(e) = handle.join() {
                tracing::warn!("MoQ audio thread join failed: {:?}", e);
                *self.audio_shared.audio_status.lock() = MoqAudioStatus::Error;
            }
        }
        self.audio_shared
            .internal_audio_ready
            .store(false, Ordering::Relaxed);
        // Preserve Error status — don't overwrite with Unavailable
        let mut status = self.audio_shared.audio_status.lock();
        if *status == MoqAudioStatus::Running || *status == MoqAudioStatus::Buffering {
            *status = MoqAudioStatus::Unavailable;
        }
        drop(status);
        if let Some(handle) = self.audio_shared.moq_audio_handle.lock().as_ref() {
            handle.set_available(false);
            handle.clear_playback_epoch();
            handle.reset_samples_played();
            handle.clear_audio_base_pts();
        }
    }
}

/// Maximum consecutive decode errors before downgrading to `Error` status.
const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 100;

/// Number of decoded audio frames to accumulate before starting playback.
/// ~170ms at 48kHz (each AAC frame = 1024 samples ≈ 21.3ms).
const AUDIO_PRE_BUFFER_FRAMES: usize = 8;

/// Pre-buffer timeout: audio must wait at least as long as video hard failsafe.
/// Derived from shared constant (+1s) so audio never leads video.
const AUDIO_PRE_BUFFER_TIMEOUT: Duration =
    Duration::from_secs(super::moq_decoder::MOQ_STARTUP_HARD_FAILSAFE_SECS + 1);

/// Audio thread main loop: receives AAC frames, decodes via symphonia, plays via rodio.
fn moq_audio_thread_main(
    audio_rx: crossbeam_channel::Receiver<MoqAudioFrame>,
    sample_rate: u32,
    channels: u32,
    description: Option<Bytes>,
    audio_handle: AudioHandle,
    stop_flag: Arc<AtomicBool>,
    audio_shared: Arc<MoqAudioShared>,
) {
    let mut player = match AudioPlayer::new_with_handle(
        AudioConfig::default(),
        Some(audio_handle.clone()),
    ) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("MoQ audio: failed to create AudioPlayer: {e}");
            #[cfg(target_os = "linux")]
            tracing::warn!("MoQ audio: on Linux, ensure libasound2-dev is installed and an audio device is available");
            *audio_shared.audio_status.lock() = MoqAudioStatus::Error;
            audio_shared
                .internal_audio_ready
                .store(false, Ordering::Relaxed);
            audio_handle.set_available(false);
            return;
        }
    };

    let mut aac_decoder =
        match SymphoniaAacDecoder::new(sample_rate, channels, description.as_ref()) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("MoQ audio: failed to create AAC decoder: {e}");
                *audio_shared.audio_status.lock() = MoqAudioStatus::Error;
                audio_shared
                    .internal_audio_ready
                    .store(false, Ordering::Relaxed);
                audio_handle.set_available(false);
                return;
            }
        };

    audio_handle.set_audio_format(sample_rate, channels);

    tracing::info!(
        "MoQ audio: thread started ({}Hz, {}ch, description={})",
        sample_rate,
        channels,
        description.as_ref().map(|d| d.len()).unwrap_or(0),
    );

    // -- Phase 1: Pre-buffer --
    // Accumulate decoded frames and wait for first successful video decode
    // before starting playback to avoid choppy audio on join.
    *audio_shared.audio_status.lock() = MoqAudioStatus::Buffering;
    let mut pre_buffer: Vec<AudioSamples> = Vec::new();
    let pre_buffer_start = std::time::Instant::now();
    let mut stopped = false;
    let mut channel_disconnected = false;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            tracing::debug!("MoQ audio: stop_flag set during pre-buffer");
            stopped = true;
            break;
        }
        if pre_buffer_start.elapsed() > AUDIO_PRE_BUFFER_TIMEOUT {
            tracing::warn!(
                "MoQ audio: pre-buffer timeout after {}ms, starting with {} frames",
                pre_buffer_start.elapsed().as_millis(),
                pre_buffer.len(),
            );
            break;
        }
        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => match aac_decoder.decode_frame(&frame.data, frame.timestamp_us) {
                Ok(samples) => {
                    pre_buffer.push(samples);
                }
                Err(e) => {
                    tracing::debug!("MoQ audio: pre-buffer decode error: {e}");
                }
            },
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                tracing::debug!("MoQ audio: channel disconnected during pre-buffer");
                channel_disconnected = true;
                break;
            }
        }
        if pre_buffer.len() >= AUDIO_PRE_BUFFER_FRAMES
            && audio_shared.video_started.load(Ordering::Relaxed)
        {
            break; // Normal exit: enough frames + video decode confirmed
        }
    }

    // -- Phase 2: Flush pre-buffer + start playback --
    if !stopped && !channel_disconnected {
        for samples in pre_buffer {
            player.queue_samples(samples);
        }

        // Advertise audio availability once playback is actually starting.
        // This avoids exposing a "present but silent" clock during pre-buffer.
        audio_handle.set_available(true);
        audio_shared
            .internal_audio_ready
            .store(true, Ordering::Relaxed);
        *audio_shared.audio_status.lock() = MoqAudioStatus::Running;
        player.play();

        tracing::info!(
            "MoQ audio: playback started after {}ms pre-buffer",
            pre_buffer_start.elapsed().as_millis(),
        );

        // -- Phase 3: Steady-state decode loop --
        let mut consecutive_errors: u32 = 0;

        loop {
            if stop_flag.load(Ordering::Relaxed) {
                tracing::debug!("MoQ audio: stop_flag set, exiting");
                break;
            }

            match audio_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(frame) => match aac_decoder.decode_frame(&frame.data, frame.timestamp_us) {
                    Ok(samples) => {
                        player.queue_samples(samples);
                        consecutive_errors = 0;
                    }
                    Err(e) => {
                        consecutive_errors += 1;
                        tracing::debug!("MoQ audio: decode error #{}: {e}", consecutive_errors);
                        if consecutive_errors >= MAX_CONSECUTIVE_DECODE_ERRORS {
                            tracing::warn!(
                                "MoQ audio: {} consecutive decode errors, likely unsupported AAC profile",
                                consecutive_errors
                            );
                            *audio_shared.audio_status.lock() = MoqAudioStatus::Error;
                            audio_shared
                                .internal_audio_ready
                                .store(false, Ordering::Relaxed);
                            break;
                        }
                    }
                },
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    tracing::debug!("MoQ audio: channel disconnected, exiting");
                    break;
                }
            }
        }
    } else if channel_disconnected {
        tracing::debug!("MoQ audio: channel disconnected during pre-buffer; skipping playback");
    }
    // else: stopped = true → skip to cleanup

    audio_shared
        .internal_audio_ready
        .store(false, Ordering::Relaxed);
    audio_handle.set_available(false);
    audio_handle.clear_playback_epoch();
    audio_handle.reset_samples_played();
    audio_handle.clear_audio_base_pts();

    let mut status = audio_shared.audio_status.lock();
    if *status != MoqAudioStatus::Error {
        *status = MoqAudioStatus::Unavailable;
    }

    tracing::info!("MoQ audio: thread exiting");
}

/// Selects the preferred audio rendition from a MoQ catalog.
///
/// Filters to AAC-only tracks and returns the one with the highest sample rate.
pub(crate) fn select_preferred_audio_rendition(
    catalog: &hang::catalog::Catalog,
) -> Option<(&str, &hang::catalog::AudioConfig)> {
    use hang::catalog::AudioCodec;

    catalog
        .audio
        .as_ref()?
        .renditions
        .iter()
        .filter(|(_, cfg)| matches!(cfg.codec, AudioCodec::AAC(_)))
        .max_by_key(|(_, cfg)| cfg.sample_rate)
        .map(|(name, cfg)| (name.as_str(), cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_edge_sender_non_blocking() {
        let (tx, rx) = crossbeam_channel::bounded(2);
        let sender = LiveEdgeSender::new(tx.clone(), rx.clone());

        // Fill the channel — first two should be Sent
        assert!(matches!(sender.send(1u32), Ok(SendResult::Sent)));
        assert!(matches!(sender.send(2u32), Ok(SendResult::Sent)));
        // Channel is full — should not block, should Dropped (eviction)
        assert!(matches!(sender.send(3u32), Ok(SendResult::Dropped)));

        // Should still be able to receive something
        let _val = rx.try_recv();
    }

    #[test]
    fn test_live_edge_sender_channel_closed() {
        // In production, LiveEdgeSender holds rx_drain from the same channel as tx.
        // The channel disconnects only when ALL receivers (including rx_drain) are
        // dropped, which happens when the LiveEdgeSender itself is dropped.
        //
        // To test the ChannelClosed code path, we create a fully disconnected tx
        // (all receivers dropped) and pair it with a separate drain receiver.
        // This validates that try_send() correctly propagates Disconnected.
        let (tx, rx) = crossbeam_channel::bounded::<u32>(2);
        drop(rx); // tx is now disconnected — no receivers left

        // rx_drain is from a separate channel (only used for drain logic, not delivery)
        let (_drain_tx, drain_rx) = crossbeam_channel::bounded::<u32>(2);
        let sender = LiveEdgeSender::new(tx, drain_rx);
        assert!(sender.send(1).is_err());
    }

    #[test]
    fn test_live_edge_sender_same_channel_lifecycle() {
        // Production wiring: tx and rx_drain are from the SAME channel.
        // LiveEdgeSender keeps the channel alive via rx_drain. After dropping
        // the sender (which drops rx_drain), the channel becomes disconnected.
        let (tx, rx) = crossbeam_channel::bounded::<u32>(2);
        let rx_drain = rx.clone();

        let sender = LiveEdgeSender::new(tx, rx_drain);

        // Sender works while channel is alive
        assert!(matches!(sender.send(1), Ok(SendResult::Sent)));
        assert!(matches!(sender.send(2), Ok(SendResult::Sent)));
        // Full channel — drain + retry, should Dropped (eviction)
        assert!(matches!(sender.send(3), Ok(SendResult::Dropped)));

        // Consumer can still receive
        assert!(rx.try_recv().is_ok());

        // Drop sender (and its rx_drain) — channel disconnects
        drop(sender);

        // Verify channel is now closed from consumer side
        // Drain remaining items first
        while rx.try_recv().is_ok() {}
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_live_edge_sender_bounded_size() {
        let (tx, rx) = crossbeam_channel::bounded(3);
        let sender = LiveEdgeSender::new(tx.clone(), rx.clone());

        // Send more than capacity — some will be Dropped
        for i in 0..10u32 {
            let result = sender.send(i);
            assert!(result.is_ok());
        }
        // Channel should never exceed capacity
        assert!(rx.len() <= 3);
    }
}
