//! MoQ audio pipeline: crossbeam handoff → symphonia AAC decode → cpal playback.
//!
//! This module is gated on `cfg(all(feature = "moq", any(target_os = "macos", target_os = "linux", target_os = "android")))`.
//! Android uses cpal/Oboe (same pipeline as desktop). Windows has no MoQ decoder yet.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;

use super::audio::{AudioHandle, AudioPlayer, AudioSamples};
use super::audio_ring_buffer::RingBufferConfig;
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
    /// Returns `Ok(())` on success or benign drop, `Err(ChannelClosed)` if
    /// the channel is permanently disconnected.
    pub fn send(&self, item: T) -> Result<(), ChannelClosed> {
        match self.tx.try_send(item) {
            Ok(()) => Ok(()),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(ChannelClosed),
            Err(crossbeam_channel::TrySendError::Full(item)) => {
                let _ = self.rx_drain.try_recv(); // best-effort drain (race ok)
                match self.tx.try_send(item) {
                    Ok(()) => Ok(()),
                    Err(crossbeam_channel::TrySendError::Disconnected(_)) => Err(ChannelClosed),
                    Err(crossbeam_channel::TrySendError::Full(_)) => Ok(()), // benign race-drop
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
    ///
    /// If the decoded output has fewer channels than `self.channels` (e.g. mono AAC
    /// but stereo playback), upmixes by duplicating each sample across channels.
    fn decode_frame(&mut self, data: &[u8], timestamp_us: u64) -> Result<AudioSamples, String> {
        use symphonia_core::formats::Packet;

        let packet = Packet::new_from_slice(0, 0, 0, data);
        let decoded = self
            .decoder
            .decode(&packet)
            .map_err(|e| format!("AAC decode error: {e}"))?;

        let spec = *decoded.spec();
        let decoded_channels = spec.channels.count();
        let num_frames = decoded.frames();

        if num_frames == 0 || decoded_channels == 0 {
            return Err("Empty decoded buffer".to_string());
        }

        let mut sample_buf =
            symphonia_core::audio::SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let raw = sample_buf.samples();

        // Upmix if decoded channels < target channels (e.g. mono → stereo)
        let target_ch = self.channels as usize;
        let data = if decoded_channels < target_ch {
            let mut upmixed = Vec::with_capacity(num_frames * target_ch);
            for frame_samples in raw.chunks(decoded_channels) {
                // Duplicate each decoded sample across target channels
                for ch in 0..target_ch {
                    upmixed.push(frame_samples[ch.min(decoded_channels - 1)]);
                }
            }
            upmixed
        } else if decoded_channels > target_ch {
            // Downmix: take only the first target_ch channels per frame
            let mut downmixed = Vec::with_capacity(num_frames * target_ch);
            for frame_samples in raw.chunks(decoded_channels) {
                for sample in frame_samples.iter().take(target_ch) {
                    downmixed.push(*sample);
                }
            }
            downmixed
        } else {
            raw.to_vec()
        };

        Ok(AudioSamples {
            data,
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
        self.audio_shared.alive.store(false, Ordering::Release);
        self.audio_shared
            .internal_audio_ready
            .store(false, Ordering::Relaxed);
        // Preserve Error status — don't overwrite with Unavailable
        let mut status = self.audio_shared.audio_status.lock();
        if *status == MoqAudioStatus::Running {
            *status = MoqAudioStatus::Unavailable;
        }
        drop(status);
        *self.audio_shared.moq_audio_handle.lock() = None;
    }
}

/// Maximum consecutive decode errors before downgrading to `Error` status.
const MAX_CONSECUTIVE_DECODE_ERRORS: u32 = 100;

/// Gap detection threshold: if two consecutive frames have a PTS gap larger than
/// this, insert silence to keep the ring buffer timeline correct.
const GAP_THRESHOLD_US: u64 = 32_000; // 32ms

/// Audio thread main loop: receives AAC frames, decodes via symphonia, writes to ring buffer.
///
/// Uses a lock-free ring buffer read by the cpal audio callback,
/// eliminating per-source transitions and queue mutex contention
/// that caused clicks/pops in the previous batch+append approach.
fn moq_audio_thread_main(
    audio_rx: crossbeam_channel::Receiver<MoqAudioFrame>,
    sample_rate: u32,
    channels: u32,
    description: Option<Bytes>,
    audio_handle: AudioHandle,
    stop_flag: Arc<AtomicBool>,
    audio_shared: Arc<MoqAudioShared>,
) {
    let ring_config = RingBufferConfig::for_format(sample_rate, channels.min(2) as u16);

    let (mut player, producer) = match AudioPlayer::new_ring_buffer(
        ring_config,
        Some(audio_handle.clone()),
        Some(sample_rate),
    ) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::warn!("MoQ audio: failed to create ring buffer AudioPlayer: {e}");
            #[cfg(target_os = "linux")]
            tracing::warn!("MoQ audio: on Linux, ensure libasound2-dev is installed and an audio device is available");
            *audio_shared.audio_status.lock() = MoqAudioStatus::Error;
            audio_shared
                .internal_audio_ready
                .store(false, Ordering::Relaxed);
            *audio_shared.moq_audio_handle.lock() = None;
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
                *audio_shared.moq_audio_handle.lock() = None;
                return;
            }
        };

    audio_handle.set_audio_format(sample_rate, channels);

    audio_shared
        .internal_audio_ready
        .store(true, Ordering::Relaxed);
    audio_shared.alive.store(true, Ordering::Release);
    *audio_shared.audio_status.lock() = MoqAudioStatus::Running;
    player.play();

    tracing::info!(
        "MoQ audio: ring buffer thread started ({}Hz, {}ch, description={})",
        sample_rate,
        channels,
        description.as_ref().map(|d| d.len()).unwrap_or(0),
    );

    let mut consecutive_errors: u32 = 0;
    let mut last_pts_us: Option<u64> = None;
    let ch = channels.min(2) as usize;
    let mut frames_since_metrics: u32 = 0;
    let mut total_frames_decoded: u64 = 0;
    let thread_start = std::time::Instant::now();
    let mut first_frame_logged = false;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            tracing::debug!("MoQ audio: stop_flag set, exiting");
            break;
        }

        match audio_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(frame) => match aac_decoder.decode_frame(&frame.data, frame.timestamp_us) {
                Ok(samples) => {
                    consecutive_errors = 0;
                    total_frames_decoded += 1;

                    // Log first decoded frame details
                    if !first_frame_logged {
                        first_frame_logged = true;
                        audio_handle.set_audio_base_pts(samples.pts);
                        tracing::info!(
                            "MoQ audio: first frame base_pts={:?}, {} interleaved samples, {}ch declared",
                            samples.pts,
                            samples.data.len(),
                            samples.channels,
                        );
                    }

                    // Set base PTS on first decoded frame
                    if last_pts_us.is_none() {
                        audio_handle.set_audio_base_pts(samples.pts);
                    }

                    // Gap detection: insert silence for PTS discontinuities > 32ms
                    if let Some(prev_pts) = last_pts_us {
                        let expected_next = prev_pts
                            + (samples.data.len() as u64 * 1_000_000)
                                / (sample_rate as u64 * ch as u64);
                        let actual = frame.timestamp_us;
                        if actual > expected_next + GAP_THRESHOLD_US {
                            let gap_us = actual - expected_next;
                            let silence_samples =
                                (gap_us as usize * sample_rate as usize * ch) / 1_000_000;
                            let silence = vec![0.0f32; silence_samples.min(48000 * ch)]; // Cap at 1s
                            producer.write(&silence);
                            tracing::debug!(
                                "MoQ audio: inserted {}ms silence ({} samples) for PTS gap",
                                gap_us / 1000,
                                silence_samples,
                            );
                        }
                    }
                    last_pts_us = Some(frame.timestamp_us);

                    // Write decoded PCM directly to ring buffer (lock-free)
                    producer.write(&samples.data);

                    // Periodically flush ring buffer metrics for UI observability (~1/sec)
                    frames_since_metrics += 1;
                    if frames_since_metrics >= 50 {
                        frames_since_metrics = 0;
                        let m = producer.metrics();
                        let elapsed = thread_start.elapsed().as_secs_f64();
                        let fps = total_frames_decoded as f64 / elapsed.max(0.001);
                        tracing::debug!(
                            "MoQ audio: {:.1} fps, fill={}% ({}/{}), wrote={}, read={}, overflows={}, stalls={}, samples/frame={}",
                            fps,
                            m.fill_samples * 100 / m.capacity_samples.max(1),
                            m.fill_samples, m.capacity_samples,
                            m.total_written, m.total_read,
                            m.overflow_count, m.stall_count,
                            samples.data.len(),
                        );
                        *audio_shared.ring_buffer_metrics.lock() = m;
                    }
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

    // Drop producer before cleanup so consumer knows we're done
    drop(producer);

    audio_shared.alive.store(false, Ordering::Release);
    audio_shared
        .internal_audio_ready
        .store(false, Ordering::Relaxed);
    *audio_shared.moq_audio_handle.lock() = None;

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

        // Fill the channel
        assert!(sender.send(1u32).is_ok());
        assert!(sender.send(2u32).is_ok());
        // Channel is full — should not block, should Ok
        assert!(sender.send(3u32).is_ok());

        // Should still be able to receive something
        let _val = rx.try_recv();
    }

    #[test]
    fn test_live_edge_sender_channel_closed() {
        let (tx, rx) = crossbeam_channel::bounded::<u32>(2);
        drop(rx);

        let (_drain_tx, drain_rx) = crossbeam_channel::bounded::<u32>(2);
        let sender = LiveEdgeSender::new(tx, drain_rx);
        assert!(sender.send(1).is_err());
    }

    #[test]
    fn test_live_edge_sender_same_channel_lifecycle() {
        let (tx, rx) = crossbeam_channel::bounded::<u32>(2);
        let rx_drain = rx.clone();

        let sender = LiveEdgeSender::new(tx, rx_drain);

        assert!(sender.send(1).is_ok());
        assert!(sender.send(2).is_ok());
        assert!(sender.send(3).is_ok());

        assert!(rx.try_recv().is_ok());

        drop(sender);

        while rx.try_recv().is_ok() {}
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn test_select_audio_empty_renditions() {
        let catalog = hang::catalog::Catalog::default();
        assert!(select_preferred_audio_rendition(&catalog).is_none());
    }

    #[test]
    fn test_select_audio_prefers_highest_sample_rate() {
        use std::collections::BTreeMap;

        let mut renditions = BTreeMap::new();
        renditions.insert(
            "audio0".to_string(),
            hang::catalog::AudioConfig {
                codec: hang::catalog::AudioCodec::AAC(hang::catalog::AAC { profile: 2 }),
                sample_rate: 44100,
                channel_count: 2,
                bitrate: None,
                description: None,
                container: hang::catalog::Container::Legacy,
                jitter: None,
            },
        );
        renditions.insert(
            "audio1".to_string(),
            hang::catalog::AudioConfig {
                codec: hang::catalog::AudioCodec::AAC(hang::catalog::AAC { profile: 2 }),
                sample_rate: 48000,
                channel_count: 2,
                bitrate: None,
                description: None,
                container: hang::catalog::Container::Legacy,
                jitter: None,
            },
        );

        let catalog = hang::catalog::Catalog {
            audio: hang::catalog::Audio { renditions },
            ..Default::default()
        };

        let (name, cfg) = select_preferred_audio_rendition(&catalog).unwrap();
        assert_eq!(name, "audio1");
        assert_eq!(cfg.sample_rate, 48000);
    }

    #[test]
    fn test_live_edge_sender_bounded_size() {
        let (tx, rx) = crossbeam_channel::bounded(3);
        let sender = LiveEdgeSender::new(tx.clone(), rx.clone());

        for i in 0..10u32 {
            assert!(sender.send(i).is_ok());
        }
        assert!(rx.len() <= 3);
    }
}
