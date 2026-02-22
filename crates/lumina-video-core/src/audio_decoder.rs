//! FFmpeg-based audio decoder for extracting audio from video files.
//!
//! This module provides audio decoding using FFmpeg, converting compressed
//! audio to PCM samples that can be played back via cpal.

use std::time::Duration;

use crate::audio::AudioSamples;

/// Audio decoder error types.
#[derive(Debug, Clone)]
pub enum AudioError {
    /// Failed to open the audio stream
    OpenFailed(String),
    /// No audio stream found
    NoAudioStream,
    /// Decoder initialization failed
    DecoderInit(String),
    /// Decoding failed
    DecodeFailed(String),
    /// Seek failed
    SeekFailed(String),
}

impl std::fmt::Display for AudioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenFailed(s) => write!(f, "Open failed: {s}"),
            Self::NoAudioStream => write!(f, "No audio stream found"),
            Self::DecoderInit(s) => write!(f, "Decoder init failed: {s}"),
            Self::DecodeFailed(s) => write!(f, "Decode failed: {s}"),
            Self::SeekFailed(s) => write!(f, "Seek failed: {s}"),
        }
    }
}

impl std::error::Error for AudioError {}

/// Audio metadata from the stream.
#[derive(Debug, Clone)]
pub struct AudioMetadata {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Audio codec name
    pub codec: String,
    /// Duration if known
    pub duration: Option<Duration>,
    /// Start time of the audio stream (first PTS), for A/V sync offset calculation
    pub start_time: Option<Duration>,
}

// ============================================================================
// Real FFmpeg implementation (when feature is enabled)
// ============================================================================

#[cfg(target_os = "macos")]
mod real_impl {
    use super::*;
    use ffmpeg_next as ffmpeg;
    use ffmpeg_next::ffi;

    /// Maximum number of empty decode attempts during seeking before treating as EOF.
    /// Allows ~1 second of empty decodes (100 attempts * 10ms sleep in frame_queue).
    const MAX_EMPTY_DECODES_SEEKING: u32 = 100;

    /// FFmpeg-based audio decoder.
    pub struct AudioDecoder {
        /// Input format context
        input: ffmpeg::format::context::Input,
        /// Audio stream index
        audio_stream_index: usize,
        /// Audio decoder
        decoder: ffmpeg::decoder::Audio,
        /// Audio resampler for format conversion
        resampler: Option<ffmpeg::software::resampling::Context>,
        /// Audio metadata
        metadata: AudioMetadata,
        /// Stream time base (numerator, denominator)
        time_base: (i32, i32),
        /// Whether EOF has been reached
        eof_reached: bool,
        /// Packet iterator state
        packet_iter_finished: bool,
        /// Target output sample rate
        target_sample_rate: u32,
        /// Whether we've logged the first frame (for debug purposes)
        logged_first_frame: bool,
        /// Whether we're in seeking state (don't treat empty packets as EOF)
        seeking: bool,
        /// Count of consecutive empty decode attempts during seeking
        empty_decode_count: u32,
    }

    impl AudioDecoder {
        /// Creates a new audio decoder for the given URL or file path.
        pub fn new(url: &str, target_sample_rate: u32) -> Result<Self, AudioError> {
            // ffmpeg::init() is safe to call multiple times (just registers codecs/formats)
            ffmpeg::init()
                .map_err(|e| AudioError::DecoderInit(format!("FFmpeg init failed: {e}")))?;

            // Open input file/stream
            let input = ffmpeg::format::input(&url)
                .map_err(|e| AudioError::OpenFailed(format!("Failed to open {url}: {e}")))?;

            // Find audio stream
            let audio_stream = input
                .streams()
                .best(ffmpeg::media::Type::Audio)
                .ok_or(AudioError::NoAudioStream)?;

            let audio_stream_index = audio_stream.index();
            let time_base = audio_stream.time_base();

            // Get codec parameters
            let codec_params = audio_stream.parameters();

            // Create decoder context from parameters
            let context =
                ffmpeg::codec::context::Context::from_parameters(codec_params).map_err(|e| {
                    AudioError::DecoderInit(format!("Failed to create codec context: {e}"))
                })?;

            // Open decoder
            let decoder = context
                .decoder()
                .audio()
                .map_err(|e| AudioError::DecoderInit(format!("Failed to open decoder: {e}")))?;

            // Extract metadata
            let duration = if input.duration() > 0 {
                Some(Duration::from_micros(
                    (input.duration() as f64 * 1_000_000.0 / ffi::AV_TIME_BASE as f64) as u64,
                ))
            } else {
                None
            };

            // Extract stream start time (convert from stream time_base to Duration)
            let start_time = {
                let st = audio_stream.start_time();
                if st >= 0 && time_base.1 > 0 {
                    // start_time is in stream time_base units
                    let us = st as i128 * time_base.0 as i128 * 1_000_000 / time_base.1 as i128;
                    Some(Duration::from_micros(us.max(0) as u64))
                } else {
                    None
                }
            };

            let metadata = AudioMetadata {
                sample_rate: decoder.rate(),
                channels: decoder.channels() as u16,
                codec: decoder
                    .codec()
                    .map(|c| c.name().to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                duration,
                start_time,
            };

            tracing::info!(
                "Audio: {}Hz -> {}Hz, {} channels, codec: {}, duration: {:?}",
                metadata.sample_rate,
                target_sample_rate,
                metadata.channels,
                metadata.codec,
                metadata.duration
            );

            Ok(Self {
                input,
                audio_stream_index,
                decoder,
                resampler: None,
                metadata,
                time_base: (time_base.0, time_base.1),
                eof_reached: false,
                packet_iter_finished: false,
                target_sample_rate,
                logged_first_frame: false,
                seeking: false,
                empty_decode_count: 0,
            })
        }

        /// Returns the audio metadata.
        pub fn metadata(&self) -> &AudioMetadata {
            &self.metadata
        }

        fn pts_to_duration(&self, pts: i64) -> Duration {
            if pts < 0 || self.time_base.1 == 0 {
                return Duration::ZERO;
            }
            let seconds = (pts as f64) * (self.time_base.0 as f64) / (self.time_base.1 as f64);
            Duration::from_secs_f64(seconds.max(0.0))
        }

        fn ensure_resampler(&mut self, frame: &ffmpeg::frame::Audio) -> Result<(), AudioError> {
            let src_format = frame.format();
            let src_rate = frame.rate();
            let src_layout = frame.channel_layout();

            // Target format: stereo, device sample rate, f32 packed (interleaved)
            let dst_format = ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed);
            let dst_rate = self.target_sample_rate;
            let dst_layout = ffmpeg::ChannelLayout::STEREO;

            // Check if we need to recreate the resampler (format/rate/layout changed)
            let needs_recreate = match &self.resampler {
                None => true,
                Some(resampler) => {
                    let input = resampler.input();
                    input.format != src_format
                        || input.rate != src_rate
                        || input.channel_layout != src_layout
                }
            };

            if needs_recreate {
                let resampler = ffmpeg::software::resampling::Context::get(
                    src_format, src_layout, src_rate, dst_format, dst_layout, dst_rate,
                )
                .map_err(|e| {
                    AudioError::DecodeFailed(format!("Failed to create resampler: {e}"))
                })?;

                self.resampler = Some(resampler);
            }

            Ok(())
        }

        fn frame_to_samples(
            &mut self,
            frame: &ffmpeg::frame::Audio,
        ) -> Result<AudioSamples, AudioError> {
            self.ensure_resampler(frame)?;

            let Some(resampler) = self.resampler.as_mut() else {
                return Err(AudioError::DecodeFailed(
                    "Resampler not initialized".to_string(),
                ));
            };

            // Create output frame
            let mut output = ffmpeg::frame::Audio::empty();

            // Run resampler
            let _delay = resampler
                .run(frame, &mut output)
                .map_err(|e| AudioError::DecodeFailed(format!("Resampling failed: {e}")))?;

            // Get the actual number of samples from the frame
            let num_samples = output.samples();

            if num_samples == 0 {
                // No samples produced yet (resampler buffering)
                return Ok(AudioSamples {
                    data: vec![],
                    sample_rate: self.target_sample_rate,
                    channels: 2,
                    pts: Duration::ZERO,
                });
            }

            // Get raw byte data and interpret as f32
            let raw_data = output.data(0);
            let bytes_per_sample = 4; // f32
            let channels = 2;
            let expected_bytes = num_samples * channels * bytes_per_sample;

            // Debug: log first frame info
            if !self.logged_first_frame {
                self.logged_first_frame = true;
                tracing::info!(
                    "Audio frame: {} samples, raw data {} bytes, expected {} bytes, format {:?}, rate {}",
                    num_samples,
                    raw_data.len(),
                    expected_bytes,
                    output.format(),
                    output.rate()
                );
            }

            // Convert raw bytes to f32 samples
            let num_floats = (raw_data.len() / 4).min(num_samples * channels);
            let mut samples = Vec::with_capacity(num_floats);

            for i in 0..num_floats {
                let offset = i * 4;
                if offset + 4 <= raw_data.len() {
                    let sample = f32::from_ne_bytes([
                        raw_data[offset],
                        raw_data[offset + 1],
                        raw_data[offset + 2],
                        raw_data[offset + 3],
                    ]);
                    samples.push(sample);
                }
            }

            let pts = frame.pts().unwrap_or(0);

            // Use actual output rate from resampler
            let actual_rate = output.rate();

            Ok(AudioSamples {
                data: samples,
                sample_rate: actual_rate,
                channels: 2,
                pts: self.pts_to_duration(pts),
            })
        }

        /// Decodes the next audio samples.
        pub fn decode_next(&mut self) -> Result<Option<AudioSamples>, AudioError> {
            if self.eof_reached {
                return Ok(None);
            }

            let mut decoded_frame = ffmpeg::frame::Audio::empty();

            loop {
                // Try to receive a frame from the decoder
                match self.decoder.receive_frame(&mut decoded_frame) {
                    Ok(()) => {
                        // Got a frame - clear seeking state and reset empty decode count
                        if self.seeking {
                            tracing::debug!("Audio: got frame after seek, clearing seeking state");
                            self.seeking = false;
                        }
                        self.empty_decode_count = 0;
                        let samples = self.frame_to_samples(&decoded_frame)?;
                        return Ok(Some(samples));
                    }
                    Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                        // Need more packets
                        if self.packet_iter_finished {
                            // During seek, don't send EOF - just return None to let caller retry
                            // But track empty decodes to detect true EOF
                            if self.seeking {
                                self.empty_decode_count += 1;
                                if self.empty_decode_count >= MAX_EMPTY_DECODES_SEEKING {
                                    tracing::warn!(
                                        "Audio: exceeded {} empty decodes during seek, treating as EOF",
                                        MAX_EMPTY_DECODES_SEEKING
                                    );
                                    self.seeking = false;
                                    self.eof_reached = true;
                                    return Ok(None);
                                }
                                tracing::trace!(
                                    "Audio: no packets during seek ({}/{}), returning None to retry",
                                    self.empty_decode_count,
                                    MAX_EMPTY_DECODES_SEEKING
                                );
                                self.packet_iter_finished = false; // Reset for next attempt
                                return Ok(None);
                            }
                            // Normal EOF path
                            tracing::debug!("Audio: sending EOF to decoder (packet_iter_finished)");
                            self.decoder.send_eof().ok();
                            self.packet_iter_finished = false;
                            continue;
                        }

                        // Read next packet
                        let mut found_audio_packet = false;
                        for (stream, packet) in self.input.packets() {
                            if stream.index() == self.audio_stream_index {
                                self.decoder.send_packet(&packet).map_err(|e| {
                                    AudioError::DecodeFailed(format!("Send packet failed: {e}"))
                                })?;
                                found_audio_packet = true;
                                break;
                            }
                        }

                        if !found_audio_packet {
                            // During seek, don't treat empty packets as EOF
                            // But track empty decodes to detect true EOF
                            if self.seeking {
                                self.empty_decode_count += 1;
                                if self.empty_decode_count >= MAX_EMPTY_DECODES_SEEKING {
                                    tracing::warn!(
                                        "Audio: exceeded {} empty decodes during seek (no packets), treating as EOF",
                                        MAX_EMPTY_DECODES_SEEKING
                                    );
                                    self.seeking = false;
                                    self.eof_reached = true;
                                    return Ok(None);
                                }
                                tracing::trace!(
                                    "Audio: no packets found during seek ({}/{}), will retry",
                                    self.empty_decode_count,
                                    MAX_EMPTY_DECODES_SEEKING
                                );
                                return Ok(None);
                            }
                            tracing::debug!(
                                "Audio: no audio packet found, marking packet_iter_finished"
                            );
                            self.packet_iter_finished = true;
                        }
                    }
                    Err(ffmpeg::Error::Eof) => {
                        // During seek, decoder EOF might be stale - reset and retry
                        if self.seeking {
                            tracing::debug!(
                                "Audio: decoder EOF during seek, flushing and retrying"
                            );
                            self.decoder.flush();
                            return Ok(None);
                        }
                        tracing::debug!("Audio: decoder returned EOF");
                        self.eof_reached = true;
                        return Ok(None);
                    }
                    Err(e) => {
                        tracing::error!("Audio decode error: {}", e);
                        return Err(AudioError::DecodeFailed(format!("Decode error: {e}")));
                    }
                }
            }
        }

        /// Seeks to a specific position.
        ///
        /// After seeking, the packet iterator is invalidated by FFmpeg's input.seek().
        /// We set seeking=true to prevent treating empty packets as EOF during buffering.
        /// The resampler is also reset to avoid filter-history bleed from pre-seek audio.
        pub fn seek(&mut self, position: Duration) -> Result<(), AudioError> {
            // input.seek() expects timestamps in AV_TIME_BASE (microseconds), not stream time_base
            let timestamp = position.as_micros() as i64;

            tracing::debug!(
                "Audio seek: position={:?}, timestamp={} (AV_TIME_BASE)",
                position,
                timestamp
            );

            // Use RangeFull (`..`) to allow FFmpeg to seek to the nearest keyframe.
            self.input
                .seek(timestamp, ..)
                .map_err(|e| AudioError::SeekFailed(format!("Seek failed: {e}")))?;

            // Flush decoder to clear any buffered frames from pre-seek position
            self.decoder.flush();

            // Drop the resampler to avoid filter-history bleed from pre-seek audio.
            // It will be lazily recreated when the next frame is decoded via ensure_resampler().
            self.resampler = None;

            // Reset all state flags - input.seek() invalidates the packet iterator
            self.eof_reached = false;
            self.packet_iter_finished = false;
            // Enter seeking state - don't treat empty packets as EOF
            self.seeking = true;
            // Reset empty decode counter for fresh seek
            self.empty_decode_count = 0;

            tracing::debug!(
                "Audio seek complete: seeking=true, eof_reached={}, packet_iter_finished={}, resampler=None",
                self.eof_reached,
                self.packet_iter_finished
            );

            Ok(())
        }
    }

    // SAFETY: AudioDecoder can be safely sent between threads because:
    // - FFmpeg's Input and Audio contexts are not thread-safe for concurrent access,
    //   but they CAN be safely moved between threads (single ownership transfer)
    // - After creation on the main thread, the decoder is moved to the audio decode thread
    //   where it has exclusive ownership and is never accessed from other threads
    // - The Send trait only guarantees safe ownership transfer, not concurrent access,
    //   which matches our usage pattern
    unsafe impl Send for AudioDecoder {}
}

// ============================================================================
// Placeholder implementation (when feature is disabled)
// ============================================================================

#[cfg(not(target_os = "macos"))]
mod placeholder_impl {
    use super::*;

    /// Placeholder audio decoder.
    pub struct AudioDecoder {
        metadata: AudioMetadata,
    }

    impl AudioDecoder {
        /// Creates a new audio decoder (placeholder).
        pub fn new(_url: &str, _target_sample_rate: u32) -> Result<Self, AudioError> {
            Err(AudioError::NoAudioStream)
        }

        /// Returns the audio metadata.
        pub fn metadata(&self) -> &AudioMetadata {
            &self.metadata
        }

        /// Decodes the next audio samples (placeholder).
        pub fn decode_next(&mut self) -> Result<Option<AudioSamples>, AudioError> {
            Ok(None)
        }

        /// Seeks to a specific position (placeholder).
        pub fn seek(&mut self, _position: Duration) -> Result<(), AudioError> {
            Ok(())
        }
    }
}

// Re-export the appropriate implementation
#[cfg(target_os = "macos")]
pub use real_impl::AudioDecoder;

#[cfg(not(target_os = "macos"))]
pub use placeholder_impl::AudioDecoder;
