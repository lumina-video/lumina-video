//! Shared MoQ worker logic for all platforms.
//!
//! Contains the single `run_moq_worker()` function that handles:
//! - QUIC connection with optional WebSocket fallback
//! - Broadcast discovery with timeout
//! - Catalog fetch, validation, and metadata extraction
//! - Audio track setup
//! - Main video/audio receive loop (with pre-allocated buffers)
//! - Deterministic audio teardown
//!
//! Each platform decoder delegates to this function with a thin wrapper.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_channel::Sender;
use bytes::{Buf, BytesMut};
use moq_lite::{Origin, PathOwned};
use moq_native::ClientConfig;

use crate::media::moq::MoqUrl;
use crate::media::moq_audio::{
    audio_codec_from_config, select_preferred_audio_rendition, ChannelClosed, LiveEdgeSender,
    MoqAudioFrame, MoqAudioThread,
};
use crate::media::moq_decoder::{
    MoqAudioStatus, MoqDecoder, MoqDecoderConfig, MoqDecoderState, MoqSharedState, MoqVideoFrame,
    MOQ_STARTUP_HARD_FAILSAFE_SECS,
};

/// Diagnostic hook: dumps raw frame data to .h264 + .csv for offline analysis.
/// Enabled by LUMINA_MOQ_DUMP_PREFIX env var. Never affects streaming.
struct FrameDumpHook {
    h264_writer: std::io::BufWriter<std::fs::File>,
    csv_writer: std::io::BufWriter<std::fs::File>,
    sps: Vec<u8>,
    pps: Vec<u8>,
    nal_length_size: usize,
    /// True when catalog has avcC description (AVCC format), false for Annex B.
    is_avcc: bool,
    max_frames: u32,
    frame_count: u32,
}

impl FrameDumpHook {
    /// Try to create from env vars + catalog avcC. Returns None (not Err) on any failure.
    fn try_init(codec_desc: &parking_lot::Mutex<Option<bytes::Bytes>>) -> Option<Self> {
        use std::io::Write;

        let prefix = std::env::var("LUMINA_MOQ_DUMP_PREFIX").ok()?;
        let max_frames: u32 = std::env::var("LUMINA_MOQ_DUMP_MAX_FRAMES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        let avcc_data = codec_desc.lock().clone()?;

        let (sps, pps, nal_length_size) = match MoqDecoder::parse_avcc_box(&avcc_data) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("FrameDumpHook: avcC parse failed ({}), dump DISABLED", e);
                return None;
            }
        };

        let h264_file = match std::fs::File::create(format!("{}.h264", prefix)) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("FrameDumpHook: can't create .h264 ({}), dump DISABLED", e);
                return None;
            }
        };
        let csv_file = match std::fs::File::create(format!("{}.csv", prefix)) {
            Ok(f) => f,
            Err(e) => {
                tracing::error!("FrameDumpHook: can't create .csv ({}), dump DISABLED", e);
                return None;
            }
        };

        let mut csv_writer = std::io::BufWriter::new(csv_file);
        if writeln!(
            csv_writer,
            "frame,is_keyframe,size_bytes,timestamp_us,nal_types,heuristic_annexb,conversion_ok"
        )
        .is_err()
        {
            tracing::error!("FrameDumpHook: can't write CSV header, dump DISABLED");
            return None;
        }

        tracing::info!(
            "FrameDumpHook: enabled, prefix={}, max_frames={}, nal_length_size={}",
            prefix,
            max_frames,
            nal_length_size
        );
        Some(Self {
            h264_writer: std::io::BufWriter::new(h264_file),
            csv_writer,
            sps,
            pps,
            nal_length_size,
            is_avcc: true, // catalog has avcC → AVCC format
            max_frames,
            frame_count: 0,
        })
    }

    /// Record one frame. Returns Ok(true) if still recording, Ok(false) if done.
    /// On I/O error: logs warning, caller should disable hook.
    fn record_frame(
        &mut self,
        data: &[u8],
        is_keyframe: bool,
        timestamp_us: u64,
    ) -> Result<bool, std::io::Error> {
        use std::io::Write;

        if self.frame_count >= self.max_frames {
            return Ok(false);
        }

        // Telemetry: heuristic result logged but NOT used for conversion decision
        let heuristic_annexb = MoqDecoder::data_is_annex_b(data);

        // SPS/PPS injection before every keyframe (first-in-group per hang semantics)
        if is_keyframe {
            self.h264_writer.write_all(&[0, 0, 0, 1])?;
            self.h264_writer.write_all(&self.sps)?;
            self.h264_writer.write_all(&[0, 0, 0, 1])?;
            self.h264_writer.write_all(&self.pps)?;
        }

        // Always AVCC→Annex B conversion (catalog guarantees AVCC format)
        let mut conversion_ok = true;
        let mut offset = 0;
        while offset + self.nal_length_size <= data.len() {
            let nal_len = match self.nal_length_size {
                1 => data[offset] as usize,
                2 => u16::from_be_bytes([data[offset], data[offset + 1]]) as usize,
                3 => u32::from_be_bytes([0, data[offset], data[offset + 1], data[offset + 2]])
                    as usize,
                4 => u32::from_be_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as usize,
                _ => {
                    conversion_ok = false;
                    break;
                }
            };
            offset += self.nal_length_size;
            if offset + nal_len > data.len() || nal_len == 0 {
                conversion_ok = false;
                break;
            }
            self.h264_writer.write_all(&[0, 0, 0, 1])?;
            self.h264_writer
                .write_all(&data[offset..offset + nal_len])?;
            offset += nal_len;
        }
        // Detect trailing bytes not consumed by AVCC parsing
        if offset != data.len() {
            conversion_ok = false;
        }

        // CSV: frame stats + format telemetry (use format-aware parsing, not heuristic)
        let (nal_types_arr, nal_count) =
            MoqDecoder::find_nal_types_for_format(data, self.nal_length_size, self.is_avcc);
        writeln!(
            self.csv_writer,
            "{},{},{},{},{:?},{},{}",
            self.frame_count,
            is_keyframe,
            data.len(),
            timestamp_us,
            &nal_types_arr[..nal_count],
            heuristic_annexb,
            conversion_ok,
        )?;

        self.frame_count += 1;
        if self.frame_count == self.max_frames {
            self.h264_writer.flush()?;
            self.csv_writer.flush()?;
            tracing::info!(
                "FrameDumpHook: captured {} frames, dump complete",
                self.max_frames
            );
        }
        Ok(true)
    }
}

/// Shared async worker for MoQ connection, catalog fetch, and frame receipt.
///
/// All three platform decoders (macOS, Android, Linux/GStreamer) delegate to
/// this function. The `label` parameter is used for tracing prefixes.
/// Why the worker is re-subscribing to the video track.
///
/// Controls whether the startup IDR gate is re-enabled after re-subscribe.
/// Decoder-initiated recovery already has its own IDR gate
/// (`MoqDecoder::waiting_for_idr_after_error`), so re-enabling the worker
/// gate would create a double-gate that starves the decoder of frames.
#[derive(Debug, Clone, Copy)]
enum ResubscribeReason {
    /// First-subscribe startup gate timeout or broken-keyframe storm.
    StartupGate,
    /// Video track ended (Ok(None) from read_frame).
    TrackEnded,
    /// Decoder requested recovery (IDR starvation, VT error escalation).
    DecoderRecovery,
    /// read_frame() hung beyond timeout — transport stall.
    ReadTimeout,
    /// read_frame() returned an error — transport/decode error.
    ReadError,
}

impl std::fmt::Display for ResubscribeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartupGate => f.write_str("startup IDR gate"),
            Self::TrackEnded => f.write_str("video track ended"),
            Self::DecoderRecovery => f.write_str("decoder requested recovery"),
            Self::ReadTimeout => f.write_str("read_frame timeout"),
            Self::ReadError => f.write_str("video read error"),
        }
    }
}

/// Result of catalog fetch and validation, avoiding positional tuple.
struct CatalogResult {
    video_track_name: String,
    max_latency: Duration,
    catalog: hang::catalog::Catalog,
    /// Whether the selected video rendition is H.264.
    selected_is_h264: bool,
    /// Codec description (avcC/hvcC) from catalog, if present.
    selected_video_description: Option<bytes::Bytes>,
}

/// Re-subscribe video (and audio, if still active) after stream end or decoder-requested recovery.
///
/// When `reason` is [`ResubscribeReason::DecoderRecovery`], the worker startup
/// IDR gate is **not** re-enabled. The decoder already has its own IDR gate
/// (`waiting_for_idr_after_error`) that filters frames post-channel. Re-enabling
/// the worker gate would create a double-gate: the worker drops non-IDR frames
/// before they reach the channel, so the decoder gate starves and cannot clear.
#[allow(clippy::too_many_arguments)]
async fn resubscribe_video_track(
    _shared: &Arc<MoqSharedState>,
    moq_broadcast: &moq_lite::BroadcastConsumer,
    video_track: &moq_lite::Track,
    max_latency: Duration,
    video_consumer: &mut hang::container::OrderedConsumer,
    idr_gate_enabled: bool,
    waiting_for_valid_idr: &mut bool,
    idr_gate_groups_seen: &mut u8,
    idr_gate_start: &mut Option<std::time::Instant>,
    resubscribe_count: &mut u32,
    recent_resubscribes: &mut VecDeque<Instant>,
    max_resubscribes_in_window: usize,
    resubscribe_window: Duration,
    resubscribe_cooldown: Duration,
    label: &str,
    reason: ResubscribeReason,
    detail: &str,
) -> bool {
    let now = Instant::now();
    while let Some(front) = recent_resubscribes.front() {
        if now.saturating_duration_since(*front) <= resubscribe_window {
            break;
        }
        recent_resubscribes.pop_front();
    }

    if recent_resubscribes.len() >= max_resubscribes_in_window {
        tracing::warn!(
            "MoQ {}: Re-subscribe storm detected ({} within {:?}); throttling for {:?}",
            label,
            recent_resubscribes.len(),
            resubscribe_window,
            resubscribe_cooldown,
        );
        tokio::time::sleep(resubscribe_cooldown).await;

        let now = Instant::now();
        while let Some(front) = recent_resubscribes.front() {
            if now.saturating_duration_since(*front) <= resubscribe_window {
                break;
            }
            recent_resubscribes.pop_front();
        }
        if recent_resubscribes.len() >= max_resubscribes_in_window {
            recent_resubscribes.pop_front();
        }
    }

    *resubscribe_count += 1;
    recent_resubscribes.push_back(Instant::now());
    tracing::warn!(
        "MoQ {}: Re-subscribing video (attempt #{}, recent={} within {:?}) — {} ({})",
        label,
        *resubscribe_count,
        recent_resubscribes.len(),
        resubscribe_window,
        reason,
        detail,
    );

    *video_consumer = hang::container::OrderedConsumer::new(
        moq_broadcast.subscribe_track(video_track),
        max_latency,
    );

    // Audio runs in its own dedicated task and is not re-subscribed here.

    // Reset startup gate and A/V startup sync — but NOT for decoder recovery.
    //
    // DecoderRecovery: the decoder already has its own IDR gate
    // (waiting_for_idr_after_error) that filters post-channel. Re-enabling
    // the worker gate here would double-gate: worker drops non-IDR frames
    // before they reach the decoder, starving its gate indefinitely.
    let skip_gate_reset = matches!(reason, ResubscribeReason::DecoderRecovery);
    if idr_gate_enabled && !skip_gate_reset {
        *waiting_for_valid_idr = true;
        *idr_gate_groups_seen = 0;
        *idr_gate_start = None;
    } else if idr_gate_enabled {
        tracing::info!(
            "MoQ {}: Skipping worker IDR gate reset (reason={}, decoder has its own gate)",
            label,
            reason,
        );
    }

    true
}

pub(crate) async fn run_moq_worker(
    shared: Arc<MoqSharedState>,
    url: MoqUrl,
    config: MoqDecoderConfig,
    frame_tx: Sender<MoqVideoFrame>,
    label: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // -- Phase 1: Connect --
    shared.set_state(MoqDecoderState::Connecting);
    shared.buffering_percent.store(10, Ordering::Relaxed);

    let (connect_url, broadcast_path) = build_connect_url(&url);

    let redacted_connect_url = connect_url
        .split_once('?')
        .map(|(base, _)| format!("{}?<redacted>", base))
        .unwrap_or_else(|| connect_url.clone());
    tracing::info!("MoQ {}: Connecting to {}", label, redacted_connect_url);
    if let Some(ref path) = broadcast_path {
        tracing::info!("MoQ {}: Will look for broadcast at path: {:?}", label, path);
    }

    let parsed_url: url::Url = connect_url
        .parse()
        .map_err(|e| format!("Invalid URL: {e}"))?;

    let (mut origin_consumer, transport_protocol, _session) =
        connect_to_relay(&parsed_url, &config, label).await?;

    *shared.transport_protocol.lock() = transport_protocol.to_string();
    tracing::info!(
        "MoQ {}: Connected ({}), waiting for broadcast announcement",
        label,
        transport_protocol,
    );

    // -- Phase 2: Discover broadcast --
    shared.set_state(MoqDecoderState::FetchingCatalog);
    shared.buffering_percent.store(30, Ordering::Relaxed);

    let moq_broadcast =
        discover_broadcast(&mut origin_consumer, broadcast_path, &url, label).await?;

    tracing::info!("MoQ {}: Found broadcast, subscribing to tracks", label);
    shared.buffering_percent.store(50, Ordering::Relaxed);

    // -- Phase 3: Fetch and validate catalog --
    let cat = fetch_and_validate_catalog(&moq_broadcast, &shared, &config, label).await?;

    // -- Phase 4: Subscribe to video track --
    let video_track = moq_lite::Track {
        name: cat.video_track_name.clone(),
        priority: 100,
    };
    let mut video_consumer = hang::container::OrderedConsumer::new(
        moq_broadcast.subscribe_track(&video_track),
        cat.max_latency,
    );

    // -- Phase 5: Audio setup --
    let (audio_consumer_opt, audio_sender_opt, mut moq_audio_thread_opt) = setup_audio(
        &cat.catalog,
        &moq_broadcast,
        cat.max_latency,
        &config,
        &shared,
        label,
    );

    // -- Phase 5.5: Spawn dedicated audio forwarding task --
    //
    // IMPORTANT: OrderedConsumer::read() (hang crate) is NOT cancellation-safe.
    // Audio gets its own task so read() is never cancelled by tokio::select!.
    let mut audio_task =
        spawn_audio_forward_task(audio_consumer_opt, audio_sender_opt, &shared.audio, label);

    // Audio track name for re-subscribing when the audio task finishes
    // (track end, error, or stream loop).
    let audio_track_for_resub: Option<moq_lite::Track> = if audio_task.is_some() {
        select_preferred_audio_rendition(&cat.catalog).map(|(name, _)| moq_lite::Track {
            name: name.to_string(),
            priority: 50,
        })
    } else {
        None
    };

    // -- Phase 5b: Frame dump hook (diagnostic, env-gated) --
    let mut frame_dump = FrameDumpHook::try_init(&shared.codec_description);

    // -- Phase 5c: Startup IDR gate (H.264 only) --
    let avcc_parsed = if cat.selected_is_h264 {
        cat.selected_video_description
            .as_ref()
            .and_then(|d| MoqDecoder::parse_avcc_box(d).ok())
    } else {
        None
    };
    let nal_length_size = avcc_parsed.as_ref().map(|(_, _, nls)| *nls).unwrap_or(4);
    let idr_gate_enabled = cat.selected_is_h264 && avcc_parsed.is_some();

    if cat.selected_is_h264 && !idr_gate_enabled {
        tracing::warn!(
            "MoQ {}: H.264 stream without parseable avcC; bypassing startup IDR gate",
            label
        );
    }

    let mut waiting_for_valid_idr = idr_gate_enabled;
    let mut idr_gate_groups_seen: u8 = 0;
    let mut idr_gate_broken_keyframes: u8 = 0;
    const IDR_GATE_MAX_GROUPS: u8 = 3;
    const IDR_GATE_TIMEOUT: Duration = Duration::from_secs(5);
    const IDR_GATE_BROKEN_KEYFRAME_THRESHOLD: u8 = 2;
    const IDR_GATE_BROKEN_KEYFRAME_MIN_ELAPSED: Duration = Duration::from_millis(800);
    let idr_gate_hard_failsafe_timeout: Duration =
        Duration::from_secs(MOQ_STARTUP_HARD_FAILSAFE_SECS);
    let mut idr_gate_start: Option<std::time::Instant> = None;

    // -- Phase 6: Streaming --
    shared.set_state(MoqDecoderState::Streaming);
    shared.buffering_percent.store(100, Ordering::Relaxed);

    tracing::info!(
        "MoQ {}: Streaming started, subscribed to video track '{}' (idr_gate={})",
        label,
        cat.video_track_name,
        idr_gate_enabled,
    );

    // Pre-allocate reusable buffers to avoid per-frame allocation
    let mut video_buf = BytesMut::with_capacity(256 * 1024);
    let mut stats_log_counter = 0u64;
    let mut resubscribe_count: u32 = 0;
    let mut recent_resubscribes: VecDeque<Instant> = VecDeque::with_capacity(8);
    const MAX_RESUBSCRIBES_IN_WINDOW: usize = 5;
    const RESUBSCRIBE_WINDOW: Duration = Duration::from_secs(8);
    const RESUBSCRIBE_COOLDOWN: Duration = Duration::from_millis(750);
    const READ_FRAME_TIMEOUT: Duration = Duration::from_secs(3);
    const MAX_READ_ERROR_RECOVERIES_IN_WINDOW: usize = 8;
    const READ_ERROR_RECOVERY_WINDOW: Duration = Duration::from_secs(30);

    // Prevent tight restart loops when audio track has ended or is unstable.
    const AUDIO_RESUBSCRIBE_RETRY_DELAY: Duration = Duration::from_millis(500);
    let mut audio_needs_resubscribe = false;
    let mut next_audio_resubscribe_attempt = Instant::now();

    // External audio health watchdog: if the audio forward task is alive but
    // hasn't forwarded a frame in this long, force-abort and resubscribe.
    // This fires as a select! arm (not at loop-top) so it can't be blocked by
    // video_consumer.read() sitting in the other arm for up to 3s.
    //
    // Budget: internal READ_TIMEOUT (8s) + 2s margin = 10s. The internal
    // timeout should handle most cases; this catches the observed failure mode
    // where the internal timeout silently fails to fire.
    const AUDIO_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(10);

    let mut recent_read_error_recoveries: VecDeque<Instant> =
        VecDeque::with_capacity(MAX_READ_ERROR_RECOVERIES_IN_WINDOW);

    // Persistent video deadline: tracks wall-clock time since last video frame.
    // Unlike wrapping read_frame() in tokio::timeout(), this isn't reset when the
    // audio arm completes — so a video-only stall is detected even while audio
    // continues producing frames.
    let mut video_deadline = tokio::time::Instant::now() + READ_FRAME_TIMEOUT;

    #[allow(unused_assignments)] // defensive default; all break paths assign before breaking
    let mut exit_reason: &str = "unknown";
    let mut loop_iter: u64 = 0;
    loop {
        loop_iter += 1;
        #[allow(clippy::manual_is_multiple_of)] // MSRV consistency: is_multiple_of requires 1.87+
        if loop_iter <= 50 || loop_iter % 100 == 0 {
            tracing::info!("MoQ {}: select loop iter #{}", label, loop_iter);
        }

        // Detect when the dedicated audio task has finished.
        // The dedicated audio task exits on track end (Ok(None)) or read error.
        // Without re-subscribing, looped/recovered streams continue video-only.
        if let Some(ref task) = audio_task {
            if task.is_finished() {
                if let Some(finished) = audio_task.take() {
                    match finished.await {
                        Ok(()) => {
                            tracing::info!("MoQ {}: Audio forward task completed normally", label);
                        }
                        Err(e) if e.is_panic() => {
                            tracing::error!(
                                "MoQ {}: Audio forward task PANICKED: {:?}",
                                label,
                                e.into_panic(),
                            );
                        }
                        Err(e) => {
                            tracing::warn!("MoQ {}: Audio forward task cancelled: {e}", label);
                        }
                    }
                }
                // Teardown the old audio decode thread (sender was moved into the task,
                // so the crossbeam channel is already disconnected)
                teardown_audio(
                    None,
                    moq_audio_thread_opt.take(),
                    &shared,
                    label,
                    "audio forward task finished",
                )
                .await;
                audio_needs_resubscribe = true;
            }
        }

        // Compute audio watchdog deadline for the select! arm below.
        // If no audio task is active, set deadline to far-future (never fires).
        let audio_watchdog_deadline = if audio_task.as_ref().is_some_and(|t| !t.is_finished()) {
            let last = *shared.audio.last_audio_forward_frame_at.lock();
            match last {
                Some(instant) => {
                    let elapsed = instant.elapsed();
                    if elapsed >= AUDIO_WATCHDOG_TIMEOUT {
                        // Already stale — fire immediately
                        tokio::time::Instant::now()
                    } else {
                        tokio::time::Instant::now() + (AUDIO_WATCHDOG_TIMEOUT - elapsed)
                    }
                }
                // No heartbeat yet (task just spawned). Give it the full
                // watchdog window to produce its first frame.
                None => tokio::time::Instant::now() + AUDIO_WATCHDOG_TIMEOUT,
            }
        } else {
            // No audio task — park the watchdog
            tokio::time::Instant::now() + Duration::from_secs(86400)
        };

        // Throttled audio re-subscribe.
        if audio_needs_resubscribe && Instant::now() >= next_audio_resubscribe_attempt {
            next_audio_resubscribe_attempt = Instant::now() + AUDIO_RESUBSCRIBE_RETRY_DELAY;

            if let Some(ref at) = audio_track_for_resub {
                tracing::info!(
                    "MoQ {}: Re-subscribing audio track '{}' after task exit",
                    label,
                    at.name,
                );
                let (new_consumer, new_sender, new_thread) = setup_audio(
                    &cat.catalog,
                    &moq_broadcast,
                    cat.max_latency,
                    &config,
                    &shared,
                    label,
                );
                moq_audio_thread_opt = new_thread;
                audio_task =
                    spawn_audio_forward_task(new_consumer, new_sender, &shared.audio, label);
                if audio_task.is_some() {
                    audio_needs_resubscribe = false;
                } else {
                    tracing::warn!(
                        "MoQ {}: audio re-subscribe attempt produced no task; retrying in {:?}",
                        label,
                        AUDIO_RESUBSCRIBE_RETRY_DELAY,
                    );
                }
            } else {
                audio_needs_resubscribe = false;
            }
        }

        tokio::select! {
            // No biased — fair scheduling prevents audio starvation
            video_result = video_consumer.read() => {
                // NOTE: video_deadline is only reset when a frame is actually
                // submitted to the decoder (past IDR gate). Frames that arrive
                // but get skipped at the IDR gate do NOT reset the watchdog.
                match video_result {
                    Ok(Some(frame)) => {
                        if shared
                            .request_video_resubscribe
                            .swap(false, Ordering::AcqRel)
                        {
                            idr_gate_broken_keyframes = 0;
                            if !resubscribe_video_track(
                                &shared,
                                &moq_broadcast,
                                &video_track,
                                cat.max_latency,
                                &mut video_consumer,
                                idr_gate_enabled,
                                &mut waiting_for_valid_idr,
                                &mut idr_gate_groups_seen,
                                &mut idr_gate_start,
                                &mut resubscribe_count,
                                &mut recent_resubscribes,
                                MAX_RESUBSCRIBES_IN_WINDOW,
                                RESUBSCRIBE_WINDOW,
                                RESUBSCRIBE_COOLDOWN,
                                label,
                                ResubscribeReason::DecoderRecovery,
                                "IDR starvation recovery",
                            )
                            .await
                            {
                                exit_reason = "resubscribe failed (decoder recovery)";
                                break;
                            }
                            video_deadline = tokio::time::Instant::now() + READ_FRAME_TIMEOUT;
                            continue;
                        }

                        let recv_count =
                            shared.frame_stats.received.fetch_add(1, Ordering::Relaxed) + 1;

                        stats_log_counter += 1;
                        #[allow(clippy::manual_is_multiple_of)] // MSRV 1.83: is_multiple_of requires 1.87+
                        if stats_log_counter % 30 == 0 {
                            shared.frame_stats.log_summary(label);
                        }

                        let data = assemble_payload(&frame.payload, &mut video_buf);

                        // Frame dump: record then check if we should disable
                        let mut disable_dump = false;
                        if let Some(ref mut dump) = frame_dump {
                            match dump.record_frame(&data, frame.keyframe, frame.timestamp.as_micros() as u64) {
                                Ok(false) => disable_dump = true,
                                Err(e) => {
                                    tracing::warn!("FrameDumpHook: I/O error ({}), disabling dump", e);
                                    disable_dump = true;
                                }
                                _ => {}
                            }
                        }
                        if disable_dump {
                            frame_dump = None;
                        }

                        // -- IDR gate: skip frames until a real IDR at group boundary --
                        if waiting_for_valid_idr {
                            let start = *idr_gate_start.get_or_insert_with(std::time::Instant::now);

                            // Log first few keyframes BEFORE gate decision for diagnostics
                            if frame.keyframe && idr_gate_groups_seen < 5 {
                                let n = data.len().min(20);
                                let is_avcc_fmt = idr_gate_enabled;
                                let (nal_arr, nal_count) =
                                    MoqDecoder::find_nal_types_for_format(&data, nal_length_size, is_avcc_fmt);
                                tracing::info!(
                                    "MoQ {}: IDR gate keyframe #{}: {} bytes, NAL types={:?}, first_20={:02x?}",
                                    label,
                                    idr_gate_groups_seen + 1,
                                    data.len(),
                                    &nal_arr[..nal_count],
                                    &data[..n],
                                );
                            }

                            if frame.keyframe {
                                idr_gate_groups_seen = idr_gate_groups_seen.saturating_add(1);
                            }
                            let elapsed = start.elapsed();
                            let timed_out = elapsed > IDR_GATE_TIMEOUT;
                            let groups_exhausted = idr_gate_groups_seen >= IDR_GATE_MAX_GROUPS;

                            // Use format-aware NAL parsing: idr_gate_enabled implies AVCC
                            // (from catalog avcC). The heuristic data_is_annex_b() misclassifies
                            // AVCC frames with 256-511 byte NALs, causing missed IDRs.
                            let is_avcc = idr_gate_enabled; // gate only enabled for avcC streams
                            let (nal_arr, nal_count) =
                                MoqDecoder::find_nal_types_for_format(&data, nal_length_size, is_avcc);
                            let has_idr = nal_arr[..nal_count].contains(&5);

                            if frame.keyframe && has_idr {
                                // Best case: real IDR at group boundary.
                                tracing::info!(
                                    "MoQ {}: IDR gate cleared — real IDR at {}ms (groups_seen={})",
                                    label,
                                    elapsed.as_millis(),
                                    idr_gate_groups_seen,
                                );
                                waiting_for_valid_idr = false;
                                idr_gate_broken_keyframes = 0;
                            } else {
                                if frame.keyframe && !has_idr {
                                    idr_gate_broken_keyframes =
                                        idr_gate_broken_keyframes.saturating_add(1);
                                    tracing::warn!(
                                        "MoQ {}: startup keyframe metadata mismatch (count={}, group_keyframe=true, NAL types={:?}, {} bytes)",
                                        label,
                                        idr_gate_broken_keyframes,
                                        &nal_arr[..nal_count],
                                        data.len(),
                                    );
                                }

                                let broken_keyframe_storm = idr_gate_broken_keyframes
                                    >= IDR_GATE_BROKEN_KEYFRAME_THRESHOLD
                                    && elapsed >= IDR_GATE_BROKEN_KEYFRAME_MIN_ELAPSED;

                                if timed_out
                                    || groups_exhausted
                                    || elapsed > idr_gate_hard_failsafe_timeout
                                    || broken_keyframe_storm
                                {
                                    let detail = format!(
                                        "elapsed={}ms, groups_seen={}, broken_keyframes={}, groups_exhausted={}, broken_keyframe_storm={}",
                                        elapsed.as_millis(),
                                        idr_gate_groups_seen,
                                        idr_gate_broken_keyframes,
                                        groups_exhausted,
                                        broken_keyframe_storm,
                                    );
                                    idr_gate_broken_keyframes = 0;
                                    if !resubscribe_video_track(
                                        &shared,
                                        &moq_broadcast,
                                        &video_track,
                                        cat.max_latency,
                                        &mut video_consumer,
                                        idr_gate_enabled,
                                        &mut waiting_for_valid_idr,
                                        &mut idr_gate_groups_seen,
                                        &mut idr_gate_start,
                                        &mut resubscribe_count,
                                        &mut recent_resubscribes,
                                        MAX_RESUBSCRIBES_IN_WINDOW,
                                        RESUBSCRIBE_WINDOW,
                                        RESUBSCRIBE_COOLDOWN,
                                        label,
                                        ResubscribeReason::StartupGate,
                                        &detail,
                                    )
                                    .await
                                    {
                                        exit_reason = "resubscribe failed (startup gate)";
                                        break;
                                    }
                                    continue;
                                }

                                shared
                                    .frame_stats
                                    .skipped_startup_frames
                                    .fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            // Gate cleared — video_started set by decoder on first successful decode
                        }

                        let moq_frame = MoqVideoFrame {
                            timestamp_us: frame.timestamp.as_micros() as u64,
                            data,
                            is_keyframe: frame.keyframe,
                        };

                        if recv_count <= 10 {
                            let mut preview = [0u8; 20];
                            let n = moq_frame.data.len().min(20);
                            preview[..n].copy_from_slice(&moq_frame.data[..n]);
                            tracing::info!(
                                "MoQ {} recv frame #{}: is_keyframe={}, {} bytes, first_20={:02x?}",
                                label, recv_count, moq_frame.is_keyframe, moq_frame.data.len(), &preview[..n],
                            );
                        }

                        // Frame passed IDR gate — reset video stall watchdog
                        video_deadline = tokio::time::Instant::now() + READ_FRAME_TIMEOUT;

                        tracing::info!("MoQ {}: sending frame #{} to channel ({} bytes, kf={})", label, recv_count, moq_frame.data.len(), moq_frame.is_keyframe);
                        if frame_tx.send(moq_frame).await.is_err() {
                            tracing::warn!("MoQ {}: Frame channel closed, stopping worker", label);
                            exit_reason = "frame_tx closed (decoder dropped receiver)";
                            break;
                        }
                        tracing::info!("MoQ {}: frame #{} sent to channel OK", label, recv_count);
                    }
                    Ok(None) => {
                        idr_gate_broken_keyframes = 0;
                        if !resubscribe_video_track(
                            &shared,
                            &moq_broadcast,
                            &video_track,
                            cat.max_latency,
                            &mut video_consumer,
                            idr_gate_enabled,
                            &mut waiting_for_valid_idr,
                            &mut idr_gate_groups_seen,
                            &mut idr_gate_start,
                            &mut resubscribe_count,
                            &mut recent_resubscribes,
                            MAX_RESUBSCRIBES_IN_WINDOW,
                            RESUBSCRIBE_WINDOW,
                            RESUBSCRIBE_COOLDOWN,
                            label,
                            ResubscribeReason::TrackEnded,
                            "read_frame returned None",
                        )
                        .await
                        {
                            exit_reason = "resubscribe failed (track ended)";
                            break;
                        }
                        video_deadline = tokio::time::Instant::now() + READ_FRAME_TIMEOUT;
                        continue;
                    }
                    Err(e) => {
                        let detail = format!("Frame read error: {e}");
                        tracing::warn!("MoQ {}: {detail}", label);
                        shared.frame_stats.log_summary(label);

                        // Recovery-window storm guard: prune stale entries, then check
                        let now = Instant::now();
                        while let Some(front) = recent_read_error_recoveries.front() {
                            if now.saturating_duration_since(*front) <= READ_ERROR_RECOVERY_WINDOW {
                                break;
                            }
                            recent_read_error_recoveries.pop_front();
                        }
                        if recent_read_error_recoveries.len() >= MAX_READ_ERROR_RECOVERIES_IN_WINDOW {
                            tracing::error!(
                                "MoQ {}: read error recovery storm ({} in {:?}), giving up",
                                label,
                                recent_read_error_recoveries.len(),
                                READ_ERROR_RECOVERY_WINDOW,
                            );
                            shared.set_error(detail);
                            exit_reason = "read error recovery storm";
                            break;
                        }

                        idr_gate_broken_keyframes = 0;
                        if !resubscribe_video_track(
                            &shared,
                            &moq_broadcast,
                            &video_track,
                            cat.max_latency,
                            &mut video_consumer,
                            idr_gate_enabled,
                            &mut waiting_for_valid_idr,
                            &mut idr_gate_groups_seen,
                            &mut idr_gate_start,
                            &mut resubscribe_count,
                            &mut recent_resubscribes,
                            MAX_RESUBSCRIBES_IN_WINDOW,
                            RESUBSCRIBE_WINDOW,
                            RESUBSCRIBE_COOLDOWN,
                            label,
                            ResubscribeReason::ReadError,
                            &detail,
                        )
                        .await
                        {
                            exit_reason = "resubscribe failed (read error)";
                            break;
                        }
                        recent_read_error_recoveries.push_back(now);
                        video_deadline = tokio::time::Instant::now() + READ_FRAME_TIMEOUT;
                        continue;
                    }
                }
            }
            // Video stall watchdog: fires when no video frame has arrived for
            // READ_FRAME_TIMEOUT, even if audio keeps producing. Unlike the old
            // tokio::timeout() wrapper, this deadline persists across select iterations.
            _ = tokio::time::sleep_until(video_deadline) => {
                tracing::warn!(
                    "MoQ {}: video stall detected ({}s, no video frames) after {} frames — resubscribing",
                    label, READ_FRAME_TIMEOUT.as_secs(),
                    shared.frame_stats.received.load(Ordering::Relaxed),
                );
                idr_gate_broken_keyframes = 0;
                if !resubscribe_video_track(
                    &shared,
                    &moq_broadcast,
                    &video_track,
                    cat.max_latency,
                    &mut video_consumer,
                    idr_gate_enabled,
                    &mut waiting_for_valid_idr,
                    &mut idr_gate_groups_seen,
                    &mut idr_gate_start,
                    &mut resubscribe_count,
                    &mut recent_resubscribes,
                    MAX_RESUBSCRIBES_IN_WINDOW,
                    RESUBSCRIBE_WINDOW,
                    RESUBSCRIBE_COOLDOWN,
                    label,
                    ResubscribeReason::ReadTimeout,
                    "video stall (no frames)",
                )
                .await
                {
                    exit_reason = "resubscribe failed (read timeout)";
                    break;
                }
                // Reset deadline after resubscribe
                video_deadline = tokio::time::Instant::now() + READ_FRAME_TIMEOUT;
                continue;
            }
            // Audio health watchdog: fires when the audio forward task is alive
            // but its heartbeat (last successful send to crossbeam) has gone stale.
            // Independent of the forward task's internal tokio::time::timeout,
            // which was observed to silently fail in production.
            _ = tokio::time::sleep_until(audio_watchdog_deadline) => {
                let stale_ms = shared
                    .audio
                    .last_audio_forward_frame_at
                    .lock()
                    .map(|i| i.elapsed().as_millis() as u64)
                    .unwrap_or(0);
                tracing::warn!(
                    "MoQ {}: AUDIO WATCHDOG: forward task alive but heartbeat stale ({}ms, threshold {}ms) — force-aborting for resubscribe",
                    label, stale_ms, AUDIO_WATCHDOG_TIMEOUT.as_millis(),
                );
                if let Some(task) = audio_task.take() {
                    task.abort();
                    match task.await {
                        Ok(()) => tracing::info!(
                            "MoQ {}: Watchdog-aborted audio task exited cleanly", label,
                        ),
                        Err(e) if e.is_cancelled() => tracing::debug!(
                            "MoQ {}: Watchdog-aborted audio task cancelled (expected)", label,
                        ),
                        Err(e) if e.is_panic() => tracing::error!(
                            "MoQ {}: Watchdog-aborted audio task PANICKED: {:?}",
                            label, e.into_panic(),
                        ),
                        Err(e) => tracing::warn!(
                            "MoQ {}: Watchdog-aborted audio task join error: {e}", label,
                        ),
                    }
                }
                teardown_audio(
                    None,
                    moq_audio_thread_opt.take(),
                    &shared,
                    label,
                    "audio watchdog stale heartbeat",
                )
                .await;
                audio_needs_resubscribe = true;
                continue;
            }
        }
    }

    // Post-loop exit log
    {
        let frames = shared.frame_stats.received.load(Ordering::Relaxed);
        let eof = shared.eof_reached.load(Ordering::Relaxed);
        if exit_reason == "frame_tx closed (decoder dropped receiver)" {
            tracing::info!(
                "MoQ {}: main loop exited — reason='{}', loop_iter={}, frames={}, eof={}",
                label,
                exit_reason,
                loop_iter,
                frames,
                eof,
            );
        } else {
            tracing::warn!(
                "MoQ {}: main loop exited — reason='{}', loop_iter={}, frames={}, eof={}",
                label,
                exit_reason,
                loop_iter,
                frames,
                eof,
            );
        }
    }

    // -- Phase 7: Teardown --
    // Abort the dedicated audio task before tearing down the audio thread
    if let Some(task) = audio_task {
        task.abort();
        match task.await {
            Ok(()) => tracing::debug!("MoQ {}: Audio task exited cleanly at teardown", label),
            Err(e) if e.is_cancelled() => {
                tracing::debug!("MoQ {}: Audio task cancelled at teardown (expected)", label)
            }
            Err(e) if e.is_panic() => {
                tracing::error!(
                    "MoQ {}: Audio task PANICKED during teardown: {:?}",
                    label,
                    e.into_panic(),
                );
            }
            Err(e) => tracing::warn!("MoQ {}: Audio task join error at teardown: {e}", label),
        }
    }
    teardown_audio(
        None, // sender was moved into the audio task
        moq_audio_thread_opt.take(),
        &shared,
        label,
        "worker loop exited",
    )
    .await;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build the connection URL and optional broadcast path from a MoqUrl.
///
/// Handles both cdn.moq.dev style (namespace in URL) and zap.stream style
/// (UUID namespace = broadcast path).
fn build_connect_url(url: &MoqUrl) -> (String, Option<PathOwned>) {
    // moqs:// → https (real TLS), moq:// → http (self-signed cert dance).
    // Matches transport.rs connect() logic. moq-native uses the scheme to
    // decide whether to fetch the server's self-signed cert hash via HTTP.
    let scheme = if url.use_tls() { "https" } else { "http" };

    // Check if namespace looks like a UUID (zap.stream broadcast ID)
    let namespace_is_broadcast_id = url.namespace().len() >= 32
        && url
            .namespace()
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c == '-');

    if namespace_is_broadcast_id {
        // zap.stream style: connect to base URL, namespace is the broadcast path
        let base = format!("{}://{}", scheme, url.server_addr());
        let connect = match url.query() {
            Some(query) => format!("{}?{}", base, query),
            None => base,
        };
        let path = sanitize_path(url.namespace());
        let broadcast = if path.is_empty() {
            None
        } else {
            Some(PathOwned::from(path))
        };
        (connect, broadcast)
    } else {
        // cdn.moq.dev style: include namespace in connection URL
        let base = format!("{}://{}/{}", scheme, url.server_addr(), url.namespace());
        let connect = match url.query() {
            Some(query) => format!("{}?{}", base, query),
            None => base,
        };
        // Use track (if specified) as the broadcast path
        (
            connect,
            url.track().map(sanitize_path).and_then(|t| {
                if t.is_empty() {
                    None
                } else {
                    Some(PathOwned::from(t))
                }
            }),
        )
    }
}

/// Strip trailing slashes from a path string.
///
/// moq-lite's `announced()` panics (unwrap on None) if the prefix has a
/// trailing slash that doesn't match the server-side path. This sanitizes
/// user-provided namespace/track strings defensively.
/// See moq-dev/moq#910.
fn sanitize_path(s: &str) -> String {
    s.trim_end_matches('/').to_string()
}

/// Two-phase connect: try QUIC first, then WebSocket fallback.
///
/// Returns (OriginConsumer, protocol_name, Session).
/// The session must be kept alive for the worker lifetime.
async fn connect_to_relay(
    parsed_url: &url::Url,
    config: &MoqDecoderConfig,
    label: &str,
) -> Result<
    (moq_lite::OriginConsumer, &'static str, moq_lite::Session),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let quic_probe_timeout = Duration::from_millis(config.transport.connect_timeout_ms.min(1500));

    if config.transport.websocket_fallback {
        // Phase 1: QUIC-only (capped at 1500ms)
        tracing::info!("MoQ {}: Trying QUIC connection...", label);
        let quic_result = try_connect(
            parsed_url,
            config.transport.disable_tls_verify,
            false,
            quic_probe_timeout,
        )
        .await;

        match quic_result {
            Ok((consumer, session)) => {
                tracing::info!("MoQ {}: Connected via QUIC", label);
                Ok((consumer, "QUIC", session))
            }
            Err(e) => {
                // Phase 2: WebSocket fallback
                tracing::debug!("MoQ {}: QUIC probe failed: {e}", label);
                tracing::info!(
                    "MoQ {}: QUIC unavailable, connecting via WebSocket...",
                    label,
                );
                let timeout = Duration::from_millis(config.transport.connect_timeout_ms);
                let (consumer, session) = try_connect(
                    parsed_url,
                    config.transport.disable_tls_verify,
                    true,
                    timeout,
                )
                .await?;
                tracing::info!("MoQ {}: Connected via WebSocket", label);
                Ok((consumer, "WebSocket", session))
            }
        }
    } else {
        // WebSocket disabled, QUIC only
        let timeout = Duration::from_millis(config.transport.connect_timeout_ms);
        let (consumer, session) = try_connect(
            parsed_url,
            config.transport.disable_tls_verify,
            false,
            timeout,
        )
        .await?;
        tracing::info!("MoQ {}: Connected via QUIC", label);
        Ok((consumer, "QUIC", session))
    }
}

/// Build a MoQ client, connect to the relay, and return the consumer + session.
async fn try_connect(
    parsed_url: &url::Url,
    disable_tls_verify: bool,
    websocket: bool,
    timeout: Duration,
) -> Result<(moq_lite::OriginConsumer, moq_lite::Session), Box<dyn std::error::Error + Send + Sync>>
{
    let mut cfg = ClientConfig::default();
    if disable_tls_verify {
        cfg.tls.disable_verify = Some(true);
    }
    cfg.websocket.enabled = websocket;
    if websocket {
        cfg.websocket.delay = Some(Duration::ZERO);
    }
    let origin = Origin::produce();
    let consumer = origin.consume();
    let client = cfg.init().map_err(|e| format!("Client init: {e}"))?;
    let session = tokio::time::timeout(
        timeout,
        client.with_consume(origin).connect(parsed_url.clone()),
    )
    .await
    .map_err(|_| "Connection timed out")?
    .map_err(|e| format!("Connection failed: {e}"))?;
    Ok((consumer, session))
}

/// Wait for a broadcast to be announced, with 10s overall timeout.
async fn discover_broadcast(
    origin_consumer: &mut moq_lite::OriginConsumer,
    specific_broadcast: Option<PathOwned>,
    url: &MoqUrl,
    label: &str,
) -> Result<moq_lite::BroadcastConsumer, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(ref path) = specific_broadcast {
        tracing::info!("MoQ {}: Looking for specific broadcast: {:?}", label, path);
    } else {
        tracing::info!(
            "MoQ {}: Auto-discovering first available broadcast...",
            label
        );
    }

    let discovery_timeout = Duration::from_secs(10);
    let discovery_start = std::time::Instant::now();

    loop {
        if discovery_start.elapsed() > discovery_timeout {
            let msg = if specific_broadcast.is_some() {
                format!(
                    "Broadcast discovery timeout - '{}' not found after {:?}",
                    url.track().unwrap_or("unknown"),
                    discovery_timeout,
                )
            } else {
                format!(
                    "Broadcast discovery timeout - no broadcasts found on '{}' after {:?}",
                    url.namespace(),
                    discovery_timeout,
                )
            };
            return Err(msg.into());
        }

        // If looking for specific broadcast, check if already available
        if let Some(ref path) = specific_broadcast {
            if let Some(broadcast) = origin_consumer.consume_broadcast(path.clone()) {
                tracing::info!("MoQ {}: Found specific broadcast at {:?}", label, path);
                return Ok(broadcast);
            }
        }

        // Wait for announcements with a short timeout to allow checking overall timeout
        let wait_result =
            tokio::time::timeout(Duration::from_secs(2), origin_consumer.announced()).await;

        match wait_result {
            Ok(Some((path, Some(broadcast)))) => {
                if let Some(ref wanted) = specific_broadcast {
                    if path == *wanted {
                        tracing::info!("MoQ {}: Found matching broadcast at {:?}", label, path);
                        return Ok(broadcast);
                    } else {
                        tracing::debug!(
                            "MoQ {}: Ignoring broadcast at {:?}, waiting for {:?}",
                            label,
                            path,
                            wanted,
                        );
                        continue;
                    }
                } else {
                    tracing::info!("MoQ {}: Auto-selected broadcast: {:?}", label, path);
                    return Ok(broadcast);
                }
            }
            Ok(Some((_path, None))) => {
                continue;
            }
            Ok(None) => {
                return Err("Origin consumer closed without broadcast".into());
            }
            Err(_) => {
                tracing::debug!("MoQ {}: Still waiting for broadcast announcement...", label);
                continue;
            }
        }
    }
}

/// Fetch catalog with 5s timeout, validate it, log renditions, and store metadata.
///
/// Returns `(video_track_name, max_latency, catalog)`.
async fn fetch_and_validate_catalog(
    moq_broadcast: &moq_lite::BroadcastConsumer,
    shared: &Arc<MoqSharedState>,
    config: &MoqDecoderConfig,
    label: &str,
) -> Result<CatalogResult, Box<dyn std::error::Error + Send + Sync>> {
    let mut catalog_consumer =
        hang::CatalogConsumer::new(moq_broadcast.subscribe_track(&hang::Catalog::default_track()));
    let catalog_timeout = Duration::from_secs(5);
    let catalog = match tokio::time::timeout(catalog_timeout, catalog_consumer.next()).await {
        Ok(Ok(Some(catalog))) => catalog,
        Ok(Ok(None)) => {
            return Err(
                "Catalog track ended before receiving catalog (broadcast may be offline)".into(),
            );
        }
        Ok(Err(e)) => {
            return Err(format!("Failed to receive catalog: {e}").into());
        }
        Err(_) => {
            return Err("Catalog timeout - broadcast may be offline or has no active video".into());
        }
    };

    tracing::info!("MoQ {}: Received catalog", label);

    // Log full catalog contents for debugging
    {
        let video = &catalog.video;
        tracing::info!(
            "MoQ {} catalog: {} video rendition(s)",
            label,
            video.renditions.len()
        );
        for (name, cfg) in &video.renditions {
            let desc_len = cfg
                .description
                .as_ref()
                .map(|d: &bytes::Bytes| d.len())
                .unwrap_or(0);
            tracing::info!(
                "  video '{}': codec={:?}, {}x{}, {:.1}fps, bitrate={:?}, description={} bytes, container={:?}, jitter={:?}",
                name,
                cfg.codec,
                cfg.coded_width.unwrap_or(0),
                cfg.coded_height.unwrap_or(0),
                cfg.framerate.unwrap_or(0.0),
                cfg.bitrate,
                desc_len,
                cfg.container,
                cfg.jitter,
            );
        }
    }
    {
        let audio = &catalog.audio;
        tracing::info!(
            "MoQ {} catalog: {} audio rendition(s)",
            label,
            audio.renditions.len()
        );
        for (name, cfg) in &audio.renditions {
            let desc_len = cfg
                .description
                .as_ref()
                .map(|d: &bytes::Bytes| d.len())
                .unwrap_or(0);
            tracing::info!(
                "  audio '{}': codec={:?}, sample_rate={}, channels={}, bitrate={:?}, description={} bytes",
                name,
                cfg.codec,
                cfg.sample_rate,
                cfg.channel_count,
                cfg.bitrate,
                desc_len,
            );
        }
    }

    // Find the first video rendition in the catalog
    let (video_track_name, video_config): (&String, &hang::catalog::VideoConfig) = catalog
        .video
        .renditions
        .iter()
        .next()
        .ok_or("No video track in catalog")?;

    // Validate container format — we only support Legacy (raw NAL units)
    match video_config.container {
        hang::catalog::Container::Legacy => {
            tracing::info!("MoQ {}: Container format: Legacy (raw frames)", label);
        }
        hang::catalog::Container::Cmaf {
            timescale,
            track_id,
        } => {
            return Err(format!(
                "Unsupported container format: CMAF (timescale={}, track_id={}). \
                 lumina-video only supports Legacy (raw NAL units).",
                timescale, track_id,
            )
            .into());
        }
    }

    // Update metadata from catalog
    {
        let mut metadata = shared.metadata.lock();
        metadata.width = video_config.coded_width.unwrap_or(1920);
        metadata.height = video_config.coded_height.unwrap_or(1080);
        // Use catalog framerate if available; default to 24fps (common for MoQ live)
        // rather than 30fps since most MoQ streams (BBB, etc.) are 24fps.
        // A wrong default affects frame scheduling and measured FPS display.
        metadata.frame_rate = video_config.framerate.unwrap_or(24.0) as f32;
        metadata.codec = format!("{:?}", video_config.codec);
    }

    // Store codec description (avcC/hvcC box with SPS/PPS) if present
    if let Some(ref desc) = video_config.description {
        tracing::info!(
            "MoQ {}: Got codec description from catalog ({} bytes)",
            label,
            desc.len(),
        );
        *shared.codec_description.lock() = Some(desc.clone());
    } else {
        tracing::warn!(
            "MoQ {}: No codec description in catalog, will try to extract from keyframes",
            label,
        );
    }

    // Determine max latency: prefer catalog jitter (publisher-recommended buffer),
    // fall back to config default (500ms) when the catalog has no jitter field.
    let max_latency = if let Some(jitter) = video_config.jitter {
        let jitter_ms = jitter.as_millis() as u64;
        let effective_ms = if jitter_ms == 0 {
            tracing::warn!(
                "MoQ {}: catalog jitter=0ms (unset), using default {}ms",
                label,
                config.max_latency_ms,
            );
            config.max_latency_ms
        } else {
            tracing::info!(
                "MoQ {}: Using catalog jitter {}ms as latency buffer",
                label,
                jitter_ms,
            );
            jitter_ms
        };
        Duration::from_millis(effective_ms)
    } else {
        tracing::info!(
            "MoQ {}: No jitter in catalog, using default {}ms",
            label,
            config.max_latency_ms,
        );
        Duration::from_millis(config.max_latency_ms)
    };

    let selected_is_h264 = matches!(video_config.codec, hang::catalog::VideoCodec::H264(_));
    let selected_video_description = video_config.description.clone();

    Ok(CatalogResult {
        video_track_name: video_track_name.clone(),
        max_latency,
        catalog,
        selected_is_h264,
        selected_video_description,
    })
}

/// Set up audio track subscription and spawn decode/playback thread.
fn setup_audio(
    catalog: &hang::catalog::Catalog,
    moq_broadcast: &moq_lite::BroadcastConsumer,
    max_latency: Duration,
    config: &MoqDecoderConfig,
    shared: &Arc<MoqSharedState>,
    label: &str,
) -> (
    Option<hang::container::OrderedConsumer>,
    Option<LiveEdgeSender<MoqAudioFrame>>,
    Option<MoqAudioThread>,
) {
    if !config.enable_audio {
        return (None, None, None);
    }

    let (track_name, audio_cfg) = match select_preferred_audio_rendition(catalog) {
        Some(pair) => pair,
        None => {
            tracing::info!("MoQ {}: No supported audio track in catalog", label);
            *shared.audio.audio_status.lock() = MoqAudioStatus::Unavailable;
            *shared.audio_codec_name.lock() = None;
            return (None, None, None);
        }
    };

    *shared.audio.audio_status.lock() = MoqAudioStatus::Starting;
    // Reset watchdog heartbeat so stale values from a previous subscription
    // don't trigger false positives before the new forward task starts.
    *shared.audio.last_audio_forward_frame_at.lock() = None;

    let audio_track = moq_lite::Track {
        name: track_name.to_string(),
        priority: 50,
    };
    let audio_consumer = hang::container::OrderedConsumer::new(
        moq_broadcast.subscribe_track(&audio_track),
        max_latency,
    );

    let (tx, rx) = crossbeam_channel::bounded(config.audio_buffer_capacity);
    let live_sender = LiveEdgeSender::new(tx, rx.clone());

    // Reuse a stable shared handle so scheduler and audio thread observe the same clock state.
    let audio_handle = {
        let mut slot = shared.audio.moq_audio_handle.lock();
        if let Some(existing) = slot.clone() {
            existing
        } else {
            let handle = crate::media::audio::AudioHandle::new();
            *slot = Some(handle.clone());
            handle
        }
    };
    audio_handle.set_available(false);
    audio_handle.set_audio_stalled(false);
    audio_handle.clear_playback_epoch();
    audio_handle.reset_samples_played();
    audio_handle.clear_audio_base_pts();
    audio_handle.clear_stream_pts_offset();

    let audio_codec = audio_codec_from_config(audio_cfg);
    *shared.audio_codec_name.lock() = Some(format!("{:?}", audio_codec));

    match MoqAudioThread::spawn(
        rx,
        audio_cfg.sample_rate,
        audio_cfg.channel_count,
        audio_cfg.description.clone(),
        audio_codec,
        audio_handle.clone(),
        shared.audio.clone(),
    ) {
        Ok(thread) => {
            tracing::info!(
                "MoQ {}: Audio subscribed to track '{}' ({:?}, {}Hz, {}ch)",
                label,
                track_name,
                audio_codec,
                audio_cfg.sample_rate,
                audio_cfg.channel_count,
            );
            (Some(audio_consumer), Some(live_sender), Some(thread))
        }
        Err(e) => {
            tracing::warn!("MoQ {}: Failed to start audio thread: {e}", label);
            *shared.audio.audio_status.lock() = MoqAudioStatus::Error;
            audio_handle.set_available(false);
            (None, None, None)
        }
    }
}

/// Spawn a dedicated tokio task that forwards audio frames from the MoQ
/// `OrderedConsumer` to the crossbeam `LiveEdgeSender`.
///
/// Returns `None` if consumer or sender is `None` (audio disabled / unavailable).
///
/// IMPORTANT: `OrderedConsumer::read()` is NOT cancellation-safe — it must run
/// in its own dedicated task, never in a shared `tokio::select!` loop.
fn spawn_audio_forward_task(
    consumer: Option<hang::container::OrderedConsumer>,
    sender: Option<LiveEdgeSender<MoqAudioFrame>>,
    audio_shared: &Arc<crate::media::moq_decoder::MoqAudioShared>,
    label: &str,
) -> Option<tokio::task::JoinHandle<()>> {
    let (mut audio_consumer, audio_sender) = match (consumer, sender) {
        (Some(c), Some(s)) => (c, s),
        _ => return None,
    };
    let label_owned = label.to_string();
    let audio_shared = audio_shared.clone();

    Some(tokio::spawn(async move {
        tracing::info!("MoQ {}: audio forward task started", label_owned);
        // Initial heartbeat so the watchdog knows we're alive before first frame.
        *audio_shared.last_audio_forward_frame_at.lock() = Some(Instant::now());
        let mut buf = BytesMut::with_capacity(4096);
        let mut frames_forwarded: u64 = 0;
        // Internal timeout: 8s to tolerate Opus DTX / silence suppression
        // periods where the relay legitimately sends no audio packets.
        // On timeout, exit immediately — do NOT retry on the same consumer.
        // OrderedConsumer::read() is not cancellation-safe (two internal await
        // points: next_frame + read_chunks). Cancelling mid-flight corrupts the
        // consumer's internal state, so subsequent reads would hang or return
        // garbage. The main loop re-subscribes with a completely fresh consumer.
        //
        // The external watchdog (AUDIO_WATCHDOG_TIMEOUT, 10s) in the main
        // select! loop provides defense-in-depth if this timeout fails to fire.
        const READ_TIMEOUT: Duration = Duration::from_secs(8);
        let mut last_periodic_log = std::time::Instant::now();
        loop {
            match tokio::time::timeout(READ_TIMEOUT, audio_consumer.read()).await {
                Ok(Ok(Some(frame))) => {
                    let data = assemble_payload(&frame.payload, &mut buf);
                    let pts_us = frame.timestamp.as_micros() as u64;
                    let moq_frame = MoqAudioFrame {
                        timestamp_us: pts_us,
                        data,
                    };
                    match audio_sender.send(moq_frame) {
                        Ok(()) => {
                            frames_forwarded += 1;
                            // Heartbeat AFTER successful send — tracks forwarded
                            // delivery, not just QUIC read liveness.
                            *audio_shared.last_audio_forward_frame_at.lock() = Some(Instant::now());
                        }
                        Err(ChannelClosed) => {
                            tracing::warn!(
                                "MoQ {}: Audio channel closed (after {} frames)",
                                label_owned,
                                frames_forwarded,
                            );
                            break;
                        }
                    }
                    // Periodic log every 5s to confirm liveness
                    if last_periodic_log.elapsed() >= Duration::from_secs(5) {
                        last_periodic_log = std::time::Instant::now();
                        tracing::info!(
                            "MoQ {}: audio forward alive: {} frames forwarded, last_pts={}us",
                            label_owned,
                            frames_forwarded,
                            pts_us,
                        );
                    }
                }
                Ok(Ok(None)) => {
                    tracing::info!(
                        "MoQ {}: Audio track ended (after {} frames)",
                        label_owned,
                        frames_forwarded,
                    );
                    break;
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        "MoQ {}: Audio read error (after {} frames): {e}",
                        label_owned,
                        frames_forwarded,
                    );
                    break;
                }
                Err(_elapsed) => {
                    // Single strike — exit immediately, don't reuse corrupted consumer
                    tracing::warn!(
                        "MoQ {}: Audio read timeout ({}s, after {} frames) — consumer stalled, exiting for re-subscribe",
                        label_owned,
                        READ_TIMEOUT.as_secs(),
                        frames_forwarded,
                    );
                    break;
                }
            }
        }
        tracing::info!(
            "MoQ {}: Audio forward task exiting (total frames forwarded: {})",
            label_owned,
            frames_forwarded,
        );
    }))
}

/// Assemble a hang payload (chunked BufList) into a contiguous `Bytes`,
/// reusing `buf` to avoid per-frame allocation.
///
/// After `split().freeze()`, the `BytesMut` retains its allocation for reuse.
fn assemble_payload(payload: &hang::container::BufList, buf: &mut BytesMut) -> bytes::Bytes {
    buf.clear();
    let needed = payload.remaining();
    buf.reserve(needed);
    for chunk in payload {
        buf.extend_from_slice(chunk);
    }
    buf.split().freeze()
}

/// Deterministic audio thread shutdown with 2s timeout.
async fn teardown_audio(
    audio_sender: Option<LiveEdgeSender<MoqAudioFrame>>,
    audio_thread: Option<MoqAudioThread>,
    shared: &Arc<MoqSharedState>,
    label: &str,
    cause: &str,
) {
    tracing::info!(
        "MoQ {}: teardown_audio: cause='{}', sender={}, thread={}",
        label,
        cause,
        audio_sender.is_some(),
        audio_thread.is_some(),
    );
    drop(audio_sender);
    {
        let mut status = shared.audio.audio_status.lock();
        if *status == MoqAudioStatus::Starting {
            *status = MoqAudioStatus::Unavailable;
        }
    }
    if let Some(thread) = audio_thread {
        let shared_for_teardown = shared.clone();
        let teardown_start = std::time::Instant::now();
        let teardown_fut = tokio::task::spawn_blocking(move || drop(thread));
        match tokio::time::timeout(Duration::from_secs(2), teardown_fut).await {
            Ok(Ok(())) => tracing::debug!("MoQ {}: audio teardown completed", label),
            Ok(Err(e)) => {
                tracing::warn!("MoQ {}: audio teardown task failed: {e}", label);
                *shared_for_teardown.audio.audio_status.lock() = MoqAudioStatus::Error;
            }
            Err(_) => {
                tracing::warn!(
                    "MoQ {}: audio teardown timed out after 2s, proceeding",
                    label,
                );
            }
        }
        shared_for_teardown
            .audio
            .internal_audio_ready
            .store(false, Ordering::Relaxed);
        shared_for_teardown
            .audio
            .alive
            .store(false, Ordering::Release);
        *shared_for_teardown.audio.last_audio_forward_frame_at.lock() = None;
        if let Some(handle) = shared_for_teardown.audio.moq_audio_handle.lock().as_ref() {
            handle.set_available(false);
            handle.set_audio_stalled(false);
            handle.clear_playback_epoch();
            handle.reset_samples_played();
            handle.clear_audio_base_pts();
            handle.clear_stream_pts_offset();
        }
        {
            let mut status = shared_for_teardown.audio.audio_status.lock();
            if *status == MoqAudioStatus::Running || *status == MoqAudioStatus::Starting {
                *status = MoqAudioStatus::Unavailable;
            }
        }
        *shared_for_teardown.audio_codec_name.lock() = None;
        let teardown_ms = teardown_start.elapsed().as_millis();
        if teardown_ms > 250 {
            tracing::warn!("MoQ {}: audio teardown took {}ms", label, teardown_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::moq::MoqUrl;

    #[test]
    fn test_sanitize_path_strips_trailing_slashes() {
        assert_eq!(sanitize_path("namespace/"), "namespace");
        assert_eq!(sanitize_path("namespace///"), "namespace");
        assert_eq!(sanitize_path("namespace"), "namespace");
        assert_eq!(sanitize_path(""), "");
        assert_eq!(sanitize_path("/"), "");
    }

    #[test]
    fn test_build_connect_url_cdn_moq_dev_style() {
        // cdn.moq.dev style: namespace in URL, track as broadcast path
        let url = MoqUrl::parse("moqs://cdn.moq.dev/bbb").unwrap();
        let (connect, path) = build_connect_url(&url);
        assert_eq!(connect, "https://cdn.moq.dev:443/bbb");
        assert!(path.is_none()); // no track → no broadcast path
    }

    #[test]
    fn test_build_connect_url_cdn_moq_dev_with_track() {
        let url = MoqUrl::parse("moqs://cdn.moq.dev/bbb/video0").unwrap();
        let (connect, path) = build_connect_url(&url);
        assert_eq!(connect, "https://cdn.moq.dev:443/bbb");
        assert_eq!(path.as_ref().map(|p| p.to_string()), Some("video0".into()));
    }

    #[test]
    fn test_build_connect_url_uuid_broadcast() {
        // zap.stream style: UUID namespace → broadcast path
        let url =
            MoqUrl::parse("moqs://relay.example.com/537a365c-f1ec-44ac-af10-22d14a7319fb").unwrap();
        let (connect, path) = build_connect_url(&url);
        assert_eq!(connect, "https://relay.example.com:443");
        assert_eq!(
            path.as_ref().map(|p| p.to_string()),
            Some("537a365c-f1ec-44ac-af10-22d14a7319fb".into())
        );
    }

    #[test]
    fn test_build_connect_url_trailing_slash_sanitized() {
        // Ensure trailing slashes in track are stripped (prevents announced() panic)
        let url = MoqUrl::parse("moqs://cdn.moq.dev/bbb/video0/").unwrap();
        let (_connect, path) = build_connect_url(&url);
        // track() returns "video0/" from the parser, sanitize_path strips the slash
        if let Some(p) = path {
            assert!(!p.to_string().ends_with('/'), "path should not end with /");
        }
    }

    #[test]
    fn test_build_connect_url_localhost_moq_uses_http() {
        // moq:// → http (self-signed cert dance), moqs:// → https
        let url = MoqUrl::parse("moq://localhost:4443/test").unwrap();
        let (connect, _path) = build_connect_url(&url);
        assert!(
            connect.starts_with("http://"),
            "moq:// should use http (moq-native fetches self-signed cert via HTTP)"
        );

        let url_tls = MoqUrl::parse("moqs://localhost:4443/test").unwrap();
        let (connect_tls, _) = build_connect_url(&url_tls);
        assert!(
            connect_tls.starts_with("https://"),
            "moqs:// should use https"
        );
    }

    #[test]
    fn test_build_connect_url_with_jwt_query() {
        let url = MoqUrl::parse("moqs://cdn.moq.dev/bbb?jwt=abc123").unwrap();
        let (connect, _path) = build_connect_url(&url);
        assert!(connect.contains("?jwt=abc123"));
    }
}
