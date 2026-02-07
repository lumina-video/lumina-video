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

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use async_channel::Sender;
use bytes::{Buf, BytesMut};
use moq_lite::{Origin, PathOwned};
use moq_native::ClientConfig;

use crate::media::moq::MoqUrl;
use crate::media::moq_audio::{
    select_preferred_audio_rendition, ChannelClosed, LiveEdgeSender, MoqAudioFrame, MoqAudioThread,
};
use crate::media::moq_decoder::{
    MoqAudioStatus, MoqDecoderConfig, MoqDecoderState, MoqSharedState, MoqVideoFrame,
};

/// Shared async worker for MoQ connection, catalog fetch, and frame receipt.
///
/// All three platform decoders (macOS, Android, Linux/GStreamer) delegate to
/// this function. The `label` parameter is used for tracing prefixes.
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
    let hang_consumer: hang::BroadcastConsumer = moq_broadcast.into();

    let (video_track_name, max_latency, catalog) =
        fetch_and_validate_catalog(&hang_consumer, &shared, &config, label).await?;

    // -- Phase 4: Subscribe to video track --
    let video_track = moq_lite::Track {
        name: video_track_name.clone(),
        priority: 1,
    };
    let mut video_consumer = hang_consumer.subscribe(&video_track, max_latency);

    // -- Phase 5: Audio setup --
    let (mut audio_consumer_opt, audio_sender_opt, mut moq_audio_thread_opt) = setup_audio(
        &catalog,
        &hang_consumer,
        max_latency,
        &config,
        &shared,
        label,
    );

    // -- Phase 6: Streaming --
    shared.set_state(MoqDecoderState::Streaming);
    shared.buffering_percent.store(100, Ordering::Relaxed);

    tracing::info!(
        "MoQ {}: Streaming started, subscribed to video track '{}'",
        label,
        video_track_name,
    );

    // Pre-allocate reusable buffers to avoid per-frame allocation
    let mut video_buf = BytesMut::with_capacity(256 * 1024);
    let mut audio_buf = BytesMut::with_capacity(4096);

    let mut stats_log_counter = 0u64;
    loop {
        tokio::select! {
            // No biased — fair scheduling prevents audio starvation
            video_result = video_consumer.read_frame() => {
                match video_result {
                    Ok(Some(frame)) => {
                        let recv_count =
                            shared.frame_stats.received.fetch_add(1, Ordering::Relaxed) + 1;

                        stats_log_counter += 1;
                        #[allow(clippy::manual_is_multiple_of)] // MSRV 1.83: is_multiple_of requires 1.87+
                        if stats_log_counter % 30 == 0 {
                            shared.frame_stats.log_summary(label);
                        }

                        let data = assemble_payload(&frame.payload, &mut video_buf);

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

                        if frame_tx.send(moq_frame).await.is_err() {
                            tracing::warn!("MoQ {}: Frame channel closed, stopping worker", label);
                            break;
                        }
                    }
                    Ok(None) => {
                        tracing::info!("MoQ {}: Video track ended", label);
                        shared.frame_stats.log_summary(label);
                        shared.set_state(MoqDecoderState::Ended);
                        shared.eof_reached.store(true, Ordering::Relaxed);
                        break;
                    }
                    Err(e) => {
                        tracing::error!("MoQ {}: Frame read error: {e}", label);
                        shared.frame_stats.log_summary(label);
                        shared.set_error(format!("Frame read error: {e}"));
                        break;
                    }
                }
            }
            audio_result = async {
                if let Some(consumer) = audio_consumer_opt.as_mut() {
                    consumer.read_frame().await
                } else {
                    std::future::pending().await
                }
            } => {
                if let Some(ref audio_sender) = audio_sender_opt {
                    match audio_result {
                        Ok(Some(frame)) => {
                            let data = assemble_payload(&frame.payload, &mut audio_buf);
                            let moq_frame = MoqAudioFrame {
                                timestamp_us: frame.timestamp.as_micros() as u64,
                                data,
                            };
                            if let Err(ChannelClosed) = audio_sender.send(moq_frame) {
                                tracing::warn!("MoQ {}: Audio channel closed", label);
                                audio_consumer_opt = None;
                            }
                        }
                        Ok(None) => {
                            tracing::info!("MoQ {}: Audio track ended", label);
                            audio_consumer_opt = None;
                        }
                        Err(e) => {
                            tracing::warn!("MoQ {}: Audio read error: {e}", label);
                            audio_consumer_opt = None;
                        }
                    }
                }
            }
        }
    }

    // -- Phase 7: Teardown --
    teardown_audio(
        audio_sender_opt,
        moq_audio_thread_opt.take(),
        &shared,
        label,
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
    let is_localhost =
        url.host() == "localhost" || url.host() == "127.0.0.1" || url.host() == "::1";
    let scheme = if url.use_tls() || !is_localhost {
        "https"
    } else {
        "http"
    };

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
    hang_consumer: &hang::BroadcastConsumer,
    shared: &Arc<MoqSharedState>,
    config: &MoqDecoderConfig,
    label: &str,
) -> Result<(String, Duration, hang::catalog::Catalog), Box<dyn std::error::Error + Send + Sync>> {
    let mut catalog_consumer = hang_consumer.catalog.clone();
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
    if let Some(ref video) = catalog.video {
        tracing::info!(
            "MoQ {} catalog: {} video rendition(s)",
            label,
            video.renditions.len()
        );
        for (name, cfg) in &video.renditions {
            tracing::info!(
                "  video '{}': codec={:?}, {}x{}, {:.1}fps, bitrate={:?}, description={} bytes, container={:?}, jitter={:?}",
                name,
                cfg.codec,
                cfg.coded_width.unwrap_or(0),
                cfg.coded_height.unwrap_or(0),
                cfg.framerate.unwrap_or(0.0),
                cfg.bitrate,
                cfg.description.as_ref().map(|d| d.len()).unwrap_or(0),
                cfg.container,
                cfg.jitter,
            );
        }
    } else {
        tracing::warn!("MoQ {} catalog: no video section", label);
    }
    if let Some(ref audio) = catalog.audio {
        tracing::info!(
            "MoQ {} catalog: {} audio rendition(s)",
            label,
            audio.renditions.len()
        );
        for (name, cfg) in &audio.renditions {
            tracing::info!(
                "  audio '{}': codec={:?}, sample_rate={}, channels={}, bitrate={:?}, description={} bytes",
                name,
                cfg.codec,
                cfg.sample_rate,
                cfg.channel_count,
                cfg.bitrate,
                cfg.description.as_ref().map(|d| d.len()).unwrap_or(0),
            );
        }
    } else {
        tracing::info!("MoQ {} catalog: no audio section", label);
    }

    // Find the first video rendition in the catalog
    let (video_track_name, video_config) = catalog
        .video
        .as_ref()
        .and_then(|v| v.renditions.iter().next())
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
        metadata.frame_rate = video_config.framerate.unwrap_or(30.0) as f32;
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
        tracing::info!(
            "MoQ {}: Using catalog jitter {}ms as latency buffer",
            label,
            jitter_ms,
        );
        Duration::from_millis(jitter_ms)
    } else {
        tracing::info!(
            "MoQ {}: No jitter in catalog, using default {}ms",
            label,
            config.max_latency_ms,
        );
        Duration::from_millis(config.max_latency_ms)
    };

    Ok((video_track_name.clone(), max_latency, catalog))
}

/// Set up audio track subscription and spawn decode/playback thread.
fn setup_audio(
    catalog: &hang::catalog::Catalog,
    hang_consumer: &hang::BroadcastConsumer,
    max_latency: Duration,
    config: &MoqDecoderConfig,
    shared: &Arc<MoqSharedState>,
    label: &str,
) -> (
    Option<hang::TrackConsumer>,
    Option<LiveEdgeSender<MoqAudioFrame>>,
    Option<MoqAudioThread>,
) {
    if !config.enable_audio {
        return (None, None, None);
    }

    let (track_name, audio_cfg) = match select_preferred_audio_rendition(catalog) {
        Some(pair) => pair,
        None => {
            tracing::info!("MoQ {}: No AAC audio track in catalog", label);
            *shared.audio.audio_status.lock() = MoqAudioStatus::Unavailable;
            return (None, None, None);
        }
    };

    *shared.audio.audio_status.lock() = MoqAudioStatus::Starting;

    let audio_track = moq_lite::Track {
        name: track_name.to_string(),
        priority: 2,
    };
    let audio_consumer = hang_consumer.subscribe(&audio_track, max_latency);

    let (tx, rx) = crossbeam_channel::bounded(config.audio_buffer_capacity);
    let live_sender = LiveEdgeSender::new(tx, rx.clone());

    let audio_handle = crate::media::audio::AudioHandle::new();
    *shared.audio.moq_audio_handle.lock() = Some(audio_handle.clone());

    match MoqAudioThread::spawn(
        rx,
        audio_cfg.sample_rate,
        audio_cfg.channel_count,
        audio_cfg.description.clone(),
        audio_handle,
        shared.audio.clone(),
    ) {
        Ok(thread) => {
            tracing::info!(
                "MoQ {}: Audio subscribed to track '{}' ({}Hz, {}ch)",
                label,
                track_name,
                audio_cfg.sample_rate,
                audio_cfg.channel_count,
            );
            (Some(audio_consumer), Some(live_sender), Some(thread))
        }
        Err(e) => {
            tracing::warn!("MoQ {}: Failed to start audio thread: {e}", label);
            *shared.audio.audio_status.lock() = MoqAudioStatus::Error;
            *shared.audio.moq_audio_handle.lock() = None;
            (None, None, None)
        }
    }
}

/// Assemble a hang payload (chunked BufList) into a contiguous `Bytes`,
/// reusing `buf` to avoid per-frame allocation.
///
/// After `split().freeze()`, the `BytesMut` retains its allocation for reuse.
fn assemble_payload(payload: &hang::BufList, buf: &mut BytesMut) -> bytes::Bytes {
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
) {
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
        *shared_for_teardown.audio.moq_audio_handle.lock() = None;
        {
            let mut status = shared_for_teardown.audio.audio_status.lock();
            if *status == MoqAudioStatus::Running || *status == MoqAudioStatus::Starting {
                *status = MoqAudioStatus::Unavailable;
            }
        }
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
    fn test_build_connect_url_localhost_no_tls() {
        let url = MoqUrl::parse("moq://localhost:4443/test").unwrap();
        let (connect, _path) = build_connect_url(&url);
        assert!(
            connect.starts_with("http://"),
            "localhost moq:// should use http"
        );
    }

    #[test]
    fn test_build_connect_url_with_jwt_query() {
        let url = MoqUrl::parse("moqs://cdn.moq.dev/bbb?jwt=abc123").unwrap();
        let (connect, _path) = build_connect_url(&url);
        assert!(connect.contains("?jwt=abc123"));
    }
}
