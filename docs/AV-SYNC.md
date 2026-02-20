# A/V Synchronization

Technical details on lumina-video's audio-video synchronization implementation.

## Overview

lumina-video supports three audio playback modes:

1. **Native audio** (MP4, MOV via VideoToolbox/GStreamer/ExoPlayer): The platform decoder handles A/V sync internally with perfect synchronization.

2. **FFmpeg fallback** (MKV, WebM): Separate audio thread with sample-based clock synchronization via cpal.

3. **MoQ live streaming** (AAC/Opus over QUIC): Wall-clock pacing with proportional drift correction and stall-on-underrun.

## Native Playback (Recommended)

For natively-supported formats (MP4, MOV, HLS), the platform decoder handles both video and audio with built-in A/V sync:

- **macOS**: AVPlayer (VideoToolbox)
- **Linux**: GStreamer
- **Android**: ExoPlayer
- **Windows**: Media Foundation

**No drift measurement** — sync is handled internally by the platform.

## FFmpeg Fallback (MKV/WebM)

For containers not supported natively, FFmpeg decodes video while a separate audio thread handles playback via cpal. This requires explicit synchronization.

### Audio Pipeline

FFmpeg decodes audio → samples written to a lock-free SPSC ring buffer → cpal output stream callback reads samples for playback. The ring buffer (`audio_ring_buffer.rs`) uses atomic operations for zero-lock producer/consumer coordination.

### Measured Performance (MKV on macOS)

| Scenario | Typical Drift | Notes |
|----------|---------------|-------|
| Initial playback | +100-200ms | Stabilizes after buffering |
| After seek | -50 to +50ms | Brief resync period |
| After pause/resume | +500-2000ms | Known issue, requires seek to recover |

- Drift can spike significantly during pause/resume operations
- Seek helps resynchronize audio and video clocks

> **Platform Note**: Results vary by audio subsystem. CoreAudio (macOS), PulseAudio/PipeWire (Linux), and AAudio (Android) have different buffer characteristics.

### Implementation Details

The FFmpeg fallback uses a **callback-based audio clock** for synchronization:

- **Sample counting**: Audio samples counted as consumed by the cpal callback, batched every 256 samples to minimize atomic operations
- **Shared playback epoch**: Video and audio clocks synchronize when the first video frame is displayed
- **PTS offset handling**: Compensates for streams where audio and video have different start timestamps
- **Recovery tracking**: Metrics exclude post-stall recovery periods for accurate steady-state measurement

### Threading Model

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   UI Thread     │     │  Decode Thread  │     │  Audio Thread   │
│                 │     │                 │     │  (cpal callback)│
│  VideoPlayer    │◄────│  FrameQueue     │     │  Ring Buffer    │
│  renders frame  │     │  buffers frames │     │  → speaker out  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲                                              │
         └──────── samples_played (atomic) ─────────────┘
```

## MoQ Live Streaming (AAC + Opus)

MoQ live audio uses a fundamentally different sync model than VOD. There is no seekable timeline — audio arrives in real-time over QUIC, and video must pace itself against wall-clock time while staying aligned with audio.

### Audio Pipeline

QUIC delivers encoded audio frames (AAC or Opus) which are decoded and played via cpal:

```
QUIC Stream ──► Async Worker ──► crossbeam channel ──► Audio Thread ──► cpal callback
                (tokio task)     (live-edge sender)    (decode loop)    (ring buffer read)
```

- **AAC**: Decoded by symphonia (pure Rust). Frame rate = sample_rate / 1024 (e.g., 48kHz → 46.875 fps).
- **Opus**: Decoded by libopus via the `opus` crate. Frame rate ≈ sample_rate / 960 at 48kHz.
- **Live-edge channel**: Bounded crossbeam channel that drops oldest frames when full, maintaining the live edge.

### Cancellation Safety

`hang::container::OrderedConsumer::read()` is **not cancellation-safe** — its internal `read_unbuffered()` has two await points. If cancelled between them (e.g., by `tokio::select!`), a frame is consumed from QUIC but never returned.

Audio MUST run in a dedicated `tokio::spawn` task, never in a shared `select!` loop with video. Violating this caused ~50% audio frame loss in testing.

### Wall-Clock Pacing

Unlike VOD where audio is the sync master, MoQ uses **wall-clock pacing** for video:

1. When the first video frame arrives, a playback epoch is established (wall-clock time mapped to PTS).
2. Subsequent frames are scheduled based on their PTS offset from the epoch, measured against wall-clock elapsed time.
3. A `video_pts_bias` adjustment keeps video aligned with the audio clock.

MoQ PTS values use publisher wall-clock time (e.g., ~7200s into a stream) while the viewer starts at ~0s. The audio handle is **not** bound to `FrameScheduler` to avoid this position jump freezing video.

### Proportional Drift Controller

A time-based proportional controller compensates for clock drift between the audio sample clock and the system wall clock:

1. **EMA smoothing** (α=0.3): Raw drift is exponentially smoothed to filter jitter before feeding the controller.
2. **Hysteresis deadband**: Correction activates when |drift| > 40ms, deactivates when |drift| < 20ms. Prevents oscillation near zero.
3. **Proportional slew**: Adjusts video pacing at most 10ms/s (1% speed change), clamped to ±500ms total correction.
4. **Step detection**: Drift > 100ms indicates a sudden desync (e.g., app backgrounding). Instead of slow proportional correction, the wall clock is immediately resynced to audio.
5. **Stall gating**: During audio stalls (ring buffer underrun or queue empty), the controller resets its EMA and hysteresis state to avoid wrong-way corrections after recovery.

### Stall-on-Underrun

When the audio ring buffer empties (3 consecutive empty cpal callbacks), video freezes to prevent A/V drift from accumulating:

- `AudioHandle.is_audio_stalled()` gates video frame advancement in `FrameScheduler`.
- When audio recovers (ring buffer refills), video resumes and the drift controller resyncs.
- This prevents the common failure mode where video runs ahead during network stalls, creating a growing offset.

### Rendering Gap Detection (Alt-Tab Recovery)

When the app is backgrounded (alt-tab), egui stops calling `get_next_frame()`. Audio continues playing via the cpal callback, creating drift equal to the gap duration.

Detection and recovery:

1. If `get_next_frame()` hasn't been called for >100ms, a rendering gap is detected.
2. Stale frames accumulated during the gap are drained (up to 100ms worth).
3. The wall clock is immediately resynced to the current audio position.
4. The drift controller resets (EMA, hysteresis, accumulated correction).
5. A grace period filters transient drift spikes during recovery.

### MoQ Threading Model

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ QUIC/Network │    │   Async Runtime  │    │   Audio Thread     │
│              │    │   (tokio)        │    │                    │
│  Video Track ─────► Video Worker     │    │                    │
│              │    │  ├─ decode (VT)   │    │                    │
│              │    │  └─► FrameQueue   │    │                    │
│              │    │                   │    │                    │
│  Audio Track ─────► Audio Forward*   │    │  crossbeam recv    │
│              │    │  (dedicated task) │    │  ├─ AAC/Opus decode│
│              │    │                   │    │  └─► Ring Buffer   │
└──────────────┘    └──────────────────┘    └─────────┬──────────┘
                                                      │
                    ┌──────────────────┐    ┌──────────▼──────────┐
                    │   UI Thread      │    │   cpal Callback     │
                    │                  │    │                     │
                    │  FrameScheduler  │    │  Ring Buffer read   │
                    │  ├─ drift ctrl   │    │  └─► speaker output │
                    │  ├─ stall gate   │    │                     │
                    │  └─ frame select │    │  samples_played     │
                    └──────────────────┘    │  (atomic update)    │
                                           └─────────────────────┘

* Audio forward task is a dedicated tokio::spawn — never in a shared select!
  (see Cancellation Safety above)
```

### Sync Targets

| Metric | Threshold |
|--------|-----------|
| Excellent | <33ms drift (1 frame at 30fps) |
| Acceptable | <100ms drift |
| Warning | <150ms (noticeable but tolerable) |
| Severe | >200ms (clearly out of sync) |

## Sync Metrics API

```rust
// Get current sync status
let snapshot = player.sync_metrics_snapshot();

println!("Current drift: {}ms", snapshot.current_drift_ms());
println!("Steady-state avg: {}ms", snapshot.steady_avg_drift_ms());
println!("Out of sync: {:.1}%", snapshot.steady_out_of_sync_percentage());
println!("Quality: {}", snapshot.quality_summary());
```

## Baseline Testing Tool

Run the sync baseline tool to measure A/V sync on your system:

```bash
# Default: 30 seconds with remote test video
cargo run -p lumina-video --example sync_baseline --features "ffmpeg,macos-native-video"

# Local file with custom duration
cargo run -p lumina-video --example sync_baseline --features "ffmpeg,macos-native-video" -- /path/to/video.mp4 60
```

## Further Reading

- **[MoQ Best Practices](../MOQ_BEST_PRACTICES.md)** — protocol-level rules, priority scheduling, cancellation safety
- **[Platforms](PLATFORMS.md)** — per-platform decoder and audio subsystem details
