# A/V Synchronization

Technical details on lumina-video's audio-video synchronization implementation.

## Overview

lumina-video supports two audio playback modes:

1. **Native audio** (MP4, MOV via VideoToolbox/GStreamer/ExoPlayer): The platform decoder handles A/V sync internally with perfect synchronization.

2. **FFmpeg fallback** (MKV, WebM): Separate audio thread with sample-based clock synchronization.

## Native Playback (Recommended)

For natively-supported formats (MP4, MOV, HLS), the platform decoder handles both video and audio with built-in A/V sync:

- **macOS**: AVPlayer (VideoToolbox)
- **Linux**: GStreamer
- **Android**: ExoPlayer
- **Windows**: Media Foundation

**No drift measurement** — sync is handled internally by the platform.

## FFmpeg Fallback (MKV/WebM)

For containers not supported natively, FFmpeg decodes video while a separate audio thread handles playback. This requires explicit synchronization.

### Measured Performance (MKV on macOS)

| Scenario | Typical Drift | Notes |
|----------|---------------|-------|
| Initial playback | +100-200ms | Stabilizes after buffering |
| After seek | -50 to +50ms | Brief resync period |
| After pause/resume | +500-2000ms | Known issue, requires seek to recover |

- Drift can spike significantly during pause/resume operations
- Seek helps resynchronize audio and video clocks

> **Platform Note**: Results vary by audio subsystem. CoreAudio (macOS), PulseAudio/PipeWire (Linux), and AAudio (Android) have different buffer characteristics.

## FFmpeg Audio Implementation Details

The FFmpeg fallback uses a **callback-based audio clock** for synchronization:

- **Sample counting**: Audio samples counted as consumed by rodio, batched every 256 samples to minimize atomic operations
- **Shared playback epoch**: Video and audio clocks synchronize when the first video frame is displayed
- **PTS offset handling**: Compensates for streams where audio and video have different start timestamps
- **Recovery tracking**: Metrics exclude post-stall recovery periods for accurate steady-state measurement

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

## Threading Model (FFmpeg Fallback)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   UI Thread     │     │  Decode Thread  │     │  Audio Thread   │
│                 │     │                 │     │                 │
│  VideoPlayer    │◄────│  FrameQueue     │     │  AudioDecoder   │
│  renders frame  │     │  buffers frames │     │  syncs playback │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

For native playback, the platform decoder handles threading internally.
