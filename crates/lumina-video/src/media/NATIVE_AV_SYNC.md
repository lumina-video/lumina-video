# Native A/V Sync Best Practices

This document captures lessons learned from implementing A/V synchronization
for native video decoders (AVFoundation, MediaFoundation, GStreamer, ExoPlayer).

## Expert Feedback → Universal Principles

| Expert Said (FFmpeg)                              | Universal Principle         | Native Application                                                   |
|---------------------------------------------------|-----------------------------|--------------------------------------------------------------------|
| "Audio packets are usually shorter than video frames" | Audio is the master clock   | Sync video presentation to AVPlayer's audio position               |
| "Binary heap to deliver both A/V packets"         | PTS ordering matters        | Present video frames in strict PTS order                           |
| "Adjustable buffer target"                        | Don't consume too early     | Buffer video frames before starting playback                       |
| "Order decoded AVFrames, not AVPackets"           | Sync after decode           | AVFoundation handles B-frame reordering, but we control presentation |
| "<100ms sync, ideally <1 frame"                   | Tight tolerance             | Target <33ms (1 frame at 30fps)                                    |

## Key Insight: Audio as Master Clock

### The Problem

Wall-clock time starts immediately when playback begins, but audio has buffer
latency before it actually plays through speakers. If video frame selection
uses wall-clock time, video will advance ahead of actual audio output.

```text
Timeline:
  t=0     Wall-clock starts, video starts rendering frames
  t=50ms  Audio buffer fills
  t=100ms Audio actually starts playing through speakers

Result: Video is ~100ms ahead of what you HEAR
```

### The Solution

Use audio position as the master clock for video frame selection:

```rust
// BAD: Wall-clock based frame selection
let current_pos = wall_clock.elapsed();
let should_show = frame.pts <= current_pos;

// GOOD: Audio-based frame selection
let current_pos = audio_handle.position_for_sync();
let should_show = frame.pts <= current_pos;
```

This ties video presentation directly to audio playback position, automatically
accounting for audio buffer latency.

## Native Sync Best Practices

### 1. Audio as Master Clock

- Use `AudioHandle.position_for_sync()` for frame timing decisions
- Fall back to wall-clock only when audio is unavailable
- Audio position accounts for buffer latency automatically

### 2. Frame Dropping/Holding

- **Video behind audio** → Skip frames to catch up
- **Video ahead of audio** → Hold current frame (don't advance)

```rust
loop {
    let next_pts = queue.peek_pts();
    if next_pts <= audio_position {
        // Frame is ready or late - display it (may skip if very late)
        frame = queue.pop();
    } else {
        // Frame is early - hold current frame
        return current_frame;
    }
}
```

### 3. Buffer Before Play

- Don't start video until ~100ms of frames are buffered
- Prevents "jumping around" at playback start
- Use `FrameHeap` with configurable `buffer_target_ms`

### 4. Precise Audio Position

For even tighter sync on macOS:
- `AVPlayer.currentTime()` has some latency
- Consider `AVPlayerItemVideoOutput.itemTime(forHostTime:)` for tighter coupling
- Host time allows correlating video frames to exact audio position

### 5. Continuous Drift Correction

- Monitor drift via `SyncMetrics`
- Log warnings when drift exceeds threshold (80ms)
- Consider gradual playback rate adjustment for persistent drift

## Platform-Specific Notes

### macOS (AVFoundation)

- AVPlayer handles both audio and video internally
- Video frames extracted via `AVPlayerItemVideoOutput`
- Audio position from `AVPlayer.currentTime()`
- B-frame reordering handled by VideoToolbox

### Windows (MediaFoundation)

- Similar to macOS: framework handles both A/V
- Use `IMFMediaSession` for playback control
- D3D11 surfaces for zero-copy video

### Linux (GStreamer)

- Pipeline handles A/V sync internally
- Can extract clock position from pipeline
- DMABuf for zero-copy video

### Android (ExoPlayer)

- ExoPlayer handles A/V sync internally
- Surface-based rendering with AHardwareBuffer

## Implementation Reference

Key files:
- `frame_queue.rs` - `FrameScheduler.sync_position()` uses audio as master clock
- `audio.rs` - `AudioHandle.position_for_sync()` returns normalized audio position
- `sync_metrics.rs` - `SyncMetrics` tracks drift and warns when out of tolerance

## Metrics

Target sync tolerance:
- **Acceptable:** <100ms drift
- **Good:** <50ms drift
- **Excellent:** <33ms drift (1 frame at 30fps)
- **Professional:** <1 frame at any framerate

The `SyncMetrics` class logs warnings when drift exceeds 80ms.
