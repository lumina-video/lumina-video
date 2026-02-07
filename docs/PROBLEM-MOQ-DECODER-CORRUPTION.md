# Technical Problem Statement: MoQ VTDecoder Frame Corruption

## Observed Behavior

When playing MoQ live streams via VTDecompressionSession on macOS:

1. **Initial frames show pixelation** - blocky artifacts, corrupted macroblocks
2. **Very low/zero measured FPS** - frames not being delivered to renderer
3. **Eventual freeze** - video stops updating entirely

![Screenshot showing pixelation](Screenshot-moq-pixelation.png)

## Environment

- Platform: macOS (Apple Silicon M4)
- Decoder: VTDecompressionSession (VideoToolbox hardware H.264)
- Source: MoQ live stream via hang crate (`moqs://cdn.moq.dev/demo/bbb`)
- Resolution: 1280x720 @ 30fps
- Codec: H.264 (AVCC format)

## Root Cause Analysis

### Primary Issue: Missing Reference Frames

The pixelation pattern is characteristic of **H.264 inter-frame decoding without valid references**:

```text
GOP Structure:  I  P  P  P  P  I  P  P  P  P  ...
                ↑              ↑
                IDR            IDR

When joining mid-stream:
                      ↓ (join here)
                I  P  P  P  P  I  P  P  P  P
                      ↑
                      Decoder receives P-frame but has no reference
                      → Outputs corrupted frame (pixelation)
```

### Contributing Factors

#### 1. IDR Detection May Be Incorrect

**Location:** `crates/lumina-video/src/media/moq_decoder.rs:547-570`

The `is_idr_frame()` function iterates AVCC NAL units looking for type 5 (IDR):

```rust
fn is_idr_frame(nal_data: &[u8]) -> bool {
    let mut offset = 0usize;
    while offset + 4 <= nal_data.len() {
        let nal_len = u32::from_be_bytes([...]) as usize;
        offset += 4;
        if nal_len == 0 || offset + nal_len > nal_data.len() {
            break;
        }
        let nal_type = nal_data[offset] & 0x1F;
        if nal_type == 5 {
            return true;
        }
        offset += nal_len;
    }
    false
}
```

**Potential issues:**
- May not handle all AVCC length prefix sizes (1, 2, or 4 bytes based on avcC box)
- First NAL in sample might be AUD (type 9), SPS (type 7), or PPS (type 8) before IDR
- Could return false for valid IDR frames if parsing fails

#### 2. Decoder State Not Reset on IDR

When an IDR frame arrives, VTDecompressionSession should:
1. Flush any pending frames
2. Reset internal reference buffer state
3. Start fresh decode sequence

**Current code does NOT explicitly flush on IDR.**

#### 3. Frame Queue May Drop Frames

The callback queue in VTDecoder has limited capacity. If frames arrive faster than they're consumed:

```rust
// In VT decode callback:
decoded_frames.lock().push_back(frame);  // Unbounded, but...
// Consumer may not keep up, causing reference frame gaps
```

#### 4. Measured FPS Shows 0.0

The FPS tracker only increments when `is_playing()` returns true, but the MoQ decoder may not properly signal playing state:

```rust
// In demo main.rs update():
if player.is_playing() {
    self.fps_tracker.update_ui_frame();
}
```

**Issue:** MoQ player state machine may not transition to "Playing" correctly.

## Decoder Data Flow

```text
hang crate
    ↓ (MoqVideoFrame with is_keyframe hint)
MoqFfmpegDecoder::decode_frame()
    ↓ (checks is_idr_frame())
    ↓ (if frame_count==0 && !IDR, returns error)
VTDecompressionSession
    ↓ (hardware decode)
CVPixelBuffer → IOSurface
    ↓ (callback queues frame)
decode_next() polls queue
    ↓ (creates MacOSGpuSurface)
VideoPlayer frame buffer
    ↓ (triple buffer)
Renderer (import_iosurface)
    ↓
Display
```

## Specific Code Locations

| Component | File | Lines | Issue |
|-----------|------|-------|-------|
| IDR detection | moq_decoder.rs | 547-570 | May misparse AVCC |
| IDR gate | moq_decoder.rs | 640-655 | Only checks frame_count==0 |
| VT callback | moq_decoder.rs | 1780-1820 | No frame drop detection |
| Queue consumer | moq_decoder.rs | 1730-1750 | May not drain fast enough |
| FPS tracking | main.rs | 330-335 | Depends on is_playing() |

## Hypotheses to Test

### H1: First frame is not actually IDR

**Test:** Add logging to show NAL types in first accepted frame:
```rust
tracing::info!("First frame NAL types: {:?}", get_all_nal_types(&data));
```

### H2: VTDecoder needs explicit flush on IDR

**Test:** Call `VTDecompressionSessionWaitForAsynchronousFrames` before decoding IDR.

### H3: hang crate's is_keyframe flag is unreliable

**Test:** Compare `moq_frame.is_keyframe` vs `is_idr_frame()` result for every frame.

### H4: Frames arriving out of order

**Test:** Log group_id and frame sequence to detect gaps or reordering.

### H5: Decoder produces frames but renderer doesn't consume

**Test:** Add counters for:
- Frames decoded (VT callback count)
- Frames queued
- Frames consumed by renderer

## Proposed Fixes

### Fix 1: Robust IDR Detection

```rust
fn is_idr_frame(nal_data: &[u8], nal_length_size: usize) -> bool {
    let mut offset = 0;
    while offset + nal_length_size <= nal_data.len() {
        let nal_len = read_nal_length(nal_data, offset, nal_length_size);
        offset += nal_length_size;
        if nal_len == 0 || offset + nal_len > nal_data.len() {
            break;
        }
        let nal_type = nal_data[offset] & 0x1F;
        if nal_type == 5 { // IDR
            return true;
        }
        offset += nal_len;
    }
    false
}
```

### Fix 2: Wait for IDR on Any Decode Error

```rust
fn decode_frame(&mut self, frame: &MoqVideoFrame) -> Result<VideoFrame, VideoError> {
    // If decoder errored, reset and wait for next IDR
    if self.decoder_errored {
        if !Self::is_idr_frame(&frame.data, self.nal_length_size) {
            return Err(VideoError::DecodeFailed("Waiting for IDR after error".into()));
        }
        self.reset_decoder();
        self.decoder_errored = false;
    }
    // ... continue with decode
}
```

### Fix 3: Flush Before IDR Decode

```rust
if is_idr {
    // Flush any pending frames before IDR
    unsafe { VTDecompressionSessionWaitForAsynchronousFrames(self.session) };
    self.callback_state.decoded_frames.lock().clear();
}
```

### Fix 4: Track Frame Statistics

Add counters to diagnose where frames are lost:
```rust
struct FrameStats {
    received: AtomicU64,      // From hang crate
    decoded: AtomicU64,       // VT callback fired
    queued: AtomicU64,        // In callback queue
    rendered: AtomicU64,      // Consumed by renderer
    dropped_no_idr: AtomicU64,
    dropped_decode_error: AtomicU64,
}
```

## Next Steps

1. Add comprehensive logging to identify exact failure point
2. Implement frame statistics tracking
3. Test with known-good local H.264 stream via MoQ
4. Compare behavior with web-based MoQ player (moq.dev/watch)
5. Consider fallback to software decode (FFmpeg) if VT fails repeatedly

## Related Issues

- `web-lumina-video-90y`: MoQ video freezes on macOS screenshot
- `web-lumina-video-gz6`: VTDecoder crash fix (kCFAllocatorNull) - CLOSED
- `web-lumina-video-8i2`: MoQ audio decoding and A/V sync
