# MoQ Late-Join Video Pixelation — Technical Problem Statement

**Date:** 2026-02-07
**Branch:** `fix/moq-worker-cleanup`
**Bead:** `web-egui-vid-duf` (P1)
**Status:** Root cause under investigation — 5 hypotheses, 2 high-priority for BBB

---

## Symptom

When joining a live MoQ stream (e.g. `moqs://cdn.moq.dev/bbb`), video displays persistent pixelation/block artifacts from the first frame onward. Audio decodes and plays correctly. The issue is **not** a freeze or stall — frames render at full rate, but the decoded picture is visually corrupted.

## Likely-Excluded Causes

| Theory | Status | Reasoning |
|--------|--------|-----------|
| Missing keyframe on late-join | Unlikely for BBB | moq-lite guarantees group-boundary joins (`keyframe: self.index == 0` at `hang/src/container/consumer.rs:194`). Confirmed by kixelated. However, internal code comment at `moq_decoder.rs:538` acknowledges hang's `is_keyframe` flag may be unreliable in some join scenarios — not fully ruled out for all streams. |
| Missing SPS/PPS | Unlikely for BBB | BBB catalog provides avcC description (41 bytes). VTDecoder initializes from catalog before first frame. For Annex B streams, `extract_h264_params` parses SPS/PPS from the keyframe itself. |
| Frozen video / metadata stall | Fixed | Fixed in commit `bfefb53` — metadata wait and PTS rejection no longer block. Runtime-verified: audio and video both play. |
| Frame backpressure drops | Fixed | Fixed in prior session — `try_send` → `send().await`. Frame delivery improved from 15% to 77%. Channel is bounded(30) with async backpressure. |

## Active Hypotheses

### H1 (HIGH): NAL length size mismatch — avcC parsed but never threaded to VT

**Location:** `moq_decoder.rs:898` (parsed), `moq_decoder.rs:669` (stored), `moq_decoder.rs:1513` (not accepted), `moq_decoder.rs:1579` (hardcoded), `moq_decoder.rs:1806` (pass-through)

The `nal_length_size` is correctly parsed from the avcC box at line 898:
```rust
let nal_length_size = ((data[4] & 0x03) + 1) as usize; // 1, 2, or 4 in practice (3 accepted by parser but not normative per ISO 14496-15)
```

And stored on the decoder struct at line 669:
```rust
self.h264_nal_length_size = nal_length_size;
```

But it is **never threaded into any downstream consumer**:

1. **`VTDecoder::new_h264`** (line 1513) has no `nal_length_size` parameter — it cannot pass it to format description creation.
2. **`create_h264_format_description`** (line 1579) hardcodes `nal_unit_header_length: i32 = 4` regardless of what the avcC specifies.
3. **`VTDecoder::decode_frame`** (line 1806) passes AVCC data through "as-is" without normalizing the NAL length prefix size.

If the publisher's avcC advertises `nal_length_size=2` (common in some encoders), VideoToolbox expects 4-byte length prefixes per the CMFormatDescription, but receives 2-byte prefixes in the sample buffer. VT then reads 2 bytes of NAL payload as part of the length field, **deterministically corrupting every NAL boundary in the stream**. VT is resilient enough to partially decode corrupted boundaries, producing visual artifacts rather than hard failures.

Similarly, `find_nal_types` at line 726 uses the stored `h264_nal_length_size` for NAL parsing, but this value could disagree with what VT was told (always 4), causing diagnostic NAL type analysis to give correct results while VT itself sees garbage boundaries.

**For BBB:** This is a direct corruption path if BBB's avcC specifies `nal_length_size != 4`. Needs runtime confirmation (see Open Questions).

**Fix direction:** Thread `nal_length_size` through `VTDecoder::new_h264` → `create_h264_format_description`. If incoming AVCC data uses a different length size than 4, either normalize the data to 4-byte prefixes before submission, or pass the actual size to `CMVideoFormatDescriptionCreateFromH264ParameterSets`.

### H2 (HIGH): `is_avcc_format` heuristic misclassifies AVCC data as Annex B

**Location:** `moq_decoder.rs:2038-2067`

The format detection checks for Annex B start codes **first**, before validating AVCC length fields:

```rust
if data[0] == 0 && data[1] == 0 {
    if data[2] == 1 { return false; }           // 3-byte start code
    if data[2] == 0 && data[3] == 1 { return false; } // 4-byte start code
}
```

Valid AVCC data gets misidentified as Annex B when the first NAL's 4-byte big-endian length happens to match a start code pattern:
- Length `0x00 0x00 0x00 0x01` (1 byte NAL) → matches 4-byte start code
- Length `0x00 0x00 0x01 0xXX` (256–511 byte NAL) → matches 3-byte start code pattern

The false-positive window covers two specific length ranges: exactly 1 byte (`0x00000001` matches the 4-byte start code check) and 256–511 bytes (`0x000001XX` matches the 3-byte start code check at `data[2] == 1`). Lengths 2–255 do **not** trigger the short-circuit because `data[2]` would be `0x00` and `data[3]` would be `0x02`–`0xFF`, matching neither Annex B pattern. While the window is narrow, it is not unrealistic: small NALs like SEI or AUD often fall in the 256–511 byte range, and if the first NAL in an access unit is one of these, the heuristic misclassifies the entire buffer.

Once misclassified, the data is run through `annex_b_to_avcc`, which searches for `00 00 01` patterns in what is actually length-prefixed data, producing corrupted output with wrong NAL boundaries.

**For BBB:** Since BBB provides avcC descriptions, frames should arrive in AVCC format. If any frame's first NAL length falls in the false-positive range, that frame gets corrupted. Needs runtime confirmation (see Open Questions).

**Fix direction:** When catalog provides a `description` field (avcC), thread the known container format through to `VTDecoder::decode_frame` so it skips the heuristic entirely. Only fall back to heuristic detection when `description: None`.

### H3 (MEDIUM): `annex_b_to_avcc` passes SPS/PPS NALs inside frame samples

**Location:** `moq_decoder.rs:2069-2128`

When data arrives in Annex B format, the first keyframe contains `[start_code][SPS][start_code][PPS][start_code][IDR]`. The `annex_b_to_avcc` function converts **every** NAL unit — including SPS (type 7), PPS (type 8), SEI (type 6), and AUD (type 9) — into length-prefixed AVCC format and includes them all in the CMSampleBuffer submitted to VTDecompressionSession.

VideoToolbox's CMFormatDescription was already created from those same SPS/PPS bytes. Whether embedding parameter sets inside the sample buffer causes issues is **implementation-specific to VT** — many decoders handle in-band SPS/PPS correctly, and some streams rely on it for mid-stream parameter set refresh.

**Applicability:** This **only** affects the `description: None` (Annex B) code path. The BBB stream provides avcC descriptions, so this path is likely not exercised for BBB. This is relevant for zap.stream and other Annex B publishers.

**Fix direction:** Validate with logs whether the Annex B path is active for the failing stream before changing. If confirmed active: consider stripping non-VCL NALs from IDR samples only when they duplicate the format description, rather than blanket-stripping all non-VCL types (which could break streams needing in-band parameter refresh).

### H4 (MEDIUM, conditional): No VT decoder flush on `OrderedConsumer` group skip

**Location:** `hang/src/container/consumer.rs:104-117`, `moq_decoder.rs:2193-2210`

When network jitter causes a group to exceed the `max_latency` budget (default 500ms), `OrderedConsumer` skips to the next group. The partially-decoded frames from the skipped group leave stale reference data in VT's Decoded Picture Buffer (DPB).

`prepare_for_idr_resync()` (which flushes the VT queue) is only called on **decode errors** (`moq_decoder.rs:815-818`), not on group skips. The new group's keyframe should theoretically reset the DPB, but if VT's internal state is inconsistent from the partial group, artifacts persist until enough keyframes flush all stale references.

**Applicability:** Only relevant if group skips are actually occurring during the corruption window. If pixelation appears from the very first frame (before any skip could happen), this is not the cause. Needs runtime confirmation (see Open Questions).

**Fix direction:** Propagate group-skip or discontinuity events from `OrderedConsumer` to the decode thread for proactive VT flush.

### H5 (LOW): `find_nal_types` uses wrong length size for format detection

**Location:** `moq_decoder.rs:721-728`

`find_nal_types` is called with `self.h264_nal_length_size` for NAL parsing, but `data_is_annex_b` is called independently. If the stored `nal_length_size` doesn't match the actual data format (e.g., after a format change mid-stream), the NAL type analysis produces incorrect results, which could cause keyframe misidentification and frame drops. This is downstream of H1/H2 — fixing those would resolve this as well.

---

## Open Questions (need runtime data)

1. **Does BBB's avcC advertise `nal_length_size=4`?** If yes, H1 is not active for BBB (though still a bug). If no, H1 is the primary cause. Check: `RUST_LOG=debug` output for the `"Parsed avcC: ... NAL length size N bytes"` log at `moq_decoder.rs:972`.

2. **Do first corrupted frames show `is_avcc_format=false` despite catalog description being present?** This confirms H2. Check: `RUST_LOG=debug` for `"VTDecoder: detected format: Annex B"` at `moq_decoder.rs:1801-1803` alongside `"initialized VTDecoder from catalog avcC"` at `moq_decoder.rs:670`.

3. **Are there `OrderedConsumer` group skips during the corruption window?** Check: `"skipping slow group"` log from `hang/src/container/consumer.rs:106`. If no skips occur before first visible corruption, H4 is not contributing.

---

## Stream Environment

| Property | Value |
|----------|-------|
| Relay | cdn.moq.dev (WebSocket + WebTransport) |
| Stream | `/bbb` (Big Buck Bunny) |
| Video codec | H.264, 720p |
| Container | avcC description in catalog (41 bytes) |
| Audio codec | AAC-LC, 44100 Hz stereo |
| Transport | QUIC via quinn (moq-lite protocol) |
| Decoder | VTDecompressionSession (macOS hardware) |
| Group delivery | moq-lite guarantees group-boundary joins |

## Recommended Investigation Order

1. **Collect runtime diagnostics** with `RUST_LOG=debug` for the first 5 frames to answer all three open questions above. This disambiguates H1 vs H2 vs H4 without code changes.

2. **Fix H1** (if `nal_length_size != 4`): Thread `nal_length_size` through VTDecoder initialization and either normalize AVCC data to 4-byte lengths or pass actual size to VT format description.

3. **Fix H2** (if misclassification confirmed): Thread known container format from catalog into `VTDecoder::decode_frame` to bypass heuristic. Even if not currently triggering, this eliminates a latent correctness bug.

4. **Fix H3** (if Annex B path is active for failing stream): Investigate VT behavior with in-band SPS/PPS before blanket-stripping.

5. **Fix H4** (if group skips measured): Add discontinuity signaling from transport to decoder.

## Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| avcC nal_length_size parsing | `moq_decoder.rs` | 897-903 |
| nal_length_size stored | `moq_decoder.rs` | 669 |
| VTDecoder::new_h264 (no nal_length_size param) | `moq_decoder.rs` | 1513-1532 |
| CMFormatDescription (hardcoded NAL len=4) | `moq_decoder.rs` | 1578-1579 |
| VTDecoder.decode_frame (AVCC pass-through) | `moq_decoder.rs` | 1806-1809 |
| `is_avcc_format` heuristic | `moq_decoder.rs` | 2038-2067 |
| `annex_b_to_avcc` conversion | `moq_decoder.rs` | 2069-2128 |
| VTDecoder init (catalog path) | `moq_decoder.rs` | 655-670 |
| VTDecoder init (keyframe path) | `moq_decoder.rs` | 678-704 |
| hang keyframe caveat comment | `moq_decoder.rs` | 538-539 |
| NAL type analysis | `moq_decoder.rs` | 721-748 |
| Frame submission to VT | `moq_decoder.rs` | 778-784 |
| Worker frame assembly | `moq/worker.rs` | 118-154, 677-685 |
| hang OrderedConsumer (group skip) | `hang/src/container/consumer.rs` | 54-121 |
| hang keyframe = index==0 | `hang/src/container/consumer.rs` | 194 |
| IDR resync (only on decode error) | `moq_decoder.rs` | 2193-2210 |
