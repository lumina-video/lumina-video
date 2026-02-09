# web-egui-vid-duf: MoQ Video Debug History

> Exported from bead `web-egui-vid-duf` on 2026-02-09 (56 comments)
> Bug: "MoQ: video broken (pixelated/frozen/artifacts) — audio works after 4s startup"
> Status: in_progress | Priority: P1 | Type: bug
> Created: 2026-02-06 07:46 | Updated: 2026-02-09 14:00

## Description

MoQ video pixelation on BBB stream + choppy audio.

### Current Status (2026-02-08 Session 5)

#### VT -12909 ROOT CAUSE: is_avcc_format() heuristic bug
- `is_avcc_format()` misclassifies AVCC frames with 256-511 byte NAL lengths
- Length prefix [0,0,1,X] looks like Annex B 3-byte start code -> Annex B conversion on AVCC data -> garbage -> -12909
- FIX: `is_avcc` field on VTDecoder set from catalog context, bypasses heuristic
- Expected: eliminates ALL decode errors + ~59 frame drops per error

#### Audio choppy ROOT CAUSE: buffer capacity vs group-boundary burst delivery
- `audio_buffer_capacity=60` (~1.3s) can't absorb 2s group bursts (~94 audio frames)
- 324 evictions in 35s = ~20% audio frame loss
- FIX: capacity increased to 180 (~3.8s)

#### Previously fixed (committed):
- PTS tolerance: 2000ms->2500ms for 2s group boundaries (674d0c70)
- Track re-subscribe on end: 5 bounded retries + IDR gate reset (674d0c70)
- Startup IDR gate: skips non-IDR startup frames, H.264-only (83c0983f)
- Audio pre-buffer: 8 frames gated on video_started (83c0983f)
- VT session recreation on error (8324a640)
- 3s metadata stall + PTS rejection fix (bfefb532)

#### Needs runtime test
- VT format fix + audio buffer increase (uncommitted, compiles clean)

### Notes
Next validation plan (determinism check): rerun the same BBB scenario with LUMINA_MOQ_ERROR_FORENSICS=1 and compare failing frame hash64 values against prior run. Prior failing triplet: d8b88f02b47590e4, af827e33672d222a, 970741a3001f2163 (with preceding IDR hash ad78eefdd683534e). Decision rule: if same hashes fail again at similar relative position, classify as deterministic content-triggered (specific AU/bitstream incompatibility). If hashes differ run-to-run, classify as nondeterministic runtime/lifecycle/timing issue and prioritize VT/session scheduling investigation.

### Dependencies
- Depends on: web-egui-vid-9feo (Remove dead get_group(seq-1) IDR fetch code from moq_decoder.rs) [P2]
- Blocks: web-egui-vid-v7r (zap.stream MoQ integration blocked) [P1 - open]

---

## Comments

### Comment 1 — 2026-02-06 15:18
Added MoqStatsSnapshot/MoqStatsHandle/MoqFrameStatsSnapshot types to moq_decoder.rs. VideoPlayer now exposes moq_stats() for UI access. Demo UI shows full MoQ pipeline state: decoder state, SPS/PPS status, frame counts (received/submitted/decoded/rendered), drops, errors. libmoq v0.2.6 analysis found get_group(sequence) could fetch previous GOP IDR on late-join — most impactful fix for pixelation.

### Comment 2 — 2026-02-06 15:27
Testing status: egui-vid demo instrumentation working (MoQ Pipeline panel renders state/errors/frame stats). cdn.moq.dev/anon has no active broadcasts. /demo namespace requires JWT we can't forge. zap.stream MoQ needs PR #69 merged. Nostr-discovered streams (6 live) connect but zap.stream won't produce decodable video until SPS/PPS fix lands. streamstr.net:1443 (ArmA Reforger) hangs on connect. Next: publish BBB to /anon ourselves to test pipeline end-to-end.

### Comment 3 — 2026-02-06 15:35
Implemented late-join IDR recovery via get_group(seq-1). When first received group has sequence > 0, fetches previous group from relay's 30s cache, reads all frames (including IDR) via hang::GroupConsumer::read(), sends them to decoder before continuing with live group. 2s timeout on fetch. Compiles clean.

### Comment 4 — 2026-02-06 15:44
TESTING RESULTS (2026-02-06): Published BBB to cdn.moq.dev/anon via hang CLI (WebSocket fallback). Demo auto-discovered and played the stream. Late-join IDR fetch TIMED OUT (group 63 not cached after 2s on WebSocket). Massive backpressure: 284 received, 215 dropped (76%), only 43 submitted/decoded/rendered. Root cause: try_send on bounded(30) channel drops frames during 4-second audio startup stall. Decoder itself works fine (43 submit=43 decode=43 render, 0 errors). Fix needed: either use blocking send with backpressure, or drain multiple frames per decode_next() call. SPS/PPS present (41 bytes from catalog), Container=Legacy, jitter=41ms.

### Comment 5 — 2026-02-06 16:11
2026-02-06: Backpressure fix landed — try_send->send().await, frame delivery 15%->77%. IDR late-join timeout reduced 2s->500ms. Late-join IDR fetch still times out in practice (WebSocket may not support get_group or relay didn't cache). Pixelation on late-join remains when first received group lacks IDR. Next: test with QUIC (not WebSocket) where get_group may work, or implement in-stream SPS/PPS extraction fallback.

### Comment 6 — 2026-02-06 16:39
AGENTS.md audit: moq_decoder.rs has 3 copies of run_moq_worker (Desktop 574L, Android 175L, GStreamer 211L). ~45% shared logic. Desktop has backpressure fix, IDR recovery, container validation, stats — Android/GStreamer lack all of these. Created web-egui-vid-tfa for future extraction into moq::client module. Decision: fix later once second platform is tested, but propagate critical point fixes (backpressure, IDR) now.

### Comment 7 — 2026-02-06 18:53
MoQ developer feedback (kixelated): description:None means Annex B format — SPS/PPS is inline before each keyframe, not missing. Our decoder already handles this correctly (extract_h264_params + annex_b_to_avcc). The real late-join fix is get_group(sequence) to fetch previous GOP IDR frame. Our zap-stream-core PR #69 (adding SPS/PPS to catalog) is valid but not required — properly-written decoders extract from bitstream. Ref: https://doc.moq.dev/concept/layer/hang.html and WebCodecs AVC registry.

### Comment 8 — 2026-02-06 22:35
NEW CONTEXT (kixelated 2026-02-06): cdn.moq.dev does NOT use MoQ FETCH messages for old groups. Instead, HTTP endpoint: `https://cdn.moq.dev/fetch/<namespace>/<track>?group=N&jwt=...` Default (no ?group) = current live group. ?group=N = specific old group. Catalog also via HTTP. BBB demo catalog confirms: video0 (H.264 720p avcC description), audio1 (AAC-LC 44100Hz stereo, no description). Late-join IDR fix should use HTTP group fetch — simpler than protocol FETCH, confirmed working by developer.

### Comment 9 — 2026-02-06 22:43
kixelated confirms cdn.moq.dev does NOT support MoQ FETCH messages. Our .get_group(seq) call at moq_decoder.rs:984 likely silently fails. Late-join IDR must use HTTP group fetch: `GET https://cdn.moq.dev/fetch/<namespace>/<track>?group=N&jwt=...` Default (no ?group) = current live group. This is the actual fix path for pixelation on cdn.moq.dev.

### Comment 10 — 2026-02-06 23:32
cdn.moq.dev self-hosting docs: https://github.com/moq-dev/moq/tree/main/cdn — useful for local testing of HTTP group fetch without relying on production relay.

### Comment 11 — 2026-02-07 15:41
kixelated PR review (moq-dev/moq#931, 2026-02-07): moq-lite GUARANTEES group-boundary joins — subscriptions always start at group boundaries, each group starts with keyframe. "There is no way to receive a frame without first receiving its dependencies." This is different from moq-transport which could deliver mid-group. Late-join pixelation root cause must be something else: decoder init, SPS/PPS timing, backpressure dropping first keyframe, container parsing, or frame buffer corruption. HTTP group fetch only needed for larger jitter buffer, not keyframe recovery. He is having Claude iterate on the upstream PR.

### Comment 12 — 2026-02-07 19:07
Session 2026-02-07 findings (from fix3 log analysis before context loss):

QUEUE FULL is the real bottleneck:
- 3,546 QUEUE FULL events on 5-slot frame queue — frames produced faster than consumed
- First frame IS IDR (NAL type 5) — keyframe detection was never the issue for BBB
- 3-second gap after first frame — metadata timeout blocking the pipeline
- 167 IDR drops + 4 decode errors even after VT sync sample marking fix
- Choppy audio after ~20 seconds of playback
- Measured FPS at 24 while video source is 30fps — ~20% frame loss

VT sync sample fix (kCMSampleAttachmentKey_DependsOnOthers=false, kCMSampleAttachmentKey_NotSync=false on keyframes) did NOT resolve the issue alone.

FPS color fix landed: Color32::YELLOW -> orange rgb(255,165,0) for readability on light backgrounds.

Root cause narrowing: the 5-slot bounded queue is the chokepoint. Decoder consumes slower than MoQ delivers, causing cascading drops. Need to either increase queue size, implement adaptive dropping (keep keyframes, drop P/B frames), or match decode throughput to delivery rate.

### Comment 13 — 2026-02-07 19:14
ROOT CAUSE ANALYSIS (2026-02-07, fix3 log — 45sec session, 39K lines):

#### 5 Issues Found

1. **3-SECOND METADATA STALL (HIGH)** — After first frame decodes, pipeline stalls waiting for video metadata with 3s timeout. 29 buffered group frames sit idle until timeout fires. This is a code bug — metadata handshake triggers timeout path unnecessarily on every session start.

2. **FRAME REJECTION — 2s PTS GAP (HIGH)** — Group-boundary join delivers burst of frames spanning 2+ seconds. Presentation clock starts at frame 0 PTS, advances real-time. All buffered frames have PTS 2s in future -> rejected as too early. 1,130 'get_next_frame: rejecting, gap=2001ms' events.

3. **OUTPUT QUEUE FULL (MEDIUM)** — 5,178 events. 5-slot output queue (separate from 30-slot async channel) with 5ms sleep retry. Decoder outruns renderer -> constant polling waste (~150/sec).

4. **A/V DESYNC at startup (MEDIUM)** — Audio base_pts 1.7s ahead of video first PTS. No sync correction applied.

5. **VT info_flags=0x4 on every frame (LOW)** — VT thinks all 839 frames are late. Systematic timing mismatch with VT internal clock. Harmless but indicates wrong presentation timestamps.

Key Facts:
- BBB source is genuinely 24fps (PTS interval=41,666us=1/24s) — but demo app displayed 30fps (BUG in FPS measurement)
- Main frame channel: async_channel::bounded(30). Output queue: 5 slots (separate)
- Only 1 actual decode error (OSStatus -12909 bad data, recovered in 13ms)
- Catalog reports 0.0fps — publisher doesn't set fps field

Fix Plan:
1. Fix 3-second metadata stall (find the timeout gate, make metadata propagate immediately)
2. Fix 2-second PTS rejection (reset presentation clock on late-join group burst)
3. Investigate FPS display showing 30 when content is 24

### Comment 14 — 2026-02-07 19:23
FIXES IMPLEMENTED (2026-02-07, fix/moq-worker-cleanup branch):

**Fix 1: 3-second metadata stall (frame_queue.rs:495-605)** — Root cause: metadata wait loop requires has_duration && has_dimensions. MoQ duration() always None, dimensions() returns (0,0) because cached_metadata only syncs in decode_next() which isn't called during the wait loop. Condition NEVER true -> always stalls 3 full seconds. Fix: Capture preview frame dimensions. If live stream (duration()==None) and preview succeeded, skip metadata wait entirely. Uses preview_dims directly.

**Fix 2: PTS rejection during audio startup (frame_queue.rs:1317-1321)** — Root cause: ahead_tolerance=100ms during 500ms audio buffer startup. At 24fps (42ms per frame), only ~2 frames accepted before 500ms rejection stall. Fix: Increased tolerance from 100ms to 500ms to match audio startup budget. After audio starts, existing 2000ms tolerance takes over.

Expected: 0s metadata stall + frames flow within 500ms = near-instant playback (was ~6.5s)

### Comment 15 — 2026-02-07 19:30
ROOT CAUSE FOUND + FIX APPLIED (2026-02-07, branch fix/moq-worker-cleanup):

TWO ROOT CAUSES in frame_queue.rs causing frozen video + choppy audio on MoQ live streams:

**Fix 1: 3-second metadata stall (frame_queue.rs ~line 547-610)** — Root cause: Metadata wait loop requires has_duration && has_dimensions to proceed. MoqDecoder.duration() returns None for live streams (no finite duration). MoqDecoder.dimensions() returns (0,0) because cached_metadata only syncs inside decode_next(), NOT during the metadata wait loop. Result: condition is NEVER true for MoQ -> always stalls for full 3-second timeout. Fix: Capture dimensions from preview frame. If stream is live (duration()==None) and preview succeeded, skip metadata wait entirely and use preview dimensions.

**Fix 2: PTS rejection during audio startup (frame_queue.rs ~line 1337-1340)** — Root cause: During 500ms audio buffer startup, ahead_tolerance was only 100ms. At 24fps (42ms intervals), only ~2 frames accepted before 500ms rejection stall. During the 3s metadata stall, MoQ worker fills 30-slot channel -> frames spanning 3+ seconds pile up -> mass PTS rejection. Fix: Increased ahead_tolerance from 100ms to 500ms to match audio startup budget. After audio starts, existing 2000ms tolerance takes over.

**Expected impact:** Before: 3s metadata stall + 500ms PTS rejection + 3s STUCK_TIMEOUT = ~6.5s before smooth playback. After: 0s metadata stall + frames flow within 500ms tolerance = near-instant playback.

Status: Both fixes compile clean. Needs runtime testing against cdn.moq.dev BBB stream.

### Comment 16 — 2026-02-07 19:36
PR #15 submitted: fix/moq-worker-cleanup -> main. Fixes initial 6.5s freeze (metadata stall + PTS rejection). Audio/video still choppy/intermittent AFTER startup — separate root cause, not addressed by this PR. Need to investigate: frame delivery timing, decode pipeline backpressure, audio buffer underruns.

### Comment 17 — 2026-02-07 19:41
RUNTIME TEST (2026-02-07 post-PR#15): Start time ~4s (down from ~6.5s but NOT instant — metadata fix only partially effective). Audio smooth after startup. Video BROKEN: pixelated artifacts and/or frozen. Non-usable. The metadata stall fix helped audio but video decode pipeline has a separate fundamental issue. Need to investigate: VTDecoder init sequence, frame reordering, decode error recovery, possible corrupt NAL delivery.

### Comment 18 — 2026-02-07 20:52
RUNTIME TEST #2 (2026-02-07 post-rebase): Audio works fine — smooth playback. Video still pixelated. Metadata stall + PTS fixes resolved audio startup but video pixelation is a separate root cause. NOT missing keyframe (moq-lite guarantees it), NOT missing SPS/PPS. Need to investigate: VTDecoder init sequence, frame data integrity (corrupt NALs?), AVCC/AnnexB format handling, frame reordering (B-frames?), or rendering pipeline issue.

### Comment 19 — 2026-02-07 21:10
Analysis doc written (MOQ_PIXELATION_ANALYSIS.md). 5 hypotheses ranked after external review: H1 HIGH (nal_length_size parsed at :898, stored at :669, but hardcoded 4 at :1579 — deterministic corruption if stream uses non-4-byte lengths), H2 HIGH (is_avcc_format heuristic false-positives for NAL lengths 1 or 256-511), H3 MEDIUM (annex_b_to_avcc includes SPS/PPS — Annex B path only), H4 MEDIUM conditional (no VT flush on group skip), H5 LOW (downstream of H1/H2). Three open questions need RUST_LOG=debug run: (1) BBB nal_length_size value, (2) is_avcc misclassification on first frames, (3) group skips during corruption window.

### Comment 20 — 2026-02-07 21:21
ROOT CAUSE FOUND (2026-02-07 session 3): VT Session Corruption After -12909. Log analysis from fix4 (20-second session, moqs://cdn.moq.dev/bbb): First 89 frames decode perfectly (3.7s at 24fps), then OSStatus -12909 on frame ~90. IDR resync drops ALL P-frames, waits for keyframe. Next keyframe ALSO fails -12909 — cascading failure. 7 errors at ~2.5s intervals, only 98/360 frames ever decoded. Root cause: prepare_for_idr_resync() only clears queue/error flag, does NOT recreate VT session — corrupted internal state persists. FIX APPLIED: self.vt_decoder = None on error, forcing fresh VTDecompressionSession creation from catalog SPS/PPS on next keyframe. All 5 hypotheses from PIXELATION_ANALYSIS.md ruled out for BBB (nal_length_size=4, AVCC format correct, no Annex B path). Compiles clean, needs runtime test.

### Comment 21 — 2026-02-07 21:33
Session 3 continued (2026-02-07): fix6 runtime test shows 0 decode errors, 0 IDR drops, 238/240 frames decoded — VT session recreation fix works! But video STILL pixelated. This rules out ALL decode-level causes. New investigation needed: possibly IOSurface rendering race, B-frame reordering (decode_time_stamp set to CMTime::invalid()), or relay serving corrupted data. BBB stream currently offline on cdn.moq.dev ('bbb not found after 10s'). Added build ID to demo UI (env LUMINA_BUILD_ID) for screenshot verification.

### Comment 22 — 2026-02-07 21:36
Session 3 fix8 runtime test (build fix6-1533): VT session recreation CONFIRMED WORKING — 'destroyed session and entering IDR resync mode' message visible, new VTDecoder created 15ms later from catalog avcC. Results: 1 error (down from 7), 275/330 decoded (83%, up from 27%). Error still occurs at ~11.4s (frame ~263) — consistent with previous sessions (~3.7-4s after join in fix4). Pattern: 263 frames decode perfectly, then one -12909, session destroyed+recreated in 15ms, 45 frames dropped during keyframe wait, then clean decode resumes. User reports pixelation ~50% of time and choppiness. Two issues remain: (1) WHY does -12909 occur at frame ~263? Possible: group boundary data issue, B-frame at GOP edge, or transport corruption. (2) Initial pixelation even with 0 errors in first 179 frames — could be from the first group's compressed quality or rendering pipeline issue.

### Comment 23 — 2026-02-08 15:35
Session 4 (2026-02-08): PR #931 review + frame pipeline investigation.

**PR #931 (merged)**: kixelated triple-confirms mid-group join impossible. Edited our docs: 'it's not possible to join mid-group' (not just 'corrupted output'). Deleted our 'Missing Keyframe on Late Join' section entirely. Also corrected: audio groups DON'T need to align with video groups. Added moqt:// scheme. New subscribe.rs canonical example.

**Frame pipeline deep-dive findings:**
1. assemble_payload() (worker.rs:673-685) — CLEAN. BytesMut properly reserved/extended. No truncation/corruption risk.
2. VT pixel format — FALSE ALARM on NV12 vs BGRA. VT explicitly configured with kCVPixelFormatType_32BGRA (line 1690), IOSurface read as PixelFormat::Bgra (line 2165). VT handles YUV->BGRA conversion internally.
3. IOSurface lifetime — SAFE. Arc<CVPixelBuffer> keeps IOSurface alive. VTDecompressionSessionWaitForAsynchronousFrames() ensures callback completes. No race.
4. decode_time_stamp = CMTime::invalid() (line 1870) — REAL CONCERN for B-frames. decode_flags=0 (line 1958), no kVTDecodeFrame_EnableTemporalProcessing.

**New primary hypothesis: B-frame decode order mismatch**
- BBB is likely Main/High profile H.264 (B-frames present)
- With DTS=invalid and no temporal processing flag, VT outputs in decode order not presentation order
- If hang delivers frames in presentation order (not decode order), B-frames referencing future P-frames would have missing DPB dependencies
- VT may produce artifacts WITHOUT returning errors (partial decode with stale references)
- This exactly matches the symptom: pixelation with 0 decode errors

**Need to verify:** (1) Does BBB use B-frames? Check SPS profile byte. (2) What order does hang deliver frames within a group — decode or presentation?

### Comment 24 — 2026-02-08 15:44
Session 4 addendum — deep analysis findings:

**B-FRAME RULED OUT**: Parsed NAL slice headers from fix8 log. All non-IDR frames are P-slices (slice_type=5, NAL header 0x41, slice header 0x9A). BBB on cdn.moq.dev is High profile (100) but uses NO B-frames. DTS=invalid and missing temporal processing flag are NOT causing pixelation.

**Frame pipeline verified clean (fix8):**
- 263 frames decoded, 0 errors, 0 drops, 0 IDR drops, 0 backpressure drops
- PTS perfectly monotonic at 24fps cadence (41666-41667us intervals)
- AVCC format, nal_length_size=4, all correct
- IOSurface->Metal->wgpu path uses standard newTextureWithDescriptor:iosurface:plane: API
- CVPixelBuffer retained via Arc, no race condition

**jitter=0s from catalog (latent bug):**
- BBB catalog reports jitter=Some(0s), our code uses it as max_latency=0 for OrderedConsumer
- With latency=0, any pending group immediately triggers skip of current group
- NOT actively causing skips for BBB (pre-buffered content arrives in bursts, 0 skip events in log)
- But IS a latent bug for truly live streams — should clamp to minimum 500ms

**First vs second group quality disparity:**
- Group 1 IDR: 6,639 bytes (tiny — loop/scene boundary)
- Group 2 IDR: 92,955 bytes (normal quality)
- 14x size difference. First group P-frames are abnormally large (11-25KB) vs second group (2-7KB)

**REMAINING PUZZLE: Pixelation with 0 errors** — User screenshot showed pixelation at 179 decoded frames — well within the clean 263-frame window. All decode metrics perfect. Possible explanations: (1) BBB stream quality — some groups have low-quality IDRs producing compression artifacts, (2) Display/rendering issue in egui texture display (unlikely, standard Metal path), (3) Frame queue timing causing wrong frame display (unlikely, PTS is monotonic).

### Comment 25 — 2026-02-08 15:50
Session 4 — critical review of root cause assessment (2026-02-08):

**Assessment: 'stream quality is highest confidence root cause' is PREMATURE.**

Key pushback points:
1. A 6.6KB IDR produces blurriness/softness, NOT structured pixelation. These are visually distinct artifacts. We're assuming what user means by 'pixelation' without distinguishing compression blur vs decode corruption vs rendering defect.
2. The consistent -12909 at frame ~263 is UNDERWEIGHTED. This is evidence of systematic data issues at group boundaries. VT doesn't randomly corrupt. If data issues are severe enough to crash VT at frame 263, subtler issues earlier could produce visual artifacts via error concealment (status=0 but degraded output).
3. We've only verified first 20 bytes of first 15 frames (~0.6s). Pixelation screenshot was at frame 179 (~7.5s). ZERO visibility into byte content of frames 15-262.
4. IOSurface->screen rendering path NEVER validated in isolation. Metal texture format, row stride, texture cache staleness could produce structured artifacts.

**Revised root cause ranking:**
1. Unknown — needs A/B test (can't rank without data)
2. Subtle data integrity issue in deep frames (medium — -12909 is evidence of systematic problems)
3. Stream quality (medium — plausible for blur, less for structured pixelation)
4. Rendering pipeline texture format/stride (low-medium — never validated)
5. jitter=0s (low for BBB, fix anyway)

**RECOMMENDED DECISIVE TEST: Save raw AVCC frame data to /tmp/bbb_frames.h264 during session, play with ffplay. Tests data independent of VT + rendering. If ffplay shows same artifacts -> stream. If clean -> our pipeline. Faster and more controlled than web player A/B.**

### Comment 26 — 2026-02-08 15:57
Session 4 — DECISIVE TEST PLAN: ffplay A/B via frame dump hook

**Goal**: Determine if pixelation is in the stream data or our VT/render pipeline by dumping raw frame data and playing with ffplay independently.

**Implementation**: FrameDumpHook in worker.rs, env-var gated (LUMINA_MOQ_DUMP_PREFIX). Writes three files:
- .avcc — raw AVCC samples concatenated (for byte inspection)
- .h264 — playable Annex B stream with SPS/PPS injected before every IDR (for ffplay)
- .csv — per-frame index (idx, pts_us, is_keyframe, size, has_desc)

**Critical fixes vs original external proposal:**
1. SPS/PPS must be injected before every IDR in .h264 — original missed this, ffplay would fail
2. Format known from catalog (has_avcc_desc flag) — no flawed heuristic guessing
3. Timestamp as_micros() returns u128, needs cast to u64
4. max_frames default u64::MAX not 0

**Integration points in worker.rs:**
- Init after catalog fetch (~line 103): `FrameDumpHook::from_env(description, label)`
- Hook after assemble_payload (~line 133): `dump.write_frame(pts, keyframe, &data, has_desc)`

**Run command:**
```
LUMINA_MOQ_DUMP_PREFIX=/tmp/bbb_frames LUMINA_MOQ_DUMP_MAX_FRAMES=400 cargo run -p lumina-video-demo
```

**Verification:**
```
ffplay -framerate 24 /tmp/bbb_frames.h264
```
- If ffplay shows same artifacts -> stream/data issue (not our bug)
- If ffplay is clean -> VT or rendering pipeline issue (our bug)

### Comment 27 — 2026-02-08 16:00
Review of external feedback on frame dump hook plan (comment #26):

**P1 parse-failure fallback: AGREE** — defaulting to nal_length_size=4 on avcC parse failure would silently produce corrupt .h264 output. Revised: disable hook init entirely if parse_avcc_box fails.

**P1 SPS/PPS keyed on is_keyframe vs IDR: DISAGREE** — reviewer suggests detecting NAL type 5 instead of using is_keyframe. This is backwards for our use case: hang's is_keyframe = (index == 0) = first frame in group = always a random access point (moq-lite guarantee). ffplay needs SPS/PPS before ANY random access point, not just IDR (NAL 5).

**P2 has_avcc_desc track mismatch: DISAGREE** — reviewer claims renditions.iter().next() might not match subscribed track. Code shows they come from the SAME .next() call. Cannot mismatch.

**P3 conversion-status CSV columns: AGREE** — adding annexb_detected and conversion_ok columns is low cost, high diagnostic value.

### Comment 28 — 2026-02-08 16:05
Review of external feedback round 2 (5 findings on frame dump hook plan v3):

All 5 findings CONFIRMED valid:

**[P0] Private helpers: AGREE** — parse_avcc_box, find_nal_types, data_is_annex_b are private inherent methods on MoqDecoder. worker.rs can't call them. FIX: make pub(crate).

**[P0] Init compile blockers: AGREE** — 3 issues: type mismatch, double drop, missing mut.

**[P1] nal_length_size=3: AGREE** — parse_avcc_box allows 1..=4 but conversion only handles 1/2/4. Add 3-byte support.

**[P1] Heuristic format branching: AGREE — strongest point.** Hook only inits when catalog provides avcC, so data MUST be AVCC. Using heuristic reintroduces the exact H2 bug we're testing for. FIX: always use AVCC->AnnexB path.

**[P2] ? kills streaming: AGREE** — dump is diagnostics, should not kill playback. FIX: wrap I/O, on error set frame_dump=None.

### Comment 29 — 2026-02-08 16:10
Review of external feedback round 3 (5 findings on frame dump hook plan v4):

All 5 confirmed valid:

**[P0] cfg(macos) on parse_avcc_box: AGREE** — parse_avcc_box is #[cfg(target_os = "macos")] but it's pure byte parsing. Calling from worker.rs breaks non-macOS builds. FIX: remove cfg gate.

**[P0] Borrow-checker on disable: AGREE** — can't assign frame_dump=None inside if-let scope. FIX: sentinel pattern.

**[P1] Jitter clamp scope: AGREE** — .max(500) overrides ALL sub-500ms values. FIX: only clamp jitter=0.

**[P1] conversion_ok false positive: AGREE** — AVCC loop exits silently. FIX: offset != data.len() check.

**[P2] Missing import: AGREE** — trivial.

### Comment 30 — 2026-02-08 16:13
Plan v5 finalized — all P0/P1/P2 findings from 3 review rounds incorporated. Ready to implement.

Changes summary (2 files):
1. moq_decoder.rs: pub(crate) on find_nal_types + data_is_annex_b, remove #[cfg] + pub(crate) on parse_avcc_box
2. worker.rs: MoqDecoder import, FrameDumpHook struct, sentinel disable pattern, jitter=0 clamp

### Comment 31 — 2026-02-08 16:29
Session 5 (2026-02-08) — DECISIVE ffplay A/B TEST COMPLETE

#### Video Pixelation Root Cause: CONFIRMED — Degenerate First Group

Frame dump hook captured 400 frames to /tmp/bbb_frames.{h264,csv}. ffplay reproduces the SAME pixelation — confirms issue is in the stream data, NOT our VT/rendering pipeline.

**Smoking gun: Frame 0 is NOT a real IDR.**
- is_keyframe=true (hang position-based), but NAL type [1] (non-IDR P-slice), only 731 bytes
- Real IDRs in this stream: 39-215 KB, all NAL type [5]
- ffprobe confirms: 'Coded slice of a non-IDR picture' + 'Frame num gap 1 -1' on first frame
- Frame 0 is a tail fragment from a prior group — references an IDR we never received

**Transport issues in first group:**
- 8 dropped frames in first 31 frames (25% loss, visible as PTS gaps)
- ffprobe 'Frame num gap' warnings at positions matching CSV PTS gaps
- Frames 72-74: Annex B fragments in AVCC stream (format mixing at group boundary)
- Frame 74: phantom IDR (NAL type 5, 361 bytes, heuristic_annexb=true) — too small, wrong format

**Fix**: Detect degenerate first group (is_keyframe=true but NAL type != 5) and skip to next group's real IDR. Adds ~2s join latency, eliminates pixelation.

#### Choppy Audio Root Cause: LiveEdgeSender drops + no jitter buffer
Audio uses LiveEdgeSender (bounded channel, capacity 60) with live-edge drop policy. Combined with fair tokio::select! and same PTS gaps. No smoothing jitter buffer.

### Comment 32 — 2026-02-08 17:28
Session 5 continued (2026-02-08) — PLAN FINALIZED after 7 review rounds

#### Plan: Fix MoQ Video Pixelation + Choppy Audio

**Fix 1: Startup IDR Gate (Video Pixelation)**
- Worker-level gate: skip frames until real IDR (NAL type 5 + frame.keyframe) at group boundary
- H.264-only (VideoCodec::H264 check from catalog), bypassed for other codecs
- Bounded fallback: 5s timeout + 3 group cap, hard failsafe accepts any frame after timeout
- Parse failure on H.264 avcC -> bypass gate entirely
- video_started signal: compare_exchange(false,true) on MoqAudioShared

**Fix 2: Audio Pre-Buffer + Video-Gated Start (Choppy Audio)**
- Pre-buffer 8 decoded AAC frames (~170ms) before player.play()
- Gated on video_started (first successful video decode)
- 6s timeout (> video gate 5s)

**Fix 3: LiveEdgeSender Observability**
- SendResult enum: Sent | Dropped
- audio_live_edge_evictions counter

### Comment 33 — 2026-02-08 18:29
Plan review round 12+ complete. Key refinements:

**IDR Gate**: groups_exhausted advisory only, hard failsafe at 10s, saturating_add, CatalogResult named struct.

**Audio Pre-Buffer**: AUDIO_PRE_BUFFER_TIMEOUT derived from shared constant, disconnect during pre-buffer = strict teardown.

**video_started Signaling**: Platform-neutral at desktop/Android/GStreamer decode paths. compare_exchange(false, true) idempotent.

**Shared Constant**: `pub(crate) const MOQ_STARTUP_HARD_FAILSAFE_SECS: u64 = 10`

### Comment 34 — 2026-02-08 18:51
Runtime test results (commit 83c0983f, 2026-02-08 session 2):

**IDR gate: WORKING but not the root cause of pixelation.**
- Gate cleared at 0ms — first frame was real IDR (NAL type 5, 41515 bytes)
- skipped_startup_frames=0
- Audio pre-buffer worked: Starting -> Buffering -> Running

**Pixelation root cause: frame_queue PTS gap (2s), NOT missing IDR.**
- Frame queue rejects subsequent frames with ~2000ms gap
- Stuck timer resets every frame acceptance so never triggers
- Frames pile up -> when finally accepted, decoder is mid-stream with stale references -> pixelation

**VT session corruption: still happening.**
- OSStatus -12909 at ~frame 120 (4.2s into stream)
- 89 frames dropped during IDR resync
- 2 decode errors total, 282/373 decoded (75.6% rate)

### Comment 35 — 2026-02-08 21:59
Session 5 Findings (2026-02-08, commit 674d0c70 + uncommitted)

#### VT -12909 ROOT CAUSE FOUND: is_avcc_format() heuristic misclassification

**Bug**: `VTDecoder::is_avcc_format()` checks first 3 bytes for Annex B start codes BEFORE checking AVCC length. When an AVCC frame's first NAL has length 256-511 bytes, the 4-byte length prefix `[0x00, 0x00, 0x01, X]` is misclassified as Annex B. Code runs annex_b_to_avcc() on already-AVCC data -> garbage -> VT -12909.

**Evidence**: BBB P-frames with small motion produce NAL units in 256-511 byte range. Consistent errors at specific positions.

**Fix**: `is_avcc: bool` field on VTDecoder, set from catalog context. Eliminates ALL misclassification.

#### Audio Eviction ROOT CAUSE: buffer capacity vs burst delivery
**Bug**: capacity=60, bursts=94 frames -> 34 evictions per burst -> ~20% loss.
**Fix**: capacity 60 -> 180.

### Comment 36 — 2026-02-08 23:16
Session 6: IDR gate fix for reference frame corruption

**Root cause**: Skip-and-continue (errors <3) kept VT session alive but did NOT set waiting_for_idr_after_error. Subsequent P-frames decoded with stale DPB references -> visible pixelation until next IDR.

**Fix**: Added self.waiting_for_idr_after_error = true to isolated error path (was only in 3+ path). Session stays alive, but P-frames gated until next IDR resets DPB.

### Comment 37 — 2026-02-08 23:48
Session 7: Runtime test reveals silent VT corruption (0 errors + pixelation)

**Key finding: pixelation occurs with ZERO decode errors.**
- SS3: recv=231, decoded=225, errors=0, 24fps — PIXELATED (submitted==decoded==rendered, no drops)
- SS4: recv=358, decoded=245, errors=2, IDR_drops=110, 5fps — Clean (IDR gate working)

**This means is_avcc fix alone does NOT solve pixelation.** VT silently produces corrupted output.

**find_nal_types() has same heuristic bug as old is_avcc_format()** — data_is_annex_b() misclassifies AVCC 256-511 byte NAL prefixes.

**Frame rate default: 30fps vs 24fps actual** — worker.rs:929 unwrap_or(30.0).

### Comment 38 — 2026-02-08 23:55
Session 7: fix find_nal_types heuristic + frame rate default

**Fixes applied:**
1. find_nal_types_for_format(): New format-aware NAL parser, bypasses data_is_annex_b()
2. MoqDecoder.is_avcc field: Set from catalog context
3. Startup IDR gate (worker.rs): Uses find_nal_types_for_format with is_avcc=true
4. should_wait_for_idr(): Uses format-aware parsing
5. Frame rate default: 30fps -> 24fps

### Comment 39 — 2026-02-09 03:00
Session 8: Runtime test confirmed pattern. First group keyframe rejected by IDR gate (groups_seen=2 before clear, 45 startup skips). First accepted IDR produces PIXELATED output with 0 errors. prepare_for_idr_resync() insufficient — VT DPB retains stale state. Fix: destroy VT session on EVERY error.

### Comment 40 — 2026-02-09 04:37
Session 8: VT quiesce barrier confirmed. Added WaitForAsynchronousFrames before Invalidate in Drop, plus 50ms delay between destruction and recreation. Test: 3 errors/3 sessions (baseline 9/6). Committed 27b9eaf7.

### Comment 41 — 2026-02-09 05:18
Session 9 runtime findings (targeted forensic rerun)

Repro symptom: Video pixelates/freeze-stalls while audio continues.

Key forensic results:
1. VT callback errors still present: OSStatus -12909 at multiple log positions
2. Error-trigger frame fingerprints (different hashes from prior run — not byte-identical deterministic)
3. First trigger: keyframe=true but nal_types=[1] (degenerate group-start)
4. After errors: prolonged "waiting for IDR resync ... got NAL type 1" streak -> freeze

### Comment 42 — 2026-02-09 05:36
Latest context (2026-02-09, post-fix reruns)

User-visible behavior remains reproducible. Key evidence from /tmp/moq-fixall-r4.log:
1. VT decode errors still occur (reduced frequency)
2. IDR starvation: "2541ms, 48 dropped, last_nal_type=1"
3. Dominant freeze mechanism: frame scheduler reject loop (gap=3505-3523ms)
4. Pipeline stats: recv=510, submit=479, decoded=380, rendered=380, errors=1, drop_idr=95

**Assessment**: Primary blocker shifted from VT lifecycle to timing/scheduler policy under live group-boundary jitter.

### Comment 43 — 2026-02-09 06:00
Session update (5+ min soak)

- Steady-state video became smooth (24fps) and visually clean for extended periods
- Startup/early phase rough (low fps / intermittent freeze before stabilizing)
- Audio remained choppy even when steady-state video looked good
- scheduler logs show audio_pos=0ns while video is active — audio clock wiring still wrong

Code actions:
1. Frame scheduler hardening (adaptive live ahead-tolerance, reject-window telemetry)
2. Audio-handle wiring fix: handles_audio_internally() keys off MoQ audio handle presence

### Comment 44 — 2026-02-09 06:30
Session update (post-soak + audio-clock patch)

- Reject windows cluster around ~4.2s gaps with tolerance capped at 4200ms
- audio_available=false, samples_played=0, raw_audio_pos=0ns during playback
- One VT callback error remains

Implemented:
1. Stable MoQ audio handle lifecycle (no handle replacement races)
2. Correct availability transitions
3. Prevent premature availability on MoQ path
4. Scheduler diagnostics retained

### Comment 45 — 2026-02-09 07:01
Session update (latest BBB MoQ runs)

From /tmp/moq-fixall-r15.log:
- Triple consecutive VT callback failures: OSStatus -12909 on seq 228/229/230
- After 3rd error: destroy VT session, enter IDR wait, stream delivers long non-IDR runs -> starvation/freeze
- IDR starvation: 2581ms, 47 dropped

Applied:
1. Isolated VT errors no longer enter IDR wait mode (only >=3 consecutive)
2. Faster IDR starvation recovery: ~1.0s/24 drops (was ~2.5s/96)
3. Immediate re-subscribe on keyframe=true but non-IDR

### Comment 46 — 2026-02-09 07:15
Hypothesis update: repeated tolerance bumps masking real failure mode — clock-offset/discontinuity issue.

Proposed direction:
1. One-shot offset rebase on stable large lead (~4s)
2. Explicit state machine: Normal -> CatchUp -> Resync
3. Hysteresis (separate enter/exit thresholds)
4. Wire MoQ discontinuity/group-skip signal for immediate rebase

### Comment 47 — 2026-02-09 14:44
Latest validation (r17):
- Scheduler reject-loop inactive in this run
- Active failure: VT callback churn — 192 frames flagged RequiredFrameDropped, zero OSStatus errors
- One-shot recreation triggered after 192 frames

Applied:
1. FrameScheduler control-loop hardening (offset rebase + state machine)
2. VT required-drop storm fail-fast: consecutive streak detector, force decode error on storm threshold

### Comment 48 — 2026-02-09 16:09
2026-02-09 validation update (post-r20 through latest reruns)

Applied:
- worker.rs: async decoder-requested resubscribe, rolling-window throttle/cooldown
- moq_decoder.rs: isolated errors don't force IDR-wait, callback-error escalation, RequiredFrameDropped fail-fast removed (kept as telemetry)

Latest evidence: symptoms still reproducible. drop(no IDR) 130-345 range, errors 24-57 range, quality FAIL.

### Comment 49 — 2026-02-09 17:06
r21 validation update

Failing frame: seq=546, group~13, size=4225, nal_types=[1], hash64=b88319e683133e58. Preceding frame: valid IDR seq=545.

Fix plan:
1. On first VT callback error, immediately request re-subscribe
2. Preserve non-IDR drop gating, pair with immediate re-subscribe

### Comment 50 — 2026-02-09 17:22
r22 validation update

- No OSStatus -12909 in this run
- Failure shifted to one-shot recreation + required-drop churn
- info_flags=0x4 correlated with visible corruption

Next: disable one-shot recreation, add bounded required-drop storm recovery

### Comment 51 — 2026-02-09 17:46
r24 update + applied fix

Dominant failure: repeated RequiredFrameDropped storms (streak=36). Previously forced hard session destroy -> reset-loop thrash (attempt #5..#11).

Applied: RequiredFrameDropped storm is now soft recovery — skip frame, keep session alive, no forced IDR wait, no forced re-subscribe. Storm streak reset at trigger.

### Comment 52 — 2026-02-09 17:55
r25 update

- Startup gate improved (broken-keyframe storm triggers early re-subscribe)
- RequiredFrameDropped storms still occur, soft-skipped
- Intermittent -12909 still appears

Applied: removed re-subscribe from storm path (local soft drop only).

### Comment 53 — 2026-02-09 19:28
r16 evidence: RequiredFrameDropped storm cycles dominant (streak=36 x8 in one window). 192 flagged frames, 0 OSStatus errors. Current skip-only behavior loops in prolonged corruption windows.

Next: bounded escalation on repeated storm cycles (time-windowed).

### Comment 54 — 2026-02-09 19:37
Applied fix (commit 900964e3): bounded escalation for repeated RequiredFrameDropped storm cycles. 4.5s window, >=3 cycles escalates to hard recovery + re-subscribe.

### Comment 55 — 2026-02-09 20:00
Session 10: Unified worker/decoder IDR gates via ResubscribeReason enum. Double-gate starvation eliminated. DecoderRecovery skips worker gate reset. Committed 359bd910.

### Comment 56 — 2026-02-09 20:25
Session 11 (r27): Runtime test revealed root cause of low FPS + freezes. VT callback returns info_flags=0x4 (RequiredFrameDropped) on 100% of frames on macOS 15/M4, even with status=0 and valid image_buffer. Our storm detection (threshold=36) fires every ~1.5s, causing periodic error escalation. Fix: when VT produces valid pixels (status=0, non-null image_buffer), treat info_flags=0x4 as informational, not error. Set required_frame_dropped=false in callback. Compiles clean, needs runtime validation.

---

## Next Test: r28 — Validate info_flags=0x4 fix (2026-02-09)

### Commits under test
- `e886a8dc` — fix(moq): ignore VT info_flags=0x4 when frame decoded successfully
- `359bd910` — fix(moq): unify worker/decoder IDR gates via ResubscribeReason enum

### Setup
1. Start local BBB publisher: `just dev` in moq repo directory
2. Run demo with logging:
   ```
   RUST_LOG=warn LUMINA_MOQ_ERROR_FORENSICS=1 cargo run -p lumina-video-demo 2>&1 | tee /tmp/moq-fixall-r28.log
   ```
3. Connect to `moqs://localhost/bbb` (or `moq://localhost:4443/anon/bbb`)
4. Let it run 2+ minutes

### Success criteria

| Metric | Before (r26) | Target |
|--------|-------------|--------|
| RequiredFrameDropped storms | Every 36 frames (~1.5s) | Zero (flag ignored) |
| Measured FPS | 11-17fps, drops to 0.0fps | Stable ~24fps |
| Decode errors | 2 per 100 frames (storm-triggered) | 0 (or only real -12909) |
| Freeze windows | Periodic 0.0fps freezes | None from storm path |
| "Skipping worker IDR gate reset" | n/a | On decoder-triggered re-subscribes |
| Pixelation | Visible macroblocking | TBD (may still exist from silent VT corruption) |

### What to look for in logs
- **No more** `RequiredFrameDropped storm` lines (storms eliminated at source)
- **No more** `VT required-frame-drop storm (streak=36)` errors
- If real VT -12909 errors still occur: `drop(no IDR)` should stop at ~24 per event (not 48-130+)
- "Skipping worker IDR gate reset" on any decoder-triggered re-subscribe

### Known remaining issues (not addressed by this fix)
- Silent VT corruption: pixelation with 0 decode errors (open hypothesis — see comment #37)
- 1-2 real VT -12909 errors per session (non-format-related, run-dependent)
- Pixelation from degenerate first-group IDR (6KB vs 92KB)
