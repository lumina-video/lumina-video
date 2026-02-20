# MoQ Best Practices for lumina-video

MoQ-specific protocol, transport, and media rules for the lumina-video project. For general Rust development standards, see [AGENTS.md](AGENTS.md). For VOD A/V sync patterns, see [NATIVE_AV_SYNC.md](crates/lumina-video/src/media/NATIVE_AV_SYNC.md).

Based on MoQ Transport draft-15 (via moq-lite).

---

## MoQ Transport Contracts

### Control Stream
- The control stream MUST NOT be closed while the session is active. Closing it tears down the entire MoQ session.
- Control stream priority is higher than any media stream — relay and client MUST process control messages before media data.
- SETUP exchange (CLIENT_SETUP → SERVER_SETUP) must complete before any SUBSCRIBE, FETCH, or media flow.

### QUIC Transport
- QUIC DATAGRAM extension negotiation is required when using datagram-based delivery.
- Stream-to-subgroup mapping: objects in the same subgroup travel on the same QUIC stream; different subgroups use different streams.
- Delivery timeout: set an explicit timeout policy for stalled subscriptions. Our catalog fetch uses a 5-second timeout (`worker.rs:1302`).

### Session Lifecycle
- GOAWAY initiates graceful session migration: the relay drains existing subscriptions while subscribers re-establish on a new session.
- Malformed track detection: respond with UNSUBSCRIBE + FETCH_CANCEL. Never cache malformed objects.

## Object Hierarchy & Segmentation

MoQ organizes media as: **Track → Group → Subgroup → Object**.

### Video
- Each IDR keyframe starts a new **Group** (one GOP = one Group).
- Object 0 within a Group is always the IDR frame.
- Group ID must increase monotonically with time.

### Audio
- Each encoded audio chunk is one **Object** (1024 samples for AAC, ~960 for Opus at 48kHz).
- AAC frame rate = `sample_rate / 1024` (e.g., 48kHz → 46.875 fps). This is independent of video fps.

### Invariants
- Objects are immutable once published.
- Group IDs increase monotonically — never reuse or go backwards.

## Priority & Congestion Control

### 4-Tier Scheduling
Priority evaluation order: **Subscriber Priority → Publisher Priority → Group Order → Object ID**.

### Audio Before Video
Audio starvation causes permanent A/V drift that compounds over time. Video frame drops are invisible if brief. Therefore:

| Track | Priority Value | Rationale |
|-------|---------------|-----------|
| Audio | 50 | Higher priority (lower numeric value in WebTransport) |
| Video | 100 | Lower priority — frame drops are recoverable |

Reference: `crates/lumina-video/src/media/moq/worker.rs` — audio track at priority 50 (line 427, 1499), video at priority 100 (line 397).

### Congestion Control
- Avoid AIMD algorithms (Reno/CUBIC) for live media — sawtooth throughput causes periodic stalls.
- Monitor app-limited conditions: when the sender is idle, congestion window should not decay.
- BBR's PROBE_RTT phase halves the congestion window periodically — beware of interaction with live media bitrate.

### Quality Layers
- Use subgroup-based layering: subgroup 0 = base layer, higher subgroups = enhancement layers.
- Under congestion, drop higher subgroups first (enhancement layers before base layer).

## A/V Sync

### Timeline Rules
- Tracks sharing the same namespace share the same timeline.
- Timestamps are integer-based: `numerator / timebase`. Never use floating-point for media timestamps.
- Audio is the master clock for video frame selection (see [NATIVE_AV_SYNC.md](crates/lumina-video/src/media/NATIVE_AV_SYNC.md) for detailed rationale).

### MoQ-Specific Pitfalls
- **Publisher vs viewer time**: MoQ PTS uses publisher wall-clock time (~7200s into a stream) while the viewer starts at ~0s. Do NOT bind MoQ audio position to `FrameScheduler` — the position jump will freeze video playback.
- **Stall-on-underrun**: When the audio ring buffer empties (3 consecutive empty cpal callbacks), freeze video to prevent A/V drift. The `AudioHandle.is_audio_stalled()` flag gates video frame advancement in `FrameScheduler`.
  - Reference: `audio.rs:539` (`STALL_CALLBACK_THRESHOLD = 3`), `frame_queue.rs:2054-2097`.

### Sync Targets
| Metric | Threshold |
|--------|-----------|
| Acceptable | <100ms drift |
| Excellent | <33ms drift (1 frame at 30fps) |

## Live vs VOD Group Order

| Mode | Group Order | Behavior |
|------|------------|----------|
| Live | Descending | Newest groups first; skip stale groups to minimize latency |
| VOD | Ascending | Sequential, complete delivery; no skipping |

- Use **SUBSCRIBE** for live future objects (real-time delivery).
- Use **FETCH** for historical/cached objects (on-demand retrieval).

## Cancellation Safety (Rust-Specific)

**`hang::container::OrderedConsumer::read()` is NOT cancellation-safe.**

Its internal `read_unbuffered()` has two await points (`next_frame()` + `read_chunks()`). If a `tokio::select!` cancels the future between them, the frame is consumed from the QUIC stream but never returned to the caller — it's permanently lost.

### Rules
1. Audio and video consumers MUST run in dedicated `tokio::spawn` tasks.
2. Never place `OrderedConsumer::read()` inside a shared `tokio::select!` loop.
3. Use channels (crossbeam, tokio mpsc) to communicate frames from the spawned task back to the consumer.

### Consequence of Violation
~50% audio frame loss (observed: 25fps received vs 47fps expected at 48kHz AAC). This was the root cause of a major audio quality bug.

Reference: `worker.rs:1566-1579` (`spawn_audio_forward_task`), comments at lines 416-417 and 1564-1565.

## CMAF/LOC Packaging Rules

### Format Selection
- **LOC** (Low Overhead Container): for low-latency and interactive use cases.
- **CMAF** (Common Media Application Format): for legacy player compatibility.
- Each track in the catalog MUST declare its packaging format: `"cmaf"` or `"loc"`.

### CMAF Requirements
- Switching sets require media-time-aligned group numbers across all tracks.
- SAP (Stream Access Point) / GOP boundaries must align with MoQ Group boundaries.
- Decode order within subgroups is maintained by Object ID ordering.
- `altGroup` alignment for ABR (Adaptive Bitrate) switching between renditions.

## Catalog Format

The catalog is a **live track** — it updates as tracks are added or removed.

### Required Fields
- `version` — catalog schema version
- `streamingFormat` — e.g., `"cmaf"` or `"loc"`
- `streamingFormatVersion` — format version string
- `tracks` or `catalogs` — track descriptions or nested catalog references

### Rules
- Track names are unique per namespace.
- Selection properties (codec, resolution, etc.) are immutable once published.
- Delta updates use JSON Patch (RFC 6902) when `supportsDeltaUpdates` is true.
- Decoder initialization via `initData` (base64-encoded) or `initTrack` reference.

Reference: `worker.rs:1293-1392` (`fetch_and_validate_catalog`).

## QoE Metrics & SLOs

### Required Metrics
Every MoQ deployment should track:
- **Startup time** — time to first rendered frame
- **Stall/rebuffer rate** — percentage of session time spent stalled
- **Dropped frames** — frames skipped due to late arrival or congestion
- **Live offset** — delay from publisher to viewer
- **Transport health** — RTT, packet loss, goodput

### Target Thresholds

| Metric | Interactive | Broadcast |
|--------|------------|-----------|
| A/V sync drift | <33ms | <100ms |
| Startup time | <1s | <2s |
| Live offset | <500ms | <2s |
| Stall rate | <0.5% | <1% |

### Regression Gates
PRs that add latency or increase stall rates must include performance data justifying the regression.

## Auth & Privacy

### Token Flow
- Authentication tokens are passed in the SETUP message via the AUTHORIZATION parameter.
- Access control is namespace-scoped: separate permissions for publish, subscribe, and fetch.
- Anonymous access is supported when the relay is configured with a public prefix (`moq-relay/src/auth.rs`).

### Token Lifecycle
- **Token expiry**: relay responds with `EXPIRED_AUTH_TOKEN` session closure.
- **Cache overflow**: relay responds with `AUTH_TOKEN_CACHE_OVERFLOW` session closure.
- Clients must handle both gracefully by re-authenticating and reconnecting.

### Privacy Considerations
- Sensitive metadata (capture timestamps, audio levels, location data) may warrant end-to-end encryption.
- The relay is a transport-level intermediary — it should not inspect media payloads.

Reference: `moq/rs/moq-relay/src/auth.rs` (JWT validation), `moq/rs/moq-token/` (token generation).

## Version Governance & Interop

### Draft Pinning
- Pin the MoQ Transport draft version in code and documentation. Current: **draft-15** via moq-lite.
- Track the ALPN version string explicitly — it changes between drafts.
- The SETUP exchange supports version negotiation across Draft14, Draft15, Draft16, and Lite Draft01/02 (`moq-lite/src/setup.rs`).

### Upgrade Policy
- Require interop verification before bumping draft versions.
- Test against at least one other MoQ implementation before merging a version bump.

### CDN Neutrality
- Relays MUST NOT understand application-level media semantics (codecs, resolutions, bitrates).
- Relay forwarding decisions use only generic transport-level headers (priority, group order, delivery timeout).
- This ensures relay implementations remain codec-agnostic and interoperable.
