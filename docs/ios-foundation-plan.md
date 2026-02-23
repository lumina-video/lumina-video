# Plan: lumina-video iOS Foundation

## Status

- **Phase 1**: COMPLETE — 6 issues, 12 commits. Branch `feat/ios-foundation`, PR #42 open against `main`.
- **Phase 2**: OPEN — 9 issues. Requires macOS + Xcode. See Phase 2 section below.

## Beads Issue Map

### Phase 1 — Closed

| Bead | Pri | Type | Title | Commit(s) |
|------|-----|------|-------|-----------|
| lv-x30 | P1 | task | Decouple zero-copy pipeline from egui | `2b5add1`..`f70a885` |
| lv-qby | P1 | feature | Extract lumina-video-core crate | (same as lv-x30) |
| lv-b3y | P1 | task | Expand cfg guards for iOS compilation | `485daa5` |
| lv-ei4 | P2 | feature | iOS cfg guards in core | (same as lv-b3y) |
| lv-24m | P1 | task | Egui regression guard | `0c3e376` |
| lv-5q3 | P2 | task | Egui regression tests | (same as lv-24m) |
| lv-6s6 | P1 | task | Define iOS C-ABI contract | `1f5e192` |
| lv-0k0 | P2 | task | iOS C-ABI contract | (same as lv-6s6) |
| lv-iii | P1 | task | Implement iOS C FFI crate | `ed837f5` |
| lv-cuh | P2 | feature | Implement iOS C FFI crate | (same as lv-iii) |
| lv-vj5 | P1 | task | FFI conformance tests | `66444f2` |
| lv-kxu | P2 | task | FFI conformance tests | (same as lv-vj5) |
| lv-alc | P1 | task | Research iOS integration approach | (pre-plan) |
| lv-5sg | P1 | task | Investigate winit iOS blockers | (pre-plan) |

### Phase 2 — Open

| Bead | Pri | Type | Status | Blocked by |
|------|-----|------|--------|------------|
| lv-791 | P1 | task | open | — |
| lv-2yc | P1 | task | open | — |
| lv-uzx | P2 | task | open | — |
| lv-77o | P2 | task | open | — |
| lv-hb9 | P1 | task | open | lv-791, lv-uzx |
| lv-ylf | P2 | task | open | lv-2yc, lv-uzx |
| lv-9gt | P2 | task | open | lv-2yc, lv-hb9 |
| lv-2lv | P3 | task | open | lv-hb9, lv-ylf |

## Context

lumina-video is a cross-platform hardware-accelerated video player. The iOS port needs an egui-free core crate so iOS can consume the decode/zero-copy pipeline via C FFI. We're on Linux — code requiring macOS/Xcode is written but cannot be fully verified locally.

**Decisions:**
- MoQ deferred — stays in lumina-video (imports rewired via module re-exports)
- API compat: top-level public types preserved, internal module paths may change
- iOS cfgs only in core + ios crate — NOT in egui layer
- Core exposes headless CorePlayer API (decoder-agnostic) for FFI

---

## Issue 1: lv-x30 — Extract lumina-video-core — COMPLETE

### What moves (~21k lines)

Core types + threading: `video.rs`, `audio.rs`, `audio_ring_buffer.rs`, `frame_queue.rs`, `triple_buffer.rs`, `sync_metrics.rs`
Platform decoders: `macos_video.rs`, `linux_video.rs`, `linux_video_gst.rs`, `android_video.rs`, `android_vulkan.rs`, `ndk_image_reader.rs`, `windows_video.rs`, `windows_audio.rs`, `video_decoder.rs`, `audio_decoder.rs`
GPU import: `zero_copy.rs` + `shaders/`
Utilities: `network.rs`, `subtitles.rs`, `vendored_runtime.rs`

### What stays in lumina-video

egui layer: `video_texture.rs`, `video_player.rs`, `video_controls.rs`, `web_video.rs`, `video.wgsl`
MoQ (deferred): `moq_decoder.rs`, `moq_audio.rs`, `moq/`, `nostr_discovery.rs`, `web_moq_decoder.rs`

### MoQ Bridge (first-class subtask)

MoQ modules stay in lumina-video but import types that move to core. The bridge works through module re-exports in `lumina-video/src/media/mod.rs`:

```rust
// Re-export moved modules — preserves super:: paths for MoQ and other local consumers
pub use lumina_video_core::video;
pub use lumina_video_core::audio;
pub use lumina_video_core::frame_queue;
pub use lumina_video_core::triple_buffer;
pub use lumina_video_core::sync_metrics;
pub use lumina_video_core::network;
pub use lumina_video_core::subtitles;

// audio_ring_buffer: pub in core for cross-crate access. NOT public API — do not stabilize.
// Required by moq_audio.rs (super::audio_ring_buffer::RingBufferConfig).
pub(crate) use lumina_video_core::audio_ring_buffer;

// Platform-specific re-exports
#[cfg(target_os = "macos")]
pub use lumina_video_core::video_decoder;
#[cfg(target_os = "macos")]
pub use lumina_video_core::macos_video;
#[cfg(target_os = "android")]
pub use lumina_video_core::android_video;
// ... etc for all platform modules
```

**Critical: audio_ring_buffer visibility fix:**
In `lumina-video-core`, `audio_ring_buffer` must be `pub mod` (not `pub(crate)`) because lumina-video needs cross-crate access for moq_audio's `super::audio_ring_buffer::RingBufferConfig` import. In lumina-video, the re-export uses `pub(crate) use` to avoid exposing it publicly.

MoQ import resolution (confirmed by grep — all work via re-exports unchanged):
| MoQ file | Import | Resolution |
|----------|--------|------------|
| `moq_decoder.rs:59` | `super::video::{CpuFrame, Plane, ...}` | via `pub use core::video` |
| `moq_decoder.rs:65` | `super::video_decoder::HwAccelConfig` | via `pub use core::video_decoder` |
| `moq_decoder.rs:73` | `super::video::MacOSGpuSurface` | via `pub use core::video` |
| `moq_audio.rs:12` | `super::audio::{AudioHandle, ...}` | via `pub use core::audio` |
| `moq_audio.rs:13` | `super::audio_ring_buffer::RingBufferConfig` | via `pub(crate) use core::audio_ring_buffer` |
| `moq/error.rs:96` | `super::super::super::video::VideoError` | via re-export chain |

### CorePlayer — Headless Playback API (Decoder-Agnostic)

CorePlayer is MoQ-agnostic. It takes a pre-created `Box<dyn VideoDecoderBackend>` — the caller decides which decoder to create. Confirmed: MoqDecoder fully implements `VideoDecoderBackend` (metadata, decode_next, handles_audio_internally, audio_handle), so it slots in cleanly.

```rust
// crates/lumina-video-core/src/player.rs
pub struct CorePlayer {
    state: VideoState,
    metadata: Option<VideoMetadata>,
    frame_queue: Arc<FrameQueue>,
    decode_thread: Option<DecodeThread>,
    scheduler: FrameScheduler,
    audio_handle: AudioHandle,
    // platform-specific fields behind cfg gates...
}

impl CorePlayer {
    /// Create player with a pre-created decoder
    pub fn with_decoder(decoder: Box<dyn VideoDecoderBackend + Send>) -> Self { ... }

    /// Create player from URL (selects platform decoder automatically, non-MoQ)
    pub fn new(url: &str) -> Result<Self, VideoError> { ... }

    pub fn play(&mut self) { ... }
    pub fn pause(&mut self) { ... }
    pub fn seek(&mut self, position: Duration) { ... }
    pub fn state(&self) -> &VideoState { ... }
    pub fn position(&self) -> Duration { ... }
    pub fn duration(&self) -> Option<Duration> { ... }

    /// Poll next decoded frame (non-blocking). Returns None if no frame ready.
    pub fn poll_frame(&mut self) -> Option<VideoFrame> { ... }

    pub fn sync_metrics(&self) -> &SyncMetrics { ... }

    /// Poll whether async initialization has completed.
    pub fn check_init_complete(&mut self) -> bool { ... }

    /// Swap audio handle (used by MoQ late-binding in VideoPlayer)
    pub fn set_audio_handle(&mut self, ah: AudioHandle) { ... }
}
```

**CorePlayer state machine (pinned before implementation):**
```text
new(url) / with_decoder(decoder)
  → Loading (init thread spawned)

check_init_complete() returns true + Ok
  → Ready (decoder + decode thread created)

check_init_complete() returns true + Err
  → Error (init failed, terminal)

play() [from Ready/Paused/Ended]
  → Playing

pause() [from Playing]
  → Paused

seek() [from Playing/Paused/Ended]
  → stays in current state, updates position

poll_frame() [from Playing]
  → returns frame if available, None otherwise

EOS detected by decode thread
  → Ended
```

Only valid transitions are enforced; invalid calls (e.g. `play()` in `Loading`) are no-ops.

**MoQ split ownership:**
- `CorePlayer::new()` handles non-MoQ URLs (platform decoder selection)
- `VideoPlayer` (egui) handles MoQ: detects MoQ URL → creates `MoqDecoder` → passes to `CorePlayer::with_decoder()`
- `VideoPlayer` calls `core.set_audio_handle()` during MoQ late-binding
- `VideoPlayer` owns MoQ-specific fields: `moq_stats`, `moq_audio_bound`

**VideoPlayer wraps CorePlayer:**
```rust
pub struct VideoPlayer {
    core: CorePlayer,
    // egui-only:
    texture: Arc<Mutex<Option<VideoTexture>>>,
    pending_frame_writer: TripleBufferWriter<PendingFrame>,
    pending_frame_reader: TripleBufferReader<PendingFrame>,
    show_controls: bool,
    controls_config: VideoControlsConfig,
    // MoQ-specific (stays in lumina-video):
    #[cfg(feature = "moq")]
    moq_stats: Arc<parking_lot::Mutex<Option<MoqStatsHandle>>>,
    #[cfg(feature = "moq")]
    moq_audio_bound: bool,
    // ...
}
```

### Feature Forwarding

```toml
# crates/lumina-video-core/Cargo.toml [features]
default = []
windows-native-video = ["dep:windows", "dep:rodio"]
vendored-runtime = []

# crates/lumina-video/Cargo.toml [features]
default = []
windows-native-video = ["lumina-video-core/windows-native-video", "dep:profiling"]
vendored-runtime = ["lumina-video-core/vendored-runtime"]
moq = [...]  # unchanged — all MoQ deps stay in lumina-video
profiling = ["dep:profiling"]
```

### WASM Compatibility

Core compiles for wasm32 via cfg guards mirroring current `mod.rs`:

```rust
// crates/lumina-video-core/src/lib.rs
#[cfg(not(target_arch = "wasm32"))]
pub mod frame_queue;
#[cfg(not(target_arch = "wasm32"))]
pub mod sync_metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod triple_buffer;
#[cfg(not(target_arch = "wasm32"))]
pub mod network;
#[cfg(not(target_arch = "wasm32"))]
pub mod player;

// Universal (compile everywhere)
pub mod video;
pub mod audio;
/// Internal bridge API — public only for cross-crate re-export by lumina-video.
/// NOT semver-stable. Do not depend on this module directly from external crates.
/// May change or be removed in any minor version.
#[doc(hidden)]
pub mod audio_ring_buffer;
pub mod subtitles;
```

Core `Cargo.toml` uses `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` for parking_lot, crossbeam, tokio, etc.

**Safeguard: WASM re-exports in lumina-video must be cfg-gated.** Every native-only module re-export in `lumina-video/src/media/mod.rs` must preserve its cfg gate:
```rust
// CORRECT — gated
#[cfg(not(target_arch = "wasm32"))]
pub use lumina_video_core::frame_queue;

// WRONG — breaks wasm
pub use lumina_video_core::frame_queue;  // frame_queue uses parking_lot
```

### vendored_runtime Re-export

Currently `pub mod vendored_runtime` at `lumina-video/src/lib.rs:50`. After move:
```rust
// lumina-video/src/lib.rs
#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
pub use lumina_video_core::vendored_runtime;
```
Preserves `lumina_video::vendored_runtime` path. Covered by compile test in lv-24m.

### Commits (split by subsystem, each compiles independently)

1. **`feat(core): create lumina-video-core crate skeleton`** — DONE `2b5add1`
   - Cargo.toml with all non-egui deps, feature flags, wasm cfg sections
   - Empty `lib.rs`, add to workspace members

2. **`refactor(core): move core types and threading primitives`** — DONE `8a03e9c`
   - `git mv`: video.rs, audio.rs, audio_ring_buffer.rs, triple_buffer.rs, sync_metrics.rs, frame_queue.rs, subtitles.rs, network.rs
   - Fix `super::` → `crate::` in moved files
   - Core `lib.rs` module declarations + cfg gates

3. **`refactor(core): move platform decoders`** — DONE `f4d0059`
   - `git mv`: macos_video.rs, linux_video.rs, linux_video_gst.rs, android_video.rs, android_vulkan.rs, ndk_image_reader.rs, windows_video.rs, windows_audio.rs, video_decoder.rs, audio_decoder.rs, vendored_runtime.rs
   - Fix imports

4. **`refactor(core): move zero-copy GPU import`** — DONE `6ec08aa`
   - `git mv`: zero_copy.rs, shaders/
   - Fix imports

5. **`refactor: wire lumina-video to re-export from core + MoQ bridge`** — DONE `646b3c6`
   - Rewrite `media/mod.rs` with re-exports
   - Update lumina-video/Cargo.toml: add core dep, forward features, remove moved deps
   - Re-export vendored_runtime in lib.rs
   - Verify MoQ bridge: `cargo check -p lumina-video --features moq`

6. **`feat(core): introduce CorePlayer with adapter methods + wire decoder init`** — DONE `dad9662`
   - New `core/src/player.rs` with `CorePlayer` struct + full API
   - CorePlayer contains: with_decoder(), new(), play/pause/seek, poll_frame, set_audio_handle
   - Include adapter methods: `init_decoder()`, `check_init_complete()`, `play()`, `pause()`, `seek()`
   - VideoPlayer's `start_async_init()` delegates to `CorePlayer::init_decoder()`

7. **`refactor(ui): complete VideoPlayer migration to CorePlayer`** — DONE `f70a885`
   - Replaced 12+ duplicate fields with `core: CorePlayer`
   - VideoPlayer delegates all playback to `self.core.*`
   - MoQ-specific logic stays in VideoPlayer, calls `core.set_audio_handle()`
   - Removed dead fields and Drop impl from VideoPlayer
   - Fixed doctest in video.rs (`lumina_video::` → `lumina_video_core::`)

### Verification Matrix

```bash
# Per-crate
cargo check -p lumina-video-core
cargo check -p lumina-video
cargo check -p lumina-video-demo

# Feature-gated
cargo check -p lumina-video-core --all-features
cargo check -p lumina-video --all-features
cargo check -p lumina-video --features moq
cargo check -p lumina-video --features vendored-runtime

# WASM
cargo check -p lumina-video-web-demo --target wasm32-unknown-unknown

# Zero egui
grep -r 'egui' crates/lumina-video-core/src/   # zero hits

# Tests + lint
cargo test --workspace
cargo clippy --workspace -- -D warnings

# Runtime
cargo run -p lumina-video-demo -- <test-url>
```

---

## Issue 2: lv-b3y — iOS cfg Guards (core only) — COMPLETE `485daa5`

Changes apply to `crates/lumina-video-core/` only.

| File (in core) | Change |
|----------------|--------|
| `zero_copy.rs:80-86` | Remove `compile_error!` for iOS |
| `zero_copy.rs:125-132` | Add `target_os = "ios"` to `is_platform_supported()` |
| `zero_copy.rs` macos module | `cfg(target_os = "macos")` → `cfg(any(target_os = "macos", target_os = "ios"))` |
| `zero_copy.rs` MTLStorageMode | Managed (macOS) vs Shared (iOS) via cfg |
| `video.rs` MacOSGpuSurface + impls | Expand cfg to include ios |
| `video.rs` DecodedFrame::MacOS | Same |
| `lib.rs` (core) | Module + re-export cfg expansion |
| `macos_video.rs` | Top-level cfg → include ios |
| `audio_decoder.rs`, `video_decoder.rs` | Same |
| `Cargo.toml` (core) | Add `[target.'cfg(target_os = "ios")'.dependencies]` mirroring macOS |

### Verification
```bash
cargo check -p lumina-video-core               # no regression
cargo clippy -p lumina-video-core -- -D warnings
# Full iOS check requires macOS CI
```

---

## Issue 3: lv-24m — Egui Regression Tests — COMPLETE `0c3e376`

1. Full verification matrix from lv-x30 above
2. **Re-export test** in `crates/lumina-video/tests/reexport_test.rs`:
   - Public types accessible via `lumina_video::VideoFrame`, etc.
   - `lumina_video::vendored_runtime` module path preserved (compile test)
3. **MoqDecoder trait + API conformance compile test** — in `crates/lumina-video/src/media/moq_decoder.rs` test module:
   ```rust
   #[cfg(test)]
   mod tests {
       use lumina_video_core::player::CorePlayer;
       use super::MoqDecoder;

       #[test]
       fn moq_decoder_satisfies_core_player_api() {
           fn assert_send<T: Send + 'static>() {}
           assert_send::<MoqDecoder>();
           // Compile-time proof: MoqDecoder can be passed to CorePlayer::with_decoder()
           fn _type_check(d: MoqDecoder) {
               let _player = CorePlayer::with_decoder(Box::new(d));
           }
       }
   }
   ```
4. `cargo test --workspace`
5. `cargo clippy --workspace -- -D warnings`
6. Runtime: `cargo run -p lumina-video-demo`

---

## Issue 4: lv-6s6 — iOS C-ABI Contract — COMPLETE `1f5e192`

### API Model: Poll-Based (No Callbacks)

The API is pure poll-based. No callback registration, no callback reentrancy rules.

### Threading Model

All functions are safe to call from any thread (internally synchronized). This matches Apple ARC semantics and simplifies Swift integration.

| Function | Thread requirement |
|----------|--------------------|
| `lumina_player_create` | Any thread. |
| `lumina_player_destroy` | Any thread (internal sync). |
| `lumina_player_play/pause/seek` | Any thread. |
| `lumina_player_poll_frame` | Any thread. |
| `lumina_frame_release` | Any thread. |
| `lumina_frame_*` accessors | Any thread (read-only). |

### Destroy Signature (safe double-call)

```c
LuminaError lumina_player_destroy(LuminaPlayer **player);
```
- Nulls `*player` after destroy. Second call receives NULL → returns `LUMINA_OK` (no-op).
- **After successful destroy, all copies of that pointer are invalid and must not be used.** This is the caller's responsibility — the C ABI cannot enforce pointer uniqueness.
- **Concurrent destroy + other call:** serialized by internal mutex. One thread completes the operation; the other finds the handle destroyed. Using a stale pointer copy after destroy is undefined behavior (documented in contract).

### Frame Lifetime & Ownership (Poll Semantics)

- `lumina_player_poll_frame()` returns owned `LuminaFrame*`. Caller MUST call `lumina_frame_release()`.
- IOSurface is valid only while `LuminaFrame` is alive.
- If Swift needs IOSurface past `lumina_frame_release()`, MUST `CFRetain()` before releasing frame.
- Backpressure: if caller doesn't poll, frame_queue drops oldest (existing behavior).

### Deliverables

1. `docs/ios-ffi-contract.md` — all sections above + error codes + nullability per param
2. `include/LuminaVideo.h` — C header, verifiable with `clang -fsyntax-only`

---

## Issue 5: lv-iii — Implement iOS C FFI — COMPLETE `ed837f5`

### Structure
```
crates/lumina-video-ios/
  Cargo.toml          # [lib] crate-type = ["staticlib", "lib"], depends on lumina-video-core
  src/
    lib.rs            # #[no_mangle] pub extern "C" entry points
    error.rs          # LuminaError enum + From<VideoError>
    handle.rs         # LuminaPlayer handle type with synchronization
    safety.rs         # ffi_boundary() wrapping catch_unwind (no thread checks — all functions any-thread)
```

### Synchronization Strategy

All FFI functions are any-thread. The handle must be thread-safe:

```rust
// handle.rs
pub struct LuminaPlayer {
    core: Arc<Mutex<CorePlayer>>,  // parking_lot::Mutex for perf
}

// LuminaFrame is immutable after creation — no synchronization needed
pub struct LuminaFrame {
    frame: VideoFrame,  // owns the decoded frame + GPU surface
}
```

FFI entry pattern:
```rust
#[no_mangle]
pub extern "C" fn lumina_player_play(player: *mut LuminaPlayer) -> LuminaError {
    ffi_boundary(|| {
        let player = unsafe { check_not_null!(player)? };
        let mut core = player.core.lock();  // parking_lot — no poisoning
        core.play();
        Ok(())
    })
}
```

`ffi_boundary()` wraps `std::panic::catch_unwind()` + converts Result→LuminaError. No thread affinity checks — purely catch_unwind + null checks.

**Locking policy (codified in safety.rs + enforced by code shape):**
- Each FFI call acquires the mutex exactly once, performs the operation, and releases.
- No callbacks or long-running work while lock is held.
- No nested lock acquisition (prevents deadlock).
- Functions return quickly — I/O and decode happen on background threads managed by CorePlayer.
- **Enforcement pattern:** every FFI entry uses a `with_core()` helper on `LuminaPlayer` that takes a closure:
  ```rust
  impl LuminaPlayer {
      pub fn with_core<F, R>(&self, f: F) -> R
      where F: FnOnce(&mut CorePlayer) -> R {
          let mut core = self.core.lock();
          f(&mut core)
      }
  }
  ```
  This ensures uniform lock scope — the closure runs under the lock, and no FFI function can accidentally hold the lock across blocking operations.

For `lumina_player_destroy(LuminaPlayer **player)`:
```rust
let player_ptr = unsafe { (*player_handle) };
if player_ptr.is_null() { return LUMINA_OK; }  // no-op for double-destroy
unsafe { *player_handle = std::ptr::null_mut(); }  // null out first
let player = unsafe { Box::from_raw(player_ptr) };
// Drop: Arc<Mutex<CorePlayer>> cleanup. If another thread is mid-call,
// its lock guard keeps CorePlayer alive until that call completes.
// After drop, the pointer is invalid — using stale copies is UB (documented).
```

FFI binds to CorePlayer — e.g. `lumina_player_create` calls `CorePlayer::new(url)`, `lumina_player_play` acquires lock and calls `core.play()`, etc.

### Verification (Linux)
```bash
cargo check -p lumina-video-ios
grep -r 'egui' crates/lumina-video-ios/   # zero hits
```

---

## Issue 6: lv-vj5 — FFI Conformance Tests — COMPLETE `66444f2`

Tests in `crates/lumina-video-ios/tests/ffi_conformance.rs`. Tests are cfg-gated to Apple platforms (`#![cfg(any(target_os = "ios", target_os = "macos"))]`) since the underlying decoders require those platforms.

| Category | Tests |
|----------|-------|
| Lifecycle | create/destroy, double destroy, destroy NULL ** |
| NULL safety | NULL args for all 13 FFI functions → proper error codes |
| State queries | initial state is Loading, initial position is zero |
| Thread safety | destroy from different thread than create |
| Stress | 100x rapid create/destroy, play/pause/seek without init |
| Frame accessors | width/height/iosurface/release with NULL → safe defaults |

### Verification (Linux)
```bash
cargo test -p lumina-video-ios               # compiles + runs on Linux
cargo check -p lumina-video-ios              # early drift detection
clang -fsyntax-only include/LuminaVideo.h    # header still valid
```

### Implementation note
Tests use direct Rust function imports (not `extern "C"` linkage) so they run on all platforms. Added `"lib"` to crate-type alongside `"staticlib"` to enable test linking. Clippy allows: `not_unsafe_ptr_arg_deref`, `macro_metavars_in_unsafe`.

---

## Commit Sequence — ALL COMPLETE

```
 1. 2b5add1  feat(core): create lumina-video-core crate skeleton           [lv-x30] DONE
 2. 8a03e9c  refactor(core): move core types and threading primitives      [lv-x30] DONE
 3. f4d0059  refactor(core): move platform decoders                        [lv-x30] DONE
 4. 6ec08aa  refactor(core): move zero-copy GPU import                     [lv-x30] DONE
 5. 646b3c6  refactor: wire lumina-video to re-export from core + MoQ bridge [lv-x30] DONE
 6. dad9662  feat(core): introduce CorePlayer with adapter methods          [lv-x30] DONE
 7. f70a885  refactor(ui): complete VideoPlayer migration to CorePlayer    [lv-x30] DONE
 8. 485daa5  feat(ios): expand cfg guards for iOS in core                  [lv-b3y] DONE
 9. 0c3e376  test: add egui regression verification                        [lv-24m] DONE
10. 1f5e192  docs(ios): define C-ABI contract                              [lv-6s6] DONE
11. ed837f5  feat(ios): implement C FFI crate                              [lv-iii] DONE
12. 66444f2  test(ios): add FFI conformance tests                          [lv-vj5] DONE
    1d079af  chore: update Cargo.lock for lumina-video-ios                          DONE
```

Branch: `feat/ios-foundation` | PR: #42 against `main`

Each commit compiles independently via `cargo check`.

## Key Files

| File | Issues | Action |
|------|--------|--------|
| `Cargo.toml` (root workspace) | x30 | Add core member |
| `crates/lumina-video-core/Cargo.toml` | x30 | New |
| `crates/lumina-video-core/src/lib.rs` | x30 | New |
| `crates/lumina-video-core/src/player.rs` | x30 | New — CorePlayer |
| `crates/lumina-video/Cargo.toml` | x30 | Add core dep, forward features |
| `crates/lumina-video/src/media/mod.rs` | x30 | Rewrite — re-exports + MoQ bridge |
| `crates/lumina-video/src/lib.rs` | x30 | vendored_runtime re-export |
| `crates/lumina-video/src/media/video_player.rs` | x30 | Wrap CorePlayer |
| All ~20 moved `*.rs` files | x30 | Fix super:: → crate:: |
| `crates/lumina-video-core/src/zero_copy.rs` | b3y | Remove iOS compile_error, expand cfgs |
| `crates/lumina-video-core/src/video.rs` | b3y | Expand MacOSGpuSurface cfgs |
| `crates/lumina-video/tests/reexport_test.rs` | 24m | New — API surface test |
| `docs/ios-ffi-contract.md` | 6s6 | New |
| `include/LuminaVideo.h` | 6s6 | New |
| `crates/lumina-video-ios/` | iii, vj5 | New crate |

---

# Phase 2: iOS Platform Integration (macOS + Xcode required)

Phase 1 (above) was completed on Linux — all Rust code, C header, FFI crate, and tests.
Phase 2 requires a macOS machine with Xcode for Swift compilation, Simulator testing, and device deployment.

## Dependency Graph

```
lv-791 (build pipeline) ──────────────┐
                                      ├──→ lv-hb9 (integration harness) ──→ lv-9gt (distribution)
lv-uzx (iOS audio) ──────────────────┤                                      ↑
                                      │                                      │
lv-2yc (Swift wrapper) ──────────────┤──→ lv-ylf (UI adapters) ────────→ lv-2lv (demo app)
                                      │                                      ↑
                                      └──────────────────────────────────────┘

lv-77o (safety instrumentation) — independent
```

## Ready to start (no blockers)

### lv-791 [P1] — iOS Build Pipeline: Rust Cross-Compilation and Static Library

**Owner:** alltheseas | **Labels:** build, ios | **Blocks:** lv-hb9

Set up Rust cross-compilation for iOS and prove we can produce a linkable artifact.

**Steps:**
- `rustup target add aarch64-apple-ios aarch64-apple-ios-sim`
- Cargo.toml: target-specific deps for iOS (objc2 crates, framework linking via `#[link]`)
- Build script (`scripts/build-ios.sh`) that produces `.a` for device + simulator
- Verify: `ar -t` and `nm` on output confirm expected FFI symbols are exported
- Minimal Xcode project (`ios/`) that links the `.a` and calls one FFI function

**Acceptance criteria:**
- [ ] `scripts/build-ios.sh` produces `target/aarch64-apple-ios/release/liblumina_video_ios.a`
- [ ] `scripts/build-ios.sh` produces `target/aarch64-apple-ios-sim/release/liblumina_video_ios.a`
- [ ] `nm` output shows expected `extern C` symbols (at least one test symbol)
- [ ] Minimal Xcode project links and runs on iOS Simulator (even if it does nothing useful)

---

### lv-2yc [P1] — LuminaVideoPlayer Swift Wrapper: Core API

**Owner:** alltheseas | **Labels:** ios, swift | **Depends on:** lv-vj5 (closed) | **Blocks:** lv-9gt, lv-ylf

Swift class wrapping the C FFI. Pure model layer — no UI concerns.

**Location:** `ios/lumina-video-bridge/Sources/LuminaVideoPlayer.swift`

**API surface:**
- `init(url: URL, audioConfig: AudioSessionConfig? = nil)`
- `play()`, `pause()`, `seek(to: TimeInterval)`
- `var volume: Float { get set }`
- `var isMuted: Bool { get set }`
- `var duration: TimeInterval? { get }`
- `var currentTime: TimeInterval? { get }`
- `var state: LuminaVideoState { get }` (enum: loading, ready, playing, paused, buffering, error, ended)

**Audio ownership hooks (per lv-uzx ownership decision):**
- `AudioSessionConfig`: category, mode, options, activateOnInit
- Default: `.playback` + `mixWithOthers` + `activateOnInit=true`
- Host app can pass `nil` to skip audio session config entirely (app manages its own)

**Delegate protocol — `LuminaVideoPlayerDelegate`:**
- `didChangeState(_ state: LuminaVideoState)`
- `didEncounterError(_ error: LuminaVideoError)`
- `didReceiveFrame(ioSurface: IOSurface, width: Int, height: Int, pts: TimeInterval)`

**Threading contract (mirrors FFI spec):**
- init, deinit: main thread only
- play, pause, seek, volume, muted: any thread
- Delegate callbacks: decode thread (caller must dispatch to main if needed)

**Lifecycle:**
- deinit calls `lumina_player_destroy` — no handle leak
- Player is not `Sendable` (main-thread-bound creation)

**Acceptance criteria:**
- [ ] Compiles and links against `liblumina_video_ios.a`
- [ ] Delegate callbacks fire for state changes and frame delivery
- [ ] deinit calls destroy — Instruments shows zero handle leaks over 100 create/destroy cycles
- [ ] Thread contract enforced: init on background thread triggers `assertionFailure` in debug, error in release
- [ ] `AudioSessionConfig=nil` skips audio session setup (verified via `AVAudioSession.sharedInstance().category` unchanged)
- [ ] Usable from both SwiftUI and UIKit without modification (no UIKit imports in this file)

---

### lv-uzx [P2] — iOS Audio: AVAudioSession Lifecycle and Interruption Handling

**Owner:** alltheseas | **Labels:** audio, ios | **Depends on:** lv-6s6 (closed) | **Blocks:** lv-hb9, lv-ylf

Configure AVAudioSession for iOS video playback.

**Ownership decision:** Host app owns AVAudioSession policy. Rationale: AVAudioSession is an app-global singleton — the host app may have its own audio category/mode requirements (e.g., a VoIP app using `.playAndRecord`). lumina-video must not override the host's session config silently.

**Consequences for FFI contract (lv-6s6):**
- Add optional `audio_session_config` param to player create function
- If NULL, lumina-video sets `.playback` + `mixWithOthers` as default
- If provided, host app has already configured AVAudioSession and lumina-video uses it as-is
- Document this in `docs/ios-ffi-contract.md` under Audio Ownership section

**Consequences for Swift wrapper (lv-2yc):**
- `LuminaVideoPlayer.init(url:audioSessionConfig:)` with default parameter
- `AudioSessionConfig` struct: category, mode, options, `activateOnInit` (bool)
- Expose `configureAudioSession()` as a separate callable for apps that set up audio later

**Scope:**
- Session activation: `setActive(true)` before cpal stream, with error handling
- Interruption handling: `AVAudioSession.interruptionNotification` — pause on `.began`, resume on `.ended` if `shouldResume=true`
- Route changes: `routeChangeNotification` — pause on headphone unplug (Apple HIG)
- Background mode: document `Info.plist` `UIBackgroundModes` audio requirement
- Sample rate: use device default, document why (avoid resampling glitches)

**Acceptance criteria:**
- [ ] Audio plays on iOS Simulator with default config
- [ ] Audio plays on physical device
- [ ] Interruption (phone call sim) pauses playback; resume works when `shouldResume=true`
- [ ] Headphone disconnect pauses playback
- [ ] Host app can pass custom AVAudioSession category — lumina-video respects it
- [ ] No audio glitches at device default sample rate
- [ ] Ownership model documented in lv-6s6 FFI contract

---

### lv-77o [P2] — iOS FFI Safety and Performance Instrumentation

**Owner:** alltheseas | **Labels:** ios, perf, safety, test | **Depends on:** lv-iii (closed) | **Independent — can be done anytime**

Cross-cutting safety and observability for the iOS FFI boundary.

**Instrumentation (puffin + os_signpost):**
- FFI entry/exit timing (per function)
- IOSurface import latency (CVPixelBuffer → MTLTexture)
- Frame callback delivery latency (decode complete → Swift callback invoked)
- Expose metrics via FFI for Swift-side `os_signpost` or dashboard integration

**Stress tests:**
- [ ] Create/destroy 1000x loop — zero leaked handles, IOSurfaces, `wgpu::Texture`s
- [ ] 4 threads concurrent play/pause/seek for 60s — no crash, deadlock, TSAN violation
- [ ] Seek storm: 100 seeks in 1s — all complete or error, no hang (timeout: 10s)
- [ ] 10 min continuous 1080p playback — RSS delta < 20MB

**Lifetime/ownership verification:**
- [ ] IOSurface refcount balanced: count before callback == count after frame consumed
- [ ] CVPixelBuffer wrapper drop triggers IOSurface release (no stale pointers)
- [ ] `wgpu::Texture` does not outlive source IOSurface (verified via use-after-free ASAN check)
- [ ] No retain cycles between player handle, callback closures, and IOSurface owners

**Leak detection:**
- [ ] Instruments Leaks: zero leaked Rust or CoreVideo objects over 5min run
- [ ] MallocStackLogging enabled in debug builds for root-cause analysis

**Platform matrix:**
- [ ] All stress tests pass on Simulator (ASAN + TSAN enabled)
- [ ] All stress tests pass on physical device

**Exit criteria:** ALL checkboxes above green.

---

## Blocked (waiting on dependencies)

### lv-hb9 [P1] — iOS Integration Harness: End-to-End Validation (blocked by: lv-791, lv-uzx)

**Owner:** alltheseas | **Labels:** ios, test, validation | **Depends on:** lv-791, lv-iii (closed), lv-uzx | **Blocks:** lv-2lv, lv-9gt

Minimal iOS app that validates the full decode→audio→render pipeline before wrapper polish. Calls C FFI directly — not through Swift wrapper.

**Location:** `ios/test-harness/`

**Validation scenarios (all must pass):**

*Decode + render:*
- [ ] HLS stream plays on iOS Simulator (decode → IOSurface → Metal → screen)
- [ ] MP4 file plays on physical device
- [ ] Zero-copy verified: no `CVPixelBufferLockBaseAddress` calls in Instruments trace
- [ ] Frame rate: >= 29fps sustained for 30fps content (< 1% dropped frames over 60s)

*Seek:*
- [ ] Seek to 0%, 50%, 95% — correct frame displayed within 500ms
- [ ] Seek during playback — no freeze, no crash
- [ ] Rapid seek (10 seeks in 2s) — player recovers to stable playback

*Audio:*
- [ ] Audio plays in sync with video (A/V drift < 50ms measured via `sync_metrics`)
- [ ] Interruption (phone call sim) pauses; resume works
- [ ] Headphone disconnect pauses playback
- [ ] Background → foreground: audio resumes without glitch

*Lifecycle:*
- [ ] Create, play 5s, destroy — 50x loop, zero leaked allocations
- [ ] Background → foreground transition: no crash, playback resumes
- [ ] Memory growth over 5 minutes continuous 1080p playback < 10MB RSS delta

**Platform matrix:**
- [ ] All scenarios pass on aarch64-apple-ios-sim (Xcode Simulator)
- [ ] All scenarios pass on physical device (aarch64-apple-ios)

**Exit criteria:** ALL checkboxes above green on both Simulator and device.

---

### lv-ylf [P2] — iOS UI Adapters: SwiftUI LuminaVideoView and UIKit Support (blocked by: lv-2yc, lv-uzx)

**Owner:** alltheseas | **Labels:** ios, swift, ui | **Depends on:** lv-2yc, lv-uzx | **Blocks:** lv-2lv

Rendering adapters that consume IOSurface frames from `LuminaVideoPlayer` delegate and display them.

**Location:** `ios/lumina-video-bridge/Sources/UI/`

**SwiftUI:**
- `LuminaVideoView`: SwiftUI View (`UIViewRepresentable` wrapping `LuminaVideoUIView`)
- Usage: `LuminaVideoView(url: myURL).aspectRatio(contentMode: .fit)`

**UIKit:**
- `LuminaVideoUIView`: UIView subclass backed by `CAMetalLayer`
- IOSurface → MTLTexture via `MTLDevice.makeTexture(descriptor:iosurface:plane:)`
- Frame pacing via `CADisplayLink` (not timer — syncs to display refresh)

**Shared rendering:**
- Aspect ratio modes: fit, fill, stretch (configurable)
- Orientation-aware layout (auto-adjusts on rotation)
- Safe area respected (no content under notch/dynamic island)
- Zero-copy invariant: IOSurface → MTLTexture only, no CPU readback

**Acceptance criteria:**
- [ ] `LuminaVideoView` renders 30fps video in SwiftUI app without dropped frames
- [ ] `LuminaVideoUIView` renders 30fps video in UIKit app without dropped frames
- [ ] No tearing or flicker during 60s continuous playback
- [ ] Aspect ratio correct after device rotation (portrait ↔ landscape)
- [ ] Instruments GPU trace: zero texture uploads (CPU → GPU) — only IOSurface imports
- [ ] Memory stable: no texture leak over 5 minutes (`MTLTexture` count does not grow)
- [ ] `CADisplayLink` callback < 2ms average (not blocking main thread)

---

### lv-9gt [P2] — iOS Distribution: XCFramework, Swift Package, CI, Versioning (blocked by: lv-2yc, lv-hb9)

**Owner:** alltheseas | **Labels:** build, ci, distribution, ios | **Depends on:** lv-2yc, lv-hb9

Package lumina-video-ios for distribution with reproducible builds, versioning, and CI.

**XCFramework:**
- `scripts/package-ios.sh`: builds device + sim, runs `xcodebuild -create-xcframework`
- Include `LuminaVideo.h` and `module.modulemap` in framework headers
- Fat simulator lib (`aarch64-apple-ios-sim` + `x86_64-apple-ios`)

**Versioning policy:**
- XCFramework version tracks `lumina-video-core` `Cargo.toml` version (single source of truth)
- Swift Package version tag format: `ios-v{major}.{minor}.{patch}` (e.g., `ios-v0.1.0`)
- `CHANGELOG.md` section for iOS-specific changes
- Semantic versioning: C-ABI changes are major bumps (header diff enforced in CI)

**Artifact reproducibility:**
- Deterministic build flags: `ZERO_AR_DATE=1`, strip debug symbols consistently
- Checksum file (`.sha256`) published alongside xcframework
- CI archives build artifacts for each tagged release

**Signing:**
- Code signing identity configurable via env var (`LUMINA_CODE_SIGN_IDENTITY`)
- Default: unsigned (consumer re-signs). Document this.
- Optional: Apple Developer ID signing for direct distribution

**Symbol/export checks:**
- CI step: `nm -gU` output diffed against `LuminaVideo.h` — fail on unexpected symbols
- CI step: abi-compliance-checker or manual header diff between tagged versions
- No C++ symbols leaked (`extern C` only)

**Swift Package:**
- `ios/Package.swift` wrapping xcframework as `binaryTarget`
- Expose `LuminaVideoPlayer` + UI adapters as Swift source
- `Package.swift` version pinned to xcframework version

**CI:**
- GitHub Actions: `cargo check --target aarch64-apple-ios`
- GitHub Actions: build xcframework (macOS runner)
- GitHub Actions: `xcodebuild test` on Simulator
- GitHub Actions: symbol check (`nm` diff)
- GitHub Actions: archive + checksum on tag push

**Acceptance criteria:**
- [ ] `scripts/package-ios.sh` produces `LuminaVideo.xcframework` with correct architectures
- [ ] Swift Package resolves and builds in a fresh Xcode project via SPM
- [ ] Version in `Package.swift` matches `Cargo.toml` version
- [ ] CI passes on PR and tag push
- [ ] `nm` symbol diff catches header/impl divergence
- [ ] Checksum file generated for xcframework artifact
- [ ] Header diff between two versions correctly identifies ABI changes

---

### lv-2lv [P3] — iOS Demo App (blocked by: lv-hb9, lv-ylf)

**Owner:** alltheseas | **Labels:** demo, ios, swift | **Depends on:** lv-hb9, lv-ylf

Polished SwiftUI demo app showcasing lumina-video on iOS. Ships after validation, not before.

**Features:**
- Play sample HLS stream and local MP4
- Transport controls (play/pause/seek scrubber/volume)
- Stream URL text input
- Dark mode support
- Orientation: landscape + portrait

**Acceptance criteria:**
- [ ] Runs on physical device and Simulator
- [ ] Plays at least one HLS and one MP4 source
- [ ] Controls work correctly
- [ ] No crashes during normal usage

---

## Suggested Commit Sequence (Phase 2)

```
13. feat(ios): add build pipeline and cross-compilation scripts    [lv-791]
14. feat(ios): LuminaVideoPlayer Swift wrapper                     [lv-2yc]
15. feat(ios): AVAudioSession lifecycle and interruption handling   [lv-uzx]
16. feat(ios): FFI safety and performance instrumentation           [lv-77o]
17. test(ios): integration harness with XCTest                      [lv-hb9]
18. feat(ios): SwiftUI LuminaVideoView and UIKit adapter            [lv-ylf]
19. feat(ios): XCFramework, Swift Package, CI                       [lv-9gt]
20. feat(ios): demo app                                             [lv-2lv]
```

---

## Fixes Applied During Implementation

| Issue | Fix |
|-------|-----|
| `AudioHandle` scope | Kept ungated (needed by `audio_handle()` return type, not just MoQ) |
| `VideoDecoderBackend` unused | cfg-gated for `feature = "moq"` only |
| MoQ re-entry in `start_async_init()` | Added `moq_init_promise.is_some()` guard |
| CorePlayer duplicate state updates | `update_frame()` no longer sets state (CorePlayer does it in `poll_frame()`) |
| VideoPlayer Drop removed | CorePlayer's own Drop handles cleanup |
| `AudioState::Disabled` | Changed to `AudioState::Uninitialized` (actual variant) |
| `VideoError::Decode` | Changed to `VideoError::DecodeFailed` (actual variant) |
| staticlib test linking | Added `"lib"` to crate-type |
| Doctest path | `lumina_video::` → `lumina_video_core::` in core's video.rs |
