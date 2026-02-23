# Agent Rules

Rules are enforced. If you skip one, your PR will be rejected.

## Before Every Commit

```bash
cargo fmt
cargo check
cargo clippy -- -D warnings
cargo test
# Must return nothing — local paths break everyone's build
grep -rE 'path = "(/|\.\.)' **/Cargo.toml
```

For platform-specific changes, also verify relevant features:

```bash
cargo check --features macos-native-video
cargo check --features linux-gstreamer-video
cargo check --features windows-native-video
cargo check --features ffmpeg
cargo check --features moq
```

## NEVER (zero exceptions)

- `unwrap()` or `expect()` in library code — use `?`, `.get()`, `if let`
- Unguarded indexing (`arr[i]`) — use `.get(i)`
- Block the render/UI loop — use `poll_promise`, channels, background threads
- Local path deps in Cargo.toml — use `git = "..."` with rev or branch
- CPU pixel copies in steady-state decode/render — zero-copy or document why
- Heap allocs, blocking I/O, or unbounded queues in per-frame hot paths
- Mutex contention in frame-critical paths — use lock-free or bounded SPSC
- `unsafe` without a `// SAFETY:` comment explaining the invariant
- Vendor external code — use git deps in Cargo.toml
- Global or thread-local state — state belongs in structs
- Hack tests to pass CI — find and fix root causes

## Media Rules

- One master clock (audio or wall). Enforce drift correction.
- Bounded queues with `drop_oldest` for live streams.
- On overload: drop frames, never grow latency.
- On discontinuity (seek/network gap): bounded resync with timeout.
- See [docs/MOQ_BEST_PRACTICES.md](docs/MOQ_BEST_PRACTICES.md) for MoQ transport rules.

## Platform Features

| Feature | Platform | Backend |
|---|---|---|
| `macos-native-video` | macOS | AVFoundation + VideoToolbox |
| `linux-gstreamer-video` | Linux | GStreamer + VA-API |
| `windows-native-video` | Windows | Media Foundation + DXVA2 |
| `ffmpeg` | All | FFmpeg fallback |
| `moq` | All | MoQ live streaming |
