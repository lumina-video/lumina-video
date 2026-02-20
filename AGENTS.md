# lumina-video Agent Development Guidelines

Development standards for lumina-video, aligned with [emilk/egui](https://github.com/emilk/egui) conventions and [damus-io/notedeck](https://github.com/damus-io/notedeck) architecture patterns.

## Commit Requirements

### Logically Distinct
Each commit addresses **one concern**. Don't mix unrelated changes.

### Standalone
Commits can be cherry-picked or reverted independently. A commit removed 50 commits later should not break the codebase.

### Human Reviewable
- Clear commit messages explaining the "why"
- A commit with 10,000 lines is not reviewable
- Prefer simplicity: 1 line > 10 lines > 100 lines

### Verified to Compile
Run `cargo check` before every commit. Code that doesn't compile doesn't get committed.

### Fixes and Refactors Belong in Original Commits
If a fix or refactor addresses code introduced in the same PR, **rebase it into the original commit** rather than adding separate "fix" or "refactor" commits. The git history should look like the code was written correctly from the start.

### Preserve Authorship
When incorporating work from other branches, use `git cherry-pick` to preserve original authorship rather than copying code manually.

## Protecting Work from Loss

**CRITICAL**: Uncommitted work can be permanently lost. Follow these rules strictly.

### Never Leave New Files Untracked
Git does NOT protect untracked files. When you switch branches, untracked files that conflict with the target branch are **deleted without warning**.

```bash
# After creating ANY new file:
git add <new-file>
git commit -m "feat: add <description>"

# NEVER do this:
# 1. Create new file
# 2. Switch branches  ← FILE IS NOW DELETED
# 3. Wonder where your code went
```

### Commit Early and Often
For multi-file implementations, commit after each logical unit:

```bash
# Good: Commit each platform separately
git add linux_video.rs && git commit -m "feat(linux): add zero-copy decoder"
git add macos_video.rs && git commit -m "feat(macos): add zero-copy decoder"

# Bad: Write everything, commit nothing, switch branches, lose everything
```

### Stay on One Branch During Implementation
Create a feature branch and stay there until work is complete:

```bash
git checkout -b feat/my-feature
# ... do ALL work here ...
# ... commit frequently ...
git checkout main  # Only after everything is committed
```

### Before Switching Branches
Always verify your work is safe:

```bash
# Check for uncommitted changes
git status

# If you see untracked files you want to keep:
git add <files>
git commit -m "wip: save work before branch switch"

# OR stash everything including untracked files:
git stash -u  # -u = include untracked files
```

### Verify Commits Exist
After committing, verify:

```bash
git log --oneline -3  # See your commits
git show --stat HEAD  # Verify files are in last commit
```

## Pre-Commit Checklist

```bash
# 1. CRITICAL: Check for local path dependencies (must return nothing!)
grep -r "path = \"/\|path = \"\.\." Cargo.toml */Cargo.toml 2>/dev/null && echo "ERROR: Local paths found!" && exit 1

# 2. Format code
cargo fmt

# 3. Verify compilation
cargo check

# 4. Run linter (warnings are errors)
cargo clippy -- -D warnings

# 5. Run tests
cargo test

# 6. For platform-specific changes, verify features compile
cargo check --features macos-native-video   # macOS
cargo check --features linux-gstreamer-video # Linux
cargo check --features windows-native-video  # Windows
cargo check --features ffmpeg                # FFmpeg
```

## Code Style (egui conventions)

### Documentation
- **Docstrings required** on all types, `struct` fields, and `pub fn`
- Include example code (doc-tests) where applicable
- Use `TODO(username):` not `FIXME`

### Formatting
- Comment format: `// Comment like this.` (space after //)
- Blank lines around `fn`, `struct`, `enum`, etc.
- Lexicographically sorted dependencies and sets
- Each type in its own file unless trivial

### Imports
- Local imports when used in only one place
- When importing traits only for their methods: `use Trait as _;`

### Naming
- Use good, descriptive names for everything
- Idiomatic Rust following [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

## Safety & Error Handling

### Never Panic
- **Never** use `unwrap()` or `expect()` in library code
- **Never** use unguarded array indexing (`arr[i]`)
- Use `.get()`, `if let Some(x)`, or `?` operator
- Code should never crash

### Avoid Unsafe
- Minimize `unsafe` blocks
- Document safety invariants when `unsafe` is necessary

### Don't Fudge CI
Never hack tests to pass. Find and fix the root cause of CI failures.

## Architecture Principles

### No Global Variables
Global state is forbidden, even thread-local. State belongs in structs passed by reference.

### No Blocking in UI Paths
The render loop runs every frame. **Never block it.**

- Use `poll_promise::Promise` for async results
- Check with `promise.ready()` (non-blocking)
- Offload CPU-heavy work to background threads
- Return results via channels or Promises

### Avoid Mutexes
Mutexes can cause UI stalls if held across frames.

**Prefer:**
- `poll_promise::Promise` for async results
- `Rc<RefCell<>>` for single-threaded interior mutability
- `parking_lot::Mutex` over `std::sync::Mutex` (2x faster, no poisoning)
- Lock-free structures (triple buffer) for hot paths
- `crossbeam_channel` for cross-thread communication

### Nevernesting
Favor early returns and guard clauses over deeply nested conditionals.

```rust
// Bad
fn process(x: Option<i32>) {
    if let Some(val) = x {
        if val > 0 {
            // deep nesting...
        }
    }
}

// Good
fn process(x: Option<i32>) {
    let Some(val) = x else { return };
    if val <= 0 { return }
    // flat logic...
}
```

### Frame-Aware Animations
For animations (GIFs, video), track `repaint_at` timestamps. Only request repaints when necessary—don't spin every frame.

## Performance

### Profiling with Puffin
Set up code for performance profiling:

```bash
cargo run --release --features puffin
```

### Profile Suspicious Code
For code suspected of impacting performance, add profiling attributes:

```rust
#[profiling::function]
fn potentially_slow_operation() {
    // ...
}
```

## Media Performance Contracts

### Zero-Copy Invariants
- Decoder output must remain on GPU/native surfaces whenever platform APIs allow.
- No CPU pixel copies in steady-state playback.
- Any required copy must be documented with platform reason and measured impact.
- PRs adding new copies in decode/render paths require perf data justifying the regression.

### Hot-Path Rules
- No heap allocations in per-frame decode/render/present paths.
- No blocking calls, no synchronous I/O, no unbounded queues in frame-critical threads.
- No mutex contention in frame-critical paths; use lock-free or bounded SPSC designs.

### MoQ Protocol Rules
- See [MOQ_BEST_PRACTICES.md](MOQ_BEST_PRACTICES.md) for MoQ-specific transport, sync, and media packaging rules.

### Clocking, Backpressure, and Recovery
- Define one master clock (audio or wall clock) and enforce drift correction policy.
- Use bounded queues with explicit drop policy (`drop_oldest` by default for live).
- On overload, prefer frame drop over latency growth.
- On discontinuity (seek/network gap), perform bounded resync with explicit timeout.

## Code Reuse

### Don't Duplicate
- Check existing code before writing new code
- Refactor existing code to be reusable rather than copying
- One implementation is better than two
- **Verify existing code can't do the job** before creating new code with the same function
- **Don't accrue duplicate code** - always revisit how existing code can be applied/refactored for new issues

### Don't Vendor
Never copy external code into the repo. In `Cargo.toml`, reference forks:

```toml
# Good
some-crate = { git = "https://github.com/org/fork", rev = "abc123" }

# Bad - vendoring
# (copying external code into src/)
```

If vendoring is absolutely necessary, document why no other option is feasible.

### No Local Path Dependencies
**NEVER** commit local path dependencies to any branch. Local paths break builds for other developers.

```toml
# FORBIDDEN - breaks everyone else's build
[patch.crates-io]
some-crate = { path = "/home/user/some-crate" }
some-crate = { path = "../local-fork" }
some-crate = { path = "/tmp/wgpu-fork/naga" }

# CORRECT - use git dependencies
[patch.crates-io]
some-crate = { git = "https://github.com/org/fork", branch = "feature" }
some-crate = { git = "https://github.com/org/fork", rev = "abc123" }
```

**Before every commit and PR**, verify no local paths exist:
```bash
# Check for local path dependencies (should return nothing)
grep -r "path = \"/\|path = \"\.\." Cargo.toml */Cargo.toml 2>/dev/null

# If you find any, replace with git dependencies before committing
```

Local paths are acceptable ONLY for local development and MUST be reverted before committing.

## Platform-Specific Code

### Feature Gating
```rust
#[cfg(target_os = "macos")]
mod macos_video;

#[cfg(feature = "macos-native-video")]
pub use macos_video::MacOSVideoDecoder;
```

### Feature Flags

| Feature | Platform | Description |
|---------|----------|-------------|
| `macos-native-video` | macOS | AVFoundation + VideoToolbox |
| `linux-gstreamer-video` | Linux | GStreamer + VA-API |
| `windows-native-video` | Windows | Media Foundation + DXVA2 |
| `ffmpeg` | All | FFmpeg fallback decoder |

## PR Requirements

- Keep PRs small and focused
- Self-review before requesting review
- Document breaking changes in PR description
- Informative titles and descriptions

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
