//! FFI safety diagnostics: handle registries + debug metrics.
//!
//! ## Always-on (release + debug)
//!
//! **Player registry**: Tracks live player pointers in a `HashSet<usize>`.
//! - `register_player(ptr)`: adds pointer, returns false if already present.
//! - `unregister_player(ptr)`: removes pointer, returns false if unknown (double-free attempt).
//!
//! Used by `lumina_player_destroy` to prevent double-free UB in production.
//!
//! ## Debug-only (`cfg(debug_assertions)`)
//!
//! **Frame registry**: Same as player registry but for frame pointers. Frame
//! poll/release happen at ~60fps — per-frame mutex+hash overhead needs benchmarking
//! before promotion to release builds.
//!
//! **Metrics**: Atomic counters for created/destroyed/peak/live players & frames,
//! plus total FFI calls. Exposed via `snapshot()` → `FfiMetricsSnapshot`.
//!
//! ## Limitations
//!
//! Address-only tracking cannot detect stale-pointer aliasing after allocator reuse
//! (ABA problem). If a frame is freed and the allocator reuses the same address for
//! a new frame, a stale pointer would pass the registry check. The registry catches
//! the common case (double-free of the same pointer without intervening reallocation)
//! but is not a complete mitigation.
//!
//! ## Scope of protection
//!
//! The player registry guards `lumina_player_destroy` — the only player `Box::from_raw`
//! call site — preventing double-free UB. Frame registry (debug-only) guards
//! `lumina_frame_release` during development. Other FFI entry points dereference raw
//! pointers via `&*player` without registry checks. The Swift wrapper mitigates the
//! broader risk by guarding all methods with `guard let ptr = playerPtr else { return }`
//! and nulling the pointer in deinit.

use std::collections::HashSet;
use std::sync::LazyLock;

use parking_lot::Mutex;

// =========================================================================
// Player registry (always-on)
// =========================================================================

static PLAYER_REGISTRY: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Registers a player pointer. Returns `false` if already registered (bug).
pub fn register_player(ptr: *const u8) -> bool {
    PLAYER_REGISTRY.lock().insert(ptr as usize)
}

/// Unregisters a player pointer. Returns `false` if unknown (double-free attempt).
pub fn unregister_player(ptr: *const u8) -> bool {
    PLAYER_REGISTRY.lock().remove(&(ptr as usize))
}

/// Returns the number of live (registered) players.
#[cfg(debug_assertions)]
pub fn live_player_count() -> usize {
    PLAYER_REGISTRY.lock().len()
}

// =========================================================================
// Frame registry (debug-only)
// =========================================================================

#[cfg(debug_assertions)]
static FRAME_REGISTRY: LazyLock<Mutex<HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Registers a frame pointer. Returns `false` if already registered (bug).
#[cfg(debug_assertions)]
pub fn register_frame(ptr: *const u8) -> bool {
    FRAME_REGISTRY.lock().insert(ptr as usize)
}

/// Unregisters a frame pointer. Returns `false` if unknown (double-free attempt).
#[cfg(debug_assertions)]
pub fn unregister_frame(ptr: *const u8) -> bool {
    FRAME_REGISTRY.lock().remove(&(ptr as usize))
}

/// Returns the number of live (registered) frames.
#[cfg(debug_assertions)]
pub fn live_frame_count() -> usize {
    FRAME_REGISTRY.lock().len()
}

// =========================================================================
// Metrics (debug-only)
// =========================================================================

#[cfg(debug_assertions)]
mod metrics {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::LazyLock;

    use parking_lot::Mutex;

    struct FfiMetrics {
        players_created: AtomicU64,
        players_destroyed: AtomicU64,
        players_peak: AtomicU64,
        frames_created: AtomicU64,
        frames_destroyed: AtomicU64,
        frames_peak: AtomicU64,
        ffi_calls: AtomicU64,
        // Protected by lock for correct peak tracking
        players_live: Mutex<u64>,
        frames_live: Mutex<u64>,
    }

    static METRICS: LazyLock<FfiMetrics> = LazyLock::new(|| FfiMetrics {
        players_created: AtomicU64::new(0),
        players_destroyed: AtomicU64::new(0),
        players_peak: AtomicU64::new(0),
        frames_created: AtomicU64::new(0),
        frames_destroyed: AtomicU64::new(0),
        frames_peak: AtomicU64::new(0),
        ffi_calls: AtomicU64::new(0),
        players_live: Mutex::new(0),
        frames_live: Mutex::new(0),
    });

    pub fn record_player_created() {
        METRICS.players_created.fetch_add(1, Ordering::Relaxed);
        let mut live = METRICS.players_live.lock();
        *live += 1;
        let current = *live;
        let peak = METRICS.players_peak.load(Ordering::Relaxed);
        if current > peak {
            METRICS.players_peak.store(current, Ordering::Relaxed);
        }
    }

    pub fn record_player_destroyed() {
        METRICS.players_destroyed.fetch_add(1, Ordering::Relaxed);
        let mut live = METRICS.players_live.lock();
        *live = live.saturating_sub(1);
    }

    pub fn record_frame_created() {
        METRICS.frames_created.fetch_add(1, Ordering::Relaxed);
        let mut live = METRICS.frames_live.lock();
        *live += 1;
        let current = *live;
        let peak = METRICS.frames_peak.load(Ordering::Relaxed);
        if current > peak {
            METRICS.frames_peak.store(current, Ordering::Relaxed);
        }
    }

    pub fn record_frame_destroyed() {
        METRICS.frames_destroyed.fetch_add(1, Ordering::Relaxed);
        let mut live = METRICS.frames_live.lock();
        *live = live.saturating_sub(1);
    }

    pub fn record_ffi_call() {
        METRICS.ffi_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot of all metrics.
    #[derive(Debug, Clone, Copy, Default)]
    #[repr(C)]
    pub struct FfiMetricsSnapshot {
        pub players_created: u64,
        pub players_destroyed: u64,
        pub players_peak: u64,
        pub players_live: u64,
        pub frames_created: u64,
        pub frames_destroyed: u64,
        pub frames_peak: u64,
        pub frames_live: u64,
        pub ffi_calls: u64,
    }

    pub fn snapshot() -> FfiMetricsSnapshot {
        FfiMetricsSnapshot {
            players_created: METRICS.players_created.load(Ordering::Relaxed),
            players_destroyed: METRICS.players_destroyed.load(Ordering::Relaxed),
            players_peak: METRICS.players_peak.load(Ordering::Relaxed),
            players_live: *METRICS.players_live.lock(),
            frames_created: METRICS.frames_created.load(Ordering::Relaxed),
            frames_destroyed: METRICS.frames_destroyed.load(Ordering::Relaxed),
            frames_peak: METRICS.frames_peak.load(Ordering::Relaxed),
            frames_live: *METRICS.frames_live.lock(),
            ffi_calls: METRICS.ffi_calls.load(Ordering::Relaxed),
        }
    }
}

#[cfg(debug_assertions)]
pub use metrics::{
    record_ffi_call, record_frame_created, record_frame_destroyed, record_player_created,
    record_player_destroyed, snapshot, FfiMetricsSnapshot,
};
