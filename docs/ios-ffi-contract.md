# iOS FFI Contract

This document defines the C-ABI contract for `lumina-video-ios`, the iOS FFI
layer over `lumina-video-core`.

## API Model: Poll-Based

The API is purely poll-based. There are no callbacks, no callback registration
functions, and no callback reentrancy concerns.

Swift integration uses a `CADisplayLink` or `Timer` to poll frames each vsync.

## Threading Model

All functions are safe to call from **any thread**. Internal synchronization
uses `parking_lot::Mutex` (non-poisoning). This matches Apple ARC semantics
and simplifies Swift integration.

| Function | Thread | Notes |
|----------|--------|-------|
| `lumina_player_create` | Any | Allocates, starts async init |
| `lumina_player_destroy` | Any | Internal sync; see Destroy section |
| `lumina_player_play` | Any | |
| `lumina_player_pause` | Any | |
| `lumina_player_seek` | Any | |
| `lumina_player_poll_frame` | Any | Returns owned frame |
| `lumina_player_state` | Any | Read-only |
| `lumina_player_position` | Any | |
| `lumina_player_duration` | Any | |
| `lumina_frame_release` | Any | Frees frame memory |
| `lumina_frame_width` | Any | Read-only accessor |
| `lumina_frame_height` | Any | Read-only accessor |
| `lumina_frame_iosurface` | Any | Read-only accessor |

## Opaque Handle Types

```c
typedef struct LuminaPlayer LuminaPlayer;
typedef struct LuminaFrame  LuminaFrame;
```

Both are opaque — Swift sees them as `OpaquePointer`.

## Error Codes

```c
typedef int32_t LuminaError;

#define LUMINA_OK                  0
#define LUMINA_ERROR_NULL_PTR      1  // A required pointer argument was NULL
#define LUMINA_ERROR_INVALID_URL   2  // URL was not valid UTF-8 or empty
#define LUMINA_ERROR_INIT_FAILED   3  // Decoder initialization failed
#define LUMINA_ERROR_DECODE        4  // Decode error during playback
#define LUMINA_ERROR_INTERNAL      5  // Unexpected internal error (panic caught)
```

All functions return `LuminaError` except accessors that return values directly.

## Nullability

| Parameter type | NULL allowed? | Behavior on NULL |
|---------------|---------------|------------------|
| `LuminaPlayer *` | No | Returns `LUMINA_ERROR_NULL_PTR` |
| `LuminaPlayer **` (destroy) | No | Returns `LUMINA_ERROR_NULL_PTR` |
| `*player` in destroy | Yes | No-op, returns `LUMINA_OK` |
| `LuminaFrame *` | No | Returns `LUMINA_ERROR_NULL_PTR` |
| `const char *url` | No | Returns `LUMINA_ERROR_NULL_PTR` |
| Out-params (`LuminaPlayer **out`) | No | Returns `LUMINA_ERROR_NULL_PTR` |

## Lifecycle

### Create

```c
LuminaError lumina_player_create(const char *url, LuminaPlayer **out_player);
```

- `url` must be valid UTF-8, null-terminated.
- On success, `*out_player` is a valid handle. Caller owns it.
- On failure, `*out_player` is NULL.
- Player starts in Loading state; async init begins immediately.

### Destroy

```c
LuminaError lumina_player_destroy(LuminaPlayer **player);
```

- Nulls `*player` after destroy. Second call receives NULL -> returns `LUMINA_OK` (no-op).
- **After successful destroy, all copies of that pointer are invalid and must
  not be used.** This is the caller's responsibility -- the C ABI cannot enforce
  pointer uniqueness.
- **Concurrent destroy + other call**: serialized by internal mutex. One thread
  completes the operation; the other finds the handle destroyed. Using a stale
  pointer copy after destroy is undefined behavior (documented here).

## Playback Control

```c
LuminaError lumina_player_play(LuminaPlayer *player);
LuminaError lumina_player_pause(LuminaPlayer *player);
LuminaError lumina_player_seek(LuminaPlayer *player, double position_secs);
```

- `position_secs` is seconds from start. Negative values are clamped to 0.

## State Query

```c
typedef int32_t LuminaState;

#define LUMINA_STATE_LOADING  0
#define LUMINA_STATE_READY    1
#define LUMINA_STATE_PLAYING  2
#define LUMINA_STATE_PAUSED   3
#define LUMINA_STATE_ENDED    4
#define LUMINA_STATE_ERROR    5

LuminaState lumina_player_state(const LuminaPlayer *player);
double      lumina_player_position(const LuminaPlayer *player);
double      lumina_player_duration(const LuminaPlayer *player);
```

- `lumina_player_duration` returns -1.0 if duration is unknown (live streams).
- `lumina_player_state` returns `LUMINA_STATE_ERROR` if player is NULL.
- `lumina_player_position` returns 0.0 if player is NULL.

## Frame Retrieval

```c
LuminaFrame *lumina_player_poll_frame(LuminaPlayer *player);
```

- Returns owned `LuminaFrame *`. Caller **MUST** call `lumina_frame_release()`.
- Returns NULL if no frame is ready (not an error).
- IOSurface is valid only while `LuminaFrame` is alive.
- If Swift needs the IOSurface past `lumina_frame_release()`, it **MUST**
  `CFRetain()` the IOSurface before releasing the frame.
- Backpressure: if caller doesn't poll, frame_queue drops oldest (existing
  behavior from CorePlayer).

## Frame Accessors

```c
uint32_t         lumina_frame_width(const LuminaFrame *frame);
uint32_t         lumina_frame_height(const LuminaFrame *frame);
IOSurfaceRef     lumina_frame_iosurface(const LuminaFrame *frame);
void             lumina_frame_release(LuminaFrame *frame);
```

- `lumina_frame_iosurface` returns NULL if the frame is CPU-only (no zero-copy).
- `lumina_frame_width/height` return 0 if frame is NULL.
- `lumina_frame_release` is a no-op if frame is NULL.

## Panic Safety

All FFI entry points wrap their bodies in `std::panic::catch_unwind()`. A Rust
panic will not unwind across the FFI boundary. Instead:

- The panic is caught.
- `LUMINA_ERROR_INTERNAL` is returned.
- The error message is logged via `tracing::error!`.

## Memory Model

- `LuminaPlayer` is heap-allocated via `Box::new()`, freed by `Box::from_raw()`.
- `LuminaFrame` is heap-allocated via `Box::new()`, freed by `lumina_frame_release()`.
- All strings passed to the API must be valid UTF-8. Invalid UTF-8 returns
  `LUMINA_ERROR_INVALID_URL`.
- The API does not retain or copy the URL string beyond the `create` call.
