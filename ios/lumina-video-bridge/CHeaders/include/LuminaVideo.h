/**
 * LuminaVideo.h — C-ABI for lumina-video-ios
 *
 * Poll-based video player API for iOS/Swift integration.
 * All functions are thread-safe (internally synchronized).
 *
 * See docs/ios-ffi-contract.md for the full contract.
 */

#ifndef LUMINA_VIDEO_H
#define LUMINA_VIDEO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __APPLE__
#include <IOSurface/IOSurfaceRef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Error codes                                                                */
/* ========================================================================= */

typedef int32_t LuminaError;

#define LUMINA_OK                  0
#define LUMINA_ERROR_NULL_PTR      1
#define LUMINA_ERROR_INVALID_URL   2
#define LUMINA_ERROR_INIT_FAILED   3
#define LUMINA_ERROR_DECODE        4
#define LUMINA_ERROR_INTERNAL      5
#define LUMINA_ERROR_INVALID_ARG   6

/* ========================================================================= */
/* Playback state                                                             */
/* ========================================================================= */

typedef int32_t LuminaState;

#define LUMINA_STATE_LOADING  0
#define LUMINA_STATE_READY    1
#define LUMINA_STATE_PLAYING  2
#define LUMINA_STATE_PAUSED   3
#define LUMINA_STATE_ENDED    4
#define LUMINA_STATE_ERROR    5

/* ========================================================================= */
/* Opaque types                                                               */
/* ========================================================================= */

/** Opaque video player handle. */
typedef struct LuminaPlayer LuminaPlayer;

/** Opaque decoded video frame. Caller must release via lumina_frame_release(). */
typedef struct LuminaFrame LuminaFrame;

/* ========================================================================= */
/* Player lifecycle                                                           */
/* ========================================================================= */

/**
 * Creates a new video player for the given URL.
 *
 * @param url           Null-terminated UTF-8 URL string. Must not be NULL.
 * @param out_player    On success, receives a valid player handle. Must not be NULL.
 *                      On failure, *out_player is set to NULL.
 * @return LUMINA_OK on success, or an error code.
 */
LuminaError lumina_player_create(const char *url,
                                 LuminaPlayer **out_player);

/**
 * Destroys a video player and frees all resources.
 *
 * Sets *player to NULL after destroy. Safe to call with *player == NULL (no-op).
 *
 * @param player    Pointer to the player handle pointer. Must not be NULL.
 *                  After return, *player is NULL.
 * @return LUMINA_OK on success, LUMINA_ERROR_NULL_PTR if player is NULL.
 */
LuminaError lumina_player_destroy(LuminaPlayer **player);

/* ========================================================================= */
/* Playback control                                                           */
/* ========================================================================= */

/**
 * Starts or resumes playback.
 *
 * @param player    Valid player handle. Must not be NULL.
 * @return LUMINA_OK on success.
 */
LuminaError lumina_player_play(LuminaPlayer *player);

/**
 * Pauses playback.
 *
 * @param player    Valid player handle. Must not be NULL.
 * @return LUMINA_OK on success.
 */
LuminaError lumina_player_pause(LuminaPlayer *player);

/**
 * Seeks to a position in seconds from the start.
 *
 * @param player        Valid player handle. Must not be NULL.
 * @param position_secs Position in seconds. Negative values are clamped to 0.
 * @return LUMINA_OK on success.
 */
LuminaError lumina_player_seek(LuminaPlayer *player, double position_secs);

/* ========================================================================= */
/* State queries                                                              */
/* ========================================================================= */

/**
 * Returns the current playback state.
 *
 * @param player    Valid player handle. Returns LUMINA_STATE_ERROR if NULL.
 * @return Current state enum value.
 */
LuminaState lumina_player_state(const LuminaPlayer *player);

/**
 * Returns the current playback position in seconds.
 *
 * @param player    Valid player handle. Returns 0.0 if NULL.
 * @return Position in seconds.
 */
double lumina_player_position(const LuminaPlayer *player);

/**
 * Returns the video duration in seconds, or -1.0 if unknown.
 *
 * @param player    Valid player handle. Returns -1.0 if NULL.
 * @return Duration in seconds, or -1.0 for live/unknown.
 */
double lumina_player_duration(const LuminaPlayer *player);

/* ========================================================================= */
/* Audio control                                                              */
/* ========================================================================= */

/**
 * Sets the muted state.
 *
 * @param player    Valid player handle. Must not be NULL.
 * @param muted     true to mute, false to unmute.
 * @return LUMINA_OK on success.
 */
LuminaError lumina_player_set_muted(LuminaPlayer *player, bool muted);

/**
 * Returns the current muted state.
 *
 * @param player    Valid player handle. Returns true (muted) if NULL.
 * @return true if muted, false if unmuted.
 */
bool lumina_player_is_muted(const LuminaPlayer *player);

/**
 * Sets the volume level (0-100).
 *
 * Values outside 0-100 are clamped.
 *
 * @param player    Valid player handle. Must not be NULL.
 * @param volume    Volume level (0 = silent, 100 = full).
 * @return LUMINA_OK on success.
 */
LuminaError lumina_player_set_volume(LuminaPlayer *player, int32_t volume);

/**
 * Returns the current volume level (0-100).
 *
 * @param player    Valid player handle. Returns 0 if NULL.
 * @return Volume level (0-100).
 */
int32_t lumina_player_volume(const LuminaPlayer *player);

/* ========================================================================= */
/* Frame retrieval                                                            */
/* ========================================================================= */

/**
 * Polls for the next decoded video frame.
 *
 * Returns NULL if no frame is ready (not an error). The returned frame is
 * owned by the caller and MUST be released via lumina_frame_release().
 *
 * The IOSurface inside the frame is valid only while the LuminaFrame is alive.
 * If you need the IOSurface to outlive the frame, CFRetain() it first.
 *
 * @param player    Valid player handle. Returns NULL if player is NULL.
 * @return Owned frame pointer, or NULL if no frame is ready.
 */
LuminaFrame *lumina_player_poll_frame(LuminaPlayer *player);

/* ========================================================================= */
/* Frame accessors                                                            */
/* ========================================================================= */

/**
 * Returns the frame width in pixels.
 *
 * @param frame    Valid frame handle. Returns 0 if NULL.
 */
uint32_t lumina_frame_width(const LuminaFrame *frame);

/**
 * Returns the frame height in pixels.
 *
 * @param frame    Valid frame handle. Returns 0 if NULL.
 */
uint32_t lumina_frame_height(const LuminaFrame *frame);

#ifdef __APPLE__
/**
 * Returns the IOSurface for zero-copy Metal rendering.
 *
 * Returns NULL if the frame is CPU-only (no zero-copy surface available).
 * The IOSurface is valid only while the LuminaFrame is alive. Call
 * CFRetain() if you need it to outlive the frame.
 *
 * @param frame    Valid frame handle. Returns NULL if frame is NULL.
 */
IOSurfaceRef lumina_frame_iosurface(const LuminaFrame *frame);
#endif

/**
 * Releases a decoded video frame and frees its resources.
 *
 * No-op if frame is NULL.
 *
 * @param frame    Frame to release (may be NULL).
 */
void lumina_frame_release(LuminaFrame *frame);

/* ========================================================================= */
/* Diagnostics                                                                */
/* ========================================================================= */

/**
 * FFI diagnostics snapshot.
 *
 * All fields are zero in release builds (debug_assertions disabled).
 * In debug builds, tracks player/frame lifecycle and FFI call counts.
 */
typedef struct LuminaDiagnostics {
    uint64_t players_created;
    uint64_t players_destroyed;
    uint64_t players_peak;
    uint64_t players_live;
    uint64_t frames_created;
    uint64_t frames_destroyed;
    uint64_t frames_peak;
    uint64_t frames_live;
    uint64_t ffi_calls;
} LuminaDiagnostics;

/**
 * Fills a diagnostics snapshot.
 *
 * @param out    Pointer to LuminaDiagnostics struct. Must not be NULL.
 * @return LUMINA_OK on success, LUMINA_ERROR_NULL_PTR if out is NULL.
 */
LuminaError lumina_diagnostics_snapshot(LuminaDiagnostics *out);

#ifdef __cplusplus
}
#endif

#endif /* LUMINA_VIDEO_H */
