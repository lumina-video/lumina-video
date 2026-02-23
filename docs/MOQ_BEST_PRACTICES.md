# MoQ Pitfalls

Hard-learned rules from real bugs. For general rules see [AGENTS.md](../AGENTS.md).

## `OrderedConsumer::read()` is NOT cancellation-safe

Its `read_unbuffered()` has two await points. If `tokio::select!` cancels
between them, the frame is consumed from QUIC but never returned — lost forever.

**Rule**: Audio and video consumers MUST run in dedicated `tokio::spawn` tasks.
Never in a shared `select!` loop.

This caused ~50% audio frame loss (25fps vs 47fps expected).

## MoQ timestamps use publisher wall-clock time

MoQ PTS reflects the publisher's time (~7200s into a stream), not the viewer's
(~0s). Do NOT bind MoQ audio position to `FrameScheduler` — the jump freezes
video playback.

## Audio starves before video

Audio gaps cause permanent A/V drift that compounds. Video drops are invisible.
Audio priority must be numerically lower (= higher priority in WebTransport)
than video. Currently: audio=50, video=100.

If adding a new media track, match this pattern.

## Integer timestamps only

Never use floating-point for media time. Use integer `numerator / timebase`.

## AAC frame rate ≠ video frame rate

AAC: `sample_rate / 1024` frames/sec (48kHz → 46.875 fps). Don't assume
audio and video run at the same rate.

## Stall-on-underrun

When the audio ring buffer empties (3 consecutive empty cpal callbacks),
video MUST freeze to prevent A/V drift. Gated by `AudioHandle.is_audio_stalled()`.
