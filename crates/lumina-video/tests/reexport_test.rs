//! Compile-time regression test for the lumina-video public API surface.
//!
//! Verifies that types moved to lumina-video-core remain accessible through
//! the original lumina-video paths. If this file compiles, the re-exports work.

// Core types accessible via lumina_video:: (compile-time import check)
#[allow(unused_imports)]
use lumina_video::{
    AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync, CpuFrame,
    DecodedFrame, HwAccelType, PixelFormat, Plane, VideoControls, VideoControlsConfig,
    VideoControlsResponse, VideoDecoderBackend, VideoError, VideoFrame, VideoMetadata,
    VideoPlayerHandle, VideoState,
};

// Native-only types (not wasm32) — compile-time import check
#[allow(unused_imports)]
use lumina_video::{
    SyncMetrics, SyncMetricsSnapshot, VideoPlayer, VideoPlayerExt, VideoPlayerResponse,
    SYNC_DRIFT_THRESHOLD_MS,
};

// Subtitle types via media:: path — compile-time import check
#[allow(unused_imports)]
use lumina_video::media::subtitles::{SubtitleCue, SubtitleError, SubtitleStyle, SubtitleTrack};

#[test]
fn public_types_are_accessible() {
    // Compile-time only — if this compiles, the re-exports work.
    fn _assert_types() {
        let _: fn() -> VideoState = || VideoState::Loading;
        let _: fn() -> HwAccelType = || HwAccelType::None;
        let _: fn() -> PixelFormat = || PixelFormat::Yuv420p;
        let _: fn() -> AudioState = || AudioState::Uninitialized;
    }
}

#[test]
fn video_player_constructors() {
    let player = VideoPlayer::new("test.mp4");
    assert!(matches!(player.state(), VideoState::Loading));
    assert!(!player.is_playing());

    let player2 = VideoPlayer::new("test.mp4")
        .with_autoplay(true)
        .with_loop(true);
    assert!(matches!(player2.state(), VideoState::Loading));
}

#[test]
fn subtitle_types_accessible() {
    let result = SubtitleTrack::from_srt("1\n00:00:01,000 --> 00:00:02,000\nHello\n");
    assert!(result.is_ok());
}
