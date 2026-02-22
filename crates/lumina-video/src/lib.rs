//! lumina-video: Cross-platform video playback for egui with hardware acceleration
//!
//! This crate provides hardware-accelerated video playback for egui applications
//! using **native platform media frameworks** - no FFmpeg required by default.
//!
//! # Native Platform Support
//!
//! Each platform uses its native media stack for optimal performance:
//!
//! | Platform | Native Framework | Hardware Acceleration |
//! |----------|------------------|----------------------|
//! | macOS | AVFoundation + VideoToolbox | Apple Silicon / Intel QuickSync |
//! | Linux | GStreamer | VA-API, NVDEC (varies by GPU/drivers) |
//! | Windows | Media Foundation | DXVA2, D3D11VA |
//! | Android | MediaCodec | Device hardware codecs |
//!
//! # Example
//!
//! ```ignore
//! use lumina_video::{VideoPlayer, VideoPlayerExt};
//!
//! // Store the player in your app state (created once):
//! let mut player = VideoPlayer::new("https://example.com/video.mp4");
//!
//! // In your egui update() function, render with controls:
//! let response = player.show(ui, available_size);
//!
//! // Or use the extension trait on Ui:
//! let response = ui.video_player(&mut player, available_size);
//! ```
//!
//! # Feature Flags
//!
//! Native decoders (recommended - one per platform):
//! - `macos-native-video`: AVFoundation + VideoToolbox on macOS
//! - `linux-gstreamer-video`: GStreamer + VA-API on Linux
//!   - Requires runtime packages: `gstreamer1.0-plugins-good`, `gstreamer1.0-libav`,
//!     `gstreamer1.0-vaapi`, and platform-specific VA-API drivers
//! - `windows-native-video`: Media Foundation + DXVA2 on Windows
//!
//! Optional fallback:
//! - `ffmpeg`: FFmpeg decoder (cross-platform fallback, requires FFmpeg installation)

#![deny(clippy::disallowed_methods)]

pub mod media;

// Vendored runtime support for Linux (bundles GStreamer libraries)
#[cfg(all(target_os = "linux", feature = "vendored-runtime"))]
pub use lumina_video_core::vendored_runtime;

// Re-export main video types for convenience
pub use media::{
    AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync, CpuFrame,
    DecodedFrame, HwAccelType, PixelFormat, Plane, VideoControls, VideoControlsConfig,
    VideoControlsResponse, VideoDecoderBackend, VideoError, VideoFrame, VideoMetadata,
    VideoPlayerHandle, VideoState,
};

// VideoPlayer and SyncMetrics are only available on native platforms (not wasm32)
#[cfg(not(target_arch = "wasm32"))]
pub use media::{
    SyncMetrics, SyncMetricsSnapshot, VideoPlayer, VideoPlayerExt, VideoPlayerResponse,
    SYNC_DRIFT_THRESHOLD_MS,
};

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use media::{FfmpegDecoder, FfmpegDecoderBuilder, HwAccelConfig};

#[cfg(target_os = "android")]
pub use media::android_video::{
    android_zero_copy_snapshot, AndroidZeroCopySnapshot, ZeroCopyStatus,
};
#[cfg(target_os = "android")]
pub use media::AndroidVideoDecoder;

#[cfg(target_arch = "wasm32")]
pub use media::{HlsBufferInfo, HlsQualityLevel, WebVideoPlayer};
