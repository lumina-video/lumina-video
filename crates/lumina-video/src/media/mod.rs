//! Video and audio playback modules for lumina-video.
//!
//! This module provides cross-platform hardware-accelerated video playback:
//!
//! - [`VideoPlayer`] - Main video player widget for egui
//! - [`VideoControls`] - Play/pause, seek, volume controls
//! - [`video`] - Core video types and decoder traits
//! - [`audio`] - Audio playback and synchronization
//! - [`subtitles`] - Subtitle parsing and rendering (SRT, VTT)
//!
//! # Platform Support
//!
//! | Platform | Decoder | Hardware Acceleration |
//! |----------|---------|----------------------|
//! | macOS | AVFoundation | VideoToolbox |
//! | Linux | GStreamer | VA-API, NVDEC |
//! | Windows | Media Foundation | DXVA2, D3D11VA |
//! | Android | MediaCodec | Hardware codecs |
//!
//! # Known Issues
//!
//! ## macOS objc2 Version Coexistence
//!
//! The native macOS decoder uses `objc2 0.6.x` for AVFoundation bindings,
//! while `winit` (used by egui) uses `objc2 0.5.x`. These versions coexist
//! safely because they bind to different Objective-C classes:
//!
//! - `objc2 0.5.x`: winit's window management classes
//! - `objc2 0.6.x`: AVFoundation media classes
//!
//! This is a known working configuration and works with upstream egui/eframe.

#[cfg(target_os = "android")]
pub mod android_video;
#[cfg(target_os = "android")]
pub mod android_vulkan;
pub mod audio;
#[cfg(target_os = "macos")]
pub mod audio_decoder;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
pub(crate) mod audio_ring_buffer;
#[cfg(not(target_arch = "wasm32"))]
pub mod frame_queue;
#[cfg(target_os = "linux")]
pub mod linux_video;
#[cfg(target_os = "linux")]
pub mod linux_video_gst;
#[cfg(target_os = "macos")]
pub mod macos_video;
#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub mod moq;
#[cfg(all(
    feature = "moq",
    any(target_os = "macos", target_os = "linux", target_os = "android")
))]
pub(crate) mod moq_audio;
#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub mod moq_decoder;
#[cfg(target_os = "android")]
pub mod ndk_image_reader;
#[cfg(not(target_arch = "wasm32"))]
pub mod network;
#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub mod nostr_discovery;
pub mod subtitles;
#[cfg(not(target_arch = "wasm32"))]
pub mod sync_metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod triple_buffer;
pub mod video;
pub mod video_controls;
#[cfg(target_os = "macos")]
pub mod video_decoder;
#[cfg(not(target_arch = "wasm32"))]
pub mod video_player;
pub mod video_texture;
#[cfg(target_arch = "wasm32")]
pub mod web_moq_decoder;
#[cfg(target_arch = "wasm32")]
pub mod web_video;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub mod windows_audio;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub mod windows_video;
// Zero-copy module available on all platforms with native decoders
#[cfg(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", feature = "windows-native-video")
))]
pub mod zero_copy;

// Re-export main types
pub use audio::{AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync};
pub use subtitles::{SubtitleCue, SubtitleError, SubtitleStyle, SubtitleTrack};
pub use video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata, VideoPlayerHandle, VideoState,
};
pub use video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
#[cfg(target_os = "macos")]
pub use video_decoder::{FfmpegDecoder, FfmpegDecoderBuilder, HwAccelConfig};
#[cfg(not(target_arch = "wasm32"))]
pub use video_player::{VideoPlayer, VideoPlayerExt, VideoPlayerResponse};

#[cfg(target_os = "android")]
pub use android_video::{AndroidVideoDecoder, AndroidZeroCopySnapshot, ZeroCopyStatus};

#[cfg(target_os = "macos")]
pub use macos_video::{MacOSVideoDecoder, MacOSZeroCopyStatsSnapshot};

#[cfg(target_os = "linux")]
pub use linux_video::{LinuxZeroCopyMetricsSnapshot, ZeroCopyGStreamerDecoder};
#[cfg(target_os = "linux")]
pub use linux_video_gst::GStreamerDecoder;

#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub use windows_video::WindowsVideoDecoder;

#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub use moq::{MoqError, MoqUrl};

#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub use moq_decoder::{
    MoqAudioStatus, MoqDecoder, MoqDecoderConfig, MoqDecoderState, MoqFrameStatsSnapshot,
    MoqStatsHandle, MoqStatsSnapshot,
};

#[cfg(all(target_os = "android", feature = "moq"))]
pub use moq_decoder::MoqAndroidDecoder;

#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub use nostr_discovery::{DiscoveryEvent, MoqStream, NostrDiscovery, StreamStatus};

#[cfg(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", feature = "windows-native-video")
))]
pub use zero_copy::{ZeroCopyError, ZeroCopyStats};

#[cfg(not(target_arch = "wasm32"))]
pub use sync_metrics::{SyncMetrics, SyncMetricsSnapshot, SYNC_DRIFT_THRESHOLD_MS};

#[cfg(target_arch = "wasm32")]
pub use web_video::{
    HlsBufferInfo, HlsQualityLevel, WebVideoPlayer, WebVideoPlayerResponse, WebVideoRenderCallback,
    WebVideoRenderResources, WebVideoTexture,
};

#[cfg(target_arch = "wasm32")]
pub use web_moq_decoder::{
    codec_strings, WebMoqDecoder, WebMoqDecoderState, WebMoqFrameInfo, WebMoqTexture, WebMoqUrl,
};

/// Maximum texture size wgpu can handle without panicking.
pub const MAX_SIZE_WGPU: usize = 8192;
