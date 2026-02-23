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

// =============================================================================
// Re-export moved modules from lumina-video-core
// =============================================================================
// These re-exports preserve super:: paths for MoQ and other local consumers.

pub use lumina_video_core::audio;
pub use lumina_video_core::subtitles;
pub use lumina_video_core::video;

// audio_ring_buffer: pub in core for cross-crate access. NOT public API — do not stabilize.
// Required by moq_audio.rs (super::audio_ring_buffer::RingBufferConfig).
// Unused without the "moq" feature, but the cfg must match core's module gate.
#[allow(unused_imports)]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
pub(crate) use lumina_video_core::audio_ring_buffer;

#[cfg(not(target_arch = "wasm32"))]
pub use lumina_video_core::frame_queue;
#[cfg(not(target_arch = "wasm32"))]
pub use lumina_video_core::network;
#[cfg(not(target_arch = "wasm32"))]
pub use lumina_video_core::player;
#[cfg(not(target_arch = "wasm32"))]
pub use lumina_video_core::sync_metrics;
#[cfg(not(target_arch = "wasm32"))]
pub use lumina_video_core::triple_buffer;

// Platform-specific re-exports
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use lumina_video_core::audio_decoder;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use lumina_video_core::macos_video;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use lumina_video_core::video_decoder;

#[cfg(target_os = "linux")]
pub use lumina_video_core::linux_video;
#[cfg(target_os = "linux")]
pub use lumina_video_core::linux_video_gst;

#[cfg(target_os = "android")]
pub use lumina_video_core::android_video;
#[cfg(target_os = "android")]
pub use lumina_video_core::android_vulkan;
#[cfg(all(target_os = "android", feature = "android-zero-copy"))]
pub use lumina_video_core::ndk_image_reader;

#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub use lumina_video_core::windows_audio;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
pub use lumina_video_core::windows_video;

// Zero-copy module
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", feature = "windows-native-video")
))]
pub use lumina_video_core::zero_copy;

// =============================================================================
// Local modules (egui layer — stay in lumina-video)
// =============================================================================

pub mod video_controls;
#[cfg(not(target_arch = "wasm32"))]
pub mod video_player;
pub mod video_texture;

// MoQ modules (stay in lumina-video, import from core via re-exports)
#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub mod moq;
#[cfg(all(
    feature = "moq",
    any(target_os = "macos", target_os = "linux", target_os = "android")
))]
pub(crate) mod moq_audio;
#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub mod moq_decoder;
#[cfg(all(not(target_arch = "wasm32"), feature = "moq"))]
pub mod nostr_discovery;

// Web/WASM modules
#[cfg(target_arch = "wasm32")]
pub mod web_moq_decoder;
#[cfg(target_arch = "wasm32")]
pub mod web_video;

// =============================================================================
// Type re-exports
// =============================================================================

// Re-export main types
pub use audio::{AudioConfig, AudioHandle, AudioPlayer, AudioSamples, AudioState, AudioSync};
pub use subtitles::{SubtitleCue, SubtitleError, SubtitleStyle, SubtitleTrack};
pub use video::{
    CpuFrame, DecodedFrame, HwAccelType, PixelFormat, Plane, VideoDecoderBackend, VideoError,
    VideoFrame, VideoMetadata, VideoPlayerHandle, VideoState,
};
pub use video_controls::{VideoControls, VideoControlsConfig, VideoControlsResponse};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use video_decoder::{FfmpegDecoder, FfmpegDecoderBuilder, HwAccelConfig};
#[cfg(not(target_arch = "wasm32"))]
pub use video_player::{VideoPlayer, VideoPlayerExt, VideoPlayerResponse};

#[cfg(target_os = "android")]
pub use android_video::{AndroidVideoDecoder, AndroidZeroCopySnapshot, ZeroCopyStatus};

#[cfg(any(target_os = "macos", target_os = "ios"))]
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
    target_os = "ios",
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
    codec_strings, WebMoqAudioRendition, WebMoqCatalog, WebMoqDecoder, WebMoqDecoderState,
    WebMoqFrameInfo, WebMoqSession, WebMoqSessionState, WebMoqStats, WebMoqTexture, WebMoqUrl,
    WebMoqVideoRendition,
};

/// Maximum texture size wgpu can handle without panicking.
pub const MAX_SIZE_WGPU: usize = 8192;
