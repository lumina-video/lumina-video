//! lumina-video-core: Core video decode and zero-copy pipeline.
//!
//! This crate provides the egui-free foundation for hardware-accelerated video
//! playback. It contains:
//!
//! - Core types: [`video`], [`audio`], [`subtitles`]
//! - Platform decoders: macOS (AVFoundation), Linux (GStreamer), Android (ExoPlayer), Windows (Media Foundation)
//! - Zero-copy GPU import: `zero_copy`
//! - Threading primitives: [`frame_queue`], [`triple_buffer`], [`sync_metrics`]
//! - Network utilities: [`network`]
//!
//! This crate has **zero egui dependency**. It is consumed by:
//! - `lumina-video` (egui integration layer)
//! - `lumina-video-ios` (C FFI for iOS/Swift)

// === Universal modules (compile on all targets including wasm32) ===

pub mod video;
pub mod audio;
pub mod subtitles;

/// Internal bridge API — public only for cross-crate re-export by lumina-video.
/// NOT semver-stable. Do not depend on this module directly from external crates.
/// May change or be removed in any minor version.
#[doc(hidden)]
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
pub mod audio_ring_buffer;

// === Native-only modules (not available on wasm32) ===

#[cfg(not(target_arch = "wasm32"))]
pub mod frame_queue;
#[cfg(not(target_arch = "wasm32"))]
pub mod sync_metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod triple_buffer;
#[cfg(not(target_arch = "wasm32"))]
pub mod network;
