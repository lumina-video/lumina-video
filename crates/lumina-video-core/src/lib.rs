//! lumina-video-core: Core video decode and zero-copy pipeline.
//!
//! This crate provides the egui-free foundation for hardware-accelerated video
//! playback. It contains:
//!
//! - Core types: [`video`], [`audio`], [`subtitles`]
//! - Platform decoders: macOS (AVFoundation), Linux (GStreamer), Android (ExoPlayer), Windows (Media Foundation)
//! - Zero-copy GPU import: [`zero_copy`]
//! - Threading primitives: [`frame_queue`], [`triple_buffer`], [`sync_metrics`]
//! - Network utilities: [`network`]
//!
//! This crate has **zero egui dependency**. It is consumed by:
//! - `lumina-video` (egui integration layer)
//! - `lumina-video-ios` (C FFI for iOS/Swift)
