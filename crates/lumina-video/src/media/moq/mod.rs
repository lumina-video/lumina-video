//! Media over QUIC (MoQ) transport layer.
//!
//! This module provides MoQ support for lumina-video, enabling live streaming
//! via the QUIC protocol. MoQ acts as a transport layer that feeds encoded
//! media into FFmpeg for decoding, reusing existing frame queue and A/V sync
//! infrastructure.
//!
//! # Architecture
//!
//! ```text
//! moq:// URL → MoqTransport (QUIC/MoQ protocol)
//!            → MoqMediaSource (reassemble objects)
//!            → FFmpeg (custom AVIO callbacks)
//!            → FrameQueue → Rendering
//! ```
//!
//! # URL Format
//!
//! - `moq://host:port/namespace/track` - Insecure (for testing)
//! - `moqs://host:port/namespace/track` - TLS required (production)
//!
//! # Feature Flag
//!
//! Enable with the `moq` feature in Cargo.toml:
//!
//! ```toml
//! lumina-video = { version = "...", features = ["moq"] }
//! ```
//!
//! # Live Streaming Considerations
//!
//! - `duration()` returns `None` for live streams
//! - `seek()` returns an error (live streams don't support seeking)
//! - Frame queue never sets EOS until stream ends
//! - Latency metrics are tracked in SyncMetrics

pub mod catalog;
pub mod error;
pub mod media_source;
pub mod subscriber;
pub mod transport;
pub mod url;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "android"))]
pub(crate) mod worker;

// Re-export main types
pub use catalog::{AudioCodec, AudioTrackInfo, MoqCatalog, VideoCodec, VideoTrackInfo};
pub use error::MoqError;
pub use media_source::{MoqMediaSource, MoqMediaSourceWriter, ReorderBufferStats};
pub use subscriber::{MoqFrame, MoqSubscriptionManager, MoqTrackSubscriber, SubscriptionState};
pub use transport::{MoqTransport, MoqTransportConfig, TransportState};
pub use url::MoqUrl;
