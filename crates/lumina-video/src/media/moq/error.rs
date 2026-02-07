//! MoQ-specific error types.
//!
//! This module defines errors that can occur during MoQ transport operations,
//! including connection errors, protocol errors, and stream errors.

use std::fmt;

/// Errors that can occur during MoQ operations.
#[derive(Debug)]
pub enum MoqError {
    /// Failed to parse moq:// URL
    InvalidUrl(String),
    /// QUIC connection failed
    ConnectionFailed(String),
    /// TLS/certificate error
    TlsError(String),
    /// MoQ session establishment failed
    SessionError(String),
    /// Catalog fetch or parse error
    CatalogError(String),
    /// Track subscription failed
    SubscriptionError(String),
    /// Object receive error
    ObjectError(String),
    /// Stream closed unexpectedly
    StreamClosed(String),
    /// Timeout waiting for data
    Timeout(String),
    /// Internal channel error
    ChannelError(String),
    /// FFmpeg AVIO bridge error
    AvioBridgeError(String),
    /// Live stream does not support this operation
    LiveStreamUnsupported(String),
}

impl fmt::Display for MoqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoqError::InvalidUrl(msg) => write!(f, "Invalid MoQ URL: {msg}"),
            MoqError::ConnectionFailed(msg) => write!(f, "MoQ connection failed: {msg}"),
            MoqError::TlsError(msg) => write!(f, "MoQ TLS error: {msg}"),
            MoqError::SessionError(msg) => write!(f, "MoQ session error: {msg}"),
            MoqError::CatalogError(msg) => write!(f, "MoQ catalog error: {msg}"),
            MoqError::SubscriptionError(msg) => write!(f, "MoQ subscription error: {msg}"),
            MoqError::ObjectError(msg) => write!(f, "MoQ object error: {msg}"),
            MoqError::StreamClosed(msg) => write!(f, "MoQ stream closed: {msg}"),
            MoqError::Timeout(msg) => write!(f, "MoQ timeout: {msg}"),
            MoqError::ChannelError(msg) => write!(f, "MoQ channel error: {msg}"),
            MoqError::AvioBridgeError(msg) => write!(f, "MoQ AVIO bridge error: {msg}"),
            MoqError::LiveStreamUnsupported(msg) => {
                write!(f, "Operation not supported on live stream: {msg}")
            }
        }
    }
}

impl std::error::Error for MoqError {}

impl From<MoqError> for super::super::video::VideoError {
    fn from(err: MoqError) -> Self {
        match err {
            MoqError::InvalidUrl(msg) => super::super::video::VideoError::OpenFailed(msg),
            MoqError::ConnectionFailed(msg) | MoqError::TlsError(msg) => {
                super::super::video::VideoError::Network(msg)
            }
            MoqError::SessionError(msg)
            | MoqError::CatalogError(msg)
            | MoqError::SubscriptionError(msg) => super::super::video::VideoError::DecoderInit(msg),
            MoqError::ObjectError(msg)
            | MoqError::StreamClosed(msg)
            | MoqError::AvioBridgeError(msg) => super::super::video::VideoError::DecodeFailed(msg),
            MoqError::Timeout(msg) | MoqError::ChannelError(msg) => {
                super::super::video::VideoError::Network(msg)
            }
            MoqError::LiveStreamUnsupported(msg) => {
                super::super::video::VideoError::SeekFailed(msg)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MoqError::InvalidUrl("missing host".to_string());
        assert!(err.to_string().contains("Invalid MoQ URL"));
        assert!(err.to_string().contains("missing host"));
    }

    #[test]
    fn test_error_conversion() {
        use super::super::super::video::VideoError;

        let moq_err = MoqError::ConnectionFailed("timeout".to_string());
        let video_err: VideoError = moq_err.into();
        match video_err {
            VideoError::Network(msg) => assert!(msg.contains("timeout")),
            _ => panic!("Expected Network error"),
        }
    }
}
