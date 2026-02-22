//! Error types for the FFI boundary.

use lumina_video_core::video::VideoError;

/// FFI error codes matching `include/LuminaVideo.h`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuminaError {
    Ok = 0,
    NullPtr = 1,
    InvalidUrl = 2,
    InitFailed = 3,
    Decode = 4,
    Internal = 5,
    InvalidArgument = 6,
}

impl From<VideoError> for LuminaError {
    fn from(e: VideoError) -> Self {
        match e {
            VideoError::DecoderInit(_) => LuminaError::InitFailed,
            VideoError::DecodeFailed(_) => LuminaError::Decode,
            VideoError::OpenFailed(_) | VideoError::Network(_) => LuminaError::InitFailed,
            _ => LuminaError::Internal,
        }
    }
}

impl LuminaError {
    /// Convert to the raw i32 for FFI return.
    pub fn as_raw(self) -> i32 {
        self as i32
    }
}
