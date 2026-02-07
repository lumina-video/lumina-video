//! MoQ catalog parsing and track metadata.
//!
//! This module handles parsing MoQ catalogs to discover available tracks
//! and their metadata (codec, dimensions, bitrate, etc.).

use super::error::MoqError;

/// Track type (video or audio).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    Video,
    Audio,
}

/// Video codec information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
    Vp9,
    Unknown(String),
}

impl VideoCodec {
    /// Parses a codec string into a VideoCodec.
    pub fn parse(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("avc") || lower.contains("h264") || lower.contains("h.264") {
            VideoCodec::H264
        } else if lower.contains("hevc")
            || lower.contains("hvc1")
            || lower.contains("h265")
            || lower.contains("h.265")
        {
            VideoCodec::H265
        } else if lower.contains("av1") || lower.contains("av01") {
            VideoCodec::Av1
        } else if lower.contains("vp9") || lower.contains("vp09") {
            VideoCodec::Vp9
        } else {
            VideoCodec::Unknown(s.to_string())
        }
    }
}

/// Audio codec information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioCodec {
    Aac,
    Opus,
    Mp3,
    Unknown(String),
}

impl AudioCodec {
    /// Parses a codec string into an AudioCodec.
    pub fn parse(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("aac") || lower.contains("mp4a") {
            AudioCodec::Aac
        } else if lower.contains("opus") {
            AudioCodec::Opus
        } else if lower.contains("mp3") || lower.contains("mp4a.40.34") {
            AudioCodec::Mp3
        } else {
            AudioCodec::Unknown(s.to_string())
        }
    }
}

/// Metadata for a video track.
#[derive(Debug, Clone)]
pub struct VideoTrackInfo {
    /// Track name/path
    pub name: String,
    /// Video codec
    pub codec: VideoCodec,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Frame rate (fps)
    pub frame_rate: f32,
    /// Bitrate in bits per second (if known)
    pub bitrate: Option<u64>,
    /// Codec-specific initialization data (e.g., SPS/PPS for H.264)
    pub init_data: Option<Vec<u8>>,
}

/// Metadata for an audio track.
#[derive(Debug, Clone)]
pub struct AudioTrackInfo {
    /// Track name/path
    pub name: String,
    /// Audio codec
    pub codec: AudioCodec,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Bitrate in bits per second (if known)
    pub bitrate: Option<u64>,
    /// Codec-specific initialization data (e.g., AudioSpecificConfig for AAC)
    pub init_data: Option<Vec<u8>>,
}

/// Parsed MoQ catalog containing available tracks.
#[derive(Debug, Clone, Default)]
pub struct MoqCatalog {
    /// Video tracks available in the broadcast
    pub video_tracks: Vec<VideoTrackInfo>,
    /// Audio tracks available in the broadcast
    pub audio_tracks: Vec<AudioTrackInfo>,
    /// Raw catalog data (for debugging)
    pub raw: Option<String>,
}

impl MoqCatalog {
    /// Creates an empty catalog.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the primary video track (first one, if any).
    pub fn primary_video(&self) -> Option<&VideoTrackInfo> {
        self.video_tracks.first()
    }

    /// Returns the primary audio track (first one, if any).
    pub fn primary_audio(&self) -> Option<&AudioTrackInfo> {
        self.audio_tracks.first()
    }

    /// Returns true if the catalog has at least one video track.
    pub fn has_video(&self) -> bool {
        !self.video_tracks.is_empty()
    }

    /// Returns true if the catalog has at least one audio track.
    pub fn has_audio(&self) -> bool {
        !self.audio_tracks.is_empty()
    }

    /// Parses a catalog from JSON data.
    ///
    /// MoQ catalogs typically follow a JSON format describing available tracks.
    /// This parser handles common catalog formats.
    pub fn parse_json(data: &[u8]) -> Result<Self, MoqError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| MoqError::CatalogError(format!("Invalid UTF-8 in catalog: {e}")))?;

        let mut catalog = MoqCatalog {
            raw: Some(text.to_string()),
            ..Default::default()
        };

        // Try to parse as JSON
        // Common catalog format:
        // {
        //   "tracks": [
        //     { "name": "video", "kind": "video", "codec": "avc1.64001f", "width": 1920, "height": 1080, "framerate": 30 },
        //     { "name": "audio", "kind": "audio", "codec": "mp4a.40.2", "samplerate": 48000, "channels": 2 }
        //   ]
        // }

        // Simple JSON parsing without serde dependency
        // In production, consider using serde_json
        if let Some(tracks_start) = text.find("\"tracks\"") {
            let after_tracks = &text[tracks_start..];
            if let Some(array_start) = after_tracks.find('[') {
                if let Some(array_end) = after_tracks.find(']') {
                    // Validate array bounds to avoid panic on malformed JSON
                    if array_end > array_start + 1 {
                        let array_content = &after_tracks[array_start + 1..array_end];
                        catalog.parse_tracks_array(array_content)?;
                    }
                    // Empty array (array_end == array_start + 1) is valid, just no tracks
                }
            }
        }

        Ok(catalog)
    }

    /// Parses individual track entries from the tracks array.
    fn parse_tracks_array(&mut self, content: &str) -> Result<(), MoqError> {
        // Split by track objects (simple heuristic)
        let mut depth = 0;
        let mut track_start = None;

        for (i, c) in content.char_indices() {
            match c {
                '{' => {
                    if depth == 0 {
                        track_start = Some(i);
                    }
                    depth += 1;
                }
                '}' => {
                    // Use saturating_sub to handle malformed JSON with extra '}'
                    if depth == 0 {
                        // Extra '}' in malformed JSON - skip it
                        continue;
                    }
                    depth -= 1;
                    if depth == 0 {
                        if let Some(start) = track_start {
                            let track_json = &content[start..=i];
                            self.parse_track_object(track_json)?;
                            track_start = None;
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Parses a single track object from JSON.
    fn parse_track_object(&mut self, json: &str) -> Result<(), MoqError> {
        // Extract fields using simple string matching
        let name = Self::extract_string_field(json, "name").unwrap_or_default();
        let kind = Self::extract_string_field(json, "kind").unwrap_or_default();
        let codec = Self::extract_string_field(json, "codec").unwrap_or_default();

        match kind.as_str() {
            "video" => {
                let width = Self::extract_number_field(json, "width").unwrap_or(0.0) as u32;
                let height = Self::extract_number_field(json, "height").unwrap_or(0.0) as u32;
                let frame_rate =
                    Self::extract_number_field(json, "framerate").unwrap_or(30.0) as f32;
                let bitrate = Self::extract_number_field(json, "bitrate").map(|b| b as u64);

                self.video_tracks.push(VideoTrackInfo {
                    name,
                    codec: VideoCodec::parse(&codec),
                    width,
                    height,
                    frame_rate,
                    bitrate,
                    init_data: None,
                });
            }
            "audio" => {
                let sample_rate =
                    Self::extract_number_field(json, "samplerate").unwrap_or(48000.0) as u32;
                let channels = Self::extract_number_field(json, "channels").unwrap_or(2.0) as u8;
                let bitrate = Self::extract_number_field(json, "bitrate").map(|b| b as u64);

                self.audio_tracks.push(AudioTrackInfo {
                    name,
                    codec: AudioCodec::parse(&codec),
                    sample_rate,
                    channels,
                    bitrate,
                    init_data: None,
                });
            }
            _ => {
                // Unknown track type, skip
            }
        }

        Ok(())
    }

    /// Extracts a string field from JSON.
    fn extract_string_field(json: &str, field: &str) -> Option<String> {
        let pattern = format!("\"{field}\"");
        let field_start = json.find(&pattern)?;
        let after_field = &json[field_start + pattern.len()..];

        // Find the colon and opening quote
        let colon_pos = after_field.find(':')?;
        let after_colon = &after_field[colon_pos + 1..];
        let quote_start = after_colon.find('"')?;
        let value_start = &after_colon[quote_start + 1..];

        // Find the closing quote (handle escaped quotes)
        let mut end_pos = 0;
        let mut chars = value_start.char_indices();
        while let Some((i, c)) = chars.next() {
            if c == '"' {
                end_pos = i;
                break;
            } else if c == '\\' {
                // Skip escaped character
                chars.next();
            }
        }

        Some(value_start[..end_pos].to_string())
    }

    /// Extracts a number field from JSON.
    fn extract_number_field(json: &str, field: &str) -> Option<f64> {
        let pattern = format!("\"{field}\"");
        let field_start = json.find(&pattern)?;
        let after_field = &json[field_start + pattern.len()..];

        // Find the colon
        let colon_pos = after_field.find(':')?;
        let after_colon = &after_field[colon_pos + 1..].trim_start();

        // Extract the number (until comma, brace, or whitespace)
        let end_pos = after_colon
            .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace())
            .unwrap_or(after_colon.len());

        after_colon[..end_pos].trim().parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_parsing() {
        assert_eq!(VideoCodec::parse("avc1.64001f"), VideoCodec::H264);
        assert_eq!(VideoCodec::parse("h264"), VideoCodec::H264);
        assert_eq!(VideoCodec::parse("hvc1.1.6.L93.B0"), VideoCodec::H265);
        assert_eq!(VideoCodec::parse("av01.0.08M.08"), VideoCodec::Av1);
        assert_eq!(VideoCodec::parse("vp09.00.10.08"), VideoCodec::Vp9);
    }

    #[test]
    fn test_audio_codec_parsing() {
        assert_eq!(AudioCodec::parse("mp4a.40.2"), AudioCodec::Aac);
        assert_eq!(AudioCodec::parse("opus"), AudioCodec::Opus);
        assert_eq!(AudioCodec::parse("mp3"), AudioCodec::Mp3);
    }

    #[test]
    fn test_catalog_json_parsing() {
        let json = r#"{
            "tracks": [
                { "name": "video", "kind": "video", "codec": "avc1.64001f", "width": 1920, "height": 1080, "framerate": 30 },
                { "name": "audio", "kind": "audio", "codec": "mp4a.40.2", "samplerate": 48000, "channels": 2 }
            ]
        }"#;

        let catalog = MoqCatalog::parse_json(json.as_bytes()).unwrap();

        assert!(catalog.has_video());
        assert!(catalog.has_audio());

        let video = catalog.primary_video().unwrap();
        assert_eq!(video.name, "video");
        assert_eq!(video.codec, VideoCodec::H264);
        assert_eq!(video.width, 1920);
        assert_eq!(video.height, 1080);

        let audio = catalog.primary_audio().unwrap();
        assert_eq!(audio.name, "audio");
        assert_eq!(audio.codec, AudioCodec::Aac);
        assert_eq!(audio.sample_rate, 48000);
        assert_eq!(audio.channels, 2);
    }

    #[test]
    fn test_empty_catalog() {
        let catalog = MoqCatalog::new();
        assert!(!catalog.has_video());
        assert!(!catalog.has_audio());
        assert!(catalog.primary_video().is_none());
        assert!(catalog.primary_audio().is_none());
    }
}
