//! Subtitle parsing and rendering for lumina-video.
//!
//! Supports SRT and VTT subtitle formats with timing-based cue lookup.

use std::time::Duration;

/// Error type for subtitle operations
#[derive(Debug, Clone)]
pub enum SubtitleError {
    /// Failed to parse timestamp
    InvalidTimestamp(String),
    /// Failed to parse subtitle file format
    ParseError(String),
    /// Empty subtitle file
    EmptyFile,
}

impl std::fmt::Display for SubtitleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubtitleError::InvalidTimestamp(s) => write!(f, "Invalid timestamp: {s}"),
            SubtitleError::ParseError(s) => write!(f, "Parse error: {s}"),
            SubtitleError::EmptyFile => write!(f, "Empty subtitle file"),
        }
    }
}

impl std::error::Error for SubtitleError {}

/// A single subtitle cue with timing and text
#[derive(Debug, Clone)]
pub struct SubtitleCue {
    /// Start time of the cue
    pub start: Duration,
    /// End time of the cue
    pub end: Duration,
    /// Text content (may contain newlines for multi-line subtitles)
    pub text: String,
}

/// Style configuration for subtitle rendering
#[derive(Debug, Clone)]
pub struct SubtitleStyle {
    /// Font size in points
    pub font_size: f32,
    /// Text color (RGBA)
    pub text_color: [u8; 4],
    /// Background color (RGBA)
    pub background_color: [u8; 4],
    /// Vertical position from bottom (in pixels)
    pub bottom_margin: f32,
    /// Whether to show background behind text
    pub show_background: bool,
}

impl Default for SubtitleStyle {
    fn default() -> Self {
        Self {
            font_size: 18.0,
            text_color: [255, 255, 255, 255],
            background_color: [0, 0, 0, 180],
            bottom_margin: 60.0,
            show_background: true,
        }
    }
}

/// A loaded subtitle track
#[derive(Debug, Clone)]
pub struct SubtitleTrack {
    /// All cues in the track, sorted by start time
    pub cues: Vec<SubtitleCue>,
    /// Optional language identifier (e.g., "en", "es")
    pub language: Option<String>,
    /// Optional track title
    pub title: Option<String>,
}

impl SubtitleTrack {
    /// Create an empty subtitle track
    pub fn new() -> Self {
        Self {
            cues: Vec::new(),
            language: None,
            title: None,
        }
    }

    /// Parse subtitle track from SRT format
    ///
    /// SRT format:
    /// ```text
    /// 1
    /// 00:00:01,000 --> 00:00:04,000
    /// First subtitle line
    ///
    /// 2
    /// 00:00:05,000 --> 00:00:08,000
    /// Second subtitle line
    /// With multiple lines
    /// ```
    pub fn from_srt(content: &str) -> Result<Self, SubtitleError> {
        let mut cues = Vec::new();

        // Normalize CRLF to LF for consistent parsing
        let content = content.replace("\r\n", "\n");

        // Split on blank lines (one or more newlines)
        for block in content.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }

            let lines: Vec<&str> = block.lines().collect();
            if lines.len() < 2 {
                continue; // Skip malformed blocks
            }

            // Find the timing line by pattern (contains "-->")
            // This handles cases where the index line might be missing
            let timing_idx = lines.iter().position(|line| line.contains("-->"));
            let Some(timing_idx) = timing_idx else {
                continue; // No timing line found, skip block
            };

            let (start, end) = parse_srt_timing(lines[timing_idx])?;

            // Text lines follow the timing line
            let text = lines[timing_idx + 1..].join("\n");
            if !text.is_empty() {
                cues.push(SubtitleCue { start, end, text });
            }
        }

        if cues.is_empty() {
            return Err(SubtitleError::EmptyFile);
        }

        // Sort by start time
        cues.sort_by_key(|c| c.start);

        Ok(Self {
            cues,
            language: None,
            title: None,
        })
    }

    /// Parse subtitle track from WebVTT format
    ///
    /// VTT format:
    /// ```text
    /// WEBVTT
    ///
    /// 00:00:01.000 --> 00:00:04.000
    /// First subtitle line
    ///
    /// 00:00:05.000 --> 00:00:08.000
    /// Second subtitle line
    /// ```
    pub fn from_vtt(content: &str) -> Result<Self, SubtitleError> {
        let mut cues = Vec::new();

        // Strip UTF-8 BOM if present
        let content = content.strip_prefix('\u{FEFF}').unwrap_or(content);

        let lines: Vec<&str> = content.lines().collect();

        // Check for WEBVTT header (may have additional text after "WEBVTT")
        if lines.is_empty() || !lines[0].trim().starts_with("WEBVTT") {
            return Err(SubtitleError::ParseError(
                "Missing WEBVTT header".to_string(),
            ));
        }

        let mut i = 1;
        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines
            if line.is_empty() {
                i += 1;
                continue;
            }

            // Skip NOTE blocks (may span multiple lines until empty line)
            if line.starts_with("NOTE") {
                i += 1;
                // Skip until empty line or end of file
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }

            // Skip STYLE blocks (span until empty line)
            if line.starts_with("STYLE") {
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }

            // Check if this is a timing line
            if line.contains("-->") {
                let (start, end) = parse_vtt_timing(line)?;

                // Collect text lines until empty line or end
                let mut text_lines = Vec::new();
                i += 1;
                while i < lines.len() && !lines[i].trim().is_empty() {
                    text_lines.push(lines[i].trim());
                    i += 1;
                }

                let text = text_lines.join("\n");
                if !text.is_empty() {
                    cues.push(SubtitleCue { start, end, text });
                }
            } else {
                // Skip cue identifiers (optional in VTT)
                i += 1;
            }
        }

        if cues.is_empty() {
            return Err(SubtitleError::EmptyFile);
        }

        // Sort by start time
        cues.sort_by_key(|c| c.start);

        Ok(Self {
            cues,
            language: None,
            title: None,
        })
    }

    /// Get the active cue at the given position, if any.
    ///
    /// Uses linear search since cues may overlap and we need the first match.
    /// For typical subtitle files (<1000 cues), this is efficient enough.
    pub fn get_cue_at(&self, position: Duration) -> Option<&SubtitleCue> {
        self.cues
            .iter()
            .find(|cue| position >= cue.start && position < cue.end)
    }

    /// Get all cues that overlap with the given time range
    pub fn get_cues_in_range(&self, start: Duration, end: Duration) -> Vec<&SubtitleCue> {
        self.cues
            .iter()
            .filter(|cue| cue.start < end && cue.end > start)
            .collect()
    }

    /// Total number of cues in the track
    pub fn len(&self) -> usize {
        self.cues.len()
    }

    /// Check if the track has no cues
    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }
}

impl Default for SubtitleTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse SRT timing line: "00:00:01,000 --> 00:00:04,000"
fn parse_srt_timing(line: &str) -> Result<(Duration, Duration), SubtitleError> {
    let (start_str, end_str) = line
        .split_once("-->")
        .ok_or_else(|| SubtitleError::InvalidTimestamp(line.to_string()))?;

    let start = parse_srt_timestamp(start_str.trim())?;
    let end = parse_srt_timestamp(end_str.trim())?;
    Ok((start, end))
}

/// Parse SRT timestamp: "00:00:01,000" (comma for milliseconds)
fn parse_srt_timestamp(s: &str) -> Result<Duration, SubtitleError> {
    // Handle both comma and period as decimal separator
    let s = s.replace(',', ".");
    parse_timestamp_common(&s)
}

/// Parse VTT timing line: "00:00:01.000 --> 00:00:04.000"
fn parse_vtt_timing(line: &str) -> Result<(Duration, Duration), SubtitleError> {
    // Find the --> separator first
    let (start_str, rest) = line
        .split_once("-->")
        .ok_or_else(|| SubtitleError::InvalidTimestamp(line.to_string()))?;

    // End timestamp is the first whitespace-delimited token after -->,
    // ignoring any cue settings that may follow
    let end_str = rest.split_whitespace().next().unwrap_or(rest.trim());

    let start = parse_vtt_timestamp(start_str.trim())?;
    let end = parse_vtt_timestamp(end_str)?;
    Ok((start, end))
}

/// Parse VTT timestamp: "00:00:01.000" or "00:01.000" (period for milliseconds)
fn parse_vtt_timestamp(s: &str) -> Result<Duration, SubtitleError> {
    parse_timestamp_common(s)
}

/// Common timestamp parsing: "HH:MM:SS.mmm" or "MM:SS.mmm"
fn parse_timestamp_common(s: &str) -> Result<Duration, SubtitleError> {
    let parts: Vec<&str> = s.split(':').collect();

    let (hours, minutes, seconds_str) = match parts.len() {
        2 => (0u64, parts[0], parts[1]),
        3 => {
            let h: u64 = parts[0]
                .parse()
                .map_err(|_| SubtitleError::InvalidTimestamp(s.to_string()))?;
            (h, parts[1], parts[2])
        }
        _ => return Err(SubtitleError::InvalidTimestamp(s.to_string())),
    };

    let minutes: u64 = minutes
        .parse()
        .map_err(|_| SubtitleError::InvalidTimestamp(s.to_string()))?;

    // Parse seconds and milliseconds
    let sec_parts: Vec<&str> = seconds_str.split('.').collect();
    let seconds: u64 = sec_parts[0]
        .parse()
        .map_err(|_| SubtitleError::InvalidTimestamp(s.to_string()))?;

    let millis: u64 = if sec_parts.len() > 1 {
        // Pad or truncate to 3 digits
        let ms_str = format!("{:0<3}", sec_parts[1]);
        ms_str[..3]
            .parse()
            .map_err(|_| SubtitleError::InvalidTimestamp(s.to_string()))?
    } else {
        0
    };

    let total_millis = hours * 3600 * 1000 + minutes * 60 * 1000 + seconds * 1000 + millis;
    Ok(Duration::from_millis(total_millis))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt_timestamp() {
        let ts = parse_srt_timestamp("00:01:23,456").unwrap();
        assert_eq!(ts, Duration::from_millis(83456));
    }

    #[test]
    fn test_parse_vtt_timestamp() {
        let ts = parse_vtt_timestamp("00:01:23.456").unwrap();
        assert_eq!(ts, Duration::from_millis(83456));

        // Short form (MM:SS.mmm)
        let ts = parse_vtt_timestamp("01:23.456").unwrap();
        assert_eq!(ts, Duration::from_millis(83456));
    }

    #[test]
    fn test_parse_srt() {
        let srt = r#"1
00:00:01,000 --> 00:00:04,000
First subtitle

2
00:00:05,000 --> 00:00:08,000
Second subtitle
with two lines
"#;

        let track = SubtitleTrack::from_srt(srt).unwrap();
        assert_eq!(track.len(), 2);
        assert_eq!(track.cues[0].text, "First subtitle");
        assert_eq!(track.cues[1].text, "Second subtitle\nwith two lines");
    }

    #[test]
    fn test_parse_vtt() {
        let vtt = r#"WEBVTT

00:00:01.000 --> 00:00:04.000
First subtitle

00:00:05.000 --> 00:00:08.000
Second subtitle
"#;

        let track = SubtitleTrack::from_vtt(vtt).unwrap();
        assert_eq!(track.len(), 2);
        assert_eq!(track.cues[0].text, "First subtitle");
    }

    #[test]
    fn test_get_cue_at() {
        let track = SubtitleTrack {
            cues: vec![
                SubtitleCue {
                    start: Duration::from_secs(1),
                    end: Duration::from_secs(4),
                    text: "First".to_string(),
                },
                SubtitleCue {
                    start: Duration::from_secs(5),
                    end: Duration::from_secs(8),
                    text: "Second".to_string(),
                },
            ],
            language: None,
            title: None,
        };

        // Before any cue
        assert!(track.get_cue_at(Duration::from_millis(500)).is_none());

        // During first cue
        let cue = track.get_cue_at(Duration::from_secs(2)).unwrap();
        assert_eq!(cue.text, "First");

        // Between cues
        assert!(track.get_cue_at(Duration::from_millis(4500)).is_none());

        // During second cue
        let cue = track.get_cue_at(Duration::from_secs(6)).unwrap();
        assert_eq!(cue.text, "Second");

        // After all cues
        assert!(track.get_cue_at(Duration::from_secs(10)).is_none());
    }
}
