//! MoQ track subscription and object receipt handling.
//!
//! This module handles subscribing to MoQ tracks and receiving media objects
//! (groups and frames) from the subscribed tracks.

use super::catalog::{AudioTrackInfo, MoqCatalog, VideoTrackInfo};
use super::error::MoqError;

use bytes::Bytes;
use moq_lite::{GroupConsumer, TrackConsumer};

/// State of a track subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptionState {
    /// Not subscribed
    Inactive,
    /// Subscription pending
    Subscribing,
    /// Actively receiving data
    Active,
    /// Subscription closed/ended
    Closed,
    /// Subscription failed
    Failed,
}

/// A received media frame from a MoQ track.
#[derive(Debug, Clone)]
pub struct MoqFrame {
    /// Group sequence number
    pub group_sequence: u64,
    /// Frame index within the group
    pub frame_index: usize,
    /// Frame data
    pub data: Bytes,
    /// Timestamp in milliseconds (if available from timing info)
    pub timestamp_ms: Option<u64>,
}

/// Subscriber for a single MoQ track.
pub struct MoqTrackSubscriber {
    /// Track info (video or audio)
    track_info: TrackInfo,
    /// Track consumer for receiving groups
    consumer: Option<TrackConsumer>,
    /// Current group being read
    current_group: Option<GroupConsumer>,
    /// Current group sequence
    current_group_seq: u64,
    /// Current frame index within group
    current_frame_idx: usize,
    /// Subscription state
    state: SubscriptionState,
    /// Total frames received
    frames_received: u64,
    /// Total bytes received
    bytes_received: u64,
}

/// Track info wrapper for either video or audio.
#[derive(Debug, Clone)]
pub enum TrackInfo {
    Video(VideoTrackInfo),
    Audio(AudioTrackInfo),
}

impl TrackInfo {
    pub fn name(&self) -> &str {
        match self {
            TrackInfo::Video(v) => &v.name,
            TrackInfo::Audio(a) => &a.name,
        }
    }

    pub fn is_video(&self) -> bool {
        matches!(self, TrackInfo::Video(_))
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, TrackInfo::Audio(_))
    }
}

impl MoqTrackSubscriber {
    /// Creates a new subscriber for a video track.
    pub fn for_video(info: VideoTrackInfo) -> Self {
        Self {
            track_info: TrackInfo::Video(info),
            consumer: None,
            current_group: None,
            current_group_seq: 0,
            current_frame_idx: 0,
            state: SubscriptionState::Inactive,
            frames_received: 0,
            bytes_received: 0,
        }
    }

    /// Creates a new subscriber for an audio track.
    pub fn for_audio(info: AudioTrackInfo) -> Self {
        Self {
            track_info: TrackInfo::Audio(info),
            consumer: None,
            current_group: None,
            current_group_seq: 0,
            current_frame_idx: 0,
            state: SubscriptionState::Inactive,
            frames_received: 0,
            bytes_received: 0,
        }
    }

    /// Returns the track info.
    pub fn track_info(&self) -> &TrackInfo {
        &self.track_info
    }

    /// Returns the subscription state.
    pub fn state(&self) -> SubscriptionState {
        self.state
    }

    /// Returns true if the subscription is active.
    pub fn is_active(&self) -> bool {
        self.state == SubscriptionState::Active
    }

    /// Returns the number of frames received.
    pub fn frames_received(&self) -> u64 {
        self.frames_received
    }

    /// Returns the number of bytes received.
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }

    /// Sets the track consumer from a MoQ session.
    ///
    /// Call this after obtaining a TrackConsumer from the broadcast.
    pub fn set_consumer(&mut self, consumer: TrackConsumer) {
        self.consumer = Some(consumer);
        self.state = SubscriptionState::Active;
    }

    /// Reads the next frame from the subscribed track.
    ///
    /// Returns `None` if the track has ended or there's no data available yet.
    pub async fn next_frame(&mut self) -> Result<Option<MoqFrame>, MoqError> {
        let consumer = match &mut self.consumer {
            Some(c) => c,
            None => {
                return Err(MoqError::SubscriptionError(
                    "Track not subscribed".to_string(),
                ))
            }
        };

        // If we have a current group, try to read the next frame
        if let Some(ref mut group) = self.current_group {
            match group.read_frame().await {
                Ok(Some(data)) => {
                    let frame = MoqFrame {
                        group_sequence: self.current_group_seq,
                        frame_index: self.current_frame_idx,
                        data,
                        timestamp_ms: None, // Would need timing info from the stream
                    };
                    self.current_frame_idx += 1;
                    self.frames_received += 1;
                    self.bytes_received += frame.data.len() as u64;
                    return Ok(Some(frame));
                }
                Ok(None) => {
                    // Group exhausted, need new group
                    self.current_group = None;
                }
                Err(e) => {
                    return Err(MoqError::ObjectError(format!("Failed to read frame: {e}")));
                }
            }
        }

        // Try to get the next group
        match consumer.next_group().await {
            Ok(Some(group)) => {
                self.current_group_seq = group.info.sequence;
                self.current_frame_idx = 0;
                self.current_group = Some(group);

                // Try to read first frame from new group
                if let Some(ref mut group) = self.current_group {
                    match group.read_frame().await {
                        Ok(Some(data)) => {
                            let frame = MoqFrame {
                                group_sequence: self.current_group_seq,
                                frame_index: self.current_frame_idx,
                                data,
                                timestamp_ms: None,
                            };
                            self.current_frame_idx += 1;
                            self.frames_received += 1;
                            self.bytes_received += frame.data.len() as u64;
                            return Ok(Some(frame));
                        }
                        Ok(None) => {
                            self.current_group = None;
                            return Ok(None);
                        }
                        Err(e) => {
                            return Err(MoqError::ObjectError(format!(
                                "Failed to read first frame: {e}"
                            )));
                        }
                    }
                }
                Ok(None)
            }
            Ok(None) => {
                // Track closed
                self.state = SubscriptionState::Closed;
                Ok(None)
            }
            Err(e) => Err(MoqError::ObjectError(format!("Failed to get group: {e}"))),
        }
    }

    /// Marks the subscription as failed.
    pub fn set_failed(&mut self) {
        self.state = SubscriptionState::Failed;
        self.consumer = None;
        self.current_group = None;
    }

    /// Closes the subscription.
    pub fn close(&mut self) {
        self.state = SubscriptionState::Closed;
        self.consumer = None;
        self.current_group = None;
    }
}

/// Manager for multiple track subscriptions.
pub struct MoqSubscriptionManager {
    /// Video track subscriber
    video: Option<MoqTrackSubscriber>,
    /// Audio track subscriber
    audio: Option<MoqTrackSubscriber>,
    /// Catalog (if available)
    catalog: Option<MoqCatalog>,
}

impl MoqSubscriptionManager {
    /// Creates a new subscription manager.
    pub fn new() -> Self {
        Self {
            video: None,
            audio: None,
            catalog: None,
        }
    }

    /// Sets the catalog.
    pub fn set_catalog(&mut self, catalog: MoqCatalog) {
        self.catalog = Some(catalog);
    }

    /// Returns the catalog.
    pub fn catalog(&self) -> Option<&MoqCatalog> {
        self.catalog.as_ref()
    }

    /// Creates subscribers from the catalog's primary tracks.
    pub fn subscribe_primary_tracks(&mut self) -> Result<(), MoqError> {
        let catalog = self
            .catalog
            .as_ref()
            .ok_or_else(|| MoqError::CatalogError("No catalog available".to_string()))?;

        if let Some(video_info) = catalog.primary_video() {
            self.video = Some(MoqTrackSubscriber::for_video(video_info.clone()));
        }

        if let Some(audio_info) = catalog.primary_audio() {
            self.audio = Some(MoqTrackSubscriber::for_audio(audio_info.clone()));
        }

        Ok(())
    }

    /// Returns a mutable reference to the video subscriber.
    pub fn video_subscriber(&mut self) -> Option<&mut MoqTrackSubscriber> {
        self.video.as_mut()
    }

    /// Returns a mutable reference to the audio subscriber.
    pub fn audio_subscriber(&mut self) -> Option<&mut MoqTrackSubscriber> {
        self.audio.as_mut()
    }

    /// Returns true if video is being subscribed.
    pub fn has_video(&self) -> bool {
        self.video.is_some()
    }

    /// Returns true if audio is being subscribed.
    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Returns the video track info.
    pub fn video_info(&self) -> Option<&VideoTrackInfo> {
        self.video.as_ref().and_then(|s| match &s.track_info {
            TrackInfo::Video(v) => Some(v),
            _ => None,
        })
    }

    /// Returns the audio track info.
    pub fn audio_info(&self) -> Option<&AudioTrackInfo> {
        self.audio.as_ref().and_then(|s| match &s.track_info {
            TrackInfo::Audio(a) => Some(a),
            _ => None,
        })
    }
}

impl Default for MoqSubscriptionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::moq::catalog::VideoCodec;

    #[test]
    fn test_track_info() {
        let video_info = VideoTrackInfo {
            name: "video".to_string(),
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            bitrate: None,
            init_data: None,
        };

        let track = TrackInfo::Video(video_info);
        assert_eq!(track.name(), "video");
        assert!(track.is_video());
        assert!(!track.is_audio());
    }

    #[test]
    fn test_subscriber_initial_state() {
        let video_info = VideoTrackInfo {
            name: "video".to_string(),
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            bitrate: None,
            init_data: None,
        };

        let subscriber = MoqTrackSubscriber::for_video(video_info);
        assert_eq!(subscriber.state(), SubscriptionState::Inactive);
        assert!(!subscriber.is_active());
        assert_eq!(subscriber.frames_received(), 0);
        assert_eq!(subscriber.bytes_received(), 0);
    }

    #[test]
    fn test_subscription_manager() {
        let manager = MoqSubscriptionManager::new();
        assert!(!manager.has_video());
        assert!(!manager.has_audio());
        assert!(manager.catalog().is_none());
    }
}
