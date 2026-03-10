//! Desktop MoqDecoder implementation (macOS with VideoToolbox, others FFmpeg fallback).

#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::Duration;

use async_channel::{Receiver, Sender};
use tokio::runtime::Handle;

// Re-export shared types from parent (mod.rs)
use super::*;

// Platform-specific imports
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
use crate::media::video::{CpuFrame, Plane};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::media::video::MacOSGpuSurface;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::media::video_decoder::HwAccelConfig;

/// MoQ video decoder using hang crate for media subscription.
///
/// On macOS, this decoder uses VTDecompressionSession for hardware-accelerated
/// zero-copy decoding with IOSurface output for direct GPU rendering.
pub struct MoqDecoder {
    /// Parsed MoQ URL
    #[allow(dead_code)]
    url: MoqUrl,
    /// Configuration
    #[allow(dead_code)]
    config: MoqDecoderConfig,
    /// Shared state with async worker
    shared: Arc<MoqSharedState>,
    /// Receiver for decoded frames from async worker
    frame_rx: Receiver<MoqVideoFrame>,
    /// Active hardware acceleration type
    active_hw_type: HwAccelType,
    /// Owned tokio runtime (created if none exists)
    _owned_runtime: Option<tokio::runtime::Runtime>,
    /// Tokio runtime handle for async operations
    _runtime: Handle,
    /// Whether audio is muted
    audio_muted: bool,
    /// Audio volume (0.0 to 1.0)
    audio_volume: f32,
    /// Cached metadata (updated from shared state to avoid unsafe access)
    cached_metadata: VideoMetadata,
    /// Last frame timestamp for timing
    #[allow(dead_code)]
    last_frame_time: Option<std::time::Instant>,
    /// Start time for PTS calculation
    #[allow(dead_code)]
    start_time: std::time::Instant,
    /// macOS VTDecompressionSession for hardware decoding (zero-copy)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    vt_decoder: Option<super::macos_vt::VTDecoder>,
    /// H.264 AVCC NAL length field size (1, 2, or 4 bytes), from avcC.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    h264_nal_length_size: usize,
    /// True if stream is AVCC format (from catalog avcC). Used for correct NAL
    /// parsing in `find_nal_types_for_format`, avoiding the `data_is_annex_b`
    /// heuristic which misclassifies 256-511 byte NALs.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    is_avcc: bool,
    /// True after 3+ consecutive decode errors; decoder must wait for next IDR
    /// to resync. Cleared only when a real IDR (NAL type 5) arrives.
    /// The VT session is destroyed on sustained decode failures.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    waiting_for_idr_after_error: bool,
    /// Consecutive decode error count. Isolated errors (1-2) skip the frame but
    /// keep the VT session alive. At 3+ consecutive errors, the session is
    /// destroyed and `waiting_for_idr_after_error` is set. Reset to 0 on
    /// successful decode or IDR resync.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    consecutive_decode_errors: u32,
    /// Lightweight DPB grace: skip non-IDR frames after an isolated VT callback
    /// error (consecutive_decode_errors == 1 only). Bypasses note_idr_wait_progress()
    /// to avoid premature resubscribe. Bounded by timeout/drop budget;
    /// on expiry, escalates to waiting_for_idr_after_error for normal recovery.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    skip_pframes_until_idr: bool,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    dpb_grace_started_at: Option<std::time::Instant>,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    dpb_grace_dropped_frames: u32,
    /// Observed real-IDR cadence (EMA, microseconds). Used to adapt DPB grace
    /// timeout/drop budget to stream cadence instead of fixed constants.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    observed_idr_interval_us: Option<u64>,
    /// Last seen real-IDR PTS (microseconds) for cadence estimation.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    last_idr_pts_us: Option<u64>,
    /// Strict recovery gate: after VT session recreation for recovery, only
    /// allow non-IDR frames once a real IDR decodes successfully on the fresh
    /// session.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    require_clean_idr_after_recreate: bool,
    /// True until the one-shot VT session recreation has fired.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    needs_session_recreation: bool,
    /// If set, defer VT session (re)creation until this instant.
    /// This avoids blocking sleeps on decode paths while still enforcing
    /// a quiesce window after session destruction.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    quiesce_until: Option<std::time::Instant>,
    /// Counts VT session creations for lifecycle diagnostics.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    vt_session_count: u32,
    /// Opt-in diagnostics for frame-level forensic logging around decode errors.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    forensic_enabled: bool,
    /// Rolling window of most-recent submitted frames (for N-3 context on failure).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    forensic_recent: VecDeque<ForensicFrameSample>,
    /// Active post-error capture window (+3 frames after failing frame).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    forensic_post_error: Option<PostErrorCapture>,
    /// Monotonic frame sequence number for forensic logs.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    forensic_seq: u64,
    /// Approximate group index, incremented on keyframe boundaries.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    forensic_group_index: u64,
    /// Start of current "waiting for IDR" starvation window.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    idr_wait_started_at: Option<std::time::Instant>,
    /// Number of frames dropped in current IDR starvation window.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    idr_wait_dropped_frames: u32,
    /// Number of keyframe boundaries seen that still lacked a real IDR while
    /// waiting for IDR recovery.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    idr_wait_broken_keyframe_boundaries: u8,
    /// Last time a decoder-side video re-subscribe request was emitted.
    /// Used to prevent rapid request churn when stream metadata is unstable.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    idr_last_resubscribe_request_at: Option<std::time::Instant>,
    /// Start of the current RequiredFrameDropped storm-cycle window.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    required_drop_window_started_at: Option<std::time::Instant>,
    /// Number of storm cycles observed in the current window.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    required_drop_storms_in_window: u8,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Clone, Copy, Debug)]
struct ForensicFrameSample {
    seq: u64,
    group_idx: u64,
    pts_us: u64,
    size: usize,
    is_keyframe: bool,
    hash64: u64,
    first16: [u8; 16],
    first16_len: usize,
    nal_types: [u8; MoqDecoder::MAX_NAL_TYPES],
    nal_count: usize,
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[derive(Clone, Copy, Debug)]
struct PostErrorCapture {
    trigger_seq: u64,
    remaining_after: u8,
}

impl MoqDecoder {
    /// Creates a new MoQ decoder for the given URL.
    pub fn new(url: &str) -> Result<Self, VideoError> {
        Self::new_with_config(url, MoqDecoderConfig::default())
    }

    /// Creates a new MoQ decoder with explicit configuration.
    pub fn new_with_config(url: &str, mut config: MoqDecoderConfig) -> Result<Self, VideoError> {
        tracing::info!("MoqDecoder::new_with_config: creating decoder for {}", url);

        // Parse the MoQ URL
        let moq_url = MoqUrl::parse(url).map_err(|e| {
            tracing::error!("MoQ: failed to parse URL: {}", e);
            VideoError::OpenFailed(e.to_string())
        })?;

        config.apply_localhost_tls_bypass(&moq_url);

        // Get existing runtime handle or create a new runtime
        let (owned_runtime, runtime) = match Handle::try_current() {
            Ok(handle) => (None, handle),
            Err(_) => {
                // No runtime exists, create one for MoQ async operations
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .thread_name("moq-runtime")
                    .build()
                    .map_err(|e| {
                        VideoError::OpenFailed(format!("Failed to create tokio runtime: {e}"))
                    })?;
                let handle = rt.handle().clone();
                (Some(rt), handle)
            }
        };

        // Create shared state
        let shared = Arc::new(MoqSharedState::new());

        // Create channel for frames (bounded to limit memory usage)
        let (frame_tx, frame_rx) = async_channel::bounded(30);

        // Spawn the async connection/subscription worker
        let worker_shared = shared.clone();
        let worker_url = moq_url.clone();
        let worker_config = config.clone();

        runtime.spawn(async move {
            tracing::info!("MoQ: worker task starting for {:?}", worker_url);
            if let Err(e) =
                Self::run_moq_worker(worker_shared.clone(), worker_url, worker_config, frame_tx)
                    .await
            {
                tracing::error!("MoQ: worker error: {}", e);
                worker_shared.set_error(format!("MoQ worker error: {e}"));
            }
        });

        let initial_volume = config.initial_volume;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let forensic_enabled = std::env::var("LUMINA_MOQ_ERROR_FORENSICS")
            .map(|v| v != "0")
            .unwrap_or(false);
        Ok(Self {
            url: moq_url,
            config,
            shared,
            frame_rx,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            active_hw_type: HwAccelType::VideoToolbox,
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            active_hw_type: HwAccelType::None,
            _owned_runtime: owned_runtime,
            _runtime: runtime,
            audio_muted: false,
            audio_volume: initial_volume,
            cached_metadata: VideoMetadata {
                width: 0,
                height: 0,
                duration: None,
                frame_rate: 0.0,
                codec: String::new(),
                pixel_aspect_ratio: 1.0,
                start_time: None,
            },
            last_frame_time: None,
            start_time: std::time::Instant::now(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            vt_decoder: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            h264_nal_length_size: 4,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            is_avcc: false, // updated when VT session is created from catalog avcC
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            waiting_for_idr_after_error: false,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            consecutive_decode_errors: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            skip_pframes_until_idr: false,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            dpb_grace_started_at: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            dpb_grace_dropped_frames: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            observed_idr_interval_us: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            last_idr_pts_us: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            require_clean_idr_after_recreate: false,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            needs_session_recreation: true,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            quiesce_until: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            vt_session_count: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            forensic_enabled,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            forensic_recent: VecDeque::with_capacity(8),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            forensic_post_error: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            forensic_seq: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            forensic_group_index: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            idr_wait_started_at: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            idr_wait_dropped_frames: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            idr_wait_broken_keyframe_boundaries: 0,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            idr_last_resubscribe_request_at: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            required_drop_window_started_at: None,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            required_drop_storms_in_window: 0,
        })
    }

    /// Async worker that handles MoQ connection, catalog fetching, and frame receipt.
    async fn run_moq_worker(
        shared: Arc<MoqSharedState>,
        url: MoqUrl,
        config: MoqDecoderConfig,
        frame_tx: Sender<MoqVideoFrame>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let platform = if cfg!(target_os = "ios") { "iOS" } else { "macOS" };
        crate::media::moq::worker::run_moq_worker(shared, url, config, frame_tx, platform).await
    }

    /// Returns true if this URL is a MoQ URL.
    pub fn is_moq_url(url: &str) -> bool {
        MoqUrl::is_moq_url(url)
    }

    /// Returns a handle to the MoQ shared state for producing stats snapshots.
    ///
    /// This handle can be stored separately (e.g. in VideoPlayer) and used to
    /// query MoQ-specific stats without needing a reference to the decoder.
    pub fn stats_handle(&self) -> MoqStatsHandle {
        MoqStatsHandle {
            shared: self.shared.clone(),
        }
    }

    /// Returns the current decoder state.
    pub fn decoder_state(&self) -> MoqDecoderState {
        *self.shared.state.lock()
    }

    /// Returns the error message if in error state.
    pub fn error_message(&self) -> Option<String> {
        self.shared.error_message.lock().clone()
    }

    /// Returns true if audio is muted.
    pub fn is_muted(&self) -> bool {
        self.audio_muted
    }

    /// Returns the current audio volume.
    pub fn volume(&self) -> f32 {
        self.audio_volume
    }

    /// Returns the audio track info, if available.
    pub fn audio_info(&self) -> Option<AudioTrackInfo> {
        self.shared.audio_info.lock().clone()
    }

    /// Checks if data contains an IDR frame (H.264 NAL type 5).
    ///
    /// Auto-detects Annex B (start codes) vs AVCC (length-prefixed) format.
    /// The hang crate's is_keyframe flag can be wrong when joining mid-stream,
    /// so we parse actual NAL types. Returns true if any NAL is type 5.
    #[allow(dead_code)]
    fn is_idr_frame(nal_data: &[u8], nal_length_size: usize) -> bool {
        let (types, count) = Self::find_nal_types(nal_data, nal_length_size);
        types[..count].contains(&5)
    }

    /// Gets the first NAL type from data for logging.
    #[allow(dead_code)]
    fn get_nal_type(nal_data: &[u8], nal_length_size: usize) -> u8 {
        let (types, count) = Self::find_nal_types(nal_data, nal_length_size);
        if count > 0 {
            types[0]
        } else {
            0
        }
    }

    /// Max NAL units we track per sample (AUD + SPS + PPS + SEI + IDR + spare).
    const MAX_NAL_TYPES: usize = 8;

    /// Returns all NAL types found in data, using known format context.
    ///
    /// When `is_avcc` is known from catalog/init context, use this to avoid
    /// the `data_is_annex_b()` heuristic which misclassifies AVCC frames
    /// whose first NAL is 256-511 bytes (length prefix `[0,0,1,X]` looks
    /// like an Annex B start code).
    pub(crate) fn find_nal_types_for_format(
        nal_data: &[u8],
        nal_length_size: usize,
        is_avcc: bool,
    ) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        if is_avcc {
            Self::find_nal_types_avcc(nal_data, nal_length_size)
        } else {
            Self::find_nal_types_annex_b(nal_data)
        }
    }

    /// Returns all NAL types found in data, auto-detecting Annex B vs AVCC format.
    ///
    /// WARNING: The heuristic `data_is_annex_b()` misclassifies AVCC frames whose
    /// first NAL is 256-511 bytes. Prefer `find_nal_types_for_format()` when the
    /// format is known from catalog context.
    pub(crate) fn find_nal_types(
        nal_data: &[u8],
        nal_length_size: usize,
    ) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        if Self::data_is_annex_b(nal_data) {
            Self::find_nal_types_annex_b(nal_data)
        } else {
            Self::find_nal_types_avcc(nal_data, nal_length_size)
        }
    }

    /// Check if data starts with Annex B start codes.
    ///
    /// WARNING: This is a heuristic that can produce false positives for AVCC data
    /// where the first NAL length is 256-511 bytes (prefix `[0,0,1,X]`).
    pub(crate) fn data_is_annex_b(data: &[u8]) -> bool {
        matches!(data, [0, 0, 0, 1, ..] | [0, 0, 1, ..])
    }

    /// Extract NAL types from Annex B bitstream (start-code delimited).
    fn find_nal_types_annex_b(data: &[u8]) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        let mut types = [0u8; Self::MAX_NAL_TYPES];
        let mut count = 0;
        let mut i = 0;
        while i < data.len() && count < Self::MAX_NAL_TYPES {
            let sc_len = if data.get(i..i + 4) == Some(&[0, 0, 0, 1]) {
                4
            } else if data.get(i..i + 3) == Some(&[0, 0, 1]) {
                3
            } else {
                i += 1;
                continue;
            };
            let nal_start = i + sc_len;
            if let Some(&byte) = data.get(nal_start) {
                types[count] = byte & 0x1F;
                count += 1;
            }
            i = nal_start + 1;
        }
        (types, count)
    }

    /// Extract NAL types from AVCC data (length-prefixed).
    fn find_nal_types_avcc(
        nal_data: &[u8],
        nal_length_size: usize,
    ) -> ([u8; Self::MAX_NAL_TYPES], usize) {
        let mut types = [0u8; Self::MAX_NAL_TYPES];
        let mut count = 0;
        if !(1..=4).contains(&nal_length_size) {
            return (types, count);
        }
        let mut offset = 0usize;
        while offset + nal_length_size <= nal_data.len() && count < Self::MAX_NAL_TYPES {
            let len_bytes = match nal_data.get(offset..offset + nal_length_size) {
                Some(b) => b,
                None => break,
            };
            let mut nal_len = 0usize;
            for &byte in len_bytes {
                nal_len = (nal_len << 8) | byte as usize;
            }
            offset += nal_length_size;
            if nal_len == 0 || offset + nal_len > nal_data.len() {
                break;
            }
            types[count] = match nal_data.get(offset) {
                Some(&b) => b & 0x1F,
                None => break,
            };
            count += 1;
            offset += nal_len;
        }
        (types, count)
    }

    /// Returns true if frame should be skipped while waiting for an IDR resync.
    /// Only accepts real IDR (NAL type 5) as valid resync point. I-frames (NAL type 1
    /// with is_keyframe=true) cannot initialize a fresh VT session because they need
    /// existing DPB reference frames.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn should_wait_for_idr(&self, moq_frame: &MoqVideoFrame) -> bool {
        if !self.waiting_for_idr_after_error {
            return false;
        }
        let (types, count) = Self::find_nal_types_for_format(
            &moq_frame.data,
            self.h264_nal_length_size,
            self.is_avcc,
        );
        let is_idr = types[..count].contains(&5);
        !is_idr // only real IDR can clear the resync gate
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn fnv1a64(data: &[u8]) -> u64 {
        // Deterministic lightweight fingerprint for cross-run frame matching.
        let mut hash: u64 = 0xcbf29ce484222325;
        for &b in data {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn record_forensic_sample(&mut self, moq_frame: &MoqVideoFrame, nal_types: &[u8]) {
        if !self.forensic_enabled {
            return;
        }

        if moq_frame.is_keyframe {
            self.forensic_group_index = self.forensic_group_index.saturating_add(1);
        }
        self.forensic_seq = self.forensic_seq.saturating_add(1);

        let mut first16 = [0u8; 16];
        let first16_len = moq_frame.data.len().min(16);
        first16[..first16_len].copy_from_slice(&moq_frame.data[..first16_len]);

        let mut nal_arr = [0u8; Self::MAX_NAL_TYPES];
        let nal_count = nal_types.len().min(Self::MAX_NAL_TYPES);
        nal_arr[..nal_count].copy_from_slice(&nal_types[..nal_count]);

        let sample = ForensicFrameSample {
            seq: self.forensic_seq,
            group_idx: self.forensic_group_index,
            pts_us: moq_frame.timestamp_us,
            size: moq_frame.data.len(),
            is_keyframe: moq_frame.is_keyframe,
            hash64: Self::fnv1a64(&moq_frame.data),
            first16,
            first16_len,
            nal_types: nal_arr,
            nal_count,
        };

        if self.forensic_recent.len() == 8 {
            let _ = self.forensic_recent.pop_front();
        }
        self.forensic_recent.push_back(sample);

        if let Some(mut post) = self.forensic_post_error {
            if sample.seq > post.trigger_seq && post.remaining_after > 0 {
                tracing::warn!(
                    "MoQ forensic post-error +{}: seq={}, group\u{2248}{}, pts={}us, size={}, keyframe={}, hash64={:016x}, nal_types={:?}, first16={:02x?}",
                    (4 - post.remaining_after) as usize,
                    sample.seq,
                    sample.group_idx,
                    sample.pts_us,
                    sample.size,
                    sample.is_keyframe,
                    sample.hash64,
                    &sample.nal_types[..sample.nal_count],
                    &sample.first16[..sample.first16_len]
                );
                post.remaining_after -= 1;
                if post.remaining_after == 0 {
                    tracing::warn!("MoQ forensic: post-error window capture complete");
                    self.forensic_post_error = None;
                } else {
                    self.forensic_post_error = Some(post);
                }
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn log_forensic_error_window(&mut self) {
        if !self.forensic_enabled {
            return;
        }
        let Some(failing) = self.forensic_recent.back().copied() else {
            return;
        };

        tracing::warn!(
            "MoQ forensic trigger: seq={}, group\u{2248}{}, pts={}us, size={}, keyframe={}, hash64={:016x}, nal_types={:?}, first16={:02x?}",
            failing.seq,
            failing.group_idx,
            failing.pts_us,
            failing.size,
            failing.is_keyframe,
            failing.hash64,
            &failing.nal_types[..failing.nal_count],
            &failing.first16[..failing.first16_len]
        );

        for sample in self.forensic_recent.iter().rev().skip(1).take(3).rev() {
            tracing::warn!(
                "MoQ forensic pre-error: seq={}, group\u{2248}{}, pts={}us, size={}, keyframe={}, hash64={:016x}, nal_types={:?}, first16={:02x?}",
                sample.seq,
                sample.group_idx,
                sample.pts_us,
                sample.size,
                sample.is_keyframe,
                sample.hash64,
                &sample.nal_types[..sample.nal_count],
                &sample.first16[..sample.first16_len]
            );
        }

        self.forensic_post_error = Some(PostErrorCapture {
            trigger_seq: failing.seq,
            remaining_after: 3,
        });
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn reset_idr_wait_tracking(&mut self) {
        self.idr_wait_started_at = None;
        self.idr_wait_dropped_frames = 0;
        self.idr_wait_broken_keyframe_boundaries = 0;
    }

    /// Ensure a VTDecompressionSession exists, creating one if needed.
    ///
    /// Tries catalog avcC description first, falls back to keyframe SPS/PPS
    /// extraction. Respects quiesce window after session destruction.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn ensure_vt_session(&mut self, moq_frame: &MoqVideoFrame) -> Result<(), VideoError> {
        if self.vt_decoder.is_some() {
            return Ok(());
        }

        // Enforce non-blocking quiesce window after session destruction
        if let Some(quiesce_until) = self.quiesce_until {
            let now = std::time::Instant::now();
            if now < quiesce_until {
                return Err(VideoError::DecodeFailed(
                    "Waiting for VT quiesce".to_string(),
                ));
            }
            self.quiesce_until = None;
            tracing::info!("MoQ: VT quiesce window complete, creating new session");
        }

        let metadata = self.shared.metadata.lock().clone();

        // First, try to use codec description from catalog (avcC/hvcC box)
        if let Some(ref desc) = *self.shared.codec_description.lock() {
            match Self::parse_avcc_box(desc) {
                Ok((sps, pps, nal_length_size)) => {
                    let decoder = super::macos_vt::VTDecoder::new_h264(
                        &sps,
                        &pps,
                        metadata.width,
                        metadata.height,
                        true, // catalog avcC = AVCC format
                    )?;
                    self.vt_decoder = Some(decoder);
                    self.h264_nal_length_size = nal_length_size;
                    self.is_avcc = true;
                    self.vt_session_count += 1;
                    tracing::info!("MoQ: initialized VTDecoder session #{} from catalog avcC ({} bytes SPS, {} bytes PPS, NAL len size {})", self.vt_session_count, sps.len(), pps.len(), nal_length_size);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("MoQ: failed to parse avcC from catalog: {}", e);
                }
            }
        }

        // Fallback: extract SPS/PPS from keyframe
        if !moq_frame.is_keyframe {
            tracing::debug!("MoQ: waiting for keyframe to initialize VTDecoder");
            return Err(VideoError::DecodeFailed(
                "Waiting for keyframe with SPS/PPS".to_string(),
            ));
        }

        match Self::extract_h264_params(&moq_frame.data) {
            Ok((sps, pps)) => {
                let decoder = super::macos_vt::VTDecoder::new_h264(
                    &sps,
                    &pps,
                    metadata.width,
                    metadata.height,
                    false, // keyframe extraction = Annex B format
                )?;
                self.vt_decoder = Some(decoder);
                self.h264_nal_length_size = 4;
                self.vt_session_count += 1;
                tracing::info!(
                    "MoQ: initialized VTDecoder session #{} from keyframe SPS/PPS (Annex B)",
                    self.vt_session_count
                );
                Ok(())
            }
            Err(e) => {
                tracing::warn!("MoQ: failed to extract H.264 params: {}", e);
                Err(e)
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn request_video_resubscribe_with_cooldown(&mut self) -> bool {
        const RESUBSCRIBE_REQUEST_COOLDOWN: Duration = Duration::from_millis(800);
        let now = std::time::Instant::now();
        if let Some(last) = self.idr_last_resubscribe_request_at {
            if now.saturating_duration_since(last) < RESUBSCRIBE_REQUEST_COOLDOWN {
                return false;
            }
        }
        if self
            .shared
            .request_video_resubscribe
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.idr_last_resubscribe_request_at = Some(now);
            return true;
        }
        false
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn note_idr_wait_progress(&mut self, nal_type: u8, frame_len: usize, is_keyframe: bool) {
        // Keep recovery bounded: if a real IDR doesn't arrive quickly, force
        // re-subscribe so we can rejoin on a fresh group boundary.
        const IDR_WAIT_MAX: Duration = Duration::from_millis(1000);
        const IDR_WAIT_MAX_DROPS: u32 = 24;
        const BROKEN_KEYFRAME_BOUNDARY_THRESHOLD: u8 = 3;

        let start = match self.idr_wait_started_at {
            Some(start) => start,
            None => {
                let now = std::time::Instant::now();
                self.idr_wait_started_at = Some(now);
                now
            }
        };
        self.idr_wait_dropped_frames = self.idr_wait_dropped_frames.saturating_add(1);

        // Metadata keyframe without a real IDR is common in degenerate groups.
        // Do not immediately re-subscribe on a single boundary; require a
        // short sequence of broken boundaries to avoid churn.
        let broken_keyframe_boundary = is_keyframe && nal_type != 5;
        if broken_keyframe_boundary {
            self.idr_wait_broken_keyframe_boundaries =
                self.idr_wait_broken_keyframe_boundaries.saturating_add(1);
            if self.idr_wait_broken_keyframe_boundaries >= BROKEN_KEYFRAME_BOUNDARY_THRESHOLD
                && self.request_video_resubscribe_with_cooldown()
            {
                tracing::warn!(
                    "MoQ: repeated keyframe boundaries without IDR (count={}, nal_type={}, {} bytes) — requesting video re-subscribe",
                    self.idr_wait_broken_keyframe_boundaries,
                    nal_type,
                    frame_len
                );
                self.idr_wait_started_at = Some(std::time::Instant::now());
                self.idr_wait_dropped_frames = 0;
                self.idr_wait_broken_keyframe_boundaries = 0;
                return;
            }
        }

        let elapsed = start.elapsed();
        if elapsed < IDR_WAIT_MAX && self.idr_wait_dropped_frames < IDR_WAIT_MAX_DROPS {
            return;
        }

        if self.request_video_resubscribe_with_cooldown() {
            tracing::warn!(
                "MoQ: IDR starvation ({}ms, {} dropped, broken_keyframes={}, last_nal_type={}, {} bytes) — requesting video re-subscribe",
                elapsed.as_millis(),
                self.idr_wait_dropped_frames,
                self.idr_wait_broken_keyframe_boundaries,
                nal_type,
                frame_len
            );
        }

        // Keep reporting at a bounded cadence if starvation persists.
        self.idr_wait_started_at = Some(std::time::Instant::now());
        self.idr_wait_dropped_frames = 0;
        self.idr_wait_broken_keyframe_boundaries = 0;
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn note_required_drop_storm_cycle(&mut self) -> u8 {
        const STORM_ESCALATION_WINDOW: Duration = Duration::from_millis(4500);

        let now = std::time::Instant::now();
        match self.required_drop_window_started_at {
            Some(start) if now.saturating_duration_since(start) <= STORM_ESCALATION_WINDOW => {
                self.required_drop_storms_in_window =
                    self.required_drop_storms_in_window.saturating_add(1);
            }
            _ => {
                self.required_drop_window_started_at = Some(now);
                self.required_drop_storms_in_window = 1;
            }
        }
        self.required_drop_storms_in_window
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn note_real_idr_timestamp(&mut self, pts_us: u64) {
        // Ignore clearly invalid cadence deltas.
        const MIN_IDR_INTERVAL_US: u64 = 300_000;
        const MAX_IDR_INTERVAL_US: u64 = 8_000_000;

        if let Some(prev_pts) = self.last_idr_pts_us {
            let delta_us = pts_us.saturating_sub(prev_pts);
            if (MIN_IDR_INTERVAL_US..=MAX_IDR_INTERVAL_US).contains(&delta_us) {
                self.observed_idr_interval_us = Some(match self.observed_idr_interval_us {
                    // Smooth noisy group boundaries while still adapting.
                    Some(current) => ((current * 3) + delta_us) / 4,
                    None => delta_us,
                });
            }
        }
        self.last_idr_pts_us = Some(pts_us);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn dpb_grace_budget(&self) -> (Duration, u32) {
        // Base timeout from observed IDR cadence + margin for jitter/bursting.
        let observed_us = self.observed_idr_interval_us.unwrap_or(2_000_000);
        let timeout_us = observed_us.saturating_add(900_000);
        let timeout_us = timeout_us.clamp(2_500_000, 5_000_000);
        let timeout = Duration::from_micros(timeout_us);

        let fps = if self.cached_metadata.frame_rate.is_finite()
            && self.cached_metadata.frame_rate > 1.0
        {
            self.cached_metadata.frame_rate as f64
        } else {
            24.0
        };
        let max_drops = ((timeout.as_secs_f64() * fps).ceil() as u32)
            .saturating_add(8)
            .clamp(60, 180);

        (timeout, max_drops)
    }

    /// Decodes an encoded frame.
    ///
    /// On macOS, uses VTDecompressionSession for zero-copy hardware decoding.
    /// On other platforms, returns a placeholder (FFmpeg integration TODO).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn decode_frame(&mut self, moq_frame: &MoqVideoFrame) -> Result<VideoFrame, VideoError> {
        // Initialize VTDecoder lazily (creates from catalog avcC or keyframe SPS/PPS)
        self.ensure_vt_session(moq_frame)?;

        // Check if we're waiting for IDR resync after a decode error
        if self.should_wait_for_idr(moq_frame) {
            let (t, c) = Self::find_nal_types_for_format(
                &moq_frame.data,
                self.h264_nal_length_size,
                self.is_avcc,
            );
            let nal_type = if c > 0 { t[0] } else { 0 };
            self.note_idr_wait_progress(nal_type, moq_frame.data.len(), moq_frame.is_keyframe);
            tracing::debug!(
                "MoQ: waiting for IDR resync after decode error (got NAL type {}, is_keyframe={}, {} bytes)",
                nal_type, moq_frame.is_keyframe, moq_frame.data.len()
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (got NAL type {})",
                nal_type
            )));
        }
        self.reset_idr_wait_tracking();

        // Check NAL types for diagnostics and keyframe validation.
        // Use format-aware parsing (self.is_avcc) to avoid data_is_annex_b() heuristic bug.
        let (nal_types_arr, nal_count) = Self::find_nal_types_for_format(
            &moq_frame.data,
            self.h264_nal_length_size,
            self.is_avcc,
        );
        let nal_types = &nal_types_arr[..nal_count];
        let is_idr = nal_types.contains(&5);
        if is_idr {
            self.note_real_idr_timestamp(moq_frame.timestamp_us);
        }
        self.record_forensic_sample(moq_frame, nal_types);

        // Bounded DPB grace: skip non-IDR frames after isolated VT callback error.
        // The skipped error frame leaves a stale DPB reference — subsequent
        // P-frames decode with status=0 but produce macroblock artifacts.
        // Bypasses note_idr_wait_progress() to avoid premature resubscribe.
        let (dpb_grace_timeout, dpb_grace_max_drops) = self.dpb_grace_budget();
        if self.skip_pframes_until_idr {
            if is_idr {
                let dpb_drops = self.dpb_grace_dropped_frames;
                // Clear DPB grace state
                self.skip_pframes_until_idr = false;
                self.dpb_grace_started_at = None;
                self.dpb_grace_dropped_frames = 0;
                // Reset error tracking — fresh session starts clean
                self.consecutive_decode_errors = 0;
                self.waiting_for_idr_after_error = false;
                self.reset_idr_wait_tracking();
                // Destroy corrupted VT session — IDR alone doesn't clear
                // VT's internal corruption state (r34: SS4 still pixelated
                // after IDR on same session, SS6 clean after recreation).
                self.vt_decoder = None;
                self.require_clean_idr_after_recreate = true;
                self.ensure_vt_session(moq_frame)?;
                tracing::info!(
                    "MoQ: DPB grace cleared by IDR — recreated VT session #{} ({} frames dropped, budget={}ms/{} drops)",
                    self.vt_session_count,
                    dpb_drops,
                    dpb_grace_timeout.as_millis(),
                    dpb_grace_max_drops
                );
                // Fall through to decode this IDR on the fresh session
            } else {
                let start = *self
                    .dpb_grace_started_at
                    .get_or_insert_with(std::time::Instant::now);
                self.dpb_grace_dropped_frames += 1;
                self.shared
                    .frame_stats
                    .dropped_dpb_grace
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let elapsed = start.elapsed();

                if elapsed > dpb_grace_timeout
                    || self.dpb_grace_dropped_frames > dpb_grace_max_drops
                {
                    // Grace expired — escalate to normal IDR-wait with resubscribe.
                    self.skip_pframes_until_idr = false;
                    let drops = self.dpb_grace_dropped_frames;
                    self.dpb_grace_started_at = None;
                    self.dpb_grace_dropped_frames = 0;
                    self.waiting_for_idr_after_error = true;
                    tracing::warn!(
                        "MoQ: DPB grace expired ({}ms, {} drops; budget={}ms/{} drops) — escalating to IDR-wait",
                        elapsed.as_millis(),
                        drops,
                        dpb_grace_timeout.as_millis(),
                        dpb_grace_max_drops
                    );
                }

                // Use distinct message that does NOT contain "Waiting for IDR frame"
                // to avoid double-counting in dropped_waiting_idr (line ~1729).
                return Err(VideoError::DecodeFailed(format!(
                    "DPB grace skip (NAL type {})",
                    nal_types.first().copied().unwrap_or(0)
                )));
            }
        }

        if self.require_clean_idr_after_recreate && !is_idr {
            tracing::debug!(
                "MoQ: post-recreate clean-IDR gate active (NAL type {}, {} bytes)",
                nal_types.first().copied().unwrap_or(0),
                moq_frame.data.len()
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (post-recreate gate, NAL type {})",
                nal_types.first().copied().unwrap_or(0)
            )));
        }

        // One-shot VT session recreation at the second IDR after startup.
        // The initial VT session can produce silently corrupted output (VT
        // status=0 but visibly pixelated). Recreating once at the next IDR
        // boundary clears the corruption. Requires at least 48 frames decoded
        // (one full group at 24fps) to ensure VT hardware has flushed, and
        // only triggers on real IDR (type 5), not I-frames (type 1).
        // A/B test: set to false to skip one-shot recreation and isolate
        // whether mid-session errors are content-dependent or lifecycle-dependent.
        // r31 result: HARMFUL — destroyed working session, caused 44 IDR drops,
        // FPS dropped to 8.0, errors increased from 2→8. Errors are content-dependent
        // (specific BBB P-frames at group position 3), not lifecycle-dependent.
        const ENABLE_ONESHOT_RECREATION: bool = false;
        if ENABLE_ONESHOT_RECREATION
            && self.needs_session_recreation
            && !self.waiting_for_idr_after_error
            && is_idr
        {
            if let Some(ref decoder) = self.vt_decoder {
                let prev_count = decoder.frame_count();
                // Only trigger after at least one full group (48 frames)
                // to give VT hardware time to fully initialize.
                if prev_count >= 48 {
                    self.needs_session_recreation = false;
                    // Drop triggers: WaitForAsync → Invalidate → CFRelease
                    self.vt_decoder = None;
                    self.quiesce_until =
                        Some(std::time::Instant::now() + std::time::Duration::from_millis(50));
                    tracing::info!(
                        "MoQ: one-shot VT recreation scheduled after quiesce window ({} bytes, {} frames on previous session)",
                        moq_frame.data.len(),
                        prev_count
                    );
                    return Err(VideoError::DecodeFailed(
                        "Waiting for VT quiesce".to_string(),
                    ));
                }
            }
        }

        if let Some(ref decoder) = self.vt_decoder {
            let frame_count = decoder.frame_count();

            // Log first 10 frames at INFO level for pipeline diagnostics
            if frame_count < 10 {
                let mut preview = [0u8; 20];
                let preview_len = moq_frame.data.len().min(20);
                preview[..preview_len].copy_from_slice(&moq_frame.data[..preview_len]);
                tracing::info!(
                    "MoQ decode frame #{}: {} bytes, is_keyframe={}, format={}, NAL types={:?}, is_idr={}, first 20 bytes={:02x?}",
                    frame_count,
                    moq_frame.data.len(),
                    moq_frame.is_keyframe,
                    if self.is_avcc { "AVCC" } else { "AnnexB" },
                    nal_types,
                    is_idr,
                    &preview[..preview_len],
                );
            }

            // First submitted frame must be a real IDR. Trusting metadata-only keyframe
            // flags (with non-IDR NAL 1 payloads) can poison VT reference state.
            if frame_count == 0 && !is_idr {
                // New VT sessions must begin on an IDR access unit.
                // Enter the normal IDR-wait path so this is treated as non-fatal,
                // counted as dropped-waiting-IDR, and eligible for bounded re-subscribe.
                self.waiting_for_idr_after_error = true;
                tracing::warn!(
                    "MoQ: dropping frame #{} — first frame is not IDR (NAL types={:?}, is_keyframe={}, {} bytes, format={})",
                    frame_count,
                    nal_types,
                    moq_frame.is_keyframe,
                    moq_frame.data.len(),
                    if self.is_avcc { "AVCC" } else { "AnnexB" },
                );
                return Err(VideoError::DecodeFailed(format!(
                    "Waiting for IDR frame (got NAL types {:?})",
                    nal_types
                )));
            }

            // Only clear IDR resync on a real IDR (NAL type 5). I-frames (NAL type 1
            // with is_keyframe=true) cannot initialize a fresh VT session — they need
            // existing reference frames in the DPB. Accepting I-frames here causes a
            // cascade: fresh session → decode fail → destroy → repeat.
            if self.waiting_for_idr_after_error && is_idr {
                self.waiting_for_idr_after_error = false;
                self.consecutive_decode_errors = 0;
                self.reset_idr_wait_tracking();
                tracing::info!(
                    "MoQ: received real IDR after error, will recreate VT session (session={})",
                    if self.vt_decoder.is_some() {
                        "exists"
                    } else {
                        "None"
                    }
                );
            }
        }

        // Treat metadata keyframe boundaries without a real IDR as discontinuities.
        // Decoding these as plain P-frames can poison visual output without emitting
        // callback errors. Instead, enter IDR wait and let the existing bounded
        // recovery/resubscribe machinery converge on a real IDR boundary.
        if moq_frame.is_keyframe && !is_idr {
            self.waiting_for_idr_after_error = true;
            tracing::warn!(
                "MoQ: keyframe metadata mismatch (is_keyframe=true but NAL types={:?}); entering IDR wait",
                nal_types
            );
            return Err(VideoError::DecodeFailed(format!(
                "Waiting for IDR frame (metadata keyframe without IDR, NAL types {:?})",
                nal_types
            )));
        }

        // Decode the frame using VTDecoder
        let vt_is_keyframe = is_idr;

        let decode_result = if let Some(ref mut decoder) = self.vt_decoder {
            decoder.decode_frame(&moq_frame.data, moq_frame.timestamp_us, vt_is_keyframe)
        } else {
            return Err(VideoError::DecodeFailed(
                "VTDecoder not initialized".to_string(),
            ));
        };

        match decode_result {
            Ok(Some(frame)) => {
                // Track decoded frame in shared stats
                let decoded_count = self
                    .shared
                    .frame_stats
                    .decoded
                    .fetch_add(1, Ordering::Relaxed)
                    + 1;

                // Reset consecutive error counter on success
                self.consecutive_decode_errors = 0;
                if self.require_clean_idr_after_recreate && is_idr {
                    self.require_clean_idr_after_recreate = false;
                    tracing::info!(
                        "MoQ: post-recreate clean-IDR gate satisfied on session #{}",
                        self.vt_session_count
                    );
                }

                // Log first few successful decodes
                if decoded_count <= 5 {
                    tracing::info!(
                        "MoQ: VT decoded frame #{} (pts={}us)",
                        decoded_count,
                        moq_frame.timestamp_us
                    );
                }
                Ok(frame)
            }
            Ok(None) => Err(VideoError::DecodeFailed(
                "VTDecoder: no frame decoded (async?)".to_string(),
            )),
            Err(e) => {
                self.consecutive_decode_errors += 1;
                let total_errors = self
                    .shared
                    .frame_stats
                    .decode_errors
                    .load(std::sync::atomic::Ordering::Relaxed)
                    + 1;
                let error_text = e.to_string();
                let hard_vt_callback_failure = error_text.contains("VT decode callback error:");
                let required_drop_storm = error_text.contains("VT required-frame-drop storm");
                if hard_vt_callback_failure {
                    // Let consecutive_decode_errors increment naturally (+= 1 below).
                    // First 1-2 errors: soft skip (keep session, no resubscribe).
                    // At 3+ consecutive: hard reset + IDR resync + resubscribe.
                    // This avoids destroying a valid VT session for isolated P-frame
                    // errors (-12909) which are common on macOS with certain H.264
                    // content. The session resets to 0 on any successful decode.
                    if self.consecutive_decode_errors >= 2 {
                        // About to hit 3 — request resubscribe for fast IDR delivery
                        if self.request_video_resubscribe_with_cooldown() {
                            tracing::warn!(
                                "MoQ: VT callback failure #{} — requesting video re-subscribe",
                                self.consecutive_decode_errors + 1
                            );
                        }
                    }
                } else if required_drop_storm {
                    // Keep isolated storms soft, but don't loop forever on repeated
                    // storm cycles in a short window.
                    const STORM_ESCALATION_THRESHOLD: u8 = 3;
                    const STORM_ESCALATION_WINDOW_MS: u128 = 4500;
                    let storms_in_window = self.note_required_drop_storm_cycle();
                    if storms_in_window >= STORM_ESCALATION_THRESHOLD {
                        self.consecutive_decode_errors = 3;
                        self.required_drop_window_started_at = Some(std::time::Instant::now());
                        self.required_drop_storms_in_window = 0;
                        let requested_resubscribe = self.request_video_resubscribe_with_cooldown();
                        tracing::warn!(
                            "MoQ: required-frame-drop storm persisted ({} storms within ~{}ms) — escalating to VT session reset{}",
                            storms_in_window,
                            STORM_ESCALATION_WINDOW_MS,
                            if requested_resubscribe {
                                " + video re-subscribe request"
                            } else {
                                ""
                            }
                        );
                    } else {
                        self.consecutive_decode_errors = 0;
                        self.waiting_for_idr_after_error = false;
                        tracing::warn!(
                            "MoQ: required-frame-drop storm detected (window_count={}) — skipping frame without re-subscribe",
                            storms_in_window
                        );
                    }
                }
                self.log_forensic_error_window();

                // Log NAL header bytes for forensic analysis of failing frames
                let data = &moq_frame.data;
                let nal_header_hex = if data.len() >= 16 {
                    format!("{:02x?}", &data[..16])
                } else {
                    format!("{:02x?}", data)
                };
                // Parse first NAL type from AVCC (4-byte length prefix)
                let first_nal_type = if self.is_avcc {
                    match data.get(self.h264_nal_length_size) {
                        Some(&nal_byte) => {
                            format!(
                                "nal_type={} ({})",
                                nal_byte & 0x1f,
                                match nal_byte & 0x1f {
                                    1 => "non-IDR slice",
                                    5 => "IDR slice",
                                    6 => "SEI",
                                    7 => "SPS",
                                    8 => "PPS",
                                    _ => "other",
                                }
                            )
                        }
                        None => "unknown".to_string(),
                    }
                } else {
                    "unknown".to_string()
                };
                tracing::warn!(
                    "MoQ: failing frame forensics: size={}, is_keyframe={}, pts={}us, {}, header={}",
                    data.len(), moq_frame.is_keyframe, moq_frame.timestamp_us,
                    first_nal_type, nal_header_hex
                );

                if self.consecutive_decode_errors >= 3 {
                    // 3+ consecutive errors: destroy VT session and wait for IDR resync.
                    // prepare_for_idr_resync() only clears the output queue — VT's
                    // internal DPB retains stale reference frames. Full session
                    // recreation from catalog SPS/PPS is needed for clean recovery.
                    self.waiting_for_idr_after_error = true;
                    self.skip_pframes_until_idr = false;
                    self.dpb_grace_started_at = None;
                    self.dpb_grace_dropped_frames = 0;
                    self.require_clean_idr_after_recreate = true;
                    self.vt_decoder = None; // Drop: WaitForAsync → Invalidate → CFRelease
                    self.quiesce_until =
                        Some(std::time::Instant::now() + std::time::Duration::from_millis(50));
                    tracing::warn!(
                        "MoQ: VT decode error #{} (consecutive={}), destroyed session for IDR resync: {}",
                        total_errors, self.consecutive_decode_errors, e
                    );
                } else {
                    self.waiting_for_idr_after_error = false;

                    if hard_vt_callback_failure && self.consecutive_decode_errors == 1 {
                        // First isolated VT callback failure: the skipped P-frame was
                        // a DPB reference. Subsequent P-frames will decode successfully
                        // but produce macroblock artifacts. Enter bounded DPB grace
                        // to skip until next natural IDR resets the DPB.
                        self.skip_pframes_until_idr = true;
                        self.dpb_grace_started_at = None;
                        self.dpb_grace_dropped_frames = 0;
                        tracing::warn!(
                            "MoQ: VT callback error #{} (isolated), DPB grace until next IDR: {}",
                            total_errors,
                            e
                        );
                    } else {
                        tracing::warn!(
                            "MoQ: VT decode error #{} (consecutive={}), skipping frame without IDR gate: {}",
                            total_errors,
                            self.consecutive_decode_errors,
                            e
                        );
                    }
                }
                Err(e)
            }
        }
    }

    /// Decodes an encoded frame to YUV (non-macOS fallback).
    ///
    /// In a full implementation, this would use FFmpeg to decode H.264/H.265/AV1.
    /// For now, we return a placeholder frame to demonstrate the pipeline.
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn decode_frame(&mut self, moq_frame: &MoqVideoFrame) -> Result<VideoFrame, VideoError> {
        let metadata = self.shared.metadata.lock();
        let width = metadata.width as usize;
        let height = metadata.height as usize;

        // Create a gray YUV420p frame (placeholder until FFmpeg integration)
        // TODO: Use FFmpeg to decode the actual H.264/H.265/AV1 NAL units
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        let y_plane = vec![128u8; y_size]; // Gray Y
        let u_plane = vec![128u8; uv_size]; // Neutral U
        let v_plane = vec![128u8; uv_size]; // Neutral V

        let cpu_frame = CpuFrame {
            format: PixelFormat::Yuv420p,
            width: metadata.width,
            height: metadata.height,
            planes: vec![
                Plane {
                    data: y_plane,
                    stride: width,
                },
                Plane {
                    data: u_plane,
                    stride: width / 2,
                },
                Plane {
                    data: v_plane,
                    stride: width / 2,
                },
            ],
        };

        // Calculate PTS from MoQ timestamp
        let pts = Duration::from_micros(moq_frame.timestamp_us);

        Ok(VideoFrame::new(pts, DecodedFrame::Cpu(cpu_frame)))
    }

    /// Parses an avcC box (H.264 decoder configuration record) to extract SPS and PPS.
    ///
    /// avcC format:
    /// - 1 byte: version (always 1)
    /// - 1 byte: profile
    /// - 1 byte: compatibility
    /// - 1 byte: level
    /// - 1 byte: 0xFC | (NAL length size - 1)
    /// - 1 byte: 0xE0 | num_sps
    /// - For each SPS: 2 bytes length (big endian) + SPS data
    /// - 1 byte: num_pps
    /// - For each PPS: 2 bytes length (big endian) + PPS data
    pub(crate) fn parse_avcc_box(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>, usize), VideoError> {
        if data.len() < 7 {
            return Err(VideoError::DecodeFailed("avcC too short".to_string()));
        }

        let version = data[0];
        if version != 1 {
            return Err(VideoError::DecodeFailed(format!(
                "Unsupported avcC version: {}",
                version
            )));
        }

        // Extract NAL length size from byte 4: (lengthSizeMinusOne & 0x03) + 1
        let nal_length_size = ((data[4] & 0x03) + 1) as usize;
        if !(1..=4).contains(&nal_length_size) {
            return Err(VideoError::DecodeFailed(format!(
                "Invalid avcC NAL length size: {}",
                nal_length_size
            )));
        }
        tracing::debug!("Parsed avcC: NAL length size {} bytes", nal_length_size);

        let mut offset = 5; // Skip version, profile, compatibility, level, NAL length size

        // Number of SPS (lower 5 bits)
        let num_sps = data[offset] & 0x1F;
        offset += 1;

        if num_sps == 0 {
            return Err(VideoError::DecodeFailed("No SPS in avcC".to_string()));
        }

        // Read first SPS
        if offset + 2 > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at SPS length".to_string(),
            ));
        }
        let sps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + sps_len > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at SPS data".to_string(),
            ));
        }
        let sps = data[offset..offset + sps_len].to_vec();
        offset += sps_len;

        // Skip remaining SPS if any
        for _ in 1..num_sps {
            if offset + 2 > data.len() {
                break;
            }
            let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2 + len;
        }

        // Number of PPS
        if offset >= data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at PPS count".to_string(),
            ));
        }
        let num_pps = data[offset];
        offset += 1;

        if num_pps == 0 {
            return Err(VideoError::DecodeFailed("No PPS in avcC".to_string()));
        }

        // Read first PPS
        if offset + 2 > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at PPS length".to_string(),
            ));
        }
        let pps_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;

        if offset + pps_len > data.len() {
            return Err(VideoError::DecodeFailed(
                "avcC truncated at PPS data".to_string(),
            ));
        }
        let pps = data[offset..offset + pps_len].to_vec();

        tracing::debug!(
            "Parsed avcC: SPS {} bytes, PPS {} bytes, NAL length size {} bytes",
            sps.len(),
            pps.len(),
            nal_length_size
        );
        Ok((sps, pps, nal_length_size))
    }

    /// Extracts SPS and PPS NAL units from H.264 Annex B bitstream.
    ///
    /// SPS NAL type = 7, PPS NAL type = 8
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn extract_h264_params(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>), VideoError> {
        let mut sps: Option<Vec<u8>> = None;
        let mut pps: Option<Vec<u8>> = None;

        let mut i = 0;
        while i < data.len() {
            // Find start code (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
            let start_code_len = if i + 4 <= data.len()
                && data[i] == 0
                && data[i + 1] == 0
                && data[i + 2] == 0
                && data[i + 3] == 1
            {
                4
            } else if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                3
            } else {
                i += 1;
                continue;
            };

            let nal_start = i + start_code_len;
            if nal_start >= data.len() {
                break;
            }

            // Get NAL unit type (lower 5 bits of first byte)
            let nal_type = data[nal_start] & 0x1F;

            // Find end of this NAL unit
            let mut nal_end = data.len();
            for j in nal_start + 1..data.len().saturating_sub(2) {
                if data[j] == 0 && data[j + 1] == 0 {
                    if j + 2 < data.len() && data[j + 2] == 1 {
                        nal_end = j;
                        break;
                    }
                    if j + 3 < data.len() && data[j + 2] == 0 && data[j + 3] == 1 {
                        nal_end = j;
                        break;
                    }
                }
            }

            let nal_data = &data[nal_start..nal_end];

            match nal_type {
                7 => {
                    // SPS
                    sps = Some(nal_data.to_vec());
                    tracing::debug!("Found SPS: {} bytes", nal_data.len());
                }
                8 => {
                    // PPS
                    pps = Some(nal_data.to_vec());
                    tracing::debug!("Found PPS: {} bytes", nal_data.len());
                }
                _ => {}
            }

            i = nal_end;
        }

        match (sps, pps) {
            (Some(s), Some(p)) => Ok((s, p)),
            (None, _) => Err(VideoError::DecodeFailed(
                "No SPS found in keyframe".to_string(),
            )),
            (_, None) => Err(VideoError::DecodeFailed(
                "No PPS found in keyframe".to_string(),
            )),
        }
    }
}

impl Drop for MoqDecoder {
    fn drop(&mut self) {
        tracing::debug!("MoQ: MoqDecoder dropped (frame_rx closing)");
    }
}

impl VideoDecoderBackend for MoqDecoder {
    fn open(url: &str) -> Result<Self, VideoError>
    where
        Self: Sized,
    {
        Self::new(url)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>, VideoError> {
        // Check if we've reached EOF
        if self.shared.eof_reached.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // Check for errors
        let state = *self.shared.state.lock();
        if state == MoqDecoderState::Error {
            return Err(VideoError::DecodeFailed(
                self.error_message()
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ));
        }

        // Sync cached metadata from shared state (safe copy under lock)
        {
            let shared_metadata = self.shared.metadata.lock();
            if shared_metadata.width != self.cached_metadata.width
                || shared_metadata.height != self.cached_metadata.height
            {
                self.cached_metadata = shared_metadata.clone();
            }
        }

        // Try to receive a frame (non-blocking)
        match self.frame_rx.try_recv() {
            Ok(moq_frame) => {
                // Track frame submitted to decoder
                self.shared
                    .frame_stats
                    .submitted_to_decoder
                    .fetch_add(1, Ordering::Relaxed);

                // Decode the frame
                match self.decode_frame(&moq_frame) {
                    Ok(frame) => {
                        // Track frame rendered
                        self.shared
                            .frame_stats
                            .rendered
                            .fetch_add(1, Ordering::Relaxed);
                        Ok(Some(frame))
                    }
                    Err(VideoError::DecodeFailed(msg))
                        if msg.contains("Waiting for keyframe")
                            || msg.contains("Waiting for IDR frame")
                            || msg.contains("Waiting for VT quiesce")
                            || msg.contains("DPB grace skip")
                            || msg.contains("no frame decoded") =>
                    {
                        // Track frames dropped waiting for IDR (but NOT DPB grace —
                        // those are already counted via dropped_dpb_grace)
                        if msg.contains("Waiting for IDR") {
                            self.shared
                                .frame_stats
                                .dropped_waiting_idr
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        // Not an error, just need to wait for keyframe or async decode
                        Ok(None)
                    }
                    Err(e) => {
                        // Track decode errors
                        self.shared
                            .frame_stats
                            .decode_errors
                            .fetch_add(1, Ordering::Relaxed);
                        Err(e)
                    }
                }
            }
            Err(async_channel::TryRecvError::Empty) => {
                // No frame available yet
                Ok(None)
            }
            Err(async_channel::TryRecvError::Closed) => {
                // Channel closed, stream ended
                tracing::info!(
                    "MoQ: frame_tx sender dropped (worker ended or shutdown), setting eof"
                );
                self.shared.eof_reached.store(true, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    fn seek(&mut self, _position: Duration) -> Result<(), VideoError> {
        // Live streams don't support seeking
        Err(VideoError::SeekFailed(
            "Seeking is not supported on live MoQ streams".to_string(),
        ))
    }

    fn metadata(&self) -> &VideoMetadata {
        // Return the locally cached metadata (safe, no lock needed)
        &self.cached_metadata
    }

    fn duration(&self) -> Option<Duration> {
        // Live streams have no duration
        None
    }

    fn is_eof(&self) -> bool {
        self.shared.eof_reached.load(Ordering::Relaxed)
    }

    fn buffering_percent(&self) -> i32 {
        self.shared.buffering_percent.load(Ordering::Relaxed)
    }

    fn hw_accel_type(&self) -> HwAccelType {
        self.active_hw_type
    }

    fn handles_audio_internally(&self) -> bool {
        self.shared
            .audio
            .internal_audio_ready
            .load(Ordering::Relaxed)
    }

    fn audio_handle(&self) -> Option<crate::media::audio::AudioHandle> {
        self.shared.audio.moq_audio_handle.lock().clone()
    }

    fn set_muted(&mut self, muted: bool) -> Result<(), VideoError> {
        self.audio_muted = muted;
        Ok(())
    }

    fn set_volume(&mut self, volume: f32) -> Result<(), VideoError> {
        self.audio_volume = volume.clamp(0.0, 1.0);
        Ok(())
    }
}
