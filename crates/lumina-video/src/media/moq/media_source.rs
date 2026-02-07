//! Media source bridge for MoQ to FFmpeg integration.
//!
//! This module provides a reorder buffer and AVIO-compatible interface that bridges
//! async MoQ frame reception to FFmpeg's synchronous demuxer.

use super::error::MoqError;
use super::subscriber::MoqFrame;

use bytes::{Bytes, BytesMut};
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;

/// Size of the reorder buffer (number of frames to hold for reordering).
const REORDER_BUFFER_SIZE: usize = 64;

/// Size of the channel buffer between async receiver and sync reader.
const CHANNEL_BUFFER_SIZE: usize = 256;

/// Reorder buffer that reassembles out-of-order MoQ frames.
///
/// MoQ delivers frames potentially out of order due to QUIC stream multiplexing.
/// This buffer collects frames and emits them in the correct order.
pub struct ReorderBuffer {
    /// Frames waiting to be emitted, keyed by sequence number
    pending: BTreeMap<u64, Bytes>,
    /// Next expected sequence number
    next_sequence: u64,
    /// Maximum frames per group (for sequence calculation)
    frames_per_group: u64,
    /// Maximum buffer size before dropping old frames
    max_size: usize,
    /// Statistics: total frames received
    frames_received: u64,
    /// Statistics: frames dropped due to arriving too late
    frames_dropped: u64,
    /// Statistics: frames emitted in order
    frames_emitted: u64,
}

impl ReorderBuffer {
    /// Creates a new reorder buffer.
    pub fn new(frames_per_group: u64) -> Self {
        Self {
            pending: BTreeMap::new(),
            next_sequence: 0,
            frames_per_group,
            max_size: REORDER_BUFFER_SIZE,
            frames_received: 0,
            frames_dropped: 0,
            frames_emitted: 0,
        }
    }

    /// Calculates global sequence number from group and frame indices.
    pub fn calc_sequence(&self, group_sequence: u64, frame_index: usize) -> u64 {
        group_sequence * self.frames_per_group + frame_index as u64
    }

    /// Inserts a frame into the buffer.
    ///
    /// Returns frames that are now ready to be emitted in order.
    pub fn insert(&mut self, frame: MoqFrame) -> Vec<Bytes> {
        self.frames_received += 1;

        let sequence = self.calc_sequence(frame.group_sequence, frame.frame_index);

        // Check if frame is too old (already passed)
        if sequence < self.next_sequence {
            self.frames_dropped += 1;
            return Vec::new();
        }

        // Insert the frame
        self.pending.insert(sequence, frame.data);

        // If buffer is too large, drop oldest frames to prevent unbounded growth.
        // For live streaming, it's better to skip frames than run out of memory.
        while self.pending.len() > self.max_size {
            if let Some((&oldest_seq, _)) = self.pending.first_key_value() {
                self.pending.remove(&oldest_seq);
                self.frames_dropped += 1;
                // If we're dropping frames we haven't emitted yet, advance next_sequence
                // to skip the gap and continue from what we have
                if oldest_seq >= self.next_sequence {
                    self.next_sequence = oldest_seq + 1;
                }
            } else {
                break;
            }
        }

        // Emit frames that are now in order
        self.emit_ready()
    }

    /// Emits all frames that are ready (in sequence order).
    fn emit_ready(&mut self) -> Vec<Bytes> {
        let mut ready = Vec::new();

        while let Some(data) = self.pending.remove(&self.next_sequence) {
            ready.push(data);
            self.next_sequence += 1;
            self.frames_emitted += 1;
        }

        ready
    }

    /// Flushes remaining frames, even if there are gaps.
    ///
    /// Use this when the stream ends to get any remaining buffered data.
    pub fn flush(&mut self) -> Vec<Bytes> {
        let mut flushed = Vec::new();

        // Get all remaining frames in order
        while let Some((&seq, _)) = self.pending.first_key_value() {
            if let Some(data) = self.pending.remove(&seq) {
                flushed.push(data);
                self.frames_emitted += 1;
            }
            // Update next_sequence to skip any gaps
            self.next_sequence = seq + 1;
        }

        flushed
    }

    /// Returns statistics about the buffer.
    pub fn stats(&self) -> ReorderBufferStats {
        ReorderBufferStats {
            frames_received: self.frames_received,
            frames_dropped: self.frames_dropped,
            frames_emitted: self.frames_emitted,
            pending_count: self.pending.len(),
            next_sequence: self.next_sequence,
        }
    }

    /// Resets the buffer state.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.next_sequence = 0;
        self.frames_received = 0;
        self.frames_dropped = 0;
        self.frames_emitted = 0;
    }
}

/// Statistics for the reorder buffer.
#[derive(Debug, Clone, Copy)]
pub struct ReorderBufferStats {
    /// Total frames inserted into the buffer.
    pub frames_received: u64,
    /// Frames dropped (arrived too late or evicted when buffer was full).
    pub frames_dropped: u64,
    /// Frames emitted in correct sequence order.
    pub frames_emitted: u64,
    /// Frames currently waiting in the buffer for missing predecessors.
    pub pending_count: usize,
    /// Next sequence number expected for in-order emission.
    pub next_sequence: u64,
}

/// Media source that provides a Read interface for FFmpeg.
///
/// This bridges the async MoQ frame receiver to FFmpeg's synchronous AVIO
/// read callbacks. It uses a channel to decouple the async receiver thread
/// from the sync FFmpeg demuxer thread.
pub struct MoqMediaSource {
    /// Receiver for ordered frame data
    receiver: Receiver<Bytes>,
    /// Current buffer being read from
    current_buffer: BytesMut,
    /// Whether the source has ended
    ended: Arc<AtomicBool>,
    /// Total bytes read
    bytes_read: AtomicU64,
    /// Error flag
    had_error: Arc<AtomicBool>,
}

/// Sender side of the media source for the async receive loop.
pub struct MoqMediaSourceWriter {
    /// Sender for frame data (Option so we can drop it to close the channel)
    sender: Mutex<Option<Sender<Bytes>>>,
    /// Reorder buffer
    reorder_buffer: Mutex<ReorderBuffer>,
    /// Whether the source has ended
    ended: Arc<AtomicBool>,
    /// Error flag
    had_error: Arc<AtomicBool>,
}

impl MoqMediaSource {
    /// Creates a new media source pair (reader, writer).
    pub fn new(frames_per_group: u64) -> (Self, MoqMediaSourceWriter) {
        let (sender, receiver) = bounded(CHANNEL_BUFFER_SIZE);
        let ended = Arc::new(AtomicBool::new(false));
        let had_error = Arc::new(AtomicBool::new(false));

        let reader = MoqMediaSource {
            receiver,
            current_buffer: BytesMut::new(),
            ended: ended.clone(),
            bytes_read: AtomicU64::new(0),
            had_error: had_error.clone(),
        };

        let writer = MoqMediaSourceWriter {
            sender: Mutex::new(Some(sender)),
            reorder_buffer: Mutex::new(ReorderBuffer::new(frames_per_group)),
            ended,
            had_error,
        };

        (reader, writer)
    }

    /// Returns the total bytes read.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Returns whether the source has ended.
    pub fn is_ended(&self) -> bool {
        self.ended.load(Ordering::Relaxed)
    }

    /// Returns whether there was an error.
    pub fn had_error(&self) -> bool {
        self.had_error.load(Ordering::Relaxed)
    }
}

impl Read for MoqMediaSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If we have data in the current buffer, use it
        if !self.current_buffer.is_empty() {
            let to_copy = std::cmp::min(buf.len(), self.current_buffer.len());
            buf[..to_copy].copy_from_slice(&self.current_buffer[..to_copy]);
            let _ = self.current_buffer.split_to(to_copy);
            self.bytes_read.fetch_add(to_copy as u64, Ordering::Relaxed);
            return Ok(to_copy);
        }

        // Check for error
        if self.had_error.load(Ordering::Relaxed) {
            return Err(io::Error::other("MoQ stream error"));
        }

        // Try to receive more data using try_recv first to drain any remaining data
        // before checking the ended flag (data may have been sent before end() was called)
        match self.receiver.try_recv() {
            Ok(data) => {
                // Copy as much as we can to the output buffer
                let to_copy = std::cmp::min(buf.len(), data.len());
                buf[..to_copy].copy_from_slice(&data[..to_copy]);

                // Store the rest in current_buffer
                if to_copy < data.len() {
                    self.current_buffer.extend_from_slice(&data[to_copy..]);
                }

                self.bytes_read.fetch_add(to_copy as u64, Ordering::Relaxed);
                Ok(to_copy)
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {
                // No data available, check if ended
                if self.ended.load(Ordering::Relaxed) {
                    return Ok(0); // EOF
                }
                // Block waiting for data
                match self.receiver.recv() {
                    Ok(data) => {
                        let to_copy = std::cmp::min(buf.len(), data.len());
                        buf[..to_copy].copy_from_slice(&data[..to_copy]);
                        if to_copy < data.len() {
                            self.current_buffer.extend_from_slice(&data[to_copy..]);
                        }
                        self.bytes_read.fetch_add(to_copy as u64, Ordering::Relaxed);
                        Ok(to_copy)
                    }
                    Err(_) => {
                        // Channel disconnected, treat as EOF
                        self.ended.store(true, Ordering::Relaxed);
                        Ok(0)
                    }
                }
            }
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                // Channel closed, treat as EOF
                self.ended.store(true, Ordering::Relaxed);
                Ok(0)
            }
        }
    }
}

impl MoqMediaSourceWriter {
    /// Writes a frame to the media source.
    ///
    /// The frame is added to the reorder buffer and any ready frames
    /// are sent to the reader.
    pub fn write_frame(&self, frame: MoqFrame) -> Result<(), MoqError> {
        if self.ended.load(Ordering::Relaxed) {
            return Err(MoqError::StreamClosed("Writer closed".to_string()));
        }

        let mut buffer = self.reorder_buffer.lock();
        let ready_frames = buffer.insert(frame);

        let sender_guard = self.sender.lock();
        let sender = sender_guard
            .as_ref()
            .ok_or_else(|| MoqError::StreamClosed("Writer closed".to_string()))?;

        for data in ready_frames {
            if sender.send(data).is_err() {
                drop(sender_guard);
                self.ended.store(true, Ordering::Relaxed);
                return Err(MoqError::ChannelError("Reader disconnected".to_string()));
            }
        }

        Ok(())
    }

    /// Flushes any remaining buffered frames.
    pub fn flush(&self) -> Result<(), MoqError> {
        let mut buffer = self.reorder_buffer.lock();
        let remaining = buffer.flush();

        let sender_guard = self.sender.lock();
        let sender = match sender_guard.as_ref() {
            Some(s) => s,
            None => return Ok(()), // Already closed
        };

        for data in remaining {
            if sender.send(data).is_err() {
                return Err(MoqError::ChannelError("Reader disconnected".to_string()));
            }
        }

        Ok(())
    }

    /// Marks the source as ended (no more data coming).
    ///
    /// This flushes remaining data and closes the channel, which will
    /// unblock any pending `recv()` calls on the reader side.
    pub fn end(&self) {
        // Flush any remaining buffered data
        let _ = self.flush();
        self.ended.store(true, Ordering::Relaxed);
        // Drop the sender to close the channel and unblock readers
        let _ = self.sender.lock().take();
    }

    /// Marks the source as having an error.
    pub fn set_error(&self) {
        self.had_error.store(true, Ordering::Relaxed);
        self.ended.store(true, Ordering::Relaxed);
    }

    /// Returns reorder buffer statistics.
    pub fn stats(&self) -> ReorderBufferStats {
        self.reorder_buffer.lock().stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reorder_buffer_in_order() {
        let mut buffer = ReorderBuffer::new(100);

        // Insert frames in order
        let frame0 = MoqFrame {
            group_sequence: 0,
            frame_index: 0,
            data: Bytes::from("frame0"),
            timestamp_ms: None,
        };
        let ready = buffer.insert(frame0);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready.first(), Some(&Bytes::from("frame0")));

        let frame1 = MoqFrame {
            group_sequence: 0,
            frame_index: 1,
            data: Bytes::from("frame1"),
            timestamp_ms: None,
        };
        let ready = buffer.insert(frame1);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready.first(), Some(&Bytes::from("frame1")));
    }

    #[test]
    fn test_reorder_buffer_out_of_order() {
        let mut buffer = ReorderBuffer::new(100);

        // Insert frame 1 first
        let frame1 = MoqFrame {
            group_sequence: 0,
            frame_index: 1,
            data: Bytes::from("frame1"),
            timestamp_ms: None,
        };
        let ready = buffer.insert(frame1);
        assert_eq!(ready.len(), 0); // Not ready yet, waiting for frame 0

        // Now insert frame 0
        let frame0 = MoqFrame {
            group_sequence: 0,
            frame_index: 0,
            data: Bytes::from("frame0"),
            timestamp_ms: None,
        };
        let ready = buffer.insert(frame0);
        assert_eq!(ready.len(), 2); // Both frames now ready
        assert_eq!(ready.first(), Some(&Bytes::from("frame0")));
        assert_eq!(ready.get(1), Some(&Bytes::from("frame1")));
    }

    #[test]
    fn test_reorder_buffer_late_frame() {
        let mut buffer = ReorderBuffer::new(100);

        // Process frames 0, 1, 2 in order
        for i in 0..3 {
            let frame = MoqFrame {
                group_sequence: 0,
                frame_index: i,
                data: Bytes::from(format!("frame{i}")),
                timestamp_ms: None,
            };
            buffer.insert(frame);
        }

        // Now try to insert frame 0 again (late)
        let late_frame = MoqFrame {
            group_sequence: 0,
            frame_index: 0,
            data: Bytes::from("late"),
            timestamp_ms: None,
        };
        let ready = buffer.insert(late_frame);
        assert_eq!(ready.len(), 0); // Frame dropped
        assert_eq!(buffer.stats().frames_dropped, 1);
    }

    #[test]
    fn test_media_source_read() {
        let (mut reader, writer) = MoqMediaSource::new(100);

        // Write a frame
        let frame = MoqFrame {
            group_sequence: 0,
            frame_index: 0,
            data: Bytes::from("hello world"),
            timestamp_ms: None,
        };
        writer.write_frame(frame).unwrap();
        writer.end();

        // Read the data
        let mut buf = [0u8; 64];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hello world");

        // Should get EOF now
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_media_source_partial_read() {
        let (mut reader, writer) = MoqMediaSource::new(100);

        // Write a frame
        let frame = MoqFrame {
            group_sequence: 0,
            frame_index: 0,
            data: Bytes::from("hello world"),
            timestamp_ms: None,
        };
        writer.write_frame(frame).unwrap();
        writer.end();

        // Read in small chunks
        let mut buf = [0u8; 5];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hello");

        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b" worl");

        let n = reader.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"d");
    }
}
