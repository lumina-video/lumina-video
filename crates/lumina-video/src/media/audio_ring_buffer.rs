//! Lock-free SPSC ring buffer for continuous audio playback.
//!
//! Used by both MoQ live streaming and FFmpeg VOD paths. Provides a single
//! continuous sample stream to the cpal audio callback, eliminating per-source
//! transitions and queue mutex contention.
//!
//! Design: true SPSC — only the producer modifies `write_pos`, only the consumer
//! modifies `read_pos`. On overflow the producer overwrites old data (advancing
//! `write_pos` past capacity); the consumer detects the skip and catches up.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for the audio ring buffer.
#[derive(Debug, Clone)]
pub struct RingBufferConfig {
    /// Total capacity in samples (interleaved). Default: 2000ms at 48kHz stereo.
    pub capacity_samples: usize,
    /// Prefill threshold in samples before playback starts. Default: 500ms.
    pub prefill_samples: usize,
}

impl RingBufferConfig {
    /// Creates config tuned for VOD playback (shorter prefill for fast seek recovery).
    pub fn for_vod(sample_rate: u32, channels: u16) -> Self {
        let sps = sample_rate as usize * channels as usize;
        Self {
            capacity_samples: sps / 2,   // 500ms
            prefill_samples: sps / 20,   // 50ms — fast seek recovery
        }
    }

    /// Creates config for the given audio format with default timing (MoQ live).
    pub fn for_format(sample_rate: u32, channels: u16) -> Self {
        let samples_per_sec = sample_rate as usize * channels as usize;
        Self {
            // 2000ms buffer — large enough to absorb MoQ burst jitter
            capacity_samples: samples_per_sec * 2,
            // 500ms prefill — build solid buffer before starting playback
            prefill_samples: samples_per_sec / 2,
        }
    }
}

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self::for_format(48000, 2)
    }
}

/// Shared state between producer and consumer.
struct RingBufferShared {
    /// The sample buffer (fixed size, power-of-2 for fast modulo).
    /// Atomic sample slots (f32 stored as bits) avoid read/write data races.
    buffer: Box<[AtomicU32]>,
    /// Write position (monotonically increasing, never wraps — use mask for index).
    /// Only modified by producer.
    write_pos: AtomicUsize,
    /// Read position (monotonically increasing, never wraps — use mask for index).
    /// Only modified by consumer.
    read_pos: AtomicUsize,
    /// Capacity mask (capacity - 1, for power-of-2 modulo).
    mask: usize,
    /// Actual capacity in samples (power-of-2).
    capacity: usize,
    /// Whether the buffer has reached the prefill threshold at least once.
    prefilled: AtomicBool,
    /// Prefill threshold in samples.
    prefill_threshold: usize,
    /// Total samples written (for metrics).
    total_written: AtomicU64,
    /// Total samples read (for metrics).
    total_read: AtomicU64,
    /// Number of overflows (oldest samples overwritten).
    overflow_count: AtomicU64,
    /// Number of underrun events (consumer found buffer empty after prefill).
    underrun_count: AtomicU64,
    /// Whether the producer is still alive.
    alive: AtomicBool,
    /// Flush generation counter — incremented by producer on seek/flush.
    flush_generation: AtomicU64,
}

/// Result of reading a sample from the ring buffer consumer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReadSample {
    /// A valid audio sample.
    Sample(f32),
    /// The producer requested a flush (seek). Consumer has snapped to write position.
    Flushed,
    /// No data available (prefill not reached or buffer empty).
    Empty,
}

/// Producer half of the ring buffer. Owned by the audio decode thread.
pub struct RingBufferProducer {
    shared: Arc<RingBufferShared>,
}

/// Consumer half of the ring buffer. Owned by the cpal audio callback.
pub struct RingBufferConsumer {
    shared: Arc<RingBufferShared>,
    /// Tracks the last observed flush generation.
    consumer_generation: u64,
}

/// Metrics snapshot for observability.
#[derive(Debug, Clone, Default)]
pub struct RingBufferMetrics {
    /// Current fill level in samples.
    pub fill_samples: usize,
    /// Buffer capacity in samples.
    pub capacity_samples: usize,
    /// Total samples written by producer.
    pub total_written: u64,
    /// Total samples read by consumer.
    pub total_read: u64,
    /// Number of overflow events (old data overwritten).
    pub overflow_count: u64,
    /// Number of underrun events (buffer empty after prefill).
    pub stall_count: u64,
    /// Whether producer is still alive.
    pub producer_alive: bool,
}

/// Round up to next power of 2.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    n.next_power_of_two()
}

/// Creates a new ring buffer split into producer and consumer halves.
pub fn audio_ring_buffer(config: RingBufferConfig) -> (RingBufferProducer, RingBufferConsumer) {
    // Round capacity up to power-of-2 for fast masking (avoids modulo)
    let capacity = next_power_of_two(config.capacity_samples.max(1024));
    let mask = capacity - 1;

    let shared = Arc::new(RingBufferShared {
        buffer: (0..capacity)
            .map(|_| AtomicU32::new(0.0f32.to_bits()))
            .collect::<Vec<_>>()
            .into_boxed_slice(),
        write_pos: AtomicUsize::new(0),
        read_pos: AtomicUsize::new(0),
        mask,
        capacity,
        prefilled: AtomicBool::new(false),
        prefill_threshold: config.prefill_samples.min(capacity / 2),
        total_written: AtomicU64::new(0),
        total_read: AtomicU64::new(0),
        overflow_count: AtomicU64::new(0),
        underrun_count: AtomicU64::new(0),
        alive: AtomicBool::new(true),
        flush_generation: AtomicU64::new(0),
    });

    (
        RingBufferProducer {
            shared: shared.clone(),
        },
        RingBufferConsumer {
            shared,
            consumer_generation: 0,
        },
    )
}

impl RingBufferProducer {
    /// Requests a flush (for seek). The consumer will snap its read position
    /// to the current write position on the next `read_sample()` call, discarding
    /// all buffered data. Prefill is reset so new data must accumulate before
    /// playback resumes.
    pub fn request_flush(&self) {
        let s = &self.shared;
        s.prefilled.store(false, Ordering::Relaxed);
        // Generation increment is the Release point — consumer's Acquire sees prefilled=false
        s.flush_generation.fetch_add(1, Ordering::Release);
    }

    /// Writes samples into the ring buffer.
    ///
    /// Always succeeds. If the buffer is full, old data is overwritten
    /// (the consumer detects the skip and catches up). The producer never
    /// touches `read_pos` — true SPSC guarantee.
    pub fn write(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }

        let s = &self.shared;
        let mask = s.mask;
        let cap = s.capacity;
        let len = samples.len();

        // Snapshot write position (we are the only writer)
        let wp = s.write_pos.load(Ordering::Relaxed);
        let rp = s.read_pos.load(Ordering::Acquire);

        // Check for overflow (write would pass consumer)
        let fill_after_write = wp.wrapping_add(len).wrapping_sub(rp);
        if fill_after_write > cap {
            s.overflow_count.fetch_add(1, Ordering::Relaxed);
        }

        // Atomic slot writes avoid undefined behavior if producer and consumer
        // touch the same slot concurrently during overwrite scenarios.
        let mut idx = wp & mask;
        for &sample in samples {
            s.buffer[idx].store(sample.to_bits(), Ordering::Relaxed);
            idx = (idx + 1) & mask;
        }

        // Advance write position (makes samples visible to consumer)
        s.write_pos.store(wp.wrapping_add(len), Ordering::Release);

        s.total_written.fetch_add(len as u64, Ordering::Relaxed);

        // Check prefill threshold
        if !s.prefilled.load(Ordering::Relaxed) {
            let new_fill = s
                .write_pos
                .load(Ordering::Relaxed)
                .wrapping_sub(s.read_pos.load(Ordering::Relaxed));
            if new_fill >= s.prefill_threshold {
                s.prefilled.store(true, Ordering::Release);
                tracing::debug!(
                    "Ring buffer prefilled: {} samples (threshold: {})",
                    new_fill,
                    s.prefill_threshold
                );
            }
        }
    }

    /// Returns current metrics for observability.
    pub fn metrics(&self) -> RingBufferMetrics {
        let s = &self.shared;
        let wp = s.write_pos.load(Ordering::Relaxed);
        let rp = s.read_pos.load(Ordering::Relaxed);
        let fill = wp.wrapping_sub(rp).min(s.capacity);
        RingBufferMetrics {
            fill_samples: fill,
            capacity_samples: s.capacity,
            total_written: s.total_written.load(Ordering::Relaxed),
            total_read: s.total_read.load(Ordering::Relaxed),
            overflow_count: s.overflow_count.load(Ordering::Relaxed),
            stall_count: s.underrun_count.load(Ordering::Relaxed),
            producer_alive: s.alive.load(Ordering::Relaxed),
        }
    }
}

impl Drop for RingBufferProducer {
    fn drop(&mut self) {
        self.shared.alive.store(false, Ordering::Release);
    }
}

#[allow(dead_code)]
impl RingBufferConsumer {
    /// Reads a single sample from the ring buffer.
    ///
    /// Returns `ReadSample::Sample(f32)` if data is available,
    /// `ReadSample::Flushed` if the producer requested a flush (seek),
    /// or `ReadSample::Empty` during initial prefill or buffer empty.
    ///
    /// If the producer has overwritten our read position (consumer fell behind),
    /// we skip forward to the oldest valid data.
    pub fn read_sample(&mut self) -> ReadSample {
        let s = &self.shared;

        // Check flush BEFORE prefill gate (flush clears prefilled)
        let gen = s.flush_generation.load(Ordering::Acquire);
        if gen != self.consumer_generation {
            self.consumer_generation = gen;
            // Snap read_pos to write_pos (discard all buffered data)
            let wp = s.write_pos.load(Ordering::Acquire);
            s.read_pos.store(wp, Ordering::Release);
            return ReadSample::Flushed;
        }

        // Initial stall: return Empty until prefill threshold is reached
        if !s.prefilled.load(Ordering::Acquire) {
            return ReadSample::Empty;
        }

        let mut rp = s.read_pos.load(Ordering::Relaxed);
        let wp = s.write_pos.load(Ordering::Acquire);

        // Buffer empty
        if rp == wp {
            s.underrun_count.fetch_add(1, Ordering::Relaxed);
            return ReadSample::Empty;
        }

        // Check if producer has lapped us (overflow overwrote our data).
        // If so, skip forward to the oldest valid data (wp - capacity/2,
        // leaving half the buffer for smooth playback).
        let fill = wp.wrapping_sub(rp);
        if fill > s.capacity {
            // We fell behind — skip to mid-buffer for some headroom
            rp = wp.wrapping_sub(s.capacity / 2);
            s.read_pos.store(rp, Ordering::Relaxed);
        }

        let idx = rp & s.mask;
        let sample = f32::from_bits(s.buffer[idx].load(Ordering::Relaxed));
        s.read_pos.store(rp.wrapping_add(1), Ordering::Release);
        s.total_read.fetch_add(1, Ordering::Relaxed);

        ReadSample::Sample(sample)
    }

    /// Returns current metrics for observability.
    pub fn metrics(&self) -> RingBufferMetrics {
        let s = &self.shared;
        let wp = s.write_pos.load(Ordering::Relaxed);
        let rp = s.read_pos.load(Ordering::Relaxed);
        let fill = wp.wrapping_sub(rp).min(s.capacity);
        RingBufferMetrics {
            fill_samples: fill,
            capacity_samples: s.capacity,
            total_written: s.total_written.load(Ordering::Relaxed),
            total_read: s.total_read.load(Ordering::Relaxed),
            overflow_count: s.overflow_count.load(Ordering::Relaxed),
            stall_count: s.underrun_count.load(Ordering::Relaxed),
            producer_alive: s.alive.load(Ordering::Relaxed),
        }
    }

    /// Returns whether the producer is still alive.
    pub fn is_producer_alive(&self) -> bool {
        self.shared.alive.load(Ordering::Relaxed)
    }
}

// Auto-traits are sufficient now — all fields are Send+Sync (atomics + Arc).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_write_read() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        // Write some samples (>= prefill threshold of 4)
        producer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Read them back
        assert_eq!(consumer.read_sample(), ReadSample::Sample(1.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(2.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(3.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(4.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(5.0));
        // Empty — returns Empty but does NOT re-enter stall mode
        assert_eq!(consumer.read_sample(), ReadSample::Empty);

        // Writing more should be immediately readable (no re-prefill needed)
        producer.write(&[6.0, 7.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(6.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(7.0));
    }

    #[test]
    fn test_stall_mode() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 10,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        // Write less than prefill threshold
        producer.write(&[1.0, 2.0, 3.0]);

        // Consumer should stall (returns Empty)
        assert_eq!(consumer.read_sample(), ReadSample::Empty);

        // Write enough to reach prefill
        producer.write(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // Now reads should succeed
        assert_eq!(consumer.read_sample(), ReadSample::Sample(1.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(2.0));
    }

    #[test]
    fn test_overflow_overwrites_old() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 1,
        };
        let (producer, _consumer) = audio_ring_buffer(config);
        // Actual capacity is next_power_of_two(1024) = 1024

        // Fill the buffer
        let fill_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        producer.write(&fill_data);

        // Write 100 more — should trigger overflow
        let overflow_data: Vec<f32> = (1024..1124).map(|i| i as f32).collect();
        producer.write(&overflow_data);

        let metrics = producer.metrics();
        assert!(
            metrics.overflow_count > 0,
            "expected overflow, got count={}",
            metrics.overflow_count
        );
    }

    #[test]
    fn test_consumer_skip_on_lap() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);
        let cap = 1024; // power-of-2

        // Write enough for prefill
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        producer.write(&data);

        // Read a few to establish read position
        assert_eq!(consumer.read_sample(), ReadSample::Sample(0.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(1.0));

        // Now write way more than capacity (lap the consumer)
        let big_data: Vec<f32> = (0..cap * 2).map(|i| (100 + i) as f32).collect();
        producer.write(&big_data);

        // Consumer should detect the lap and skip forward
        let sample = consumer.read_sample();
        assert!(
            matches!(sample, ReadSample::Sample(_)),
            "should get a sample after skip"
        );
    }

    #[test]
    fn test_wraparound() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        // Write and read many rounds to force multiple wraparounds
        for round in 0..300 {
            let base = round as f32 * 4.0;
            producer.write(&[base, base + 1.0, base + 2.0, base + 3.0]);
            assert_eq!(consumer.read_sample(), ReadSample::Sample(base));
            assert_eq!(consumer.read_sample(), ReadSample::Sample(base + 1.0));
            assert_eq!(consumer.read_sample(), ReadSample::Sample(base + 2.0));
            assert_eq!(consumer.read_sample(), ReadSample::Sample(base + 3.0));
        }
    }

    #[test]
    fn test_producer_drop_signals_dead() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 2,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        producer.write(&[1.0, 2.0, 3.0]);
        assert!(consumer.is_producer_alive());

        drop(producer);
        assert!(!consumer.is_producer_alive());

        // Can still read remaining samples
        assert_eq!(consumer.read_sample(), ReadSample::Sample(1.0));
    }

    #[test]
    fn test_metrics() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        producer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let pm = producer.metrics();
        assert_eq!(pm.total_written, 5);
        assert_eq!(pm.fill_samples, 5);

        consumer.read_sample();
        consumer.read_sample();

        let cm = consumer.metrics();
        assert_eq!(cm.total_read, 2);
        assert_eq!(cm.fill_samples, 3);
    }

    #[test]
    fn test_concurrent_write_read() {
        use std::thread;

        let config = RingBufferConfig {
            capacity_samples: 4096,
            prefill_samples: 100,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        let write_count = 10_000usize;

        let writer = thread::spawn(move || {
            for i in 0..write_count {
                producer.write(&[i as f32]);
                if i % 100 == 0 {
                    thread::yield_now();
                }
            }
            drop(producer);
        });

        let reader = thread::spawn(move || {
            let mut consumer = consumer;
            let mut read_count = 0u64;
            let mut last_val = -1.0f32;
            loop {
                match consumer.read_sample() {
                    ReadSample::Sample(val) => {
                        // Values should be monotonically increasing
                        // (may skip values due to overflow, but never go backwards)
                        assert!(
                            val >= last_val,
                            "went backwards: {} after {}",
                            val,
                            last_val
                        );
                        last_val = val;
                        read_count += 1;
                    }
                    ReadSample::Empty | ReadSample::Flushed => {
                        if !consumer.is_producer_alive() {
                            break;
                        }
                        thread::yield_now();
                    }
                }
            }
            read_count
        });

        writer.join().unwrap();
        let read_count = reader.join().unwrap();

        assert!(
            read_count > 0,
            "should have read some samples, got {}",
            read_count
        );
    }

    #[test]
    fn test_flush_basic() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        // Write and read some data
        producer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(1.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(2.0));

        // Request flush
        producer.request_flush();

        // Next read should return Flushed
        assert_eq!(consumer.read_sample(), ReadSample::Flushed);

        // After flush, buffer is empty (prefill reset) — should return Empty
        assert_eq!(consumer.read_sample(), ReadSample::Empty);

        // Write new data past prefill threshold, should be readable
        producer.write(&[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(10.0));
        assert_eq!(consumer.read_sample(), ReadSample::Sample(11.0));
    }

    #[test]
    fn test_flush_resets_prefill() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 10,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        // Fill past prefill threshold
        let data: Vec<f32> = (0..15).map(|i| i as f32).collect();
        producer.write(&data);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(0.0));

        // Flush resets prefill
        producer.request_flush();
        assert_eq!(consumer.read_sample(), ReadSample::Flushed);

        // Write less than prefill — should still be in stall mode
        producer.write(&[100.0, 101.0, 102.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Empty);

        // Write enough to reach prefill
        producer.write(&[103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(100.0));
    }

    #[test]
    fn test_multiple_rapid_flushes() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, mut consumer) = audio_ring_buffer(config);

        producer.write(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(1.0));

        // Multiple rapid flushes — consumer should land on latest generation
        producer.request_flush();
        producer.request_flush();
        producer.request_flush();

        // Only one Flushed signal should appear (consumer catches up to latest gen)
        assert_eq!(consumer.read_sample(), ReadSample::Flushed);
        assert_eq!(consumer.read_sample(), ReadSample::Empty);

        // New data works fine
        producer.write(&[50.0, 51.0, 52.0, 53.0]);
        assert_eq!(consumer.read_sample(), ReadSample::Sample(50.0));
    }
}
