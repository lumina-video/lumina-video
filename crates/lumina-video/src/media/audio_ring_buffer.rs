//! Lock-free SPSC ring buffer for continuous MoQ audio playback.
//!
//! Replaces per-source `sink.append(SamplesBuffer)` with a single continuous
//! `RingBufferSource` that never exhausts, eliminating rodio resampler state
//! resets and queue mutex contention that cause clicks/pops.
//!
//! Design: true SPSC — only the producer modifies `write_pos`, only the consumer
//! modifies `read_pos`. On overflow the producer overwrites old data (advancing
//! `write_pos` past capacity); the consumer detects the skip and catches up.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
    /// Creates config for the given audio format with default timing.
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
    buffer: Box<[f32]>,
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
}

/// Producer half of the ring buffer. Owned by the audio decode thread.
pub struct RingBufferProducer {
    shared: Arc<RingBufferShared>,
}

/// Consumer half of the ring buffer. Owned by the rodio Source.
pub struct RingBufferConsumer {
    shared: Arc<RingBufferShared>,
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
        buffer: vec![0.0f32; capacity].into_boxed_slice(),
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
    });

    (
        RingBufferProducer {
            shared: shared.clone(),
        },
        RingBufferConsumer { shared },
    )
}

impl RingBufferProducer {
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

        // Write samples into the circular buffer (may overwrite old data)
        // SAFETY: single producer, power-of-2 masking ensures valid indices.
        let ptr = s.buffer.as_ptr() as *mut f32;
        let start_idx = wp & mask;
        let first_chunk = (cap - start_idx).min(len);

        unsafe {
            std::ptr::copy_nonoverlapping(samples.as_ptr(), ptr.add(start_idx), first_chunk);
        }
        if first_chunk < len {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    samples.as_ptr().add(first_chunk),
                    ptr,
                    len - first_chunk,
                );
            }
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
    /// Returns `Some(sample)` if data is available, or `None` during initial
    /// prefill or if the buffer is empty.
    ///
    /// If the producer has overwritten our read position (consumer fell behind),
    /// we skip forward to the oldest valid data.
    pub fn read_sample(&self) -> Option<f32> {
        let s = &self.shared;

        // Initial stall: return None until prefill threshold is reached
        if !s.prefilled.load(Ordering::Acquire) {
            return None;
        }

        let mut rp = s.read_pos.load(Ordering::Relaxed);
        let wp = s.write_pos.load(Ordering::Acquire);

        // Buffer empty
        if rp == wp {
            s.underrun_count.fetch_add(1, Ordering::Relaxed);
            return None;
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
        let sample = s.buffer[idx];
        s.read_pos.store(rp.wrapping_add(1), Ordering::Release);
        s.total_read.fetch_add(1, Ordering::Relaxed);

        Some(sample)
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

// SAFETY: The ring buffer is designed for single-producer single-consumer use.
// Producer only modifies write_pos, consumer only modifies read_pos.
// Atomic positions with Acquire/Release ordering provide synchronization.
unsafe impl Send for RingBufferProducer {}
unsafe impl Send for RingBufferConsumer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_write_read() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        // Write some samples (>= prefill threshold of 4)
        producer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Read them back
        assert_eq!(consumer.read_sample(), Some(1.0));
        assert_eq!(consumer.read_sample(), Some(2.0));
        assert_eq!(consumer.read_sample(), Some(3.0));
        assert_eq!(consumer.read_sample(), Some(4.0));
        assert_eq!(consumer.read_sample(), Some(5.0));
        // Empty — returns None but does NOT re-enter stall mode
        assert_eq!(consumer.read_sample(), None);

        // Writing more should be immediately readable (no re-prefill needed)
        producer.write(&[6.0, 7.0]);
        assert_eq!(consumer.read_sample(), Some(6.0));
        assert_eq!(consumer.read_sample(), Some(7.0));
    }

    #[test]
    fn test_stall_mode() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 10,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        // Write less than prefill threshold
        producer.write(&[1.0, 2.0, 3.0]);

        // Consumer should stall (returns None)
        assert_eq!(consumer.read_sample(), None);

        // Write enough to reach prefill
        producer.write(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        // Now reads should succeed
        assert_eq!(consumer.read_sample(), Some(1.0));
        assert_eq!(consumer.read_sample(), Some(2.0));
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
        let (producer, consumer) = audio_ring_buffer(config);
        let cap = 1024; // power-of-2

        // Write enough for prefill
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        producer.write(&data);

        // Read a few to establish read position
        assert_eq!(consumer.read_sample(), Some(0.0));
        assert_eq!(consumer.read_sample(), Some(1.0));

        // Now write way more than capacity (lap the consumer)
        let big_data: Vec<f32> = (0..cap * 2).map(|i| (100 + i) as f32).collect();
        producer.write(&big_data);

        // Consumer should detect the lap and skip forward
        let sample = consumer.read_sample();
        assert!(sample.is_some(), "should get a sample after skip");
    }

    #[test]
    fn test_wraparound() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        // Write and read many rounds to force multiple wraparounds
        for round in 0..300 {
            let base = round as f32 * 4.0;
            producer.write(&[base, base + 1.0, base + 2.0, base + 3.0]);
            assert_eq!(consumer.read_sample(), Some(base));
            assert_eq!(consumer.read_sample(), Some(base + 1.0));
            assert_eq!(consumer.read_sample(), Some(base + 2.0));
            assert_eq!(consumer.read_sample(), Some(base + 3.0));
        }
    }

    #[test]
    fn test_producer_drop_signals_dead() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 2,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        producer.write(&[1.0, 2.0, 3.0]);
        assert!(consumer.is_producer_alive());

        drop(producer);
        assert!(!consumer.is_producer_alive());

        // Can still read remaining samples
        assert_eq!(consumer.read_sample(), Some(1.0));
    }

    #[test]
    fn test_metrics() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 4,
        };
        let (producer, consumer) = audio_ring_buffer(config);

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
            let mut read_count = 0u64;
            let mut last_val = -1.0f32;
            loop {
                match consumer.read_sample() {
                    Some(val) => {
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
                    None => {
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
}
