//! Lock-free SPSC ring buffer for continuous MoQ audio playback.
//!
//! Replaces per-source `sink.append(SamplesBuffer)` with a single continuous
//! `RingBufferSource` that never exhausts, eliminating rodio resampler state
//! resets and queue mutex contention that cause clicks/pops.
//!
//! Modeled after moq-dev/moq's `AudioRingBuffer` (browser AudioWorklet pattern).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Configuration for the audio ring buffer.
#[derive(Debug, Clone)]
pub struct RingBufferConfig {
    /// Total capacity in samples (interleaved). Default: 500ms at 48kHz stereo = 48000.
    pub capacity_samples: usize,
    /// Prefill threshold in samples before playback starts. Default: 240ms = 23040.
    pub prefill_samples: usize,
}

impl RingBufferConfig {
    /// Creates config for the given audio format with default timing.
    pub fn for_format(sample_rate: u32, channels: u16) -> Self {
        let samples_per_sec = sample_rate as usize * channels as usize;
        Self {
            // 500ms buffer
            capacity_samples: samples_per_sec / 2,
            // 240ms prefill
            prefill_samples: samples_per_sec * 240 / 1000,
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
    /// The sample buffer (fixed size, never reallocated).
    buffer: Box<[f32]>,
    /// Write position (monotonically increasing, wraps via modulo).
    write_pos: AtomicUsize,
    /// Read position (monotonically increasing, wraps via modulo).
    read_pos: AtomicUsize,
    /// Capacity in samples.
    capacity: usize,
    /// Whether the buffer has reached the prefill threshold at least once.
    prefilled: AtomicBool,
    /// Prefill threshold in samples.
    prefill_threshold: usize,
    /// Total samples written (for metrics).
    total_written: AtomicU64,
    /// Total samples read (for metrics).
    total_read: AtomicU64,
    /// Number of overflows (oldest samples dropped).
    overflow_count: AtomicU64,
    /// Number of stall events (consumer returned silence waiting for prefill).
    stall_count: AtomicU64,
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
    /// Number of overflow events (dropped oldest).
    pub overflow_count: u64,
    /// Number of stall events (silence while prefilling).
    pub stall_count: u64,
    /// Whether producer is still alive.
    pub producer_alive: bool,
}

/// Creates a new ring buffer split into producer and consumer halves.
pub fn audio_ring_buffer(config: RingBufferConfig) -> (RingBufferProducer, RingBufferConsumer) {
    let capacity = config.capacity_samples.max(1024); // Minimum 1024 samples
    let shared = Arc::new(RingBufferShared {
        buffer: vec![0.0f32; capacity].into_boxed_slice(),
        write_pos: AtomicUsize::new(0),
        read_pos: AtomicUsize::new(0),
        capacity,
        prefilled: AtomicBool::new(false),
        prefill_threshold: config.prefill_samples.min(capacity),
        total_written: AtomicU64::new(0),
        total_read: AtomicU64::new(0),
        overflow_count: AtomicU64::new(0),
        stall_count: AtomicU64::new(0),
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
    /// On overflow (buffer full), advances the read pointer to drop oldest samples
    /// (live-edge policy — stay near real-time, never block).
    pub fn write(&self, samples: &[f32]) {
        let s = &self.shared;
        let cap = s.capacity;

        for &sample in samples {
            let wp = s.write_pos.load(Ordering::Relaxed);
            let rp = s.read_pos.load(Ordering::Relaxed);

            // Check if buffer is full
            let next_wp = wp.wrapping_add(1);
            if next_wp.wrapping_sub(rp) > cap {
                // Overflow: advance read pointer (drop oldest sample)
                s.read_pos.store(rp.wrapping_add(1), Ordering::Release);
                s.overflow_count.fetch_add(1, Ordering::Relaxed);
            }

            // Write sample
            // SAFETY: We use modulo to index within bounds. The buffer is fixed-size
            // and we're the only writer (SPSC guarantee).
            let idx = wp % cap;
            // Safety: single producer, so no concurrent writes to the same index.
            // The consumer only reads indices behind read_pos, which we've already passed.
            unsafe {
                let ptr = s.buffer.as_ptr() as *mut f32;
                ptr.add(idx).write(sample);
            }
            s.write_pos.store(next_wp, Ordering::Release);
        }

        s.total_written
            .fetch_add(samples.len() as u64, Ordering::Relaxed);

        // Check prefill threshold
        if !s.prefilled.load(Ordering::Relaxed) {
            let wp = s.write_pos.load(Ordering::Relaxed);
            let rp = s.read_pos.load(Ordering::Relaxed);
            let fill = wp.wrapping_sub(rp);
            if fill >= s.prefill_threshold {
                s.prefilled.store(true, Ordering::Release);
                tracing::debug!(
                    "Ring buffer prefilled: {} samples (threshold: {})",
                    fill,
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
        RingBufferMetrics {
            fill_samples: wp.wrapping_sub(rp),
            capacity_samples: s.capacity,
            total_written: s.total_written.load(Ordering::Relaxed),
            total_read: s.total_read.load(Ordering::Relaxed),
            overflow_count: s.overflow_count.load(Ordering::Relaxed),
            stall_count: s.stall_count.load(Ordering::Relaxed),
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
    /// Returns `Some(sample)` if data is available, or `None` if in stall mode
    /// (prefill not yet reached) or buffer is empty.
    pub fn read_sample(&self) -> Option<f32> {
        let s = &self.shared;

        // Stall mode: return None until prefill threshold is reached
        if !s.prefilled.load(Ordering::Acquire) {
            s.stall_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let rp = s.read_pos.load(Ordering::Relaxed);
        let wp = s.write_pos.load(Ordering::Acquire);

        if rp == wp {
            // Buffer empty — enter stall mode again if producer is alive
            if s.alive.load(Ordering::Relaxed) {
                s.prefilled.store(false, Ordering::Release);
                s.stall_count.fetch_add(1, Ordering::Relaxed);
                tracing::debug!("Ring buffer underrun, entering stall mode");
            }
            return None;
        }

        let idx = rp % s.capacity;
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
        RingBufferMetrics {
            fill_samples: wp.wrapping_sub(rp),
            capacity_samples: s.capacity,
            total_written: s.total_written.load(Ordering::Relaxed),
            total_read: s.total_read.load(Ordering::Relaxed),
            overflow_count: s.overflow_count.load(Ordering::Relaxed),
            stall_count: s.stall_count.load(Ordering::Relaxed),
            producer_alive: s.alive.load(Ordering::Relaxed),
        }
    }

    /// Returns whether the producer is still alive.
    pub fn is_producer_alive(&self) -> bool {
        self.shared.alive.load(Ordering::Relaxed)
    }
}

// SAFETY: The ring buffer is designed for single-producer single-consumer use.
// Producer only writes, consumer only reads. Atomic positions provide synchronization.
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

        // Write some samples
        producer.write(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Read them back
        assert_eq!(consumer.read_sample(), Some(1.0));
        assert_eq!(consumer.read_sample(), Some(2.0));
        assert_eq!(consumer.read_sample(), Some(3.0));
        assert_eq!(consumer.read_sample(), Some(4.0));
        assert_eq!(consumer.read_sample(), Some(5.0));
        // Empty
        assert_eq!(consumer.read_sample(), None);
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
    fn test_overflow_drops_oldest() {
        let config = RingBufferConfig {
            capacity_samples: 1024,
            prefill_samples: 1,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        // Fill the entire 1024-sample buffer
        let fill_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        producer.write(&fill_data);

        // Write 100 more — should overflow and drop oldest 100
        let overflow_data: Vec<f32> = (1024..1124).map(|i| i as f32).collect();
        producer.write(&overflow_data);

        let metrics = producer.metrics();
        assert!(
            metrics.overflow_count > 0,
            "expected overflow, got count={}",
            metrics.overflow_count
        );

        // First readable sample should NOT be 0.0 (oldest were dropped)
        let first = consumer.read_sample().expect("should have data");
        assert!(first > 0.0, "oldest samples should have been dropped, got {}", first);
    }

    #[test]
    fn test_wraparound() {
        let config = RingBufferConfig {
            capacity_samples: 8,
            prefill_samples: 2,
        };
        let (producer, consumer) = audio_ring_buffer(config);

        // Write and read multiple rounds to force wraparound
        for round in 0..10 {
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
                // Simulate decode timing
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
                        // Values should be monotonically increasing (no duplicates, no reversals)
                        assert!(
                            val > last_val,
                            "non-monotonic: {} after {}",
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

        // Should have read most samples (some may be lost to overflow or timing)
        assert!(
            read_count > 0,
            "should have read some samples, got {}",
            read_count
        );
    }
}
