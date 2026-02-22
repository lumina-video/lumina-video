/**
 * lumina-video MoQ Audio Worklet Processor
 *
 * Standalone AudioWorklet for low-latency audio playback of MoQ streams.
 * Ported from moq/js/hang/src/watch/audio/ring-buffer.ts
 *
 * Messages from main thread:
 *   { type: "init", rate, channels, latency }  - Initialize ring buffer
 *   { type: "data", timestamp, data }           - Audio samples (Float32Array[])
 *   { type: "latency", latency }                - Resize buffer for new latency
 *
 * Messages to main thread:
 *   { type: "state", timestamp, stalled }       - Periodic state update (~75/sec at 48kHz)
 */

/**
 * Circular ring buffer for audio samples.
 * Handles wrap-around, gap-filling, overflow, and stall detection.
 */
class AudioRingBuffer {
  /** @type {Float32Array[]} */
  #buffer;
  #writeIndex = 0;
  #readIndex = 0;
  #stalled = true;
  // Diagnostics
  #diagWriteCount = 0;
  #diagGapCount = 0;
  #diagOverflowCount = 0;
  #diagDropCount = 0;
  // Clock drift correction (Strategy D: sample insert/delete)
  #driftAccum = 0;
  #driftInserts = 0;
  #driftDeletes = 0;

  /** @param {{ rate: number, channels: number, latency: number }} props */
  constructor(props) {
    if (props.channels <= 0) throw new Error("invalid channels");
    if (props.rate <= 0) throw new Error("invalid sample rate");
    if (props.latency <= 0) throw new Error("invalid latency");

    this.rate = props.rate;
    this.channels = props.channels;

    // latency is in milliseconds, convert to samples
    const samples = Math.ceil(props.rate * (props.latency / 1000));
    if (samples === 0) throw new Error("empty buffer");

    this.#buffer = [];
    for (let i = 0; i < this.channels; i++) {
      this.#buffer[i] = new Float32Array(samples);
    }
  }

  get stalled() {
    return this.#stalled;
  }

  get diagWriteCount() { return this.#diagWriteCount; }
  get diagGapCount() { return this.#diagGapCount; }
  get diagOverflowCount() { return this.#diagOverflowCount; }
  get diagDropCount() { return this.#diagDropCount; }
  get driftInserts() { return this.#driftInserts; }
  get driftDeletes() { return this.#driftDeletes; }

  /** Current playback timestamp in microseconds (rebased to wall-clock) */
  get timestamp() {
    return (this.#readIndex / this.rate) * 1_000_000;
  }

  get length() {
    return this.#writeIndex - this.#readIndex;
  }

  get capacity() {
    return this.#buffer[0]?.length ?? 0;
  }

  /** @param {number} latency - New latency in milliseconds */
  resize(latency) {
    const newCapacity = Math.ceil(this.rate * (latency / 1000));
    if (newCapacity === this.capacity) return;
    if (newCapacity === 0) throw new Error("empty buffer");

    const newBuffer = [];
    for (let i = 0; i < this.channels; i++) {
      newBuffer[i] = new Float32Array(newCapacity);
    }

    const samplesToKeep = Math.min(this.length, newCapacity);
    if (samplesToKeep > 0) {
      const copyStart = this.#writeIndex - samplesToKeep;
      for (let channel = 0; channel < this.channels; channel++) {
        const src = this.#buffer[channel];
        const dst = newBuffer[channel];
        for (let i = 0; i < samplesToKeep; i++) {
          dst[i % dst.length] = src[(copyStart + i) % src.length];
        }
      }
    }

    this.#buffer = newBuffer;
    this.#readIndex = this.#writeIndex - samplesToKeep;
    this.#stalled = true;
  }

  /**
   * Write audio samples into the ring buffer.
   *
   * Append-only: ignores timestamp for positioning and simply appends after
   * the last written sample.  MoQ delivers frames in order over reliable
   * QUIC, so timestamp-based positioning is unnecessary.  More importantly,
   * CMAF timestamps can drift relative to the AudioContext sample clock,
   * causing the write pointer to creep ahead of the read pointer and
   * producing constant tiny overflows (~20/sec) that sound like static.
   *
   * @param {number} _timestamp - Unused (kept for API compat)
   * @param {Float32Array[]} data - Per-channel sample arrays
   */
  write(_timestamp, data) {
    if (data.length !== this.channels) throw new Error("wrong number of channels");

    const samples = data[0].length;
    const start = this.#writeIndex;
    const end = start + samples;

    this.#diagWriteCount++;

    // Clear stall once buffer is half-full.
    if (this.#stalled && this.length >= this.#buffer[0].length / 2) {
      this.#stalled = false;
    }

    // Overflow: new samples would exceed capacity.
    // The initial network burst (relay sends buffered data faster than
    // real-time on subscribe) fills the buffer to 100%.  At 100%, every
    // write overflows because the 960-sample write arrives before the
    // AudioWorklet has consumed 960 samples (it reads 128 per quantum).
    // Reset to 50% fill on overflow — this is a one-time correction;
    // after that, producer and consumer both run at 48kHz so the buffer
    // stays near 50% with no further overflows.
    const overflow = end - this.#readIndex - this.#buffer[0].length;
    if (overflow > 0) {
      this.#stalled = false;
      this.#diagOverflowCount++;
      const targetFill = Math.floor(this.#buffer[0].length / 2);
      const newReadIndex = end - targetFill;
      if (this.#diagOverflowCount <= 5) {
        console.debug(`[ring-diag] overflow#${this.#diagOverflowCount}: reset to 50% (readIdx ${this.#readIndex} -> ${newReadIndex})`);
      }
      this.#readIndex = newReadIndex;
    }

    // Write samples
    for (let channel = 0; channel < this.channels; channel++) {
      const src = data[channel];
      const dst = this.#buffer[channel];
      for (let i = 0; i < samples; i++) {
        dst[(start + i) % dst.length] = src[i];
      }
    }

    this.#writeIndex = end;
  }

  /**
   * Read samples from the ring buffer into output arrays.
   *
   * Includes clock drift correction: after each read, the fill level is
   * compared to the 50% target.  If the buffer is draining (producer slower
   * than consumer), readIndex is nudged back by 1–3 samples so the same
   * sample is read twice — effectively slowing consumption.  If the buffer
   * is filling, readIndex is nudged forward to skip samples.  A fractional
   * accumulator with proportional gain keeps corrections smooth and
   * prevents oscillation.  At 48 kHz a single duplicated/skipped sample is
   * a 20.8 µs discontinuity — well below the audibility threshold.
   *
   * @param {Float32Array[]} output - Per-channel output arrays
   * @returns {number} Number of samples read
   */
  read(output) {
    if (output.length !== this.channels) throw new Error("wrong number of channels");
    if (this.#stalled) return 0;

    const available = this.#writeIndex - this.#readIndex;
    if (available <= 0) {
      this.#stalled = true;
      return 0;
    }
    const samples = Math.min(available, output[0].length);

    for (let channel = 0; channel < this.channels; channel++) {
      const dst = output[channel];
      const src = this.#buffer[channel];
      for (let i = 0; i < samples; i++) {
        dst[i] = src[(this.#readIndex + i) % src.length];
      }
    }

    this.#readIndex += samples;

    // --- Clock drift correction ---
    // Proportional controller: correction rate scales with distance from
    // the 50% fill target.  A ±5% deadzone prevents jitter when centred.
    //
    // Gain 0.1 → at 10% error (fill 40% or 60%), correct ~480 samples/sec.
    // At the observed ~0.8% ffmpeg drift this stabilises fill around 42%.
    const fill = (this.#writeIndex - this.#readIndex) / this.#buffer[0].length;
    const error = 0.5 - fill; // positive = draining, negative = filling
    const absError = Math.abs(error);
    const DEADZONE = 0.05;

    if (absError > DEADZONE) {
      const netError = (absError - DEADZONE) * Math.sign(error);
      const correctionsPerSec = netError * this.rate * 0.1;
      this.#driftAccum += (correctionsPerSec * samples) / this.rate;

      const correction = Math.trunc(this.#driftAccum);
      if (correction !== 0) {
        const clamped = Math.max(-3, Math.min(3, correction));
        if (clamped > 0) {
          // Insert: nudge readIndex back so next read re-reads samples
          this.#readIndex -= clamped;
          this.#driftInserts += clamped;
        } else {
          // Delete: nudge readIndex forward, but don't skip past writeIndex
          const maxSkip = this.#writeIndex - this.#readIndex;
          const skip = Math.min(-clamped, maxSkip);
          this.#readIndex += skip;
          this.#driftDeletes += skip;
        }
        this.#driftAccum -= clamped;
      }
    } else {
      // Inside deadzone — reset accumulator to prevent integral windup
      this.#driftAccum = 0;
    }

    return samples;
  }
}

/**
 * AudioWorklet processor for MoQ audio playback.
 */
class MoqAudioRender extends AudioWorkletProcessor {
  /** @type {AudioRingBuffer|undefined} */
  #buffer;
  #underflow = 0;
  #stateCounter = 0;
  #diagCounter = 0;

  constructor() {
    super();

    this.port.onmessage = (event) => {
      const { type } = event.data;
      if (type === "init") {
        try {
          this.#buffer = new AudioRingBuffer(event.data);
          this.#underflow = 0;
        } catch (e) {
          this.port.postMessage({ type: "error", message: "AudioRingBuffer init failed", detail: e.message });
        }
      } else if (type === "data") {
        if (!this.#buffer) return;
        this.#buffer.write(event.data.timestamp, event.data.data);
      } else if (type === "latency") {
        if (!this.#buffer) return;
        this.#buffer.resize(event.data.latency);
      }
    };
  }

  process(_inputs, outputs, _parameters) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const samplesRead = this.#buffer?.read(output) ?? 0;

    // Zero any unfilled output to prevent stale data from playing as static.
    // The WebAudio spec says outputs are zero-initialized, but some browsers
    // reuse buffers across process() calls without clearing.
    if (samplesRead < output[0].length) {
      for (let ch = 0; ch < output.length; ch++) {
        output[ch].fill(0, samplesRead);
      }
      this.#underflow += output[0].length - samplesRead;
    } else if (this.#underflow > 0 && this.#buffer) {
      this.#underflow = 0;
    }

    // Diagnostic: compute output RMS every 375 blocks (~1/sec at 48kHz)
    this.#diagCounter++;
    if (this.#diagCounter >= 375 && samplesRead > 0 && output[0]) {
      this.#diagCounter = 0;
      let sumSq = 0;
      for (let i = 0; i < samplesRead; i++) sumSq += output[0][i] * output[0][i];
      const outRms = Math.sqrt(sumSq / samplesRead);
      this.port.postMessage({
        type: "diag",
        outRms: outRms.toFixed(4),
        gaps: this.#buffer?.diagGapCount ?? 0,
        overflows: this.#buffer?.diagOverflowCount ?? 0,
        drops: this.#buffer?.diagDropCount ?? 0,
        writes: this.#buffer?.diagWriteCount ?? 0,
        driftInserts: this.#buffer?.driftInserts ?? 0,
        driftDeletes: this.#buffer?.driftDeletes ?? 0,
      });
    }

    // Send state update every 5 blocks (5 × 128 = 640 samples → ~75/sec at 48kHz)
    this.#stateCounter++;
    if (this.#buffer && this.#stateCounter >= 5) {
      this.#stateCounter = 0;
      this.port.postMessage({
        type: "state",
        timestamp: this.#buffer.timestamp,
        stalled: this.#buffer.stalled,
        underflowSamples: this.#underflow,
        bufferLength: this.#buffer.length,
        bufferCapacity: this.#buffer.capacity,
      });
    }

    return true;
  }
}

registerProcessor("moq-audio-render", MoqAudioRender);
