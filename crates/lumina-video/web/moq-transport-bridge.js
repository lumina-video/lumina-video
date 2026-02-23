/**
 * lumina-video MoQ Transport Bridge
 *
 * Self-contained JavaScript bridge for MoQ live streaming with:
 * - Transport connection via @moq/lite (bundled as window.MoqLite)
 * - Catalog JSON parsing
 * - WebCodecs VideoDecoder for video tracks
 * - WebCodecs AudioDecoder + AudioWorklet for audio tracks
 *
 * Called from Rust/WASM via wasm_bindgen(module = "/web/moq-transport-bridge.js")
 *
 * Architecture:
 *   WASM ←→ moq-transport-bridge.js ←→ MoqLite (IIFE bundle) ←→ Relay (WebTransport/WS)
 *                                    ├→ WebCodecs VideoDecoder → VideoFrame (polled by WASM)
 *                                    └→ WebCodecs AudioDecoder → AudioWorklet (autonomous)
 */

// ============================================================================
// Hot-path constants & helpers
// ============================================================================

/** Max frames queued in WebCodecs decoder before applying backpressure. */
const MAX_VIDEO_DECODE_QUEUE = 3;
const MAX_AUDIO_DECODE_QUEUE = 5;

/** Timeout for MoQ transport reads (ms). Prevents indefinite stall on network partition. */
const TRANSPORT_READ_TIMEOUT_MS = 10_000;

/**
 * Wait until the decoder's internal queue has capacity.
 * Prevents unbounded queue growth when the decoder falls behind.
 * @param {VideoDecoder|AudioDecoder} decoder
 * @param {number} maxQueue
 */
function waitForDecodeCapacity(decoder, maxQueue) {
  if (decoder.decodeQueueSize <= maxQueue) return Promise.resolve();
  return new Promise((resolve) => {
    decoder.addEventListener("dequeue", function onDequeue() {
      if (decoder.decodeQueueSize <= maxQueue) {
        decoder.removeEventListener("dequeue", onDequeue);
        resolve();
      }
    });
  });
}

/**
 * Wrap a promise with a timeout. Rejects with an error if the promise
 * doesn't resolve within `ms` milliseconds.
 * @template T
 * @param {Promise<T>} promise
 * @param {number} ms
 * @returns {Promise<T>}
 */
function withTimeout(promise, ms) {
  let timer;
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("transport read timeout")), ms);
    }),
  ]).finally(() => clearTimeout(timer));
}

// ============================================================================
// Session management
// ============================================================================

/** @type {Map<number, MoqSession>} */
const activeSessions = new Map();
let sessionIdCounter = 0;

/** @type {Map<number, VideoDecoderHandle>} */
const activeVideoDecoders = new Map();
let videoDecoderIdCounter = 0;

/** @type {Map<number, AudioHandle>} */
const activeAudioHandles = new Map();
let audioIdCounter = 0;

// ============================================================================
// MoQ Session
// ============================================================================

class MoqSession {
  constructor(id, url, namespace) {
    this.id = id;
    this.url = url;
    this.namespace = namespace;
    this.connection = null;
    this.broadcast = null;
    this.catalog = null;
    this.state = "connecting";
    this.error = null;
    this.stats = { videoFrames: 0, audioFrames: 0 };
    this._catalogTrack = null;
    this._closed = false;
  }

  close() {
    this._closed = true;
    this.state = "closed";
    if (this._catalogTrack) {
      try { this._catalogTrack.close(); } catch (_) { /* ignore */ }
    }
    if (this.broadcast) {
      try { this.broadcast.close(); } catch (_) { /* ignore */ }
    }
    if (this.connection) {
      try { this.connection.close(); } catch (_) { /* ignore */ }
    }
  }
}

// ============================================================================
// Video Decoder Handle
// ============================================================================

class VideoDecoderHandle {
  constructor(id, sessionId) {
    this.id = id;
    this.sessionId = sessionId;
    this.decoder = null;
    this.lastFrame = null;
    this.frameCount = 0;
    this.errorMessage = null;
    this.closed = false;
    this._track = null;
    this._abortController = null;
  }

  close() {
    this.closed = true;
    if (this._abortController) {
      this._abortController.abort();
    }
    if (this.lastFrame) {
      try { this.lastFrame.close(); } catch (_) { /* ignore */ }
      this.lastFrame = null;
    }
    if (this.decoder) {
      try { this.decoder.close(); } catch (_) { /* ignore */ }
    }
    if (this._track) {
      try { this._track.close(); } catch (_) { /* ignore */ }
    }
  }
}

// ============================================================================
// Audio Handle
// ============================================================================

class AudioHandle {
  constructor(id, sessionId) {
    this.id = id;
    this.sessionId = sessionId;
    this.context = null;
    this.workletNode = null;
    this.gainNode = null;
    this.decoder = null;
    this.muted = false;
    this.volume = 1.0;
    this.stalled = true;
    this.timestampMs = 0;
    this.frameCount = 0;
    this.closed = false;
    this._track = null;
    this._abortController = null;
    // Buffer health stats from worklet
    this.underflowSamples = 0;
    this.bufferLength = 0;
    this.bufferCapacity = 0;
  }

  close() {
    this.closed = true;
    if (this._abortController) {
      this._abortController.abort();
    }
    if (this.decoder) {
      try { this.decoder.close(); } catch (_) { /* ignore */ }
    }
    if (this.workletNode) {
      try { this.workletNode.disconnect(); } catch (_) { /* ignore */ }
    }
    if (this.gainNode) {
      try { this.gainNode.disconnect(); } catch (_) { /* ignore */ }
    }
    if (this.context) {
      try { this.context.close(); } catch (_) { /* ignore */ }
    }
    if (this._track) {
      try { this._track.close(); } catch (_) { /* ignore */ }
    }
  }
}

// ============================================================================
// VarInt decoder (QUIC RFC 9000)
// ============================================================================

function decodeVarInt(data, offset) {
  if (offset >= data.length) throw new Error("varint: buffer empty");
  const first = data[offset];
  const prefix = (first & 0xc0) >> 6;
  const size = 1 << prefix;

  if (offset + size > data.length) {
    throw new Error(`varint: need ${size} bytes, have ${data.length - offset}`);
  }

  const view = new DataView(data.buffer, data.byteOffset + offset, size);
  let value;

  if (size === 1) {
    value = first & 0x3f;
  } else if (size === 2) {
    value = view.getUint16(0) & 0x3fff;
  } else if (size === 4) {
    value = view.getUint32(0) & 0x3fffffff;
  } else {
    // 8 bytes — use BigInt for precision, convert back to Number
    value = Number(view.getBigUint64(0) & 0x3fffffffffffffffn);
  }

  return { value, length: size };
}

// ============================================================================
// CMAF (fMP4) parser — minimal ISO BMFF box reader
// ============================================================================

/**
 * Parse ISO BMFF boxes from a buffer.
 * Returns array of { type, offset, size, data? } objects.
 */
function parseBoxes(data) {
  const boxes = [];
  let pos = 0;

  while (pos + 8 <= data.length) {
    const view = new DataView(data.buffer, data.byteOffset + pos);
    let size = view.getUint32(0);
    const type = String.fromCharCode(data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]);

    let headerSize = 8;
    if (size === 1) {
      // Extended size (64-bit)
      if (pos + 16 > data.length) break;
      size = Number(view.getBigUint64(8));
      headerSize = 16;
    } else if (size === 0) {
      size = data.length - pos;
    }

    if (pos + size > data.length) break;

    boxes.push({
      type,
      offset: pos,
      headerSize,
      size,
      data: data.subarray(pos + headerSize, pos + size),
    });
    pos += size;
  }

  return boxes;
}

/**
 * Recursively find a box by type in nested ISO BMFF data.
 */
function findBox(data, type) {
  const boxes = parseBoxes(data);
  for (const box of boxes) {
    if (box.type === type) return box;
    // Container boxes: moof, traf, moov, trak, mdia, minf, stbl
    if (["moof", "traf", "moov", "trak", "mdia", "minf", "stbl"].includes(box.type)) {
      const found = findBox(box.data, type);
      if (found) return found;
    }
  }
  return null;
}

/**
 * Parse tfdt (Track Fragment Decode Time) box.
 * Returns baseMediaDecodeTime as a number.
 */
function parseTfdt(data) {
  const view = new DataView(data.buffer, data.byteOffset);
  const version = data[0];
  if (version === 1) {
    return Number(view.getBigUint64(4));
  }
  return view.getUint32(4);
}

/**
 * Parse trun (Track Run) box.
 * Returns { sampleCount, firstSampleFlags, dataOffset, samples[] }
 */
function parseTrun(data) {
  const view = new DataView(data.buffer, data.byteOffset);
  const version = data[0];
  const flags = (data[1] << 16) | (data[2] << 8) | data[3];
  let pos = 4;

  const sampleCount = view.getUint32(pos); pos += 4;

  let dataOffset = 0;
  if (flags & 0x000001) { dataOffset = view.getInt32(pos); pos += 4; }

  let firstSampleFlags;
  if (flags & 0x000004) { firstSampleFlags = view.getUint32(pos); pos += 4; }

  const samples = [];
  for (let i = 0; i < sampleCount; i++) {
    const sample = {};
    if (flags & 0x000100) { sample.duration = view.getUint32(pos); pos += 4; }
    if (flags & 0x000200) { sample.size = view.getUint32(pos); pos += 4; }
    if (flags & 0x000400) { sample.flags = view.getUint32(pos); pos += 4; }
    if (flags & 0x000800) {
      sample.compositionOffset = version === 0 ? view.getUint32(pos) : view.getInt32(pos);
      pos += 4;
    }
    samples.push(sample);
  }

  return { sampleCount, dataOffset, firstSampleFlags, samples };
}

/**
 * Parse tfhd (Track Fragment Header) box.
 */
function parseTfhd(data) {
  const view = new DataView(data.buffer, data.byteOffset);
  const flags = (data[1] << 16) | (data[2] << 8) | data[3];
  let pos = 4;

  const trackId = view.getUint32(pos); pos += 4;

  let baseDataOffset, defaultDuration, defaultSize, defaultFlags;
  if (flags & 0x000001) { baseDataOffset = Number(view.getBigUint64(pos)); pos += 8; }
  if (flags & 0x000002) { pos += 4; } // sample description index
  if (flags & 0x000008) { defaultDuration = view.getUint32(pos); pos += 4; }
  if (flags & 0x000010) { defaultSize = view.getUint32(pos); pos += 4; }
  if (flags & 0x000020) { defaultFlags = view.getUint32(pos); pos += 4; }

  return { trackId, baseDataOffset, defaultDuration, defaultSize, defaultFlags };
}

/**
 * Decode CMAF data segment (moof+mdat) into samples.
 * @param {Uint8Array} segment
 * @param {number} timescale - Time units per second
 * @returns {{ data: Uint8Array, timestamp: number, keyframe: boolean }[]}
 */
function decodeCmafSegment(segment, timescale) {
  const tfdtBox = findBox(segment, "tfdt");
  const trunBox = findBox(segment, "trun");
  const tfhdBox = findBox(segment, "tfhd");
  const mdatBox = findBox(segment, "mdat");

  if (!trunBox || !mdatBox) {
    console.warn("[moq-bridge] CMAF segment missing trun or mdat");
    return [];
  }

  const baseDecodeTime = tfdtBox ? parseTfdt(tfdtBox.data) : 0;
  const trun = parseTrun(trunBox.data);
  const tfhd = tfhdBox ? parseTfhd(tfhdBox.data) : {};

  const defaultDuration = tfhd.defaultDuration ?? 0;
  const defaultSize = tfhd.defaultSize ?? 0;
  const defaultFlags = tfhd.defaultFlags ?? 0;

  const mdatData = mdatBox.data;
  const samples = [];
  let dataOffset = 0;
  let decodeTime = baseDecodeTime;

  for (let i = 0; i < trun.sampleCount; i++) {
    const s = trun.samples[i] ?? {};
    const sampleSize = s.size ?? defaultSize;
    const sampleDuration = s.duration ?? defaultDuration;

    if (sampleSize <= 0 || dataOffset + sampleSize > mdatData.length) break;

    const sampleFlags = (i === 0 && trun.firstSampleFlags !== undefined)
      ? trun.firstSampleFlags
      : (s.flags ?? defaultFlags);

    const compositionOffset = s.compositionOffset ?? 0;
    const pts = decodeTime + compositionOffset;
    const timestamp = Math.round((pts * 1_000_000) / timescale);
    const keyframe = sampleFlags === 0 || (sampleFlags & 0x00010000) === 0;

    samples.push({
      data: mdatData.subarray(dataOffset, dataOffset + sampleSize),
      timestamp,
      keyframe,
    });

    dataOffset += sampleSize;
    decodeTime += sampleDuration;
  }

  return samples;
}

// ============================================================================
// Catalog parsing
// ============================================================================

/**
 * Parse a MoQ catalog JSON into a structured object.
 * @param {string} jsonStr
 * @returns {{ video: Object[], audio: Object[] }}
 */
function parseCatalog(jsonStr) {
  const catalog = JSON.parse(jsonStr);

  const video = [];
  if (catalog.video?.renditions) {
    for (const [name, config] of Object.entries(catalog.video.renditions)) {
      video.push({
        name,
        codec: config.codec,
        width: config.codedWidth,
        height: config.codedHeight,
        framerate: config.framerate,
        bitrate: config.bitrate,
        description: config.description, // hex string
        container: config.container ?? { kind: "legacy" },
        optimizeForLatency: config.optimizeForLatency,
      });
    }
  }

  const audio = [];
  if (catalog.audio?.renditions) {
    for (const [name, config] of Object.entries(catalog.audio.renditions)) {
      audio.push({
        name,
        codec: config.codec,
        sampleRate: config.sampleRate,
        channels: config.numberOfChannels,
        bitrate: config.bitrate,
        description: config.description, // hex string
        container: config.container ?? { kind: "legacy" },
      });
    }
  }

  return { video, audio, raw: catalog };
}

// ============================================================================
// Hex utility
// ============================================================================

function hexToBytes(hex) {
  if (!hex || hex.length === 0) return undefined;
  if (hex.length % 2 !== 0) {
    throw new Error(`hexToBytes: hex string must have even length, got ${hex.length}`);
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  return bytes;
}

// ============================================================================
// Exported API — called from WASM via wasm_bindgen
// ============================================================================

/**
 * Connect to a MoQ relay and subscribe to a broadcast.
 * @param {string} url - WebTransport URL (http:// or https://)
 * @param {string} namespace - Broadcast namespace
 * @returns {Promise<number>} Session ID
 */
export async function moqConnect(url, namespace) {
  const Moq = window.MoqLite;
  if (!Moq) throw new Error("MoqLite bundle not loaded. Include moq-lite-bundle.js before WASM.");

  const id = sessionIdCounter++;
  const session = new MoqSession(id, url, namespace);
  activeSessions.set(id, session);

  try {
    console.log("[moq-transport] Connecting to", url);

    // Race WebTransport and WebSocket with no head start for either.
    // Add a timeout to avoid hanging if both transports fail to settle.
    const connectPromise = Moq.Connection.connect(new URL(url), {
      websocket: { delay: 0 },
    });
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(() => reject(new Error("Connection timed out after 10s")), 10_000)
    );

    const connection = await Promise.race([connectPromise, timeoutPromise]);

    if (session._closed) {
      connection.close();
      throw new Error("Session closed during connect");
    }

    session.connection = connection;
    session.state = "connected";
    console.log("[moq-transport] Connected, version:", connection.version?.toString(16));

    // Monitor connection lifecycle
    connection.closed.then(
      (v) => console.debug("[moq-transport] connection closed:", v),
    );

    // Subscribe to the broadcast namespace
    const broadcast = connection.consume(Moq.Path.from(namespace));
    session.broadcast = broadcast;
    console.log("[moq-transport] Subscribing to catalog.json");

    // Subscribe to catalog.json (priority 100)
    const catalogTrack = broadcast.subscribe("catalog.json", 100);
    session._catalogTrack = catalogTrack;

    // Fetch catalog asynchronously
    fetchCatalog(session, catalogTrack);

    return id;
  } catch (err) {
    session.state = "error";
    session.error = err.message || String(err);
    console.error("[moq-transport] Connect failed:", err);
    throw err;
  }
}

/**
 * Fetch catalog from a track and update session state.
 */
async function fetchCatalog(session, track) {
  try {
    // Read the latest catalog frame
    const frame = await track.readFrame();
    if (!frame || session._closed) return;

    const text = new TextDecoder().decode(frame);
    console.debug("[moq-transport] Received catalog:", text);

    session.catalog = parseCatalog(text);
    session.state = "catalog";

    // Keep reading for catalog updates
    fetchCatalogUpdates(session, track);
  } catch (err) {
    if (!session._closed) {
      console.warn("[moq-transport] Catalog fetch error:", err);
      session.state = "error";
      session.error = err.message || String(err);
    }
  }
}

async function fetchCatalogUpdates(session, track) {
  try {
    for (;;) {
      const frame = await track.readFrame();
      if (!frame || session._closed) break;

      const text = new TextDecoder().decode(frame);
      console.debug("[moq-transport] Catalog update:", text);
      session.catalog = parseCatalog(text);
    }
  } catch (_) {
    // Catalog track closed — normal for stream end
  }
}

/**
 * Disconnect a MoQ session.
 * @param {number} sessionId
 */
export function moqDisconnect(sessionId) {
  const session = activeSessions.get(sessionId);
  if (!session) return;

  session.close();
  activeSessions.delete(sessionId);
  console.debug("[moq-transport] Disconnected session", sessionId);
}

/**
 * Get session state.
 * @param {number} sessionId
 * @returns {string} "connecting"|"connected"|"catalog"|"playing"|"error"|"closed"
 */
export function moqGetSessionState(sessionId) {
  const session = activeSessions.get(sessionId);
  return session?.state ?? "closed";
}

/**
 * Get error message for a session.
 * @param {number} sessionId
 * @returns {string|null}
 */
export function moqGetError(sessionId) {
  const session = activeSessions.get(sessionId);
  return session?.error ?? null;
}

/**
 * Get parsed catalog as JSON string.
 * @param {number} sessionId
 * @returns {string|null} JSON string or null if not yet available
 */
export function moqGetCatalog(sessionId) {
  const session = activeSessions.get(sessionId);
  if (!session?.catalog) return null;
  return JSON.stringify(session.catalog);
}

/**
 * Start video decoding for a track.
 * @param {number} sessionId
 * @param {string} trackName - Track name from catalog
 * @param {string} codec - WebCodecs codec string
 * @param {number} width - Coded width
 * @param {number} height - Coded height
 * @param {string} containerKind - "legacy" or "cmaf"
 * @param {number} timescale - For CMAF containers
 * @param {string|null} description - Hex-encoded codec description
 * @param {Function} onFrame - Callback: (timestamp, width, height, duration) => void
 * @param {Function} onError - Callback: (errorMessage) => void
 * @returns {number} Decoder ID
 */
export function moqStartVideo(
  sessionId, trackName, codec, width, height,
  containerKind, timescale, description,
  onFrame, onError
) {
  const session = activeSessions.get(sessionId);
  if (!session?.broadcast) throw new Error("Session not connected");

  const id = videoDecoderIdCounter++;
  const handle = new VideoDecoderHandle(id, sessionId);
  handle._abortController = new AbortController();
  activeVideoDecoders.set(id, handle);

  // Create WebCodecs VideoDecoder
  const decoder = new VideoDecoder({
    output: (frame) => {
      if (handle.closed) { frame.close(); return; }
      if (handle.lastFrame) { handle.lastFrame.close(); }
      handle.lastFrame = frame;
      handle.frameCount++;

      const sess = activeSessions.get(sessionId);
      if (sess) sess.stats.videoFrames++;

      try {
        onFrame(frame.timestamp, frame.displayWidth, frame.displayHeight, frame.duration || 0);
      } catch (e) {
        console.error("[moq-transport] Video frame callback error:", e);
      }
    },
    error: (e) => {
      handle.errorMessage = e.message;
      handle.closed = true;
      console.error("[moq-transport] VideoDecoder error:", e);
      try { onError(e.message); } catch (_) { /* ignore */ }
    },
  });

  // Configure decoder
  const config = {
    codec,
    optimizeForLatency: true,
  };
  if (width > 0) config.codedWidth = width;
  if (height > 0) config.codedHeight = height;
  if (description) {
    config.description = hexToBytes(description);
  }

  decoder.configure(config);
  handle.decoder = decoder;

  // Subscribe to track and start decode loop
  const track = session.broadcast.subscribe(trackName, 60);
  handle._track = track;

  if (containerKind === "cmaf") {
    runCmafVideoLoop(handle, track, timescale);
  } else {
    runLegacyVideoLoop(handle, track);
  }

  session.state = "playing";
  console.debug("[moq-transport] Started video decoder", id, "for track", trackName);
  return id;
}

async function runLegacyVideoLoop(handle, track) {
  try {
    for (;;) {
      const group = await withTimeout(track.nextGroup(), TRANSPORT_READ_TIMEOUT_MS);
      if (!group || handle.closed) break;

      let isKeyframe = true;
      try {
        for (;;) {
          const frame = await withTimeout(group.readFrame(), TRANSPORT_READ_TIMEOUT_MS);
          if (!frame || handle.closed) break;

          const { value: timestamp, length } = decodeVarInt(frame, 0);
          const payload = frame.subarray(length);

          if (handle.decoder.state === "configured") {
            await waitForDecodeCapacity(handle.decoder, MAX_VIDEO_DECODE_QUEUE);
            const chunk = new EncodedVideoChunk({
              type: isKeyframe ? "key" : "delta",
              timestamp,
              data: payload,
            });
            handle.decoder.decode(chunk);
          }

          isKeyframe = false;
        }
      } finally {
        group.close();
      }
    }
  } catch (err) {
    if (!handle.closed) {
      console.warn("[moq-transport] Legacy video loop error:", err);
    }
  }
}

async function runCmafVideoLoop(handle, track, timescale) {
  try {
    for (;;) {
      const group = await withTimeout(track.nextGroup(), TRANSPORT_READ_TIMEOUT_MS);
      if (!group || handle.closed) break;

      await processCmafVideoGroup(handle, group, timescale);
    }
  } catch (err) {
    if (!handle.closed) {
      console.warn("[moq-transport] CMAF video loop error:", err);
    }
  }
}

async function processCmafVideoGroup(handle, group, timescale) {
  try {
    for (;;) {
      const segment = await withTimeout(group.readFrame(), TRANSPORT_READ_TIMEOUT_MS);
      if (!segment || handle.closed) break;

      const samples = decodeCmafSegment(segment, timescale);
      for (const sample of samples) {
        if (handle.closed || handle.decoder.state !== "configured") break;

        await waitForDecodeCapacity(handle.decoder, MAX_VIDEO_DECODE_QUEUE);
        const chunk = new EncodedVideoChunk({
          type: sample.keyframe ? "key" : "delta",
          timestamp: sample.timestamp,
          data: sample.data,
        });
        handle.decoder.decode(chunk);
      }
    }
  } catch (err) {
    if (!handle.closed) {
      console.warn("[moq-transport] CMAF video group error:", err);
    }
  } finally {
    group.close();
  }
}

/**
 * Get the latest decoded VideoFrame (transfers ownership — caller must close).
 * @param {number} decoderId
 * @returns {VideoFrame|null}
 */
export function moqGetVideoFrame(decoderId) {
  const handle = activeVideoDecoders.get(decoderId);
  if (!handle?.lastFrame) return null;

  const frame = handle.lastFrame;
  handle.lastFrame = null;
  return frame;
}

/**
 * Close a video decoder.
 * @param {number} decoderId
 */
export function moqCloseVideo(decoderId) {
  const handle = activeVideoDecoders.get(decoderId);
  if (!handle) return;
  handle.close();
  activeVideoDecoders.delete(decoderId);
}

/**
 * Start audio decoding and playback for a track.
 * @param {number} sessionId
 * @param {string} trackName
 * @param {string} codec - e.g., "opus" or "mp4a.40.2"
 * @param {number} sampleRate
 * @param {number} channels
 * @param {string} containerKind - "legacy" or "cmaf"
 * @param {number} timescale
 * @param {string|null} description - Hex-encoded codec description
 * @returns {Promise<number>} Audio handle ID
 */
export async function moqStartAudio(
  sessionId, trackName, codec, sampleRate, channels,
  containerKind, timescale, description
) {
  const session = activeSessions.get(sessionId);
  if (!session?.broadcast) throw new Error("Session not connected");

  // Check AudioDecoder support for this codec
  if (typeof AudioDecoder === "undefined") {
    console.warn("[moq-transport] AudioDecoder not available, skipping audio");
    return -1;
  }

  try {
    const support = await AudioDecoder.isConfigSupported({
      codec,
      sampleRate,
      numberOfChannels: channels,
    });
    if (!support.supported) {
      console.warn(`[moq-transport] Audio codec ${codec} not supported, skipping`);
      return -1;
    }
  } catch (_) {
    console.warn(`[moq-transport] Audio codec check failed for ${codec}, skipping`);
    return -1;
  }

  const id = audioIdCounter++;
  const handle = new AudioHandle(id, sessionId);
  handle._abortController = new AbortController();
  activeAudioHandles.set(id, handle);

  // Create AudioContext
  const context = new AudioContext({
    latencyHint: "interactive",
    sampleRate,
  });
  handle.context = context;

  // Create GainNode for volume control
  const gainNode = context.createGain();
  gainNode.gain.value = handle.volume;
  gainNode.connect(context.destination);
  handle.gainNode = gainNode;

  // Load AudioWorklet
  try {
    await context.audioWorklet.addModule("moq-audio-worklet.js");
  } catch (err) {
    console.error("[moq-transport] Failed to load audio worklet:", err);
    handle.close();
    activeAudioHandles.delete(id);
    return -1;
  }

  if (handle.closed || context.state === "closed") {
    handle.close();
    activeAudioHandles.delete(id);
    return -1;
  }

  // Create AudioWorkletNode
  const workletNode = new AudioWorkletNode(context, "moq-audio-render", {
    channelCount: channels,
    channelCountMode: "explicit",
  });
  workletNode.connect(gainNode);
  handle.workletNode = workletNode;

  // Initialize ring buffer in worklet
  // 500ms gives enough headroom for startup transient + jitter.
  // The buffer fills to 100% during stall, then drains during the
  // startup burst; at 200ms it would drop to <10%, causing noise.
  workletNode.port.postMessage({
    type: "init",
    rate: sampleRate,
    channels,
    latency: 500,
  });

  // Listen for state updates from worklet
  workletNode.port.onmessage = (event) => {
    if (event.data.type === "state") {
      handle.stalled = event.data.stalled;
      handle.timestampMs = event.data.timestamp / 1000; // us → ms
      handle.underflowSamples = event.data.underflowSamples ?? 0;
      handle.bufferLength = event.data.bufferLength ?? 0;
      handle.bufferCapacity = event.data.bufferCapacity ?? 0;
    } else if (event.data.type === "diag") {
      console.debug(
        `[ring-diag] output rms=${event.data.outRms}`,
        `writes=${event.data.writes} gaps=${event.data.gaps}`,
        `overflows=${event.data.overflows} drops=${event.data.drops}`
      );
    }
  };

  // Create AudioDecoder
  const decoder = new AudioDecoder({
    output: (audioData) => {
      if (handle.closed) { audioData.close(); return; }

      handle.frameCount++;
      const sess = activeSessions.get(sessionId);
      if (sess) sess.stats.audioFrames++;

      // Allocate Float32Array per channel — copyTo() requires a destination
      // buffer; these are transferred to the AudioWorklet via postMessage
      // (zero-copy via Transferable), so the allocation is not wasted.
      const channelData = [];
      for (let ch = 0; ch < audioData.numberOfChannels; ch++) {
        const data = new Float32Array(audioData.numberOfFrames);
        audioData.copyTo(data, { format: "f32-planar", planeIndex: ch });
        channelData.push(data);
      }

      workletNode.port.postMessage(
        { type: "data", data: channelData, timestamp: audioData.timestamp },
        channelData.map((d) => d.buffer),
      );

      audioData.close();
    },
    error: (err) => {
      console.error("[moq-transport] AudioDecoder error:", err);
    },
  });
  handle.decoder = decoder;

  // Configure decoder
  const decoderConfig = {
    codec,
    sampleRate,
    numberOfChannels: channels,
  };
  if (description) {
    decoderConfig.description = hexToBytes(description);
  }
  decoder.configure(decoderConfig);

  // Subscribe to track and start decode loop
  const track = session.broadcast.subscribe(trackName, 80);
  handle._track = track;

  if (containerKind === "cmaf") {
    runCmafAudioLoop(handle, track, timescale);
  } else {
    runLegacyAudioLoop(handle, track);
  }

  // Resume AudioContext (may need user gesture)
  context.resume().catch(() => {});

  console.debug("[moq-transport] Started audio", id, "for track", trackName);
  return id;
}

async function runLegacyAudioLoop(handle, track) {
  try {
    for (;;) {
      const group = await withTimeout(track.nextGroup(), TRANSPORT_READ_TIMEOUT_MS);
      if (!group || handle.closed) break;

      let isKeyframe = true;
      try {
        for (;;) {
          const frame = await withTimeout(group.readFrame(), TRANSPORT_READ_TIMEOUT_MS);
          if (!frame || handle.closed) break;

          const { value: timestamp, length } = decodeVarInt(frame, 0);
          const payload = frame.subarray(length);

          if (handle.decoder.state === "configured") {
            await waitForDecodeCapacity(handle.decoder, MAX_AUDIO_DECODE_QUEUE);
            const chunk = new EncodedAudioChunk({
              type: isKeyframe ? "key" : "delta",
              timestamp,
              data: payload,
            });
            handle.decoder.decode(chunk);
          }
          isKeyframe = false;
        }
      } finally {
        group.close();
      }
    }
  } catch (err) {
    if (!handle.closed) {
      console.warn("[moq-transport] Legacy audio loop error:", err);
    }
  }
}

async function runCmafAudioLoop(handle, track, timescale) {
  try {
    for (;;) {
      const group = await withTimeout(track.nextGroup(), TRANSPORT_READ_TIMEOUT_MS);
      if (!group || handle.closed) break;

      await processCmafAudioGroup(handle, group, timescale);
    }
  } catch (err) {
    if (!handle.closed) {
      console.warn("[moq-transport] CMAF audio loop error:", err);
    }
  }
}

async function processCmafAudioGroup(handle, group, timescale) {
  try {
    for (;;) {
      const segment = await withTimeout(group.readFrame(), TRANSPORT_READ_TIMEOUT_MS);
      if (!segment || handle.closed) break;

      const samples = decodeCmafSegment(segment, timescale);
      for (const sample of samples) {
        if (handle.closed || handle.decoder.state !== "configured") break;

        await waitForDecodeCapacity(handle.decoder, MAX_AUDIO_DECODE_QUEUE);
        const chunk = new EncodedAudioChunk({
          type: sample.keyframe ? "key" : "delta",
          timestamp: sample.timestamp,
          data: sample.data,
        });
        handle.decoder.decode(chunk);
      }
    }
  } catch (err) {
    if (!handle.closed) {
      console.warn("[moq-transport] CMAF audio group error:", err);
    }
  } finally {
    group.close();
  }
}

/**
 * Set audio muted state.
 * @param {number} audioId
 * @param {boolean} muted
 */
export function moqSetAudioMuted(audioId, muted) {
  const handle = activeAudioHandles.get(audioId);
  if (!handle) return;
  handle.muted = muted;
  if (handle.gainNode) {
    handle.gainNode.gain.value = muted ? 0 : handle.volume;
  }
}

/**
 * Set audio volume (0.0 to 1.0).
 * @param {number} audioId
 * @param {number} volume
 */
export function moqSetAudioVolume(audioId, volume) {
  const handle = activeAudioHandles.get(audioId);
  if (!handle) return;
  handle.volume = volume;
  if (handle.gainNode && !handle.muted) {
    handle.gainNode.gain.value = volume;
  }
}

/**
 * Get audio state.
 * @param {number} audioId
 * @returns {{ stalled: boolean, timestampMs: number }}
 */
export function moqGetAudioState(audioId) {
  const handle = activeAudioHandles.get(audioId);
  if (!handle) return { stalled: true, timestampMs: 0 };
  return { stalled: handle.stalled, timestampMs: handle.timestampMs };
}

/**
 * Close an audio handle.
 * @param {number} audioId
 */
export function moqCloseAudio(audioId) {
  const handle = activeAudioHandles.get(audioId);
  if (!handle) return;
  handle.close();
  activeAudioHandles.delete(audioId);
}

/**
 * Get session stats.
 * @param {number} sessionId
 * @returns {{ videoFrames: number, audioFrames: number, state: string }}
 */
export function moqGetStats(sessionId) {
  const session = activeSessions.get(sessionId);
  if (!session) return { videoFrames: 0, audioFrames: 0, state: "closed" };
  return {
    videoFrames: session.stats.videoFrames,
    audioFrames: session.stats.audioFrames,
    state: session.state,
  };
}

/**
 * Get extended stats for diagnostics overlay.
 * @param {number} sessionId
 * @returns {Object} Extended stats object
 */
export function moqGetExtendedStats(sessionId) {
  const session = activeSessions.get(sessionId);
  if (!session) return null;

  const stats = {
    connectionVersion: null,
    videoDecodeQueueSize: 0,
    audioDecodeQueueSize: 0,
    audioContextState: "closed",
    audioStalled: true,
    audioTimestampMs: 0,
    audioFrameCount: 0,
    audioUnderflowSamples: 0,
    audioBufferLength: 0,
    audioBufferCapacity: 0,
  };

  // Connection version
  if (session.connection?.version != null) {
    stats.connectionVersion = session.connection.version;
  }

  // Video decoder queue depth
  for (const [, handle] of activeVideoDecoders) {
    if (handle.sessionId === sessionId && handle.decoder?.state === "configured") {
      stats.videoDecodeQueueSize = handle.decoder.decodeQueueSize ?? 0;
      break;
    }
  }

  // Audio decoder queue depth + buffer stats
  for (const [, handle] of activeAudioHandles) {
    if (handle.sessionId === sessionId) {
      if (handle.decoder?.state === "configured") {
        stats.audioDecodeQueueSize = handle.decoder.decodeQueueSize ?? 0;
      }
      if (handle.context) {
        stats.audioContextState = handle.context.state;
      }
      stats.audioStalled = handle.stalled;
      stats.audioTimestampMs = handle.timestampMs;
      stats.audioFrameCount = handle.frameCount;
      stats.audioUnderflowSamples = handle.underflowSamples;
      stats.audioBufferLength = handle.bufferLength;
      stats.audioBufferCapacity = handle.bufferCapacity;
      break;
    }
  }

  return stats;
}

// Debug: expose internals
window.__moqTransportBridge = { activeSessions, activeVideoDecoders, activeAudioHandles };
