/**
 * lumina-video MoQ WebCodecs Bridge
 *
 * JavaScript bridge for MoQ live streaming with WebCodecs decoding.
 * This file provides:
 * - WebCodecs VideoDecoder management
 * - Frame callback handling with proper lifecycle
 * - Zero-copy GPU texture operations via copyExternalImageToTexture
 *
 * Architecture:
 *   MoQ NAL units (from Rust/WASM) -> EncodedVideoChunk -> VideoDecoder
 *   -> VideoFrame -> copyExternalImageToTexture -> wgpu texture
 *
 * This file is imported by Rust/WASM via wasm-bindgen.
 */

// Track active decoders for cleanup
const activeDecoders = new Map();
let decoderIdCounter = 0;

/**
 * Decoder state object returned to Rust
 */
class MoqDecoderHandle {
  constructor(id, decoder, onFrame, onError) {
    this.id = id;
    this.decoder = decoder;
    this.onFrame = onFrame;
    this.onError = onError;
    this.lastFrame = null;
    this.frameCount = 0;
    this.errorMessage = null;
    this.closed = false;
  }
}

/**
 * Creates a new WebCodecs VideoDecoder for MoQ streams.
 *
 * @param {string} codec - Codec string (e.g., "avc1.42E01E" for H.264 baseline)
 * @param {number} width - Video width in pixels (optional, for optimization)
 * @param {number} height - Video height in pixels (optional, for optimization)
 * @param {Uint8Array|null} description - Codec-specific description (SPS/PPS for H.264)
 * @param {Function} onFrame - Callback invoked with VideoFrame when decoded
 * @param {Function} onError - Callback invoked with error message on failure
 * @returns {number} Decoder handle ID
 * @throws {Error} If VideoDecoder is not supported or configuration fails
 */
export function createMoqDecoder(codec, width, height, description, onFrame, onError) {
  if (typeof VideoDecoder === 'undefined') {
    throw new Error('WebCodecs VideoDecoder is not supported in this browser');
  }

  const id = decoderIdCounter++;

  // Create the decoder with output and error callbacks
  const decoder = new VideoDecoder({
    output: (frame) => {
      const handle = activeDecoders.get(id);
      if (!handle || handle.closed) {
        frame.close();
        return;
      }

      // Close previous frame to prevent memory leaks
      if (handle.lastFrame) {
        handle.lastFrame.close();
      }
      handle.lastFrame = frame;
      handle.frameCount++;

      // Invoke Rust callback with frame info
      // Note: We pass frame metadata, not the frame itself (Rust will call getLastFrame)
      try {
        onFrame(frame.timestamp, frame.displayWidth, frame.displayHeight, frame.duration || 0);
      } catch (e) {
        console.error('[moq-bridge] Error in frame callback:', e);
      }
    },
    error: (e) => {
      const handle = activeDecoders.get(id);
      if (handle) {
        handle.errorMessage = e.message || 'Unknown decoder error';
        handle.closed = true;
      }
      console.error('[moq-bridge] VideoDecoder error:', e);
      try {
        onError(e.message || 'Unknown decoder error');
      } catch (err) {
        console.error('[moq-bridge] Error in error callback:', err);
      }
    }
  });

  // Build decoder configuration
  const config = {
    codec: codec,
    optimizeForLatency: true, // Critical for live streaming
  };

  // Add optional parameters
  if (width > 0) config.codedWidth = width;
  if (height > 0) config.codedHeight = height;
  if (description && description.length > 0) {
    config.description = description;
  }

  // Configure the decoder
  decoder.configure(config);

  const handle = new MoqDecoderHandle(id, decoder, onFrame, onError);
  activeDecoders.set(id, handle);

  console.debug('[moq-bridge] Created decoder', id, 'with config:', config);
  return id;
}

/**
 * Checks if a codec configuration is supported.
 *
 * @param {string} codec - Codec string to check
 * @param {number} width - Video width (optional)
 * @param {number} height - Video height (optional)
 * @returns {Promise<boolean>} Whether the configuration is supported
 */
export async function isCodecSupported(codec, width, height) {
  if (typeof VideoDecoder === 'undefined') {
    return false;
  }

  const config = {
    codec: codec,
    optimizeForLatency: true,
  };

  if (width > 0) config.codedWidth = width;
  if (height > 0) config.codedHeight = height;

  try {
    const support = await VideoDecoder.isConfigSupported(config);
    return support.supported === true;
  } catch (e) {
    console.warn('[moq-bridge] Codec support check failed:', e);
    return false;
  }
}

/**
 * Decodes an encoded video chunk (NAL unit).
 *
 * @param {number} decoderId - Decoder handle ID
 * @param {Uint8Array} data - Encoded frame data (NAL units)
 * @param {number} timestampUs - Presentation timestamp in microseconds
 * @param {boolean} isKeyframe - Whether this is a keyframe
 * @throws {Error} If decoder not found or in error state
 */
export function decodeChunk(decoderId, data, timestampUs, isKeyframe) {
  const handle = activeDecoders.get(decoderId);
  if (!handle) {
    throw new Error(`Decoder ${decoderId} not found`);
  }
  if (handle.closed) {
    throw new Error(`Decoder ${decoderId} is closed`);
  }

  const chunk = new EncodedVideoChunk({
    type: isKeyframe ? 'key' : 'delta',
    timestamp: timestampUs,
    data: data,
  });

  handle.decoder.decode(chunk);
}

/**
 * Gets the last decoded VideoFrame for texture upload.
 * The frame must be closed after use by calling releaseFrame().
 *
 * @param {number} decoderId - Decoder handle ID
 * @returns {VideoFrame|null} The last decoded frame, or null if none available
 */
export function getLastFrame(decoderId) {
  const handle = activeDecoders.get(decoderId);
  if (!handle || !handle.lastFrame) {
    return null;
  }
  // Clone the frame so caller can work with it independently
  // The original stays in lastFrame for the next getLastFrame call
  try {
    return handle.lastFrame.clone();
  } catch (e) {
    // Frame may have been closed already
    return null;
  }
}

/**
 * Gets the decoder state information.
 *
 * @param {number} decoderId - Decoder handle ID
 * @returns {Object} State object with decodeQueueSize, state, frameCount
 */
export function getDecoderState(decoderId) {
  const handle = activeDecoders.get(decoderId);
  if (!handle) {
    return {
      state: 'closed',
      decodeQueueSize: 0,
      frameCount: 0,
      errorMessage: null,
    };
  }

  return {
    state: handle.decoder.state,
    decodeQueueSize: handle.decoder.decodeQueueSize,
    frameCount: handle.frameCount,
    errorMessage: handle.errorMessage,
  };
}

/**
 * Flushes the decoder, waiting for all pending frames to be output.
 *
 * @param {number} decoderId - Decoder handle ID
 * @returns {Promise<void>}
 */
export async function flushDecoder(decoderId) {
  const handle = activeDecoders.get(decoderId);
  if (!handle || handle.closed) {
    return;
  }

  try {
    await handle.decoder.flush();
  } catch (e) {
    console.warn('[moq-bridge] Flush failed:', e);
  }
}

/**
 * Resets the decoder state for seeking or error recovery.
 *
 * @param {number} decoderId - Decoder handle ID
 */
export function resetDecoder(decoderId) {
  const handle = activeDecoders.get(decoderId);
  if (!handle || handle.closed) {
    return;
  }

  try {
    handle.decoder.reset();
    // Clear cached frame on reset
    if (handle.lastFrame) {
      handle.lastFrame.close();
      handle.lastFrame = null;
    }
  } catch (e) {
    console.warn('[moq-bridge] Reset failed:', e);
  }
}

/**
 * Closes and cleans up a decoder.
 *
 * @param {number} decoderId - Decoder handle ID
 */
export function closeDecoder(decoderId) {
  const handle = activeDecoders.get(decoderId);
  if (!handle) {
    return;
  }

  handle.closed = true;

  // Close the last frame if any
  if (handle.lastFrame) {
    try {
      handle.lastFrame.close();
    } catch (e) {
      // Ignore close errors
    }
    handle.lastFrame = null;
  }

  // Close the decoder
  try {
    handle.decoder.close();
  } catch (e) {
    console.warn('[moq-bridge] Error closing decoder:', e);
  }

  activeDecoders.delete(decoderId);
  console.debug('[moq-bridge] Closed decoder', decoderId);
}

/**
 * Copies a VideoFrame to a WebGPU texture using copyExternalImageToTexture.
 * This is the zero-copy path for GPU rendering.
 *
 * @param {VideoFrame} frame - The VideoFrame to copy
 * @param {GPUDevice} device - WebGPU device
 * @param {GPUTexture} texture - Target WebGPU texture
 * @returns {boolean} Whether the copy succeeded
 */
export function copyFrameToTexture(frame, device, texture) {
  if (!frame || !device || !texture) {
    return false;
  }

  try {
    device.queue.copyExternalImageToTexture(
      {
        source: frame,
        origin: { x: 0, y: 0 },
        flipY: false,
      },
      {
        texture: texture,
        origin: { x: 0, y: 0, z: 0 },
        aspect: 'all',
        mipLevel: 0,
        colorSpace: 'srgb',
        premultipliedAlpha: false,
      },
      {
        width: frame.displayWidth,
        height: frame.displayHeight,
        depthOrArrayLayers: 1,
      }
    );
    return true;
  } catch (e) {
    console.error('[moq-bridge] copyExternalImageToTexture failed:', e);
    return false;
  }
}

/**
 * Closes a VideoFrame to release its resources.
 * Must be called after getLastFrame() when done with the frame.
 *
 * @param {VideoFrame} frame - The frame to close
 */
export function closeFrame(frame) {
  if (frame) {
    try {
      frame.close();
    } catch (e) {
      // Ignore close errors (frame may already be closed)
    }
  }
}

/**
 * Checks if WebCodecs VideoDecoder is available.
 *
 * @returns {boolean} Whether VideoDecoder is supported
 */
export function isWebCodecsSupported() {
  return typeof VideoDecoder !== 'undefined' &&
         typeof EncodedVideoChunk !== 'undefined' &&
         typeof VideoFrame !== 'undefined';
}

/**
 * Gets the WebCodecs API capabilities.
 *
 * @returns {Object} Capabilities object
 */
export function getWebCodecsCapabilities() {
  return {
    videoDecoder: typeof VideoDecoder !== 'undefined',
    videoEncoder: typeof VideoEncoder !== 'undefined',
    audioDecoder: typeof AudioDecoder !== 'undefined',
    audioEncoder: typeof AudioEncoder !== 'undefined',
    videoFrame: typeof VideoFrame !== 'undefined',
    encodedVideoChunk: typeof EncodedVideoChunk !== 'undefined',
  };
}

// Export for debugging
window.__moqBridgeDecoders = activeDecoders;
