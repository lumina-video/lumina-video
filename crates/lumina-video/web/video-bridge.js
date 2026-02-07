/**
 * lumina-video Web Video Bridge
 *
 * JavaScript bridge for web video playback with:
 * - HLS.js integration for adaptive streaming
 * - requestVideoFrameCallback for frame-accurate sync
 * - Performance monitoring and buffer statistics
 *
 * This file is imported by Rust/WASM via wasm-bindgen.
 */

// HLS.js configuration for best-in-class streaming performance
const HLS_CONFIG = {
  // Buffer configuration for optimal playback
  maxBufferLength: 30,           // Max buffer ahead in seconds
  maxMaxBufferLength: 600,       // Max buffer for VOD content
  maxBufferSize: 60 * 1000000,   // 60MB max buffer size
  maxBufferHole: 0.5,            // Max gap to skip

  // Startup optimization
  startLevel: -1,                // Auto quality selection on start
  autoStartLoad: true,           // Start loading immediately
  startPosition: -1,             // Start from beginning

  // ABR (Adaptive Bitrate) tuning for fast quality switching
  abrEwmaDefaultEstimate: 500000,   // Default bandwidth estimate (500kbps)
  abrEwmaFastLive: 3.0,             // Fast EMA factor for live
  abrEwmaSlowLive: 9.0,             // Slow EMA factor for live
  abrEwmaFastVoD: 3.0,              // Fast EMA factor for VOD
  abrEwmaSlowVoD: 9.0,              // Slow EMA factor for VOD
  abrBandWidthFactor: 0.95,         // Conservative bandwidth usage
  abrBandWidthUpFactor: 0.7,        // More conservative for upgrades

  // Low latency settings (when applicable)
  lowLatencyMode: false,            // Enable for live streams if needed
  backBufferLength: 30,             // Keep 30s of back buffer

  // Error recovery
  manifestLoadingMaxRetry: 4,
  manifestLoadingRetryDelay: 1000,
  levelLoadingMaxRetry: 4,
  levelLoadingRetryDelay: 1000,
  fragLoadingMaxRetry: 6,
  fragLoadingRetryDelay: 1000,

  // Performance
  enableWorker: true,               // Use web worker for parsing
  enableSoftwareAES: true,          // Software AES for encrypted streams
};

/**
 * Initializes HLS.js and attaches it to a video element.
 *
 * @param {HTMLVideoElement} video - The video element to attach to
 * @param {string} url - The HLS manifest URL (.m3u8)
 * @returns {Hls} The HLS.js instance
 * @throws {Error} If HLS.js is not supported or fails to initialize
 */
export function initHls(video, url) {
  // Check if HLS.js is available (should be loaded via CDN or bundled)
  if (typeof Hls === 'undefined') {
    throw new Error('HLS.js is not loaded. Include hls.js in your HTML.');
  }

  if (!Hls.isSupported()) {
    throw new Error('HLS.js is not supported in this browser');
  }

  const hls = new Hls(HLS_CONFIG);

  // Track recovery attempts to prevent infinite loops on persistent errors
  let networkRecoveryAttempts = 0;
  let mediaRecoveryAttempts = 0;
  const MAX_RECOVERY_ATTEMPTS = 3;

  // Set up event listeners for debugging and monitoring
  hls.on(Hls.Events.MANIFEST_PARSED, (event, data) => {
    console.debug('[lumina-video] HLS manifest parsed:', data.levels.length, 'quality levels');
    // Reset recovery counters on successful manifest parse
    networkRecoveryAttempts = 0;
    mediaRecoveryAttempts = 0;
    // Auto-play after manifest is ready (if allowed)
    video.play().catch(() => {
      // Autoplay blocked - user interaction required
      console.debug('[lumina-video] Autoplay blocked, waiting for user interaction');
    });
  });

  hls.on(Hls.Events.LEVEL_SWITCHED, (event, data) => {
    const level = hls.levels[data.level];
    console.debug('[lumina-video] Quality switched to:', level?.height + 'p', '@', Math.round(level?.bitrate / 1000), 'kbps');
  });

  hls.on(Hls.Events.ERROR, (event, data) => {
    if (data.fatal) {
      console.error('[lumina-video] Fatal HLS error:', data.type, data.details);
      switch (data.type) {
        case Hls.ErrorTypes.NETWORK_ERROR:
          networkRecoveryAttempts++;
          if (networkRecoveryAttempts <= MAX_RECOVERY_ATTEMPTS) {
            console.log('[lumina-video] Attempting to recover from network error (attempt', networkRecoveryAttempts, 'of', MAX_RECOVERY_ATTEMPTS + ')...');
            hls.startLoad();
          } else {
            console.error('[lumina-video] Max network recovery attempts reached, destroying HLS instance');
            hls.destroy();
          }
          break;
        case Hls.ErrorTypes.MEDIA_ERROR:
          mediaRecoveryAttempts++;
          if (mediaRecoveryAttempts <= MAX_RECOVERY_ATTEMPTS) {
            console.log('[lumina-video] Attempting to recover from media error (attempt', mediaRecoveryAttempts, 'of', MAX_RECOVERY_ATTEMPTS + ')...');
            hls.recoverMediaError();
          } else {
            console.error('[lumina-video] Max media recovery attempts reached, destroying HLS instance');
            hls.destroy();
          }
          break;
        default:
          console.error('[lumina-video] Unrecoverable error, destroying HLS instance');
          hls.destroy();
          break;
      }
    } else {
      console.warn('[lumina-video] Non-fatal HLS error:', data.type, data.details);
    }
  });

  // Bandwidth estimation logging (helpful for debugging)
  hls.on(Hls.Events.FRAG_LOADED, (event, data) => {
    const stats = data.frag.stats;
    if (stats && stats.total && stats.loading) {
      const bandwidth = Math.round((stats.total * 8) / (stats.loading.end - stats.loading.start) * 1000);
      console.debug('[lumina-video] Fragment loaded, bandwidth estimate:', Math.round(bandwidth / 1000), 'kbps');
    }
  });

  // Attach to video and load source
  hls.loadSource(url);
  hls.attachMedia(video);

  return hls;
}

/**
 * Destroys an HLS.js instance and cleans up resources.
 *
 * @param {Hls} hls - The HLS.js instance to destroy
 */
export function destroyHls(hls) {
  if (hls && typeof hls.destroy === 'function') {
    hls.destroy();
  }
}

/**
 * Registers a requestVideoFrameCallback on the video element.
 * Falls back to requestAnimationFrame if not supported.
 *
 * @param {HTMLVideoElement} video - The video element
 * @param {Function} callback - Callback function(now, metadata)
 */
export function requestVideoFrameCallback(video, callback) {
  if ('requestVideoFrameCallback' in video) {
    // Modern browsers with frame-accurate callback
    const wrappedCallback = (now, metadata) => {
      // Stop re-registering if video ended or removed from DOM
      if (video.ended || !video.isConnected) {
        return;
      }
      callback(now, metadata);
      // Re-register for next frame
      video.requestVideoFrameCallback(wrappedCallback);
    };
    video.requestVideoFrameCallback(wrappedCallback);
  } else {
    // Fallback for older browsers with cancellation support
    console.warn('[lumina-video] requestVideoFrameCallback not supported, using requestAnimationFrame fallback');
    let rafId = null;

    const rafCallback = () => {
      // Check termination conditions (stop if ended OR disconnected)
      if (video.ended || !video.isConnected) {
        if (rafId !== null) {
          cancelAnimationFrame(rafId);
          rafId = null;
        }
        return;
      }

      if (!video.paused && !video.ended) {
        callback(performance.now(), {
          presentationTime: video.currentTime * 1000,
          expectedDisplayTime: performance.now(),
          width: video.videoWidth,
          height: video.videoHeight,
          mediaTime: video.currentTime,
        });
      }
      rafId = requestAnimationFrame(rafCallback);
    };

    rafId = requestAnimationFrame(rafCallback);

    // Store cancellation function on video element for explicit cleanup
    video.__eguiVidRafCancel = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
    };
  }
}

/**
 * Gets the available HLS quality levels.
 *
 * @param {Hls} hls - The HLS.js instance
 * @returns {Array} Array of level objects with bitrate, width, height, codec
 */
export function getHlsLevels(hls) {
  if (!hls || !hls.levels) {
    return [];
  }

  return hls.levels.map((level, index) => ({
    index,
    bitrate: level.bitrate || 0,
    width: level.width || 0,
    height: level.height || 0,
    codec: level.videoCodec || level.audioCodec || 'unknown',
  }));
}

/**
 * Sets the current HLS quality level.
 *
 * @param {Hls} hls - The HLS.js instance
 * @param {number} level - Level index, or -1 for automatic
 */
export function setHlsLevel(hls, level) {
  if (!hls) return;

  if (level === -1) {
    // Enable automatic bitrate selection
    hls.currentLevel = -1;
    hls.nextLevel = -1;
    console.debug('[lumina-video] ABR enabled (automatic quality)');
  } else {
    // Force specific level
    hls.currentLevel = level;
    hls.nextLevel = level;
    const levelInfo = hls.levels[level];
    console.debug('[lumina-video] Quality locked to:', levelInfo?.height + 'p');
  }
}

/**
 * Gets HLS buffer and bandwidth statistics.
 *
 * @param {Hls} hls - The HLS.js instance
 * @returns {Object} Buffer info with bufferLength, bandwidth, currentLevel
 */
export function getHlsBufferInfo(hls) {
  if (!hls || !hls.media) {
    return {
      bufferLength: 0,
      bandwidth: 0,
      currentLevel: -1,
    };
  }

  const video = hls.media;
  const buffered = video.buffered;
  let bufferLength = 0;

  if (buffered.length > 0) {
    // Find buffer range containing current time
    const currentTime = video.currentTime;
    for (let i = 0; i < buffered.length; i++) {
      if (currentTime >= buffered.start(i) && currentTime <= buffered.end(i)) {
        bufferLength = buffered.end(i) - currentTime;
        break;
      }
    }
  }

  return {
    bufferLength,
    bandwidth: hls.bandwidthEstimate || 0,
    currentLevel: hls.currentLevel,
  };
}

/**
 * Preloads an HLS stream without playing.
 * Useful for reducing time-to-first-frame.
 *
 * @param {string} url - The HLS manifest URL
 * @returns {Hls} HLS instance (not attached to video yet)
 */
export function preloadHls(url) {
  if (typeof Hls === 'undefined' || !Hls.isSupported()) {
    return null;
  }

  const hls = new Hls({
    ...HLS_CONFIG,
    autoStartLoad: true,
  });

  hls.loadSource(url);
  return hls;
}

/**
 * Attaches a preloaded HLS instance to a video element.
 *
 * @param {Hls} hls - Preloaded HLS instance
 * @param {HTMLVideoElement} video - Video element to attach to
 */
export function attachPreloadedHls(hls, video) {
  if (hls && video) {
    hls.attachMedia(video);
  }
}

// Export support check functions (run at call time for proper detection)
export function isHLSSupported() {
  return typeof Hls !== 'undefined' && Hls.isSupported();
}

export function isRVFCSupported() {
  return 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
}
