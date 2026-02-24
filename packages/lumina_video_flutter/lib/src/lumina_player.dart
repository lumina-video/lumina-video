import 'dart:async';
import 'dart:ui' show Size;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Playback state, matching native enum indices on both platforms.
enum LuminaPlaybackState { loading, ready, playing, paused, ended, error }

/// Error from the native video player.
class LuminaVideoException implements Exception {
  const LuminaVideoException({
    required this.code,
    this.message,
    this.platformCode,
  });

  /// Error code string (e.g. "LUMINA_INIT_FAILED", "EXOPLAYER_ERROR").
  final String code;

  /// Human-readable error message, if available.
  final String? message;

  /// Platform-specific error code (C error code on iOS, ExoPlayer error code on Android).
  final int? platformCode;

  @override
  String toString() {
    final parts = ['LuminaVideoException($code'];
    if (message != null) parts.add(', $message');
    if (platformCode != null) parts.add(', platformCode=$platformCode');
    parts.add(')');
    return parts.join();
  }
}

/// Immutable snapshot of a [LuminaPlayer]'s current state.
class LuminaPlayerValue {
  const LuminaPlayerValue({
    this.textureId = -1,
    this.state = LuminaPlaybackState.loading,
    this.position = 0.0,
    this.duration = -1.0,
    this.videoSize,
    this.error,
    this.fps,
    this.maxFps,
    this.zeroCopy,
    this.videoCodec,
    this.audioCodec,
    this.format,
  });

  const LuminaPlayerValue.uninitialized()
      : textureId = -1,
        state = LuminaPlaybackState.loading,
        position = 0.0,
        duration = -1.0,
        videoSize = null,
        error = null,
        fps = null,
        maxFps = null,
        zeroCopy = null,
        videoCodec = null,
        audioCodec = null,
        format = null;

  /// Flutter texture ID, or -1 if not yet registered.
  final int textureId;

  /// Current playback state.
  final LuminaPlaybackState state;

  /// Current position in seconds.
  final double position;

  /// Duration in seconds, or -1.0 if unknown.
  final double duration;

  /// Video dimensions, or null if not yet known.
  final Size? videoSize;

  /// Error details when [state] == [LuminaPlaybackState.error].
  final LuminaVideoException? error;

  /// Measured decode FPS (frames polled per second), or null if not yet measured.
  final double? fps;

  /// Display max refresh rate (Hz), or null if unknown.
  final int? maxFps;

  /// Whether zero-copy rendering is active (IOSurface on iOS), or null if unknown.
  final bool? zeroCopy;

  /// Video decoder/codec name (e.g. "VideoToolbox", "H.264"), or null if unknown.
  final String? videoCodec;

  /// Audio decoder/codec name (e.g. "AAC (FFmpeg)"), or null if unknown.
  final String? audioCodec;

  /// Container format (e.g. "MP4", "WEBM"), or null if unknown.
  final String? format;

  /// Whether the player has a valid texture ready for rendering.
  bool get isInitialized => textureId >= 0;

  LuminaPlayerValue copyWith({
    int? textureId,
    LuminaPlaybackState? state,
    double? position,
    double? duration,
    Size? videoSize,
    LuminaVideoException? error,
    double? fps,
    int? maxFps,
    bool? zeroCopy,
    String? videoCodec,
    String? audioCodec,
    String? format,
  }) {
    return LuminaPlayerValue(
      textureId: textureId ?? this.textureId,
      state: state ?? this.state,
      position: position ?? this.position,
      duration: duration ?? this.duration,
      videoSize: videoSize ?? this.videoSize,
      error: error ?? this.error,
      fps: fps ?? this.fps,
      maxFps: maxFps ?? this.maxFps,
      zeroCopy: zeroCopy ?? this.zeroCopy,
      videoCodec: videoCodec ?? this.videoCodec,
      audioCodec: audioCodec ?? this.audioCodec,
      format: format ?? this.format,
    );
  }

  @override
  String toString() =>
      'LuminaPlayerValue(texture=$textureId, state=$state, '
      'pos=${position.toStringAsFixed(1)}, dur=${duration.toStringAsFixed(1)})';
}

/// Internal lifecycle states.
enum _LifecycleState { uninitialized, creating, ready, error, disposed }

/// Hardware-accelerated video player using platform-native decoders.
///
/// Uses lumina-video C FFI on iOS (zero-copy IOSurface textures) and
/// ExoPlayer on Android (SurfaceTexture).
///
/// Usage:
/// ```dart
/// final player = LuminaPlayer();
/// await player.open('https://example.com/video.mp4');
/// await player.play();
/// // In build:
/// ValueListenableBuilder<LuminaPlayerValue>(
///   valueListenable: player,
///   builder: (_, val, __) => val.isInitialized
///       ? Texture(textureId: val.textureId)
///       : const CircularProgressIndicator(),
/// )
/// // Cleanup:
/// await player.close();
/// player.dispose();
/// ```
class LuminaPlayer extends ValueNotifier<LuminaPlayerValue> {
  /// Creates a new player instance.
  ///
  /// [onCleanupError] is called when best-effort cleanup (dispose/tryDestroy)
  /// fails. Defaults to [debugPrint]. Pass a custom callback for structured
  /// logging (e.g., Sentry, Crashlytics).
  LuminaPlayer({
    void Function(String message)? onCleanupError,
  })  : _onCleanupError = onCleanupError ?? debugPrint,
        super(const LuminaPlayerValue.uninitialized());

  static const _channel = MethodChannel('lumina_video_flutter');
  static const _createTimeout = Duration(seconds: 10);

  /// Instance-level cleanup error callback. No global mutable state.
  final void Function(String message) _onCleanupError;

  /// Safe wrapper — ensures callback throws never break cleanup control flow.
  void _reportCleanupError(String message) {
    try {
      _onCleanupError(message);
    } catch (_) {
      // Swallow — telemetry failure must never mask original cleanup failure.
    }
  }

  int? _playerId;
  StreamSubscription<dynamic>? _eventSub;
  Future<void>? _closeFuture;
  Completer<void>? _createCompleter;
  _LifecycleState _lifecycle = _LifecycleState.uninitialized;
  bool _isDisposed = false;
  bool _closeRequested = false;

  bool get _isTerminal => _lifecycle == _LifecycleState.disposed;
  bool get _shouldAbort => _isDisposed || _closeRequested;

  /// Best-effort native destroy. Swallows UNKNOWN_PLAYER (idempotent).
  Future<void> _tryDestroy() async {
    if (_playerId == null) return;
    try {
      await _channel.invokeMethod<void>('destroy', {'playerId': _playerId});
      _playerId = null;
    } catch (e) {
      if (e is PlatformException && e.code == 'UNKNOWN_PLAYER') {
        _playerId = null;
      } else {
        _reportCleanupError(
          'LuminaPlayer: _tryDestroy failed (playerId=$_playerId): $e',
        );
      }
    }
  }

  /// Opens a video URL for playback. Must be called exactly once.
  ///
  /// Throws [LuminaVideoException] if initialization fails.
  /// Throws [StateError] if already opened or closed during init.
  Future<void> open(String url) async {
    if (_isDisposed) throw StateError('LuminaPlayer is disposed');
    if (_lifecycle != _LifecycleState.uninitialized) {
      throw StateError('LuminaPlayer already opened');
    }
    _lifecycle = _LifecycleState.creating;
    _createCompleter = Completer<void>();

    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'create',
        {'url': url},
      );
      if (result == null) {
        throw const LuminaVideoException(
          code: 'NULL_RESPONSE',
          message: 'Platform returned null from create',
        );
      }
      _playerId = (result['playerId'] as num).toInt();
      if (!_createCompleter!.isCompleted) _createCompleter!.complete();

      if (_shouldAbort) {
        await _tryDestroy();
        throw StateError('Player was closed during initialization');
      }

      final snap = _parseSnapshot(result);
      value = value.copyWith(
        textureId: result['textureId'] as int,
        state: snap.state,
        position: snap.position,
        duration: snap.duration,
        videoSize: snap.videoSize,
        error: snap.error,
        fps: snap.fps,
        maxFps: snap.maxFps,
        zeroCopy: snap.zeroCopy,
        videoCodec: snap.videoCodec,
        audioCodec: snap.audioCodec,
        format: snap.format,
      );

      if (snap.state == LuminaPlaybackState.error) {
        if (!_isTerminal) _lifecycle = _LifecycleState.error;
        throw snap.error!;
      }

      final events = EventChannel('lumina_video_flutter/events/$_playerId');
      _eventSub = events.receiveBroadcastStream().listen(
        _handleEvent,
        onError: (Object e) {
          _eventSub?.cancel();
          _eventSub = null;
          if (!_isTerminal && !_shouldAbort) {
            _lifecycle = _LifecycleState.error;
            value = value.copyWith(
              state: LuminaPlaybackState.error,
              error: LuminaVideoException(code: 'STREAM_ERROR', message: '$e'),
            );
          }
        },
        onDone: () {
          _eventSub = null;
        },
        cancelOnError: false,
      );

      if (!_isTerminal) _lifecycle = _LifecycleState.ready;
    } catch (e) {
      if (!_createCompleter!.isCompleted) {
        _createCompleter!.complete();
      }
      if (_playerId != null && !_isTerminal) {
        await _tryDestroy();
        if (_playerId != null) {
          _reportCleanupError(
            'LuminaPlayer: native player leaked (playerId=$_playerId). '
            'detachFromEngine will clean up on hot restart/termination.',
          );
        }
      }
      if (!_isTerminal && !_shouldAbort && _lifecycle != _LifecycleState.error) {
        _lifecycle = _LifecycleState.error;
      }
      rethrow;
    }
  }

  /// Starts or resumes playback.
  Future<void> play() async {
    if (_closeRequested || _lifecycle != _LifecycleState.ready) {
      throw StateError('not ready');
    }
    await _channel.invokeMethod<void>('play', {'playerId': _playerId});
  }

  /// Pauses playback.
  Future<void> pause() async {
    if (_closeRequested || _lifecycle != _LifecycleState.ready) {
      throw StateError('not ready');
    }
    await _channel.invokeMethod<void>('pause', {'playerId': _playerId});
  }

  /// Seeks to a position in seconds.
  Future<void> seek(double positionSeconds) async {
    if (_closeRequested || _lifecycle != _LifecycleState.ready) {
      throw StateError('not ready');
    }
    await _channel.invokeMethod<void>(
      'seek',
      {'playerId': _playerId, 'position': positionSeconds},
    );
  }

  /// Sets the muted state.
  Future<void> setMuted(bool muted) async {
    if (_closeRequested || _lifecycle != _LifecycleState.ready) {
      throw StateError('not ready');
    }
    await _channel.invokeMethod<void>(
      'setMuted',
      {'playerId': _playerId, 'muted': muted},
    );
  }

  /// Sets volume (0-100).
  Future<void> setVolume(int volume) async {
    if (_closeRequested || _lifecycle != _LifecycleState.ready) {
      throw StateError('not ready');
    }
    await _channel.invokeMethod<void>(
      'setVolume',
      {'playerId': _playerId, 'volume': volume},
    );
  }

  /// Async cleanup. Frees Dart resources and native player.
  ///
  /// After calling close(), the instance is permanently unusable for playback
  /// regardless of outcome. Throws on failure so caller can retry native cleanup.
  Future<void> close() {
    _closeRequested = true;
    if (_isTerminal) return Future<void>.value();
    return _closeFuture ??= _doClose();
  }

  Future<void> _doClose() async {
    _eventSub?.cancel();
    _eventSub = null;

    if (_createCompleter != null && !_createCompleter!.isCompleted) {
      try {
        await _createCompleter!.future.timeout(_createTimeout);
      } on TimeoutException {
        _closeFuture = null;
        throw TimeoutException(
          'Native create still in-flight after $_createTimeout. '
          'Retry close() or rely on detachFromEngine.',
          _createTimeout,
        );
      } catch (_) {
        // Proceed to destroy.
      }
    }

    if (_playerId != null) {
      try {
        await _channel.invokeMethod<void>('destroy', {'playerId': _playerId});
      } catch (e) {
        if (e is PlatformException && e.code == 'UNKNOWN_PLAYER') {
          // Idempotent: player already gone.
        } else {
          _closeFuture = null;
          rethrow;
        }
      }
      _playerId = null;
    }
    _lifecycle = _LifecycleState.disposed;
  }

  @override
  void dispose() {
    _isDisposed = true;
    close().catchError((Object e) {
      _reportCleanupError('LuminaPlayer.dispose: close failed: $e');
    });
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Event handling
  // ---------------------------------------------------------------------------

  void _handleEvent(dynamic event) {
    if (_shouldAbort) return;
    if (event is! Map) return;

    final rawV = event['v'];
    final version = (rawV is num) ? rawV.toInt() : null;
    if (version == null || version != 1) return;

    try {
      final snap = _parseSnapshot(event, fallbackState: value.state);
      value = value.copyWith(
        state: snap.state,
        position: snap.position,
        duration: snap.duration,
        videoSize: snap.videoSize,
        error: snap.error,
        fps: snap.fps,
        maxFps: snap.maxFps,
        zeroCopy: snap.zeroCopy,
        videoCodec: snap.videoCodec,
        audioCodec: snap.audioCodec,
        format: snap.format,
      );
    } catch (e) {
      if (!_isTerminal && !_shouldAbort) {
        _lifecycle = _LifecycleState.error;
        value = value.copyWith(
          state: LuminaPlaybackState.error,
          error: LuminaVideoException(code: 'EVENT_PARSE_ERROR', message: '$e'),
        );
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Snapshot parsing — shared by open() initial response and event stream
  // ---------------------------------------------------------------------------

  static ({
    LuminaPlaybackState state,
    double position,
    double duration,
    Size? videoSize,
    LuminaVideoException? error,
    double? fps,
    int? maxFps,
    bool? zeroCopy,
    String? videoCodec,
    String? audioCodec,
    String? format,
  }) _parseSnapshot(
    Map<dynamic, dynamic> map, {
    LuminaPlaybackState? fallbackState,
  }) {
    final rawState = map['state'];
    final stateIndex = (rawState is num) ? rawState.toInt() : -1;
    final state = (stateIndex >= 0 &&
            stateIndex < LuminaPlaybackState.values.length)
        ? LuminaPlaybackState.values[stateIndex]
        : (fallbackState ?? LuminaPlaybackState.loading);

    final rawPos = map['position'];
    final position = (rawPos is num) ? rawPos.toDouble() : 0.0;
    final rawDur = map['duration'];
    final duration = (rawDur is num) ? rawDur.toDouble() : -1.0;

    LuminaVideoException? error;
    var resolvedState = state;
    final rawError = map['error'];
    if (rawError is Map) {
      final rawCode = rawError['code'];
      final rawMsg = rawError['message'];
      final rawPlatCode = rawError['platformCode'];
      error = LuminaVideoException(
        code: (rawCode is String) ? rawCode : 'UNKNOWN',
        message: (rawMsg is String) ? rawMsg : null,
        platformCode: (rawPlatCode is num) ? rawPlatCode.toInt() : null,
      );
      resolvedState = LuminaPlaybackState.error;
    } else if (state == LuminaPlaybackState.error) {
      error = const LuminaVideoException(
        code: 'UNKNOWN',
        message: 'No error details from platform',
      );
    }

    // Diagnostic fields — all optional
    final rawFps = map['fps'];
    final fps = (rawFps is num) ? rawFps.toDouble() : null;
    final rawMaxFps = map['maxFps'];
    final maxFps = (rawMaxFps is num) ? rawMaxFps.toInt() : null;
    final rawZeroCopy = map['zeroCopy'];
    final zeroCopy = (rawZeroCopy is bool) ? rawZeroCopy : null;
    final rawVideoCodec = map['videoCodec'];
    final videoCodec = (rawVideoCodec is String) ? rawVideoCodec : null;
    final rawAudioCodec = map['audioCodec'];
    final audioCodec = (rawAudioCodec is String) ? rawAudioCodec : null;
    final rawFormat = map['format'];
    final format = (rawFormat is String) ? rawFormat : null;

    return (
      state: resolvedState,
      position: position,
      duration: duration,
      videoSize: _parseSize(map),
      error: error,
      fps: fps,
      maxFps: maxFps,
      zeroCopy: zeroCopy,
      videoCodec: videoCodec,
      audioCodec: audioCodec,
      format: format,
    );
  }

  static Size? _parseSize(Map<dynamic, dynamic> map) {
    final rawW = map['videoWidth'];
    final rawH = map['videoHeight'];
    if (rawW is num && rawH is num && rawW > 0 && rawH > 0) {
      return Size(rawW.toDouble(), rawH.toDouble());
    }
    return null;
  }
}
