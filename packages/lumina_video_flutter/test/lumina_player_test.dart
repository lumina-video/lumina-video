import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:lumina_video_flutter/lumina_video_flutter.dart';

/// Tracks MethodChannel calls for assertions.
class MockMethodChannel {
  final List<MethodCall> calls = [];
  FutureOr<Object?> Function(MethodCall)? handler;

  void setUp() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
      const MethodChannel('lumina_video_flutter'),
      (MethodCall call) async {
        calls.add(call);
        return handler?.call(call);
      },
    );
  }

  void tearDown() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
      const MethodChannel('lumina_video_flutter'),
      null,
    );
  }

  /// Sends an event to a player's EventChannel.
  void sendEvent(int playerId, Map<String, dynamic> event) {
    ServicesBinding.instance.defaultBinaryMessenger.handlePlatformMessage(
      'lumina_video_flutter/events/$playerId',
      const StandardMethodCodec().encodeSuccessEnvelope(event),
      (_) {},
    );
  }

  int get createCount => calls.where((c) => c.method == 'create').length;
  int get destroyCount => calls.where((c) => c.method == 'destroy').length;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late MockMethodChannel mock;
  int nextId = 1;

  setUp(() {
    mock = MockMethodChannel();
    nextId = 1;
    mock.handler = (call) {
      switch (call.method) {
        case 'create':
          final id = nextId++;
          return {
            'playerId': id,
            'textureId': id * 100,
            'v': 1,
            'state': 0, // loading
            'position': 0.0,
            'duration': -1.0,
          };
        case 'destroy':
        case 'play':
        case 'pause':
        case 'seek':
        case 'setMuted':
        case 'setVolume':
          return null;
        default:
          return null;
      }
    };
    mock.setUp();
  });

  tearDown(() {
    mock.tearDown();
  });

  // Test 1: Lifecycle guard
  test('play() before open() throws StateError', () async {
    final player = LuminaPlayer();
    expect(() => player.play(), throwsStateError);
    player.dispose();
  });

  // Test 2: Idempotent close
  test('close() twice after success is idempotent', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');
    await player.close();
    await player.close(); // second call is no-op
    expect(mock.destroyCount, 1);
    player.dispose();
  });

  // Test 3: Close mid-open
  test('close() during open() force-destroys', () async {
    final createCompleter = Completer<Map<String, dynamic>>();
    mock.handler = (call) {
      if (call.method == 'create') return createCompleter.future;
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    final openFuture = player.open('https://example.com/video.mp4');

    // Close while create is pending
    final closeFuture = player.close();

    // Complete the create
    createCompleter.complete({
      'playerId': 1,
      'textureId': 100,
      'v': 1,
      'state': 0,
      'position': 0.0,
      'duration': -1.0,
    });

    // open() should throw StateError
    await expectLater(openFuture, throwsStateError);
    await closeFuture;
    player.dispose();
  });

  // Test 4: State propagation
  test('event updates value.state to playing', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    mock.sendEvent(1, {
      'v': 1,
      'state': 2, // playing
      'position': 5.0,
      'duration': 60.0,
    });
    await Future<void>.delayed(Duration.zero);

    expect(player.value.state, LuminaPlaybackState.playing);
    expect(player.value.position, 5.0);
    expect(player.value.duration, 60.0);

    await player.close();
    player.dispose();
  });

  // Test 5: Error mapping
  test('event with error sets error state', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    mock.sendEvent(1, {
      'v': 1,
      'state': 5,
      'position': 0.0,
      'duration': -1.0,
      'error': {'code': 'LUMINA_INIT_FAILED', 'message': 'decoder failed'},
    });
    await Future<void>.delayed(Duration.zero);

    expect(player.value.state, LuminaPlaybackState.error);
    expect(player.value.error, isNotNull);
    expect(player.value.error!.code, 'LUMINA_INIT_FAILED');

    await player.close();
    player.dispose();
  });

  // Test 6: Error fallback
  test('error state without error payload synthesizes UNKNOWN error', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    mock.sendEvent(1, {
      'v': 1,
      'state': 5, // error
      'position': 0.0,
      'duration': -1.0,
    });
    await Future<void>.delayed(Duration.zero);

    expect(player.value.state, LuminaPlaybackState.error);
    expect(player.value.error!.code, 'UNKNOWN');

    await player.close();
    player.dispose();
  });

  // Test 7: Event isolation
  test('two players receive independent events', () async {
    final player1 = LuminaPlayer();
    final player2 = LuminaPlayer();
    await player1.open('https://example.com/a.mp4');
    await player2.open('https://example.com/b.mp4');

    mock.sendEvent(1, {
      'v': 1,
      'state': 2, // playing
      'position': 10.0,
      'duration': 60.0,
    });
    mock.sendEvent(2, {
      'v': 1,
      'state': 3, // paused
      'position': 5.0,
      'duration': 30.0,
    });
    await Future<void>.delayed(Duration.zero);

    expect(player1.value.state, LuminaPlaybackState.playing);
    expect(player1.value.position, 10.0);
    expect(player2.value.state, LuminaPlaybackState.paused);
    expect(player2.value.position, 5.0);

    await player1.close();
    await player2.close();
    player1.dispose();
    player2.dispose();
  });

  // Test 8: Late event discard
  test('events after close() are discarded', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');
    await player.close();

    final valueBefore = player.value;
    mock.sendEvent(1, {
      'v': 1,
      'state': 2,
      'position': 99.0,
      'duration': 100.0,
    });
    await Future<void>.delayed(Duration.zero);

    // Value unchanged
    expect(player.value.position, valueBefore.position);
    player.dispose();
  });

  // Test 9: Version gate
  test('event with v:2 is silently dropped', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    final valueBefore = player.value;
    mock.sendEvent(1, {
      'v': 2,
      'state': 2,
      'position': 50.0,
      'duration': 100.0,
    });
    await Future<void>.delayed(Duration.zero);

    expect(player.value.position, valueBefore.position);

    await player.close();
    player.dispose();
  });

  // Test 10: Bounds-safe state
  test('event with out-of-range state keeps previous', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    mock.sendEvent(1, {
      'v': 1,
      'state': 99,
      'position': 1.0,
      'duration': 10.0,
    });
    await Future<void>.delayed(Duration.zero);

    // Position updated but state falls back to previous (loading)
    expect(player.value.position, 1.0);
    expect(player.value.state, LuminaPlaybackState.loading);

    await player.close();
    player.dispose();
  });

  // Test 11: Destroy failure + retry
  test('destroy failure allows close() retry', () async {
    var destroyAttempts = 0;
    mock.handler = (call) {
      if (call.method == 'create') {
        return {
          'playerId': 1,
          'textureId': 100,
          'v': 1,
          'state': 0,
          'position': 0.0,
          'duration': -1.0,
        };
      }
      if (call.method == 'destroy') {
        destroyAttempts++;
        if (destroyAttempts == 1) {
          throw PlatformException(code: 'DESTROY_FAILED', message: 'oops');
        }
        return null;
      }
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    // First close fails
    await expectLater(player.close(), throwsA(isA<PlatformException>()));

    // Retry succeeds
    await player.close();
    expect(destroyAttempts, 2);
    player.dispose();
  });

  // Test 12: Init error → lifecycle
  test('create returns error state → open() throws', () async {
    mock.handler = (call) {
      if (call.method == 'create') {
        return {
          'playerId': 1,
          'textureId': 100,
          'v': 1,
          'state': 5, // error
          'position': 0.0,
          'duration': -1.0,
          'error': {'code': 'LUMINA_INIT_FAILED', 'message': 'no decoder'},
        };
      }
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    await expectLater(
      player.open('https://example.com/video.mp4'),
      throwsA(isA<LuminaVideoException>()),
    );
    expect(() => player.play(), throwsStateError);
    player.dispose();
  });

  // Test 13: Native-generated playerId
  test('playerId comes from native create response', () async {
    mock.handler = (call) {
      if (call.method == 'create') {
        return {
          'playerId': 42,
          'textureId': 4200,
          'v': 1,
          'state': 0,
          'position': 0.0,
          'duration': -1.0,
        };
      }
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    // Verify destroy is called with the native-assigned ID
    await player.close();
    final destroyCall = mock.calls.firstWhere((c) => c.method == 'destroy');
    expect(destroyCall.arguments['playerId'], 42);
    player.dispose();
  });

  // Test 16: Destroy idempotency (UNKNOWN_PLAYER)
  test('destroy returning UNKNOWN_PLAYER is treated as success', () async {
    mock.handler = (call) {
      if (call.method == 'create') {
        return {
          'playerId': 1,
          'textureId': 100,
          'v': 1,
          'state': 0,
          'position': 0.0,
          'duration': -1.0,
        };
      }
      if (call.method == 'destroy') {
        throw PlatformException(code: 'UNKNOWN_PLAYER', message: 'gone');
      }
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');
    // Should NOT throw — UNKNOWN_PLAYER is idempotent
    await player.close();
    player.dispose();
  });

  // Test 17: Defensive snapshot parsing
  test('event with missing position defaults to 0.0', () async {
    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    mock.sendEvent(1, {
      'v': 1,
      'state': 2,
      // position missing
      'duration': 60.0,
    });
    await Future<void>.delayed(Duration.zero);

    expect(player.value.position, 0.0);
    expect(player.value.duration, 60.0);

    await player.close();
    player.dispose();
  });

  // Test 24: Failed close makes playback APIs unusable
  test('after failed close(), play() throws StateError', () async {
    var destroyAttempts = 0;
    mock.handler = (call) {
      if (call.method == 'create') {
        return {
          'playerId': 1,
          'textureId': 100,
          'v': 1,
          'state': 0,
          'position': 0.0,
          'duration': -1.0,
        };
      }
      if (call.method == 'destroy') {
        destroyAttempts++;
        if (destroyAttempts == 1) {
          throw PlatformException(code: 'DESTROY_FAILED', message: 'oops');
        }
        return null;
      }
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    // close() fails
    await expectLater(player.close(), throwsA(isA<PlatformException>()));

    // play() should throw — _closeRequested is permanent
    expect(() => player.play(), throwsStateError);

    // Retry close succeeds
    await player.close();
    player.dispose();
  });

  // Test 25: Failed close leaves no event subscription
  test('close() cancels event subscription before destroy attempt', () async {
    mock.handler = (call) {
      if (call.method == 'create') {
        return {
          'playerId': 1,
          'textureId': 100,
          'v': 1,
          'state': 0,
          'position': 0.0,
          'duration': -1.0,
        };
      }
      if (call.method == 'destroy') {
        throw PlatformException(code: 'DESTROY_FAILED', message: 'oops');
      }
      return null;
    };
    mock.setUp();

    final player = LuminaPlayer();
    await player.open('https://example.com/video.mp4');

    // close() fails but eventSub should be cancelled
    await expectLater(player.close(), throwsA(isA<PlatformException>()));

    final valueBefore = player.value;
    mock.sendEvent(1, {
      'v': 1,
      'state': 2,
      'position': 50.0,
      'duration': 100.0,
    });
    await Future<void>.delayed(Duration.zero);

    // Value should be unchanged — events discarded
    expect(player.value.position, valueBefore.position);
    player.dispose();
  });
}
