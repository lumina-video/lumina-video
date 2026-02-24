# Flutter Plugin — Agent Rules

Supplements [AGENTS.md](AGENTS.md). All rules there apply here. This file covers **only** what's unique to the Flutter/FFI boundary.

## Pre-Commit Checklist

```bash
dart format .
flutter analyze --no-pub
flutter test
cargo check -p lumina-video-ffi
```

If the C header changed, regenerate bindings: `dart run ffigen`. Never hand-edit generated bindings.

## FFI Boundary Rules

These are the rules agents violate most. Read carefully.

1. **Every FFI pointer return must be null-checked.** Never use `!` on a value from native code.

```dart
// WRONG
final player = _bindings.lumina_player_create()!;

// RIGHT
final ptr = _bindings.lumina_player_create();
if (ptr == nullptr) throw LuminaVideoException('create failed');
```

2. **Every `malloc.allocate` / `toNativeUtf8()` must have a matching `free()` in a `finally` block.**

```dart
final namePtr = url.toNativeUtf8();
try {
  _bindings.lumina_set_url(handle, namePtr.cast());
} finally {
  malloc.free(namePtr);
}
```

3. **Use `NativeFinalizer` on wrapper objects.** Prevent leaks when Dart GC collects a controller that wasn't manually disposed.

4. **No panics across FFI.** Every `extern "C"` function in Rust must wrap its body in `std::panic::catch_unwind`. A panic across FFI is instant UB.

5. **No `Pointer` in public API.** Callers interact with Dart classes. Raw pointers stay inside `src/`.

## Rendering Rules

6. **Frames go GPU → Flutter Texture → compositor. No pixel data through Dart.** If you find yourself passing `Uint8List` of pixels from native to Dart for display, you're doing it wrong. Use `Texture(textureId: id)`.

7. **No Dart allocations in the per-frame path.** No `Map`, `List`, `Future`, or string formatting per frame. The frame callback must be trivial.

8. **Frame pacing is native-side.** Dart notifies Flutter with `textureFrameAvailable()`. Don't poll with `Timer.periodic` — use `SchedulerBinding.addPostFrameCallback` if Dart-side timing is needed.

## Concurrency

9. **Short FFI calls (get position, set volume) are fine on the main isolate.** Blocking FFI calls (open URL, seek to network position) must run on a background `Isolate` or native thread.

10. **Multiple players must be independent.** No shared mutable globals on either side of the FFI boundary.
