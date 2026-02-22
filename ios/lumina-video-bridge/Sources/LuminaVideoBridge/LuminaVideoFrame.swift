import Foundation
import IOSurface
import CLuminaVideo

/// Owns a decoded video frame from the Rust decoder.
///
/// Holds the Rust `LuminaFrame*` alive until deinit, so the IOSurface
/// (which is backed by the frame) remains valid for this object's lifetime.
///
/// Thread-safety: The frame pointer is immutable after init and released
/// exactly once in deinit. The IOSurface is ARC-retained independently.
public final class LuminaVideoFrame: @unchecked Sendable {
    private let framePtr: OpaquePointer

    /// Frame width in pixels.
    public let width: Int

    /// Frame height in pixels.
    public let height: Int

    /// IOSurface for zero-copy Metal rendering.
    /// ARC-retained — safe to hold beyond this object's lifetime.
    public let ioSurface: IOSurface?

    init(framePtr: OpaquePointer) {
        self.framePtr = framePtr
        self.width = Int(lumina_frame_width(framePtr))
        self.height = Int(lumina_frame_height(framePtr))

        // lumina_frame_iosurface returns IOSurfaceRef with no CF ownership
        // annotation → Swift imports as Unmanaged<IOSurface>?.
        //
        // takeUnretainedValue(): extracts without changing refcount.
        // Assigning to self.ioSurface (strong let) → ARC retains (+1).
        //
        // Refcount lifecycle:
        //   1. Rust frame owns IOSurface (refcount >= 1)
        //   2. takeUnretainedValue() + ARC strong store → +1 (refcount >= 2)
        //   3. deinit: lumina_frame_release() → Rust drops its retain (-1)
        //   4. deinit: ARC releases self.ioSurface → -1 → freed when last ref drops
        if let surfaceUnmanaged = lumina_frame_iosurface(framePtr) {
            self.ioSurface = surfaceUnmanaged.takeUnretainedValue()
        } else {
            self.ioSurface = nil
        }
    }

    deinit {
        lumina_frame_release(framePtr)
    }
}
