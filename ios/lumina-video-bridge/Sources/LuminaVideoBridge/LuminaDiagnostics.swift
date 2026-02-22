import Foundation
import CLuminaVideo

/// FFI diagnostics snapshot from the Rust layer.
///
/// In debug builds, tracks player/frame lifecycle counts and FFI call totals.
/// In release builds, all fields are zero.
public struct LuminaVideoDiagnostics: CustomStringConvertible {
    public let playersCreated: UInt64
    public let playersDestroyed: UInt64
    public let playersPeak: UInt64
    public let playersLive: UInt64
    public let framesCreated: UInt64
    public let framesDestroyed: UInt64
    public let framesPeak: UInt64
    public let framesLive: UInt64
    public let ffiCalls: UInt64

    /// Takes a snapshot of current FFI diagnostics.
    public static func snapshot() -> LuminaVideoDiagnostics? {
        var raw = LuminaDiagnostics()
        let err = lumina_diagnostics_snapshot(&raw)
        guard err == 0 else { return nil }
        return LuminaVideoDiagnostics(
            playersCreated: raw.players_created,
            playersDestroyed: raw.players_destroyed,
            playersPeak: raw.players_peak,
            playersLive: raw.players_live,
            framesCreated: raw.frames_created,
            framesDestroyed: raw.frames_destroyed,
            framesPeak: raw.frames_peak,
            framesLive: raw.frames_live,
            ffiCalls: raw.ffi_calls
        )
    }

    public var description: String {
        """
        LuminaVideoDiagnostics:
          Players: \(playersLive) live, \(playersPeak) peak, \(playersCreated) created, \(playersDestroyed) destroyed
          Frames:  \(framesLive) live, \(framesPeak) peak, \(framesCreated) created, \(framesDestroyed) destroyed
          FFI calls: \(ffiCalls)
        """
    }
}
