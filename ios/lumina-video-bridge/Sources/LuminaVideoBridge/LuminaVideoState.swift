import Foundation

/// Playback state, mirroring the C enum values from LuminaVideo.h.
public enum LuminaVideoState: Int32, Sendable {
    case loading  = 0
    case ready    = 1
    case playing  = 2
    case paused   = 3
    case ended    = 4
    case error    = 5
}
