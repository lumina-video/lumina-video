import Foundation

/// Errors from the lumina-video FFI layer.
public enum LuminaVideoError: Int32, Error, Sendable {
    case nullPointer    = 1
    case invalidUrl     = 2
    case initFailed     = 3
    case decode         = 4
    case `internal`     = 5
    case invalidArg     = 6
}
