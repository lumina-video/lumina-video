import Foundation

/// Delegate protocol for receiving video player events.
public protocol LuminaVideoPlayerDelegate: AnyObject {
    /// Called when the playback state changes.
    func luminaPlayer(_ player: LuminaVideoPlayer, didChangeState state: LuminaVideoState)

    /// Called when a new decoded video frame is available.
    /// The frame (and its IOSurface) stays alive until the `LuminaVideoFrame` is deallocated.
    func luminaPlayer(_ player: LuminaVideoPlayer, didReceiveFrame frame: LuminaVideoFrame)
}
