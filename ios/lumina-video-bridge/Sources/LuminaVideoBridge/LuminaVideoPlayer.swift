import Foundation
import QuartzCore
import CLuminaVideo

/// Swift wrapper around the lumina-video Rust FFI.
///
/// Uses CADisplayLink to poll the Rust player at vsync for state changes
/// and decoded video frames. Matches the DamusVideoPlayer API shape.
///
/// - Note: Must be used from the main thread (@MainActor).
/// - Note: Audio is not yet implemented (placeholder `is_muted`).
@MainActor
public final class LuminaVideoPlayer: ObservableObject {
    // MARK: - Published properties

    /// Current playback state.
    @Published public private(set) var state: LuminaVideoState = .loading

    /// Whether the player is currently playing.
    @Published public private(set) var is_playing: Bool = false

    /// Current playback position in seconds.
    @Published public private(set) var current_time: TimeInterval = 0

    /// Video duration in seconds. Nil if unknown (e.g., live stream).
    @Published public private(set) var duration: TimeInterval? = nil

    /// Whether the player is loading.
    @Published public private(set) var is_loading: Bool = true

    /// Video dimensions from the first decoded frame.
    @Published public private(set) var video_size: CGSize? = nil

    /// Mute state placeholder (audio not yet implemented).
    @Published public var is_muted: Bool = true

    // MARK: - Delegate

    /// Delegate for state changes and frame delivery.
    public weak var delegate: LuminaVideoPlayerDelegate?

    // MARK: - Private state

    /// Opaque Rust player handle.
    private var playerPtr: OpaquePointer?

    /// CADisplayLink for vsync-driven polling.
    private var displayLink: CADisplayLink?

    /// Last observed state, for change detection.
    private var lastState: LuminaVideoState = .loading

    // MARK: - Lifecycle

    /// Creates a new video player for the given URL.
    ///
    /// The player starts in `.loading` state. Decoder initialization happens
    /// asynchronously in Rust. Poll via CADisplayLink detects when ready.
    ///
    /// - Parameter url: Video URL string (http/https/file).
    /// - Throws: `LuminaVideoError` if the URL is invalid or creation fails.
    public init(url: String) throws {
        var ptr: OpaquePointer?
        let err = url.withCString { cStr in
            lumina_player_create(cStr, &ptr)
        }
        guard err == 0, let validPtr = ptr else {
            throw LuminaVideoError(rawValue: err) ?? .internal
        }
        self.playerPtr = validPtr
        startDisplayLink()
    }

    deinit {
        displayLink?.invalidate()
        displayLink = nil
        if playerPtr != nil {
            lumina_player_destroy(&playerPtr)
        }
    }

    // MARK: - Playback control

    /// Starts or resumes playback.
    public func play() {
        guard let ptr = playerPtr else { return }
        lumina_player_play(ptr)
        resumeDisplayLink()
    }

    /// Pauses playback.
    public func pause() {
        guard let ptr = playerPtr else { return }
        lumina_player_pause(ptr)
    }

    /// Seeks to a position in seconds.
    public func seek(to position: TimeInterval) {
        guard let ptr = playerPtr else { return }
        lumina_player_seek(ptr, position)
    }

    // MARK: - Display link

    private func startDisplayLink() {
        let link = CADisplayLink(target: self, selector: #selector(displayLinkFired))
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    private func stopDisplayLink() {
        displayLink?.invalidate()
        displayLink = nil
    }

    private func resumeDisplayLink() {
        displayLink?.isPaused = false
    }

    // MARK: - Poll loop

    @objc private func displayLinkFired(_ link: CADisplayLink) {
        guard let ptr = playerPtr else { return }

        // 1. Poll state
        let rawState = lumina_player_state(ptr)
        if let newState = LuminaVideoState(rawValue: rawState) {
            if newState != lastState {
                state = newState
                is_playing = newState == .playing
                is_loading = newState == .loading
                lastState = newState
                delegate?.luminaPlayer(self, didChangeState: newState)

                // Pause display link on terminal states
                if newState == .ended || newState == .error {
                    link.isPaused = true
                }
            }
        }

        // 2. Poll position / duration
        current_time = lumina_player_position(ptr)
        let dur = lumina_player_duration(ptr)
        duration = dur < 0 ? nil : dur

        // 3. Poll frame
        if let framePtr = lumina_player_poll_frame(ptr) {
            let frame = LuminaVideoFrame(framePtr: framePtr)

            // Capture video dimensions from first frame
            if video_size == nil && frame.width > 0 && frame.height > 0 {
                video_size = CGSize(width: frame.width, height: frame.height)
            }

            delegate?.luminaPlayer(self, didReceiveFrame: frame)
        }
    }
}
