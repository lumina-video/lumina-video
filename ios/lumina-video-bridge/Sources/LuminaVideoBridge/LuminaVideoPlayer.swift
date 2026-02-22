import Foundation
import AVFoundation
import QuartzCore
import CLuminaVideo

/// AVAudioSession configuration for video playback.
public struct AudioSessionConfig {
    public let category: AVAudioSession.Category
    public let options: AVAudioSession.CategoryOptions

    /// Default: `.playback` category with `.mixWithOthers` option.
    public static let `default` = AudioSessionConfig(
        category: .playback,
        options: .mixWithOthers
    )
}

/// Weak-target proxy that breaks the CADisplayLink → LuminaVideoPlayer retain cycle.
///
/// CADisplayLink strongly retains its target. Without this proxy, the cycle
/// `LuminaVideoPlayer → displayLink → LuminaVideoPlayer` prevents deallocation.
private class DisplayLinkProxy {
    weak var target: LuminaVideoPlayer?

    init(target: LuminaVideoPlayer) {
        self.target = target
    }

    @objc func displayLinkFired(_ link: CADisplayLink) {
        guard let target = target else {
            link.invalidate()
            return
        }
        target.displayLinkFired(link)
    }
}

/// Swift wrapper around the lumina-video Rust FFI.
///
/// Uses CADisplayLink to poll the Rust player at vsync for state changes
/// and decoded video frames. Matches the DamusVideoPlayer API shape.
///
/// - Note: Must be used from the main thread (@MainActor).
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

    /// Mute state. Defaults to `true` (muted) — appropriate for autoplay/feed scenarios.
    /// Setting this property immediately syncs to the Rust player.
    @Published public var is_muted: Bool = true {
        didSet {
            guard oldValue != is_muted, let ptr = playerPtr else { return }
            lumina_player_set_muted(ptr, is_muted)
        }
    }

    /// Volume level (0-100). Defaults to 100 (full volume).
    /// Setting this property immediately syncs to the Rust player.
    @Published public var volume: Int32 = 100 {
        didSet {
            guard oldValue != volume, let ptr = playerPtr else { return }
            lumina_player_set_volume(ptr, volume)
        }
    }

    /// Whether the video URL has audio tracks. Detected asynchronously after init.
    @Published public private(set) var has_audio: Bool = false

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
    /// Audio session is configured with the provided config (default: `.playback` + `.mixWithOthers`).
    ///
    /// - Parameters:
    ///   - url: Video URL string (http/https/file path).
    ///   - audioSessionConfig: Audio session configuration. Defaults to `.default`.
    /// - Throws: `LuminaVideoError` if the URL is invalid or creation fails.
    public init(url: String, audioSessionConfig: AudioSessionConfig = .default) throws {
        var ptr: OpaquePointer?
        let err = url.withCString { cStr in
            lumina_player_create(cStr, &ptr)
        }
        guard err == 0, let validPtr = ptr else {
            throw LuminaVideoError(rawValue: err) ?? .internal
        }
        self.playerPtr = validPtr

        // Configure audio session
        configureAudioSession(audioSessionConfig)

        // Sync initial mute state to Rust player
        lumina_player_set_muted(validPtr, is_muted)

        startDisplayLink()
        detectAudioTracks(url: url)
    }

    deinit {
        // CADisplayLink must be invalidated on the run loop it was added to.
        // @MainActor does NOT isolate deinit, so dispatch to main as a safety net.
        // (The DisplayLinkProxy already handles cleanup via its weak reference,
        // but this ensures immediate invalidation rather than waiting for next tick.)
        if let link = displayLink {
            RunLoop.main.perform { link.invalidate() }
        }
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

    // MARK: - Audio session

    private func configureAudioSession(_ config: AudioSessionConfig) {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(config.category, options: config.options)
            try session.setActive(true)
        } catch {
            // Non-fatal: audio may not work but video playback continues
        }
    }

    // MARK: - Audio track detection

    /// Detects whether the video URL has audio tracks.
    private func detectAudioTracks(url: String) {
        // Normalize URL: plain paths (no scheme) → file URL, otherwise parse as URL string
        let videoURL: URL
        if url.contains("://") {
            guard let parsed = URL(string: url) else { return }
            videoURL = parsed
        } else {
            videoURL = URL(fileURLWithPath: url)
        }
        let asset = AVURLAsset(url: videoURL)
        Task { @MainActor [weak self] in
            let tracks = try? await asset.load(.tracks)
            let hasAudioTrack = tracks?.contains(where: { $0.mediaType == .audio }) ?? false
            self?.has_audio = hasAudioTrack
        }
    }

    // MARK: - Display link

    private func startDisplayLink() {
        let proxy = DisplayLinkProxy(target: self)
        let link = CADisplayLink(target: proxy, selector: #selector(DisplayLinkProxy.displayLinkFired))
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

    @objc fileprivate func displayLinkFired(_ link: CADisplayLink) {
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

        // 2. Poll position / duration (only update @Published when changed to avoid unnecessary SwiftUI re-renders)
        let newTime = lumina_player_position(ptr)
        if newTime != current_time { current_time = newTime }
        let dur = lumina_player_duration(ptr)
        let newDuration: TimeInterval? = dur < 0 ? nil : dur
        if newDuration != duration { duration = newDuration }

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
