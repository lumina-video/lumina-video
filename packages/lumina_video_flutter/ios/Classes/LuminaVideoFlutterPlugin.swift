import Flutter
import UIKit
import AVFoundation

/// Per-player state. Implements FlutterTexture for zero-copy IOSurface rendering.
private class PlayerEntry: NSObject, FlutterTexture, FlutterStreamHandler {
    var playerPtr: OpaquePointer?
    let playerId: Int
    var textureId: Int64 = -1
    var displayLink: CADisplayLink?
    var eventChannel: FlutterEventChannel?
    var eventSink: FlutterEventSink?
    var latestPixelBuffer: CVPixelBuffer?
    let lock = NSLock()
    var lastState: Int32 = -1
    var lastEventPush: CFTimeInterval = 0
    var videoWidth: Int = 0
    var videoHeight: Int = 0
    var wasPlayingBeforeBackground = false

    // Diagnostics
    var frameCount: Int = 0
    var fpsWindowStart: CFTimeInterval = 0
    var currentFps: Double = 0
    var isZeroCopy: Bool = false
    var sourceUrl: String = ""

    init(playerPtr: OpaquePointer, playerId: Int) {
        self.playerPtr = playerPtr
        self.playerId = playerId
        super.init()
    }

    // MARK: - FlutterTexture

    func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
        lock.lock()
        defer { lock.unlock() }
        guard let pb = latestPixelBuffer else { return nil }
        return Unmanaged.passRetained(pb)
    }

    // MARK: - FlutterStreamHandler

    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        eventSink = events
        // Emit current snapshot immediately so Dart doesn't miss state between
        // create response and EventChannel subscription.
        events(LuminaVideoFlutterPlugin.buildEvent(entry: self))
        return nil
    }

    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        eventSink = nil
        return nil
    }
}

public class LuminaVideoFlutterPlugin: NSObject, FlutterPlugin {
    private var textureRegistry: FlutterTextureRegistry
    private var messenger: FlutterBinaryMessenger
    private var players: [Int: PlayerEntry] = [:]

    /// Monotonic player ID counter. Static — survives plugin instance recreation.
    /// Accessed only on main thread (MethodChannel handler).
    private static var nextPlayerId: Int = 1

    init(textureRegistry: FlutterTextureRegistry, messenger: FlutterBinaryMessenger) {
        self.textureRegistry = textureRegistry
        self.messenger = messenger
        super.init()

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appWillResignActive),
            name: UIApplication.willResignActiveNotification,
            object: nil
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(appDidBecomeActive),
            name: UIApplication.didBecomeActiveNotification,
            object: nil
        )
    }

    // MARK: - FlutterPlugin registration

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "lumina_video_flutter",
            binaryMessenger: registrar.messenger()
        )
        let instance = LuminaVideoFlutterPlugin(
            textureRegistry: registrar.textures(),
            messenger: registrar.messenger()
        )
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    // MARK: - MethodChannel handler

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "create":
            handleCreate(call, result: result)
        case "play":
            handlePlayerCommand(call, result: result) { entry in
                guard let ptr = entry.playerPtr else { return }
                entry.displayLink?.isPaused = false
                let err = lumina_player_play(ptr)
                if err != LUMINA_OK {
                    let info = self.mapLuminaError(err)
                    result(FlutterError(code: info.code, message: info.message, details: nil))
                    return
                }
                result(nil)
            }
        case "pause":
            handlePlayerCommand(call, result: result) { entry in
                guard let ptr = entry.playerPtr else { return }
                let err = lumina_player_pause(ptr)
                if err != LUMINA_OK {
                    let info = self.mapLuminaError(err)
                    result(FlutterError(code: info.code, message: info.message, details: nil))
                    return
                }
                result(nil)
            }
        case "seek":
            handlePlayerCommand(call, result: result) { entry in
                guard let ptr = entry.playerPtr, let args = call.arguments as? [String: Any],
                      let position = args["position"] as? Double else { return }
                entry.displayLink?.isPaused = false
                let err = lumina_player_seek(ptr, position)
                if err != LUMINA_OK {
                    let info = self.mapLuminaError(err)
                    result(FlutterError(code: info.code, message: info.message, details: nil))
                    return
                }
                result(nil)
            }
        case "setMuted":
            handlePlayerCommand(call, result: result) { entry in
                guard let ptr = entry.playerPtr, let args = call.arguments as? [String: Any],
                      let muted = args["muted"] as? Bool else { return }
                lumina_player_set_muted(ptr, muted)
                result(nil)
            }
        case "setVolume":
            handlePlayerCommand(call, result: result) { entry in
                guard let ptr = entry.playerPtr, let args = call.arguments as? [String: Any],
                      let volume = args["volume"] as? Int else { return }
                lumina_player_set_volume(ptr, Int32(volume))
                result(nil)
            }
        case "destroy":
            handleDestroy(call, result: result)
        #if DEBUG
        case "_debugGetPlayersLive":
            result(players.count)
        #endif
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - Create

    private func handleCreate(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let url = args["url"] as? String else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing url", details: nil))
            return
        }

        // Configure audio session before creating native player
        configureAudioSession()

        var ptr: OpaquePointer?
        let err = url.withCString { cStr in
            lumina_player_create(cStr, &ptr)
        }
        guard err == LUMINA_OK, let validPtr = ptr else {
            let errorInfo = mapLuminaError(err)
            result(FlutterError(code: errorInfo.code, message: errorInfo.message, details: nil))
            return
        }

        let playerId = LuminaVideoFlutterPlugin.nextPlayerId
        LuminaVideoFlutterPlugin.nextPlayerId += 1

        let entry = PlayerEntry(playerPtr: validPtr, playerId: playerId)
        entry.sourceUrl = url

        // Register texture
        entry.textureId = textureRegistry.register(entry)

        // Create per-player EventChannel
        let channelName = "lumina_video_flutter/events/\(playerId)"
        let eventChannel = FlutterEventChannel(name: channelName, binaryMessenger: messenger)
        eventChannel.setStreamHandler(entry)
        entry.eventChannel = eventChannel

        players[playerId] = entry

        // Start display link for frame polling
        startDisplayLink(for: entry)

        // Return initial state
        var response = Self.buildEvent(entry: entry)
        response["playerId"] = playerId
        response["textureId"] = entry.textureId
        result(response)
    }

    // MARK: - Player commands

    private func handlePlayerCommand(
        _ call: FlutterMethodCall,
        result: @escaping FlutterResult,
        action: (PlayerEntry) -> Void
    ) {
        guard let args = call.arguments as? [String: Any],
              let playerId = args["playerId"] as? Int,
              let entry = players[playerId] else {
            result(FlutterError(code: "UNKNOWN_PLAYER", message: "Player not found", details: nil))
            return
        }
        action(entry)
    }

    // MARK: - Destroy

    private func handleDestroy(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let playerId = args["playerId"] as? Int else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing playerId", details: nil))
            return
        }
        guard players[playerId] != nil else {
            result(FlutterError(code: "UNKNOWN_PLAYER", message: "Player not found", details: nil))
            return
        }
        destroyPlayer(id: playerId)
        result(nil)
    }

    private func destroyPlayer(id: Int) {
        guard let entry = players.removeValue(forKey: id) else { return }

        // 1. Stop display link — no more ticks
        entry.displayLink?.invalidate()
        entry.displayLink = nil

        // 2. Tear down EventChannel
        entry.eventSink = nil
        entry.eventChannel?.setStreamHandler(nil)
        entry.eventChannel = nil

        // 3. Nil pixel buffer under lock — prevents raster thread from returning stale buffer
        entry.lock.lock()
        entry.latestPixelBuffer = nil
        entry.lock.unlock()

        // 4. Unregister texture
        textureRegistry.unregisterTexture(entry.textureId)

        // 5. Destroy native player
        if entry.playerPtr != nil {
            var ptr = entry.playerPtr
            lumina_player_destroy(&ptr)
            entry.playerPtr = nil
        }
    }

    // MARK: - Display link

    private func startDisplayLink(for entry: PlayerEntry) {
        let link = CADisplayLink(target: self, selector: #selector(displayLinkFired(_:)))
        // Store playerId in the link's name-value coding isn't available,
        // so we use the link's reference equality to find the entry.
        // Actually, we store the entry reference via a wrapper approach.
        // Simpler: iterate players to find the matching link.
        link.add(to: .main, forMode: .common)
        entry.displayLink = link
    }

    @objc private func displayLinkFired(_ link: CADisplayLink) {
        // Find the entry owning this display link
        guard let entry = players.values.first(where: { $0.displayLink === link }) else {
            link.invalidate()
            return
        }
        guard let ptr = entry.playerPtr else { return }

        // 1. Poll frame
        if let framePtr = lumina_player_poll_frame(ptr) {
            let w = Int(lumina_frame_width(framePtr))
            let h = Int(lumina_frame_height(framePtr))

            // FPS tracking
            entry.frameCount += 1
            let now = CACurrentMediaTime()
            let elapsed = now - entry.fpsWindowStart
            if elapsed >= 1.0 {
                entry.currentFps = Double(entry.frameCount) / elapsed
                entry.frameCount = 0
                entry.fpsWindowStart = now
            }

            // Get IOSurface for zero-copy texture.
            if let surfaceUnmanaged = lumina_frame_iosurface(framePtr) {
                entry.isZeroCopy = true
                let ioSurface = surfaceUnmanaged.takeUnretainedValue()
                var pixelBuffer: Unmanaged<CVPixelBuffer>?
                let status = CVPixelBufferCreateWithIOSurface(
                    nil,
                    ioSurface,
                    nil,
                    &pixelBuffer
                )
                lumina_frame_release(framePtr)

                if status == kCVReturnSuccess, let pb = pixelBuffer {
                    entry.lock.lock()
                    entry.latestPixelBuffer = pb.takeRetainedValue()
                    entry.lock.unlock()
                    textureRegistry.textureFrameAvailable(entry.textureId)
                }
            } else {
                entry.isZeroCopy = false
                lumina_frame_release(framePtr)
            }

            // Update video dimensions from frame
            if w > 0 && h > 0 && (w != entry.videoWidth || h != entry.videoHeight) {
                entry.videoWidth = w
                entry.videoHeight = h
                // Push event immediately on size change
                pushEvent(entry: entry)
            }
        }

        // 2. Throttled event push (250ms) or immediate on state change
        let currentState = lumina_player_state(ptr)
        let now = CACurrentMediaTime()
        if currentState != entry.lastState {
            entry.lastState = currentState
            pushEvent(entry: entry)
            entry.lastEventPush = now

            // Pause display link on terminal states
            if currentState == LUMINA_STATE_ENDED || currentState == LUMINA_STATE_ERROR {
                link.isPaused = true
            }
        } else if now - entry.lastEventPush >= 0.25 {
            pushEvent(entry: entry)
            entry.lastEventPush = now
        }
    }

    // MARK: - Event building (centralized)

    private func pushEvent(entry: PlayerEntry, error: [String: Any]? = nil) {
        entry.eventSink?(Self.buildEvent(entry: entry, error: error))
    }

    /// Maps C LuminaState to Dart LuminaPlaybackState index.
    private static func mapLuminaStateToDartState(_ cState: Int32) -> Int {
        switch cState {
        case LUMINA_STATE_LOADING: return 0
        case LUMINA_STATE_READY:   return 1
        case LUMINA_STATE_PLAYING: return 2
        case LUMINA_STATE_PAUSED:  return 3
        case LUMINA_STATE_ENDED:   return 4
        case LUMINA_STATE_ERROR:   return 5
        default:                   return 5 // unknown → error
        }
    }

    fileprivate static func buildEvent(
        entry: PlayerEntry,
        error: [String: Any]? = nil
    ) -> [String: Any] {
        let rawState = entry.playerPtr.map { lumina_player_state($0) } ?? LUMINA_STATE_ERROR
        let dartState = (error != nil) ? 5 : mapLuminaStateToDartState(rawState)
        var event: [String: Any] = [
            "v": 1,
            "state": dartState,
            "position": entry.playerPtr.map { lumina_player_position($0) } ?? 0.0,
            "duration": entry.playerPtr.map { lumina_player_duration($0) } ?? -1.0,
        ]
        if entry.videoWidth > 0 {
            event["videoWidth"] = entry.videoWidth
            event["videoHeight"] = entry.videoHeight
        }
        if let error = error { event["error"] = error }

        // Diagnostics
        if entry.currentFps > 0 {
            event["fps"] = entry.currentFps
        }
        // maxFps omitted on iOS — C FFI does not expose video source framerate
        event["zeroCopy"] = entry.isZeroCopy
        event["videoCodec"] = "VideoToolbox"
        event["audioCodec"] = "Native"
        if let ext = URL(string: entry.sourceUrl)?.pathExtension.lowercased(), !ext.isEmpty {
            event["format"] = ext.uppercased()
        }

        return event
    }

    // MARK: - Error mapping

    private func mapLuminaError(_ code: Int32) -> (code: String, message: String) {
        switch code {
        case LUMINA_ERROR_NULL_PTR:    return ("LUMINA_NULL_PTR", "Required pointer was NULL")
        case LUMINA_ERROR_INVALID_URL: return ("LUMINA_INVALID_URL", "URL string invalid or unsupported")
        case LUMINA_ERROR_INIT_FAILED: return ("LUMINA_INIT_FAILED", "Decoder initialization failed")
        case LUMINA_ERROR_DECODE:      return ("LUMINA_DECODE", "Decode error during playback")
        case LUMINA_ERROR_INTERNAL:    return ("LUMINA_INTERNAL", "Rust panic caught at FFI boundary")
        case LUMINA_ERROR_INVALID_ARG: return ("LUMINA_INVALID_ARG", "Invalid argument value")
        default:                       return ("LUMINA_UNKNOWN", "Unknown error code: \(code)")
        }
    }

    // MARK: - Audio session

    private func configureAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, options: .mixWithOthers)
            try session.setActive(true)
        } catch {
            // Non-fatal: audio may not work but video playback continues
        }
    }

    // MARK: - App lifecycle

    @objc private func appWillResignActive() {
        for (_, entry) in players {
            guard let ptr = entry.playerPtr else { continue }
            let state = lumina_player_state(ptr)
            entry.wasPlayingBeforeBackground = (state == LUMINA_STATE_PLAYING)
            if entry.wasPlayingBeforeBackground { lumina_player_pause(ptr) }
            entry.displayLink?.isPaused = true
        }
    }

    @objc private func appDidBecomeActive() {
        for (_, entry) in players {
            guard let ptr = entry.playerPtr else { continue }
            entry.displayLink?.isPaused = false
            if entry.wasPlayingBeforeBackground {
                lumina_player_play(ptr)
                entry.wasPlayingBeforeBackground = false
            }
        }
    }

    // MARK: - Engine detach (hot restart cleanup)

    public func detachFromEngine(for registrar: FlutterPluginRegistrar) {
        NotificationCenter.default.removeObserver(self)
        for id in Array(players.keys) { destroyPlayer(id: id) }
    }
}
