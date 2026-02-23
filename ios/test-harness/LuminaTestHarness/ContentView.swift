import SwiftUI
import AVFoundation
import LuminaVideoBridge

struct ContentView: View {
    @StateObject private var viewModel = VideoViewModel()
    @State private var isFullscreen = false

    var body: some View {
        ZStack {
            if isFullscreen {
                fullscreenView
            } else {
                normalView
            }
        }
        .onAppear {
            viewModel.load(url: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8")
        }
        .statusBarHidden(isFullscreen)
    }

    // MARK: - Normal (inline) layout

    private var normalView: some View {
        ScrollView {
            VStack(spacing: 0) {
                videoArea
                    .aspectRatio(16.0 / 9.0, contentMode: .fit)

                controlsView
                    .padding()

                DiagnosticsPanel(viewModel: viewModel)
                    .padding(.horizontal)
                    .padding(.bottom)
            }
        }
    }

    // MARK: - Fullscreen layout

    private var fullscreenView: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            videoArea
                .ignoresSafeArea()

            // Floating exit button
            Button(action: { withAnimation { isFullscreen = false } }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title)
                    .foregroundStyle(.white.opacity(0.8))
            }
            .padding()
        }
    }

    // MARK: - Shared subviews

    private var videoArea: some View {
        ZStack {
            Color.black
            if viewModel.hasFrame {
                MetalVideoView(viewModel: viewModel)
            } else if viewModel.state == .error {
                Text("Error")
                    .foregroundColor(.red)
                    .font(.headline)
            } else {
                ProgressView()
                    .tint(.white)
            }
        }
    }

    private var controlsView: some View {
        VStack(spacing: 12) {
            // State label
            HStack {
                Circle()
                    .fill(stateColor)
                    .frame(width: 8, height: 8)
                Text(stateText)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                if let size = viewModel.videoSize {
                    Text("\(Int(size.width))x\(Int(size.height))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Seek slider
            if let duration = viewModel.duration, duration > 0 {
                HStack {
                    Text(formatTime(viewModel.currentTime))
                        .font(.caption.monospacedDigit())
                    Slider(
                        value: $viewModel.seekPosition,
                        in: 0...duration,
                        onEditingChanged: { editing in
                            if !editing {
                                viewModel.seek(to: viewModel.seekPosition)
                            }
                        }
                    )
                    Text(formatTime(duration))
                        .font(.caption.monospacedDigit())
                }
            }

            // Play/pause + volume + fullscreen
            HStack(spacing: 20) {
                Button(action: { viewModel.togglePlayPause() }) {
                    Image(systemName: viewModel.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                }

                Button(action: { viewModel.toggleMute() }) {
                    Image(systemName: viewModel.isMuted ? "speaker.slash.fill" : "speaker.wave.2.fill")
                        .font(.title3)
                }

                Slider(value: $viewModel.volumeLevel, in: 0...100, step: 1) {
                    Text("Volume")
                }
                .frame(maxWidth: 150)

                Spacer()

                Button(action: { withAnimation { isFullscreen = true } }) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                        .font(.title3)
                }
            }
        }
    }

    private var stateColor: Color {
        switch viewModel.state {
        case .loading: return .orange
        case .ready:   return .blue
        case .playing: return .green
        case .paused:  return .yellow
        case .ended:   return .gray
        case .error:   return .red
        }
    }

    private var stateText: String {
        switch viewModel.state {
        case .loading: return "Loading"
        case .ready:   return "Ready"
        case .playing: return "Playing"
        case .paused:  return "Paused"
        case .ended:   return "Ended"
        case .error:   return "Error"
        }
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}

// MARK: - Diagnostics Panel

struct DiagnosticsPanel: View {
    @ObservedObject var viewModel: VideoViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Diagnostics")
                .font(.caption.bold())
                .foregroundColor(.primary)

            DiagRow(label: "Render path",
                    value: viewModel.renderPath)
            DiagRow(label: "Video codec",
                    value: viewModel.videoCodec ?? "—")
            DiagRow(label: "Audio codec",
                    value: viewModel.audioCodec ?? "—")
            DiagRow(label: "Nominal FPS",
                    value: viewModel.nominalFPS.map { String(format: "%.1f", $0) } ?? "—")
            DiagRow(label: "Measured FPS",
                    value: String(format: "%.1f", viewModel.measuredFPS))
            DiagRow(label: "A/V sync",
                    value: viewModel.isLive ? viewModel.clockDriftLabel : "AVPlayer (native)")
        }
        .padding(10)
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
    }
}

struct DiagRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption2.monospacedDigit())
                .foregroundColor(.primary)
        }
    }
}

// MARK: - ViewModel

@MainActor
final class VideoViewModel: ObservableObject {
    @Published var state: LuminaVideoState = .loading
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval?
    @Published var videoSize: CGSize?
    @Published var hasFrame = false
    @Published var seekPosition: TimeInterval = 0
    @Published var diagnostics: LuminaVideoDiagnostics?

    // Diagnostics
    @Published var renderPath: String = "—"
    @Published var videoCodec: String?
    @Published var audioCodec: String?
    @Published var nominalFPS: Float?
    @Published var measuredFPS: Double = 0
    @Published var clockDriftLabel: String = "—"

    /// True for live streams (MoQ), false for VOD (HLS/MP4). Clock drift only shown for live.
    var isLive: Bool { duration == nil }

    @Published var isMuted = true {
        didSet { player?.is_muted = isMuted }
    }

    @Published var volumeLevel: Double = 100 {
        didSet { player?.volume = Int32(volumeLevel) }
    }

    private var player: LuminaVideoPlayer?

    /// Current frame's IOSurface for Metal rendering.
    var currentFrame: LuminaVideoFrame?

    // FPS measurement
    private var frameTimestamps: [CFTimeInterval] = []
    private var fpsWindowSeconds: CFTimeInterval = 1.0

    // Nominal FPS inference from unique position changes
    private var lastFramePosition: TimeInterval = -1
    private var uniquePositionTimestamps: [CFTimeInterval] = []

    // A/V sync: track wall-clock vs reported position
    private var playbackStartWall: CFTimeInterval?
    private var playbackStartPosition: TimeInterval?

    func load(url: String) {
        do {
            let p = try LuminaVideoPlayer(url: url)
            p.delegate = self
            player = p
            p.play()
            probeMediaInfo(url: url)
        } catch {
            state = .error
        }
    }

    func togglePlayPause() {
        guard let player else { return }
        if isPlaying {
            player.pause()
            playbackStartWall = nil
        } else {
            player.play()
        }
    }

    func toggleMute() {
        isMuted.toggle()
    }

    func seek(to position: TimeInterval) {
        player?.seek(to: position)
        // Reset A/V sync anchor after seek
        playbackStartWall = nil
        playbackStartPosition = nil
    }

    // MARK: - FPS measurement

    private func recordFrameArrival() {
        let now = CACurrentMediaTime()
        frameTimestamps.append(now)
        // Trim to window
        let cutoff = now - fpsWindowSeconds
        frameTimestamps.removeAll { $0 < cutoff }
        if frameTimestamps.count >= 2 {
            let span = frameTimestamps.last! - frameTimestamps.first!
            if span > 0 {
                measuredFPS = Double(frameTimestamps.count - 1) / span
            }
        }

        // Infer nominal FPS from unique position changes (content frame rate)
        if nominalFPS == nil {
            let pos = currentTime
            if pos != lastFramePosition {
                lastFramePosition = pos
                uniquePositionTimestamps.append(now)
                let uniqCutoff = now - 2.0 // 2s window for stability
                uniquePositionTimestamps.removeAll { $0 < uniqCutoff }
                if uniquePositionTimestamps.count >= 10 {
                    let span = uniquePositionTimestamps.last! - uniquePositionTimestamps.first!
                    if span > 0 {
                        let raw = Double(uniquePositionTimestamps.count - 1) / span
                        nominalFPS = snapToStandardFPS(raw)
                    }
                }
            }
        }
    }

    private func snapToStandardFPS(_ fps: Double) -> Float {
        let standards: [Double] = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]
        let closest = standards.min(by: { abs($0 - fps) < abs($1 - fps) }) ?? fps
        return Float(closest)
    }

    // MARK: - A/V sync

    private func updateAVSync() {
        let now = CACurrentMediaTime()
        if playbackStartWall == nil {
            playbackStartWall = now
            playbackStartPosition = currentTime
            clockDriftLabel = "calibrating…"
            return
        }
        guard let startWall = playbackStartWall,
              let startPos = playbackStartPosition else { return }
        let wallElapsed = now - startWall
        let posElapsed = currentTime - startPos
        let drift = (posElapsed - wallElapsed) * 1000.0 // ms
        clockDriftLabel = String(format: "%+.0f ms", drift)
    }

    // MARK: - Media info probing

    private func probeMediaInfo(url: String) {
        guard let parsedURL = URL(string: url) else { return }

        // For HLS: fetch the master playlist and parse EXT-X-STREAM-INF CODECS
        if url.hasSuffix(".m3u8") || url.contains("m3u8") {
            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    let (data, _) = try await URLSession.shared.data(from: parsedURL)
                    guard let manifest = String(data: data, encoding: .utf8) else { return }
                    self.parseHLSManifest(manifest)
                } catch {}
            }
            return
        }

        // For non-HLS: probe with AVURLAsset
        let asset = AVURLAsset(url: parsedURL)
        Task { @MainActor [weak self] in
            guard let self else { return }
            guard let tracks = try? await asset.load(.tracks) else { return }
            for track in tracks {
                if track.mediaType == .video, self.videoCodec == nil {
                    if let descs = try? await track.load(.formatDescriptions),
                       let desc = descs.first {
                        self.videoCodec = fourCCToString(CMFormatDescriptionGetMediaSubType(desc))
                    }
                    if let rate = try? await track.load(.nominalFrameRate), rate > 0 {
                        self.nominalFPS = rate
                    }
                } else if track.mediaType == .audio, self.audioCodec == nil {
                    if let descs = try? await track.load(.formatDescriptions),
                       let desc = descs.first {
                        self.audioCodec = fourCCToString(CMFormatDescriptionGetMediaSubType(desc))
                    }
                }
            }
        }
    }

    private func parseHLSManifest(_ manifest: String) {
        // Parse CODECS="..." from EXT-X-STREAM-INF
        // e.g. CODECS="avc1.64001e,mp4a.40.2"
        for line in manifest.components(separatedBy: .newlines) {
            guard line.hasPrefix("#EXT-X-STREAM-INF") else { continue }
            guard let codecsRange = line.range(of: "CODECS=\"") else { continue }
            let afterCodecs = line[codecsRange.upperBound...]
            guard let closeQuote = afterCodecs.firstIndex(of: "\"") else { continue }
            let codecsStr = String(afterCodecs[..<closeQuote])
            let codecs = codecsStr.components(separatedBy: ",")

            for codec in codecs {
                let c = codec.trimmingCharacters(in: .whitespaces).lowercased()
                if c.hasPrefix("avc1") {
                    videoCodec = "H.264"
                } else if c.hasPrefix("hvc1") || c.hasPrefix("hev1") {
                    videoCodec = "HEVC"
                } else if c.hasPrefix("vp09") {
                    videoCodec = "VP9"
                } else if c.hasPrefix("av01") {
                    videoCodec = "AV1"
                } else if c.hasPrefix("mp4a") {
                    audioCodec = "AAC"
                } else if c.hasPrefix("opus") || c.hasPrefix("Opus") {
                    audioCodec = "Opus"
                } else if c.hasPrefix("ac-3") {
                    audioCodec = "AC-3"
                } else if c.hasPrefix("ec-3") {
                    audioCodec = "E-AC-3"
                } else if c.hasPrefix("flac") {
                    audioCodec = "FLAC"
                }
            }

            // Parse FRAME-RATE if present
            if let fpsRange = line.range(of: "FRAME-RATE=") {
                let afterFPS = line[fpsRange.upperBound...]
                let fpsStr = afterFPS.prefix(while: { $0.isNumber || $0 == "." })
                if let fps = Float(fpsStr), fps > 0 {
                    nominalFPS = fps
                }
            }

            if videoCodec != nil { break } // Use first variant
        }
    }
}

private func fourCCToString(_ code: FourCharCode) -> String {
    // Known codec mappings
    switch code {
    case kCMVideoCodecType_H264:           return "H.264"
    case kCMVideoCodecType_HEVC:           return "HEVC"
    case kCMVideoCodecType_VP9:            return "VP9"
    case kCMVideoCodecType_AV1:            return "AV1"
    case kAudioFormatMPEG4AAC:             return "AAC"
    case kAudioFormatOpus:                 return "Opus"
    case kAudioFormatMPEGLayer3:           return "MP3"
    case kAudioFormatFLAC:                 return "FLAC"
    case kAudioFormatAppleLossless:        return "ALAC"
    case kAudioFormatLinearPCM:            return "PCM"
    default:
        // Decode FourCC bytes
        let chars: [Character] = [
            Character(UnicodeScalar((code >> 24) & 0xFF)!),
            Character(UnicodeScalar((code >> 16) & 0xFF)!),
            Character(UnicodeScalar((code >> 8) & 0xFF)!),
            Character(UnicodeScalar(code & 0xFF)!)
        ]
        return String(chars).trimmingCharacters(in: .whitespaces)
    }
}

extension VideoViewModel: LuminaVideoPlayerDelegate {
    nonisolated func luminaPlayer(_ player: LuminaVideoPlayer, didChangeState state: LuminaVideoState) {
        Task { @MainActor in
            self.state = state
            self.isPlaying = state == .playing
            self.currentTime = player.current_time
            self.duration = player.duration
            self.videoSize = player.video_size
            if state == .playing {
                // Reset A/V sync anchor on play
                self.playbackStartWall = nil
                self.playbackStartPosition = nil
            }
        }
    }

    nonisolated func luminaPlayer(_ player: LuminaVideoPlayer, didReceiveFrame frame: LuminaVideoFrame) {
        Task { @MainActor in
            let isFirst = !self.hasFrame
            self.currentFrame = frame
            self.hasFrame = true
            self.currentTime = player.current_time
            self.seekPosition = player.current_time

            // Keep polling duration until it resolves (HLS reports lazily)
            if self.duration == nil {
                self.duration = player.duration
            }

            // Render path: zero-copy if IOSurface present
            self.renderPath = frame.ioSurface != nil ? "Zero-copy (IOSurface)" : "CPU fallback"

            // FPS + A/V sync
            self.recordFrameArrival()
            if self.isPlaying {
                self.updateAVSync()
            }

            if isFirst {
                self.videoSize = player.video_size
            }
        }
    }
}
