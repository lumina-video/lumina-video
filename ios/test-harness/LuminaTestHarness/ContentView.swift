import SwiftUI
import LuminaVideoBridge

struct ContentView: View {
    @StateObject private var viewModel = VideoViewModel()

    var body: some View {
        VStack(spacing: 0) {
            // Video area
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
            .aspectRatio(16.0 / 9.0, contentMode: .fit)

            // Controls
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

                // Play/pause + volume
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
                }

                // Diagnostics
                if let diag = viewModel.diagnostics {
                    HStack {
                        Text("P:\(diag.playersLive) F:\(diag.framesLive) FFI:\(diag.ffiCalls)")
                            .font(.caption2.monospacedDigit())
                            .foregroundColor(.secondary)
                        Spacer()
                    }
                }
            }
            .padding()

            Spacer()
        }
        .onAppear {
            viewModel.load(url: "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8")
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

    @Published var isMuted = true {
        didSet { player?.is_muted = isMuted }
    }

    @Published var volumeLevel: Double = 100 {
        didSet { player?.volume = Int32(volumeLevel) }
    }

    private var player: LuminaVideoPlayer?

    /// Current frame's IOSurface for Metal rendering.
    var currentFrame: LuminaVideoFrame?

    func load(url: String) {
        do {
            let p = try LuminaVideoPlayer(url: url)
            p.delegate = self
            player = p
            p.play()
        } catch {
            state = .error
        }
    }

    func togglePlayPause() {
        guard let player else { return }
        if isPlaying {
            player.pause()
        } else {
            player.play()
        }
    }

    func toggleMute() {
        isMuted.toggle()
    }

    func seek(to position: TimeInterval) {
        player?.seek(to: position)
    }

    func refreshDiagnostics() {
        diagnostics = LuminaVideoDiagnostics.snapshot()
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
            self.refreshDiagnostics()
        }
    }

    nonisolated func luminaPlayer(_ player: LuminaVideoPlayer, didReceiveFrame frame: LuminaVideoFrame) {
        Task { @MainActor in
            let isFirst = !self.hasFrame
            self.currentFrame = frame
            self.hasFrame = true
            self.currentTime = player.current_time
            self.seekPosition = player.current_time
            if isFirst {
                self.videoSize = player.video_size
            }
        }
    }
}
