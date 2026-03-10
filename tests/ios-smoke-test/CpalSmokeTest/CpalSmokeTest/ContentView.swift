import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var status = "Tap to test cpal audio"
    @State private var isRunning = false
    @State private var deviceInfo = ""

    var body: some View {
        VStack(spacing: 24) {
            Text("cpal iOS Smoke Test")
                .font(.title2)
                .fontWeight(.bold)

            Text(status)
                .font(.title)
                .foregroundColor(statusColor)
                .multilineTextAlignment(.center)

            if !deviceInfo.isEmpty {
                Text(deviceInfo)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

            Button(action: runTest) {
                Text(isRunning ? "Playing..." : "Play 440 Hz Tone (2s)")
                    .font(.headline)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(isRunning ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            }
            .disabled(isRunning)
            .padding(.horizontal)

            Text("Expects a 2-second A4 tone from the speaker.\nCheck Xcode console for cpal device logs.")
                .font(.footnote)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private var statusColor: Color {
        if status.contains("PASS") { return .green }
        if status.contains("FAIL") { return .red }
        return .primary
    }

    private func runTest() {
        isRunning = true
        status = "Configuring AVAudioSession..."

        // Show audio route info
        let session = AVAudioSession.sharedInstance()
        let route = session.currentRoute
        let outputs = route.outputs.map { "\($0.portName) (\($0.portType.rawValue))" }.joined(separator: ", ")
        deviceInfo = "Output: \(outputs)\nHW Sample Rate: \(session.sampleRate) Hz"

        DispatchQueue.global(qos: .userInitiated).async {
            // Configure AVAudioSession — cpal does NOT do this
            do {
                try session.setCategory(.playback, options: .mixWithOthers)
                try session.setActive(true)
            } catch {
                DispatchQueue.main.async {
                    status = "FAIL: AVAudioSession error: \(error)"
                    isRunning = false
                }
                return
            }

            DispatchQueue.main.async {
                status = "Playing 440 Hz tone..."
            }

            let code = lumina_audio_smoke_test(2.0)

            DispatchQueue.main.async {
                if code == 0 {
                    status = "PASS (code 0) — did you hear the tone?"
                } else {
                    status = "FAIL (code \(code))"
                }
                isRunning = false
            }
        }
    }
}
