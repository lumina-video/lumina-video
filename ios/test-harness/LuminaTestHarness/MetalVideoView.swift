import SwiftUI
import LuminaVideoBridge

/// SwiftUI wrapper around MetalVideoUIView.
struct MetalVideoView: UIViewRepresentable {
    @ObservedObject var viewModel: VideoViewModel

    func makeUIView(context: Context) -> MetalVideoUIView {
        MetalVideoUIView(frame: .zero)
    }

    func updateUIView(_ uiView: MetalVideoUIView, context: Context) {
        uiView.currentFrame = viewModel.currentFrame
    }
}
