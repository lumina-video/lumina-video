import IOSurface
import LuminaVideoBridge
import Metal
import MetalKit
import UIKit

/// UIView backed by CAMetalLayer that renders IOSurface frames via Metal.
///
/// Zero-copy path: IOSurface → MTLTexture (no CPU readback).
/// Falls back to a "No IOSurface" label on simulator if CPU-only decode returns nil.
final class MetalVideoUIView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }

    private var metalLayer: CAMetalLayer? { layer as? CAMetalLayer }
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var pipelineState: MTLRenderPipelineState?
    private var sampler: MTLSamplerState?

    // Cached descriptor — reused when frame dimensions haven't changed
    private var cachedTexDesc: MTLTextureDescriptor?
    private var cachedDescWidth: Int = 0
    private var cachedDescHeight: Int = 0

    /// Set by the SwiftUI coordinator to trigger redraws.
    var currentFrame: LuminaVideoFrame? {
        didSet { draw() }
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let metalLayer else { return }
        self.device = device
        self.commandQueue = device.makeCommandQueue()

        metalLayer.device = device
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = true
        metalLayer.contentsScale = UIScreen.main.scale

        buildPipeline(device: device)
        buildSampler(device: device)
    }

    private func buildPipeline(device: MTLDevice) {
        guard let library = device.makeDefaultLibrary() else { return }
        guard let vertexFn = library.makeFunction(name: "vertexShader") else { return }
        guard let fragmentFn = library.makeFunction(name: "fragmentShader") else { return }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm

        pipelineState = try? device.makeRenderPipelineState(descriptor: desc)
    }

    private func buildSampler(device: MTLDevice) {
        let desc = MTLSamplerDescriptor()
        desc.minFilter = .linear
        desc.magFilter = .linear
        desc.sAddressMode = .clampToEdge
        desc.tAddressMode = .clampToEdge
        sampler = device.makeSamplerState(descriptor: desc)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        guard let metalLayer else { return }
        metalLayer.drawableSize = CGSize(
            width: bounds.width * metalLayer.contentsScale,
            height: bounds.height * metalLayer.contentsScale
        )
    }

    private func draw() {
        guard let device, let commandQueue, let pipelineState, let sampler,
              let metalLayer else { return }
        guard let frame = currentFrame, let ioSurface = frame.ioSurface else { return }

        // Reuse texture descriptor when frame dimensions haven't changed
        if frame.width != cachedDescWidth || frame.height != cachedDescHeight {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: frame.width,
                height: frame.height,
                mipmapped: false
            )
            desc.usage = [.shaderRead]
            desc.storageMode = .shared
            cachedTexDesc = desc
            cachedDescWidth = frame.width
            cachedDescHeight = frame.height
        }
        guard let texDesc = cachedTexDesc else { return }

        // New texture each frame — each IOSurface is a different backing store
        guard let texture = device.makeTexture(
            descriptor: texDesc,
            iosurface: ioSurface as IOSurfaceRef,
            plane: 0
        ) else { return }

        guard let drawable = metalLayer.nextDrawable() else { return }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let passDesc = MTLRenderPassDescriptor()
        passDesc.colorAttachments[0].texture = drawable.texture
        passDesc.colorAttachments[0].loadAction = .clear
        passDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        passDesc.colorAttachments[0].storeAction = .store

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc) else { return }
        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.setFragmentSamplerState(sampler, index: 0)
        // Fullscreen triangle: 3 vertices, no vertex buffer needed
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
