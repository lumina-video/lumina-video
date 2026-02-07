# lumina-video

Hardware-accelerated video player for [egui](https://github.com/emilk/egui) with zero-copy GPU rendering via [wgpu](https://github.com/gfx-rs/wgpu).

Decoded frames are delivered as `wgpu::Texture` — on supported platforms, frames go directly from the hardware decoder to GPU texture with no CPU copies.

## Usage

```toml
[dependencies]
lumina-video = { git = "https://github.com/lumina-video/lumina-video" }
```

```rust
use lumina_video::VideoPlayer;

// In your egui app:
if self.player.is_none() {
    self.player = Some(VideoPlayer::new("https://example.com/video.mp4"));
}
if let Some(player) = &mut self.player {
    player.show(ui, [640.0, 360.0].into());
}
```

## Platform Support

| Platform | Decoder | Zero-Copy Path |
|----------|---------|----------------|
| macOS | VideoToolbox | IOSurface -> wgpu::Texture |
| Linux | GStreamer + VA-API | DMA-BUF -> wgpu::Texture |
| Android | MediaCodec | AHardwareBuffer -> Vulkan -> wgpu::Texture |
| Windows | Media Foundation | D3D11 shared handle -> D3D12 -> wgpu::Texture |
| Web | HTMLVideoElement / WebCodecs | WebGPU texture |

## Integration

lumina-video uses the `wgpu::Device` and `wgpu::Queue` from your egui/eframe render state. If your app already uses eframe (the default egui backend), this works automatically.

For custom wgpu setups, the player accepts any `wgpu::Device`/`Queue` — it doesn't depend on egui for rendering, only for the UI widget layer.

## Optional Features

| Feature | Description |
|---------|-------------|
| `moq` | Media over QUIC live streaming (experimental) |
| `windows-native-video` | Windows Media Foundation decoder (opt-in) |
| `vendored-runtime` | Bundle GStreamer with binary (Linux) |

## License

Dual-licensed under MIT or Apache 2.0, at your option.

See the [repository root](https://github.com/lumina-video/lumina-video) for full documentation.
