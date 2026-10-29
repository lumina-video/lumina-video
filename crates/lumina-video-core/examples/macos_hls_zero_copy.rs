#[cfg(any(target_os = "macos", target_os = "ios"))]
fn main() {
    use std::time::{Duration, Instant};

    use lumina_video_core::macos_video::MacOSVideoDecoder;
    use lumina_video_core::video::{DecodedFrame, VideoDecoderBackend};
    use lumina_video_core::zero_copy::macos::import_iosurface;

    let url = std::env::args().nth(1).expect("usage: macos_hls_zero_copy <url> [seconds]");
    let seconds: u64 = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(15);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
            .expect("device");

    let mut decoder = MacOSVideoDecoder::new(&url).expect("decoder");
    decoder.resume().expect("resume");

    let start = Instant::now();
    let (mut gpu, mut cpu, mut imported, mut failed) = (0u32, 0u32, 0u32, 0u32);
    let mut size = (0, 0);
    let mut last_pts = Duration::ZERO;
    while start.elapsed() < Duration::from_secs(seconds) {
        unsafe {
            if let Some(mode) = objc2_core_foundation::kCFRunLoopDefaultMode {
                objc2_core_foundation::CFRunLoop::run_in_mode(Some(mode), 0.005, false);
            }
        }
        match decoder.decode_next() {
            Ok(Some(frame)) => {
                last_pts = frame.pts;
                match frame.frame {
                    DecodedFrame::MacOS(s) => {
                        gpu += 1;
                        size = (s.width, s.height);
                        match unsafe {
                            import_iosurface(
                                &device,
                                s.io_surface,
                                s.width,
                                s.height,
                                wgpu::TextureFormat::Bgra8Unorm,
                            )
                        } {
                            Ok(_) => imported += 1,
                            Err(_) => failed += 1,
                        }
                    }
                    DecodedFrame::Cpu(_) => cpu += 1,
                    #[allow(unreachable_patterns)]
                    _ => {}
                }
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(2)),
            Err(e) => {
                eprintln!("decode error: {e}");
                break;
            }
        }
    }
    println!(
        "gpu_surface_frames={gpu} cpu_frames={cpu} imported={imported} import_failed={failed} size={size:?} last_pts={last_pts:?}"
    );
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {}
