#![cfg(target_os = "macos")]

use std::ffi::c_void;

use lumina_video_core::zero_copy::macos::import_iosurface;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{NSMutableDictionary, NSNumber, NSString};

extern "C" {
    fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut u8;
    fn IOSurfaceGetBytesPerRow(surface: *mut c_void) -> usize;
    fn CFRelease(cf: *const c_void);
}

const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;

fn pixel(x: u32, y: u32) -> [u8; 4] {
    [(x * 4) as u8, (y * 8) as u8, (x + y) as u8, 255]
}

fn make_surface() -> *mut c_void {
    let dict: Retained<NSMutableDictionary<NSString, AnyObject>> = NSMutableDictionary::new();
    let set = |key: &str, value: u32| unsafe {
        let n = NSNumber::new_u32(value);
        let k = NSString::from_str(key);
        dict.setObject_forKey(&*n, objc2::runtime::ProtocolObject::from_ref(&*k));
    };
    set("IOSurfaceWidth", WIDTH);
    set("IOSurfaceHeight", HEIGHT);
    set("IOSurfaceBytesPerElement", 4);
    set("IOSurfacePixelFormat", u32::from_be_bytes(*b"BGRA"));

    let surface = unsafe { IOSurfaceCreate(Retained::as_ptr(&dict) as *const c_void) };
    assert!(!surface.is_null(), "IOSurfaceCreate failed");

    unsafe {
        IOSurfaceLock(surface, 0, std::ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface);
        let stride = IOSurfaceGetBytesPerRow(surface);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let p = pixel(x, y);
                std::ptr::copy_nonoverlapping(
                    p.as_ptr(),
                    base.add(y as usize * stride + x as usize * 4),
                    4,
                );
            }
        }
        IOSurfaceUnlock(surface, 0, std::ptr::null_mut());
    }
    surface
}

#[test]
fn iosurface_imported_into_wgpu_metal_texture_has_same_pixels() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("device");

    let surface = make_surface();
    let texture = unsafe {
        import_iosurface(&device, surface, WIDTH, HEIGHT, wgpu::TextureFormat::Bgra8Unorm)
    }
    .expect("import");

    let stride = (WIDTH * 4).next_multiple_of(256);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (stride * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range().expect("mapped range");
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let off = (y * stride + x * 4) as usize;
            assert_eq!(&data[off..off + 4], &pixel(x, y), "pixel {x},{y}");
        }
    }
    drop(data);
    drop(texture);
    unsafe { CFRelease(surface) };
}
