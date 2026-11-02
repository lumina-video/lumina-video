use ash::vk;
use std::ffi::CStr;

const REQUIRED_EXTENSIONS: &[&CStr] = &[
    c"VK_ANDROID_external_memory_android_hardware_buffer",
    c"VK_KHR_external_memory",
    c"VK_KHR_dedicated_allocation",
    c"VK_KHR_sampler_ycbcr_conversion",
    c"VK_KHR_external_semaphore",
    c"VK_KHR_external_semaphore_fd",
    c"VK_KHR_bind_memory2",
    c"VK_KHR_get_memory_requirements2",
    c"VK_EXT_queue_family_foreign",
];

pub fn android_wgpu_setup(
) -> Result<egui_wgpu::WgpuSetupExisting, Box<dyn std::error::Error + Send + Sync>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster_block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;

    let features = adapter.features() & wgpu::Features::default();
    let limits = adapter.limits();
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("lumina-video android"),
        required_features: features,
        required_limits: limits.clone(),
        ..Default::default()
    };

    let opened = unsafe {
        let hal_adapter = adapter
            .as_hal::<wgpu::hal::api::Vulkan>()
            .ok_or("adapter is not Vulkan")?;
        let supported: Vec<_> = hal_adapter
            .shared_instance()
            .raw_instance()
            .enumerate_device_extension_properties(hal_adapter.raw_physical_device())?;
        let has = |name: &CStr| {
            supported
                .iter()
                .any(|e| e.extension_name_as_c_str().map(|n| n == name).unwrap_or(false))
        };
        let extra: Vec<&'static CStr> = REQUIRED_EXTENSIONS
            .iter()
            .copied()
            .filter(|e| has(e))
            .collect();

        hal_adapter.open_with_callback(
            features,
            &limits,
            &wgpu::MemoryHints::Performance,
            Some(Box::new(move |args| {
                for e in &extra {
                    if !args.extensions.contains(e) {
                        args.extensions.push(e);
                    }
                }
                let ycbcr: &'static mut vk::PhysicalDeviceSamplerYcbcrConversionFeatures<'static> =
                    Box::leak(Box::new(
                        vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
                            .sampler_ycbcr_conversion(true),
                    ));
                *args.create_info = std::mem::take(args.create_info).push_next(ycbcr);
            })),
        )?
    };

    let (device, queue) = unsafe { adapter.create_device_from_hal(opened, &descriptor)? };

    Ok(egui_wgpu::WgpuSetupExisting {
        instance,
        adapter,
        device,
        queue,
    })
}

fn pollster_block_on<F: std::future::Future>(f: F) -> F::Output {
    pollster::block_on(f)
}
