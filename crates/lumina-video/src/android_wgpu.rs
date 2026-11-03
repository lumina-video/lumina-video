#[cfg(target_os = "android")]
use ash::vk;
use std::ffi::CStr;

const REQUIRED_EXTENSIONS: &[&CStr] = &[
    c"VK_ANDROID_external_memory_android_hardware_buffer",
    c"VK_KHR_external_memory",
    c"VK_KHR_dedicated_allocation",
    YCBCR_EXTENSION,
    c"VK_KHR_external_semaphore",
    c"VK_KHR_external_semaphore_fd",
    c"VK_KHR_bind_memory2",
    c"VK_KHR_get_memory_requirements2",
    c"VK_EXT_queue_family_foreign",
];

const YCBCR_EXTENSION: &CStr = c"VK_KHR_sampler_ycbcr_conversion";

#[cfg(target_os = "android")]
pub fn android_wgpu_setup(
) -> Result<egui_wgpu::WgpuSetupExisting, Box<dyn std::error::Error + Send + Sync>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;

    let features = wgpu::Features::empty();
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
            supported.iter().any(|e| {
                e.extension_name_as_c_str()
                    .map(|n| n == name)
                    .unwrap_or(false)
            })
        };
        let extra = supported_extensions(REQUIRED_EXTENSIONS, has);
        let mut ycbcr_storage = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
            .sampler_ycbcr_conversion(true);
        let mut ycbcr_features = extra
            .contains(&YCBCR_EXTENSION)
            .then_some(&mut ycbcr_storage);

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
                if let Some(ycbcr) = ycbcr_features.take() {
                    *args.create_info = std::mem::take(args.create_info).push_next(ycbcr);
                }
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

fn supported_extensions(
    required: &[&'static CStr],
    has: impl Fn(&CStr) -> bool,
) -> Vec<&'static CStr> {
    required.iter().copied().filter(|e| has(e)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_extensions_are_not_requested() {
        let present = [c"VK_KHR_external_memory", c"VK_KHR_bind_memory2"];
        let got = supported_extensions(REQUIRED_EXTENSIONS, |e| present.contains(&e));
        assert_eq!(got, present.to_vec());
        assert!(!got.contains(&YCBCR_EXTENSION));
    }
}
