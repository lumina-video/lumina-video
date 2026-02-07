//! lumina-video Android Demo Native Library
//!
//! This crate provides the native Android library for the lumina-video demo app.
//! It exports:
//! - `android_main` for GameActivity integration
//! - JNI functions for ExoPlayerBridge communication

use android_activity::AndroidApp;
use eframe::egui;
use lumina_video::{VideoPlayer, VideoPlayerExt};
use tracing::{error, info, warn};

// Re-export JNI functions from lumina-video so they're available in this cdylib
// The linker will include these symbols from lumina-video
pub use lumina_video::*;

/// HAL viability spike: test if we can access raw Vulkan handles from wgpu
///
/// This verifies that the zero-copy bypass approach is viable on this device.
fn test_hal_viability(render_state: &egui_wgpu::RenderState) -> HalViabilityResult {
    use wgpu_hal::api::Vulkan;

    info!("=== HAL VIABILITY SPIKE ===");

    let device = &render_state.device;
    let adapter = &render_state.adapter;

    // Log adapter info
    let adapter_info = adapter.get_info();
    info!(
        "Adapter: {} ({:?})",
        adapter_info.name, adapter_info.backend
    );
    info!("Driver: {}", adapter_info.driver);
    info!("Driver info: {}", adapter_info.driver_info);

    // Check if Vulkan backend
    if adapter_info.backend != wgpu::Backend::Vulkan {
        warn!("Not using Vulkan backend: {:?}", adapter_info.backend);
        return HalViabilityResult {
            success: false,
            backend: format!("{:?}", adapter_info.backend),
            has_hal_access: false,
            error: Some("Not Vulkan backend".to_string()),
        };
    }

    // Attempt to get raw Vulkan handles via as_hal
    // SAFETY: This unsafe block is sound because:
    // 1. `device` is a valid wgpu::Device passed by reference from the caller
    // 2. The closure receives the HAL device only if the backend is Vulkan
    // 3. All Vulkan handles (raw_device, raw_instance, raw_physical_device) are
    //    only used within the closure scope and not stored beyond it
    // 4. The FFI calls (enumerate_device_extension_properties) use the ash crate's
    //    safe wrapper over the Vulkan C API
    let hal_result = unsafe {
        device.as_hal::<Vulkan, _, _>(|hal_device| {
            match hal_device {
                Some(d) => {
                    // Get raw VkDevice handle
                    let raw_device = d.raw_device();
                    info!("✓ Got raw VkDevice handle: {:?}", raw_device.handle());

                    // Get shared instance
                    let instance = d.shared_instance();
                    let raw_instance = instance.raw_instance();
                    info!("✓ Got raw VkInstance handle: {:?}", raw_instance.handle());

                    // Get physical device
                    let physical_device = d.raw_physical_device();
                    info!("✓ Got VkPhysicalDevice: {:?}", physical_device);

                    // Query device extensions using ash
                    let instance_fns = raw_instance;
                    let extension_props = instance_fns
                        .enumerate_device_extension_properties(physical_device)
                        .unwrap_or_default();

                    // Check for required extensions
                    let required_extensions = [
                        "VK_ANDROID_external_memory_android_hardware_buffer",
                        "VK_KHR_external_memory",
                        "VK_KHR_external_semaphore_fd",
                        "VK_KHR_sampler_ycbcr_conversion",
                    ];

                    let mut missing = Vec::new();
                    for required in &required_extensions {
                        let found = extension_props.iter().any(|ext| {
                            let name = ext
                                .extension_name_as_c_str()
                                .map(|s| s.to_string_lossy())
                                .unwrap_or_default();
                            name == *required
                        });
                        if found {
                            info!("✓ Extension present: {}", required);
                        } else {
                            warn!("✗ Extension MISSING: {}", required);
                            missing.push(*required);
                        }
                    }

                    if missing.is_empty() {
                        info!("=== HAL VIABILITY: SUCCESS ===");
                        info!("All required extensions present. Zero-copy bypass is viable!");
                        (true, None)
                    } else {
                        warn!("=== HAL VIABILITY: PARTIAL ===");
                        warn!("Missing extensions: {:?}", missing);
                        (true, Some(format!("Missing extensions: {:?}", missing)))
                    }
                }
                None => {
                    error!("✗ as_hal returned None - HAL access not available");
                    (false, Some("as_hal returned None".to_string()))
                }
            }
        })
    };

    HalViabilityResult {
        success: hal_result.0,
        backend: "Vulkan".to_string(),
        has_hal_access: hal_result.0,
        error: hal_result.1,
    }
}

/// Result of HAL viability test for zero-copy video rendering.
#[derive(Debug)]
struct HalViabilityResult {
    /// Whether the HAL viability test succeeded (all required extensions present)
    success: bool,
    /// The graphics backend name (e.g., "Vulkan")
    backend: String,
    /// Whether HAL access is available (device, physical device, and instance handles)
    has_hal_access: bool,
    /// Error message if test failed
    error: Option<String>,
}

/// Sample video URL for testing
const SAMPLE_VIDEO_URL: &str =
    "https://storage.googleapis.com/exoplayer-test-media-1/mp4/android-screens-10s.mp4";

/// Android entry point
///
/// This function is called by GameActivity when the native library is loaded.
/// It initializes the egui application with wgpu Vulkan backend.
#[no_mangle]
pub extern "C" fn android_main(app: AndroidApp) {
    // Initialize Android logging
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(tracing::log::LevelFilter::Debug)
            .with_tag("lumina-video"),
    );

    info!("android_main: Starting lumina-video demo");

    // Note: ndk-context is initialized by eframe/android-activity automatically.
    // DO NOT call ndk_context::initialize_android_context here - it will panic
    // with "assertion failed: previous.is_none()" if called twice.

    let options = eframe::NativeOptions {
        android_app: Some(app),
        viewport: egui::ViewportBuilder::default().with_title("lumina-video Demo"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            // Use Vulkan for AHardwareBuffer import support
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(egui_wgpu::WgpuSetupCreateNew {
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::VULKAN,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    if let Err(e) = eframe::run_native(
        "lumina-video Demo",
        options,
        Box::new(|cc| Ok(Box::new(DemoApp::new(cc)))),
    ) {
        tracing::error!("eframe::run_native failed: {}", e);
    }
}

/// Demo application state
struct DemoApp {
    /// Video player instance
    player: Option<VideoPlayer>,
    /// Whether player has been initialized
    initialized: bool,
}

impl DemoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("DemoApp::new: Creating demo app");

        // Run HAL viability spike first
        if let Some(render_state) = cc.wgpu_render_state.as_ref() {
            let result = test_hal_viability(render_state);
            info!("HAL viability result: {:?}", result);
        } else {
            warn!("No wgpu render state - cannot test HAL viability");
        }

        // Create video player if wgpu is available
        let player = cc.wgpu_render_state.as_ref().map(|render_state| {
            info!("Creating VideoPlayer with wgpu");
            VideoPlayer::with_wgpu(SAMPLE_VIDEO_URL, render_state)
                .with_autoplay(true)
                .with_loop(true)
                .with_controls(true)
        });

        let initialized = player.is_some();
        Self {
            player,
            initialized,
        }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref mut player) = self.player {
                // Get available size for video
                let available_size = ui.available_size();

                // Calculate video size maintaining aspect ratio
                let video_size = if let Some(metadata) = player.metadata() {
                    let aspect = metadata.width as f32 / metadata.height as f32;
                    let max_width = available_size.x;
                    let max_height = available_size.y;

                    if max_width / aspect <= max_height {
                        egui::vec2(max_width, max_width / aspect)
                    } else {
                        egui::vec2(max_height * aspect, max_height)
                    }
                } else {
                    // Default 16:9 while loading
                    let aspect = 16.0 / 9.0;
                    egui::vec2(available_size.x, available_size.x / aspect).min(available_size)
                };

                // Render the video player
                ui.centered_and_justified(|ui| {
                    ui.video_player(player, video_size);
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.heading("Initializing...");
                    if !self.initialized {
                        ui.label("wgpu render state not available");
                    }
                });
            }
        });

        // Request continuous repaint for video playback
        ctx.request_repaint();
    }
}
