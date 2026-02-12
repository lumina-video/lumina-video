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
            backend: format!("{:?}", adapter_info.backend),
            has_hal_access: false,
            ext_memory_ahb: false,
            ext_memory: false,
            ext_semaphore: false,
            ycbcr_conv: false,
            all_extensions_present: false,
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
                    let raw_device = d.raw_device();
                    info!("Got raw VkDevice handle: {:?}", raw_device.handle());

                    let instance = d.shared_instance();
                    let raw_instance = instance.raw_instance();
                    info!("Got raw VkInstance handle: {:?}", raw_instance.handle());

                    let physical_device = d.raw_physical_device();
                    info!("Got VkPhysicalDevice: {:?}", physical_device);

                    // Query device extensions using ash
                    match raw_instance.enumerate_device_extension_properties(physical_device) {
                        Ok(extension_props) => {
                            let has_ext = |name: &str| -> bool {
                                extension_props.iter().any(|ext| {
                                    ext.extension_name_as_c_str()
                                        .map(|s| s.to_string_lossy())
                                        .unwrap_or_default()
                                        == name
                                })
                            };

                            let ext_memory_ahb =
                                has_ext("VK_ANDROID_external_memory_android_hardware_buffer");
                            let ext_memory = has_ext("VK_KHR_external_memory");
                            let ext_semaphore = has_ext("VK_KHR_external_semaphore_fd");
                            let ycbcr_conv = has_ext("VK_KHR_sampler_ycbcr_conversion");

                            for (name, present) in [
                                (
                                    "VK_ANDROID_external_memory_android_hardware_buffer",
                                    ext_memory_ahb,
                                ),
                                ("VK_KHR_external_memory", ext_memory),
                                ("VK_KHR_external_semaphore_fd", ext_semaphore),
                                ("VK_KHR_sampler_ycbcr_conversion", ycbcr_conv),
                            ] {
                                if present {
                                    info!("Extension present: {}", name);
                                } else {
                                    warn!("Extension MISSING: {}", name);
                                }
                            }

                            let all = ext_memory_ahb && ext_memory && ext_semaphore && ycbcr_conv;

                            if all {
                                info!("=== HAL VIABILITY: SUCCESS ===");
                            } else {
                                warn!("=== HAL VIABILITY: PARTIAL ===");
                            }

                            (
                                true,
                                ext_memory_ahb,
                                ext_memory,
                                ext_semaphore,
                                ycbcr_conv,
                                all,
                                None,
                            )
                        }
                        Err(e) => {
                            error!("Failed to enumerate Vulkan device extensions: {:?}", e);
                            (
                                true, // HAL access worked, query didn't
                                false,
                                false,
                                false,
                                false,
                                false,
                                Some(format!(
                                    "Failed to enumerate Vulkan device extensions: {:?}",
                                    e
                                )),
                            )
                        }
                    }
                }
                None => {
                    error!("as_hal returned None - HAL access not available");
                    (
                        false,
                        false,
                        false,
                        false,
                        false,
                        false,
                        Some("as_hal returned None".to_string()),
                    )
                }
            }
        })
    };

    HalViabilityResult {
        backend: "Vulkan".to_string(),
        has_hal_access: hal_result.0,
        ext_memory_ahb: hal_result.1,
        ext_memory: hal_result.2,
        ext_semaphore: hal_result.3,
        ycbcr_conv: hal_result.4,
        all_extensions_present: hal_result.5,
        error: hal_result.6,
    }
}

/// Result of HAL viability test for zero-copy video rendering.
#[derive(Debug)]
struct HalViabilityResult {
    /// The graphics backend name (e.g., "Vulkan")
    backend: String,
    /// Whether as_hal() succeeded (VkPhysicalDevice available)
    has_hal_access: bool,
    /// VK_ANDROID_external_memory_android_hardware_buffer
    ext_memory_ahb: bool,
    /// VK_KHR_external_memory
    ext_memory: bool,
    /// VK_KHR_external_semaphore_fd
    ext_semaphore: bool,
    /// VK_KHR_sampler_ycbcr_conversion
    ycbcr_conv: bool,
    /// All 4 required extensions present
    all_extensions_present: bool,
    /// Error message if any failure occurred
    error: Option<String>,
}

/// Sample video URL for testing.
/// VideoPlayer::with_wgpu() creates an AndroidVideoDecoder which calls
/// LuminaVideo.createPlayer() to get a self-contained ExoPlayer instance.
const SAMPLE_VIDEO_URL: &str =
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4";

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

/// Fetches Android device model and API level via JNI.
fn fetch_device_info() -> (String, i32) {
    // TODO(e): extract shared JNI device-info helper to avoid drift with android_video.rs
    // SAFETY: ndk_context::android_context().vm() returns the JavaVM pointer set by
    // android-activity during initialization. The pointer is valid for the process lifetime.
    // JavaVM::from_raw wraps it without taking ownership (we don't call destroy_java_vm).
    let vm = match unsafe { jni::JavaVM::from_raw(ndk_context::android_context().vm().cast()) } {
        Ok(vm) => vm,
        Err(e) => {
            error!("Failed to get JavaVM: {}", e);
            return ("Unknown".to_string(), 0);
        }
    };

    let mut env = match vm.attach_current_thread() {
        Ok(env) => env,
        Err(e) => {
            error!("Failed to attach JNI thread: {}", e);
            return ("Unknown".to_string(), 0);
        }
    };

    let model = (|| -> Option<String> {
        let build_class = env.find_class("android/os/Build").ok()?;
        let model_obj = env
            .get_static_field(&build_class, "MODEL", "Ljava/lang/String;")
            .ok()?
            .l()
            .ok()?;
        env.get_string((&model_obj).into()).map(|s| s.into()).ok()
    })()
    .unwrap_or_else(|| "Unknown".to_string());

    let api_level = (|| -> Option<i32> {
        let version_class = env.find_class("android/os/Build$VERSION").ok()?;
        env.get_static_field(&version_class, "SDK_INT", "I")
            .ok()?
            .i()
            .ok()
    })()
    .unwrap_or(0);

    info!("Device: {} (API {})", model, api_level);
    (model, api_level)
}

/// Demo application state
struct DemoApp {
    /// Video player instance
    player: Option<VideoPlayer>,
    /// Whether player has been initialized
    initialized: bool,
    /// HAL viability result (None if no render state)
    hal_result: Option<HalViabilityResult>,
    /// Device model name
    device_model: String,
    /// Android API level (SDK_INT)
    api_level: i32,
    /// Whether debug panel is visible
    show_debug_panel: bool,
}

impl DemoApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("DemoApp::new: Creating demo app");

        let (device_model, api_level) = fetch_device_info();

        // Run HAL viability spike
        let hal_result = cc.wgpu_render_state.as_ref().map(|render_state| {
            let result = test_hal_viability(render_state);
            info!("HAL viability result: {:?}", result);
            result
        });
        if hal_result.is_none() {
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
            hal_result,
            device_model,
            api_level,
            show_debug_panel: true,
        }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Top bar with debug toggle
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("lumina-video");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let label = if self.show_debug_panel {
                        "Debug [-]"
                    } else {
                        "Debug [+]"
                    };
                    if ui.button(label).clicked() {
                        self.show_debug_panel = !self.show_debug_panel;
                    }
                });
            });
        });

        // Debug side panel
        if self.show_debug_panel {
            egui::SidePanel::right("debug_panel")
                .max_width(260.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.render_debug_panel(ui);
                    });
                });
        }

        // Main video area
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref mut player) = self.player {
                let available_size = ui.available_size();

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
                    let aspect = 16.0 / 9.0;
                    egui::vec2(available_size.x, available_size.x / aspect).min(available_size)
                };

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

        // Only spin the render loop when video is active — avoids burning CPU/GPU when idle
        if self.player.as_ref().is_some_and(|p| p.is_playing()) {
            ctx.request_repaint();
        } else {
            // Repaint at reduced rate for UI responsiveness (debug panel updates, state changes)
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
    }
}

impl DemoApp {
    fn render_debug_panel(&self, ui: &mut egui::Ui) {
        let check = |ok: bool| if ok { "[Y]" } else { "[X]" };

        // --- Device section ---
        ui.heading("Device");
        ui.separator();
        egui::Grid::new("device_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                ui.label("Model:");
                ui.label(&self.device_model);
                ui.end_row();

                ui.label("API Level:");
                ui.label(format!("{}", self.api_level));
                ui.end_row();

                ui.label("Backend:");
                match &self.hal_result {
                    Some(hal) => {
                        ui.label(&hal.backend);
                    }
                    None => {
                        ui.label("Unknown");
                    }
                }
                ui.end_row();

                ui.label("Vulkan HAL:");
                match &self.hal_result {
                    Some(hal) if hal.has_hal_access => {
                        ui.colored_label(egui::Color32::GREEN, format!("{} Active", check(true)));
                    }
                    Some(hal) => {
                        ui.colored_label(
                            egui::Color32::RED,
                            format!(
                                "{} {}",
                                check(false),
                                hal.error.as_deref().unwrap_or("Unavailable")
                            ),
                        );
                    }
                    None => {
                        ui.label("Unknown (no render state)");
                    }
                }
                ui.end_row();
            });

        ui.add_space(8.0);

        // --- Zero-Copy section ---
        ui.heading("Zero-Copy Extensions");
        ui.separator();
        egui::Grid::new("zerocopy_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                // AHB API availability (OS-level, independent of Vulkan extensions)
                ui.label("AHB API (26+):");
                if self.api_level >= 26 {
                    ui.colored_label(
                        egui::Color32::GREEN,
                        format!("{} (API {})", check(true), self.api_level),
                    );
                } else if self.api_level > 0 {
                    ui.colored_label(
                        egui::Color32::RED,
                        format!("{} (API {})", check(false), self.api_level),
                    );
                } else {
                    ui.label("Unknown");
                }
                ui.end_row();

                // Per-extension rows
                match &self.hal_result {
                    Some(hal) if hal.has_hal_access => {
                        for (label, present) in [
                            ("VK AHB import:", hal.ext_memory_ahb),
                            ("VK ext_memory:", hal.ext_memory),
                            ("VK semaphore_fd:", hal.ext_semaphore),
                            ("VK ycbcr_conv:", hal.ycbcr_conv),
                        ] {
                            ui.label(label);
                            let color = if present {
                                egui::Color32::GREEN
                            } else {
                                egui::Color32::RED
                            };
                            ui.colored_label(color, check(present));
                            ui.end_row();
                        }

                        ui.label("All present:");
                        if hal.all_extensions_present {
                            ui.colored_label(egui::Color32::GREEN, check(true));
                        } else {
                            ui.colored_label(egui::Color32::RED, check(false));
                        }
                        ui.end_row();

                        if let Some(ref err) = hal.error {
                            ui.label("Error:");
                            ui.colored_label(egui::Color32::YELLOW, err.as_str());
                            ui.end_row();
                        }
                    }
                    Some(_) => {
                        ui.label("Extensions:");
                        ui.colored_label(egui::Color32::RED, "No HAL access");
                        ui.end_row();
                    }
                    None => {
                        ui.label("Extensions:");
                        ui.label("Unknown (no render state)");
                        ui.end_row();
                    }
                }
            });

        ui.add_space(8.0);

        // --- Rendering section ---
        ui.heading("Rendering");
        ui.separator();
        egui::Grid::new("rendering_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                let zc = lumina_video::android_zero_copy_snapshot(0);
                let total = zc.total();

                // Status based on most recent frame, not lifetime totals
                ui.label("Zero-Copy:");
                match zc.current_status {
                    lumina_video::ZeroCopyStatus::Waiting => {
                        ui.colored_label(egui::Color32::GRAY, "Waiting...");
                    }
                    lumina_video::ZeroCopyStatus::ZeroCopy => {
                        ui.colored_label(egui::Color32::GREEN, "YES");
                    }
                    lumina_video::ZeroCopyStatus::CpuAssisted => {
                        ui.colored_label(egui::Color32::YELLOW, "CPU FALLBACK");
                    }
                    lumina_video::ZeroCopyStatus::Failed => {
                        ui.colored_label(egui::Color32::RED, "FAILED");
                    }
                }
                ui.end_row();

                if total > 0 {
                    ui.label("Frames:");
                    ui.label(format!(
                        "{} zero-copy, {} cpu, {} failed",
                        zc.true_zero_copy_frames, zc.cpu_assisted_frames, zc.failed_frames
                    ));
                    ui.end_row();
                }
            });

        ui.add_space(8.0);

        // --- Video / Playback section ---
        ui.heading("Video / Playback");
        ui.separator();
        egui::Grid::new("playback_grid")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                if let Some(ref player) = self.player {
                    if let Some(meta) = player.metadata() {
                        ui.label("Resolution:");
                        ui.label(format!("{}x{}", meta.width, meta.height));
                        ui.end_row();

                        ui.label("Codec:");
                        ui.label(&meta.codec);
                        ui.end_row();

                        ui.label("FPS:");
                        ui.label(format!("{:.1}", meta.frame_rate));
                        ui.end_row();
                    } else {
                        ui.label("Resolution:");
                        ui.colored_label(egui::Color32::GRAY, "Loading...");
                        ui.end_row();
                    }

                    ui.label("State:");
                    ui.label(format!("{:?}", player.state()));
                    ui.end_row();

                    ui.label("Position:");
                    let pos = player.position();
                    let pos_s = pos.as_secs_f64();
                    if let Some(dur) = player.duration() {
                        let dur_s = dur.as_secs_f64();
                        ui.label(format!("{:.1}s / {:.1}s", pos_s, dur_s));
                    } else {
                        ui.label(format!("{:.1}s / --", pos_s));
                    }
                    ui.end_row();
                } else {
                    ui.label("Player:");
                    ui.colored_label(egui::Color32::GRAY, "Not initialized");
                    ui.end_row();
                }
            });
    }
}
