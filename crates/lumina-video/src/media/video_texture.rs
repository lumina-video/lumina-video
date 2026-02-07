//! Video texture management for wgpu rendering.
//!
//! This module handles uploading decoded video frames to GPU textures
//! and provides the rendering pipeline for YUV to RGB conversion.

use std::borrow::Cow;
use std::num::NonZeroU64;
#[cfg(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", feature = "windows-native-video")
))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "android",
    all(target_os = "windows", feature = "windows-native-video")
))]
use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};
#[cfg(not(target_arch = "wasm32"))]
use egui::Rect;

use super::video::{CpuFrame, PixelFormat};

// Zero-copy imports for platform-specific GPU texture import
// Each platform module import requires both zero-copy AND the platform-specific feature

// ZeroCopyError is platform-agnostic (defined at module root, not in platform submodules)
// Only import on platforms where from_imported_yuv_planes is used
#[cfg(any(target_os = "linux", target_os = "android"))]
use super::zero_copy::ZeroCopyError;

// Platform-specific module aliases (for accessing ::macos, ::linux, etc. submodules)
#[cfg(target_os = "android")]
use super::zero_copy as zero_copy_android;

// libc for fence sync on Android (poll/close)
#[cfg(target_os = "android")]
extern crate libc;
#[cfg(target_os = "macos")]
use super::zero_copy;
#[cfg(all(target_os = "windows", feature = "windows-native-video"))]
use super::zero_copy as zero_copy_windows;
#[cfg(target_os = "linux")]
use super::zero_copy as zero_copy_linux;

// NOTE: Per-instance fallback logging is now in VideoRenderCallback.fallback_logged
// This ensures multiple video players don't suppress each other's warnings.

/// wgpu requires bytes_per_row to be aligned to this value.
const WGPU_COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Aligns a value up to the nearest multiple of alignment.
fn align_up(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Pads row data to meet wgpu's bytes_per_row alignment requirement.
/// Returns (aligned_bytes_per_row, data) - uses Cow to avoid copying when already aligned.
fn pad_plane_data(data: &[u8], stride: usize, height: u32) -> (u32, Cow<'_, [u8]>) {
    let stride_u32 = stride as u32;
    let aligned_stride = align_up(stride_u32, WGPU_COPY_BYTES_PER_ROW_ALIGNMENT);

    if aligned_stride == stride_u32 {
        // Already aligned - borrow without copying
        return (stride_u32, Cow::Borrowed(data));
    }

    // Need to pad each row
    let mut padded = Vec::with_capacity((aligned_stride * height) as usize);
    for row in 0..height as usize {
        let row_start = row * stride;
        let row_end = row_start + stride;
        if row_end <= data.len() {
            padded.extend_from_slice(&data[row_start..row_end]);
        } else {
            // Truncated plane data - zero-fill missing bytes
            let available = data.len().saturating_sub(row_start);
            if available > 0 {
                padded.extend_from_slice(&data[row_start..row_start + available]);
            }
            padded.resize(padded.len() + stride - available, 0);
        }
        // Add alignment padding
        padded.resize(padded.len() + (aligned_stride - stride_u32) as usize, 0);
    }

    (aligned_stride, Cow::Owned(padded))
}

/// Resources for rendering video frames via wgpu.
///
/// This struct is stored in egui's callback resources and contains
/// all the GPU resources needed to render video frames.
#[allow(dead_code)] // Fields used only on non-wasm32 platforms
pub struct VideoRenderResources {
    /// The render pipeline for YUV to RGB conversion
    pipeline_yuv420p: wgpu::RenderPipeline,
    /// The render pipeline for NV12 format
    pipeline_nv12: wgpu::RenderPipeline,
    /// The render pipeline for RGB passthrough
    pipeline_rgb: wgpu::RenderPipeline,
    /// Bind group layout for video textures
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for transform data
    uniform_buffer: wgpu::Buffer,
    /// Texture sampler
    sampler: wgpu::Sampler,
    /// Cached 1x1 dummy texture for unused UV planes in RGB zero-copy path
    #[cfg(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        all(target_os = "windows", feature = "windows-native-video")
    ))]
    dummy_texture: wgpu::Texture,
    /// Depth format used to build pipelines (must match host render pass).
    depth_format: Option<wgpu::TextureFormat>,
}

impl VideoRenderResources {
    /// Creates video render resources.
    ///
    /// This should be called once during application initialization.
    ///
    /// # Arguments
    ///
    /// * `wgpu_render_state` - The egui wgpu render state
    /// * `depth_format` - Optional depth format to match the render pass. Pass `None` if
    ///   the render pass has no depth attachment, or `Some(format)` to match the host app's
    ///   depth buffer format (e.g., `Depth24Plus`, `Depth32Float`).
    pub fn new(
        wgpu_render_state: &egui_wgpu::RenderState,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("video.wgsl"))),
        });

        // Create sampler for texture sampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create uniform buffer for video transform
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("video_uniform_buffer"),
            size: 16, // 4 floats for transform
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("video_bind_group_layout"),
            entries: &[
                // Uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
                // Y texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // U texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // V texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("video_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipelines for different formats
        let create_pipeline = |entry_point: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu_render_state.target_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                // Video rendering doesn't need depth testing - it's always drawn on top
                // But we must match the render pass depth format to avoid wgpu validation errors
                // Use the host app's depth format with testing disabled (Always compare, no writes)
                depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let pipeline_yuv420p = create_pipeline("fs_main", "video_pipeline_yuv420p");
        let pipeline_nv12 = create_pipeline("fs_main_nv12", "video_pipeline_nv12");
        let pipeline_rgb = create_pipeline("fs_main_rgb", "video_pipeline_rgb");

        // Create cached 1x1 dummy texture for zero-copy RGB path (avoids per-frame allocation)
        #[cfg(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "android",
            all(target_os = "windows", feature = "windows-native-video")
        ))]
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("video_dummy_texture_cached"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        Self {
            pipeline_yuv420p,
            pipeline_nv12,
            pipeline_rgb,
            bind_group_layout,
            uniform_buffer,
            sampler,
            #[cfg(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "android",
                all(target_os = "windows", feature = "windows-native-video")
            ))]
            dummy_texture,
            depth_format,
        }
    }

    /// Returns the depth format used to create pipelines.
    pub fn depth_format(&self) -> Option<wgpu::TextureFormat> {
        self.depth_format
    }

    /// Registers the video render resources with egui's callback system.
    ///
    /// # Behavior
    ///
    /// - **Last registration wins**: If resources already exist with a different `depth_format`,
    ///   they are replaced to ensure pipeline compatibility. This allows apps to correct a
    ///   previous registration (e.g., calling `with_wgpu_and_depth` after an initial `with_wgpu`).
    /// - **Idempotent**: If resources exist with the same `depth_format`, this is a no-op.
    /// - **Thread-safe**: Uses double-checked locking to minimize contention. Pipeline creation
    ///   happens outside the write lock to avoid blocking the renderer.
    ///
    /// # Arguments
    ///
    /// * `wgpu_render_state` - The egui wgpu render state
    /// * `depth_format` - Optional depth format to match the render pass. Pass `None` if
    ///   the render pass has no depth attachment, or `Some(format)` to match the host app's
    ///   depth buffer format (e.g., `Depth24Plus`, `Depth32Float`).
    ///
    /// # Note for Multi-Pass Apps
    ///
    /// If your app uses multiple render passes with different depth formats on the same
    /// `RenderState`, be aware that all `VideoPlayer` instances share these resources.
    /// The last `depth_format` registered will be used for all video rendering.
    pub fn register(
        wgpu_render_state: &egui_wgpu::RenderState,
        depth_format: Option<wgpu::TextureFormat>,
    ) {
        // First check under read lock to see if we can early-return
        {
            let renderer = wgpu_render_state.renderer.read();
            if let Some(existing) = renderer.callback_resources.get::<VideoRenderResources>() {
                if existing.depth_format() == depth_format {
                    return;
                }
            }
        }

        // Build resources without holding the renderer lock (pipeline creation is expensive)
        let resources = Self::new(wgpu_render_state, depth_format);

        // Re-check and insert under write lock
        let mut renderer = wgpu_render_state.renderer.write();
        if let Some(existing) = renderer.callback_resources.get::<VideoRenderResources>() {
            if existing.depth_format() == depth_format {
                // Another thread already registered with correct format
                return;
            }
            tracing::info!(
                "VideoRenderResources depth format changed from {:?} to {:?}, rebuilding pipelines",
                existing.depth_format(),
                depth_format
            );
        }
        renderer.callback_resources.insert(resources);
    }
}

/// A video texture that can be uploaded to the GPU and rendered.
///
/// This handles the GPU-side representation of a video frame, including
/// texture creation, upload, and bind group management.
#[allow(dead_code)] // Fields used only on non-wasm32 platforms
pub struct VideoTexture {
    /// Y plane texture (or RGB texture for RGB formats)
    y_texture: wgpu::Texture,
    /// U plane texture (or UV texture for NV12)
    u_texture: wgpu::Texture,
    /// V plane texture (unused for NV12/RGB)
    v_texture: wgpu::Texture,
    /// Current bind group for shader access to texture views.
    ///
    /// # wgpu Internal Reference Behavior
    ///
    /// `wgpu::BindGroup` holds strong (Arc) references to the texture views
    /// passed during creation. This means we don't need to store `y_view`,
    /// `u_view`, and `v_view` separately - they're kept alive by the bind group.
    ///
    /// **Warning**: If wgpu ever changes this behavior to use weak references
    /// or require external lifetime management, this code could regress since
    /// we removed explicit y_view/u_view/v_view storage. Monitor wgpu releases
    /// for any changes to `BindGroup` resource lifetime semantics.
    bind_group: wgpu::BindGroup,
    /// Video dimensions
    width: u32,
    height: u32,
    /// Pixel format
    format: PixelFormat,
    /// Keeps the Android AHardwareBuffer alive for the texture's lifetime.
    /// The Vulkan-imported texture references the underlying buffer memory,
    /// so we must prevent AHardwareBuffer_release until the texture is dropped.
    #[cfg(target_os = "android")]
    _android_owner: Option<super::android_video::AndroidVideoFrame>,
}

impl VideoTexture {
    /// Creates a new video texture with the specified dimensions and format.
    pub fn new(
        device: &wgpu::Device,
        resources: &VideoRenderResources,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Self {
        // Calculate texture sizes based on format
        // Use ceiling division for chroma planes to handle odd dimensions
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let (y_size, u_size, v_size) = match format {
            PixelFormat::Yuv420p => (
                (width, height),
                (chroma_width, chroma_height),
                (chroma_width, chroma_height),
            ),
            PixelFormat::Nv12 => ((width, height), (chroma_width, chroma_height), (1, 1)),
            PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgra => {
                ((width, height), (1, 1), (1, 1))
            }
        };

        // Determine texture format
        let y_format = match format {
            PixelFormat::Yuv420p | PixelFormat::Nv12 => wgpu::TextureFormat::R8Unorm,
            PixelFormat::Rgb24 => wgpu::TextureFormat::Rgba8Unorm, // Will need conversion
            PixelFormat::Rgba => wgpu::TextureFormat::Rgba8Unorm,
            PixelFormat::Bgra => wgpu::TextureFormat::Bgra8Unorm,
        };

        let u_format = match format {
            PixelFormat::Yuv420p => wgpu::TextureFormat::R8Unorm,
            PixelFormat::Nv12 => wgpu::TextureFormat::Rg8Unorm, // Interleaved UV
            _ => wgpu::TextureFormat::R8Unorm,
        };

        let v_format = wgpu::TextureFormat::R8Unorm;

        // Create textures
        let create_texture = |size: (u32, u32), format: wgpu::TextureFormat, label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let y_texture = create_texture(y_size, y_format, "video_y_texture");
        let u_texture = create_texture(u_size, u_format, "video_u_texture");
        let v_texture = create_texture(v_size, v_format, "video_v_texture");

        let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let u_view = u_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = v_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video_bind_group"),
            layout: &resources.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        });

        Self {
            y_texture,
            u_texture,
            v_texture,
            bind_group,
            width,
            height,
            format,
            #[cfg(target_os = "android")]
            _android_owner: None,
        }
    }

    /// Creates a VideoTexture from an externally imported RGBA/BGRA texture.
    ///
    /// This is used for zero-copy rendering where the texture is imported directly
    /// from a platform-specific GPU surface (e.g., IOSurface on macOS).
    /// The imported texture is used as the Y texture with the RGB pipeline.
    #[cfg(any(
        target_os = "macos",
        target_os = "linux",
        target_os = "android",
        all(target_os = "windows", feature = "windows-native-video")
    ))]
    pub fn from_imported_rgba(
        device: &wgpu::Device,
        resources: &VideoRenderResources,
        imported_texture: wgpu::Texture,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Self {
        // Create view for the imported texture
        let y_view = imported_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use cached dummy texture for U/V planes (not used for RGB rendering)
        // We create views from the cached texture but reference it for u_texture/v_texture fields
        let u_view = resources
            .dummy_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = resources
            .dummy_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group with the imported texture
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video_bind_group_imported"),
            layout: &resources.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        });

        // Clone the cached dummy texture for the struct fields
        // (wgpu textures are cheap to clone - Arc internally)
        let dummy_texture = resources.dummy_texture.clone();

        Self {
            y_texture: imported_texture,
            u_texture: dummy_texture.clone(),
            v_texture: dummy_texture,
            bind_group,
            width,
            height,
            format,
            #[cfg(target_os = "android")]
            _android_owner: None,
        }
    }

    /// Creates a VideoTexture from externally imported YUV plane textures.
    ///
    /// This is used for zero-copy rendering of multi-plane formats (NV12, YUV420p)
    /// where each plane is imported as a separate wgpu texture.
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu device
    /// * `resources` - Video render resources with bind group layout
    /// * `textures` - Vector of imported plane textures:
    ///   - NV12: `[Y (R8), UV (RG8)]`
    ///   - YUV420p: `[Y (R8), U (R8), V (R8)]`
    /// * `width` - Full frame width
    /// * `height` - Full frame height
    /// * `format` - Pixel format (NV12 or YUV420p)
    ///
    /// # Errors
    ///
    /// Returns `ZeroCopyError::NotAvailable` if:
    /// - Insufficient planes provided (NV12 needs 2, YUV420p needs 3)
    /// - Non-YUV format passed (only NV12 and YUV420p supported)
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn from_imported_yuv_planes(
        device: &wgpu::Device,
        resources: &VideoRenderResources,
        mut textures: Vec<wgpu::Texture>,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<Self, ZeroCopyError> {
        // Handle NV12 (2 planes) vs YUV420p (3 planes)
        let (y_texture, u_texture, v_texture) = match format {
            PixelFormat::Nv12 => {
                // NV12: Y (plane 0), UV interleaved (plane 1)
                // The UV texture is used for both U and V bindings
                // The shader will sample .r for U and .g for V from the RG8 texture
                let uv_texture = textures.pop().ok_or_else(|| {
                    ZeroCopyError::NotAvailable(format!(
                        "NV12 requires 2 planes, got {}",
                        textures.len()
                    ))
                })?;
                let y_texture = textures.pop().ok_or_else(|| {
                    ZeroCopyError::NotAvailable(format!(
                        "NV12 requires 2 planes, got {} remaining after UV pop",
                        textures.len()
                    ))
                })?;
                // Clone UV texture for V binding (same underlying texture)
                // Note: wgpu textures are cheap to clone (Arc internally)
                (y_texture, uv_texture.clone(), uv_texture)
            }
            PixelFormat::Yuv420p => {
                // YUV420p: Y (plane 0), U (plane 1), V (plane 2)
                let v_texture = textures.pop().ok_or_else(|| {
                    ZeroCopyError::NotAvailable(format!(
                        "YUV420p requires 3 planes, got {}",
                        textures.len()
                    ))
                })?;
                let u_texture = textures.pop().ok_or_else(|| {
                    ZeroCopyError::NotAvailable(format!(
                        "YUV420p requires 3 planes, got {} remaining after V pop",
                        textures.len()
                    ))
                })?;
                let y_texture = textures.pop().ok_or_else(|| {
                    ZeroCopyError::NotAvailable(format!(
                        "YUV420p requires 3 planes, got {} remaining after U pop",
                        textures.len()
                    ))
                })?;
                (y_texture, u_texture, v_texture)
            }
            _ => {
                return Err(ZeroCopyError::NotAvailable(format!(
                    "from_imported_yuv_planes only supports NV12/YUV420p, got {format:?}"
                )));
            }
        };

        // Create views for each plane
        let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let u_view = u_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let v_view = v_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group with all plane textures
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video_bind_group_imported_yuv"),
            layout: &resources.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&y_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&u_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&v_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        });

        Ok(Self {
            y_texture,
            u_texture,
            v_texture,
            bind_group,
            width,
            height,
            format,
            #[cfg(target_os = "android")]
            _android_owner: None,
        })
    }

    /// Uploads a video frame to the GPU textures.
    pub fn upload(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        match frame.format {
            PixelFormat::Yuv420p => {
                self.upload_yuv420p(queue, frame);
            }
            PixelFormat::Nv12 => {
                self.upload_nv12(queue, frame);
            }
            PixelFormat::Rgba | PixelFormat::Bgra => {
                self.upload_rgba(queue, frame);
            }
            PixelFormat::Rgb24 => {
                self.upload_rgb24(queue, frame);
            }
        }
    }

    fn upload_yuv420p(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        // Upload Y plane
        if let Some(y_plane) = frame.plane(0) {
            let (bytes_per_row, data) = pad_plane_data(&y_plane.data, y_plane.stride, frame.height);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Use ceiling division for chroma dimensions to match texture size (handles odd dimensions)
        let uv_width = frame.width.div_ceil(2);
        let uv_height = frame.height.div_ceil(2);

        // Upload U plane
        if let Some(u_plane) = frame.plane(1) {
            let (bytes_per_row, data) = pad_plane_data(&u_plane.data, u_plane.stride, uv_height);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.u_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(uv_height),
                },
                wgpu::Extent3d {
                    width: uv_width,
                    height: uv_height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Upload V plane
        if let Some(v_plane) = frame.plane(2) {
            let (bytes_per_row, data) = pad_plane_data(&v_plane.data, v_plane.stride, uv_height);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.v_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(uv_height),
                },
                wgpu::Extent3d {
                    width: uv_width,
                    height: uv_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn upload_nv12(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        // Upload Y plane
        if let Some(y_plane) = frame.plane(0) {
            let (bytes_per_row, data) = pad_plane_data(&y_plane.data, y_plane.stride, frame.height);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Use ceiling division for chroma dimensions to match texture size (handles odd dimensions)
        let uv_width = frame.width.div_ceil(2);
        let uv_height = frame.height.div_ceil(2);

        // Upload interleaved UV plane
        if let Some(uv_plane) = frame.plane(1) {
            let (bytes_per_row, data) = pad_plane_data(&uv_plane.data, uv_plane.stride, uv_height);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.u_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(uv_height),
                },
                wgpu::Extent3d {
                    width: uv_width,
                    height: uv_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn upload_rgba(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        if let Some(plane) = frame.plane(0) {
            let (bytes_per_row, data) = pad_plane_data(&plane.data, plane.stride, frame.height);

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn upload_rgb24(&self, queue: &wgpu::Queue, frame: &CpuFrame) {
        // RGB24 needs to be converted to RGBA with proper alignment
        if let Some(plane) = frame.plane(0) {
            let rgba_stride = frame.width * 4;
            let aligned_stride = align_up(rgba_stride, WGPU_COPY_BYTES_PER_ROW_ALIGNMENT);
            let padding = (aligned_stride - rgba_stride) as usize;

            let mut rgba_data = Vec::with_capacity((aligned_stride * frame.height) as usize);
            let mut truncated = false;
            for y in 0..frame.height as usize {
                for x in 0..frame.width as usize {
                    let offset = y * plane.stride + x * 3;
                    if offset + 2 < plane.data.len() {
                        rgba_data.push(plane.data[offset]); // R
                        rgba_data.push(plane.data[offset + 1]); // G
                        rgba_data.push(plane.data[offset + 2]); // B
                        rgba_data.push(255); // A
                    } else {
                        // Frame data truncated - fill with black
                        rgba_data.extend_from_slice(&[0, 0, 0, 255]);
                        truncated = true;
                    }
                }
                // Add padding bytes for alignment
                rgba_data.resize(rgba_data.len() + padding, 0);
            }
            if truncated {
                tracing::warn!(
                    "RGB24 frame data truncated: expected {}x{} with stride {}, got {} bytes",
                    frame.width,
                    frame.height,
                    plane.stride,
                    plane.data.len()
                );
            }

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.y_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_stride),
                    rows_per_image: Some(frame.height),
                },
                wgpu::Extent3d {
                    width: frame.width,
                    height: frame.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Returns the video dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Returns the pixel format.
    pub fn format(&self) -> PixelFormat {
        self.format
    }
}

/// Data passed to the video render callback.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VideoRenderData {
    /// Transform: [scale_x, scale_y, offset_x, offset_y]
    pub transform: [f32; 4],
}

/// Callback for rendering video frames via egui's paint callback system.
/// Only available on native platforms (not wasm32) as it depends on video_player.
#[cfg(not(target_arch = "wasm32"))]
mod native_render_callback {
    use super::super::triple_buffer::TripleBufferReader;
    use super::super::video_player::PendingFrame;
    use super::*;

    /// Callback for rendering video frames via egui's paint callback system.
    ///
    /// This struct implements [`egui_wgpu::CallbackTrait`] to handle GPU texture
    /// uploads and rendering within egui's wgpu callback infrastructure.
    ///
    /// # Thread Safety
    /// Uses `Arc<Mutex<>>` for the texture and `TripleBufferReader` for lock-free
    /// frame reads from the render thread.
    pub struct VideoRenderCallback {
        /// The video texture to render
        pub texture: std::sync::Arc<parking_lot::Mutex<Option<VideoTexture>>>,
        /// Triple buffer reader for pending frame (lock-free reads from render thread)
        pub pending_frame_reader: TripleBufferReader<PendingFrame>,
        /// The pixel format of the current frame
        pub format: PixelFormat,
        /// The destination rectangle in clip space
        pub rect: Rect,
        /// Rate-limit "no CPU fallback" warning (log once per player instance, persists across frames)
        #[cfg(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "android",
            all(target_os = "windows", feature = "windows-native-video")
        ))]
        pub fallback_logged: Arc<AtomicBool>,
        /// Android player ID for multi-player frame isolation
        #[cfg(target_os = "android")]
        pub player_id: u64,
    }

    impl egui_wgpu::CallbackTrait for VideoRenderCallback {
        fn prepare(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            _screen_descriptor: &egui_wgpu::ScreenDescriptor,
            _egui_encoder: &mut wgpu::CommandEncoder,
            resources: &mut egui_wgpu::CallbackResources,
        ) -> Vec<wgpu::CommandBuffer> {
            let Some(video_resources): Option<&VideoRenderResources> = resources.get() else {
                tracing::warn!("VideoRenderResources not registered, skipping video render");
                return Vec::new();
            };

            // Handle pending frame: create texture if needed and upload data (lock-free read)
            if let Some(pending) = self.pending_frame_reader.read() {
                // NOTE: Zero-copy import blocks the render thread (lumina-video-d70)
                //
                // Current behavior: Import happens synchronously here in prepare().
                // This typically takes <1ms but can cause frame hitches under load.
                //
                // Future improvement: Move import to a background thread with poll_promise,
                // passing pre-imported Arc<wgpu::Texture> through the triple buffer.
                // This requires careful lifetime management since GPU surfaces are
                // only valid for one frame.
                //
                // For now, this is acceptable for most use cases. Latency-sensitive
                // applications (VR/AR) may need the async import optimization.

                // Zero-copy path: import macOS IOSurface directly as wgpu texture
                #[cfg(target_os = "macos")]
                if let Some(ref surface) = pending.macos_surface {
                    let mut texture_guard = self.texture.lock();
                    let mut imported_ok = false;

                    // For zero-copy, we need to recreate the texture each frame because
                    // the IOSurface changes with each decoded frame
                    match unsafe {
                        zero_copy::macos::import_iosurface(
                            device,
                            surface.io_surface,
                            surface.width,
                            surface.height,
                            wgpu::TextureFormat::Bgra8Unorm,
                        )
                    } {
                        Ok(imported_texture) => {
                            // Create a VideoTexture wrapper for the imported texture
                            // This creates the bind group needed for rendering
                            let video_texture = VideoTexture::from_imported_rgba(
                                device,
                                video_resources,
                                imported_texture,
                                surface.width,
                                surface.height,
                                surface.format,
                            );
                            *texture_guard = Some(video_texture);
                            imported_ok = true;
                            tracing::trace!(
                                "Zero-copy: imported IOSurface as wgpu texture {}x{}",
                                surface.width,
                                surface.height
                            );
                        }
                        Err(zero_copy::ZeroCopyError::DeviceBusy) => {
                            // GPU device is busy (e.g., macOS screenshot in progress)
                            // Keep the existing texture and skip this frame
                            tracing::debug!("Zero-copy: device busy, keeping previous frame");
                            if texture_guard.is_some() {
                                imported_ok = true; // Use cached texture
                            }
                            // If no cached texture, fall through to CPU path
                        }
                        Err(e) => {
                            tracing::warn!("Zero-copy import failed, will use CPU fallback: {}", e);
                            // Fall through to CPU path if there's a CPU frame available
                        }
                    }

                    // If we successfully imported this frame, skip the CPU path
                    if imported_ok {
                        // Update transform and return
                        let transform = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
                        queue.write_buffer(
                            &video_resources.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&transform),
                        );
                        return Vec::new();
                    }
                }

                // Zero-copy path: import Windows D3D11 shared handle as wgpu texture
                #[cfg(all(target_os = "windows", feature = "windows-native-video"))]
                if let Some(ref surface) = pending.windows_surface {
                    let mut texture_guard = self.texture.lock();
                    let mut imported_ok = false;

                    // For zero-copy, we need to recreate the texture each frame because
                    // the shared handle changes with each decoded frame
                    match unsafe {
                        zero_copy_windows::windows::import_d3d11_shared_handle(
                            device,
                            surface.shared_handle,
                            surface.width,
                            surface.height,
                            wgpu::TextureFormat::Bgra8Unorm,
                        )
                    } {
                        Ok(imported_texture) => {
                            // Create a VideoTexture wrapper for the imported texture
                            // This creates the bind group needed for rendering
                            let video_texture = VideoTexture::from_imported_rgba(
                                device,
                                video_resources,
                                imported_texture,
                                surface.width,
                                surface.height,
                                surface.format,
                            );
                            *texture_guard = Some(video_texture);
                            imported_ok = true;
                            tracing::trace!(
                                "Zero-copy: imported D3D11 shared handle as wgpu texture {}x{}",
                                surface.width,
                                surface.height
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Windows zero-copy import failed, will use CPU fallback: {}",
                                e
                            );
                            // Fall through to CPU path if there's a CPU frame available
                        }
                    }

                    // If we successfully imported, skip the CPU path
                    if imported_ok {
                        // Update transform and return
                        let transform = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
                        queue.write_buffer(
                            &video_resources.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&transform),
                        );
                        return Vec::new();
                    }
                }

                // Zero-copy path: check for HardwareBuffer from ExoPlayerBridge queue
                // This is the primary zero-copy path for Android using the Kotlin bridge
                //
                // ## Known Limitation: ImageFormat.PRIVATE yields YUV, not RGBA
                //
                // ExoPlayer's ImageReader with ImageFormat.PRIVATE typically provides YUV data
                // (usually NV12), NOT RGBA. The HardwareBuffer format will be something like
                // AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420 rather than R8G8B8A8_UNORM.
                //
                // The format check below correctly rejects non-RGBA buffers, but this means
                // zero-copy will almost always fall back to the CPU path on Android. This is
                // expected behavior given current implementation constraints.
                //
                // Future work: Implement YUV multi-plane import using VkSamplerYcbcrConversion
                // to handle NV12/YUV420p HardwareBuffers directly on the GPU. This would require:
                // 1. Detecting YUV format from HardwareBuffer
                // 2. Creating Vulkan YCbCr sampler with appropriate format
                // 3. Importing each plane as a separate texture
                // 4. Using the existing YUV shader pipeline for conversion
                //
                // See: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSamplerYcbcrConversion.html
                //
                // NOTE: Android timing limitation (lumina-video-m07)
                //
                // This path bypasses FrameScheduler and displays frames immediately upon receipt.
                // Frame timestamps (timestamp_ns) are ignored, which can cause timing drift or
                // display of stale frames under load. This is acceptable for most use cases.
                //
                // Full fix would require passing (playback_start_time, playback_start_position)
                // through the triple buffer and checking: frame_pts <= current_position.
                #[cfg(target_os = "android")]
                {
                    tracing::debug!(
                        "prepare: checking HardwareBuffer queue for player_id={}",
                        self.player_id
                    );
                }
                #[cfg(target_os = "android")]
                if let Some(mut frame) =
                    crate::media::android_video::try_receive_hardware_buffer_for_player(
                        self.player_id,
                    )
                {
                    use crate::media::android_video::{
                        is_yuv_candidate_hardware_buffer_format, is_yv12_format,
                        AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
                    };

                    tracing::info!(
                        "Received HardwareBuffer frame: {}x{} format=0x{:x}",
                        frame.width,
                        frame.height,
                        frame.format
                    );

                    let mut texture_guard = self.texture.lock();
                    let mut imported_ok = false;

                    // Capture dimensions before frame is moved into _android_owner
                    let (frame_width, frame_height) = (frame.width, frame.height);

                    // Take ownership of the fence FD so we can pass it to Vulkan.
                    // After this, the frame's Drop won't close the FD since it's -1.
                    // The import function will either:
                    // - Import it into Vulkan (which takes ownership with TEMPORARY flag), or
                    // - Close it on error
                    let fence_fd = std::mem::replace(&mut frame.fence_fd, -1);

                    if frame.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM {
                        // RGBA path - direct single-plane import
                        // Check fence_fd (if valid) with non-blocking poll before reading buffer
                        // This ensures the producer (MediaCodec) has finished writing
                        let fence_ready = if fence_fd >= 0 {
                            // Non-blocking poll (timeout=0) to check if fence is signaled
                            // poll() returns >0 on success (ready), 0 on not ready, -1 on error
                            let mut pfd = libc::pollfd {
                                fd: fence_fd,
                                events: libc::POLLIN as i16,
                                revents: 0,
                            };
                            let ret = unsafe { libc::poll(&mut pfd, 1, 0) };
                            unsafe {
                                libc::close(fence_fd);
                            }
                            if ret <= 0 {
                                tracing::debug!(
                                    "RGBA fence not ready, dropping frame to avoid blocking"
                                );
                                false
                            } else {
                                true
                            }
                        } else {
                            // No fence means already signaled
                            true
                        };

                        // Only proceed with zero-copy import if fence is ready
                        // If fence isn't ready, skip import to avoid reading incomplete data
                        if fence_ready {
                            match unsafe {
                                zero_copy_android::android::import_ahardwarebuffer(
                                    device,
                                    frame.buffer,
                                    frame_width,
                                    frame_height,
                                    wgpu::TextureFormat::Rgba8Unorm,
                                )
                            } {
                                Ok(imported_texture) => {
                                    let mut video_texture = VideoTexture::from_imported_rgba(
                                        device,
                                        video_resources,
                                        imported_texture,
                                        frame_width,
                                        frame_height,
                                        PixelFormat::Rgba,
                                    );
                                    video_texture._android_owner = Some(frame);
                                    *texture_guard = Some(video_texture);
                                    imported_ok = true;
                                    tracing::trace!(
                                        "Zero-copy: imported RGBA HardwareBuffer {}x{}",
                                        frame_width,
                                        frame_height
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!("Android RGBA zero-copy import failed: {}", e);
                                }
                            }
                        }
                    } else if is_yuv_candidate_hardware_buffer_format(frame.format) {
                        // YUV path - try true zero-copy first, fall back to CPU-assisted path
                        //
                        // True zero-copy: Uses raw Vulkan to import YUV planes and do GPU-side
                        // YUV→RGB conversion without any CPU memory access (~186 MB/s saved at 1080p60).
                        //
                        // CPU-assisted: Uses AHardwareBuffer_lockPlanes to read plane data, copies
                        // to wgpu textures. Fallback when raw Vulkan path fails (driver bugs, etc.)
                        match unsafe {
                            zero_copy_android::android::import_ahardwarebuffer_yuv_zero_copy(
                                device,
                                frame.buffer,
                                frame_width,
                                frame_height,
                                None, // TODO: Extract color space from decoder metadata
                                fence_fd,
                            )
                        } {
                            Ok(rgba_texture) => {
                                // True zero-copy succeeded - got a single RGBA texture
                                let mut video_texture = VideoTexture::from_imported_rgba(
                                    device,
                                    video_resources,
                                    rgba_texture,
                                    frame_width,
                                    frame_height,
                                    PixelFormat::Rgba,
                                );
                                video_texture._android_owner = Some(frame);
                                *texture_guard = Some(video_texture);
                                imported_ok = true;
                                tracing::trace!(
                                "True zero-copy: imported YUV HardwareBuffer via Vulkan YUV pipeline {}x{}",
                                frame_width,
                                frame_height
                            );
                            }
                            Err(zero_copy_err) => {
                                // True zero-copy failed, try CPU-assisted fallback
                                tracing::info!(
                                    "True zero-copy failed ({}), trying CPU-assisted path",
                                    zero_copy_err
                                );

                                // Determine the correct pixel format based on HardwareBuffer format
                                // YV12 returns 3 planes [Y, U, V] -> use Yuv420p shader
                                // NV12 returns 2 planes [Y, UV] -> use Nv12 shader
                                let pixel_format = if is_yv12_format(frame.format) {
                                    PixelFormat::Yuv420p
                                } else {
                                    PixelFormat::Nv12
                                };

                                match unsafe {
                                    zero_copy_android::android::import_ahardwarebuffer_yuv(
                                        device,
                                        queue,
                                        frame.buffer,
                                        frame_width,
                                        frame_height,
                                        frame.format,
                                    )
                                } {
                                    Ok(plane_textures) => {
                                        match VideoTexture::from_imported_yuv_planes(
                                            device,
                                            video_resources,
                                            plane_textures,
                                            frame_width,
                                            frame_height,
                                            pixel_format,
                                        ) {
                                            Ok(mut video_texture) => {
                                                video_texture._android_owner = Some(frame);
                                                *texture_guard = Some(video_texture);
                                                imported_ok = true;
                                                tracing::info!(
                                                "CPU-assisted: imported YUV HardwareBuffer as multi-plane {}x{} format={:?}",
                                                frame_width,
                                                frame_height,
                                                pixel_format
                                            );
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                "Failed to create VideoTexture from YUV planes: {}",
                                                e
                                            );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                        "Android YUV import failed (format {}): {}. Using CPU fallback.",
                                        frame.format, e
                                    );
                                    }
                                }
                            }
                        }
                    } else {
                        // Close fence_fd since we're not using it
                        if fence_fd >= 0 {
                            extern "C" {
                                fn close(fd: i32) -> i32;
                            }
                            unsafe {
                                close(fence_fd);
                            }
                        }
                        tracing::warn!(
                        "Android zero-copy: unsupported HardwareBuffer format {}. Using CPU fallback.",
                        frame.format
                    );
                    }

                    if imported_ok {
                        tracing::info!(
                            "Android frame import successful, texture ready for rendering"
                        );
                        let transform = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
                        queue.write_buffer(
                            &video_resources.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&transform),
                        );
                        return Vec::new();
                    } else {
                        tracing::warn!("Android frame import failed, frame will be dropped");
                    }
                    // frame is dropped here if not imported, calling AHardwareBuffer_release
                }

                // Fallback: import Android AHardwareBuffer from decode path (legacy)
                #[cfg(target_os = "android")]
                if let Some(ref surface) = pending.android_surface {
                    let mut texture_guard = self.texture.lock();
                    let mut imported_ok = false;

                    // For zero-copy, we need to recreate the texture each frame because
                    // the AHardwareBuffer changes with each decoded frame
                    match unsafe {
                        zero_copy_android::android::import_ahardwarebuffer(
                            device,
                            surface.ahardware_buffer,
                            surface.width,
                            surface.height,
                            wgpu::TextureFormat::Rgba8Unorm, // Android typically uses RGBA
                        )
                    } {
                        Ok(imported_texture) => {
                            // Create a VideoTexture wrapper for the imported texture
                            // This creates the bind group needed for rendering
                            let video_texture = VideoTexture::from_imported_rgba(
                                device,
                                video_resources,
                                imported_texture,
                                surface.width,
                                surface.height,
                                surface.format,
                            );
                            *texture_guard = Some(video_texture);
                            imported_ok = true;
                            tracing::trace!(
                                "Zero-copy: imported AHardwareBuffer as wgpu texture {}x{}",
                                surface.width,
                                surface.height
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Android zero-copy import failed, will use CPU fallback: {}",
                                e
                            );
                            // Fall through to CPU path if there's a CPU frame available
                        }
                    }

                    // If we successfully imported, skip the CPU path
                    if imported_ok {
                        // Update transform and return
                        let transform = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
                        queue.write_buffer(
                            &video_resources.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&transform),
                        );
                        return Vec::new();
                    }
                }

                // Zero-copy path: import Linux DMABuf directly as wgpu texture
                // Supports both single-plane (RGBA/BGRA) and multi-plane (NV12/YUV420p) formats
                #[cfg(target_os = "linux")]
                if let Some(ref surface) = pending.linux_surface {
                    let mut texture_guard = self.texture.lock();
                    let mut imported_ok = false;

                    // Handle different formats with appropriate import paths
                    match surface.format {
                        // Single-plane formats: RGBA, BGRA
                        PixelFormat::Rgba | PixelFormat::Bgra => {
                            let wgpu_format = if surface.format == PixelFormat::Rgba {
                                wgpu::TextureFormat::Rgba8Unorm
                            } else {
                                wgpu::TextureFormat::Bgra8Unorm
                            };

                            let dmabuf_handle = zero_copy_linux::linux::DmaBufHandle::single_plane(
                                surface.primary_fd(),
                                surface.primary_size(),
                                surface.primary_offset(),
                                surface.primary_stride(),
                                surface.modifier,
                            );

                            match unsafe {
                                zero_copy_linux::linux::import_dmabuf(
                                    device,
                                    dmabuf_handle,
                                    surface.width,
                                    surface.height,
                                    wgpu_format,
                                )
                            } {
                                Ok(imported_texture) => {
                                    let video_texture = VideoTexture::from_imported_rgba(
                                        device,
                                        video_resources,
                                        imported_texture,
                                        surface.width,
                                        surface.height,
                                        surface.format,
                                    );
                                    *texture_guard = Some(video_texture);
                                    imported_ok = true;
                                    tracing::trace!(
                                    "Zero-copy: imported DMABuf fd={} as RGBA/BGRA texture {}x{}",
                                    surface.primary_fd(),
                                    surface.width,
                                    surface.height
                                );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                    "Linux zero-copy import failed for {:?}, will use CPU fallback: {}",
                                    surface.format,
                                    e
                                );
                                }
                            }
                        }

                        // Multi-plane formats: NV12, YUV420p
                        PixelFormat::Nv12 | PixelFormat::Yuv420p => {
                            // Convert LinuxGpuSurface planes to DmaBufPlaneHandle
                            let plane_handles: Vec<_> = surface
                                .planes
                                .iter()
                                .map(|p| zero_copy_linux::linux::DmaBufPlaneHandle {
                                    fd: p.fd,
                                    offset: p.offset,
                                    stride: p.stride,
                                    size: p.size,
                                })
                                .collect();

                            // Choose import path based on single-FD vs multi-FD layout
                            if surface.is_single_fd {
                                // Single-FD multi-plane: all planes share one FD with different offsets
                                // Common with VA-API output
                                //
                                // NOTE: The spec-correct disjoint NV12 import (VK_FORMAT_G8_B8R8_2PLANE_420_UNORM
                                // with VK_IMAGE_CREATE_DISJOINT_BIT) is not supported by Intel ANV driver
                                // for external DMABuf memory. Validation error:
                                //   "Format VK_FORMAT_G8_B8R8_2PLANE_420_UNORM is not supported for this
                                //    combination of parameters"
                                //
                                // Falling back to separate-plane import (R8 for Y, RG8 for UV).
                                // This may have color corruption on Intel ANV due to UV plane offset handling.

                                // Normalize modifier: INVALID -> LINEAR for VA-API linear buffers
                                let effective_modifier = if surface.modifier
                                    == zero_copy_linux::linux::drm_modifiers::DRM_FORMAT_MOD_INVALID
                                {
                                    zero_copy_linux::linux::drm_modifiers::DRM_FORMAT_MOD_LINEAR
                                } else {
                                    surface.modifier
                                };

                                tracing::debug!(
                                    "Importing single-FD {:?} as separate plane textures (fd={}, modifier={:#x})",
                                    surface.format,
                                    surface.primary_fd(),
                                    effective_modifier
                                );
                                match unsafe {
                                    zero_copy_linux::linux::import_dmabuf_single_fd_multi_plane(
                                        device,
                                        surface.primary_fd(),
                                        &plane_handles,
                                        surface.width,
                                        surface.height,
                                        surface.format,
                                        effective_modifier,
                                    )
                                } {
                                    Ok(plane_textures) => {
                                        match VideoTexture::from_imported_yuv_planes(
                                            device,
                                            video_resources,
                                            plane_textures,
                                            surface.width,
                                            surface.height,
                                            surface.format,
                                        ) {
                                            Ok(video_texture) => {
                                                *texture_guard = Some(video_texture);
                                                imported_ok = true;
                                                tracing::trace!(
                                                    "Zero-copy: imported single-FD DMABuf {} planes as YUV texture {}x{}",
                                                    surface.num_planes(),
                                                    surface.width,
                                                    surface.height
                                                );
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    "Linux zero-copy YUV plane binding failed for single-FD {:?}, will use CPU fallback: {}",
                                                    surface.format,
                                                    e
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Linux zero-copy single-FD import failed for {:?}, will use CPU fallback: {}",
                                            surface.format,
                                            e
                                        );
                                    }
                                }
                            } else {
                                // Multi-FD: each plane has its own FD
                                // Normalize modifier: INVALID -> LINEAR (same as single-FD branch)
                                let effective_modifier = if surface.modifier
                                    == zero_copy_linux::linux::drm_modifiers::DRM_FORMAT_MOD_INVALID
                                {
                                    zero_copy_linux::linux::drm_modifiers::DRM_FORMAT_MOD_LINEAR
                                } else {
                                    surface.modifier
                                };

                                let dmabuf_handle = zero_copy_linux::linux::DmaBufHandle::new(
                                    plane_handles,
                                    effective_modifier,
                                );
                                match unsafe {
                                    zero_copy_linux::linux::import_dmabuf_multi_plane(
                                        device,
                                        dmabuf_handle,
                                        surface.width,
                                        surface.height,
                                        surface.format,
                                    )
                                } {
                                    Ok(plane_textures) => {
                                        match VideoTexture::from_imported_yuv_planes(
                                            device,
                                            video_resources,
                                            plane_textures,
                                            surface.width,
                                            surface.height,
                                            surface.format,
                                        ) {
                                            Ok(video_texture) => {
                                                *texture_guard = Some(video_texture);
                                                imported_ok = true;
                                                tracing::trace!(
                                                    "Zero-copy: imported multi-FD DMABuf {} planes as YUV texture {}x{}",
                                                    surface.num_planes(),
                                                    surface.width,
                                                    surface.height
                                                );
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    "Linux zero-copy YUV plane binding failed for {:?}, will use CPU fallback: {}",
                                                    surface.format,
                                                    e
                                                );
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Linux zero-copy multi-FD import failed for {:?}, will use CPU fallback: {}",
                                            surface.format,
                                            e
                                        );
                                    }
                                }
                            }
                        }

                        // Unsupported format: RGB24 (3-byte pixels not supported)
                        PixelFormat::Rgb24 => {
                            tracing::debug!(
                            "Linux zero-copy: skipping RGB24 format (3-byte pixels not supported), using CPU path"
                        );
                        }
                    }

                    // If we successfully imported, skip the CPU path
                    if imported_ok {
                        // Update transform and return
                        let transform = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
                        queue.write_buffer(
                            &video_resources.uniform_buffer,
                            0,
                            bytemuck::bytes_of(&transform),
                        );
                        return Vec::new();
                    }
                }

                // CPU fallback path
                if let Some(ref cpu_frame) = pending.frame {
                    let mut texture_guard = self.texture.lock();

                    // On Android with zero-copy, skip CPU fallback if we have a valid texture
                    // with more pixels than the incoming frame. This prevents 1x1 placeholder frames
                    // from decode_next_frame() from overwriting good zero-copy HardwareBuffer frames.
                    // Using pixel count comparison handles wide/tall aspect ratio mismatches safely.
                    #[cfg(target_os = "android")]
                    {
                        if let Some(ref existing) = *texture_guard {
                            let existing_pixels = existing.width as u64 * existing.height as u64;
                            let incoming_pixels = cpu_frame.width as u64 * cpu_frame.height as u64;
                            // If existing texture has more pixels, it's likely a real video frame
                            // and the incoming frame is a placeholder. Skip upload.
                            if existing_pixels >= incoming_pixels && incoming_pixels < 16 {
                                tracing::trace!(
                                    "CPU fallback: skipping {}x{} ({} px) frame, keeping existing {}x{} ({} px) zero-copy texture",
                                    cpu_frame.width, cpu_frame.height, incoming_pixels,
                                    existing.width, existing.height, existing_pixels
                                );
                                return Vec::new();
                            }
                        }
                    }

                    // Check if texture format matches expected format from GPU surface
                    // This prevents first-frame mismatch when GPU surface format differs
                    let format_mismatch = pending.pixel_format.is_some_and(|expected_format| {
                        texture_guard
                            .as_ref()
                            .is_some_and(|tex| tex.format() != expected_format)
                    });

                    // Create texture if needed (size change, format mismatch, or first frame)
                    if pending.needs_recreate || format_mismatch || texture_guard.is_none() {
                        let new_texture = VideoTexture::new(
                            device,
                            video_resources,
                            cpu_frame.width,
                            cpu_frame.height,
                            cpu_frame.format,
                        );
                        *texture_guard = Some(new_texture);
                    }

                    // Upload frame data
                    if let Some(ref texture) = *texture_guard {
                        texture.upload(queue, cpu_frame);
                    }
                } else {
                    // Check if we had a GPU surface but no CPU fallback
                    // This can happen when zero-copy import fails and decoder didn't provide CPU frame
                    #[cfg(any(
                        target_os = "macos",
                        target_os = "linux",
                        target_os = "android",
                        all(target_os = "windows", feature = "windows-native-video")
                    ))]
                    {
                        #[cfg(target_os = "macos")]
                        let has_gpu_surface = pending.macos_surface.is_some();
                        #[cfg(target_os = "windows")]
                        let has_gpu_surface = pending.windows_surface.is_some();
                        #[cfg(target_os = "android")]
                        let has_gpu_surface = pending.android_surface.is_some();
                        #[cfg(target_os = "linux")]
                        let has_gpu_surface = pending.linux_surface.is_some();
                        #[cfg(not(any(
                            target_os = "macos",
                            target_os = "windows",
                            target_os = "android",
                            target_os = "linux"
                        )))]
                        let has_gpu_surface = false;

                        if has_gpu_surface && !self.fallback_logged.swap(true, Ordering::Relaxed) {
                            tracing::warn!(
                                "Zero-copy import failed with no CPU fallback available. \
                             Frame dropped. Check GPU driver compatibility if this persists."
                            );
                        }
                    }
                }
            }

            // egui-wgpu paint callbacks render within the clip rect that was specified.
            // We just need to draw a fullscreen quad that fills the clip rect.
            // Use identity transform (scale=1, offset=0) to fill the entire callback area.
            let transform = [1.0f32, 1.0f32, 0.0f32, 0.0f32];

            queue.write_buffer(
                &video_resources.uniform_buffer,
                0,
                bytemuck::bytes_of(&transform),
            );

            Vec::new()
        }

        fn paint(
            &self,
            info: egui::PaintCallbackInfo,
            render_pass: &mut wgpu::RenderPass<'static>,
            resources: &egui_wgpu::CallbackResources,
        ) {
            let Some(video_resources): Option<&VideoRenderResources> = resources.get() else {
                tracing::warn!("VideoRenderResources not registered, skipping video paint");
                return;
            };

            // Set viewport to match our callback rect
            // This maps the -1..1 NDC quad to fill the callback rect
            let viewport = info.viewport_in_pixels();
            render_pass.set_viewport(
                viewport.left_px as f32,
                viewport.top_px as f32,
                viewport.width_px as f32,
                viewport.height_px as f32,
                0.0,
                1.0,
            );

            // Set scissor rect to clip to our area
            let clip = info.clip_rect_in_pixels();
            render_pass.set_scissor_rect(
                clip.left_px.max(0) as u32,
                clip.top_px.max(0) as u32,
                clip.width_px.max(0) as u32,
                clip.height_px.max(0) as u32,
            );

            // Select pipeline based on format
            let pipeline = match self.format {
                PixelFormat::Yuv420p => &video_resources.pipeline_yuv420p,
                PixelFormat::Nv12 => &video_resources.pipeline_nv12,
                PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgra => {
                    &video_resources.pipeline_rgb
                }
            };

            // Get the video texture
            let texture_guard = self.texture.lock();
            if let Some(ref texture) = *texture_guard {
                tracing::debug!(
                    "paint: rendering {}x{} texture with format {:?}",
                    texture.width,
                    texture.height,
                    texture.format
                );
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, &texture.bind_group, &[]);
                render_pass.draw(0..6, 0..1); // Draw fullscreen quad
            } else {
                tracing::debug!("paint: texture is None, skipping render");
            }
        }
    }
} // end native_render_callback module

#[cfg(not(target_arch = "wasm32"))]
pub use native_render_callback::VideoRenderCallback;
