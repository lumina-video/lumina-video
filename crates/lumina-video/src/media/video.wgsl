// YUV to RGB conversion shader for video playback.
//
// This shader converts YUV420p (planar) or NV12 (semi-planar) video frames
// to RGB for display. The conversion uses the BT.709 color matrix, which is
// standard for HD video content.
//
// For SD content (BT.601), the coefficients would be different, but BT.709
// is the safe default for most modern video.

// Vertex shader output / Fragment shader input
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

// Uniform buffer containing the output rect position
struct VideoUniforms {
    // Transform from clip space to texture space
    // [scale_x, scale_y, offset_x, offset_y]
    transform: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: VideoUniforms;

// YUV texture samplers
// For YUV420p: Y, U, V are separate textures
// For NV12: Y is separate, UV is interleaved in one texture
@group(0) @binding(1) var y_texture: texture_2d<f32>;
@group(0) @binding(2) var u_texture: texture_2d<f32>;
@group(0) @binding(3) var v_texture: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;

// Fullscreen quad vertices (two triangles)
// We generate these in the vertex shader to avoid needing a vertex buffer
var<private> QUAD_POSITIONS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),  // Bottom-left
    vec2<f32>( 1.0, -1.0),  // Bottom-right
    vec2<f32>(-1.0,  1.0),  // Top-left
    vec2<f32>(-1.0,  1.0),  // Top-left
    vec2<f32>( 1.0, -1.0),  // Bottom-right
    vec2<f32>( 1.0,  1.0),  // Top-right
);

var<private> QUAD_TEX_COORDS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0),  // Bottom-left (V flipped for video)
    vec2<f32>(1.0, 1.0),  // Bottom-right
    vec2<f32>(0.0, 0.0),  // Top-left
    vec2<f32>(0.0, 0.0),  // Top-left
    vec2<f32>(1.0, 1.0),  // Bottom-right
    vec2<f32>(1.0, 0.0),  // Top-right
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    let pos = QUAD_POSITIONS[vertex_index];
    let tex = QUAD_TEX_COORDS[vertex_index];

    // Apply transform to position the video in the correct location
    output.position = vec4<f32>(
        pos.x * uniforms.transform.x + uniforms.transform.z,
        pos.y * uniforms.transform.y + uniforms.transform.w,
        0.0,
        1.0
    );

    output.tex_coord = tex;

    return output;
}

// BT.709 YUV to RGB conversion matrix (video range)
//
// Video range (most common for H.264/HEVC):
//   Y: 16-235 (scaled to 0-1)
//   UV: 16-240 (centered at 128)
//
// The Y channel is scaled from [16/255, 235/255] to [0, 1] using:
//   y_scaled = (y - 16/255) * (255/219) = (y - 0.0627) * 1.164
//
// BT.709 conversion (normalized coefficients, U/V shifted by 0.5):
// R = y_scaled + 1.5748 * (V - 0.5)
// G = y_scaled - 0.1873 * (U - 0.5) - 0.4681 * (V - 0.5)
// B = y_scaled + 1.8556 * (U - 0.5)

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample Y, U, V textures
    let y = textureSample(y_texture, tex_sampler, in.tex_coord).r;
    let u = textureSample(u_texture, tex_sampler, in.tex_coord).r;
    let v = textureSample(v_texture, tex_sampler, in.tex_coord).r;

    // Convert to RGB using BT.709 coefficients (video range)
    // Scale Y from [16/255, 235/255] to [0, 1]
    let y_scaled = (y - 0.0627) * 1.164;
    // Shift U and V from [0, 1] to [-0.5, 0.5]
    let u_shifted = u - 0.5;
    let v_shifted = v - 0.5;

    // BT.709 conversion
    let r = y_scaled + 1.5748 * v_shifted;
    let g = y_scaled - 0.1873 * u_shifted - 0.4681 * v_shifted;
    let b = y_scaled + 1.8556 * u_shifted;

    // Clamp to valid range and return
    return vec4<f32>(
        clamp(r, 0.0, 1.0),
        clamp(g, 0.0, 1.0),
        clamp(b, 0.0, 1.0),
        1.0
    );
}

// Alternative fragment shader for NV12 format (Y + interleaved UV)
// In NV12, U and V are packed together: UVUVUV...
// This requires a different texture setup where the UV texture has 2 channels
@fragment
fn fs_main_nv12(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample Y texture (full resolution)
    let y = textureSample(y_texture, tex_sampler, in.tex_coord).r;

    // Sample UV texture (half resolution, 2 channels)
    // u_texture is repurposed as the UV texture for NV12
    let uv = textureSample(u_texture, tex_sampler, in.tex_coord).rg;
    let u = uv.r;
    let v = uv.g;

    // Convert to RGB using BT.709 coefficients (video range)
    // Scale Y from [16/255, 235/255] to [0, 1]
    let y_scaled = (y - 0.0627) * 1.164;
    let u_shifted = u - 0.5;
    let v_shifted = v - 0.5;

    let r = y_scaled + 1.5748 * v_shifted;
    let g = y_scaled - 0.1873 * u_shifted - 0.4681 * v_shifted;
    let b = y_scaled + 1.8556 * u_shifted;

    return vec4<f32>(
        clamp(r, 0.0, 1.0),
        clamp(g, 0.0, 1.0),
        clamp(b, 0.0, 1.0),
        1.0
    );
}

// Simple passthrough for RGB/RGBA textures (no conversion needed)
// This is used when the video decoder outputs RGB directly
@fragment
fn fs_main_rgb(in: VertexOutput) -> @location(0) vec4<f32> {
    // y_texture is repurposed as the RGB texture
    return textureSample(y_texture, tex_sampler, in.tex_coord);
}
