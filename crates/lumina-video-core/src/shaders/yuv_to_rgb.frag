#version 450
// YUV (NV12) to RGB conversion fragment shader
// Supports configurable color matrix and offsets via uniforms for:
// - BT.709 / BT.601 color space
// - Limited range (16-235) / Full range (0-255) input

layout(set = 0, binding = 0) uniform sampler2D y_tex;
layout(set = 0, binding = 1) uniform sampler2D uv_tex;

// YUV→RGB conversion uniforms
// The CPU/decoder side sets these based on the video's color space and range
layout(set = 0, binding = 2) uniform YuvParams {
    // 3x3 YUV to RGB conversion matrix (column-major, GLSL default)
    // The CPU also uploads this matrix in column-major format
    // For BT.709 limited range:
    //   [ 1.164,  0.000,  1.5748 ]
    //   [ 1.164, -0.1873, -0.4681 ]
    //   [ 1.164,  1.8556,  0.000 ]
    mat3 yuv_to_rgb;

    // Offset to subtract from normalized [0,1] YUV values before matrix multiply
    // For limited range: (16/255, 128/255, 128/255) ≈ (0.0627, 0.5, 0.5)
    // For full range: (0.0, 0.5, 0.5)
    vec3 yuv_offset;
} params;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 color;

void main() {
    // Sample Y plane (full resolution, R8)
    float y = texture(y_tex, v_uv).r;

    // Sample UV plane (half resolution, RG8 interleaved)
    vec2 uv_val = texture(uv_tex, v_uv).rg;

    // Construct YUV vector and apply offset
    vec3 yuv = vec3(y, uv_val.r, uv_val.g) - params.yuv_offset;

    // Apply YUV→RGB conversion matrix
    vec3 rgb = params.yuv_to_rgb * yuv;

    // Clamp to valid range and output
    color = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
