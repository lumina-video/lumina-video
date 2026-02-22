#version 450
// YCbCr blit fragment shader
// Samples from a combined YCbCr sampler (hardware does YUV->RGB conversion)
// and outputs RGBA. This is true zero-copy - no manual color matrix needed.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// Combined image sampler with YCbCr conversion
// The sampler is immutable and configured with VkSamplerYcbcrConversion
layout(set = 0, binding = 0) uniform sampler2D u_ycbcr_texture;

void main() {
    // Sample the YCbCr texture - hardware converts to RGB during the fetch
    vec3 rgb = texture(u_ycbcr_texture, v_uv).rgb;

    // Output with full alpha
    o_color = vec4(rgb, 1.0);
}
