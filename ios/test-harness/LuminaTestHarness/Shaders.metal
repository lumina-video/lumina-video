#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

/// Fullscreen triangle — generates 3 vertices that cover the screen.
/// No vertex buffer needed; vertex_id 0,1,2 map to a triangle
/// that fully covers the [-1,1] clip space.
vertex VertexOut vertexShader(uint vid [[vertex_id]]) {
    VertexOut out;
    // Oversized triangle covering clip space:
    //   vid 0: (-1, -1)  texCoord (0, 1)
    //   vid 1: ( 3, -1)  texCoord (2, 1)
    //   vid 2: (-1,  3)  texCoord (0, -1)
    float2 pos = float2((vid << 1) & 2, vid & 2);
    out.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    out.texCoord = float2(pos.x, 1.0 - pos.y);
    return out;
}

/// Passthrough fragment — samples the video texture.
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               texture2d<float> tex [[texture(0)]],
                               sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.texCoord);
}
