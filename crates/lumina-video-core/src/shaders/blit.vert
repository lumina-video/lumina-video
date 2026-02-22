#version 450
// Fullscreen triangle vertex shader for blit operations
// Generates a triangle that covers the entire screen from vertex ID

layout(location = 0) out vec2 v_uv;

void main() {
    // Generate UV coordinates from vertex index
    // Vertex 0: (0, 0), Vertex 1: (2, 0), Vertex 2: (0, 2)
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

    // Convert UV [0,2] to NDC [-1,3] for fullscreen triangle
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
