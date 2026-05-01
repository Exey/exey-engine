#version 450
//
// sprite.vert — M2 vertex shader.
//
// Input vertex layout matches `draw::Vertex2D` (engine-side struct):
//   location 0 — vec2 position in pixels (top-left origin, +Y down)
//   location 1 — vec4 vertex color (modulates the texture sample, supplies alpha)
//   location 2 — vec2 texture UV in [0..1]
//
// The push constant encodes a tiny screen→clip transform:
//   ndc.xy = position * scale + offset
// which we use as a stand-in for a real ortho projection until M4 wires up
// `IsometricCamera2D`. With `scale = 2/extent` and `offset = (-1, -1)` we get
// pixel coords with origin at the top-left of the framebuffer.

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

layout(push_constant) uniform PushConstants {
    vec2 scale;
    vec2 offset;
    vec4 tint;
} pc;

void main() {
    vec2 ndc = in_pos * pc.scale + pc.offset;
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color * pc.tint;
    v_uv = in_uv;
}
