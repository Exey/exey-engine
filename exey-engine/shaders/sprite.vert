#version 450
//
// sprite.vert — M3 vertex shader.
//
// Input vertex layout matches `draw::Vertex2D` (engine-side struct):
//   location 0 — vec2 position in unit-quad local space, range [0..1]
//   location 1 — vec4 vertex color (modulates the texture sample, supplies alpha)
//   location 2 — vec2 texture UV in [0..1]
//
// The push constant encodes:
//   - per-frame:  screen_scale, screen_offset    — pixel→clip transform
//   - per-sprite: world_pos, world_size          — top-left and size in pixels
//   - per-sprite: tint                           — multiplied into vertex color
//
// World transform: pixel_pos = local * world_size + world_pos
// Clip transform:  ndc.xy    = pixel_pos * screen_scale + screen_offset
//
// With `screen_scale = 2/extent` and `screen_offset = (-1, -1)` we get pixel
// coords with origin at the top-left of the framebuffer. Both transforms
// stand in for a real ortho projection until M4 wires up `IsometricCamera2D`.
//
// CHANGED IN M3 from M2: vertex `in_pos` is now [0..1] local coords, not
// pixel coords. The push constant gained `world_pos`/`world_size`. This lets
// every sprite share the same vertex/index buffers — only the push constant
// differs per draw.

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

layout(push_constant) uniform PushConstants {
    vec2 screen_scale;
    vec2 screen_offset;
    vec2 world_pos;
    vec2 world_size;
    vec4 tint;
} pc;

void main() {
    vec2 pixel_pos = in_pos * pc.world_size + pc.world_pos;
    vec2 ndc      = pixel_pos * pc.screen_scale + pc.screen_offset;
    gl_Position   = vec4(ndc, 0.0, 1.0);
    v_color       = in_color * pc.tint;
    v_uv          = in_uv;
}
