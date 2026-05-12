#version 450
//
// sprite.vert — M5 vertex shader.
//
// Inputs (matches `draw::Vertex2D`):
//   location 0 — vec2 in_pos    unit-quad local position, [0..1]
//   location 1 — vec4 in_color  per-vertex color
//   location 2 — vec2 in_uv     per-vertex texture coordinate, [0..1] of the unit quad
//
// The push constant encodes per-frame view transform + per-sprite world
// transform + per-sprite UV sub-region. The vertex shader composes:
//
//   world_pixel = local * world_size + world_pos
//   ndc.xy      = world_pixel * view_scale + view_offset
//   sample_uv   = in_uv * uv_scale + uv_offset
//
// CHANGED IN M5: added `uv_offset` / `uv_scale` so atlas-based sprites
// (bitmap font glyphs, future sprite atlases) share this pipeline.
// Tiles use uv_offset=(0,0), uv_scale=(1,1) for the full texture.

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

layout(push_constant) uniform PushConstants {
    vec2 view_scale;
    vec2 view_offset;
    vec2 world_pos;
    vec2 world_size;
    vec4 tint;
    vec2 uv_offset;
    vec2 uv_scale;
} pc;

void main() {
    vec2 world_pixel = in_pos * pc.world_size + pc.world_pos;
    vec2 ndc        = world_pixel * pc.view_scale + pc.view_offset;
    gl_Position     = vec4(ndc, 0.0, 1.0);
    v_color         = in_color * pc.tint;
    v_uv            = in_uv * pc.uv_scale + pc.uv_offset;
}
