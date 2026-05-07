#version 450
//
// sprite.vert — M4 vertex shader.
//
// Input vertex layout matches `draw::Vertex2D` (engine-side struct):
//   location 0 — vec2 position in unit-quad local space, range [0..1]
//   location 1 — vec4 vertex color (modulates the texture sample, supplies alpha)
//   location 2 — vec2 texture UV in [0..1]
//
// The push constant encodes:
//   - per-frame:  view_scale, view_offset    — world→clip transform from camera
//   - per-sprite: world_pos, world_size      — top-left and size in world pixels
//   - per-sprite: tint                        — multiplied into vertex color
//
// World transform: world_pixel = local * world_size + world_pos
// Camera transform: ndc.xy     = world_pixel * view_scale + view_offset
//
// `view_scale` and `view_offset` come from the camera's view_transform()
// (see exey-engine/src/render/camera). For a default camera at pos=0
// zoom=1 with a 1280×720 viewport, this produces:
//   view_scale = (2/1280, 2/720)    →  world pixels scaled to NDC range
//   view_offset = (0, 0)             →  world origin at NDC origin (screen centre)
// Panning the camera shifts view_offset; zooming scales view_scale.
//
// CHANGED IN M4 from M3: per-frame fields renamed `screen_*` → `view_*`
// to reflect that they now encode the camera's full view transform
// (pan + zoom + ortho), not just a fixed pixel→clip mapping. The byte
// layout is identical — same `vec2 vec2 vec2 vec2 vec4` totalling 48
// bytes, same offsets — so this is a semantic rename only.

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
} pc;

void main() {
    vec2 world_pixel = in_pos * pc.world_size + pc.world_pos;
    vec2 ndc        = world_pixel * pc.view_scale + pc.view_offset;
    gl_Position     = vec4(ndc, 0.0, 1.0);
    v_color         = in_color * pc.tint;
    v_uv            = in_uv;
}
