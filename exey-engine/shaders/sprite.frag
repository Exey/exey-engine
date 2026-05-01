#version 450
//
// sprite.frag — M2 fragment shader.
//
// Single sampled image at descriptor set 0, binding 0. NEAREST filtering is
// chosen on the engine side for crisp pixel-art tiles (matches the AS3
// engine's `MIPNEAREST` setup).
//
// Premultiplied-alpha-friendly: we sample, multiply by the per-vertex color,
// and let the pipeline's blend state do the right thing. M2 uses non-PMA
// straight blending; M3+ may switch to PMA once we wire up `Sprite2D.alpha`.

layout(set = 0, binding = 0) uniform sampler2D u_tex;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = texture(u_tex, v_uv) * v_color;
}
