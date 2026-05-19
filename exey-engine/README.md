# exey-engine

The engine. Rust + Vulkan port of `ExeyEngine` by Exey Panteleev (AS3, 2014).

For the algorithm write-ups (BigBuffer, IsometricRectangleSorter), see the
[top-level README](../README.md). This README is just the API surface.

## Public types

```rust
use exey_engine::{
    Engine, EngineConfig, RendererKind, FrameClock,
    Sprite, SpriteMesh, Texture, Vertex2D,
    AnimationState, FrameStrip, LoopMode, // M7
};
```

- [`Engine`](src/core.rs) — the equivalent of `ExeyEngineCore`. Owns Vulkan,
  exposes `draw_frame(&Window, dt: f32, &dyn ICamera2D, &[&SpriteMesh], world: &mut [Sprite], gui: &mut [Sprite])`,
  `on_resize((u32, u32))`. `dt` drives the M7 animation tick; world/gui are
  `&mut` because the tick writes `uv_offset`/`uv_scale` on sprites whose
  `anim` is `Some`.
- [`EngineConfig`](src/core.rs) — start-time options (app name, renderer kind).
- [`RendererKind`](src/render/mod.rs) — `Simple | Batch | BigBuffer`.
  Maps from `--renderer` CLI strings via `RendererKind::from_cli`.
- [`ICamera2D`](src/render/camera/mod.rs) — camera interface. Two
  concrete kinds: `SimpleCamera2D` for screen-space content, `IsometricCamera2D`
  for world-space content. Both share `AbstractCamera2D` state (position,
  zoom, viewport).
- [`iso`](src/render/iso.rs) — logic↔world conversions for the 2:1 iso
  projection. Mirrors AS3 `IsoUtil.spaceToScreen` / `screenToSpace`.
- [`Sprite`](src/render/sprite.rs) — per-sprite CPU state (position, size,
  velocity, tint). Mutate freely between frames.
- [`SpriteMesh`](src/render/sprite.rs) — shared GPU geometry (unit-quad
  vertex/index buffers) plus a single descriptor bound to a texture. One
  per (geometry, texture) pair the engine draws.
- [`Texture`](src/gfx/texture.rs) — owns a `vk::Image` + view + sampler.
  Build via `from_rgba(...)`, `from_png_bytes(...)`, or
  `from_image_file_with_luma_key(...)` (M7 — for PNG/JPEG character sheets
  on a black background; promotes near-black pixels to alpha 0 with a
  small luma ramp to hide JPEG halos).
- [`Vertex2D`](src/draw/vertex.rs) — pos/color/uv vertex. M3+ uses unit-quad
  local coords in `pos`; per-sprite world transform travels through the
  push constant.
- [`FrameStrip`](src/draw/animation.rs) — M7. Atlas metadata + timing
  for one animation: where its frames live in the texture, how many,
  how fast, what loop mode. Built once at scene setup via
  `RenderCore::register_strip(strip) -> u16`. Shared across all sprites
  playing that animation.
- [`AnimationState`](src/draw/animation.rs) — M7. Per-sprite playback
  state (`strip_id`, `time`, `paused`). Lives on `Sprite::anim` as
  `Option<AnimationState>`; `None` means the sprite is static and the
  per-frame animation tick skips it entirely.
- [`LoopMode`](src/draw/animation.rs) — M7. `Loop | Once | PingPong`.
- [`FrameClock`](src/time.rs) — delta time + smoothed FPS.

## Module map vs. the AS3 sources

| AS3 package                                   | Rust module                                              |
|-----------------------------------------------|----------------------------------------------------------|
| `exey.engine.ExeyEngineCore`                  | `core::Engine`                                           |
| `exey.engine.stage3d.*`                       | `gfx::{instance, device, swapchain, frame, buffer, memory, texture}` |
| `exey.engine.stage3d.VertexDataBinary`        | `draw::Vertex2D`                                         |
| `exey.engine.render.RenderCore`               | `render::RenderCore`                                     |
| `exey.engine.render.renderers.IRenderer`      | `render::IRenderer`                                      |
| `exey.engine.render.renderers.SimpleRenderer` | `render::SimpleRenderer` (M3; functional, one draw/sprite) |
| `exey.engine.render.renderers.BatchRenderer`  | `render::BatchRenderer` (M3 stub → M5)                   |
| `exey.engine.render.renderers.BigBufferRenderer` | `render::BigBufferRenderer` (M3 stub → M6) ★          |
| `exey.engine.render.sorting.ISorter`          | `render::sort::ISorter`                                  |
| `exey.engine.render.sorting.IsometricRectangleSorter` | `render::sort::IsometricRectangleSorter` (M5 ✓) ★ |
| `exey.engine.common.graph.Graph` / `GraphTopologicalSorter` | `render::sort::graph` (Kahn's algorithm) |
| `exey.engine.render.camera.AbstractCamera2D`  | `render::camera::AbstractCamera2D`                       |
| `exey.engine.render.camera.ICamera2D`         | `render::camera::ICamera2D`                              |
| `exey.engine.render.camera.SimpleCamera2D`    | `render::camera::SimpleCamera2D`                         |
| `exey.engine.render.camera.IsometricCamera2D` | `render::camera::IsometricCamera2D`                      |
| `exey.moss.utils.IsoUtil`                     | `render::iso` (`logic_to_world`, `world_to_logic`)       |
| `exey.engine.draw.*`                          | `draw::*` (M3+)                                          |
| `exey.engine.draw.animation.*`                | `draw::animation` (M7 ✓ — `FrameStrip` + `AnimationState`) |

The naming preserves the original conventions where it improves discoverability,
and Rust-ifies it where AS3 conventions don't translate (e.g. `IRenderable`
trait → just `Renderable` would be more idiomatic, but I kept the I-prefix
in `IRenderer` and `ISorter` because the strategy-pattern relationship is the
whole point and renaming would obscure the lineage).

## Shaders

GLSL sources live in `shaders/`; the matching SPIR-V blobs are committed
under `shaders/spv/` and `include_bytes!`d at engine load time. After editing
any `*.vert` / `*.frag`, run:

```sh
../tools/compile_shaders.sh
```

This needs `glslang`, `glslangValidator`, or `glslc` on `PATH` — install
`glslang-tools` (Linux apt), `glslang` (Homebrew on macOS), or the LunarG
Vulkan SDK. The script picks whichever is available. The vanilla `cargo
build` does **not** need any of these because the SPIR-V is committed.

## Building from a fresh tree

```sh
cargo build -p exey-engine
```

Requires the Vulkan SDK at runtime, but only `libloading` at build time.
