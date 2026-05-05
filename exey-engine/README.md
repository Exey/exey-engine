# exey-engine

The engine. Rust + Vulkan port of `ExeyEngine` by Exey Panteleev (AS3, 2014).

For the algorithm write-ups (BigBuffer, IsometricRectangleSorter), see the
[top-level README](../README.md). This README is just the API surface.

## Public types

```rust
use exey_engine::{Engine, EngineConfig, RendererKind, FrameClock, Sprite, Texture, Vertex2D};
```

- [`Engine`](src/core.rs) — the equivalent of `ExeyEngineCore`. Owns Vulkan,
  exposes `draw_frame(&Window, &SpriteMesh, &[Sprite])`, `on_resize((u32, u32))`.
- [`EngineConfig`](src/core.rs) — start-time options (app name, renderer kind).
- [`RendererKind`](src/render/mod.rs) — `Simple | Batch | BigBuffer`.
  Maps from `--renderer` CLI strings via `RendererKind::from_cli`.
- [`Sprite`](src/render/sprite.rs) — per-sprite CPU state (position, size,
  velocity, tint). Mutate freely between frames.
- [`SpriteMesh`](src/render/sprite.rs) — shared GPU geometry (unit-quad
  vertex/index buffers) plus a single descriptor bound to a texture. One
  per (geometry, texture) pair the engine draws.
- [`Texture`](src/gfx/texture.rs) — owns a `vk::Image` + view + sampler.
  Build via `from_rgba(...)` or `from_png_bytes(...)`.
- [`Vertex2D`](src/draw/vertex.rs) — pos/color/uv vertex. M3 uses unit-quad
  local coords in `pos`; per-sprite world transform travels through the
  push constant.
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
| `exey.engine.render.sorting.IsometricRectangleSorter` | `render::sort::iso_rect` (M5) ★                  |
| `exey.engine.render.sorting.ScreenYSorter`    | `render::sort::screen_y`                                 |
| `exey.engine.render.camera.*`                 | `render::camera`                                         |
| `exey.engine.draw.*`                          | `draw::*` (M3+)                                          |
| `exey.engine.draw.animation.*`                | `draw::animation` (M7)                                   |

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
