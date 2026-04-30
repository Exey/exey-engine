# exey-engine

The engine. Rust + Vulkan port of `ExeyEngine` by Exey Panteleev (AS3, 2014).

For the algorithm write-ups (BigBuffer, IsometricRectangleSorter), see the
[top-level README](../README.md). This README is just the API surface.

## Public types

```rust
use exey_engine::{Engine, EngineConfig, RendererKind, FrameClock};
```

- [`Engine`](src/core.rs) — the equivalent of `ExeyEngineCore`. Owns Vulkan,
  exposes `draw_frame(&Window)`, `on_resize((u32, u32))`.
- [`EngineConfig`](src/core.rs) — start-time options (app name, renderer kind).
- [`RendererKind`](src/render/mod.rs) — `Simple | Batch | BigBuffer`.
  Maps from `--renderer` CLI strings via `RendererKind::from_cli`.
- [`FrameClock`](src/time.rs) — delta time + smoothed FPS.

## Module map vs. the AS3 sources

| AS3 package                    | Rust module                                        |
|--------------------------------|----------------------------------------------------|
| `ragcat.engine.ExeyEngineCore` | `core::Engine`                                     |
| `ragcat.engine.stage3d.*`      | `gfx::{instance, device, swapchain, frame}`        |
| `ragcat.engine.render.RenderCore` | `render::RenderCore`                            |
| `ragcat.engine.render.renderers.IRenderer` | `render::IRenderer`                    |
| `ragcat.engine.render.renderers.SimpleRenderer` | `render::simple` (M3)             |
| `ragcat.engine.render.renderers.BatchRenderer`  | `render::batch_renderer` (M6)      |
| `ragcat.engine.render.renderers.BigBufferRenderer` | `render::big_buffer` (M6) ★      |
| `ragcat.engine.render.sorting.ISorter`          | `render::sort::ISorter`            |
| `ragcat.engine.render.sorting.IsometricRectangleSorter` | `render::sort::iso_rect` (M5) ★ |
| `ragcat.engine.render.sorting.ScreenYSorter`    | `render::sort::screen_y`           |
| `ragcat.engine.render.camera.*`                 | `render::camera`                   |
| `ragcat.engine.draw.*`                          | `render::draw` (M3+)               |
| `ragcat.engine.draw.animation.*`                | `render::animation` (M7)           |

The naming preserves the original conventions where it improves discoverability,
and Rust-ifies it where AS3 conventions don't translate (e.g. `IRenderable`
trait → just `Renderable` would be more idiomatic, but I kept the I-prefix
in `IRenderer` and `ISorter` because the strategy-pattern relationship is the
whole point and renaming would obscure the lineage).

## Building from a fresh tree

```sh
cargo build -p exey-engine
```

Requires the Vulkan SDK at runtime, but only `libloading` at build time.
