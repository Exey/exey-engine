//! ExeyEngine — a Rust + Vulkan port of `ExeyEngine` by Exey Panteleev (AS3/Stage3D, 2014).
//!
//! The architecture mirrors the original:
//! - [`Engine`]                  is the equivalent of `ExeyEngineCore` — owns the device, the
//!                               render core, and per-frame update.
//! - [`render::RenderCore`]      is the equivalent of `RenderCore` — three layers
//!                               (background, world, gui), a pluggable [`render::IRenderer`],
//!                               and a pluggable [`render::sort::ISorter`].
//! - [`render::sprite_pipeline`] is the textured-quad graphics pipeline (one
//!                               combined image sampler + view/world push constants).
//! - [`render::SpriteMesh`]      is the shared unit-quad geometry + texture descriptor.
//! - [`render::Sprite`]          is per-sprite CPU state (position, size, velocity, tint).
//! - [`render::camera`]          mirrors AS3's `SimpleCamera2D` and `IsometricCamera2D`,
//!                               built on a shared [`AbstractCamera2D`](render::camera::AbstractCamera2D).
//! - [`render::iso`]             holds the iso ↔ logic ↔ world math —
//!                               replaces AS3's `IsoUtil.spaceToScreen`.
//! - [`draw::animation`]         M7: frame strips + per-sprite playback state
//!                               (mirrors AS3 `draw.animation.*`).
//!
//! M7 scope: every animated sprite owns an `AnimationState` referring to a
//! `FrameStrip` registered with `RenderCore::register_strip`. `Engine::draw_frame`
//! takes `dt` and walks the sprite slices once per frame, resolving the
//! strip's current frame into the sprite's `uv_offset` / `uv_scale`. The
//! renderers stay animation-agnostic.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod core;
pub mod draw;
pub mod gfx;
pub mod render;
pub mod time;

// Re-export the most-used types so demo code stays clean.
pub use crate::core::{Engine, EngineConfig};
pub use crate::draw::{AnimationState, FrameStrip, LoopMode, Vertex2D};
pub use crate::gfx::Texture;
pub use crate::render::{
    ICamera2D, IsometricCamera2D, RendererKind, SimpleCamera2D, Sprite, SpriteMesh,
};
// Sorting (M5+).
pub use crate::render::sort::{depth_compare, IsoBounds, IsoSortable, IsometricRectangleSorter};
pub use crate::time::FrameClock;

// Re-export the iso math as a module so the demo can call
// `exey_engine::iso::logic_to_world(...)` without going through `render`.
pub use crate::render::iso;

// Re-export glam so the demo and engine speak the exact same math types.
pub use glam;
