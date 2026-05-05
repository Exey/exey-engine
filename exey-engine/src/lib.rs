//! ExeyEngine — a Rust + Vulkan port of `ExeyEngine` by Exey Panteleev (AS3/Stage3D, 2014).
//!
//! The architecture mirrors the original:
//! - [`Engine`]                  is the equivalent of `ExeyEngineCore` — owns the device, the
//!                               render core, and per-frame update.
//! - [`render::RenderCore`]      is the equivalent of `RenderCore` — three layers
//!                               (background, world, gui), a pluggable [`render::IRenderer`],
//!                               and a pluggable [`render::sort::ISorter`].
//! - [`render::sprite_pipeline`] is the textured-quad graphics pipeline (one
//!                               combined image sampler + screen/world push constants).
//! - [`render::SpriteMesh`]      is the shared unit-quad geometry + texture descriptor.
//! - [`render::Sprite`]          is per-sprite CPU state (position, size, velocity, tint).
//!
//! M3 scope: a flock of textured quads bouncing off the window edges, all
//! sharing one mesh + descriptor + pipeline; the demo updates sprite state
//! per frame and the renderer records one draw per sprite via push
//! constants. Batch and BigBuffer renderers still delegate to
//! [`render::SimpleRenderer`] until M5/M6 ship the real algorithms.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod core;
pub mod draw;
pub mod gfx;
pub mod render;
pub mod time;

// Re-export the most-used types so demo code stays clean.
pub use crate::core::{Engine, EngineConfig};
pub use crate::draw::Vertex2D;
pub use crate::gfx::Texture;
pub use crate::render::{RendererKind, Sprite, SpriteMesh};
pub use crate::time::FrameClock;

// Re-export glam so the demo and engine speak the exact same math types.
pub use glam;
