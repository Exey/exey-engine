//! ExeyEngine — a Rust + Vulkan port of `ExeyEngine` by Exey Panteleev (AS3/Stage3D, 2014).
//!
//! The architecture mirrors the original:
//! - [`Engine`]                  is the equivalent of `ExeyEngineCore` — owns the device, the
//!                               render core, and per-frame update.
//! - [`render::RenderCore`]      is the equivalent of `RenderCore` — three layers
//!                               (background, world, gui), a pluggable [`render::IRenderer`],
//!                               and a pluggable [`render::sort::ISorter`].
//! - [`render::sprite_pipeline`] is the M2 textured-quad graphics pipeline (one
//!                               combined image sampler + screen→clip push constant).
//! - [`render::Sprite`]          is the M2 drawable; M3 replaces it with the AS3
//!                               `Sprite2D` / `IRenderable` plumbing.
//!
//! M2 scope: textured quads on screen via dynamic-rendering, with a procedural
//! checkerboard texture supplied by the demo. The renderer trait now records
//! draws into the per-frame command buffer; Batch and BigBuffer renderers
//! delegate to [`render::SimpleRenderer`] until M5/M6 ship the real algorithms.

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
pub use crate::render::{RendererKind, Sprite};
pub use crate::time::FrameClock;

// Re-export glam so the demo and engine speak the exact same math types.
pub use glam;
