//! ExeyEngine — a Rust + Vulkan port of `RCatEngine` by Exey Panteleev (AS3/Stage3D, 2014).
//!
//! The architecture mirrors the original:
//! - [`Engine`]                  is the equivalent of `RCatEngineCore` — owns the device, the
//!                               render core, and per-frame update.
//! - [`render::RenderCore`]      is the equivalent of `RenderCore` — three layers
//!                               (background, world, gui), a pluggable [`render::IRenderer`],
//!                               and a pluggable [`render::sort::ISorter`].
//! - [`render::big_buffer`]      is the BigBufferRenderer — see the README for the algorithm.
//! - [`render::sort::iso_rect`]  is the topological iso depth sorter.
//!
//! M1 scope: window, instance, device, swapchain, dynamic-rendering clear,
//! frame loop with FPS counter. No drawing yet.

#![forbid(unsafe_op_in_unsafe_fn)]

pub mod core;
pub mod gfx;
pub mod render;
pub mod time;

// Re-export the most-used types so demo code stays clean.
pub use crate::core::{Engine, EngineConfig};
pub use crate::render::RendererKind;
pub use crate::time::FrameClock;

// Re-export glam so the demo and engine speak the exact same math types.
pub use glam;
