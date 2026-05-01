//! Drawable primitives. Mirrors the AS3 `draw/` package.
//!
//! M2 has just the vertex format. M3 adds `IRenderable`, `Sprite2D`,
//! `FrameData`, and `RenderOperationData`.

pub mod vertex;

pub use vertex::Vertex2D;
