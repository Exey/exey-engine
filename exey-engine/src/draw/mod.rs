//! Drawable primitives. Mirrors the AS3 `draw/` package.
//!
//! M2 added the vertex format. M7 adds `animation` (the AS3
//! `draw.animation.*` cluster — frame strips + per-sprite playback state).

pub mod animation;
pub mod vertex;

pub use animation::{AnimationState, FrameStrip, LoopMode};
pub use vertex::Vertex2D;
