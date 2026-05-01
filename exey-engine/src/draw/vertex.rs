//! [`Vertex2D`] — the engine's single sprite vertex format.
//!
//! Mirrors the AS3 `VertexDataBinary` layout (position, color, texcoord) but
//! uses a `vec4` color instead of a packed `uint` RGBA. The packed-int format
//! made sense on Stage3D where it saved a `setVertexBufferAt` slot; on Vulkan
//! the saving is invisible and the readability cost is real, so we widen it.
//!
//! Stride is 32 bytes (8 × f32). M6's BigBufferRenderer caps a single buffer
//! pair at 65,536 vertices, so a pair is exactly 2 MiB — comfortable for
//! per-frame streaming uploads.

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct Vertex2D {
    /// Pixel-space position. Origin is top-left, +Y down (matches AS3 / web /
    /// most 2D engines). The vertex shader maps this to clip space using a
    /// push constant in M2; M4 replaces that with the camera matrix.
    pub pos: [f32; 2],
    /// Per-vertex tint, multiplied with the texture sample in the fragment
    /// shader. Maps to AS3's `setColorAndAlpha`. Default white = `[1, 1, 1, 1]`.
    pub color: [f32; 4],
    /// Texture coords in [0..1] with origin top-left.
    pub uv: [f32; 2],
}

impl Vertex2D {
    pub const STRIDE: u32 = std::mem::size_of::<Self>() as u32;

    pub const fn new(pos: [f32; 2], color: [f32; 4], uv: [f32; 2]) -> Self {
        Self { pos, color, uv }
    }
}
