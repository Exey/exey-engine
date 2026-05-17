//! [`RenderBufferPair`] — the streaming vertex + index pair used by
//! [`crate::render::big_buffer::BigBufferRenderer`].
//!
//! Direct port of AS3 `bigbuffer.RenderBufferPair`. The shape changes a
//! little because Vulkan host-visible memory replaces Stage3D's
//! `uploadFrom*` calls, but the algorithm is the same: bake one sprite
//! after another into the buffer until adding another would exceed the
//! u16 index cap, then close out and let the renderer start a new pair.
//!
//! ## The 65,493 cap
//!
//! Stage3D's index buffer is u16, so a single buffer can address at most
//! 65,535 vertices. AS3 watched `lastVertexIndex * 4 >= 65535 - 4*4` (slack
//! for one more quad). We keep the same literal — [`TRIP_VERTEX_COUNT`] —
//! so the algorithm's batching characteristics transfer. Vulkan itself
//! would happily use `UINT32` and let us pack much more, but the README
//! documents this constraint as intentional.
//!
//! ## Bake-into-vertex strategy
//!
//! The M5 sprite pipeline applies per-sprite `world_pos` / `world_size` /
//! `tint` / `uv_offset` / `uv_scale` through push constants — that's how
//! `SimpleRenderer` keeps per-vertex data tiny. BigBuffer can't push
//! per-sprite data because the whole point is to coalesce many sprites
//! into one draw call; everything per-sprite has to live in the vertex
//! data. So [`Self::populate`]:
//!
//! 1. transforms the unit-quad locals into pixel-space world positions
//!    using each sprite's own `pos` / `size`,
//! 2. transforms unit-quad UVs into the sprite's atlas sub-region using
//!    `uv_offset` / `uv_scale`,
//! 3. bakes `tint` into the per-vertex color.
//!
//! The renderer then pushes "neutered" per-sprite constants
//! (`world_pos=[0,0] world_size=[1,1] tint=white uv_offset=[0,0]
//! uv_scale=[1,1]`) once per camera/run. The vertex shader's
//! `in_pos * world_size + world_pos` reduces to `in_pos`, which is now
//! the baked world-pixel position. Same for UVs and tint. The view
//! transform applies as usual.

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::draw::Vertex2D;
use crate::gfx::{Buffer, Device, Instance};
use crate::render::Sprite;

/// Max vertices we'll pack into one pair before closing it. Lifted
/// verbatim from AS3:
///
/// ```actionscript
///     if (lastVertexIndex * 4 >= 65493) { // 65535-4*4
/// ```
///
/// (The literal `65493` doesn't algebraically match the inline comment,
/// but it's what shipped — and the README is explicit that we preserve
/// AS3's batching characteristics so head-to-head timings transfer. The
/// effective cap is "stop comfortably below u16's 65,535".)
pub const TRIP_VERTEX_COUNT: usize = 65_493;

/// Buffer capacity. Headroom of one quad above [`TRIP_VERTEX_COUNT`].
pub const BUFFER_VERTEX_CAPACITY: usize = 65_536;
/// Per-pair index capacity: 6 indices per quad, max 16_384 quads.
pub const BUFFER_INDEX_CAPACITY: usize = (BUFFER_VERTEX_CAPACITY / 4) * 6;

/// Outcome of [`RenderBufferPair::populate`]. Either we exhausted the
/// sprite slice or we tripped the cap mid-loop.
#[derive(Copy, Clone, Debug)]
pub enum PopulateOutcome {
    /// Reached the end of the sprite slice. This was the last pair.
    Exhausted,
    /// Hit the cap. The next pair should resume at this sprite index.
    Continue { next_start: usize },
}

/// One vertex + index buffer pair, persistently host-mapped. Reusable
/// across frames via [`Self::reset`]. AS3 used `ObjectPool`; here we
/// keep a `Vec<RenderBufferPair>` on the renderer and re-borrow.
pub struct RenderBufferPair {
    pub vertices: Buffer,
    pub indices: Buffer,
    /// Sprite indices `[start_sprite_idx .. end_sprite_idx)` that were
    /// packed into this pair on the last [`Self::populate`] call. The
    /// renderer uses this range to walk the merged sprite stream and
    /// emit one `cmd_draw_indexed` per state-change run.
    pub start_sprite_idx: usize,
    pub end_sprite_idx: usize,
    /// Vertex count written this frame. Range: 0..=BUFFER_VERTEX_CAPACITY.
    pub vertex_count: usize,
    /// Index count written this frame. Always `vertex_count / 4 * 6`.
    pub index_count: usize,
}

impl RenderBufferPair {
    pub fn new(instance: &Instance, device: &Device) -> Result<Self> {
        let vertex_size = (BUFFER_VERTEX_CAPACITY * std::mem::size_of::<Vertex2D>())
            as vk::DeviceSize;
        let index_size = (BUFFER_INDEX_CAPACITY * std::mem::size_of::<u16>())
            as vk::DeviceSize;
        let mut vertices = Buffer::host_visible(
            instance,
            device,
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        vertices.map_persistent(device)?;
        let mut indices = Buffer::host_visible(
            instance,
            device,
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        indices.map_persistent(device)?;
        log::info!(
            "RenderBufferPair: allocated vertex={} bytes, index={} bytes \
             (cap {} verts / {} indices, trip @ {} verts)",
            vertex_size, index_size,
            BUFFER_VERTEX_CAPACITY, BUFFER_INDEX_CAPACITY, TRIP_VERTEX_COUNT,
        );
        Ok(Self {
            vertices,
            indices,
            start_sprite_idx: 0,
            end_sprite_idx: 0,
            vertex_count: 0,
            index_count: 0,
        })
    }

    /// Mark the pair empty. Doesn't free memory — buffers stay mapped.
    pub fn reset(&mut self) {
        self.start_sprite_idx = 0;
        self.end_sprite_idx = 0;
        self.vertex_count = 0;
        self.index_count = 0;
    }

    /// Populate this pair with sprites starting at `start`, packing into
    /// the persistently-mapped vertex+index buffers. The caller is
    /// responsible for ordering: pass the sprite slice in the order you
    /// want it drawn (i.e. already iso-sorted for the world layer; in
    /// input order for the GUI layer; or any concatenation thereof).
    ///
    /// Returns [`PopulateOutcome::Continue`] if the cap tripped — the
    /// renderer should construct another pair starting at the returned
    /// index — or [`PopulateOutcome::Exhausted`] if we packed everything.
    pub fn populate(&mut self, sprites: &[Sprite], start: usize) -> PopulateOutcome {
        self.start_sprite_idx = start;
        self.vertex_count = 0;
        self.index_count = 0;

        for i in start..sprites.len() {
            // Before appending the next quad: would it cross the cap?
            // AS3 used `lastVertexIndex * 4 >= 65493` inside the per-
            // vertex loop; the cleaner equivalent is "we've packed N
            // quads, would N+1 fit?" — checked at the top of the
            // iteration, before adding more.
            if self.vertex_count + 4 > TRIP_VERTEX_COUNT {
                self.end_sprite_idx = i;
                return PopulateOutcome::Continue { next_start: i };
            }

            let s = &sprites[i];

            // Bake unit-quad locals → world pixel positions. Vertex
            // shader will multiply by view_scale + add view_offset.
            // Layout matches `SpriteMesh::unit_quad`: (0,0) (1,0) (1,1)
            // (0,1) — same CCW winding as the M3 unit quad.
            let x0 = s.pos[0];
            let y0 = s.pos[1];
            let x1 = s.pos[0] + s.size[0];
            let y1 = s.pos[1] + s.size[1];
            let u0 = s.uv_offset[0];
            let v0 = s.uv_offset[1];
            let u1 = s.uv_offset[0] + s.uv_scale[0];
            let v1 = s.uv_offset[1] + s.uv_scale[1];
            let c = s.tint;

            let quad = [
                Vertex2D::new([x0, y0], c, [u0, v0]),
                Vertex2D::new([x1, y0], c, [u1, v0]),
                Vertex2D::new([x1, y1], c, [u1, v1]),
                Vertex2D::new([x0, y1], c, [u0, v1]),
            ];
            let base_vertex = self.vertex_count as u16;
            let quad_indices: [u16; 6] = [
                base_vertex,
                base_vertex + 1,
                base_vertex + 2,
                base_vertex,
                base_vertex + 2,
                base_vertex + 3,
            ];

            let v_offset = (self.vertex_count * std::mem::size_of::<Vertex2D>())
                as vk::DeviceSize;
            self.vertices
                .write_at_offset(v_offset, bytemuck::cast_slice(&quad));

            let i_offset = (self.index_count * std::mem::size_of::<u16>())
                as vk::DeviceSize;
            self.indices
                .write_at_offset(i_offset, bytemuck::cast_slice(&quad_indices));

            self.vertex_count += 4;
            self.index_count += 6;
        }
        self.end_sprite_idx = sprites.len();
        PopulateOutcome::Exhausted
    }

    pub fn destroy(&mut self, device: &Device) {
        self.vertices.destroy(device);
        self.indices.destroy(device);
    }
}
