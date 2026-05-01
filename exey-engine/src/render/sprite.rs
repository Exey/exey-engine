//! [`Sprite`] — the M2 drawable. Owns the GPU resources for one textured quad:
//! its vertex/index buffers and its descriptor set. The descriptor set
//! references a `gfx::Texture` whose lifetime the caller manages separately
//! (textures will live in an asset manager from M3 onward).
//!
//! This is **not** the AS3 `Sprite2D` — that arrives in M3 with `IRenderable`,
//! `FrameData`, and the per-frame transform matrix. For M2 we keep the
//! engine surface minimal: the demo builds a couple of these once and the
//! renderer redraws them each frame.

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::draw::Vertex2D;
use crate::gfx::{Buffer, Device, Instance, Texture};
use crate::render::sprite_pipeline::{SpritePipeline, SpritePushConstants};

/// One textured quad worth of draw data.
pub struct Sprite {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub descriptor_set: vk::DescriptorSet,
    /// Per-sprite tint. Position is encoded in the vertices for M2; M3 will
    /// move position into the push-constant matrix and bake just `[0..w]`,
    /// `[0..h]` quads in the vertex buffer.
    pub tint: [f32; 4],
}

impl Sprite {
    /// Build a textured quad whose vertices live in pixel coords. `(x, y)`
    /// is the top-left, `(x + w, y + h)` the bottom-right. UVs are the full
    /// [0..1] square — i.e. one tile = the whole texture.
    pub fn quad(
        instance: &Instance,
        device: &Device,
        pipeline: &SpritePipeline,
        texture: &Texture,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
    ) -> Result<Self> {
        let white = [1.0, 1.0, 1.0, 1.0];
        let verts = [
            Vertex2D::new([x,     y    ], white, [0.0, 0.0]),
            Vertex2D::new([x + w, y    ], white, [1.0, 0.0]),
            Vertex2D::new([x + w, y + h], white, [1.0, 1.0]),
            Vertex2D::new([x,     y + h], white, [0.0, 1.0]),
        ];
        // Two triangles, COUNTER_CLOCKWISE wound (cull mode is NONE so this
        // doesn't matter for visibility, just stays consistent if we later
        // turn culling on).
        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

        let v_bytes: &[u8] = bytemuck::cast_slice(&verts);
        let i_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = Buffer::host_visible(
            instance,
            device,
            v_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;
        vertex_buffer.write_bytes(device, v_bytes)?;

        let index_buffer = Buffer::host_visible(
            instance,
            device,
            i_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;
        index_buffer.write_bytes(device, i_bytes)?;

        let descriptor_set = pipeline.allocate_descriptor(device, texture)?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            descriptor_set,
            tint: white,
        })
    }

    /// Record the draw commands for this sprite into `cb`. Caller has
    /// already bound the pipeline + set the viewport (via [`SpritePipeline::bind`]).
    pub fn record(
        &self,
        device: &Device,
        cb: vk::CommandBuffer,
        pipeline: &SpritePipeline,
        extent: vk::Extent2D,
    ) {
        let pc = SpritePushConstants::for_extent(extent, self.tint);
        pipeline.push_constants(device, cb, &pc);
        pipeline.bind_texture(device, cb, self.descriptor_set);
        let offsets: [vk::DeviceSize; 1] = [0];
        unsafe {
            device.logical.cmd_bind_vertex_buffers(
                cb,
                0,
                &[self.vertex_buffer.handle],
                &offsets,
            );
            device.logical.cmd_bind_index_buffer(
                cb,
                self.index_buffer.handle,
                0,
                vk::IndexType::UINT16,
            );
            device.logical.cmd_draw_indexed(cb, self.index_count, 1, 0, 0, 0);
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        // The descriptor set is freed implicitly when the pool is destroyed.
        // (We allocated from a non-FREE_DESCRIPTOR_SET pool, so explicit
        // free isn't legal anyway.)
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}
