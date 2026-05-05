//! M3 sprite types: lightweight state ([`Sprite`]) and shared GPU geometry
//! ([`SpriteMesh`]).
//!
//! ## Why this differs from M2
//!
//! In M2 every `Sprite` owned its own vertex buffer, index buffer, and
//! descriptor set. With one or two sprites that was fine; once the demo
//! moved to a flock of dozens it became wasteful (every sprite uses the
//! same unit-quad geometry, and most share the same texture). M3 splits
//! the responsibilities:
//!
//! * [`SpriteMesh`] owns the shared unit-quad vertex/index buffers and a
//!   single descriptor set per texture. Built once, used by N sprites.
//! * [`Sprite`] is plain CPU state — position, size, velocity, tint. No
//!   Vulkan handles. Mutating it is just `f32` writes; the renderer reads
//!   it each frame and emits push-constant + draw calls.
//!
//! ## Why not the AS3 IRenderable / FrameData / RenderOpInstance triad
//!
//! The AS3 plumbing was shaped by AS3's lack of generics and its preference
//! for interface-based dispatch. In Rust that translates to either a trait
//! object zoo (`Box<dyn IRenderable>`) or a tagged-union enum, neither of
//! which buys us anything for the M3 deliverable. We'll revisit when M5/M6
//! introduce real per-renderer behaviour (batching, big-buffer streaming).
//!
//! ## Coordinate system
//!
//! `pos` is the sprite's top-left in pixels (framebuffer coords, +Y down).
//! `size` is the sprite's pixel size. The vertex shader composes:
//!
//! ```text
//!   pixel_pos = local * size + pos     // local ∈ [0..1]² from the unit quad
//!   ndc.xy    = pixel_pos * (2/extent) + (-1, -1)
//! ```
//!
//! See `shaders/sprite.vert`.

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::draw::Vertex2D;
use crate::gfx::{Buffer, Device, Instance, Texture};
use crate::render::sprite_pipeline::SpritePipeline;

/// Shared GPU geometry + texture binding for a group of sprites.
///
/// One instance is built per (geometry, texture) pair the engine needs to
/// draw. M3 only ever has one — a unit quad bound to the procedural
/// checkerboard — but the type is structured so M5 can hold many.
pub struct SpriteMesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub index_count: u32,
    pub descriptor_set: vk::DescriptorSet,
}

impl SpriteMesh {
    /// Build the unit-quad mesh and bind it to `texture`.
    ///
    /// The quad is two CCW-wound triangles spanning local space `[0..1]²`
    /// with UVs covering the full texture. World position and size travel
    /// through the push constant per draw.
    pub fn unit_quad(
        instance: &Instance,
        device: &Device,
        pipeline: &SpritePipeline,
        texture: &Texture,
    ) -> Result<Self> {
        log::info!(
            "SpriteMesh::unit_quad — building shared unit-quad mesh, texture {}x{}",
            texture.width, texture.height,
        );
        let white = [1.0, 1.0, 1.0, 1.0];
        // Local-space unit quad. pos is in [0..1]² — the vertex shader
        // multiplies by the per-sprite world_size and adds world_pos.
        let verts = [
            Vertex2D::new([0.0, 0.0], white, [0.0, 0.0]),
            Vertex2D::new([1.0, 0.0], white, [1.0, 0.0]),
            Vertex2D::new([1.0, 1.0], white, [1.0, 1.0]),
            Vertex2D::new([0.0, 1.0], white, [0.0, 1.0]),
        ];
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

        log::info!(
            "SpriteMesh::unit_quad — built  vbuf={:?}  ibuf={:?}  desc={:?}",
            vertex_buffer.handle, index_buffer.handle, descriptor_set,
        );

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            descriptor_set,
        })
    }

    /// Bind this mesh's vertex/index buffers and descriptor set on the
    /// command buffer. Call once before drawing N sprites that share it.
    pub fn bind(&self, device: &Device, cb: vk::CommandBuffer, pipeline: &SpritePipeline) {
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
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        // Descriptor set is freed implicitly when the pool is destroyed
        // (pool was not created with FREE_DESCRIPTOR_SET so explicit free
        // would be illegal).
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
    }
}

/// CPU-side sprite state. Mutate freely between frames; the renderer reads
/// these fields each frame and pushes them through a per-draw push constant.
#[derive(Copy, Clone, Debug)]
pub struct Sprite {
    /// Top-left in pixels (framebuffer coords, +Y down).
    pub pos: [f32; 2],
    /// Width and height in pixels.
    pub size: [f32; 2],
    /// Velocity in pixels per second. Used by the demo's bounce update;
    /// the renderer ignores this.
    pub velocity: [f32; 2],
    /// Per-sprite color modulator. Multiplies the sampled texel.
    pub tint: [f32; 4],
}

impl Sprite {
    pub fn new(pos: [f32; 2], size: [f32; 2], velocity: [f32; 2], tint: [f32; 4]) -> Self {
        Self { pos, size, velocity, tint }
    }
}
