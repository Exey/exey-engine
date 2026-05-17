//! [`BigBufferRenderer`] — the M6 algorithm.
//!
//! See the top-level [README](../../../README.md) for the algorithm prose.
//! This module is the implementation.
//!
//! ## Per-frame flow
//!
//! 1. Pick the pair-ring slot for this frame-in-flight. The ring has
//!    [`MAX_FRAMES_IN_FLIGHT`](crate::gfx::frame::MAX_FRAMES_IN_FLIGHT)
//!    slots so we never write a pair whose previous contents the GPU
//!    might still be reading.
//! 2. Concatenate the world (in iso-sorted order) and gui sprite slices
//!    into a single logical input stream. Walk it via
//!    [`RenderBufferPair::populate`] in a loop, borrowing the slot's
//!    pre-allocated pairs in turn until the stream is exhausted.
//! 3. For each populated pair: bind its vertex+index buffer, push the
//!    "neutered" per-sprite constants (vertex data is already baked,
//!    see [`crate::render::buffer_pair`]), then walk the sprite range
//!    that pair owns and emit one `cmd_draw_indexed` per state-change
//!    run.
//!
//! ## Run-break criteria
//!
//! In the AS3 original the loop broke runs on changes in `texture`,
//! `transformation`, `camera`, `alpha`, or `blend mode`. In the Rust
//! port (M6) the only per-sprite state that lives outside the vertex
//! data is the texture descriptor — selected via `Sprite::mesh_idx`.
//! Per-sprite alpha and tint are baked into the vertex `color`;
//! per-sprite UV sub-region is baked into the vertex `uv`; world
//! position and size are baked into vertex `pos`. There's no
//! per-sprite blend mode or camera in the current API. So the run-
//! break is just `mesh_idx changed`. The other checks are left as
//! `// TODO(M-future)` hooks — see inside [`Self::record`].
//!
//! ## Pool sizing
//!
//! Each frame-in-flight slot pre-allocates [`PAIRS_PER_SLOT`] pairs in
//! `init`. With one pair = 16,373 quads, two pairs cover ~32k visible
//! sprites — far more than the current demo's ~6k. If a future scene
//! exceeds this, `record` panics with a clear message; bump
//! [`PAIRS_PER_SLOT`] and rebuild. We don't grow the pool from inside
//! `record` because growth needs `&Instance` and widening every
//! `IRenderer::record` for the rare case isn't worth it.

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::gfx::frame::MAX_FRAMES_IN_FLIGHT;
use crate::gfx::{Device, Instance};
use crate::render::buffer_pair::{PopulateOutcome, RenderBufferPair};
use crate::render::sprite_pipeline::SpritePushConstants;
use crate::render::{IRenderer, RenderContext, RendererKind, Sprite};

/// Pairs allocated per frame-in-flight slot. Two is enough for ~32k
/// sprites, which is well past anything the engine renders today.
/// Bump if a scene ever overflows (the panic message inside `record`
/// will tell you to).
pub const PAIRS_PER_SLOT: usize = 2;

/// Per-frame-in-flight slot. Owns its pre-allocated pairs and a `used`
/// cursor that resets to 0 at the top of each `record` call.
struct PoolSlot {
    pairs: Vec<RenderBufferPair>,
    used: usize,
}

impl PoolSlot {
    fn new() -> Self {
        Self {
            pairs: Vec::new(),
            used: 0,
        }
    }
    fn destroy(&mut self, device: &Device) {
        for p in &mut self.pairs {
            p.destroy(device);
        }
        self.pairs.clear();
    }
}

pub struct BigBufferRenderer {
    /// Ring of per-frame-in-flight pools. Indexed by `frame_count %
    /// MAX_FRAMES_IN_FLIGHT`. Slot `k` is safe to overwrite as soon as
    /// frame `k`'s fence has been waited on — which `gfx::frame::acquire`
    /// does before we get here.
    slots: [PoolSlot; MAX_FRAMES_IN_FLIGHT],
    /// Monotonic counter to pick the slot for this frame. Bumps once
    /// per `record` call.
    frame_count: u64,
    /// Scratch buffer for the merged sprite stream (world-sorted ++ gui).
    /// Reused per frame; capacity grows but never shrinks.
    merged_scratch: Vec<Sprite>,
}

impl BigBufferRenderer {
    pub fn new() -> Self {
        Self {
            slots: [PoolSlot::new(), PoolSlot::new()],
            frame_count: 0,
            merged_scratch: Vec::new(),
        }
    }

    /// Allocate the per-slot pair pool. Called from the widened
    /// [`IRenderer::init`] inside `RenderCore::new`.
    pub(crate) fn allocate_pools(
        &mut self,
        instance: &Instance,
        device: &Device,
    ) -> Result<()> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            slot.pairs.reserve_exact(PAIRS_PER_SLOT);
            for _ in 0..PAIRS_PER_SLOT {
                slot.pairs.push(RenderBufferPair::new(instance, device)?);
            }
            log::info!(
                "BigBufferRenderer: slot {} pre-allocated {} pairs",
                i,
                slot.pairs.len(),
            );
        }
        Ok(())
    }
}

impl Default for BigBufferRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl IRenderer for BigBufferRenderer {
    fn kind(&self) -> RendererKind {
        RendererKind::BigBuffer
    }

    fn init(&mut self, _device: &Device) -> Result<()> {
        log::info!(
            "renderer init: bigbuffer (M6 — 65k-cap streaming, state-change batched, \
             {} frames in flight × {} pairs/slot)",
            MAX_FRAMES_IN_FLIGHT, PAIRS_PER_SLOT,
        );
        // The actual pair allocation happens in `allocate_pools`, called
        // by `RenderCore::new` after this — it needs `&Instance` which
        // we don't get through `IRenderer::init`. (See the module-level
        // doc comment in `render/mod.rs` for the rationale.)
        Ok(())
    }

    fn record(&mut self, device: &Device, cb: vk::CommandBuffer, ctx: &RenderContext) {
        // Pick this frame's pool slot and reset its `used` cursor.
        let slot_idx = (self.frame_count as usize) % MAX_FRAMES_IN_FLIGHT;
        self.frame_count = self.frame_count.wrapping_add(1);
        let slot = &mut self.slots[slot_idx];
        slot.used = 0;

        // Merge world (in sorted order) and gui (in input order) into one
        // sprite stream. The merged stream is then fed to populate().
        let merged = &mut self.merged_scratch;
        merged.clear();
        merged.reserve(ctx.world_sprites.len() + ctx.gui_sprites.len());

        if !ctx.world_sprites.is_empty() {
            if ctx.world_sort_order.len() == ctx.world_sprites.len() {
                for &i in ctx.world_sort_order {
                    merged.push(ctx.world_sprites[i as usize]);
                }
            } else {
                log::warn!(
                    "BigBufferRenderer: world_sort_order length ({}) != \
                     world_sprites length ({}); drawing world in input order",
                    ctx.world_sort_order.len(),
                    ctx.world_sprites.len(),
                );
                merged.extend_from_slice(ctx.world_sprites);
            }
        }
        merged.extend_from_slice(ctx.gui_sprites);

        if merged.is_empty() {
            if ctx.verbose {
                log::info!(
                    "  BigBufferRenderer::record  → 0 sprites (clear-only frame)"
                );
            }
            return;
        }

        // Bind pipeline + dynamic state (viewport/scissor) once.
        ctx.pipeline.bind(device, cb, ctx.extent);

        // The "neutered" per-sprite push constant. With these values the
        // vertex shader composes:
        //   world_pixel = in_pos * 1 + 0    = in_pos       (already baked)
        //   sample_uv   = in_uv  * 1 + 0    = in_uv        (already baked)
        //   v_color     = in_color * white  = in_color     (already baked)
        // and the view transform applies as usual. One push per frame,
        // shared by every draw.
        let push = SpritePushConstants {
            view_scale: ctx.view.view_scale,
            view_offset: ctx.view.view_offset,
            world_pos: [0.0, 0.0],
            world_size: [1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
        };
        ctx.pipeline.push_constants(device, cb, &push);

        // Populate as many pairs as it takes to cover the merged stream.
        let mut start = 0usize;
        let total = merged.len();
        let mut total_draws = 0u32;
        let mut total_mesh_changes = 0u32;
        let mut pairs_used = 0u32;

        loop {
            if slot.used >= slot.pairs.len() {
                panic!(
                    "BigBufferRenderer: scene needs more than {} pairs/slot \
                     (merged sprite count = {}); bump `PAIRS_PER_SLOT` in \
                     render::big_buffer",
                    PAIRS_PER_SLOT, total,
                );
            }
            let pair_idx = slot.used;
            slot.used += 1;
            pairs_used += 1;
            let pair = &mut slot.pairs[pair_idx];
            pair.reset();
            let outcome = pair.populate(merged, start);

            // Bind this pair's vertex + index buffers.
            let v_offsets: [vk::DeviceSize; 1] = [0];
            unsafe {
                device.logical.cmd_bind_vertex_buffers(
                    cb,
                    0,
                    &[pair.vertices.handle],
                    &v_offsets,
                );
                device.logical.cmd_bind_index_buffer(
                    cb,
                    pair.indices.handle,
                    0,
                    vk::IndexType::UINT16,
                );
            }

            // Walk this pair's sprite range. Each sprite occupies 6
            // consecutive indices in the pair's index buffer (indices
            // are written densely: sprite 0 = indices 0..6, sprite 1 =
            // 6..12, …). A "run" is a maximal stretch where mesh_idx
            // doesn't change.
            let pair_first = pair.start_sprite_idx;
            let pair_last = pair.end_sprite_idx;
            let mut run_first_index: u32 = 0;
            let mut run_index_count: u32 = 0;
            let mut current_mesh_idx: Option<u8> = None;

            let pipeline = ctx.pipeline;
            let meshes = ctx.meshes;
            let bind_texture_for = |mesh_idx: u8| {
                let clamped = if (mesh_idx as usize) < meshes.len() {
                    mesh_idx
                } else {
                    if ctx.verbose {
                        log::warn!(
                            "BigBufferRenderer: sprite.mesh_idx={} out of range \
                             (have {} meshes); using mesh 0",
                            mesh_idx,
                            meshes.len(),
                        );
                    }
                    0
                };
                pipeline.bind_texture(device, cb, meshes[clamped as usize].descriptor_set);
            };

            for (local_i, sprite_i) in (pair_first..pair_last).enumerate() {
                let mesh_idx = merged[sprite_i].mesh_idx;
                let mesh_changed = current_mesh_idx.map_or(true, |c| c != mesh_idx);
                // TODO(M-future): break also on blend mode change, when
                //   `Sprite` grows a blend field. AS3 had MODE_OPAQUE /
                //   MODE_ADD; we'd flush the run and rebind the pipeline.
                // TODO(M-future): break also on real per-sprite alpha
                //   (today's tint alpha is baked into the vertex color
                //   so this break never fires).
                // TODO(M-future): break also on camera change when the
                //   engine grows multi-camera support.
                if mesh_changed && run_index_count > 0 {
                    unsafe {
                        device.logical.cmd_draw_indexed(
                            cb,
                            run_index_count,
                            1,
                            run_first_index,
                            0,
                            0,
                        );
                    }
                    total_draws += 1;
                    run_first_index += run_index_count;
                    run_index_count = 0;
                }
                if mesh_changed {
                    bind_texture_for(mesh_idx);
                    current_mesh_idx = Some(mesh_idx);
                    if local_i > 0 {
                        total_mesh_changes += 1;
                    }
                }
                run_index_count += 6;
            }
            if run_index_count > 0 {
                unsafe {
                    device.logical.cmd_draw_indexed(
                        cb,
                        run_index_count,
                        1,
                        run_first_index,
                        0,
                        0,
                    );
                }
                total_draws += 1;
            }

            match outcome {
                PopulateOutcome::Exhausted => break,
                PopulateOutcome::Continue { next_start } => {
                    start = next_start;
                    if start == total {
                        // Defensive — populate should have returned
                        // Exhausted in this case. Break anyway.
                        break;
                    }
                }
            }
        }

        if ctx.verbose {
            log::info!(
                "  BigBufferRenderer::record  → world={} (sorted), gui={}, meshes={} \
                 | pairs={} draws={} mesh_changes={} verts={}",
                ctx.world_sprites.len(),
                ctx.gui_sprites.len(),
                ctx.meshes.len(),
                pairs_used,
                total_draws,
                total_mesh_changes,
                total * 4,
            );
        }
    }

    fn destroy(&mut self, device: &Device) {
        for slot in &mut self.slots {
            slot.destroy(device);
        }
    }
}
