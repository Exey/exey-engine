//! Render layer. Mirrors the AS3 `render/` package.
//!
//! - [`RenderCore`] holds three layers (background / world / gui), the cameras,
//!   and the chosen [`IRenderer`] strategy. Per frame it sorts the world layer
//!   (via [`sort::ISorter`]), builds a [`RenderContext`], and hands that to
//!   the renderer.
//! - [`IRenderer`] is the strategy trait. M2 wires up a functional
//!   [`SimpleRenderer`]; the Batch and BigBuffer kinds delegate to it for
//!   identical output until M5/M6 ship the real algorithms.
//!
//! The `Kind` enum exists so the demo can pick a renderer at startup via a CLI
//! flag (`--renderer simple|batch|bigbuffer`).

pub mod camera;
pub mod iso;
pub mod sort;
pub mod sprite;
pub mod sprite_pipeline;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::gfx::{Device, Swapchain};

pub use camera::{ICamera2D, IsometricCamera2D, SimpleCamera2D, ViewTransform};
pub use sprite::{Sprite, SpriteMesh};
pub use sprite_pipeline::{SpritePipeline, SpritePushConstants};

/// Picks which IRenderer implementation to construct. Mirrors the comments
/// in the AS3 `ExeyEngineCore.context3dCreated_handler` where the user
/// chose between `BigBufferRenderer`, `BatchRenderer`, and `SimpleRenderer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum RendererKind {
    /// One draw call per sprite. Slowest, easiest to debug.
    Simple,
    /// Group identical render-ops, one draw call per group.
    Batch,
    /// Pack everything into 65k-vertex buffers, draw on state change.
    /// This is the algorithm the README documents at length.
    #[default]
    BigBuffer,
}

impl RendererKind {
    pub fn from_cli(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "simple" => Some(Self::Simple),
            "batch" => Some(Self::Batch),
            "bigbuffer" | "big-buffer" | "big_buffer" => Some(Self::BigBuffer),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Batch => "batch",
            Self::BigBuffer => "bigbuffer",
        }
    }
}

/// Per-frame data passed from `RenderCore` into [`IRenderer::record`].
///
/// In M5 the context carries:
/// * `pipeline` — the textured-quad pipeline (shared)
/// * `meshes` — slice of [`SpriteMesh`]es; sprites pick which via `mesh_idx`
/// * `extent` — framebuffer size in pixels (for viewport/scissor)
/// * `view` — the camera's world→clip transform for this frame
/// * `world_sprites` — sprites rendered with iso-sorted draw order
/// * `world_sort_order` — permutation of `0..world_sprites.len()` from
///   the iso sorter; the renderer iterates this rather than the slice directly
/// * `gui_sprites` — sprites rendered after `world_sprites` in input order
///   (overlay text, HUD, debug). Not sorted by the iso sorter.
/// * `clear_color`, `verbose` — orchestration / diagnostics
///
/// M6 will let the BigBuffer renderer access the streaming vertex pool
/// through here.
pub struct RenderContext<'a> {
    pub pipeline: &'a SpritePipeline,
    pub meshes: &'a [&'a SpriteMesh],
    pub extent: vk::Extent2D,
    pub view: ViewTransform,
    pub world_sprites: &'a [Sprite],
    pub world_sort_order: &'a [u32],
    pub gui_sprites: &'a [Sprite],
    pub clear_color: [f32; 4],
    /// Diagnostic logging toggle. The engine sets this for the first few
    /// frames after startup so we can confirm the renderer is actually
    /// recording draws; otherwise the renderer stays silent.
    pub verbose: bool,
}

/// Trait equivalent of AS3 `IRenderer`. The two methods correspond to the
/// AS3 `init` (called once after device creation) and `render` (per-frame).
/// In the Vulkan port `render` becomes [`record`](IRenderer::record) — it
/// records draw commands into a command buffer that the caller has already
/// transitioned and put inside a dynamic-rendering scope.
pub trait IRenderer {
    fn kind(&self) -> RendererKind;
    /// Called once after device creation. Late milestones use this to build
    /// per-renderer pipelines or persistent buffers; M2 has nothing to do here.
    fn init(&mut self, device: &Device) -> Result<()>;
    /// Called per frame inside `cmd_begin_rendering` / `cmd_end_rendering`.
    /// Implementations issue `cmd_bind_pipeline`, `cmd_push_constants`,
    /// `cmd_bind_descriptor_sets`, vertex/index binds, and draws.
    fn record(&mut self, device: &Device, cb: vk::CommandBuffer, ctx: &RenderContext);
    /// Called once before the device is destroyed.
    fn destroy(&mut self, device: &Device);
}

/// One draw call per sprite. The simplest possible IRenderer — easy to read,
/// easy to debug, and the M2/M3 baseline against which Batch and BigBuffer
/// will be benchmarked once they ship.
pub struct SimpleRenderer;

impl SimpleRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl IRenderer for SimpleRenderer {
    fn kind(&self) -> RendererKind {
        RendererKind::Simple
    }
    fn init(&mut self, _device: &Device) -> Result<()> {
        log::info!("renderer init: simple (one draw per sprite)");
        Ok(())
    }
    fn record(&mut self, device: &Device, cb: vk::CommandBuffer, ctx: &RenderContext) {
        let total = ctx.world_sprites.len() + ctx.gui_sprites.len();
        if total == 0 {
            if ctx.verbose {
                log::info!("  SimpleRenderer::record  → 0 sprites (clear-only frame)");
            }
            return;
        }
        if ctx.verbose {
            log::info!(
                "  SimpleRenderer::record  → world={} (sorted), gui={}, meshes={}",
                ctx.world_sprites.len(), ctx.gui_sprites.len(), ctx.meshes.len(),
            );
        }
        // Per-frame state: pipeline + viewport/scissor.
        ctx.pipeline.bind(device, cb, ctx.extent);

        // We rebind the mesh (vertex buffer + index buffer + descriptor)
        // only when the sprite's mesh_idx changes from the last drawn
        // sprite. Grouping sprites in input order minimises rebinds —
        // for the M5 demo the world is tiles-then-buildings, so the
        // mesh changes only once between groups.
        let mut current_mesh_idx: Option<u8> = None;

        // Helper to draw one sprite, handling mesh rebind. Inlined as
        // a closure to keep the loop bodies single-purpose.
        let pipeline = ctx.pipeline;
        let meshes = ctx.meshes;
        let view = ctx.view;
        let draw_one = |sprite: &Sprite, current: &mut Option<u8>| {
            // Clamp mesh_idx defensively — out-of-range falls back to 0,
            // logged once per offending sprite (verbose only).
            let mesh_idx = if (sprite.mesh_idx as usize) < meshes.len() {
                sprite.mesh_idx
            } else {
                if ctx.verbose {
                    log::warn!(
                        "sprite.mesh_idx={} out of range (have {} meshes); using mesh 0",
                        sprite.mesh_idx, meshes.len(),
                    );
                }
                0
            };
            if current.map_or(true, |c| c != mesh_idx) {
                meshes[mesh_idx as usize].bind(device, cb, pipeline);
                *current = Some(mesh_idx);
            }
            let mesh = meshes[mesh_idx as usize];
            let pc = SpritePushConstants::for_sprite(view, sprite);
            pipeline.push_constants(device, cb, &pc);
            unsafe {
                device.logical.cmd_draw_indexed(cb, mesh.index_count, 1, 0, 0, 0);
            }
        };

        // World sprites in sort order. Defensive: if sort_order is
        // empty (no sorter ran) but sprites are present, draw them in
        // input order — better than dropping them.
        if !ctx.world_sprites.is_empty() {
            if ctx.world_sort_order.len() == ctx.world_sprites.len() {
                for (rank, &sprite_idx) in ctx.world_sort_order.iter().enumerate() {
                    let sprite = &ctx.world_sprites[sprite_idx as usize];
                    draw_one(sprite, &mut current_mesh_idx);
                    if ctx.verbose && rank < 3 {
                        log::info!(
                            "    world[rank={rank} idx={sprite_idx}]: \
                             pos=({:.1},{:.1}) iso=({:.1},{:.1})±({:.1},{:.1}) mesh={}",
                            sprite.pos[0], sprite.pos[1],
                            sprite.iso_grid[0], sprite.iso_grid[1],
                            sprite.iso_grid_size[0], sprite.iso_grid_size[1],
                            sprite.mesh_idx,
                        );
                    }
                }
            } else {
                log::warn!(
                    "world_sort_order length ({}) != world_sprites length ({}); \
                     drawing in input order",
                    ctx.world_sort_order.len(), ctx.world_sprites.len(),
                );
                for sprite in ctx.world_sprites {
                    draw_one(sprite, &mut current_mesh_idx);
                }
            }
        }

        // GUI sprites in input order, drawn after world. Iso sorter
        // doesn't touch these.
        for sprite in ctx.gui_sprites {
            draw_one(sprite, &mut current_mesh_idx);
        }
    }
    fn destroy(&mut self, _device: &Device) {}
}

/// M2 stub. Until M3+'s state-change-batching arrives, this is identical
/// to [`SimpleRenderer`] — same draw calls, same output. The CLI flag is
/// preserved so we can validate the wiring end-to-end before the real
/// algorithm lands.
pub struct BatchRenderer {
    inner: SimpleRenderer,
}

impl BatchRenderer {
    pub fn new() -> Self {
        Self {
            inner: SimpleRenderer::new(),
        }
    }
}

impl Default for BatchRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl IRenderer for BatchRenderer {
    fn kind(&self) -> RendererKind {
        RendererKind::Batch
    }
    fn init(&mut self, device: &Device) -> Result<()> {
        log::info!("renderer init: batch (M2 stub — delegates to simple)");
        self.inner.init(device)
    }
    fn record(&mut self, device: &Device, cb: vk::CommandBuffer, ctx: &RenderContext) {
        self.inner.record(device, cb, ctx);
    }
    fn destroy(&mut self, device: &Device) {
        self.inner.destroy(device);
    }
}

/// M2 stub. Same shape as [`BatchRenderer`] — placeholder until M6 builds
/// the 65k-cap streaming buffer pool and the state-change loop.
pub struct BigBufferRenderer {
    inner: SimpleRenderer,
}

impl BigBufferRenderer {
    pub fn new() -> Self {
        Self {
            inner: SimpleRenderer::new(),
        }
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
    fn init(&mut self, device: &Device) -> Result<()> {
        log::info!("renderer init: bigbuffer (M2 stub — delegates to simple)");
        self.inner.init(device)
    }
    fn record(&mut self, device: &Device, cb: vk::CommandBuffer, ctx: &RenderContext) {
        self.inner.record(device, cb, ctx);
    }
    fn destroy(&mut self, device: &Device) {
        self.inner.destroy(device);
    }
}

fn make_renderer(kind: RendererKind) -> Box<dyn IRenderer> {
    match kind {
        RendererKind::Simple => Box::new(SimpleRenderer::new()),
        RendererKind::Batch => Box::new(BatchRenderer::new()),
        RendererKind::BigBuffer => Box::new(BigBufferRenderer::new()),
    }
}

/// Top-level render orchestrator. Mirrors AS3 `RenderCore`. Owns the
/// strategy renderer, the iso sorter, and the textured-quad pipeline.
pub struct RenderCore {
    pub renderer: Box<dyn IRenderer>,
    pub sorter: Box<dyn sort::ISorter>,
    pub sprite_pipeline: SpritePipeline,
    pub clear_color: [f32; 4],
    /// Scratch buffer reused each frame for the iso bounds passed to
    /// the sorter. Kept here to avoid per-frame allocation.
    pub sort_bounds_scratch: Vec<sort::IsoBounds>,
}

impl RenderCore {
    pub fn new(kind: RendererKind, device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let mut renderer = make_renderer(kind);
        renderer.init(device)?;
        let sprite_pipeline = SpritePipeline::new(device, swapchain)?;
        Ok(Self {
            renderer,
            sorter: Box::new(sort::IsometricRectangleSorter::new()),
            sprite_pipeline,
            // Cornflower blue. Easy to recognise, easy to spot stuck pipelines.
            clear_color: [0.39, 0.58, 0.93, 1.0],
            sort_bounds_scratch: Vec::new(),
        })
    }

    pub fn destroy(&mut self, device: &Device) {
        self.renderer.destroy(device);
        self.sprite_pipeline.destroy(device);
    }
}
