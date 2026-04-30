//! Render layer. Mirrors the AS3 `render/` package.
//!
//! - [`RenderCore`] holds three layers (background / world / gui), the cameras,
//!   and the chosen [`IRenderer`] strategy. Per frame it sorts the world layer
//!   (via [`sort::ISorter`]), builds a [`RenderBatchData`], and hands that to
//!   the renderer.
//! - [`IRenderer`] is the strategy trait. M1 wires up the kinds; the real
//!   implementations land in M3 (Simple), M4–M5 (Iso math + Sort), M6 (BigBuffer).
//!
//! The `Kind` enum exists so the demo can pick a renderer at startup via a CLI
//! flag (`--renderer simple|batch|bigbuffer`).

pub mod camera;
pub mod sort;

use crate::gfx::Device;

/// Picks which IRenderer implementation to construct. Mirrors the comments
/// in the AS3 `RCatEngineCore.context3dCreated_handler` where the user
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

/// The data passed from `RenderCore` to an `IRenderer` each frame.
/// In later milestones this will hold a Vec of render-op instances; for M1
/// it's just the clear color so we have a non-empty interface.
#[derive(Default)]
pub struct RenderBatchData {
    pub clear_color: [f32; 4],
}

/// Trait equivalent of AS3 `IRenderer`. Method names match the original
/// (`init`, `render`) so future readers cross-referencing the AS3 sources
/// will recognize the contract.
pub trait IRenderer {
    fn kind(&self) -> RendererKind;
    /// Called once after device creation. Late milestones use this to build
    /// pipelines, allocate persistent buffers, etc.
    fn init(&mut self, device: &Device) -> anyhow::Result<()>;
    /// Called per frame. M1 implementation is a no-op — the actual clear is
    /// performed in `gfx::frame::record_clear` regardless of renderer.
    fn render(&mut self, batch: &RenderBatchData);
    /// Called once before the device is destroyed.
    fn destroy(&mut self, device: &Device);
}

/// M1 placeholder — all three kinds share this empty body. The real ones
/// land in their own files in M3/M6.
pub struct StubRenderer {
    kind: RendererKind,
}

impl StubRenderer {
    pub fn new(kind: RendererKind) -> Self {
        Self { kind }
    }
}

impl IRenderer for StubRenderer {
    fn kind(&self) -> RendererKind {
        self.kind
    }
    fn init(&mut self, _device: &Device) -> anyhow::Result<()> {
        log::info!("renderer init: {} (M1 stub — clear-only)", self.kind.as_str());
        Ok(())
    }
    fn render(&mut self, _batch: &RenderBatchData) {
        // M1: clear-only. Future milestones populate this.
    }
    fn destroy(&mut self, _device: &Device) {}
}

/// Top-level render orchestrator. M1 holds only what's needed to wire up
/// the renderer-selection flag; cameras and sort live here in M4+.
pub struct RenderCore {
    pub renderer: Box<dyn IRenderer>,
    pub clear_color: [f32; 4],
}

impl RenderCore {
    pub fn new(kind: RendererKind, device: &Device) -> anyhow::Result<Self> {
        let mut renderer: Box<dyn IRenderer> = Box::new(StubRenderer::new(kind));
        renderer.init(device)?;
        Ok(Self {
            renderer,
            // Cornflower blue. Easy to recognise, easy to spot stuck pipelines.
            clear_color: [0.39, 0.58, 0.93, 1.0],
        })
    }

    pub fn destroy(&mut self, device: &Device) {
        self.renderer.destroy(device);
    }

    pub fn build_batch(&self) -> RenderBatchData {
        RenderBatchData {
            clear_color: self.clear_color,
        }
    }
}
