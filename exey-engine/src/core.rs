//! [`Engine`] is the Rust equivalent of `ExeyEngineCore`.
//!
//! Owns the Vulkan stack and the [`render::RenderCore`]. The demo binary
//! creates one `Engine` per `winit::Window` and drives it from its
//! `ApplicationHandler`.

use anyhow::Result;
use winit::window::Window;

use crate::gfx::{self, Device, Instance, Swapchain, frame::AcquireResult};
use crate::render::{RenderCore, RendererKind, Sprite};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub app_name: String,
    pub renderer: RendererKind,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            app_name: "ExeyEngine".to_string(),
            renderer: RendererKind::default(),
        }
    }
}

pub struct Engine {
    // Drop order matters: render core (uses device) → frames → swapchain → device → instance.
    pub render: RenderCore,
    pub frames: gfx::FramesInFlight,
    pub swapchain: Swapchain,
    pub device: Device,
    pub instance: Instance,
    /// Suspended state — we set this when the swapchain reports OUT_OF_DATE
    /// or when the window is minimized (extent 0×0). The next non-zero
    /// resize event recreates the swapchain.
    pub needs_recreate: bool,
    /// Monotonic frame counter. Used by the diagnostic logging path so we
    /// log the first few frames in detail (to confirm wiring) and then
    /// stay quiet to avoid console spam.
    pub frame_index: u64,
}

impl Engine {
    pub fn new(window: &Window, config: EngineConfig) -> Result<Self> {
        let instance = Instance::new(window, &config.app_name)?;
        let device = Device::new(&instance)?;
        let size = window.inner_size();
        let swapchain = Swapchain::new(&instance, &device, (size.width, size.height))?;
        let frames = gfx::FramesInFlight::new(&device)?;
        let render = RenderCore::new(config.renderer, &device, &swapchain)?;
        Ok(Self {
            render,
            frames,
            swapchain,
            device,
            instance,
            needs_recreate: false,
            frame_index: 0,
        })
    }

    /// Called by the demo when winit reports a resize. We don't recreate
    /// immediately — we set the flag and recreate at the top of `draw_frame`
    /// to avoid recreating in the middle of a frame.
    pub fn on_resize(&mut self, _new_size: (u32, u32)) {
        self.needs_recreate = true;
    }

    /// Draw one frame.
    ///
    /// * `mesh`    — shared unit-quad geometry + texture descriptor (M3
    ///   demos own one of these and pass it in; M5+ may pass several).
    /// * `camera`  — the camera whose view transform applies to this frame.
    ///   The engine reads its position/zoom/viewport and computes the
    ///   `(view_scale, view_offset)` push-constant fields from them.
    /// * `sprites` — CPU-side state for sprites to draw this frame. Empty
    ///   slice = clear-only frame.
    pub fn draw_frame(
        &mut self,
        window: &Window,
        camera: &dyn crate::render::ICamera2D,
        mesh: &crate::render::SpriteMesh,
        sprites: &[Sprite],
    ) -> Result<()> {
        // Log only the first few frames in detail so we can confirm wiring
        // end-to-end without spamming the console at ~vsync rate. After
        // VERBOSE_FRAMES, this method is silent on the happy path.
        const VERBOSE_FRAMES: u64 = 3;
        let verbose = self.frame_index < VERBOSE_FRAMES;

        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            // Minimized — skip the frame entirely. Trying to acquire here
            // gives undefined behaviour on some drivers.
            if verbose {
                log::info!(
                    "frame {}: skipped (window minimized, extent {}x{})",
                    self.frame_index, size.width, size.height
                );
            }
            return Ok(());
        }
        if self.needs_recreate {
            log::info!(
                "frame {}: recreating swapchain ({}x{})",
                self.frame_index, size.width, size.height
            );
            self.recreate_swapchain((size.width, size.height))?;
            self.needs_recreate = false;
        }

        let frame = self.frames.current();
        let image_index = match gfx::frame::acquire(&self.device, &self.swapchain, frame)? {
            AcquireResult::Ok(i) => i,
            AcquireResult::Recreate => {
                if verbose {
                    log::info!("frame {}: acquire said recreate", self.frame_index);
                }
                self.needs_recreate = true;
                return Ok(());
            }
        };

        // Build the per-frame context. The borrow checker needs disjoint
        // field accesses here: the closure captures `&mut self.render.renderer`
        // and the context borrows `&self.render.sprite_pipeline` — both
        // through `self.render`, so we name them as separate locals first.
        let extent = self.swapchain.extent;
        let device = &self.device;
        let pipeline = &self.render.sprite_pipeline;
        let clear = self.render.clear_color;
        let renderer = &mut self.render.renderer;
        // Read the camera's view transform once per frame. The renderer
        // copies it into every push constant. We could cache this on the
        // engine when the camera changes, but at 1024 sprites × ~20 ns
        // per copy it's cheaper than the bookkeeping.
        let view = camera.view_transform();

        if verbose {
            log::info!(
                "frame {}: extent={}x{}  image_index={}  sprites={}  renderer={:?}",
                self.frame_index,
                extent.width,
                extent.height,
                image_index,
                sprites.len(),
                renderer.kind(),
            );
            log::info!(
                "  view: scale=({:.4},{:.4})  offset=({:.4},{:.4})",
                view.view_scale[0], view.view_scale[1],
                view.view_offset[0], view.view_offset[1],
            );
        }

        let ctx = crate::render::RenderContext {
            pipeline,
            mesh,
            extent,
            view,
            sprites,
            clear_color: clear,
            verbose,
        };

        gfx::frame::record_frame(
            device,
            &self.swapchain,
            frame,
            image_index,
            clear,
            |cb| {
                renderer.record(device, cb, &ctx);
            },
        )?;

        let needs_recreate = gfx::frame::submit_and_present(
            &self.device,
            &self.swapchain,
            frame,
            image_index,
        )?;
        self.needs_recreate |= needs_recreate;

        self.frames.advance();
        self.frame_index += 1;
        Ok(())
    }

    fn recreate_swapchain(&mut self, size: (u32, u32)) -> Result<()> {
        // Spec: must wait for all GPU work to complete before swapchain teardown.
        self.device.wait_idle();
        self.swapchain.recreate(&self.instance, &self.device, size)?;
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Wait for the GPU before tearing anything down — otherwise we'd
        // free resources that are still in flight.
        self.device.wait_idle();
        self.render.destroy(&self.device);
        self.frames.destroy(&self.device);
        self.swapchain.destroy(&self.device);
        // Device / Instance drop themselves via their `Drop` impls.
    }
}
