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
    /// * `camera`        — the view to render through.
    /// * `meshes`        — slice of meshes; sprites' `mesh_idx` indexes into this.
    /// * `world_sprites` — iso-positioned sprites; sorted by the iso sorter.
    /// * `gui_sprites`   — overlay sprites; rendered after world in input order.
    ///
    /// Empty slices are fine; an empty world+gui draws a clear-only frame.
    pub fn draw_frame(
        &mut self,
        window: &Window,
        camera: &dyn crate::render::ICamera2D,
        meshes: &[&crate::render::SpriteMesh],
        world_sprites: &[Sprite],
        gui_sprites: &[Sprite],
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
        let view = camera.view_transform();

        // Compute iso bounds for the world sprites and run the sorter.
        // For an empty world we skip the sort entirely. Bounds buffer is
        // reused across frames.
        let world_sort_order: Vec<u32> = if world_sprites.is_empty() {
            Vec::new()
        } else {
            let bounds = &mut self.render.sort_bounds_scratch;
            bounds.clear();
            bounds.reserve(world_sprites.len());
            for s in world_sprites {
                use crate::render::sort::IsoSortable;
                bounds.push(s.iso_bounds());
            }
            self.render.sorter.sort(bounds)
        };

        let renderer = &mut self.render.renderer;

        if verbose {
            log::info!(
                "frame {}: extent={}x{}  image_index={}  world={} gui={} meshes={} renderer={:?}",
                self.frame_index,
                extent.width,
                extent.height,
                image_index,
                world_sprites.len(),
                gui_sprites.len(),
                meshes.len(),
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
            meshes,
            extent,
            view,
            world_sprites,
            world_sort_order: &world_sort_order,
            gui_sprites,
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
