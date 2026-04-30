//! [`Engine`] is the Rust equivalent of `ExeyEngineCore`.
//!
//! Owns the Vulkan stack and the [`render::RenderCore`]. The demo binary
//! creates one `Engine` per `winit::Window` and drives it from its
//! `ApplicationHandler`.

use anyhow::Result;
use winit::window::Window;

use crate::gfx::{self, Device, Instance, Swapchain, frame::AcquireResult};
use crate::render::{RenderCore, RendererKind};

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
}

impl Engine {
    pub fn new(window: &Window, config: EngineConfig) -> Result<Self> {
        let instance = Instance::new(window, &config.app_name)?;
        let device = Device::new(&instance)?;
        let size = window.inner_size();
        let swapchain = Swapchain::new(&instance, &device, (size.width, size.height))?;
        let frames = gfx::FramesInFlight::new(&device)?;
        let render = RenderCore::new(config.renderer, &device)?;
        Ok(Self {
            render,
            frames,
            swapchain,
            device,
            instance,
            needs_recreate: false,
        })
    }

    /// Called by the demo when winit reports a resize. We don't recreate
    /// immediately — we set the flag and recreate at the top of `draw_frame`
    /// to avoid recreating in the middle of a frame.
    pub fn on_resize(&mut self, _new_size: (u32, u32)) {
        self.needs_recreate = true;
    }

    pub fn draw_frame(&mut self, window: &Window) -> Result<()> {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            // Minimized — skip the frame entirely. Trying to acquire here
            // gives undefined behaviour on some drivers.
            return Ok(());
        }
        if self.needs_recreate {
            self.recreate_swapchain((size.width, size.height))?;
            self.needs_recreate = false;
        }

        let frame = self.frames.current();
        let image_index = match gfx::frame::acquire(&self.device, &self.swapchain, frame)? {
            AcquireResult::Ok(i) => i,
            AcquireResult::Recreate => {
                self.needs_recreate = true;
                return Ok(());
            }
        };

        // Build per-frame batch (just clear color in M1) and call into the
        // selected renderer. The renderer no-ops in M1; the actual clear is
        // performed by `record_clear`.
        let batch = self.render.build_batch();
        self.render.renderer.render(&batch);

        gfx::frame::record_clear(
            &self.device,
            &self.swapchain,
            frame,
            image_index,
            batch.clear_color,
        )?;

        let needs_recreate = gfx::frame::submit_and_present(
            &self.device,
            &self.swapchain,
            frame,
            image_index,
        )?;
        self.needs_recreate |= needs_recreate;

        self.frames.advance();
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
