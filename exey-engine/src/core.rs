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
        let render = RenderCore::new(config.renderer, &instance, &device, &swapchain)?;
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
    /// * `dt`            — seconds since the last frame; drives the M7
    ///                     animation tick. Pass `0.0` to advance no
    ///                     animations (useful on the very first frame
    ///                     when the clock hasn't been ticked yet).
    /// * `camera`        — the view to render through.
    /// * `meshes`        — slice of meshes; sprites' `mesh_idx` indexes into this.
    /// * `world_sprites` — iso-positioned sprites; sorted by the iso sorter.
    ///                     Mutable because the M7 animation pass writes
    ///                     `uv_offset` / `uv_scale` on sprites whose
    ///                     `anim` is `Some`. Static sprites are not touched.
    /// * `gui_sprites`   — overlay sprites; rendered after world in input order.
    ///                     Also mutable for the same reason (animated UI).
    ///
    /// Empty slices are fine; an empty world+gui draws a clear-only frame.
    pub fn draw_frame(
        &mut self,
        window: &Window,
        dt: f32,
        camera: &dyn crate::render::ICamera2D,
        meshes: &[&crate::render::SpriteMesh],
        world_sprites: &mut [Sprite],
        gui_sprites: &mut [Sprite],
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

        // M7 — animation tick. Walk both sprite slices once and, for each
        // sprite carrying an AnimationState, advance time and resolve the
        // current frame's UV. Static sprites (anim == None) are not
        // touched, so per-frame cost is O(animated), not O(total).
        //
        // We run the tick *before* iso bounds collection because the new
        // UV doesn't change iso footprint (sort math reads iso_grid /
        // iso_grid_size, which the tick doesn't touch). The order is
        // chosen for readability, not correctness.
        if !self.render.frame_strips.is_empty() {
            let strips = &self.render.frame_strips;
            let mut animated = 0u32;
            tick_animations(world_sprites, strips, dt, &mut animated);
            tick_animations(gui_sprites, strips, dt, &mut animated);
            if verbose {
                log::info!(
                    "frame {}: M7 animation tick — dt={:.5}s strips={} animated={}",
                    self.frame_index, dt, strips.len(), animated,
                );
            }
        }

        // Compute iso bounds for the world sprites and run the sorter.
        // For an empty world we skip the sort entirely. Bounds buffer is
        // reused across frames.
        let world_sort_order: Vec<u32> = if world_sprites.is_empty() {
            Vec::new()
        } else {
            let bounds = &mut self.render.sort_bounds_scratch;
            bounds.clear();
            bounds.reserve(world_sprites.len());
            for s in world_sprites.iter() {
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

        // Reborrow `&mut [Sprite]` → `&[Sprite]` for RenderContext. The
        // engine has already done its mutating pass (the M7 animation
        // tick above); from here on the renderer only reads.
        let ctx = crate::render::RenderContext {
            pipeline,
            meshes,
            extent,
            view,
            world_sprites: &*world_sprites,
            world_sort_order: &world_sort_order,
            gui_sprites: &*gui_sprites,
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

/// Walk a sprite slice once, advancing `time` and writing UV for any
/// sprite that holds an [`crate::draw::AnimationState`]. Called twice
/// per `draw_frame` (world + gui). Lives at module scope rather than
/// as a method so it can take `&mut [Sprite]` and `&[FrameStrip]`
/// borrowed from disjoint pieces of `self` without fighting the borrow
/// checker.
///
/// Defensive: an out-of-range `strip_id` falls back to the first strip
/// and logs once per offender. Never panics on bad data.
fn tick_animations(
    sprites: &mut [Sprite],
    strips: &[crate::draw::FrameStrip],
    dt: f32,
    animated_count: &mut u32,
) {
    if strips.is_empty() {
        return;
    }
    for s in sprites.iter_mut() {
        // Snapshot the strip lookup keys out of the (mut-borrowed) anim
        // state so the borrow ends before we write to sibling fields of
        // the Sprite. NLL would handle the overlapping disjoint-field
        // case, but spelling it out keeps the loop body obvious.
        let (strip_id, time) = match s.anim.as_mut() {
            None => continue,
            Some(anim) => {
                if !anim.paused {
                    anim.time += dt;
                }
                (anim.strip_id, anim.time)
            }
        };
        *animated_count += 1;
        let strip = if (strip_id as usize) < strips.len() {
            &strips[strip_id as usize]
        } else {
            log::warn!(
                "tick_animations: sprite has strip_id={} but only {} strip(s) registered; \
                 falling back to strip 0",
                strip_id, strips.len(),
            );
            &strips[0]
        };
        let idx = strip.frame_index_at(time);
        s.uv_offset = strip.uv_offset_for(idx);
        s.uv_scale = strip.frame_uv_scale;
    }
}
