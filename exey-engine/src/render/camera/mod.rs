//! Cameras. Mirrors the AS3 `render/camera/` hierarchy:
//!
//! ```text
//!   AbstractCamera2D (state: pos, zoom, viewport)
//!         ▲
//!     ┌───┴────┐
//!     │        │
//! Simple   Isometric
//! ```
//!
//! Both [`SimpleCamera2D`] and [`IsometricCamera2D`] implement [`ICamera2D`]
//! and produce a [`ViewTransform`] (a pair of `vec2`s: scale + offset)
//! that the sprite vertex shader uses to map world pixels to clip space.
//!
//! ## Why two near-identical concrete cameras
//!
//! In AS3 the two concrete classes have an identical `getCompoundMatrix3D`
//! body — the type difference is purely declarative ("this camera is for
//! iso-space content, this one is for screen-space content"). The
//! distinction earns its keep elsewhere: `IsometricCamera2D` ships static
//! helpers for logic↔world conversion (now in [`crate::render::iso`]),
//! and a future divergence (e.g. iso-aware viewport culling) lives on
//! the iso camera type.
//!
//! Mirroring this in Rust gives us `SimpleCamera2D` and
//! `IsometricCamera2D` as separate concrete types with the same trait
//! body. The *content* placement convention — "iso-space content uses
//! the iso camera" — lives outside this module (in the demo's scene
//! setup).
//!
//! ## ViewTransform
//!
//! The sprite vertex shader does:
//!
//! ```glsl
//! pixel_pos = local * world_size + world_pos;     // local→world
//! ndc.xy    = pixel_pos * view_scale + view_offset; // world→NDC
//! ```
//!
//! [`ViewTransform`] packs `(view_scale, view_offset)`. The camera builds
//! these once per frame from `(viewport_extent, position, zoom)`.

use glam::Vec2;

/// Pair of `vec2`s pushed to the vertex shader: a scale and an offset
/// that together map world pixels to NDC clip coords. See module docs.
#[derive(Copy, Clone, Debug)]
pub struct ViewTransform {
    pub view_scale: [f32; 2],
    pub view_offset: [f32; 2],
}

/// Shared camera state: position in world pixels, zoom factor, and the
/// viewport (framebuffer extent in pixels). All concrete cameras hold
/// one of these.
#[derive(Copy, Clone, Debug)]
pub struct AbstractCamera2D {
    /// Camera position in world pixels. The world point at `position`
    /// renders at the centre of the viewport. Default: world origin.
    pub position: Vec2,
    /// Linear zoom factor. `1.0` = world pixels = framebuffer pixels.
    /// `> 1.0` zooms in (world looks bigger), `< 1.0` zooms out.
    pub zoom: f32,
    /// Framebuffer extent in pixels — `(width, height)`. The engine
    /// updates this on swapchain (re)creation; the camera reads it to
    /// build the projection. Defaults to `(1, 1)` to avoid divide-by-zero.
    pub viewport: Vec2,
}

impl Default for AbstractCamera2D {
    fn default() -> Self {
        Self {
            position: Vec2::ZERO,
            zoom: 1.0,
            viewport: Vec2::new(1.0, 1.0),
        }
    }
}

impl AbstractCamera2D {
    /// Build the world→clip transform from the current state.
    ///
    /// Derivation: we want world point `p` to land at NDC point
    /// `(p - cam.pos) * (2/extent) * zoom`. Decomposing into a scale and
    /// an offset gives:
    ///
    /// ```text
    ///   view_scale  = (2/extent.w * zoom, 2/extent.h * zoom)
    ///   view_offset = (-cam.pos.x * view_scale.x, -cam.pos.y * view_scale.y)
    /// ```
    ///
    /// (No `(-1, -1)` baseline like in M3: the `-cam.pos * scale` term
    /// already centers the camera at NDC origin. M3's "pixel coords →
    /// NDC top-left" convention is recovered by `cam.pos = -extent/2`
    /// and `zoom = 1` — i.e. a default `SimpleCamera2D` with no pan.)
    pub fn view_transform(&self) -> ViewTransform {
        let scale_x = 2.0 / self.viewport.x.max(1.0) * self.zoom;
        let scale_y = 2.0 / self.viewport.y.max(1.0) * self.zoom;
        ViewTransform {
            view_scale: [scale_x, scale_y],
            view_offset: [-self.position.x * scale_x, -self.position.y * scale_y],
        }
    }
}

/// Behaviour shared by every concrete camera. Currently small — both
/// concrete impls produce the same view transform — but the trait gives
/// the renderer one type to talk to and leaves room for divergence.
///
/// Mirrors AS3 `ICamera2D`. The AS3 method `getCompoundMatrix3D(transform)`
/// returned a 4×4 matrix multiplying the sprite's transform by the camera's
/// projection. We've split that into "view transform" (camera-only, here)
/// and "world transform" (per-sprite, written into the push constant by
/// [`crate::render::SimpleRenderer`]).
pub trait ICamera2D {
    /// Read-only access to the shared camera state.
    fn abstract_state(&self) -> &AbstractCamera2D;
    /// Mutable access — used by callers that pan/zoom.
    fn abstract_state_mut(&mut self) -> &mut AbstractCamera2D;
    /// Build the world→clip transform. Default impl is "delegate to the
    /// abstract state"; concrete cameras may override if they ever
    /// diverge (none do today).
    fn view_transform(&self) -> ViewTransform {
        self.abstract_state().view_transform()
    }

    // Convenience accessors so callers don't always reach through
    // `abstract_state()`. Have default bodies; overriding them is unusual.
    fn position(&self) -> Vec2 {
        self.abstract_state().position
    }
    fn set_position(&mut self, pos: Vec2) {
        self.abstract_state_mut().position = pos;
    }
    fn zoom(&self) -> f32 {
        self.abstract_state().zoom
    }
    fn set_zoom(&mut self, zoom: f32) {
        self.abstract_state_mut().zoom = zoom;
    }
    fn set_viewport(&mut self, viewport: Vec2) {
        self.abstract_state_mut().viewport = viewport;
    }
}

/// Screen-space camera. Mirrors AS3 `SimpleCamera2D`. Used for content
/// that isn't iso-projected — GUI, HUD, debug overlays.
///
/// In M4 this is functionally identical to [`IsometricCamera2D`]. The
/// distinction is what content the demo *places* through it: iso tiles
/// go through `IsometricCamera2D`; screen-space sprites would go through
/// `SimpleCamera2D`. The renderer doesn't care which.
#[derive(Copy, Clone, Debug, Default)]
pub struct SimpleCamera2D {
    pub state: AbstractCamera2D,
}

impl SimpleCamera2D {
    pub fn new() -> Self { Self::default() }
}

impl ICamera2D for SimpleCamera2D {
    fn abstract_state(&self) -> &AbstractCamera2D { &self.state }
    fn abstract_state_mut(&mut self) -> &mut AbstractCamera2D { &mut self.state }
}

/// Iso-projected camera. Mirrors AS3 `IsometricCamera2D`. Used for the
/// world: tiles, characters, decor — anything placed via
/// [`crate::render::iso::logic_to_world`].
///
/// AS3 hangs four static helpers off this class
/// (`convertLogicToWorld`/`convertWorldToLogic`/`convertWorldToScreen`/
/// `convertScreenToWorld`). We've moved the first two to the
/// [`crate::render::iso`] module (free functions are easier to test and
/// don't need a camera instance to call). The other two — world↔screen
/// — are the camera's job and are accessible via [`Self::world_to_screen`]
/// and [`Self::screen_to_world`].
#[derive(Copy, Clone, Debug, Default)]
pub struct IsometricCamera2D {
    pub state: AbstractCamera2D,
}

impl IsometricCamera2D {
    pub fn new() -> Self { Self::default() }

    /// Map a world-pixel position to screen-pixel coords (origin at the
    /// top-left of the viewport, +Y down). Useful for hit testing,
    /// HUD-anchoring, debug overlays.
    pub fn world_to_screen(&self, world: Vec2) -> Vec2 {
        let s = &self.state;
        // `(world - cam.pos) * zoom` in centred-viewport coords; add
        // half-extent to shift origin from centre to top-left.
        let centred = (world - s.position) * s.zoom;
        centred + s.viewport * 0.5
    }

    /// Inverse of [`Self::world_to_screen`].
    pub fn screen_to_world(&self, screen: Vec2) -> Vec2 {
        let s = &self.state;
        let centred = screen - s.viewport * 0.5;
        centred / s.zoom + s.position
    }
}

impl ICamera2D for IsometricCamera2D {
    fn abstract_state(&self) -> &AbstractCamera2D { &self.state }
    fn abstract_state_mut(&mut self) -> &mut AbstractCamera2D { &mut self.state }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_transform_default_centres_origin() {
        // A default camera (pos=0, zoom=1, viewport=(1280, 720)) puts
        // world origin at NDC origin.
        let mut s = AbstractCamera2D::default();
        s.viewport = Vec2::new(1280.0, 720.0);
        let vt = s.view_transform();
        // World origin → NDC: (0,0) * scale + offset = offset, and
        // offset is -pos * scale = 0 since pos=0. So world (0,0) maps
        // to clip (0, 0) — the centre of the screen.
        assert_eq!(vt.view_offset, [0.0, 0.0]);
    }

    #[test]
    fn world_to_screen_round_trip() {
        let mut cam = IsometricCamera2D::new();
        cam.state.viewport = Vec2::new(1280.0, 720.0);
        cam.state.position = Vec2::new(100.0, 50.0);
        cam.state.zoom = 0.75;

        let p = Vec2::new(42.0, -7.0);
        let s = cam.world_to_screen(p);
        let p2 = cam.screen_to_world(s);
        assert!((p - p2).length() < 1e-3);
    }
}
