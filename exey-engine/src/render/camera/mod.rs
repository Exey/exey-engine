//! Cameras. Two concrete kinds in the original AS3:
//!   - `SimpleCamera2D`     — screen-space orthographic, used for GUI
//!   - `IsometricCamera2D`  — iso-projected orthographic, used for the world
//!
//! Both implement the `ICamera2D` interface (here: [`ICamera2D`] trait).
//! Real bodies arrive in M4. M1 just has the trait so the rest of the
//! engine compiles cleanly.

use glam::Mat4;

pub trait ICamera2D {
    fn view_projection(&self) -> Mat4;
    fn screen_size(&self) -> (f32, f32);
}
