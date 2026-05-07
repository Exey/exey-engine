//! Isometric coordinate conversions.
//!
//! ExeyEngine uses the standard 2:1 "diamond" isometric projection:
//! tiles are drawn as diamonds twice as wide as they are tall. Three
//! coordinate spaces are involved:
//!
//! * **Logic (grid) space** — discrete or fractional `(grid_x, grid_y)`
//!   integer-ish tile coordinates. (5, 3) means "tile at column 5, row 3".
//!   The world is laid out on this grid, and gameplay code (pathfinding,
//!   placement, occupancy) lives here.
//! * **World (pixel) space** — `(world_x, world_y)` in pixels, +Y down,
//!   origin at logic (0, 0). This is where sprites are positioned, where
//!   the camera lives, and where the iso projection actually applies.
//! * **Screen (clip) space** — `(ndc_x, ndc_y)` in `[-1, 1]`, what the
//!   GPU rasterizes against. The camera produces a transform that maps
//!   world → clip; the vertex shader applies it.
//!
//! This module covers logic ↔ world. World ↔ screen is the camera's job
//! ([`crate::render::camera`]).
//!
//! ## The math
//!
//! Mirrors the AS3 `IsoUtil.spaceToScreen` / `screenToSpace` formulas as
//! used by `IsometricCamera2D.convertLogicToWorld`. With tile size
//! `(tile_w, tile_h)`:
//!
//! ```text
//!   world_x = tile_h * (gx - gy)
//!   world_y = tile_h * (gx + gy) / 2
//!
//!   gx = (world_x / tile_h + 2 * world_y / tile_h) / 2
//!   gy = (2 * world_y / tile_h - world_x / tile_h) / 2
//! ```
//!
//! Note that the AS3 source scales by `tileSize.y` (height) on both axes
//! before the iso transform. That's the "2:1 iso" convention: a tile is
//! drawn `2*tile_h` wide and `tile_h` tall. We pass `tile_w` for clarity
//! but only `tile_h` participates in the iso math; `tile_w` is used by
//! the demo for sizing the sprite quad.

use glam::Vec2;

/// Convert grid (logic) coordinates to world (pixel) coordinates.
///
/// `tile_h` is the tile *height* in pixels — the iso diamond's vertical
/// half-extent times two. The width is implicit: `tile_w = 2 * tile_h` for
/// canonical 2:1 iso. We don't take `tile_w` here because it doesn't
/// participate in the projection — only in sprite *sizing*, which is the
/// caller's concern.
pub fn logic_to_world(grid: Vec2, tile_h: f32) -> Vec2 {
    Vec2::new(
        tile_h * (grid.x - grid.y),
        tile_h * (grid.x + grid.y) * 0.5,
    )
}

/// Inverse of [`logic_to_world`]. Returns fractional grid coords; floor
/// for a tile lookup, fract for sub-tile position.
pub fn world_to_logic(world: Vec2, tile_h: f32) -> Vec2 {
    let inv = world / tile_h;
    Vec2::new(
        (inv.x + 2.0 * inv.y) * 0.5,
        (2.0 * inv.y - inv.x) * 0.5,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_logic_to_world_to_logic() {
        let tile_h = 32.0;
        for gx in 0..10 {
            for gy in 0..10 {
                let g = Vec2::new(gx as f32, gy as f32);
                let w = logic_to_world(g, tile_h);
                let g2 = world_to_logic(w, tile_h);
                assert!((g - g2).length() < 1e-3, "g={g:?} -> w={w:?} -> g2={g2:?}");
            }
        }
    }

    #[test]
    fn diamond_corners() {
        let tile_h = 32.0;
        // Four corners of a 2:1 iso tile centered at logic (0, 0).
        // The tile is drawn 2*tile_h wide × tile_h tall, with its
        // logic origin at the top corner of the diamond... actually
        // the AS3 convention places logic origin at the top corner,
        // so a unit move in +grid_x goes "down-right" (+world_x,
        // +world_y/2), and +grid_y goes "down-left" (-world_x,
        // +world_y/2). Verify:
        assert_eq!(logic_to_world(Vec2::ZERO, tile_h), Vec2::ZERO);
        assert_eq!(logic_to_world(Vec2::new(1.0, 0.0), tile_h), Vec2::new(32.0, 16.0));
        assert_eq!(logic_to_world(Vec2::new(0.0, 1.0), tile_h), Vec2::new(-32.0, 16.0));
        assert_eq!(logic_to_world(Vec2::new(1.0, 1.0), tile_h), Vec2::new(0.0, 32.0));
    }
}
