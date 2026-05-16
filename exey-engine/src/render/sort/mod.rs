//! Render-order sorting. Mirrors AS3's `engine/render/sorting/`.
//!
//! M5 ships [`IsometricRectangleSorter`] — a graph-topological sort over
//! iso bounds, faithful to the AS3 algorithm by Exey Panteleev. It
//! correctly orders overlapping iso decor (multi-tile footprints, tall
//! sprites) where naive Y-sorting fails.
//!
//! The sorter operates on logic-space iso bounds, not world or screen
//! coords. Each sprite carries a `(iso_grid, iso_grid_size)` pair; the
//! sorter reads these via the [`IsoSortable`] trait. The output is a
//! permutation of the sprite indices that the renderer iterates in
//! order.
//!
//! Architecture decision: the sorter consumes a slice of bounds (one
//! per sprite, in input order) and returns a `Vec<u32>` of sorted
//! indices. We don't permute the sprite slice in place because:
//!   - sprites are large (32+ bytes); permuting copies them around
//!   - the renderer can index into the original slice from sorted indices cheaply
//!   - the caller often wants to keep the original ordering for game logic
//!
//! The trait [`ISorter`] mirrors AS3's `ISorter` interface; the
//! implementation lives in [`iso_rect`].

pub mod graph;
pub mod iso_rect;

pub use iso_rect::{depth_compare, IsoBounds, IsoSortable, IsometricRectangleSorter};

/// Sort produces a permutation of the input indices: `result[i]` is the
/// sprite that should be drawn at position `i` in the final order.
pub trait ISorter {
    /// Compute a draw order for the given bounds. `bounds[i]` are the
    /// iso bounds for sprite `i`. Returns a permutation of `0..len`
    /// (always — even on degenerate input the sorter returns *some*
    /// order so the renderer can proceed).
    fn sort(&mut self, bounds: &[IsoBounds]) -> Vec<u32>;
}
