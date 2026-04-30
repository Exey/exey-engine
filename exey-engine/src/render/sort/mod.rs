//! Render-order sorting. Two kinds in the AS3 engine:
//!   - `ScreenYSorter`              — sort by `y` (cheap, wrong for overlapping iso)
//!   - `IsometricRectangleSorter`   — graph-topological sort over iso bounds (correct)
//!
//! Real implementations land in M5. The trait sits here so RenderCore can
//! hold a `Box<dyn ISorter>` from the start.

pub trait ISorter {
    /// Reorder the indices in-place so that drawing them in the new order
    /// produces correct depth.
    fn sort(&mut self, indices: &mut [u32]);
}
