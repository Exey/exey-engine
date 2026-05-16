//! Iso-rectangle topological sorter.
//!
//! Faithful Rust port of AS3 `IsometricRectangleSorter` by Exey Panteleev,
//! itself a port of the algorithm described in
//! `IsometricDepthSorting2D.h` (ragcat/game_orcs).
//!
//! ## The problem
//!
//! In an isometric scene, sprites' draw order depends on their *iso
//! depth*, which is well-defined only for *overlapping* sprites. Two
//! non-overlapping sprites can be drawn in any order. Naive y-sorting
//! imposes a total order from this partial relation and produces wrong
//! results when tall decor or multi-tile footprints occlude things they
//! shouldn't.
//!
//! ## The algorithm
//!
//! 1. **Bounds** — each sprite has logic-space bounds
//!    `(iso_x1, iso_y1, iso_x2, iso_y2)` plus derived
//!    `iso_left = iso_x1 - iso_y2`, `iso_right = iso_x2 - iso_y1`.
//!    These map the sprite's logic-space footprint to a 1D screen-x-like
//!    range `[iso_left, iso_right]`. Two sprites overlap in screen-x
//!    iff their ranges overlap.
//!
//! 2. **Sweep** — sort sprites by `iso_left` (left-to-right). Walk in
//!    order, maintaining a set of "active" sprites (whose `iso_right`
//!    we haven't yet passed):
//!    - Evict any active sprite whose `iso_right <= new sprite's
//!      iso_left` (no longer overlaps).
//!    - For each new sprite, find its slot in the depth-sorted active
//!      set (via the depth comparator). Add a graph edge from the
//!      sprite immediately before (must render first) and to the
//!      sprite immediately after (must render after).
//!    - Insert the new sprite into the active set.
//!
//! 3. **Toposort** — run Kahn's on the resulting DAG to produce the
//!    final draw order.
//!
//! ## Why this is correct
//!
//! The depth comparator `depth_compare(a, b)` returns positive iff `a`
//! is in front of `b` along the dominant separating axis. It's only
//! meaningful for overlapping sprites — for disjoint ones, both sides
//! of the `max` are negative and the result is geometric nonsense.
//! The sweep only adds graph edges between sprites that *actually
//! overlap* in screen-x (because they're co-active), so every edge
//! encodes a meaningful constraint. The toposort then threads them
//! into a globally consistent total order.
//!
//! ## Complexity
//!
//! - Initial sort by `iso_left`: `O(n log n)`.
//! - Sweep with sorted-Vec active sets: `O(n × k)` where `k` is the
//!   average active-set size. For typical iso scenes (most sprites
//!   are disjoint in x) `k` is small — tens, not thousands.
//! - Toposort: `O(n + edges)`. Edges are bounded by 2n (each sprite
//!   adds at most 2 edges when inserted).
//!
//! Total expected: `O(n log n + n × k)`. For 1054 sprites with `k ≈ 30`
//! that's roughly 30K comparator calls — sub-millisecond.

use glam::Vec2;
use super::ISorter;
use super::graph::{Graph, topological_sort};

/// Logic-space iso bounds. Cached `iso_left`/`iso_right` so the sweep
/// doesn't recompute them. Index is set by the sorter (it's just the
/// sprite's input position; we copy it here so comparators can break
/// ties stably).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IsoBounds {
    /// Back corner X in logic (grid) space. AS3: `isoX1 = startX - sizeX`.
    pub iso_x1: f32,
    /// Back corner Y in logic space. AS3: `isoY1 = startY - sizeY`.
    pub iso_y1: f32,
    /// Front corner X in logic space. AS3: `isoX2 = startX`.
    pub iso_x2: f32,
    /// Front corner Y in logic space. AS3: `isoY2 = startY`.
    pub iso_y2: f32,
    /// Sprite's input index. Populated by the sorter; callers leave 0.
    pub index: u32,
}

impl IsoBounds {
    /// Compute bounds from the sprite's logic-space front corner and
    /// footprint size. Mirrors AS3 `updateIsometric`.
    pub fn from_grid(grid: Vec2, grid_size: Vec2) -> Self {
        Self {
            iso_x1: grid.x - grid_size.x,
            iso_y1: grid.y - grid_size.y,
            iso_x2: grid.x,
            iso_y2: grid.y,
            index: 0,
        }
    }

    /// Screen-x left edge in logic coords. AS3: `isoX1 - isoY2`.
    #[inline]
    pub fn iso_left(&self) -> f32 { self.iso_x1 - self.iso_y2 }

    /// Screen-x right edge in logic coords. AS3: `isoX2 - isoY1`.
    #[inline]
    pub fn iso_right(&self) -> f32 { self.iso_x2 - self.iso_y1 }
}

/// Sprites implement this to expose their iso bounds to the sorter.
/// We don't store the trait object — the sorter takes a slice of
/// pre-computed bounds — but the trait is here for callers who want
/// to compute bounds from sprite state. The engine's `Sprite` impls
/// this via its `iso_grid` / `iso_grid_size` fields.
pub trait IsoSortable {
    fn iso_bounds(&self) -> IsoBounds;
}

/// AS3 `IsometricDepthCompare`: positive iff `a` is in front of `b`
/// (must render after) for sprites that overlap in screen-x.
///
/// Returns negative/zero/positive in the spirit of `Ordering` but as
/// an `i32` so the index tiebreak is explicit. For unequal bounds, the
/// sign of the result is what matters; ties (`a == b` value-wise)
/// disambiguate by `index` so the sort is stable.
pub fn depth_compare(a: &IsoBounds, b: &IsoBounds) -> i32 {
    // a-front-of-b along the dominant separating axis.
    let a_term = (a.iso_x1 - b.iso_x2).max(a.iso_y1 - b.iso_y2);
    let b_term = (b.iso_x1 - a.iso_x2).max(b.iso_y1 - a.iso_y2);
    if a_term > b_term {
        1
    } else if a_term < b_term {
        -1
    } else if a.index < b.index {
        1
    } else if a.index > b.index {
        -1
    } else {
        0
    }
}

/// Sorted insertion: find the insertion position for `b` in the sorted
/// slice `xs` using `depth_compare`. AS3's `SortedListImproved.lowerBound`.
///
/// Returns the index where inserting `b` keeps the slice sorted. If
/// `b` is in front of all elements, returns `xs.len()`. If behind all,
/// returns 0.
fn lower_bound_by_depth(xs: &[IsoBounds], b: &IsoBounds) -> usize {
    // Binary search: smallest i such that xs[i] is not-less-than b.
    // depth_compare(xs[i], b) >= 0 means xs[i] is in-front or equal.
    let mut lo = 0;
    let mut hi = xs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if depth_compare(&xs[mid], b) < 0 {
            // xs[mid] is behind b; b goes after mid.
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The sorter. Stateless aside from scratch buffers reused across
/// calls to avoid allocating per frame.
pub struct IsometricRectangleSorter {
    /// Reusable buffer for the sorted-by-iso-left bounds.
    by_left_scratch: Vec<IsoBounds>,
    /// Active set ordered by depth. AS3: `currentRectanglesByDepth`.
    active_by_depth: Vec<IsoBounds>,
    /// Active set ordered by iso_right ascending. AS3:
    /// `currentRectanglesByRight`. Used to evict sprites whose
    /// `iso_right` is past in the sweep.
    active_by_right: Vec<IsoBounds>,
}

impl Default for IsometricRectangleSorter {
    fn default() -> Self { Self::new() }
}

impl IsometricRectangleSorter {
    pub fn new() -> Self {
        Self {
            by_left_scratch: Vec::new(),
            active_by_depth: Vec::new(),
            active_by_right: Vec::new(),
        }
    }

    /// Build the graph for the sweep. Pure function over the sorted-by-left
    /// bounds; isolated so it's unit-testable.
    fn build_graph(&mut self, by_left: &[IsoBounds]) -> Graph {
        self.active_by_depth.clear();
        self.active_by_right.clear();
        let mut graph = Graph::with_nodes(by_left.len());

        for rect in by_left {
            // Evict any active sprite whose iso_right is past rect.iso_left.
            // active_by_right is sorted ascending; remove from the front
            // while the first's iso_right <= rect.iso_left.
            while let Some(first) = self.active_by_right.first() {
                if first.iso_right() <= rect.iso_left() {
                    let evicted = self.active_by_right.remove(0);
                    // Also remove from active_by_depth. We find by index
                    // (the bounds' own index field) because the same
                    // sprite is in both lists; identity is its index.
                    if let Some(pos) = self
                        .active_by_depth
                        .iter()
                        .position(|r| r.index == evicted.index)
                    {
                        self.active_by_depth.remove(pos);
                    }
                } else {
                    break;
                }
            }

            // Find rect's slot in the depth-sorted active set.
            let next_idx = lower_bound_by_depth(&self.active_by_depth, rect);

            // Edge from the sprite just before (it renders first).
            if next_idx > 0 {
                let prev = &self.active_by_depth[next_idx - 1];
                graph.add_edge(prev.index, rect.index);
            }
            // Edge to the sprite just after (it renders last).
            if next_idx < self.active_by_depth.len() {
                let next = &self.active_by_depth[next_idx];
                graph.add_edge(rect.index, next.index);
            }

            // Insert rect into both active sets.
            self.active_by_depth.insert(next_idx, *rect);
            // For active_by_right, find the right insertion slot.
            let r = rect.iso_right();
            let rpos = self
                .active_by_right
                .binary_search_by(|x| {
                    let xr = x.iso_right();
                    xr.partial_cmp(&r)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Stable tie-break by index, mirroring AS3.
                        .then_with(|| x.index.cmp(&rect.index))
                })
                .unwrap_or_else(|p| p);
            self.active_by_right.insert(rpos, *rect);
        }

        graph
    }
}

impl ISorter for IsometricRectangleSorter {
    fn sort(&mut self, bounds: &[IsoBounds]) -> Vec<u32> {
        if bounds.is_empty() {
            return Vec::new();
        }
        // Stamp indices and copy into scratch for sorting.
        self.by_left_scratch.clear();
        self.by_left_scratch.reserve(bounds.len());
        for (i, b) in bounds.iter().enumerate() {
            let mut b = *b;
            b.index = i as u32;
            self.by_left_scratch.push(b);
        }
        // AS3 `sortLeftToRight`: sort by iso_left ascending. Stable on
        // ties (slice::sort_by is stable in Rust).
        self.by_left_scratch.sort_by(|a, b| {
            a.iso_left()
                .partial_cmp(&b.iso_left())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Move the sorted slice out so we can call `build_graph`
        // without conflicting `&mut self` and `&self.by_left_scratch`
        // borrows. Restore it afterwards — net cost: two pointer
        // swaps, no allocation.
        let by_left = std::mem::take(&mut self.by_left_scratch);
        let graph = self.build_graph(&by_left);
        self.by_left_scratch = by_left;

        topological_sort(&graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two non-overlapping tiles: any order is fine.
    #[test]
    fn disjoint_pair() {
        let mut s = IsometricRectangleSorter::new();
        let bounds = vec![
            IsoBounds::from_grid(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)),
            IsoBounds::from_grid(Vec2::new(5.0, 5.0), Vec2::new(1.0, 1.0)),
        ];
        let order = s.sort(&bounds);
        assert_eq!(order.len(), 2);
        // Both indices appear exactly once.
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
    }

    /// Two overlapping tiles where (1,1) is in front of (0,0):
    /// (0,0) must render before (1,1).
    #[test]
    fn two_overlapping_tiles_in_a_diagonal_line() {
        let mut s = IsometricRectangleSorter::new();
        let bounds = vec![
            // Tile at logic (0, 0), size 1×1.
            IsoBounds::from_grid(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)),
            // Tile at logic (1, 1), size 1×1 — directly "in front of" (0,0)
            // in iso depth.
            IsoBounds::from_grid(Vec2::new(1.0, 1.0), Vec2::new(1.0, 1.0)),
        ];
        let order = s.sort(&bounds);
        // (0,0) should come first.
        assert_eq!(order, vec![0, 1]);
    }

    /// A 2×2 building covering tiles (0,0)..(1,1) and a 1×1 tile at
    /// (2, 2). The tile at (2,2) is in front of the building; it must
    /// render second.
    #[test]
    fn building_then_front_tile() {
        let mut s = IsometricRectangleSorter::new();
        let bounds = vec![
            // 2×2 building, front corner at (1, 1).
            IsoBounds::from_grid(Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0)),
            // Single tile at (2, 2).
            IsoBounds::from_grid(Vec2::new(2.0, 2.0), Vec2::new(1.0, 1.0)),
        ];
        let order = s.sort(&bounds);
        assert_eq!(order, vec![0, 1], "building (idx 0) must render before front tile (idx 1)");
    }

    /// Reverse input order test: building given second should still
    /// render first.
    #[test]
    fn building_then_front_tile_reversed_input() {
        let mut s = IsometricRectangleSorter::new();
        let bounds = vec![
            // Single tile at (2, 2).
            IsoBounds::from_grid(Vec2::new(2.0, 2.0), Vec2::new(1.0, 1.0)),
            // 2×2 building, front corner at (1, 1).
            IsoBounds::from_grid(Vec2::new(1.0, 1.0), Vec2::new(2.0, 2.0)),
        ];
        let order = s.sort(&bounds);
        // Building was input as index 1; tile as index 0. Building must
        // render first → order starts with 1.
        assert_eq!(order, vec![1, 0]);
    }

    /// Three tiles in a row along the iso diagonal — each in front of
    /// the previous.
    #[test]
    fn diagonal_row_of_three() {
        let mut s = IsometricRectangleSorter::new();
        let bounds = vec![
            IsoBounds::from_grid(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)),
            IsoBounds::from_grid(Vec2::new(1.0, 1.0), Vec2::new(1.0, 1.0)),
            IsoBounds::from_grid(Vec2::new(2.0, 2.0), Vec2::new(1.0, 1.0)),
        ];
        let order = s.sort(&bounds);
        assert_eq!(order, vec![0, 1, 2]);
    }
}
