//! M7 — 2D frame animation.
//!
//! Mirrors AS3 `exey.engine.draw.animation.*`. The original used a
//! `FrameManager` that held a flat array of `FrameData` and pumped time
//! through `IRenderable` instances. Two design notes from porting:
//!
//! 1. **Strips are shared; playback state is per-sprite.** A `FrameStrip`
//!    is pure metadata: where in an atlas the frames live, how many, how
//!    fast they play, how they loop. N sprites can share one strip and pay
//!    only [`AnimationState`] storage per sprite (~12 bytes).
//!
//! 2. **The engine ticks animations inside `draw_frame`.** It owns the
//!    `Vec<FrameStrip>` registry (built by the demo at startup, then
//!    immutable) and walks `&mut [Sprite]` once per frame to resolve the
//!    current frame index → `uv_offset` / `uv_scale`. Renderers stay
//!    animation-agnostic; they just read whatever UV the sprite holds.
//!
//! ## Atlas layout assumption
//!
//! A `FrameStrip` describes a *contiguous horizontal run* of frames in an
//! atlas: `uv_offset` is the top-left of frame 0, `frame_uv_scale` is the
//! size of *one* frame in UV space, and frame `i` lives at
//! `uv = uv_offset + (i * frame_uv_scale.x, 0)`. That covers the wolf
//! sheet (rows of frames, picked one row at a time) and the font atlas
//! pattern (one horizontal strip). A future strip that wraps to multiple
//! rows would need either a different layout struct or a 2D step vector;
//! M7 doesn't need that.
//!
//! ## Loop modes
//!
//! [`LoopMode::Loop`] wraps `time` modulo `frame_count / fps`.
//! [`LoopMode::Once`] clamps to the last frame after one pass — the
//! AS3 `playOnce` flag.
//! [`LoopMode::PingPong`] bounces 0..N-1..0; the period is
//! `2 * (frame_count - 1) / fps`. Useful for breathing-style idles where
//! frame 0 and frame N-1 are the rest poses and the middle frames are
//! the squash.

/// How a [`FrameStrip`] resolves time → frame index past one full cycle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopMode {
    /// Wrap forever. The default. `time` past the end folds back to 0.
    Loop,
    /// Play once, then hold the last frame indefinitely. Mirrors AS3's
    /// `playOnce` flag on `FrameManager`.
    Once,
    /// 0 → N-1 → 0 → N-1 → … Bounces back and forth.
    PingPong,
}

impl Default for LoopMode {
    fn default() -> Self {
        Self::Loop
    }
}

/// Atlas metadata + timing for one animation. Shared by all sprites that
/// play this animation.
///
/// `uv_offset` and `frame_uv_scale` are in normalized texture space
/// (`[0..1]`). For an atlas that's a horizontal strip of `frame_count`
/// equal cells in one row of an `atlas_cells_x × atlas_cells_y` grid,
/// build with [`FrameStrip::from_grid_row`].
#[derive(Copy, Clone, Debug)]
pub struct FrameStrip {
    /// Top-left of frame 0 in UV space.
    pub uv_offset: [f32; 2],
    /// Size of one frame in UV space. Frames step by `(frame_uv_scale.x, 0)`.
    pub frame_uv_scale: [f32; 2],
    /// Number of frames in the strip. Must be ≥ 1.
    pub frame_count: u32,
    /// Frames per second. 0.0 means "hold frame 0" (a paused sprite).
    pub fps: f32,
    pub loop_mode: LoopMode,
}

impl FrameStrip {
    /// Build a strip for a contiguous run of `frame_count` cells in row
    /// `row` of an `atlas_cells_x × atlas_cells_y` grid, starting at
    /// column `col0`.
    ///
    /// Example: the wolf sheet is 15×16 cells of 64×64 px. Row 9 starts
    /// the SW idle; the first 2 frames are:
    ///
    /// ```ignore
    /// FrameStrip::from_grid_row(15, 16, /*row=*/9, /*col0=*/0, /*count=*/2, 4.0, LoopMode::Loop)
    /// ```
    pub fn from_grid_row(
        atlas_cells_x: u32,
        atlas_cells_y: u32,
        row: u32,
        col0: u32,
        frame_count: u32,
        fps: f32,
        loop_mode: LoopMode,
    ) -> Self {
        assert!(frame_count >= 1, "FrameStrip needs at least 1 frame");
        assert!(
            col0 + frame_count <= atlas_cells_x,
            "FrameStrip::from_grid_row: row {} col0={} count={} runs off the {}-cell-wide atlas",
            row, col0, frame_count, atlas_cells_x,
        );
        assert!(
            row < atlas_cells_y,
            "FrameStrip::from_grid_row: row {} is past the {}-row atlas",
            row, atlas_cells_y,
        );
        let cw = 1.0 / atlas_cells_x as f32;
        let ch = 1.0 / atlas_cells_y as f32;
        Self {
            uv_offset: [col0 as f32 * cw, row as f32 * ch],
            frame_uv_scale: [cw, ch],
            frame_count,
            fps,
            loop_mode,
        }
    }

    /// Period of one full cycle, in seconds. For `Once` this is the
    /// total play time before the strip clamps to its last frame.
    /// Returns `f32::INFINITY` if `fps == 0`.
    pub fn cycle_seconds(&self) -> f32 {
        if self.fps <= 0.0 {
            return f32::INFINITY;
        }
        match self.loop_mode {
            LoopMode::Loop | LoopMode::Once => self.frame_count as f32 / self.fps,
            // PingPong of N frames visits N + (N-2) = 2N-2 frame slots
            // before repeating. (For N=1 the period is 0 — degenerate
            // single-frame animation; we clamp to a tiny value to avoid
            // a divide-by-zero in `frame_index_at`.)
            LoopMode::PingPong => {
                if self.frame_count <= 1 {
                    f32::INFINITY
                } else {
                    (2 * (self.frame_count - 1)) as f32 / self.fps
                }
            }
        }
    }

    /// Resolve `time` (seconds) into a frame index in `0..frame_count`.
    ///
    /// Defensive against degenerate inputs:
    /// * `frame_count == 1` → always 0.
    /// * `fps <= 0` → always 0 (the strip is effectively paused).
    /// * `time` negative → treated as 0 (no rewind semantics in M7).
    pub fn frame_index_at(&self, time: f32) -> u32 {
        if self.frame_count <= 1 || self.fps <= 0.0 {
            return 0;
        }
        let t = time.max(0.0);
        let raw = t * self.fps; // floating-point frame number
        match self.loop_mode {
            LoopMode::Loop => (raw as u32) % self.frame_count,
            LoopMode::Once => {
                let i = raw as u32;
                i.min(self.frame_count - 1)
            }
            LoopMode::PingPong => {
                // 2N-2 unique slots; second half mirrors the first.
                let period = 2 * (self.frame_count - 1);
                let i = (raw as u32) % period;
                if i < self.frame_count {
                    i
                } else {
                    // i in [N..2N-2] → mirror back to [N-2..0].
                    // e.g. N=4: i=4→2, i=5→1, i=6→would be 0 but wraps.
                    period - i
                }
            }
        }
    }

    /// UV offset for frame `frame_idx`. Caller must clamp `frame_idx` to
    /// `0..frame_count`; [`frame_index_at`](Self::frame_index_at) does this.
    pub fn uv_offset_for(&self, frame_idx: u32) -> [f32; 2] {
        [
            self.uv_offset[0] + frame_idx as f32 * self.frame_uv_scale[0],
            self.uv_offset[1],
        ]
    }
}

/// Per-sprite playback state. Lives on the [`Sprite`](crate::render::Sprite)
/// as `Option<AnimationState>`; `None` means the sprite is static and the
/// frame-manager loop skips it entirely.
///
/// Fields are public so callers can construct directly. The engine
/// mutates `time` per frame.
#[derive(Copy, Clone, Debug)]
pub struct AnimationState {
    /// Index into the engine's `Vec<FrameStrip>` registry.
    /// Must be `< render_core.frame_strips.len()` when `draw_frame` runs.
    pub strip_id: u16,
    /// Accumulated seconds since the strip started. The engine adds
    /// `dt` to this each frame (when `paused == false`).
    pub time: f32,
    /// If true, [`time`](Self::time) does not advance. The strip still
    /// resolves to its current frame, so a paused sprite shows whatever
    /// frame it was on when the pause flipped.
    pub paused: bool,
}

impl AnimationState {
    /// Construct a fresh playback state at `time = 0`, not paused.
    pub fn new(strip_id: u16) -> Self {
        Self {
            strip_id,
            time: 0.0,
            paused: false,
        }
    }

    /// As [`Self::new`] but with a starting time offset. Used to stagger
    /// many sprites sharing one strip so they don't tick in lockstep.
    pub fn with_offset(strip_id: u16, time_offset: f32) -> Self {
        Self {
            strip_id,
            time: time_offset,
            paused: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(count: u32, fps: f32, mode: LoopMode) -> FrameStrip {
        FrameStrip {
            uv_offset: [0.0, 0.0],
            frame_uv_scale: [1.0 / count as f32, 1.0],
            frame_count: count,
            fps,
            loop_mode: mode,
        }
    }

    #[test]
    fn loop_wraps() {
        let s = strip(4, 1.0, LoopMode::Loop);
        assert_eq!(s.frame_index_at(0.0), 0);
        assert_eq!(s.frame_index_at(0.99), 0);
        assert_eq!(s.frame_index_at(1.0), 1);
        assert_eq!(s.frame_index_at(3.5), 3);
        assert_eq!(s.frame_index_at(4.0), 0);
        assert_eq!(s.frame_index_at(9.5), 1); // 9.5 * 1fps = 9.5, %4 = 1
    }

    #[test]
    fn once_clamps() {
        let s = strip(3, 2.0, LoopMode::Once);
        assert_eq!(s.frame_index_at(0.0), 0);
        assert_eq!(s.frame_index_at(0.4), 0);
        assert_eq!(s.frame_index_at(0.6), 1);
        assert_eq!(s.frame_index_at(1.0), 2);
        assert_eq!(s.frame_index_at(100.0), 2);
    }

    #[test]
    fn pingpong_bounces() {
        // 4 frames: visits 0,1,2,3,2,1, then repeats. Period = 6 frames.
        let s = strip(4, 1.0, LoopMode::PingPong);
        assert_eq!(s.frame_index_at(0.0), 0);
        assert_eq!(s.frame_index_at(1.0), 1);
        assert_eq!(s.frame_index_at(2.0), 2);
        assert_eq!(s.frame_index_at(3.0), 3);
        assert_eq!(s.frame_index_at(4.0), 2);
        assert_eq!(s.frame_index_at(5.0), 1);
        assert_eq!(s.frame_index_at(6.0), 0); // wraps
        assert_eq!(s.frame_index_at(7.0), 1);
    }

    #[test]
    fn single_frame_is_stable() {
        let s = strip(1, 10.0, LoopMode::Loop);
        assert_eq!(s.frame_index_at(0.0), 0);
        assert_eq!(s.frame_index_at(99.0), 0);
    }

    #[test]
    fn zero_fps_holds_frame_zero() {
        let s = strip(4, 0.0, LoopMode::Loop);
        assert_eq!(s.frame_index_at(0.0), 0);
        assert_eq!(s.frame_index_at(100.0), 0);
    }

    #[test]
    fn grid_row_builds_correct_uv() {
        // 15×16 grid, row 9, cols 0..2.
        let s = FrameStrip::from_grid_row(15, 16, 9, 0, 2, 4.0, LoopMode::Loop);
        let cw = 1.0 / 15.0;
        let ch = 1.0 / 16.0;
        assert!((s.uv_offset[0] - 0.0).abs() < 1e-6);
        assert!((s.uv_offset[1] - 9.0 * ch).abs() < 1e-6);
        assert!((s.frame_uv_scale[0] - cw).abs() < 1e-6);
        assert!((s.frame_uv_scale[1] - ch).abs() < 1e-6);
        // Frame 1's uv_offset: column 1 of row 9.
        let f1 = s.uv_offset_for(1);
        assert!((f1[0] - cw).abs() < 1e-6);
        assert!((f1[1] - 9.0 * ch).abs() < 1e-6);
    }

    #[test]
    fn cycle_seconds() {
        let loop4 = strip(4, 2.0, LoopMode::Loop);
        assert!((loop4.cycle_seconds() - 2.0).abs() < 1e-6); // 4 / 2 = 2s

        let pp4 = strip(4, 2.0, LoopMode::PingPong);
        // 2*(4-1) / 2 = 3s
        assert!((pp4.cycle_seconds() - 3.0).abs() < 1e-6);

        let zero = strip(4, 0.0, LoopMode::Loop);
        assert!(zero.cycle_seconds().is_infinite());
    }
}
