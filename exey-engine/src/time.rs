//! Frame timing — delta seconds and a rolling FPS counter.
//!
//! The original engine kept its own `getDelta()` and stats counter on
//! `ExeyEngineCore`; we factor that out into a small struct so the demo
//! can render its own FPS overlay (a requirement) without poking into
//! engine internals.

use std::time::Instant;

pub struct FrameClock {
    last: Instant,
    /// Smoothed FPS, exponential moving average so the readout doesn't flicker.
    fps_ema: f32,
    /// Last instantaneous delta in seconds.
    last_dt: f32,
}

impl FrameClock {
    pub fn new() -> Self {
        Self {
            last: Instant::now(),
            fps_ema: 0.0,
            last_dt: 0.0,
        }
    }

    /// Call once per frame. Returns delta time in seconds.
    pub fn tick(&mut self) -> f32 {
        let now = Instant::now();
        let dt = now.duration_since(self.last).as_secs_f32();
        self.last = now;
        self.last_dt = dt;

        // EMA, alpha=0.1. Gives a stable readout without lagging too far behind.
        let inst_fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
        if self.fps_ema == 0.0 {
            self.fps_ema = inst_fps;
        } else {
            self.fps_ema = self.fps_ema * 0.9 + inst_fps * 0.1;
        }
        dt
    }

    pub fn fps(&self) -> f32 {
        self.fps_ema
    }

    pub fn last_dt(&self) -> f32 {
        self.last_dt
    }
}

impl Default for FrameClock {
    fn default() -> Self {
        Self::new()
    }
}
