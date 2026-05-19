//! IsometricWorldGenerator — demo for ExeyEngine.
//!
//! M5 scope: a 32×32 grid of iso tiles **plus** ~24 scattered 2×2 buildings,
//! drawn through an `IsometricCamera2D` with `IsometricRectangleSorter`
//! providing correct depth ordering. Mouse drag pans the camera; the +/-
//! (and `=` / `_` for keyboards without a numpad) keys zoom in and out.
//! An on-screen FPS counter renders in the top-left via a tiny embedded
//! bitmap font.
//!
//! M7 scope: a handful of wolves are placed on the grid, each running a
//! 2-frame idle from a 4-facing strip (SW/SE/NW/NE). Wolves stagger their
//! playback time so they don't tick in lockstep. If `assets/wolf-all.png`
//! is missing, the demo falls back to a procedural 2-frame silhouette so
//! it always runs out-of-the-box.
//!
//! Four textures:
//! * Tile diamond (procedural, 64×32) — 1024 tiles
//! * Building 2×2 (procedural, taller than a tile so occlusion is visible)
//! * Font atlas (embedded const, 16 glyphs in a horizontal strip)
//! * Wolf spritesheet (15×16 cells of 64×64; or procedural 2-frame fallback)
//!
//! Renderer choice via `--renderer simple|batch|bigbuffer` CLI flag
//! (passed through from `run.sh`). All three currently produce the same
//! output — Batch and BigBuffer ship their real algorithms in M5/M6.
//!
//! Mouse controls:
//! * Left-click drag — pan the camera (scaled by `1/zoom` so the
//!   world drags one-to-one with the cursor).
//! * `+` / `=`        — zoom in 10% (screen-centered).
//! * `-` / `_`        — zoom out 10%.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use exey_engine::glam::Vec2;
use exey_engine::{
    depth_compare, AnimationState, Engine, EngineConfig, FrameClock, FrameStrip,
    IsometricCamera2D, IsoBounds, IsoSortable, LoopMode, RendererKind, Sprite, SpriteMesh,
    Texture, iso,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

mod font;

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Iso tile size, in world pixels. 2:1 iso → `tile_w = 2 * tile_h`. Only
/// `tile_h` participates in the projection math; `tile_w` is the rendered
/// sprite width.
const TILE_W: f32 = 64.0;
const TILE_H: f32 = 32.0;

/// Grid dimensions. 32×32 = 1024 tiles.
const GRID_W: usize = 32;
const GRID_H: usize = 32;

/// Visual margin between auto-fit zoom and the actual viewport edges,
/// expressed as a multiplicative factor on the computed zoom.
const ZOOM_FIT_MARGIN: f32 = 0.95;

/// Debug: place exactly 2 buildings at fixed positions.
/// Building 0: back corner at (0,0); Building 1: center of grid.
const BUILDING_COUNT: usize = 2;

/// Zoom multiplier per +/- keypress.
const ZOOM_STEP: f32 = 1.10;
/// Clamp to keep the scene visible.
const ZOOM_MIN: f32 = 0.05;
const ZOOM_MAX: f32 = 4.0;

/// Mesh indices used by the demo. Match the order of meshes passed to
/// `Engine::draw_frame`. `Sprite::mesh_idx` is `u8`.
const MESH_TILE: u8 = 0;
const MESH_BUILDING: u8 = 1;
const MESH_FONT: u8 = 2;
const MESH_WOLF: u8 = 3;

// M7 — wolf-sheet layout. The asset (when present) is 960×1024 with
// 15 columns × 16 rows of 64×64 cells. Rows 9–12 are idle, one row per
// facing direction (SW/SE/NW/NE, in that order). For M7 we register 4
// FrameStrips, each = first 2 frames of one of those rows.
const WOLF_ATLAS_CELLS_X: u32 = 15;
const WOLF_ATLAS_CELLS_Y: u32 = 16;
const WOLF_CELL_PX: u32 = 64;
const WOLF_IDLE_FRAME_COUNT: u32 = 2;
const WOLF_IDLE_FPS: f32 = 2.5;
/// Idle row per facing, in registry order.
const WOLF_IDLE_ROWS: [u32; 4] = [9, 10, 11, 12]; // SW, SE, NW, NE

/// How many wolves to scatter on the iso grid.
const WOLF_COUNT: usize = 12;
/// Wolf draw size on the grid, in world pixels. The atlas cells are
/// 64×64 — same as `TILE_W` — so 1:1 keeps the wolf reading at tile
/// scale. Tuned upward slightly so the silhouette is recognisable.
const WOLF_SPRITE_W: f32 = 64.0;
const WOLF_SPRITE_H: f32 = 64.0;

/// CLI args. Tiny hand-rolled parser — pulling in `clap` for one flag is
/// overkill at M2. Add more flags as the demo grows.
struct Args {
    renderer: RendererKind,
}

impl Args {
    fn parse() -> Self {
        let mut renderer = RendererKind::default();
        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--renderer" | "-r" => {
                    if let Some(v) = iter.next() {
                        if let Some(k) = RendererKind::from_cli(&v) {
                            renderer = k;
                        } else {
                            println!("unknown renderer '{v}', falling back to default");
                        }
                    }
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other if other.starts_with("--renderer=") => {
                    let v = other.trim_start_matches("--renderer=");
                    if let Some(k) = RendererKind::from_cli(v) {
                        renderer = k;
                    }
                }
                _ => {}
            }
        }
        Self { renderer }
    }
}

fn print_help() {
    println!("isometric-world-generator — ExeyEngine demo\n");
    println!("USAGE:");
    println!("    isometric-world-generator [OPTIONS]\n");
    println!("OPTIONS:");
    println!("    -r, --renderer <KIND>   simple | batch | bigbuffer  (default: bigbuffer)");
    println!("    -h, --help              print this message");
}

/// Simple stdout-flushing logger. We use this instead of env_logger because
/// on some terminals (notably some macOS setups) stderr appears buffered or
/// silenced; stdout is more reliably user-visible. Every log line is followed
/// by an explicit flush so output appears immediately.
struct StdoutLogger;
impl log::Log for StdoutLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = writeln!(
            out,
            "[{:>5}] {}: {}",
            record.level(),
            record.target(),
            record.args()
        );
        let _ = out.flush();
    }
    fn flush(&self) {
        use std::io::Write;
        let _ = std::io::stdout().lock().flush();
    }
}
static LOGGER: StdoutLogger = StdoutLogger;

/// Dump sort-order debug info for buildings and their neighbours.
/// Called once per session on the first rendered frame.
fn log_sort_debug(sort_order: &[u32], world_sprites: &[Sprite]) {
    log::info!("=== SORT DEBUG  ({} sprites, {} in order) ===", world_sprites.len(), sort_order.len());

    // Build a rank lookup: rank_of[sprite_idx] = rank in sort_order.
    let mut rank_of = vec![0usize; world_sprites.len()];
    for (rank, &idx) in sort_order.iter().enumerate() {
        rank_of[idx as usize] = rank;
    }

    for (rank, &idx) in sort_order.iter().enumerate() {
        let s = &world_sprites[idx as usize];
        if s.mesh_idx != MESH_BUILDING { continue; }

        let b = s.iso_bounds();
        log::info!(
            "BUILDING [input_idx={}] rank={} grid=[{},{}] size=[{},{}]",
            idx, rank, s.iso_grid[0], s.iso_grid[1], s.iso_grid_size[0], s.iso_grid_size[1],
        );
        log::info!(
            "  iso: x1={:.1} y1={:.1} x2={:.1} y2={:.1}  left={:.1} right={:.1}",
            b.iso_x1, b.iso_y1, b.iso_x2, b.iso_y2, b.iso_left(), b.iso_right(),
        );

        // Show ±10 neighbours in draw order.
        let lo = rank.saturating_sub(10);
        let hi = (rank + 11).min(sort_order.len());
        log::info!("  -- neighbours in draw order (rank {} to {}) --", lo, hi - 1);
        for r in lo..hi {
            let ni = sort_order[r] as usize;
            let ns = &world_sprites[ni];
            let nb = ns.iso_bounds();
            let a_term = (b.iso_x1 - nb.iso_x2).max(b.iso_y1 - nb.iso_y2);
            let b_term = (nb.iso_x1 - b.iso_x2).max(nb.iso_y1 - b.iso_y2);
            let cmp = depth_compare(&b, &nb);
            let marker = if r == rank { ">>>" } else { "   " };
            log::info!(
                "  {} rank={:4} idx={:4} mesh={} grid=[{:.0},{:.0}] \
                 iso_left={:.1} iso_right={:.1}  \
                 depth_cmp(bldg,this)={:+} (a={:.1} b={:.1})",
                marker, r, ni, ns.mesh_idx,
                ns.iso_grid[0], ns.iso_grid[1],
                nb.iso_left(), nb.iso_right(),
                cmp, a_term, b_term,
            );
        }

        // Explicitly check the 4 footprint tiles and 4 side-adjacent tiles.
        let gx = (b.iso_x1) as isize;
        let gy = (b.iso_y1) as isize;
        log::info!("  -- footprint + side tiles depth_compare(building, tile) --");
        let check_coords: &[(isize, isize, &str)] = &[
            (gx,   gy,   "back corner"),
            (gx+1, gy,   "footprint front-x"),
            (gx,   gy+1, "footprint front-y"),
            (gx+1, gy+1, "footprint front"),
            (gx-1, gy,   "side left-x"),
            (gx,   gy-1, "side left-y"),
            (gx+2, gy,   "side right-x"),
            (gx,   gy+2, "side right-y"),
            (gx+2, gy+2, "diagonal front"),
        ];
        for &(tx, ty, label) in check_coords {
            if tx < 0 || ty < 0 || tx >= GRID_W as isize || ty >= GRID_H as isize { continue; }
            let ti = (ty as usize) * GRID_W + tx as usize;
            let ts = &world_sprites[ti];
            let tb: IsoBounds = ts.iso_bounds();
            let cmp = depth_compare(&b, &tb);
            let a_term = (b.iso_x1 - tb.iso_x2).max(b.iso_y1 - tb.iso_y2);
            let b_term = (tb.iso_x1 - b.iso_x2).max(tb.iso_y1 - b.iso_y2);
            let expected = if label.starts_with("footprint") { ">0 (bldg in front)" }
                           else if label.starts_with("side") || label == "diagonal front"
                               { "<0 (tile in front)" } else { "?" };
            log::info!(
                "    tile({:2},{:2}) [rank={:4}] {:16}  cmp={:+}  a={:+.1} b={:+.1}  want: {}",
                tx, ty, rank_of[ti], label, cmp, a_term, b_term, expected,
            );
        }
    }
    log::info!("=== END SORT DEBUG ===");
}

fn main() -> Result<()> {
    // Stdout-flushed banner: this is the very first thing main() does. If
    // you don't see this in your terminal, the binary isn't reaching main()
    // (or stdout is being swallowed by something outside our control).
    println!("=== isometric-world-generator: main() entered");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    log::set_logger(&LOGGER).expect("failed to install logger");
    log::set_max_level(log::LevelFilter::Info);
    log::info!("logger installed (stdout, flushed)");

    let args = Args::parse();
    log::info!("renderer = {}", args.renderer.as_str());

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        config: EngineConfig {
            app_name: "IsometricWorldGenerator".to_string(),
            renderer: args.renderer,
        },
        window: None,
        engine: None,
        scene: None,
        clock: FrameClock::new(),
        last_fps_print: Instant::now(),
    };
    log::info!("entering event loop");
    event_loop.run_app(&mut app)?;
    log::info!("event loop exited cleanly");
    Ok(())
}

/// Build a procedural diamond texture: a filled iso-tile shape with a
/// 1-pixel outline. The texture is `tile_w × tile_h` pixels — the same
/// dimensions as the rendered sprite — so UV [0..1] maps 1:1 to the
/// diamond. Pixels outside the diamond shape are fully transparent so
/// the alpha-blend pipeline cuts them out cleanly.
fn build_diamond_rgba(tile_w: u32, tile_h: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (tile_w * tile_h * 4) as usize];
    let cx = (tile_w - 1) as f32 * 0.5;
    let cy = (tile_h - 1) as f32 * 0.5;
    for y in 0..tile_h {
        for x in 0..tile_w {
            let nx = (x as f32 - cx).abs() / cx;
            let ny = (y as f32 - cy).abs() / cy;
            let d = nx + ny;
            let i = ((y * tile_w + x) * 4) as usize;
            if d <= 1.0 {
                let outline_zone = 0.94;
                if d > outline_zone {
                    rgba[i] = 32; rgba[i + 1] = 24; rgba[i + 2] = 16; rgba[i + 3] = 255;
                } else {
                    let t = y as f32 / tile_h as f32;
                    let lo = 0.55; let hi = 0.85;
                    let v = lo + (hi - lo) * (1.0 - t);
                    rgba[i] = (110.0 * v) as u8;
                    rgba[i + 1] = (165.0 * v) as u8;
                    rgba[i + 2] = (95.0 * v) as u8;
                    rgba[i + 3] = 255;
                }
            } else {
                rgba[i] = 0; rgba[i + 1] = 0; rgba[i + 2] = 0; rgba[i + 3] = 0;
            }
        }
    }
    rgba
}

/// Build a procedural 2×2 building texture: an iso prism shape (diamond
/// base + vertical walls + sloped top). The texture is sized so the
/// building's base is 2 tiles wide × 2 tiles deep, and its top extends
/// `BUILDING_H` extra pixels upward to create occlusion that the sorter
/// must resolve.
///
/// Texture coords:
///   Width  = 2 * tile_w           — base diamond is 2 tiles wide
///   Height = 2 * tile_h + extra   — base diamond is 2 tiles tall, plus
///                                    `extra` pixels of vertical body
///
/// Shape: a "house"-style iso prism. The renderer treats this as one
/// big sprite — the depth sorter sees a 2×2 footprint, so tiles in front
/// occlude it correctly and tiles behind are occluded by it.
fn build_building_rgba(tile_w: u32, tile_h: u32, body_extra: u32) -> (u32, u32, Vec<u8>) {
    let base_w = tile_w * 2;
    let base_h = tile_h * 2;
    let height = base_h + body_extra;
    let mut rgba = vec![0u8; (base_w * height * 4) as usize];
    let cx = (base_w - 1) as f32 * 0.5;
    // Base diamond centre y: middle of the bottom 2 tiles.
    let base_top_y = body_extra as f32;
    let base_cy = base_top_y + (base_h - 1) as f32 * 0.5;
    let half_w = cx;
    let half_h = (base_h - 1) as f32 * 0.5;

    for y in 0..height {
        for x in 0..base_w {
            let yf = y as f32;
            let xf = x as f32;
            let i = ((y * base_w + x) * 4) as usize;

            // 1) Check the body slab: vertical column from base_top_y down to
            //    base_cy (i.e. the back half of the base diamond extruded up).
            //    The slab's left/right edges follow the top half of the diamond
            //    (since the building's top is where the back of the base is).
            let dy_from_top = base_top_y - yf;
            let in_body = if yf < base_top_y {
                // Above the base entirely. Body slab here: x range is the
                // diamond's top-half width at the relative y.
                let t = (base_top_y - yf) / body_extra as f32;
                let _ = t;
                // Body x extents at this height = same as diamond top-corner
                // width at y = base_top_y (i.e. full top of base).
                // For simplicity treat body as a clipped trapezoid: top edges
                // are the back-corners of the base diamond, bottom edges are
                // the side-corners (full diamond width at base_cy).
                let frac = yf / base_top_y; // 0 at top, 1 at base_top_y
                let w_at_y = half_w * frac; // narrow at top, wide at base
                (xf - cx).abs() <= w_at_y
            } else {
                false
            };

            // 2) Check the base diamond: |x-cx|/half_w + |y-base_cy|/half_h <= 1
            let nx = (xf - cx).abs() / half_w;
            let ny = (yf - base_cy).abs() / half_h;
            let in_base = (nx + ny) <= 1.0 && yf >= base_top_y;

            if in_body {
                // Body fill — warm brown.
                let _ = dy_from_top;
                let shade = 0.7 + 0.3 * (yf / height as f32);
                rgba[i] = (180.0 * shade) as u8;
                rgba[i + 1] = (130.0 * shade) as u8;
                rgba[i + 2] = (90.0 * shade) as u8;
                rgba[i + 3] = 255;
            } else if in_base {
                // Base ring + fill — slightly lighter than body so the
                // building looks footed.
                let d = nx + ny;
                if d > 0.92 {
                    rgba[i] = 40; rgba[i + 1] = 30; rgba[i + 2] = 20; rgba[i + 3] = 255;
                } else {
                    rgba[i] = 200; rgba[i + 1] = 160; rgba[i + 2] = 110; rgba[i + 3] = 255;
                }
            } else {
                // Transparent.
                rgba[i] = 0; rgba[i + 1] = 0; rgba[i + 2] = 0; rgba[i + 3] = 0;
            }
        }
    }
    (base_w, height, rgba)
}

// =====================================================================
// M7 — wolf spritesheet
// =====================================================================

/// Path to the real wolf spritesheet (CC-BY asset; not redistributed).
/// If absent at runtime we fall back to a procedural 2-frame silhouette
/// so the demo always runs out-of-the-box (a warning is logged).
fn wolf_asset_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("assets");
    p.push("wolf-all.png");
    p
}

/// Build a tiny procedural 2-frame wolf-shaped silhouette atlas, used as
/// a fallback when `assets/wolf-all.png` isn't on disk. We mimic the
/// real sheet's 15×16 layout so the demo's `FrameStrip`s work without
/// branching on which texture we got. The real sheet would have actual
/// frames in rows 9–12; we only paint frame 0 and 1 of row 9 (SW idle)
/// and leave the rest transparent. The other three facings reuse the
/// same two cells via UV — they'll all look identical, but the animation
/// still ticks, which is the M7 deliverable.
///
/// Returns RGBA bytes for a (15*64) × (16*64) atlas with luma-key-friendly
/// near-black background.
fn build_wolf_fallback_rgba() -> Vec<u8> {
    let atlas_w = WOLF_ATLAS_CELLS_X * WOLF_CELL_PX; // 960
    let atlas_h = WOLF_ATLAS_CELLS_Y * WOLF_CELL_PX; // 1024
    let mut rgba = vec![0u8; (atlas_w * atlas_h * 4) as usize];

    // Paint a wolf-ish silhouette into cell (row, col). Frame index drives
    // a small breathing offset so the two idle frames read as different.
    // The silhouette is approximate — body ellipse + head circle + four
    // leg blocks + tail — at this resolution it just needs to read as
    // "an animal" against the green tiles. Frame 0 is the rest pose;
    // frame 1 lifts the back two legs and tucks the body 1px to suggest
    // a breath.
    let paint_cell = |rgba: &mut [u8], row: u32, col: u32, frame: u32| {
        let cx = col * WOLF_CELL_PX;
        let cy = row * WOLF_CELL_PX;
        let breath = if frame == 0 { 0i32 } else { -1i32 };
        // Body ellipse: centred slightly back and below mid-cell.
        let body_cx = 30i32;
        let body_cy = 36i32 + breath;
        let body_a = 18i32; // semi-major (x)
        let body_b = 10i32; // semi-minor (y)
        // Head circle: in front-bottom-left (SW facing wolf looks down-left).
        let head_cx = 14i32;
        let head_cy = 34i32 + breath;
        let head_r = 6i32;
        // Legs: 4 vertical bars below the body.
        let leg_y0 = 46i32 + breath;
        let leg_y1 = 56i32;
        let leg_xs = [16i32, 24i32, 34i32, 42i32];
        // Frame 1 lifts back legs by 2px.
        let leg_lift = [0i32, 0i32, if frame == 1 { 2 } else { 0 }, if frame == 1 { 2 } else { 0 }];
        // Tail: short diagonal stub off the rear.
        let tail_pts = [(48i32, 32i32), (50i32, 30i32), (52i32, 28i32)];

        for py in 0..WOLF_CELL_PX as i32 {
            for px in 0..WOLF_CELL_PX as i32 {
                let mut hit = false;
                // Body ellipse
                let dx = px - body_cx;
                let dy = py - body_cy;
                if (dx * dx * body_b * body_b + dy * dy * body_a * body_a)
                    <= (body_a * body_a * body_b * body_b)
                {
                    hit = true;
                }
                // Head
                let hdx = px - head_cx;
                let hdy = py - head_cy;
                if hdx * hdx + hdy * hdy <= head_r * head_r {
                    hit = true;
                }
                // Legs
                for (i, lx) in leg_xs.iter().enumerate() {
                    let ly0 = leg_y0 - leg_lift[i];
                    let ly1 = leg_y1 - leg_lift[i];
                    if px >= *lx - 1 && px <= *lx + 1 && py >= ly0 && py <= ly1 {
                        hit = true;
                    }
                }
                // Tail
                for (tx, ty) in tail_pts.iter() {
                    if (px - tx).abs() <= 1 && (py - ty).abs() <= 1 {
                        hit = true;
                    }
                }
                if hit {
                    let x = cx as i32 + px;
                    let y = cy as i32 + py;
                    if x >= 0 && y >= 0 && (x as u32) < atlas_w && (y as u32) < atlas_h {
                        let i = ((y as u32 * atlas_w + x as u32) * 4) as usize;
                        // Dark slate, similar to the real asset's tone.
                        rgba[i] = 60;
                        rgba[i + 1] = 70;
                        rgba[i + 2] = 80;
                        rgba[i + 3] = 255;
                    }
                }
            }
        }
    };

    // Paint frame 0 and frame 1 into the SW idle row (row 9, cols 0..2)
    // so registered strips against rows 9–12 col 0..2 will all sample
    // these same two cells. Other rows stay fully transparent.
    paint_cell(&mut rgba, 9, 0, 0);
    paint_cell(&mut rgba, 9, 1, 1);
    log::warn!(
        "build_wolf_fallback_rgba — painted procedural fallback into row 9 cols 0..2; \
         FrameStrips for rows 10/11/12 will sample transparent cells (wolves at those \
         facings will be invisible until a real wolf-all.png is dropped in assets/)"
    );

    rgba
}

/// Load the wolf texture: try the real PNG/JPEG asset first, fall back
/// to the procedural builder if it's missing or fails to decode. Always
/// returns a texture sized as `(15*64)×(16*64)` so the demo's
/// `FrameStrip`s work either way.
fn build_wolf_texture(engine: &Engine) -> Result<Texture> {
    let path = wolf_asset_path();
    if path.exists() {
        log::info!(
            "build_wolf_texture — found asset at {} ; loading with luma-key alpha",
            path.display()
        );
        match Texture::from_image_file_with_luma_key(&engine.instance, &engine.device, &path) {
            Ok(tex) => {
                // Sanity-check dimensions against our strip assumptions.
                let expected_w = WOLF_ATLAS_CELLS_X * WOLF_CELL_PX;
                let expected_h = WOLF_ATLAS_CELLS_Y * WOLF_CELL_PX;
                if tex.width != expected_w || tex.height != expected_h {
                    log::warn!(
                        "build_wolf_texture — asset is {}x{} but FrameStrip math assumes {}x{} \
                         ({} cells × {} cells of {} px). UV coords will be off; \
                         consider re-exporting or updating WOLF_ATLAS_CELLS_* constants.",
                        tex.width, tex.height, expected_w, expected_h,
                        WOLF_ATLAS_CELLS_X, WOLF_ATLAS_CELLS_Y, WOLF_CELL_PX,
                    );
                }
                return Ok(tex);
            }
            Err(e) => {
                log::warn!(
                    "build_wolf_texture — asset present but failed to decode: {e:#}; \
                     falling back to procedural"
                );
            }
        }
    } else {
        log::warn!(
            "build_wolf_texture — asset not found at {} ; using procedural fallback. \
             Drop the real wolf-all.png there for the proper sprite.",
            path.display()
        );
    }
    let rgba = build_wolf_fallback_rgba();
    let w = WOLF_ATLAS_CELLS_X * WOLF_CELL_PX;
    let h = WOLF_ATLAS_CELLS_Y * WOLF_CELL_PX;
    Texture::from_rgba(&engine.instance, &engine.device, w, h, &rgba)
}

/// Tiny deterministic PRNG so wolf placement is reproducible across runs
/// without pulling in `rand`. xorshift32, seeded with a constant.
struct DemoRng(u32);
impl DemoRng {
    fn new(seed: u32) -> Self {
        Self(seed | 1) // non-zero
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn range(&mut self, n: u32) -> u32 {
        self.next_u32() % n.max(1)
    }
    fn unit(&mut self) -> f32 {
        // Roughly uniform [0..1).
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

struct Scene {
    // GPU resources (kept alive for the scene's lifetime).
    tile_mesh: SpriteMesh,
    tile_texture: Texture,
    building_mesh: SpriteMesh,
    building_texture: Texture,
    font_mesh: SpriteMesh,
    font_texture: Texture,
    /// M7 — wolf spritesheet + mesh. The mesh's UV space spans the full
    /// atlas; per-sprite `uv_offset`/`uv_scale` are written each frame
    /// by the engine's animation tick from the registered FrameStrips.
    wolf_mesh: SpriteMesh,
    wolf_texture: Texture,

    /// World sprites: tiles + buildings, sorted by the iso sorter each frame.
    world_sprites: Vec<Sprite>,
    /// GUI sprites: FPS overlay text, drawn after world.
    gui_sprites: Vec<Sprite>,

    /// Camera used to project world → clip.
    camera: IsometricCamera2D,

    /// Drag state: previous cursor position when LMB is held.
    drag_anchor: Option<PhysicalPosition<f64>>,
    /// Last known cursor position (for screen-space hit math when zooming).
    cursor: PhysicalPosition<f64>,

    /// Static tile coordinate label sprites (built once, appended to gui each
    /// frame). Stored separately so rebuild_fps_overlay doesn't recompute them.
    label_sprites: Vec<Sprite>,
    /// Last-known FPS string content; we rebuild gui_sprites only when it
    /// changes (each second) to avoid per-frame allocation.
    last_fps_text: String,
    /// Set to true after the first-frame sort debug log fires.
    sort_logged: bool,
}

impl Scene {
    fn new(engine: &mut Engine) -> Result<Self> {
        log::info!(
            "Scene::new — building {GRID_W}×{GRID_H} iso grid + {BUILDING_COUNT} buildings",
        );

        // -- Tile texture + mesh --
        let tex_w = TILE_W as u32;
        let tex_h = TILE_H as u32;
        let tile_rgba = build_diamond_rgba(tex_w, tex_h);
        let tile_texture =
            Texture::from_rgba(&engine.instance, &engine.device, tex_w, tex_h, &tile_rgba)?;
        let tile_mesh = SpriteMesh::unit_quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &tile_texture,
        )?;

        // -- Building texture + mesh --
        // Body extends 1.5 tile-heights above the base so occlusion is
        // visually obvious. With 2×2 base and 48px of extra body, the
        // sprite is 128×112 (texture); we draw it at the same world size.
        let body_extra = (TILE_H * 1.5) as u32;
        let (bw, bh, building_rgba) = build_building_rgba(tex_w, tex_h, body_extra);
        let building_texture =
            Texture::from_rgba(&engine.instance, &engine.device, bw, bh, &building_rgba)?;
        let building_mesh = SpriteMesh::unit_quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &building_texture,
        )?;

        // -- Font texture + mesh --
        let font_rgba = font::build_atlas_rgba();
        let font_texture = Texture::from_rgba(
            &engine.instance, &engine.device, font::ATLAS_W, font::ATLAS_H, &font_rgba,
        )?;
        let font_mesh = SpriteMesh::unit_quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &font_texture,
        )?;

        // -- M7: wolf texture + mesh + frame strip registry --
        // The wolf atlas is shared GPU geometry just like the tile/font;
        // wolves differ only in per-sprite uv_offset/uv_scale, which the
        // engine writes each frame from their AnimationState.
        let wolf_texture = build_wolf_texture(engine)?;
        let wolf_mesh = SpriteMesh::unit_quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &wolf_texture,
        )?;
        // Register one strip per facing. `wolf_strips[i]` is the strip_id
        // for facing `i` in `WOLF_IDLE_ROWS` order (SW, SE, NW, NE).
        let mut wolf_strips: [u16; 4] = [0; 4];
        for (i, row) in WOLF_IDLE_ROWS.iter().enumerate() {
            let strip = FrameStrip::from_grid_row(
                WOLF_ATLAS_CELLS_X,
                WOLF_ATLAS_CELLS_Y,
                *row,
                /*col0=*/ 0,
                WOLF_IDLE_FRAME_COUNT,
                WOLF_IDLE_FPS,
                LoopMode::Loop,
            );
            wolf_strips[i] = engine.render.register_strip(strip);
        }
        log::info!(
            "Scene::new — registered {} wolf idle strips (rows {:?}, {} frames each @ {:.1} fps)",
            wolf_strips.len(), WOLF_IDLE_ROWS, WOLF_IDLE_FRAME_COUNT, WOLF_IDLE_FPS,
        );

        // -- Build world: tiles in row-major, then buildings on random cells --
        let mut world_sprites = Vec::with_capacity(GRID_W * GRID_H + BUILDING_COUNT + WOLF_COUNT);
        for gy in 0..GRID_H {
            for gx in 0..GRID_W {
                let g = Vec2::new(gx as f32, gy as f32);
                let world = iso::logic_to_world(g, TILE_H);
                let pos = [world.x - TILE_W * 0.5, world.y];
                let parity = (gx + gy) % 2 == 0;
                let base = if parity { 1.0 } else { 0.85 };
                let tint = [
                    base * (0.85 + 0.15 * (gx as f32 / GRID_W as f32)),
                    base,
                    base * (0.85 + 0.15 * (gy as f32 / GRID_H as f32)),
                    1.0,
                ];
                let mut s = Sprite::new(pos, [TILE_W, TILE_H], [0.0, 0.0], tint);
                s.iso_grid = [gx as f32, gy as f32];
                s.iso_grid_size = [1.0, 1.0];
                s.mesh_idx = MESH_TILE;
                // uv_offset/scale default to (0,0)/(1,1) from Sprite::new.
                world_sprites.push(s);
            }
        }
        log::info!("Scene::new — placed {} tile sprites", world_sprites.len());

        // Two fixed debug buildings. Back corner (gx,gy) → sprite anchored at
        // world(gx,gy) (the top corner of the back tile). iso_grid is set to
        // [gx+2, gy+2] so the building's sort front corner is one step past
        // the footprint's frontmost tile.
        let building_sprite_w = TILE_W * 2.0;
        let building_sprite_h = TILE_H * 2.0 + body_extra as f32;
        let debug_backs: [(usize, usize); BUILDING_COUNT] = [
            (0, 0),                             // Building 0: top-left corner
            (GRID_W / 2 - 1, GRID_H / 2 - 1), // Building 1: center
        ];
        for (i, (gx, gy)) in debug_backs.iter().enumerate() {
            let (gx, gy) = (*gx, *gy);
            let anchor = iso::logic_to_world(Vec2::new(gx as f32, gy as f32), TILE_H);
            let pos = [anchor.x - building_sprite_w * 0.5, anchor.y - body_extra as f32];
            let tint = [1.0, 1.0, 1.0, 1.0];
            let mut s = Sprite::new(pos, [building_sprite_w, building_sprite_h], [0.0, 0.0], tint);
            s.iso_grid = [(gx + 2) as f32, (gy + 2) as f32];
            s.iso_grid_size = [2.0, 2.0];
            s.mesh_idx = MESH_BUILDING;
            world_sprites.push(s);
            log::info!(
                "Scene::new — building {} back=({},{}) iso_grid=[{},{}] anchor=({:.1},{:.1})",
                i, gx, gy, gx + 2, gy + 2, anchor.x, anchor.y,
            );
        }
        log::info!("Scene::new — placed {BUILDING_COUNT} debug buildings");

        // -- M7: scatter wolves on the iso grid --
        // Random tile, random facing, staggered AnimationState.time so
        // wolves don't all flip frame on the same tick. Footprint is 1×1
        // (a single tile) so the iso sorter treats them like tile-scale
        // characters. Anchor matches a tile's screen position; the sprite
        // is drawn aligned to the tile's top corner (same pattern as the
        // building anchor math, just at 1× scale).
        let mut rng = DemoRng::new(0xC0FFEE);
        for w in 0..WOLF_COUNT {
            // Pick a grid cell. We don't bother de-duping against
            // building footprints — at WOLF_COUNT=12 in a 32×32 grid
            // the collision probability is negligible and visual overlap
            // is actually a useful test for the sorter.
            let gx = rng.range(GRID_W as u32) as usize;
            let gy = rng.range(GRID_H as u32) as usize;
            let facing = rng.range(4) as usize; // 0..3 → SW/SE/NW/NE
            let time_offset = rng.unit() * (WOLF_IDLE_FRAME_COUNT as f32 / WOLF_IDLE_FPS);

            let anchor = iso::logic_to_world(Vec2::new(gx as f32, gy as f32), TILE_H);
            // Center the sprite horizontally on the tile's top corner;
            // bias upward so the wolf's feet sit roughly on the tile's
            // diamond center, not on its top vertex.
            let pos = [
                anchor.x - WOLF_SPRITE_W * 0.5,
                anchor.y - (WOLF_SPRITE_H - TILE_H) * 0.5,
            ];
            let mut s = Sprite::new(pos, [WOLF_SPRITE_W, WOLF_SPRITE_H], [0.0, 0.0], [1.0, 1.0, 1.0, 1.0]);
            // 1×1 iso footprint. The wolf's "front corner" for sorting
            // is just the tile it stands on, shifted by 1 like other
            // sprites — i.e. iso_grid is (gx+1, gy+1) so the wolf sorts
            // in front of the tile it's drawn over.
            s.iso_grid = [(gx + 1) as f32, (gy + 1) as f32];
            s.iso_grid_size = [1.0, 1.0];
            s.mesh_idx = MESH_WOLF;
            s.anim = Some(AnimationState::with_offset(wolf_strips[facing], time_offset));
            world_sprites.push(s);
            log::info!(
                "Scene::new — wolf {w} at grid=({gx},{gy}) facing={} time_offset={:.3}s",
                ["SW", "SE", "NW", "NE"][facing], time_offset,
            );
        }
        log::info!("Scene::new — placed {WOLF_COUNT} wolves");

        // -- Tile coordinate labels (static, built once) --
        // Each tile gets a "gx,gy" text label rendered as a GUI sprite on top.
        // Scale: small enough to fit in a tile but readable when zoomed in.
        let label_scale = 1.5f32;
        let label_gw = font::GLYPH_W as f32 * label_scale;
        let label_advance = label_gw + label_scale;
        let mut label_sprites = Vec::with_capacity(GRID_W * GRID_H * 5);
        for gy in 0..GRID_H {
            for gx in 0..GRID_W {
                let world = iso::logic_to_world(Vec2::new(gx as f32, gy as f32), TILE_H);
                let text = format!("{},{}", gx, gy);
                let char_count = text.chars()
                    .filter(|c| font::glyph_index(*c).is_some())
                    .count() as f32;
                let label_w = char_count * label_advance;
                // Center the label horizontally over the tile's top corner,
                // shifted down just a few pixels so it sits inside the diamond.
                let lx = world.x - label_w * 0.5;
                let ly = world.y + 4.0;
                font::FontAtlas::emit(
                    &mut label_sprites,
                    &text,
                    lx, ly,
                    label_scale,
                    [1.0, 1.0, 0.0, 0.85],
                    MESH_FONT,
                );
            }
        }
        log::info!("Scene::new — built {} tile label sprites", label_sprites.len());

        // -- Camera --
        let world_w = TILE_H * 2.0 * (GRID_W.max(GRID_H) as f32 - 1.0) + TILE_W;
        let world_h = TILE_H * (GRID_W + GRID_H - 2) as f32 * 0.5 + TILE_H;
        let centre_x = 0.0;
        let centre_y = world_h * 0.5;
        let viewport = Vec2::new(WINDOW_W as f32, WINDOW_H as f32);
        let zoom = (viewport.x / world_w).min(viewport.y / world_h) * ZOOM_FIT_MARGIN;
        let mut camera = IsometricCamera2D::new();
        camera.state.position = Vec2::new(centre_x, centre_y);
        camera.state.zoom = zoom;
        camera.state.viewport = viewport;

        log::info!(
            "Scene::new — camera centre=({centre_x:.1},{centre_y:.1}) zoom={zoom:.4}",
        );

        Ok(Self {
            tile_mesh,
            tile_texture,
            building_mesh,
            building_texture,
            font_mesh,
            font_texture,
            wolf_mesh,
            wolf_texture,
            world_sprites,
            gui_sprites: Vec::new(),
            label_sprites,
            camera,
            drag_anchor: None,
            cursor: PhysicalPosition::new(0.0, 0.0),
            last_fps_text: String::new(),
            sort_logged: false,
        })
    }

    fn on_resize(&mut self, width: u32, height: u32) {
        self.camera.state.viewport = Vec2::new(width as f32, height as f32);
    }

    /// Apply a multiplicative zoom centred on the *screen* (viewport
    /// centre). Since the camera position is the world point at screen
    /// centre, screen-centered zoom leaves `camera.position` unchanged.
    fn apply_zoom(&mut self, factor: f32) {
        let new_zoom = (self.camera.state.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
        self.camera.state.zoom = new_zoom;
    }

    /// Drag pan: the camera moves *opposite* the cursor delta (in screen
    /// pixels) so the world content drags with the cursor. The screen-
    /// pixel delta is divided by `zoom` to convert to world pixels.
    fn drag_to(&mut self, new_pos: PhysicalPosition<f64>) {
        if let Some(anchor) = self.drag_anchor {
            let dx = (new_pos.x - anchor.x) as f32;
            let dy = (new_pos.y - anchor.y) as f32;
            let zoom = self.camera.state.zoom.max(1e-6);
            self.camera.state.position.x -= dx / zoom;
            self.camera.state.position.y -= dy / zoom;
            self.drag_anchor = Some(new_pos);
        }
    }

    /// Rebuild the FPS overlay sprites if the text changed.
    /// `viewport_top_left` is the position (in world pixels) of the
    /// top-left of the viewport; GUI sprites are positioned relative
    /// to it so they stay glued to the corner regardless of camera pan.
    ///
    /// We anchor GUI text in *world coords* but compute that anchor
    /// from the inverse camera transform. M8 will introduce a proper
    /// screen-space GUI camera; for now we hack it.
    fn rebuild_fps_overlay(&mut self, fps: f32) {
        let text = format!("FPS: {:.0}", fps);
        if text == self.last_fps_text {
            return;
        }
        self.last_fps_text = text.clone();
        self.gui_sprites.clear();
        // Position the text in the top-left of the *viewport*. The
        // viewport top-left in world coords is `camera.position - viewport/2/zoom`.
        let s = &self.camera.state;
        let half = s.viewport * 0.5 / s.zoom.max(1e-6);
        let topleft = s.position - half;
        // Anchor 12 (world) px in from the corner so glyphs aren't
        // flush against the edge. Scale = 1/zoom so the rendered text
        // stays roughly the same screen size regardless of zoom.
        let scale = 4.0 / s.zoom.max(1e-6);
        let pad = 8.0 / s.zoom.max(1e-6);
        font::FontAtlas::emit(
            &mut self.gui_sprites,
            &text,
            topleft.x + pad,
            topleft.y + pad,
            scale,
            [1.0, 1.0, 0.4, 1.0], // yellow
            MESH_FONT,
        );
        // Append static tile labels after the FPS text.
        self.gui_sprites.extend_from_slice(&self.label_sprites);
    }

    fn destroy(&mut self, engine: &Engine) {
        engine.device.wait_idle();
        self.tile_mesh.destroy(&engine.device);
        self.tile_texture.destroy(&engine.device);
        self.building_mesh.destroy(&engine.device);
        self.building_texture.destroy(&engine.device);
        self.font_mesh.destroy(&engine.device);
        self.font_texture.destroy(&engine.device);
        self.wolf_mesh.destroy(&engine.device);
        self.wolf_texture.destroy(&engine.device);
    }
}

struct App {
    config: EngineConfig,
    /// Wrapped in Arc so winit can keep its own ref alongside our reads.
    /// Created lazily in `resumed` per winit 0.30 idiom.
    window: Option<Arc<Window>>,
    engine: Option<Engine>,
    scene: Option<Scene>,
    clock: FrameClock,
    last_fps_print: Instant,
}

/// Dispatch a keyboard event to the scene. Free function so it can take
/// `Option<&mut Scene>` without fighting the borrow checker against the
/// surrounding `self` access.
fn handle_key(scene: Option<&mut Scene>, ev: KeyEvent) {
    if ev.state != ElementState::Pressed {
        return;
    }
    let Some(scene) = scene else { return };
    // Treat both the named numpad-style keys and the shifted/non-shifted
    // top-row variants as zoom controls. `=` is `+` without shift.
    match &ev.logical_key {
        Key::Character(s) => {
            for c in s.chars() {
                match c {
                    '+' | '=' => scene.apply_zoom(ZOOM_STEP),
                    '-' | '_' => scene.apply_zoom(1.0 / ZOOM_STEP),
                    _ => {}
                }
            }
        }
        Key::Named(NamedKey::ArrowUp) => scene.apply_zoom(ZOOM_STEP),
        Key::Named(NamedKey::ArrowDown) => scene.apply_zoom(1.0 / ZOOM_STEP),
        _ => {}
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("=== App::resumed (window already? {})", self.window.is_some());
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(format!(
                "IsometricWorldGenerator [{}]",
                self.config.renderer.as_str()
            ))
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_W, WINDOW_H));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                println!("=== create_window FAILED: {e}");
                log::error!("create_window failed: {e}");
                event_loop.exit();
                return;
            }
        };
        println!("=== window created  inner_size={:?}", window.inner_size());
        let mut engine = match Engine::new(&window, self.config.clone()) {
            Ok(e) => e,
            Err(e) => {
                println!("=== Engine::new FAILED: {e:#}");
                log::error!("engine init failed: {e:#}");
                event_loop.exit();
                return;
            }
        };
        println!("=== Engine::new ok");
        match Scene::new(&mut engine) {
            Ok(scene) => self.scene = Some(scene),
            Err(e) => {
                println!("=== Scene::new FAILED: {e:#}");
                log::error!("scene init failed: {e:#}");
                event_loop.exit();
                return;
            }
        }
        println!("=== Scene::new ok");
        self.engine = Some(engine);
        // Kick the redraw loop. On macOS / winit 0.30 the `RedrawRequested`
        // event isn't fired automatically after window creation; we have to
        // request the first one ourselves. Each subsequent frame requests
        // the next one inside `RedrawRequested`, so we self-perpetuate.
        println!("=== resumed: requesting first redraw");
        log::info!("App::resumed — requesting first redraw");
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        // First-event diagnostic: print exactly once per event variant we
        // care about so we can confirm winit is feeding us events.
        match &event {
            WindowEvent::RedrawRequested => {
                static FIRST: std::sync::OnceLock<()> = std::sync::OnceLock::new();
                if FIRST.set(()).is_ok() {
                    println!("=== first WindowEvent::RedrawRequested received");
                }
            }
            WindowEvent::Resized(size) => {
                println!("=== WindowEvent::Resized {}x{}", size.width, size.height);
            }
            WindowEvent::CloseRequested => {
                println!("=== WindowEvent::CloseRequested");
            }
            _ => {}
        }
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.on_resize((size.width, size.height));
                }
                if let Some(scene) = self.scene.as_mut() {
                    scene.on_resize(size.width, size.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(scene) = self.scene.as_mut() {
                    scene.cursor = position;
                    if scene.drag_anchor.is_some() {
                        scene.drag_to(position);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    if let Some(scene) = self.scene.as_mut() {
                        match state {
                            ElementState::Pressed => {
                                scene.drag_anchor = Some(scene.cursor);
                            }
                            ElementState::Released => {
                                scene.drag_anchor = None;
                            }
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event: ke, .. } => {
                handle_key(self.scene.as_mut(), ke);
            }
            WindowEvent::RedrawRequested => {
                let dt = self.clock.tick();
                // Rebuild the FPS overlay if the text changed (once per
                // second-ish). The check is cheap; we do it every frame.
                if let Some(scene) = self.scene.as_mut() {
                    scene.rebuild_fps_overlay(self.clock.fps());
                }
                if let (Some(engine), Some(scene)) =
                    (self.engine.as_mut(), self.scene.as_mut())
                {
                    // First-frame sort debug: compute sort order and log it.
                    if !scene.sort_logged {
                        scene.sort_logged = true;
                        let bounds: Vec<_> = scene.world_sprites.iter()
                            .map(|s| s.iso_bounds()).collect();
                        let order = engine.render.sorter.sort(&bounds);
                        log_sort_debug(&order, &scene.world_sprites);
                    }
                    // Build the mesh array and capture the camera ref
                    // *before* taking the two `&mut` borrows of the sprite
                    // vecs. The borrow checker accepts this because every
                    // field on Scene is a distinct memory location, and
                    // we never overlap a shared borrow with a mutable one.
                    let meshes: [&SpriteMesh; 4] = [
                        &scene.tile_mesh,
                        &scene.building_mesh,
                        &scene.font_mesh,
                        &scene.wolf_mesh,
                    ];
                    let camera = &scene.camera;
                    let world: &mut [Sprite] = &mut scene.world_sprites;
                    let gui: &mut [Sprite] = &mut scene.gui_sprites;
                    if let Err(e) = engine.draw_frame(
                        &window,
                        dt,
                        camera,
                        &meshes,
                        world,
                        gui,
                    ) {
                        log::error!("draw_frame: {e:#}");
                        event_loop.exit();
                    }
                }
                // Keep the title FPS indicator too — easier to glance at
                // when the in-window overlay is off-screen during panning.
                if self.last_fps_print.elapsed() >= Duration::from_millis(500) {
                    let title = format!(
                        "IsometricWorldGenerator [{}]  {:.0} fps  ({} tiles + {} bldgs + {} wolves)",
                        self.config.renderer.as_str(),
                        self.clock.fps(),
                        GRID_W * GRID_H,
                        BUILDING_COUNT,
                        WOLF_COUNT,
                    );
                    window.set_title(&title);
                    self.last_fps_print = Instant::now();
                }
                window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Drop scene before engine so its GPU resources tear down with a
        // valid device, then engine before window so Vulkan teardown happens
        // with a valid surface.
        if let (Some(scene), Some(engine)) = (self.scene.as_mut(), self.engine.as_ref()) {
            scene.destroy(engine);
        }
        self.scene = None;
        self.engine = None;
        self.window = None;
    }
}
