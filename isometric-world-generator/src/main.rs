//! IsometricWorldGenerator — demo for ExeyEngine.
//!
//! M5 scope: a 32×32 grid of iso tiles **plus** ~24 scattered 2×2 buildings,
//! drawn through an `IsometricCamera2D` with `IsometricRectangleSorter`
//! providing correct depth ordering. Mouse drag pans the camera; the +/-
//! (and `=` / `_` for keyboards without a numpad) keys zoom in and out.
//! An on-screen FPS counter renders in the top-left via a tiny embedded
//! bitmap font.
//!
//! Three textures:
//! * Tile diamond (procedural, 64×32) — 1024 tiles
//! * Building 2×2 (procedural, taller than a tile so occlusion is visible)
//! * Font atlas (embedded const, 16 glyphs in a horizontal strip)
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

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use exey_engine::glam::Vec2;
use exey_engine::{
    Engine, EngineConfig, FrameClock, ICamera2D, IsometricCamera2D, RendererKind, Sprite,
    SpriteMesh, Texture, iso,
};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

mod font;
use font::FontAtlas;

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

/// Number of 2×2 buildings to scatter. Pseudo-random placement, but
/// deterministic per-run (LCG seed). Each building has a 2×2 iso
/// footprint so the sorter has interesting occlusion to resolve.
const BUILDING_COUNT: usize = 24;

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

/// Tiny LCG for deterministic building placement.
struct Lcg(u32);
impl Lcg {
    fn new(seed: u32) -> Self { Self(if seed == 0 { 1 } else { seed }) }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn next_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u32() as usize) % (hi - lo)
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

    /// Vertical pixel height the building sprite extends above its base.
    /// Stored so we can compute the building's sprite quad size and the
    /// vertical offset to place its base diamond at the right tile.
    building_body_extra: f32,

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

    /// Last-known FPS string content; we rebuild gui_sprites only when it
    /// changes (each second) to avoid per-frame allocation.
    last_fps_text: String,
}

impl Scene {
    fn new(engine: &Engine) -> Result<Self> {
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

        // -- Build world: tiles in row-major, then buildings on random cells --
        let mut world_sprites = Vec::with_capacity(GRID_W * GRID_H + BUILDING_COUNT);
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

        // Buildings on deterministic-random 2x2 anchors. We track placed
        // cells to avoid stacking.
        let mut lcg = Lcg::new(0xBEEF);
        let mut placed_cells: Vec<(usize, usize)> = Vec::new();
        let building_sprite_w = TILE_W * 2.0;
        let building_sprite_h = TILE_H * 2.0 + body_extra as f32;
        for _ in 0..BUILDING_COUNT {
            // Try a few times to find a clear 2x2 spot. If we exhaust
            // attempts, drop this building — the sorter doesn't care.
            let mut placed = false;
            for _try in 0..32 {
                let gx = lcg.next_range(1, GRID_W - 2);
                let gy = lcg.next_range(1, GRID_H - 2);
                // Skip if any placed building's anchor is within 2 cells
                // (rough collision avoidance — accepts adjacency but not
                // overlap of the 2x2 footprints).
                let too_close = placed_cells.iter().any(|&(px, py)| {
                    (gx as isize - px as isize).abs() < 2
                        && (gy as isize - py as isize).abs() < 2
                });
                if too_close { continue; }
                placed_cells.push((gx, gy));
                // The building's iso anchor (front corner) is at
                // (gx+1, gy+1) — the front corner of a 2x2 footprint
                // anchored with its back corner at (gx, gy).
                let front_gx = gx + 1;
                let front_gy = gy + 1;
                let g = Vec2::new(front_gx as f32, front_gy as f32);
                let world_front = iso::logic_to_world(g, TILE_H);
                // World position of the building's sprite top-left:
                //   x: front_world.x - building_sprite_w/2   (centre under front corner... no)
                // Actually we want the building's *base* diamond to land
                // on tiles [gx..gx+1, gy..gy+1]. The base diamond spans
                // world-x [world(gx, gy+1).x .. world(gx+1, gy).x], which
                // is [-tile_h..+tile_h] = [-tile_w/2..+tile_w/2] relative
                // to world(gx, gy). The diamond's top corner sits at
                // world(gx, gy) - (tile_w/2, 0) — no, world(gx,gy) is the
                // top corner of cell (gx, gy)'s diamond.
                //
                // Simpler: the building's base diamond's top corner
                // (highest y) is at world((gx, gy)). The building sprite
                // extends from that point: y starts at world(gx,gy).y -
                // body_extra (the top of the body) and width is 2*tile_w
                // centred on world(gx,gy).x.
                let anchor = iso::logic_to_world(Vec2::new(gx as f32, gy as f32), TILE_H);
                let pos = [
                    anchor.x - building_sprite_w * 0.5,
                    anchor.y - body_extra as f32,
                ];
                let tint = [1.0, 1.0, 1.0, 1.0];
                let mut s = Sprite::new(pos, [building_sprite_w, building_sprite_h], [0.0, 0.0], tint);
                // Sort using the *front* iso corner of the 2x2 footprint
                // and size 2x2 so iso_x1/y1 = front-2 = back corner.
                s.iso_grid = [front_gx as f32, front_gy as f32];
                s.iso_grid_size = [2.0, 2.0];
                s.mesh_idx = MESH_BUILDING;
                world_sprites.push(s);
                placed = true;
                break;
            }
            let _ = placed;
        }
        log::info!(
            "Scene::new — placed {} buildings (target {BUILDING_COUNT})",
            placed_cells.len(),
        );

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
            building_body_extra: body_extra as f32,
            world_sprites,
            gui_sprites: Vec::new(),
            camera,
            drag_anchor: None,
            cursor: PhysicalPosition::new(0.0, 0.0),
            last_fps_text: String::new(),
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
    }

    fn destroy(&mut self, engine: &Engine) {
        engine.device.wait_idle();
        self.tile_mesh.destroy(&engine.device);
        self.tile_texture.destroy(&engine.device);
        self.building_mesh.destroy(&engine.device);
        self.building_texture.destroy(&engine.device);
        self.font_mesh.destroy(&engine.device);
        self.font_texture.destroy(&engine.device);
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
        let engine = match Engine::new(&window, self.config.clone()) {
            Ok(e) => e,
            Err(e) => {
                println!("=== Engine::new FAILED: {e:#}");
                log::error!("engine init failed: {e:#}");
                event_loop.exit();
                return;
            }
        };
        println!("=== Engine::new ok");
        match Scene::new(&engine) {
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
                let _dt = self.clock.tick();
                // Rebuild the FPS overlay if the text changed (once per
                // second-ish). The check is cheap; we do it every frame.
                if let Some(scene) = self.scene.as_mut() {
                    scene.rebuild_fps_overlay(self.clock.fps());
                }
                if let (Some(engine), Some(scene)) =
                    (self.engine.as_mut(), self.scene.as_ref())
                {
                    let meshes: [&SpriteMesh; 3] =
                        [&scene.tile_mesh, &scene.building_mesh, &scene.font_mesh];
                    if let Err(e) = engine.draw_frame(
                        &window,
                        &scene.camera,
                        &meshes,
                        &scene.world_sprites,
                        &scene.gui_sprites,
                    ) {
                        log::error!("draw_frame: {e:#}");
                        event_loop.exit();
                    }
                }
                // Keep the title FPS indicator too — easier to glance at
                // when the in-window overlay is off-screen during panning.
                if self.last_fps_print.elapsed() >= Duration::from_millis(500) {
                    let title = format!(
                        "IsometricWorldGenerator [{}]  {:.0} fps  ({} tiles + {} bldgs)",
                        self.config.renderer.as_str(),
                        self.clock.fps(),
                        GRID_W * GRID_H,
                        BUILDING_COUNT,
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
