//! IsometricWorldGenerator — demo for ExeyEngine.
//!
//! M4 scope: a 32×32 grid of iso-projected tiles drawn through an
//! `IsometricCamera2D`. Tiles are placed by `iso::logic_to_world` so the
//! grid renders as a flat diamond. The camera auto-fits the grid into
//! the viewport (zoom + centred position computed at scene init), and
//! its viewport is updated whenever the window resizes.
//!
//! No motion in M4 — no flock, no pan. The milestone proves the iso
//! math and the camera plumbing. M5 adds the iso-rectangle sorter, at
//! which point we can start placing tiles at different heights and
//! characters that can occlude tiles.
//!
//! Renderer choice via `--renderer simple|batch|bigbuffer` CLI flag
//! (passed through from `run.sh`). All three currently produce the same
//! output — Batch and BigBuffer ship their real algorithms in M5/M6.
//!
//! Texture is a procedural diamond — a filled iso-tile shape with a
//! 1-pixel outline. M9 replaces it with the scrabling tileset (or any
//! PNG dropped into `assets/`) once the TMX loader and map generator
//! land.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use exey_engine::glam::Vec2;
use exey_engine::{
    Engine, EngineConfig, FrameClock, ICamera2D, IsometricCamera2D, RendererKind, Sprite,
    SpriteMesh, Texture, iso,
};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Iso tile size, in world pixels. The "2:1 isometric" convention has
/// `tile_w = 2 * tile_h`. Only `tile_h` participates in the projection
/// math; `tile_w` is the rendered sprite width.
const TILE_W: f32 = 64.0;
const TILE_H: f32 = 32.0;

/// Grid dimensions. 32×32 = 1024 tiles. SimpleRenderer issues one draw
/// per tile, so this exercises the per-frame push-constant path at a
/// scale where M5/M6 will start to win.
const GRID_W: usize = 32;
const GRID_H: usize = 32;

/// Visual margin between auto-fit zoom and the actual viewport edges,
/// expressed as a multiplicative factor on the computed zoom. 0.95
/// leaves a small breathing room so the corner tiles aren't flush against
/// the window edge.
const ZOOM_FIT_MARGIN: f32 = 0.95;

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
///
/// Diamond test: a pixel `(x, y)` is inside the iso diamond iff
/// `|x - cx|/cx + |y - cy|/cy <= 1` (centre at `(cx, cy)`, half-extents
/// `(cx, cy)`).
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
                // Inside the diamond. Outline ring near d=1, fill below.
                let outline_zone = 0.94;
                if d > outline_zone {
                    // Dark outline (almost black with a hint of warmth).
                    rgba[i]     = 32;
                    rgba[i + 1] = 24;
                    rgba[i + 2] = 16;
                    rgba[i + 3] = 255;
                } else {
                    // Subtle vertical gradient inside the tile so adjacent
                    // tiles are visually distinguishable when packed.
                    let t = y as f32 / tile_h as f32;
                    let lo = 0.55;
                    let hi = 0.85;
                    let v = lo + (hi - lo) * (1.0 - t);
                    rgba[i]     = (110.0 * v) as u8;  // R: muted earthy
                    rgba[i + 1] = (165.0 * v) as u8;  // G: greenish
                    rgba[i + 2] = (95.0 * v) as u8;   // B: low
                    rgba[i + 3] = 255;
                }
            } else {
                // Outside the diamond — fully transparent.
                rgba[i] = 0;
                rgba[i + 1] = 0;
                rgba[i + 2] = 0;
                rgba[i + 3] = 0;
            }
        }
    }
    rgba
}

struct Scene {
    /// Shared GPU geometry + texture descriptor. All tiles share this.
    mesh: SpriteMesh,
    /// Texture backing the mesh (the diamond). Kept here so its drop runs
    /// when the scene drops; the descriptor in `mesh` references it.
    texture: Texture,
    /// One [`Sprite`] per grid cell, in row-major order
    /// `idx = grid_y * GRID_W + grid_x`.
    sprites: Vec<Sprite>,
    /// The iso-projected world camera. Demo-owned so we can update its
    /// viewport on resize and (later) pan/zoom.
    camera: IsometricCamera2D,
}

impl Scene {
    fn new(engine: &Engine) -> Result<Self> {
        log::info!(
            "Scene::new — building {GRID_W}×{GRID_H} iso grid \
             ({} tiles, tile {TILE_W}×{TILE_H})",
            GRID_W * GRID_H,
        );

        // Diamond texture sized to the iso tile: width = 2 * tile_h,
        // height = tile_h. Use power-of-two-friendly multiples just in
        // case (current sampler is NEAREST so no mipmaps; this stays
        // OK for pixel-art-style tiles).
        let tex_w = TILE_W as u32;
        let tex_h = TILE_H as u32;
        let rgba = build_diamond_rgba(tex_w, tex_h);
        let texture =
            Texture::from_rgba(&engine.instance, &engine.device, tex_w, tex_h, &rgba)?;

        let mesh = SpriteMesh::unit_quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &texture,
        )?;

        // Build the grid. Each tile's sprite top-left is the diamond's
        // top-left bounding-box corner, so we render the iso diamond
        // texture in a quad of size (tile_w × tile_h) at world position
        //
        //   sprite_pos = iso::logic_to_world(grid, tile_h) - (tile_w/2, 0)
        //
        // The `-(tile_w/2, 0)` shifts the diamond so its *top* corner
        // lands at the iso world position (which is what the iso math
        // returns — the canonical AS3 convention).
        //
        // Light per-tile tint variation so the grid doesn't look flat;
        // we modulate by (gx + gy) parity for a checkerboard hint and
        // a small linear gradient across the grid.
        let mut sprites = Vec::with_capacity(GRID_W * GRID_H);
        for gy in 0..GRID_H {
            for gx in 0..GRID_W {
                let g = Vec2::new(gx as f32, gy as f32);
                let world = iso::logic_to_world(g, TILE_H);
                let pos = [world.x - TILE_W * 0.5, world.y];
                // Tint: parity-based light/dark + a small gradient so we
                // can visually trace rows/columns when debugging.
                let parity = (gx + gy) % 2 == 0;
                let base = if parity { 1.0 } else { 0.85 };
                let tint = [
                    base * (0.85 + 0.15 * (gx as f32 / GRID_W as f32)),
                    base,
                    base * (0.85 + 0.15 * (gy as f32 / GRID_H as f32)),
                    1.0,
                ];
                sprites.push(Sprite::new(pos, [TILE_W, TILE_H], [0.0, 0.0], tint));
            }
        }
        log::info!("Scene::new — built {} sprites", sprites.len());

        // Compute camera state. Centre on the grid's bounding-box centre
        // and zoom to fit the diamond into the viewport with a margin.
        //
        // Grid bounding box in world space (32×32 grid, tile_h=32):
        //   gx=GRID_W-1, gy=0       → world (+992, +496)   max world_x
        //   gx=0,        gy=GRID_H-1→ world (-992, +496)   min world_x
        //   gx=0,        gy=0       → world (   0,    0)   min world_y
        //   gx=GRID_W-1, gy=GRID_H-1→ world (   0, +992)   max world_y
        // Each rendered tile sprite extends `tile_w/2` further in ±x
        // (sprite top-left is `world - tile_w/2`) and `tile_h` further
        // in +y (sprite spans `[world_y, world_y + tile_h]`).
        // So rendered bbox: x ∈ [-(GRID_H-1)*tile_h - tile_w/2 ..
        //                          +(GRID_W-1)*tile_h + tile_w/2],
        //                   y ∈ [0 .. (GRID_W+GRID_H-2)*tile_h/2 + tile_h].
        let world_w =
            TILE_H * 2.0 * (GRID_W.max(GRID_H) as f32 - 1.0) + TILE_W;
        let world_h =
            TILE_H * (GRID_W + GRID_H - 2) as f32 * 0.5 + TILE_H;
        let centre_x = 0.0; // diamond is symmetric in x around 0
        let centre_y = world_h * 0.5;

        let viewport = Vec2::new(WINDOW_W as f32, WINDOW_H as f32);
        let zoom_x = viewport.x / world_w;
        let zoom_y = viewport.y / world_h;
        let zoom = zoom_x.min(zoom_y) * ZOOM_FIT_MARGIN;

        let mut camera = IsometricCamera2D::new();
        camera.state.position = Vec2::new(centre_x, centre_y);
        camera.state.zoom = zoom;
        camera.state.viewport = viewport;
        log::info!(
            "Scene::new — camera centre=({centre_x:.1},{centre_y:.1}) zoom={zoom:.4} \
             viewport={WINDOW_W}×{WINDOW_H}  world_bbox≈{world_w:.0}×{world_h:.0}",
        );

        Ok(Self { mesh, texture, sprites, camera })
    }

    /// Update the camera viewport when the window resizes. The camera's
    /// auto-fit zoom isn't recomputed — keeping the original zoom means
    /// the grid stays at the same world-pixel scale, just with more or
    /// less margin around it. Re-fitting is a one-line change if we want
    /// it later.
    fn on_resize(&mut self, width: u32, height: u32) {
        self.camera.state.viewport = Vec2::new(width as f32, height as f32);
    }

    fn destroy(&mut self, engine: &Engine) {
        // Wait for the GPU before tearing down sprite resources — they may
        // still be in flight from the most recent frame. The engine's own
        // Drop will wait again before destroying the swapchain etc.
        engine.device.wait_idle();
        self.mesh.destroy(&engine.device);
        self.texture.destroy(&engine.device);
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
                // Camera viewport tracks the framebuffer extent so the
                // world→clip transform stays correct after the swapchain
                // recreates.
                if let Some(scene) = self.scene.as_mut() {
                    scene.on_resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                let _dt = self.clock.tick(); // M4: no scene motion; kept for the FPS counter.
                if let (Some(engine), Some(scene)) =
                    (self.engine.as_mut(), self.scene.as_ref())
                {
                    if let Err(e) =
                        engine.draw_frame(&window, &scene.camera, &scene.mesh, &scene.sprites)
                    {
                        log::error!("draw_frame: {e:#}");
                        event_loop.exit();
                    }
                }
                // Print FPS in the window title once per second. This is the
                // M1/M2 FPS indicator; an in-window text overlay arrives in M8.
                if self.last_fps_print.elapsed() >= Duration::from_millis(500) {
                    let title = format!(
                        "IsometricWorldGenerator [{}]  {:.0} fps  ({} tiles)",
                        self.config.renderer.as_str(),
                        self.clock.fps(),
                        GRID_W * GRID_H,
                    );
                    window.set_title(&title);
                    self.last_fps_print = Instant::now();
                }
                // Continuous redraw — we want a real game loop, not redraws-on-event.
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
