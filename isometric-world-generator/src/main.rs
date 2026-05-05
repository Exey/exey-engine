//! IsometricWorldGenerator — demo for ExeyEngine.
//!
//! M3 scope: a flock of textured quads bouncing off the window edges.
//! All sprites share one mesh + descriptor + pipeline; the demo updates
//! sprite state (position via velocity, sign-flip on edge contact) per
//! frame and the renderer issues one push-constant + draw per sprite.
//! FPS in the title bar each second. Renderer choice via
//! `--renderer simple|batch|bigbuffer` CLI flag (passed through from
//! `run.sh`). All three currently produce the same output — Batch and
//! BigBuffer ship their real algorithms in M5/M6.
//!
//! Texture is a procedural 64×64 magenta/black checkerboard. M9 replaces
//! it with the scrabling tileset (or any PNG dropped into `assets/`) once
//! the TMX loader and map generator land.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use exey_engine::{Engine, EngineConfig, FrameClock, RendererKind, Sprite, SpriteMesh, Texture};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

/// Sprite count for the bouncing flock. Roughly the M3 deliverable size —
/// big enough to stress the per-draw push-constant path, small enough that
/// `SimpleRenderer` (one draw per sprite) still hits 60+ fps trivially.
const FLOCK_SIZE: usize = 32;
/// Edge length of each sprite, in framebuffer pixels.
const SPRITE_PX: f32 = 64.0;
/// Min/max initial speed magnitude per axis, in pixels per second. Sign is
/// chosen at random, so actual range is [-MAX..-MIN] ∪ [MIN..MAX].
const SPEED_MIN: f32 = 80.0;
const SPEED_MAX: f32 = 220.0;

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

/// Tiny linear congruential generator for deterministic seeded randomness.
/// Pulling in `rand` for one initialization pass would be overkill; this
/// is good enough to scatter sprite positions and velocities.
///
/// Constants from Numerical Recipes (LCG with full period over u32).
struct Lcg(u32);
impl Lcg {
    fn new(seed: u32) -> Self {
        // Avoid zero-state which would lock the LCG at zero.
        Self(if seed == 0 { 0x1234_5678 } else { seed })
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    /// Uniform float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // Take the high 24 bits so the result fits in f32's mantissa
        // without precision loss.
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Uniform float in [lo, hi).
    fn next_f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
    /// +1.0 or -1.0 with equal probability.
    fn next_sign(&mut self) -> f32 {
        if self.next_u32() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

struct Scene {
    /// Shared GPU geometry + texture descriptor for the flock. All sprites
    /// share this — the per-sprite differences travel through push constants.
    mesh: SpriteMesh,
    /// Texture backing the mesh. Kept here so its drop runs when the scene
    /// drops; the descriptor in `mesh` references it.
    texture: Texture,
    /// CPU state for each sprite. Mutated each frame by `tick`.
    sprites: Vec<Sprite>,
}

impl Scene {
    fn new(engine: &Engine) -> Result<Self> {
        log::info!("Scene::new — building procedural checkerboard + flock of {FLOCK_SIZE} sprites");

        // 64×64 magenta-on-black checkerboard, 8×8-pixel cells. Bright enough
        // to be unmistakable when rendered, generic enough to also serve as a
        // sanity check for sampler / format configuration.
        let tile_size: u32 = 64;
        let cell: u32 = 8;
        let mut rgba = vec![0u8; (tile_size * tile_size * 4) as usize];
        for y in 0..tile_size {
            for x in 0..tile_size {
                let on = ((x / cell) + (y / cell)) % 2 == 0;
                let i = ((y * tile_size + x) * 4) as usize;
                if on {
                    rgba[i] = 255; // R
                    rgba[i + 1] = 0; // G
                    rgba[i + 2] = 255; // B
                    rgba[i + 3] = 255; // A
                } else {
                    rgba[i] = 16;
                    rgba[i + 1] = 16;
                    rgba[i + 2] = 16;
                    rgba[i + 3] = 255;
                }
            }
        }

        let texture =
            Texture::from_rgba(&engine.instance, &engine.device, tile_size, tile_size, &rgba)?;

        let mesh = SpriteMesh::unit_quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &texture,
        )?;

        // Spawn the flock with a deterministic seed so successive runs
        // produce the same starting layout — useful for debugging.
        let mut rng = Lcg::new(0xC0FFEE);
        // Use the *logical* window dimensions to bound spawn positions.
        // The renderer scales by the actual framebuffer extent at draw
        // time, so initial layout looks the same on retina vs non-retina.
        let bound_w = WINDOW_W as f32;
        let bound_h = WINDOW_H as f32;
        let sprites: Vec<Sprite> = (0..FLOCK_SIZE)
            .map(|_| {
                let x = rng.next_f32_in(0.0, bound_w - SPRITE_PX);
                let y = rng.next_f32_in(0.0, bound_h - SPRITE_PX);
                let vx = rng.next_sign() * rng.next_f32_in(SPEED_MIN, SPEED_MAX);
                let vy = rng.next_sign() * rng.next_f32_in(SPEED_MIN, SPEED_MAX);
                // Slight per-sprite tint variation for visual interest.
                // Pure white is the default; we lerp toward a random hue
                // by ~20% so the flock isn't a uniform mass.
                let r = 0.8 + 0.2 * rng.next_f32();
                let g = 0.8 + 0.2 * rng.next_f32();
                let b = 0.8 + 0.2 * rng.next_f32();
                Sprite::new([x, y], [SPRITE_PX, SPRITE_PX], [vx, vy], [r, g, b, 1.0])
            })
            .collect();

        log::info!("Scene::new — done (sprites in logical {bound_w}x{bound_h})");
        Ok(Self { mesh, texture, sprites })
    }

    /// Advance one frame: integrate velocity, bounce off the framebuffer
    /// edges. `extent` is the *physical* framebuffer size in pixels —
    /// what the renderer actually rasterizes against. We bounce in that
    /// space because bouncing in logical pixels would let sprites drift
    /// off-screen on retina displays where physical > logical.
    fn tick(&mut self, dt: f32, extent: (u32, u32)) {
        let w = extent.0 as f32;
        let h = extent.1 as f32;
        for s in self.sprites.iter_mut() {
            s.pos[0] += s.velocity[0] * dt;
            s.pos[1] += s.velocity[1] * dt;
            // Clamp to the bounce arena and reverse velocity on contact.
            // We test both sides; a sprite small enough to overshoot per
            // frame would still resolve correctly because we re-clamp.
            if s.pos[0] < 0.0 {
                s.pos[0] = 0.0;
                s.velocity[0] = -s.velocity[0];
            } else if s.pos[0] + s.size[0] > w {
                s.pos[0] = w - s.size[0];
                s.velocity[0] = -s.velocity[0];
            }
            if s.pos[1] < 0.0 {
                s.pos[1] = 0.0;
                s.velocity[1] = -s.velocity[1];
            } else if s.pos[1] + s.size[1] > h {
                s.pos[1] = h - s.size[1];
                s.velocity[1] = -s.velocity[1];
            }
        }
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
            }
            WindowEvent::RedrawRequested => {
                let dt = self.clock.tick();
                if let (Some(engine), Some(scene)) =
                    (self.engine.as_mut(), self.scene.as_mut())
                {
                    // Advance simulation in framebuffer-pixel space so the
                    // bounce arena matches the actual visible region (retina-
                    // aware). Cap dt to avoid sprites teleporting if the
                    // event loop hiccuped — 50 ms = 1 frame at 20 fps.
                    let dt = dt.min(0.05);
                    let extent = (
                        engine.swapchain.extent.width,
                        engine.swapchain.extent.height,
                    );
                    scene.tick(dt, extent);
                    if let Err(e) = engine.draw_frame(&window, &scene.mesh, &scene.sprites) {
                        log::error!("draw_frame: {e:#}");
                        event_loop.exit();
                    }
                }
                // Print FPS in the window title once per second. This is the
                // M1/M2 FPS indicator; an in-window text overlay arrives in M8.
                if self.last_fps_print.elapsed() >= Duration::from_millis(500) {
                    let title = format!(
                        "IsometricWorldGenerator [{}]  {:.0} fps",
                        self.config.renderer.as_str(),
                        self.clock.fps()
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
