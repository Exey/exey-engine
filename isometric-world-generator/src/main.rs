//! IsometricWorldGenerator — demo for ExeyEngine.
//!
//! M2 scope: open a window, dynamic-rendering clear + one textured quad in
//! the centre, FPS in the title bar each second. Renderer choice via
//! `--renderer simple|batch|bigbuffer` CLI flag (passed through from `run.sh`).
//! All three currently produce the same output — Batch and BigBuffer ship
//! their real algorithms in M5/M6.
//!
//! Texture is a procedural 64×64 magenta/black checkerboard. M9 replaces it
//! with the scrabling tileset (or any PNG dropped into `assets/`) once the
//! TMX loader and map generator land.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use exey_engine::{Engine, EngineConfig, FrameClock, RendererKind, Sprite, Texture};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_W: u32 = 1280;
const WINDOW_H: u32 = 720;

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

/// The demo-owned GPU resources that need to outlive any single frame.
/// Drop order matters: descriptor set is freed implicitly with the engine's
/// descriptor pool, so `Sprite` (which holds the descriptor set handle and
/// the vertex/index buffers) is destroyed manually below before the engine.
struct Scene {
    sprite: Sprite,
    texture: Texture,
}

impl Scene {
    fn new(engine: &Engine) -> Result<Self> {
        log::info!("Scene::new — building procedural checkerboard");

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

        // Centre a 256×256 quad on the window. Pixel coords; +Y down.
        let quad_w = 256.0_f32;
        let quad_h = 256.0_f32;
        let x = (WINDOW_W as f32 - quad_w) * 0.5;
        let y = (WINDOW_H as f32 - quad_h) * 0.5;

        let sprite = Sprite::quad(
            &engine.instance,
            &engine.device,
            &engine.render.sprite_pipeline,
            &texture,
            x,
            y,
            quad_w,
            quad_h,
        )?;

        log::info!("Scene::new — done");
        Ok(Self { sprite, texture })
    }

    fn destroy(&mut self, engine: &Engine) {
        // Wait for the GPU before tearing down sprite resources — they may
        // still be in flight from the most recent frame. The engine's own
        // Drop will wait again before destroying the swapchain etc.
        engine.device.wait_idle();
        self.sprite.destroy(&engine.device);
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
                let _ = dt; // M3+: pass to a time-step function.
                if let (Some(engine), Some(scene)) =
                    (self.engine.as_mut(), self.scene.as_ref())
                {
                    let sprites: [&Sprite; 1] = [&scene.sprite];
                    if let Err(e) = engine.draw_frame(&window, &sprites) {
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
