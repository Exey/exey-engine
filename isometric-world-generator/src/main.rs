//! IsometricWorldGenerator — demo for ExeyEngine.
//!
//! M1 scope: open a window, clear-color via Vulkan dynamic rendering,
//! print FPS to the console title bar each second. Renderer choice via
//! `--renderer simple|batch|bigbuffer` CLI flag (passed through from `run.sh`).
//! Subsequent milestones add the actual sprite drawing, iso math, sort,
//! TMX loader, map generator, and pathfinding.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use exey_engine::{Engine, EngineConfig, FrameClock, RendererKind};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

/// CLI args. Tiny hand-rolled parser — pulling in `clap` for one flag is
/// overkill at M1. Add more flags as the demo grows.
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
                            eprintln!("unknown renderer '{v}', falling back to default");
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

fn main() -> Result<()> {
    // env_logger respects RUST_LOG env var. Default to info so users see
    // GPU pick / swapchain info on stdout.
    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }
    env_logger::init();

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
        clock: FrameClock::new(),
        last_fps_print: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    config: EngineConfig,
    /// Wrapped in Arc so winit can keep its own ref alongside our reads.
    /// Created lazily in `resumed` per winit 0.30 idiom.
    window: Option<Arc<Window>>,
    engine: Option<Engine>,
    clock: FrameClock,
    last_fps_print: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(format!(
                "IsometricWorldGenerator [{}]",
                self.config.renderer.as_str()
            ))
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("create_window failed: {e}");
                event_loop.exit();
                return;
            }
        };
        match Engine::new(&window, self.config.clone()) {
            Ok(engine) => {
                self.engine = Some(engine);
            }
            Err(e) => {
                log::error!("engine init failed: {e:#}");
                event_loop.exit();
                return;
            }
        }
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
                let _ = dt; // M2+: pass to a time-step function.
                if let Some(engine) = self.engine.as_mut() {
                    if let Err(e) = engine.draw_frame(&window) {
                        log::error!("draw_frame: {e:#}");
                        event_loop.exit();
                    }
                }
                // Print FPS in the window title once per second. This is the
                // M1 FPS indicator; an in-window text overlay arrives in M8.
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
        // Drop engine before window so Vulkan teardown happens with a valid surface.
        self.engine = None;
        self.window = None;
    }
}
