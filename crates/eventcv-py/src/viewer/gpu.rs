//! GPU backend for the viewer (wgpu → Metal / Vulkan / DX12). Owns the window and draw
//! loop; [`super`] hands it a [`Scene`]. Clouds render as instanced billboard splats with a
//! depth buffer and orbit controls; images as a single textured quad.
//!
//! The winit `EventLoop` is a process-global (a platform can hold only one), reused across
//! `view()` calls via `run_app_on_demand`. It must run on the main thread — the same
//! constraint the old minifb viewer had.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowId};

use super::roi::MaskEditor;
use super::{CloudPoint, Scene};

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.025,
    b: 0.05,
    a: 1.0,
};
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const AXIS_X: u32 = 0xffca3a;
const AXIS_Y: u32 = 0x8ac926;
const AXIS_Z: u32 = 0xc77dff;
/// Base splat half-size as a fraction of the viewport height.
const POINT_SIZE: f32 = 0.006;
const CAMERA_DISTANCE: f32 = 3.4;
const MIN_DISTANCE: f32 = 1.3;
const MAX_DISTANCE: f32 = 12.0;
const DEFAULT_PITCH: f32 = -0.55;
/// Live viewer refresh cadence (~60 FPS).
const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

thread_local! {
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

/// Opens a window and renders `scene` until the user closes it (Esc / window close).
pub(crate) fn run(scene: Scene) -> Result<(), String> {
    run_app(App::new(scene))
}

/// Opens a window and displays a stream of frames, pulling a fresh one from `producer` every
/// `interval` until the user closes the window or `producer` returns an error. The producer renders
/// on this (main) thread — it returns `Ok(Some(image))` for a new frame, `Ok(None)` when nothing
/// changed since the last call, or `Err` to abort. Reuses the same process-global event loop as
/// [`run`], so a session mixes `.view()` and streaming freely (one window at a time).
///
/// Not camera-specific: a recorded file played back at a chosen rate is the same loop with a
/// different producer, which is why `interval` is a parameter rather than the live viewer's
/// hardcoded 60 FPS.
pub(crate) fn run_live<P>(
    producer: P,
    width: u32,
    height: u32,
    title: String,
    interval: std::time::Duration,
) -> Result<(), String>
where
    P: FnMut() -> Result<Option<super::Rgb8Image>, String>,
{
    run_app(LiveApp::new(producer, width, height, title, None, interval))
}

/// Like [`run_live`], but `editor` draws an ROI over the displayed frames: pointer drags become
/// shapes and the excluded region is dimmed in place. Returns once the window closes; whether the
/// drawing was accepted is read back from the editor. A `producer` that yields one frame and then
/// `Ok(None)` makes this a still-image editor, which is how a file-derived frame is drawn on.
pub(crate) fn run_draw<P>(
    producer: P,
    width: u32,
    height: u32,
    title: String,
    editor: &mut MaskEditor,
) -> Result<(), String>
where
    P: FnMut() -> Result<Option<super::Rgb8Image>, String>,
{
    run_app(LiveApp::new(
        producer,
        width,
        height,
        title,
        Some(editor),
        FRAME_INTERVAL,
    ))
}

/// Runs an app to completion on the process-global event loop, surfacing whatever error ended it.
fn run_app(mut app: impl ApplicationHandler + Failable) -> Result<(), String> {
    EVENT_LOOP.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(EventLoop::new().map_err(|error| error.to_string())?);
        }
        let event_loop = slot.as_mut().expect("event loop just initialised");
        event_loop
            .run_app_on_demand(&mut app)
            .map_err(|error| error.to_string())?;
        match app.take_error() {
            Some(message) => Err(message),
            None => Ok(()),
        }
    })
}

/// An app that can end on a fatal error, so [`run_app`] can report it.
trait Failable {
    fn take_error(&mut self) -> Option<String>;
}

/// The live counterpart to [`App`]: instead of one static [`Scene`] it pulls a frame from `producer`
/// each ~[`FRAME_INTERVAL`] and re-uploads the image texture in place.
struct LiveApp<'a, P> {
    producer: P,
    width: u32,
    height: u32,
    title: String,
    window: Option<Arc<Window>>,
    render: Option<Render>,
    error: Option<String>,
    /// ROI drawing, when the window was opened by `draw_mask()`. It repaints every displayed frame,
    /// so the newest frame is kept here to composite onto even when the producer has nothing new.
    editor: Option<&'a mut MaskEditor>,
    frame: Option<super::Rgb8Image>,
    /// How long to sleep between producer polls — the display cadence, not the data rate.
    interval: std::time::Duration,
}

impl<P> Failable for LiveApp<'_, P> {
    fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }
}

impl<'a, P> LiveApp<'a, P>
where
    P: FnMut() -> Result<Option<super::Rgb8Image>, String>,
{
    fn new(
        producer: P,
        width: u32,
        height: u32,
        title: String,
        editor: Option<&'a mut MaskEditor>,
        interval: std::time::Duration,
    ) -> Self {
        Self {
            producer,
            width,
            height,
            title,
            window: None,
            render: None,
            error: None,
            editor,
            frame: None,
            interval,
        }
    }

    /// Pushes the newest frame to the GPU, with the ROI overlay painted on when drawing.
    fn present(&mut self) {
        let (Some(render), Some(frame)) = (&mut self.render, &self.frame) else {
            return;
        };
        match &self.editor {
            Some(editor) => {
                let mut overlaid = frame.clone();
                editor.paint(&mut overlaid);
                render.update_image(&overlaid);
            }
            None => render.update_image(frame),
        }
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    /// Routes a window event to the editor. Returns whether it accepted the drawing (`Enter`).
    fn edit(&mut self, event: &WindowEvent) -> bool {
        let size = self.window.as_ref().map(|window| window.inner_size());
        let Some(editor) = &mut self.editor else {
            return false;
        };
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let size = size.unwrap_or(PhysicalSize::new(1, 1));
                editor.cursor_moved(
                    position.x / size.width.max(1) as f64,
                    position.y / size.height.max(1) as f64,
                );
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                editor.shift_held(modifiers.state().shift_key())
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => editor.press(),
                ElementState::Released => editor.release(),
            },
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                return match &event.logical_key {
                    Key::Named(NamedKey::Enter) => editor.key('\r'),
                    Key::Character(text) => text
                        .chars()
                        .next()
                        .is_some_and(|key| editor.key(key.to_ascii_lowercase())),
                    _ => false,
                };
            }
            _ => {}
        }
        false
    }

    /// Sleeps until the next display frame is due.
    fn wait(&self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + self.interval));
    }

    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        self.render = None;
        self.window = None;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.exit();
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, message: String) {
        self.error = Some(message);
        self.shutdown(event_loop);
    }
}

impl<P> ApplicationHandler for LiveApp<'_, P>
where
    P: FnMut() -> Result<Option<super::Rgb8Image>, String>,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // resumed can fire more than once
        }
        // Upscale small sensors so the window is comfortably visible; the texture stays at sensor
        // resolution (nearest sampling) so the events aren't blurred.
        let factor = ((480 + self.height - 1) / self.height.max(1)).clamp(1, 4);
        let title = match self.editor {
            Some(_) => format!("eventcv - {} - {}", self.title, super::roi::LEGEND),
            None => format!("eventcv - {}", self.title),
        };
        let attributes = Window::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(self.width * factor, self.height * factor));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => return self.fail(event_loop, error.to_string()),
        };
        // Start from a black frame at sensor resolution; live frames re-upload it in place.
        let black = super::Rgb8Image {
            width: self.width as usize,
            height: self.height as usize,
            pixels: vec![0; (self.width * self.height * 3) as usize],
        };
        match Render::new(window.clone(), Scene::Image(black)) {
            Ok(render) => {
                self.render = Some(render);
                self.window = Some(window);
                event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + self.interval));
            }
            Err(error) => self.fail(event_loop, error),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Drawing consumes pointer and character input; Esc and close still end the window (as a
        // cancel), and everything below keeps working unchanged when no editor is attached.
        if self.edit(&event) {
            return self.shutdown(event_loop);
        }
        match event {
            WindowEvent::CloseRequested => self.shutdown(event_loop),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                self.shutdown(event_loop)
            }
            WindowEvent::Resized(size) => {
                if let Some(render) = &mut self.render {
                    render.resize(size);
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(render) = &mut self.render {
                    // Image content ignores the camera args.
                    if let Err(error) = render.draw(0.0, 0.0, CAMERA_DISTANCE) {
                        self.fail(event_loop, error);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        match (self.producer)() {
            Ok(Some(image)) => self.frame = Some(image),
            // Nothing new from the producer, but a drag may have moved: an editor still repaints.
            Ok(None) if self.editor.is_none() => return self.wait(event_loop),
            Ok(None) => {}
            Err(message) => return self.fail(event_loop, message),
        }
        self.present();
        self.wait(event_loop);
    }
}

struct App {
    scene: Option<Scene>,
    is_cloud: bool,
    window: Option<Arc<Window>>,
    render: Option<Render>,
    pitch: f32,
    yaw: f32,
    /// Camera distance from the volume centre — smaller = zoomed in (scroll wheel).
    distance: f32,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    error: Option<String>,
}

impl Failable for App {
    fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }
}

impl App {
    fn new(scene: Scene) -> Self {
        let is_cloud = matches!(scene, Scene::Cloud { .. });
        Self {
            scene: Some(scene),
            is_cloud,
            window: None,
            render: None,
            pitch: DEFAULT_PITCH,
            yaw: 0.0,
            distance: CAMERA_DISTANCE,
            dragging: false,
            last_cursor: None,
            error: None,
        }
    }

    /// Tears the window down and asks the loop to return.
    ///
    /// Order matters on macOS: the wgpu `Surface` holds its own `Arc<Window>` clone, so the
    /// window can't be destroyed (and leaves the screen) until the render state is dropped
    /// first. We also switch to `Poll` so `run_app_on_demand` processes the exit immediately
    /// instead of stalling in `Wait` for an event that never arrives.
    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        self.render = None;
        self.window = None;
        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.exit();
    }

    /// Records a fatal error and tears down.
    fn fail(&mut self, event_loop: &ActiveEventLoop, message: String) {
        self.error = Some(message);
        self.shutdown(event_loop);
    }

    fn redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Idle between events; a prior `shutdown` on a reused loop left it in `Poll`.
        event_loop.set_control_flow(ControlFlow::Wait);
        if self.window.is_some() {
            return; // already initialised (resumed can fire more than once)
        }
        let Some(scene) = self.scene.take() else {
            return;
        };

        let (width, height, title) = match &scene {
            Scene::Image(image) => (image.width as u32, image.height as u32, "eventcv"),
            Scene::Cloud { name, .. } => (960, 720, name.as_str()),
        };
        let attributes = Window::default_attributes()
            .with_title(format!("eventcv - {title}"))
            .with_inner_size(LogicalSize::new(width, height));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => return self.fail(event_loop, error.to_string()),
        };

        match Render::new(window.clone(), scene) {
            Ok(render) => {
                self.render = Some(render);
                self.window = Some(window);
                self.redraw();
            }
            Err(error) => self.fail(event_loop, error),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.shutdown(event_loop),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                self.shutdown(event_loop)
            }
            WindowEvent::Resized(size) => {
                if let Some(render) = &mut self.render {
                    render.resize(size);
                }
                self.redraw();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging && self.is_cloud {
                    if let Some((last_x, last_y)) = self.last_cursor {
                        self.yaw += (position.x - last_x) as f32 * 0.008;
                        self.pitch =
                            (self.pitch + (position.y - last_y) as f32 * 0.008).clamp(-1.45, 1.45);
                        self.redraw();
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::MouseWheel { delta, .. } if self.is_cloud => {
                // Positive scroll (wheel up / two-finger up) zooms in. Multiplicative so each
                // notch feels the same at any distance; line vs pixel deltas are normalised.
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                };
                self.distance =
                    (self.distance * (1.0 - scroll * 0.1)).clamp(MIN_DISTANCE, MAX_DISTANCE);
                self.redraw();
            }
            WindowEvent::RedrawRequested => {
                if let Some(render) = &mut self.render {
                    if let Err(error) = render.draw(self.pitch, self.yaw, self.distance) {
                        self.fail(event_loop, error);
                    }
                }
            }
            _ => {}
        }
    }
}

// ---- GPU state -------------------------------------------------------------------------

struct Render {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    content: Content,
}

// Exactly one `Content` exists per window, so the variant size gap costs nothing.
#[allow(clippy::large_enum_variant)]
enum Content {
    Cloud {
        points_pipeline: wgpu::RenderPipeline,
        axes_pipeline: wgpu::RenderPipeline,
        labels_pipeline: wgpu::RenderPipeline,
        uniform: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        quad: wgpu::Buffer,
        instances: wgpu::Buffer,
        instance_count: u32,
        axes: wgpu::Buffer,
        axes_count: u32,
        labels: wgpu::Buffer,
        labels_count: u32,
    },
    Image {
        pipeline: wgpu::RenderPipeline,
        bind_group: wgpu::BindGroup,
        // Kept so the live viewer and the ROI editor can re-upload frames in place (the sensor
        // size is fixed).
        texture: wgpu::Texture,
        width: u32,
        height: u32,
    },
}

/// Pick the best available adapter, degrading gracefully rather than failing on GPU-less machines:
/// a real high-performance GPU first, then any low-power/integrated (or OpenGL) adapter, then
/// wgpu's software fallback (WARP on Windows, llvmpipe/lavapipe on Linux via Mesa). `None` only
/// when the machine has no renderable adapter at all — e.g. a headless box with no Mesa installed.
fn request_adapter(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> Option<wgpu::Adapter> {
    // (power preference, force software fallback). Ordered best → worst.
    let attempts = [
        (wgpu::PowerPreference::HighPerformance, false),
        (wgpu::PowerPreference::LowPower, false),
        (wgpu::PowerPreference::None, true),
    ];
    attempts.into_iter().find_map(|(power_preference, force_fallback_adapter)| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface: Some(surface),
            force_fallback_adapter,
        }))
    })
}

impl Render {
    fn new(window: Arc<Window>, scene: Scene) -> Result<Self, String> {
        // `all()` (not just `PRIMARY`) so machines with no native Vulkan/Metal/DX12 GPU — headless
        // boxes, VMs, remote desktops — can still reach an OpenGL or software adapter and render,
        // slowly, instead of failing outright. A real GPU is still preferred (see `request_adapter`).
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(|error| error.to_string())?;
        let adapter = request_adapter(&instance, &surface)
            .ok_or_else(|| "no compatible GPU or software adapter found".to_owned())?;
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            eprintln!(
                "eventcv: no GPU found — falling back to software rendering ({}); the viewer will be slow.",
                adapter.get_info().name
            );
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("eventcv-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|error| error.to_string())?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth = depth_view(&device, config.width, config.height);

        let content = match scene {
            Scene::Cloud { points, .. } => build_cloud(&device, format, &points),
            Scene::Image(image) => build_image(&device, &queue, format, &image),
        };

        Ok(Self {
            surface,
            device,
            queue,
            config,
            depth,
            content,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth = depth_view(&self.device, size.width, size.height);
    }

    /// Re-uploads a live frame into the image texture in place. Frames whose size differs from the
    /// texture (fixed at the sensor resolution) are ignored, as is a non-image content.
    fn update_image(&mut self, image: &super::Rgb8Image) {
        let Content::Image {
            texture,
            width,
            height,
            ..
        } = &self.content
        else {
            return;
        };
        if image.width as u32 != *width || image.height as u32 != *height {
            return;
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgb_to_rgba(&image.pixels),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(*width * 4),
                rows_per_image: Some(*height),
            },
            wgpu::Extent3d {
                width: *width,
                height: *height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn draw(&mut self, pitch: f32, yaw: f32, distance: f32) -> Result<(), String> {
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            // Reconfigure and skip this frame on a lost/outdated surface.
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        match &self.content {
            Content::Cloud {
                points_pipeline,
                axes_pipeline,
                labels_pipeline,
                uniform,
                bind_group,
                quad,
                instances,
                instance_count,
                axes,
                axes_count,
                labels,
                labels_count,
            } => {
                let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
                let uniforms = Uniforms {
                    view_proj: view_proj(pitch, yaw, distance, aspect),
                    point_size: [POINT_SIZE / aspect, POINT_SIZE],
                    aspect,
                    _pad: 0.0,
                };
                self.queue
                    .write_buffer(uniform, 0, bytemuck::bytes_of(&uniforms));

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("cloud"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(BACKGROUND),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_pipeline(axes_pipeline);
                pass.set_vertex_buffer(0, axes.slice(..));
                pass.draw(0..*axes_count, 0..1);
                pass.set_pipeline(points_pipeline);
                pass.set_vertex_buffer(0, quad.slice(..));
                pass.set_vertex_buffer(1, instances.slice(..));
                pass.draw(0..6, 0..*instance_count);
                // Axis labels last, depth-independent, so they read on top of the cloud.
                pass.set_pipeline(labels_pipeline);
                pass.set_vertex_buffer(0, labels.slice(..));
                pass.draw(0..*labels_count, 0..1);
            }
            Content::Image {
                pipeline,
                bind_group,
                ..
            } => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("image"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}

// ---- Scene → GPU resources -------------------------------------------------------------

fn build_cloud(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    points: &[CloudPoint],
) -> Content {
    let instances: Vec<Instance> = cloud_instances(points);
    let axes = axis_vertices();
    let labels = label_vertices();

    let quad = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("quad"),
        contents: bytemuck::cast_slice(&QUAD),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instances"),
        contents: bytemuck::cast_slice(&instances),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let axes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("axes"),
        contents: bytemuck::cast_slice(&axes),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let labels_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("labels"),
        contents: bytemuck::cast_slice(&labels),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: std::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cloud-uniforms"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cloud-uniforms"),
        layout: &bind_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cloud"),
        source: wgpu::ShaderSource::Wgsl(CLOUD_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cloud"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });

    let points_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("points"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_points",
            buffers: &[QUAD_LAYOUT, INSTANCE_LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_points",
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(depth_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    let axes_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("axes"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_axes",
            buffers: &[AXIS_LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_axes",
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(depth_state()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });
    // Labels are billboarded (screen-facing) and drawn without depth testing so they stay
    // legible on top of the cloud.
    let labels_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("labels"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_labels",
            buffers: &[LABEL_LAYOUT],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_axes",
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: Some(depth_state_overlay()),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    Content::Cloud {
        points_pipeline,
        axes_pipeline,
        labels_pipeline,
        uniform,
        bind_group,
        quad,
        instances: instance_buffer,
        instance_count: instances.len() as u32,
        axes: axes_buffer,
        axes_count: axes.len() as u32,
        labels: labels_buffer,
        labels_count: labels.len() as u32,
    }
}

fn build_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    image: &super::Rgb8Image,
) -> Content {
    let size = wgpu::Extent3d {
        width: image.width as u32,
        height: image.height as u32,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("frame"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgb_to_rgba(&image.pixels),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(image.width as u32 * 4),
            rows_per_image: Some(image.height as u32),
        },
        size,
    );
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("frame"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("image"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("image"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("image"),
        source: wgpu::ShaderSource::Wgsl(IMAGE_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("image"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("image"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_image",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_image",
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    Content::Image {
        pipeline,
        bind_group,
        texture,
        width: image.width as u32,
        height: image.height as u32,
    }
}

/// Converts cloud points to GPU instances, baking size + brightness from each point's
/// strength relative to the busiest (matching the old CPU renderer's look).
fn cloud_instances(points: &[CloudPoint]) -> Vec<Instance> {
    let maximum = points.iter().map(|p| p.strength).fold(0.0_f32, f32::max);
    points
        .iter()
        .map(|point| {
            let relative = if maximum > 0.0 {
                (point.strength / maximum).sqrt()
            } else {
                0.0
            };
            let brightness = 0.45 + relative * 0.55;
            let [r, g, b] = srgb_to_linear(point.color);
            Instance {
                position: [
                    point.x as f32,
                    point.y as f32,
                    point.z as f32,
                    1.0 + relative,
                ],
                color: [r * brightness, g * brightness, b * brightness, 1.0],
            }
        })
        .collect()
}

fn axis_vertices() -> Vec<AxisVertex> {
    let origin = [-1.0, -1.0, -1.0];
    let make = |end: [f32; 3], color: u32| {
        let [r, g, b] = srgb_to_linear(color);
        [
            AxisVertex {
                position: origin,
                color: [r, g, b],
            },
            AxisVertex {
                position: end,
                color: [r, g, b],
            },
        ]
    };
    let mut vertices = Vec::with_capacity(6);
    vertices.extend(make([1.0, -1.0, -1.0], AXIS_X));
    vertices.extend(make([-1.0, 1.0, -1.0], AXIS_Y));
    vertices.extend(make([-1.0, -1.0, 1.0], AXIS_Z));
    vertices
}

/// Billboarded `X` / `Y` / `T` labels at each axis tip: the glyph is a set of screen-space
/// line segments anchored to the (rotating) axis end, so it stays upright and constant-size.
/// The depth axis is time, hence `T`.
fn label_vertices() -> Vec<LabelVertex> {
    // (anchor at the axis tip, screen-space push away from the line, glyph, colour).
    let labels = [
        ([1.0, -1.0, -1.0], [0.045, 0.0], 'X', AXIS_X),
        ([-1.0, 1.0, -1.0], [0.0, 0.05], 'Y', AXIS_Y),
        ([-1.0, -1.0, 1.0], [0.045, 0.0], 'T', AXIS_Z),
    ];
    let mut vertices = Vec::new();
    for (anchor, push, glyph, color) in labels {
        let [r, g, b] = srgb_to_linear(color);
        for [(ax, ay), (bx, by)] in glyph_segments(glyph) {
            // Centre the glyph box on the push point, scaled to `LABEL_SIZE`.
            let place = |gx: f32, gy: f32| {
                [
                    push[0] + (gx - 0.5) * LABEL_SIZE,
                    push[1] + (gy - 0.5) * LABEL_SIZE,
                ]
            };
            vertices.push(LabelVertex {
                anchor,
                offset: place(*ax, *ay),
                color: [r, g, b],
            });
            vertices.push(LabelVertex {
                anchor,
                offset: place(*bx, *by),
                color: [r, g, b],
            });
        }
    }
    vertices
}

/// Line segments for a glyph in a `[0, 1]²` box (origin bottom-left).
fn glyph_segments(glyph: char) -> &'static [[(f32, f32); 2]] {
    match glyph {
        'X' => &[[(0.0, 0.0), (1.0, 1.0)], [(0.0, 1.0), (1.0, 0.0)]],
        'Y' => &[
            [(0.0, 1.0), (0.5, 0.5)],
            [(1.0, 1.0), (0.5, 0.5)],
            [(0.5, 0.5), (0.5, 0.0)],
        ],
        // 'T': top bar + centre stem.
        _ => &[[(0.0, 1.0), (1.0, 1.0)], [(0.5, 1.0), (0.5, 0.0)]],
    }
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    // Indexed writes into a preallocated buffer, rather than `extend_from_slice` per pixel: this
    // runs once per displayed live frame at full sensor resolution, and the per-call overhead of
    // `extend_from_slice` (a length check plus a memcpy call for each 3-byte chunk) adds up at that
    // rate on cores with less headroom to hide it.
    let mut rgba = vec![0u8; rgb.len() / 3 * 4];
    let (sources, _) = rgb.as_chunks::<3>();
    let (destinations, _) = rgba.as_chunks_mut::<4>();
    for (src, dst) in sources.iter().zip(destinations.iter_mut()) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
    }
    rgba
}

/// sRGB byte colour → linear float, so the sRGB surface's encode step reproduces the palette.
fn srgb_to_linear(color: u32) -> [f32; 3] {
    let channel = |shift: u32| {
        let value = ((color >> shift) & 0xff) as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    [channel(16), channel(8), channel(0)]
}

fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Depth state for overlays (axis labels): always passes, never writes — drawn on top.
fn depth_state_overlay() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

// ---- Vertex data & layouts -------------------------------------------------------------

/// Two triangles covering `[-1, 1]²`; billboarded per instance in the shader.
const QUAD: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

const QUAD_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: 8,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
};
const INSTANCE_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<Instance>() as u64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &wgpu::vertex_attr_array![1 => Float32x4, 2 => Float32x4],
};
const AXIS_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<AxisVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
};
const LABEL_LAYOUT: wgpu::VertexBufferLayout = wgpu::VertexBufferLayout {
    array_stride: std::mem::size_of::<LabelVertex>() as u64,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x3],
};

/// Axis-label glyph height as a fraction of the viewport (NDC units).
const LABEL_SIZE: f32 = 0.06;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    position: [f32; 4], // xyz + per-splat size multiplier
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AxisVertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LabelVertex {
    anchor: [f32; 3], // 3-D anchor at the axis tip
    offset: [f32; 2], // screen-space glyph offset (billboarded)
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    point_size: [f32; 2],
    aspect: f32, // width / height — corrects billboard x offsets
    _pad: f32,
}

// ---- Camera (column-major 4×4, matching WGSL's mat4x4 column layout) --------------------

fn view_proj(pitch: f32, yaw: f32, distance: f32, aspect: f32) -> [[f32; 4]; 4] {
    let projection = perspective(45.0_f32.to_radians(), aspect, 0.1, 100.0);
    let view = mul(
        translate(0.0, 0.0, -distance),
        mul(rotation_x(pitch), rotation_y(yaw)),
    );
    mul(projection, view)
}

fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0_f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

fn translate(x: f32, y: f32, z: f32) -> [[f32; 4]; 4] {
    let mut m = identity();
    m[3] = [x, y, z, 1.0];
    m
}

fn rotation_x(angle: f32) -> [[f32; 4]; 4] {
    let (sin, cos) = angle.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cos, sin, 0.0],
        [0.0, -sin, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn rotation_y(angle: f32) -> [[f32; 4]; 4] {
    let (sin, cos) = angle.sin_cos();
    [
        [cos, 0.0, -sin, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sin, 0.0, cos, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fovy / 2.0).tan();
    let mut m = [[0.0_f32; 4]; 4];
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][2] = far / (near - far);
    m[2][3] = -1.0;
    m[3][2] = near * far / (near - far);
    m
}

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

// ---- Shaders ---------------------------------------------------------------------------

const CLOUD_SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    point_size: vec2<f32>,
    aspect: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct PointOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_points(
    @location(0) corner: vec2<f32>,
    @location(1) center: vec4<f32>,
    @location(2) color: vec4<f32>,
) -> PointOut {
    var out: PointOut;
    var clip = u.view_proj * vec4<f32>(center.xyz, 1.0);
    // Offset by clip.w so the splat keeps a constant screen size after perspective divide.
    clip = vec4<f32>(clip.xy + corner * u.point_size * center.w * clip.w, clip.z, clip.w);
    out.position = clip;
    out.color = color.rgb;
    out.uv = corner;
    return out;
}

@fragment
fn fs_points(in: PointOut) -> @location(0) vec4<f32> {
    if (dot(in.uv, in.uv) > 1.0) { discard; }
    return vec4<f32>(in.color, 1.0);
}

struct AxisOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_axes(@location(0) position: vec3<f32>, @location(1) color: vec3<f32>) -> AxisOut {
    var out: AxisOut;
    out.position = u.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

// Billboarded axis label: project the 3-D anchor, then add the screen-space glyph offset
// (x corrected for aspect) so the letter stays upright and constant-size as the cloud orbits.
@vertex
fn vs_labels(
    @location(0) anchor: vec3<f32>,
    @location(1) offset: vec2<f32>,
    @location(2) color: vec3<f32>,
) -> AxisOut {
    var out: AxisOut;
    var clip = u.view_proj * vec4<f32>(anchor, 1.0);
    let screen = vec2<f32>(offset.x / u.aspect, offset.y);
    clip = vec4<f32>(clip.xy + screen * clip.w, clip.z, clip.w);
    out.position = clip;
    out.color = color;
    return out;
}

@fragment
fn fs_axes(in: AxisOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

const IMAGE_SHADER: &str = r#"
struct ImageOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_image(@builtin(vertex_index) index: u32) -> ImageOut {
    var out: ImageOut;
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    out.uv = vec2<f32>(x, y);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var frame_texture: texture_2d<f32>;
@group(0) @binding(1) var frame_sampler: sampler;

@fragment
fn fs_image(in: ImageOut) -> @location(0) vec4<f32> {
    return textureSample(frame_texture, frame_sampler, in.uv);
}
"#;

#[cfg(test)]
mod tests {
    use super::{cloud_instances, mul, view_proj};
    use crate::viewer::CloudPoint;

    #[test]
    fn strength_scales_instance_size_and_brightness() {
        let points = [
            CloudPoint {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                strength: 1.0,
                color: 0xffffff,
            },
            CloudPoint {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                strength: 0.0,
                color: 0xffffff,
            },
        ];
        let instances = cloud_instances(&points);
        // The strongest point is larger and brighter than the weakest.
        assert!(instances[0].position[3] > instances[1].position[3]);
        assert!(instances[0].color[0] > instances[1].color[0]);
    }

    #[test]
    fn identity_times_matrix_is_unchanged() {
        let m = view_proj(0.0, 0.0, super::CAMERA_DISTANCE, 1.0);
        let identity = super::identity();
        assert_eq!(mul(identity, m), m);
    }

    #[test]
    fn zooming_in_brings_the_volume_closer() {
        // A smaller camera distance magnifies: a fixed world point projects further from the
        // screen centre. Project (0.5, 0, 0) and compare its NDC x (clip.x / clip.w).
        let ndc_x = |distance: f32| {
            let m = view_proj(0.0, 0.0, distance, 1.0);
            let point = [0.5_f32, 0.0, 0.0, 1.0];
            let apply = |row: usize| (0..4).map(|col| m[col][row] * point[col]).sum::<f32>();
            apply(0) / apply(3) // clip.x / clip.w
        };
        assert!(ndc_x(super::MIN_DISTANCE).abs() > ndc_x(super::MAX_DISTANCE).abs());
    }

    #[test]
    fn every_axis_gets_a_billboarded_label() {
        // X, Y, T each contribute at least one glyph segment (2 vertices), all screen-offset.
        let labels = super::label_vertices();
        assert!(labels.len() >= 3 * 2);
        assert!(labels.iter().any(|v| v.offset != [0.0, 0.0]));
    }
}
