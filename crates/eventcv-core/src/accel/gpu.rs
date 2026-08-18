//! The wgpu device, and the one dispatch path every representation kernel goes through.
//!
//! Compiled only with the `gpu` feature. Everything here is about *plumbing* — opening an adapter,
//! packing events, running one shader entry point, reading the cells back. The kernels themselves
//! are in `kernels.wgsl`, and what each does with the cells it gets back lives beside the CPU
//! implementation it mirrors.

use std::sync::Mutex;

use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::EventStream;

const SHADER: &str = include_str!("kernels.wgsl");

/// Matches `@workgroup_size` in the shader.
const WORKGROUP: u32 = 256;

/// Fixed-point scale for the accumulating kernels (voxel, averaged time surface): Q16.16.
///
/// Integer accumulation is what makes those kernels order-independent, and 16 fractional bits put
/// the per-event quantisation at 2^-17 after rounding — below f32's own resolution for the values
/// involved — while leaving ±32768 of range on a cell. A cell that would exceed that sets the
/// shader's saturation flag and the call becomes an error rather than a wrapped number.
pub(crate) const FIXED_ONE: f32 = 65536.0;

/// An opened GPU. One per process: adapter enumeration costs milliseconds and the queue is shared
/// safely, so every call reuses this.
pub(crate) struct Context {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    module: wgpu::ShaderModule,
    /// Declared rather than derived from the shader. Deriving gives each entry point a layout with
    /// only the buffers *it* names — `count` touches neither ages nor the saturation flag — so one
    /// bind group could not serve them all.
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// Compiled pipelines by entry point. Building one costs a shader compilation, which is worth
    /// paying once rather than on every window of a stream.
    pipelines: std::sync::Mutex<std::collections::HashMap<&'static str, wgpu::ComputePipeline>>,
    /// The simulator's pipeline and layout, compiled on first use.
    ///
    /// It lives on this context rather than owning a device of its own for a reason found the hard
    /// way: two open wgpu devices on one adapter fault intermittently while the process tears down,
    /// and a process that has both simulated *and* built representations would hit it. One device is
    /// also simply what a single library should hold.
    pub(crate) sim: std::sync::OnceLock<super::sim::SimPipeline>,
}

/// One storage-buffer binding in the shared layout.
fn layout_entry(binding: u32, uniform: bool, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The process-wide GPU: `None` before the first probe, `Some(None)` once a probe found nothing.
///
/// A `Mutex` rather than a `OnceLock` for one reason: a `OnceLock` static is never dropped, and a
/// wgpu device that outlives the process's own teardown is what makes some drivers fault on the way
/// out — an intermittent segfault *after* the last line of a script ran, which is about the worst
/// failure mode a library can have. Holding it here means [`shutdown`] can close it deliberately,
/// which the bindings do from `atexit`.
static CONTEXT: Mutex<Option<Option<Context>>> = Mutex::new(None);

/// Runs `body` with the process-wide GPU, or returns `None` when there is no adapter.
///
/// The lock is held for the whole call. Submissions to one device serialise in the driver anyway,
/// so this costs nothing real, and it is what lets the context be closed at exit without racing a
/// dispatch that is still running.
pub(crate) fn with_context<T>(body: impl FnOnce(&Context) -> T) -> Option<T> {
    let mut guard = CONTEXT.lock().unwrap_or_else(|error| error.into_inner());
    body(guard.get_or_insert_with(open).as_ref()?).into()
}

/// Closes the GPU, if one was opened. Idempotent; a later call reopens it.
pub(crate) fn shutdown() {
    let mut guard = CONTEXT.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(Some(context)) = guard.take() {
        // Everything queued must land before the device goes away, or the driver is left tearing
        // down work that is still in flight.
        context.device.poll(wgpu::Maintain::Wait);
        drop(context);
    }
}

fn open() -> Option<Context> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("eventcv"),
            required_features: wgpu::Features::empty(),
            // The kernels use only storage buffers and integer atomics, so the downlevel defaults
            // are enough — which keeps software rasterisers and older drivers usable.
            // The downlevel defaults allow four storage buffers per compute stage; the simulator
            // kernel needs five, and asking for six rather than the adapter's maximum keeps the
            // widest set of devices able to open this.
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 6,
                ..wgpu::Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eventcv representations"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("eventcv kernels"),
        entries: &[
            layout_entry(0, true, true),   // params
            layout_entry(1, false, true),  // coords
            layout_entry(2, false, true),  // ages
            layout_entry(3, false, false), // cells
            layout_entry(4, false, false), // saturation flag
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("eventcv kernels"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    Some(Context {
        device,
        queue,
        module,
        layout,
        pipeline_layout,
        pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
        sim: std::sync::OnceLock::new(),
    })
}

impl Context {
    /// Runs `body` with the pipeline for `entry`, compiling it the first time it is asked for.
    fn with_pipeline<T>(&self, entry: &'static str, body: impl FnOnce(&wgpu::ComputePipeline) -> T) -> T {
        let mut pipelines = self
            .pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let pipeline = pipelines.entry(entry).or_insert_with(|| {
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&self.pipeline_layout),
                    module: &self.module,
                    entry_point: entry,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        });
        body(pipeline)
    }
}

/// The uniform block, laid out to match `Params` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    width: u32,
    height: u32,
    n_events: u32,
    bins: u32,
    scale_ms: f32,
    span_ms: f32,
    fixed_one: f32,
    max_age_ticks: u32,
}

// `Params` is plain scalars with no padding of its own, so its bytes are its representation.
unsafe impl bytemuck::Zeroable for Params {}
unsafe impl bytemuck::Pod for Params {}

/// What a kernel needs beyond the events themselves.
pub(crate) struct Dispatch {
    /// Entry point in `kernels.wgsl`.
    pub(crate) entry: &'static str,
    /// Cells to allocate — usually `channels * width * height`.
    pub(crate) cells: usize,
    /// Value every cell starts at. `0` for the accumulating kernels, `i32::MAX` for the
    /// minimising one.
    pub(crate) initial: i32,
    pub(crate) bins: u32,
    pub(crate) span_ms: f32,
    pub(crate) fixed_one: f32,
    /// Window in milliseconds for a kernel that drops events outside one (`None` keeps every
    /// event). Converted to a tick count here, so the kernel's cut-off is an integer comparison —
    /// see `max_age_ticks` in the shader.
    pub(crate) window_ms: Option<f64>,
    /// Whether the kernel reads the age of each event.
    ///
    /// The counting kernels do not, and the upload is where a GPU call spends most of its time on a
    /// multi-million-event slice — so skipping the ages halves the bytes for exactly the kernels
    /// whose arithmetic is too cheap to pay for the trip otherwise.
    pub(crate) needs_ages: bool,
}

/// Why a GPU pass could not produce a result.
pub(crate) enum GpuError {
    /// A fixed-point accumulator would have wrapped. Reported rather than returned wrapped.
    Saturated,
    /// The driver refused something — an allocation, a mapping. Carries its own words.
    Driver(String),
}

/// Runs one kernel over `stream` and returns the raw cells.
///
/// The whole stream is packed into two `u32` buffers — coordinates with the polarity in the top
/// bit, and the age of each event in timestamp ticks measured back from the newest — because every
/// kernel here wants exactly those. Doing it once, here, is also what keeps each kernel's own code
/// down to its arithmetic.
pub(crate) fn run(
    context: &Context,
    stream: &EventStream,
    dispatch: &Dispatch,
) -> Result<Vec<i32>, GpuError> {
    let (width, height) = stream.sensor_size();
    let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
    let newest = ts.iter().copied().max().unwrap_or(0);

    // Packing is the fixed cost of every GPU call and, on a multi-million-event slice, most of it —
    // the kernels themselves finish in a fraction of the time it takes to get the events to them.
    // It is also embarrassingly parallel, so it runs on rayon: without this the GPU path spends its
    // advantage before the first workgroup starts.
    let mut coords: Vec<u32> = (0..stream.len())
        .into_par_iter()
        .map(|index| {
            u32::from(xs[index])
                | ((u32::from(ys[index]) & 0x7fff) << 16)
                | ((ps[index] as u32) << 31)
        })
        .collect();
    // wgpu refuses a zero-sized binding, and an empty window is ordinary during a live capture —
    // one padding entry keeps the binding legal, and the kernels index by invocation against
    // `n_events`, so nothing reads it.
    if coords.is_empty() {
        coords.push(0);
    }
    // Saturating, so a timestamp beyond a `u32` of ticks reads as "very old" — which is what it is
    // — instead of wrapping into the current window. A single zero stands in for a kernel that
    // never reads them, because the binding still has to exist.
    let ages: Vec<u32> = if dispatch.needs_ages {
        ts.par_iter()
            .map(|t| newest.saturating_sub(*t).try_into().unwrap_or(u32::MAX))
            .collect()
    } else {
        Vec::new()
    };
    let ages = if ages.is_empty() { vec![0] } else { ages };

    let params = Params {
        width: width as u32,
        height: height as u32,
        n_events: stream.len() as u32,
        bins: dispatch.bins,
        scale_ms: stream.timestamp_scale_ms() as f32,
        span_ms: dispatch.span_ms,
        fixed_one: dispatch.fixed_one,
        max_age_ticks: dispatch
            .window_ms
            .map_or(u32::MAX, |window| max_age_ticks(window, stream.timestamp_scale_ms())),
    };

    let device = &context.device;
    let storage = wgpu::BufferUsages::STORAGE;
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let coords = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coords"),
        contents: bytemuck::cast_slice(&coords),
        usage: storage,
    });
    let ages = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ages"),
        contents: bytemuck::cast_slice(&ages),
        usage: storage,
    });
    let cells = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cells"),
        contents: bytemuck::cast_slice(&vec![dispatch.initial; dispatch.cells]),
        usage: storage | wgpu::BufferUsages::COPY_SRC,
    });
    let saturated = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("saturated"),
        contents: bytemuck::bytes_of(&0u32),
        usage: storage | wgpu::BufferUsages::COPY_SRC,
    });

    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &context.layout,
        entries: &[
            binding(0, &uniform),
            binding(1, &coords),
            binding(2, &ages),
            binding(3, &cells),
            binding(4, &saturated),
        ],
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (dispatch.cells * 4 + 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    context.with_pipeline(dispatch.entry, |pipeline| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((stream.len() as u32).div_ceil(WORKGROUP).max(1), 1, 1);
    });
    encoder.copy_buffer_to_buffer(&cells, 0, &readback, 0, (dispatch.cells * 4) as u64);
    encoder.copy_buffer_to_buffer(&saturated, 0, &readback, (dispatch.cells * 4) as u64, 4);
    context.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|error| GpuError::Driver(error.to_string()))?
        .map_err(|error| GpuError::Driver(error.to_string()))?;

    let mapped = slice.get_mapped_range();
    let values: Vec<i32> = bytemuck::cast_slice::<u8, i32>(&mapped[..dispatch.cells * 4]).to_vec();
    let saturated = mapped[dispatch.cells * 4..].iter().any(|byte| *byte != 0);
    drop(mapped);
    readback.unmap();

    if saturated {
        return Err(GpuError::Saturated);
    }
    Ok(values)
}

/// The largest age in ticks whose milliseconds still satisfy the CPU's `age <= window` test.
///
/// Deliberately evaluates `ticks as f64 * scale_ms` — the very expression `representation::age_ms`
/// uses — and then walks the candidate by one in each direction, rather than trusting a division to
/// land on the same side of the boundary. That is what makes the two backends drop exactly the same
/// events instead of differing by one wherever an age falls on the edge.
fn max_age_ticks(window_ms: f64, scale_ms: f64) -> u32 {
    if scale_ms <= 0.0 || !scale_ms.is_finite() {
        return u32::MAX;
    }
    let inside = |ticks: i64| ticks >= 0 && (ticks as f64 * scale_ms) <= window_ms;
    let mut ticks = (window_ms / scale_ms).floor() as i64;
    while ticks > 0 && !inside(ticks) {
        ticks -= 1;
    }
    while inside(ticks + 1) {
        ticks += 1;
    }
    ticks.clamp(0, i64::from(u32::MAX)) as u32
}

fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}
