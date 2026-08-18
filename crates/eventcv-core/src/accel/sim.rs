//! Driving `simulate.wgsl` — the GPU backend for [`crate::simulate::Simulator`].
//!
//! The pixel model itself is in the shader; this owns the buffers, the frame-pair upload, and the
//! readback. It is separate from [`super::gpu`] because the two have almost nothing in common: the
//! representation kernels scatter events into a fixed grid, while this walks a per-pixel state
//! machine forwards in time and *produces* events whose count is not known until it has.

use wgpu::util::DeviceExt;

use crate::simulate::{PixelState, SimEvent, SimulatorConfig};

const SHADER: &str = include_str!("simulate.wgsl");

/// Matches `@workgroup_size` in the shader.
const WORKGROUP: u32 = 64;

/// Events per pixel the output buffer is first sized for.
///
/// A frame pair that produces more is not an error: the kernel reports the count it *wanted* and
/// the host grows the buffer and runs it again. Two is enough for a typical interval (the adaptive
/// upsampler exists precisely to keep it near one), so the retry is rare.
const EVENTS_PER_PIXEL: usize = 2;

/// Everything the shader needs that is not per pixel. Mirrors `SimParams` in `simulate.wgsl`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SimParams {
    width: u32,
    height: u32,
    steps: u32,
    step_offset: u32,
    total_steps: u32,
    capacity: u32,
    seed_lo: u32,
    seed_hi: u32,
    pos_thres: f32,
    neg_thres: f32,
    refractory_us: i32,
    cutoff_hz: f32,
    leak_rate_hz: f32,
    shot_noise_rate_hz: f32,
    frame: u32,
    _pad: u32,
}

unsafe impl bytemuck::Zeroable for SimParams {}
unsafe impl bytemuck::Pod for SimParams {}

/// The GPU's copy of one pixel's state. Mirrors `PixelState` in the shader, and is kept in step
/// with [`crate::simulate::PixelState`] by [`from_cpu`](GpuPixelState::from_cpu).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GpuPixelState {
    log_ref: f32,
    lowpass: f32,
    thres_pos: f32,
    thres_neg: f32,
    last_t: i32,
}

unsafe impl bytemuck::Zeroable for GpuPixelState {}
unsafe impl bytemuck::Pod for GpuPixelState {}

/// The GPU's copy of one produced event. Mirrors `SimEvent` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GpuEvent {
    t: i32,
    x: u32,
    y: u32,
    positive: u32,
}

unsafe impl bytemuck::Zeroable for GpuEvent {}
unsafe impl bytemuck::Pod for GpuEvent {}

impl GpuPixelState {
    /// The CPU's state as the shader sees it.
    ///
    /// `last_t` is the one lossy field: the CPU seeds it at `i64::MIN / 4` to mean "never fired",
    /// which does not fit an `i32`. `i32::MIN / 4` means the same thing to the refractory check —
    /// any real timestamp is further away than any refractory period — and stays clear of the
    /// overflow that `i32::MIN` itself would risk in `t - last_t`.
    pub(crate) fn from_cpu(state: &PixelState) -> Self {
        Self {
            log_ref: state.log_ref,
            lowpass: state.lowpass,
            thres_pos: state.thres_pos,
            thres_neg: state.thres_neg,
            last_t: state.last_t.clamp(i64::from(i32::MIN / 4), i64::from(i32::MAX)) as i32,
        }
    }

    /// Writes the kernel's state back over the CPU's, so the two can be interleaved — which is what
    /// makes a run that falls back mid-recording still coherent.
    pub(crate) fn to_cpu(self, state: &mut PixelState) {
        state.log_ref = self.log_ref;
        state.lowpass = self.lowpass;
        state.thres_pos = self.thres_pos;
        state.thres_neg = self.thres_neg;
        state.last_t = i64::from(self.last_t);
    }
}

/// The simulator's compiled pipeline, built on the shared device the representation kernels use.
pub(crate) struct SimPipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// High-water mark of events produced by one dispatch, so the next interval sizes its buffer
    /// from what the last one actually needed rather than from a worst case nobody hits.
    capacity: std::sync::atomic::AtomicUsize,
}

/// Compiles the kernel against `context`'s device.
///
/// It borrows that device rather than opening one of its own for a reason found the hard way: two
/// wgpu devices open on one adapter fault intermittently while the process tears down, and any
/// program that both simulated *and* built representations would have hit it.
fn build(context: &super::gpu::Context) -> SimPipeline {
    let device = &context.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("eventcv simulator"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..6)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: match binding {
                    0 => wgpu::BufferBindingType::Uniform,
                    // The pixel state, the events and the cursor are written; the rest is input.
                    2..=4 => wgpu::BufferBindingType::Storage { read_only: false },
                    _ => wgpu::BufferBindingType::Storage { read_only: true },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("eventcv simulator"),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("eventcv simulator"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("simulate"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "simulate",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    SimPipeline {
        pipeline,
        layout,
        capacity: std::sync::atomic::AtomicUsize::new(0),
    }
}

/// One frame pair on the GPU: advances `state` and returns the interval's events, unsorted.
///
/// `bounds` are the `steps + 1` sub-interval boundaries in microseconds, computed by the caller in
/// `f64` exactly as the CPU path computes them — the shader has no `f64` to compute them with, and
/// timestamps that merely land nearby would be a difference in the output rather than in the
/// arithmetic.
///
/// Returns `None` when there is no GPU, so the caller can say so rather than guess.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_interval(
    width: usize,
    height: usize,
    config: &SimulatorConfig,
    frame: u64,
    bounds: &[i64],
    prev_log: &[f32],
    log_now: &[f32],
    prev_luma: &[f32],
    luma_now: &[f32],
    state: &mut [PixelState],
) -> Option<Vec<SimEvent>> {
    let total_steps = bounds.len().saturating_sub(1);
    if total_steps == 0 {
        return Some(Vec::new());
    }
    let mut events = Vec::new();
    run_range(
        width, height, config, frame, 0, total_steps, total_steps, bounds, prev_log, log_now,
        prev_luma, luma_now, state, &mut events,
    )
    .then_some(events)
}

/// Runs `count` sub-steps starting at `offset`, halving the range if the events they produce would
/// not fit one buffer binding.
///
/// Splitting is exact: the pixel state carries from one dispatch to the next exactly as it carries
/// from one sub-step to the next, and the interpolation and the noise key both use a sub-step's
/// place in the *whole* interval. So this is purely a memory decision. It is done reactively — try
/// the whole interval, split only when the kernel says it did not fit — because a worst case
/// computed up front (every pixel firing every sub-step) is dozens of times what a real frame pair
/// produces, and budgeting for it turns one dispatch into sixteen.
#[allow(clippy::too_many_arguments)]
fn run_range(
    width: usize,
    height: usize,
    config: &SimulatorConfig,
    frame: u64,
    offset: usize,
    count: usize,
    total_steps: usize,
    bounds: &[i64],
    prev_log: &[f32],
    log_now: &[f32],
    prev_luma: &[f32],
    luma_now: &[f32],
    state: &mut [PixelState],
    events: &mut Vec<SimEvent>,
) -> bool {
    match run_steps(
        width,
        height,
        config,
        frame,
        offset,
        total_steps,
        &bounds[offset..=offset + count],
        prev_log,
        log_now,
        prev_luma,
        luma_now,
        state,
    ) {
        Dispatch::Events(produced) => {
            events.extend(produced);
            true
        }
        Dispatch::NoDevice => false,
        // One sub-step that still does not fit is beyond what this device can do at this sensor
        // size; the caller falls back to the CPU, which has no such ceiling.
        Dispatch::TooLarge if count <= 1 => false,
        Dispatch::TooLarge => {
            let half = count / 2;
            run_range(
                width, height, config, frame, offset, half, total_steps, bounds, prev_log, log_now,
                prev_luma, luma_now, state, events,
            ) && run_range(
                width,
                height,
                config,
                frame,
                offset + half,
                count - half,
                total_steps,
                bounds,
                prev_log,
                log_now,
                prev_luma,
                luma_now,
                state,
                events,
            )
        }
    }
}

/// What one dispatch produced, or why it did not.
enum Dispatch {
    Events(Vec<SimEvent>),
    /// More events than one buffer binding can hold — the caller splits the sub-step range.
    TooLarge,
    NoDevice,
}

/// One dispatch: sub-steps `offset + 1 ..= offset + bounds.len() - 1` of an interval that has
/// `total_steps` of them.
#[allow(clippy::too_many_arguments)]
fn run_steps(
    width: usize,
    height: usize,
    config: &SimulatorConfig,
    frame: u64,
    step_offset: usize,
    total_steps: usize,
    bounds: &[i64],
    prev_log: &[f32],
    log_now: &[f32],
    prev_luma: &[f32],
    luma_now: &[f32],
    state: &mut [PixelState],
) -> Dispatch {
    super::gpu::with_context(|context| {
        let sim = context.sim.get_or_init(|| build(context));
        run_dispatch(
            context, sim, width, height, config, frame, step_offset, total_steps, bounds, prev_log,
            log_now, prev_luma, luma_now, state,
        )
    })
    .unwrap_or(Dispatch::NoDevice)
}

/// One dispatch against the already-opened device and compiled pipeline.
#[allow(clippy::too_many_arguments)]
fn run_dispatch(
    context: &super::gpu::Context,
    sim: &SimPipeline,
    width: usize,
    height: usize,
    config: &SimulatorConfig,
    frame: u64,
    step_offset: usize,
    total_steps: usize,
    bounds: &[i64],
    prev_log: &[f32],
    log_now: &[f32],
    prev_luma: &[f32],
    luma_now: &[f32],
    state: &mut [PixelState],
) -> Dispatch {
    let device = &context.device;
    let pixels = width * height;
    let steps = bounds.len().saturating_sub(1);

    let gpu_state: Vec<GpuPixelState> = state.iter().map(GpuPixelState::from_cpu).collect();
    let bounds: Vec<i32> = bounds
        .iter()
        .map(|t| (*t).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32)
        .collect();

    let storage = wgpu::BufferUsages::STORAGE;
    // The four planes go up as one buffer; see `planes` in the shader.
    let mut plane_data = Vec::with_capacity(pixels * 4);
    for plane in [prev_log, log_now, prev_luma, luma_now] {
        plane_data.extend_from_slice(plane);
    }
    let planes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("planes"),
        contents: bytemuck::cast_slice(&plane_data),
        usage: storage,
    });
    let bounds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bounds"),
        contents: bytemuck::cast_slice(&bounds),
        usage: storage,
    });

    // The kernel reports how many events it *wanted* to write, so an undersized buffer costs a
    // second pass rather than a wrong answer. The counter-based RNG makes that second pass produce
    // exactly the events the first one would have.
    let ceiling =
        device.limits().max_storage_buffer_binding_size as usize / std::mem::size_of::<GpuEvent>();
    // Start from a little more than the busiest interval so far — a guess that is too small costs
    // one extra dispatch, and one that is too large costs the allocation every frame.
    let previous = sim.capacity.load(std::sync::atomic::Ordering::Relaxed);
    let mut capacity = match previous {
        0 => (pixels * EVENTS_PER_PIXEL).min(ceiling),
        previous => (previous + previous / 4).min(ceiling),
    }
    .max(1);
    loop {
        let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("state"),
            contents: bytemuck::cast_slice(&gpu_state),
            usage: storage | wgpu::BufferUsages::COPY_SRC,
        });
        let events = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("events"),
            size: (capacity * std::mem::size_of::<GpuEvent>()) as u64,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let cursor = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cursor"),
            contents: bytemuck::bytes_of(&0u32),
            usage: storage | wgpu::BufferUsages::COPY_SRC,
        });
        let params = SimParams {
            width: width as u32,
            height: height as u32,
            steps: steps as u32,
            step_offset: step_offset as u32,
            total_steps: total_steps as u32,
            capacity: capacity as u32,
            seed_lo: config.seed as u32,
            seed_hi: (config.seed >> 32) as u32,
            pos_thres: config.pos_thres,
            neg_thres: config.neg_thres,
            refractory_us: config.refractory_us.clamp(0, i64::from(i32::MAX)) as i32,
            cutoff_hz: config.cutoff_hz,
            leak_rate_hz: config.leak_rate_hz,
            shot_noise_rate_hz: config.shot_noise_rate_hz,
            frame: frame as u32,
            _pad: 0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &sim.layout,
            entries: &[
                entry(0, &uniform),
                entry(1, &planes),
                entry(2, &state_buffer),
                entry(3, &events),
                entry(4, &cursor),
                entry(5, &bounds),
            ],
        });

        let state_bytes = (pixels * std::mem::size_of::<GpuPixelState>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&sim.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((pixels as u32).div_ceil(WORKGROUP).max(1), 1, 1);
        }
        // How many events there are comes back first, on its own, and only then are that many
        // copied. Copying the whole buffer instead would move the *capacity* every interval — tens
        // of times the events actually produced on a quiet frame, and enough to make the GPU path
        // slower than the CPU it is meant to beat.
        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cursor readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&cursor, 0, &counter, 0, 4);
        context.queue.submit(Some(encoder.finish()));
        let Some(wanted) = read_u32(device, &counter) else {
            return Dispatch::NoDevice;
        };
        let wanted = wanted as usize;

        if wanted > capacity {
            if capacity >= ceiling {
                return Dispatch::TooLarge;
            }
            capacity = wanted.min(ceiling);
            continue;
        }

        let event_bytes = (wanted * std::mem::size_of::<GpuEvent>()) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim readback"),
            size: (state_bytes + event_bytes).max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&state_buffer, 0, &readback, 0, state_bytes);
        if event_bytes > 0 {
            encoder.copy_buffer_to_buffer(&events, 0, &readback, state_bytes, event_bytes);
        }
        context.queue.submit(Some(encoder.finish()));

        let Some(mapped) = map_read(device, &readback) else {
            return Dispatch::NoDevice;
        };
        let states: &[GpuPixelState] = bytemuck::cast_slice(&mapped[..state_bytes as usize]);
        for (gpu, cpu) in states.iter().zip(state.iter_mut()) {
            gpu.to_cpu(cpu);
        }
        let produced: &[GpuEvent] =
            bytemuck::cast_slice(&mapped[state_bytes as usize..(state_bytes + event_bytes) as usize]);
        let events: Vec<SimEvent> = produced
            .iter()
            .map(|event| SimEvent {
                t: i64::from(event.t),
                x: event.x as u16,
                y: event.y as u16,
                positive: event.positive != 0,
            })
            .collect();
        drop(mapped);
        readback.unmap();
        // Next interval starts from what this one needed. Video is temporally coherent, so after
        // the first frame pair the buffer is the right size and the retry above never fires.
        sim.capacity.store(capacity.max(wanted), std::sync::atomic::Ordering::Relaxed);
        return Dispatch::Events(events);
    }
}

/// Blocks until `buffer` is mapped, returning its bytes.
fn map_read<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
) -> Option<wgpu::BufferView<'a>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv().ok()?.ok()?;
    Some(slice.get_mapped_range())
}

/// Reads a single `u32` back, unmapping before returning so the buffer can be dropped.
fn read_u32(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Option<u32> {
    let mapped = map_read(device, buffer)?;
    let value = u32::from_le_bytes(mapped[..4].try_into().ok()?);
    drop(mapped);
    buffer.unmap();
    Some(value)
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
