//! Frames → events: a DVS pixel simulator.
//!
//! Turns a sequence of intensity frames into an [`EventStream`], modelling the pixel closely enough
//! that the result is usable as training data rather than only as a demo. The model follows Hu et
//! al., *v2e: From Video Frames to Realistic DVS Events* (CVPRW 2021): log-intensity differencing
//! with per-pixel threshold mismatch, a finite photoreceptor bandwidth, shot noise, leak events and
//! a refractory period.
//!
//! # What is modelled, and why each part matters
//!
//! - **sRGB linearisation, then a lin-log map.** Video is gamma-encoded; taking `ln` of the stored
//!   8-bit value measures contrast in the wrong space. Below ~20/255 the map is linear rather than
//!   logarithmic, because `ln` of a near-zero value amplifies quantisation noise into events that a
//!   real sensor would never emit.
//! - **Per-pixel threshold mismatch.** Real thresholds vary pixel to pixel by a few percent. Fixed
//!   thresholds make every pixel on an edge fire in lockstep, which is the most visible way
//!   synthetic events look synthetic.
//! - **Photoreceptor bandwidth.** A first-order lowpass whose cutoff falls with intensity, so dark
//!   scenes lag — the dominant artefact in low-light DVS recordings.
//! - **Shot noise and leak events.** Background activity that does not come from motion at all. A
//!   model trained on noiseless synthetic data has never seen it.
//! - **Interpolated timestamps.** When a pixel crosses its threshold several times between two
//!   frames, the crossings are spread across the interval by when they actually occurred, rather
//!   than all being stamped with the frame time. Timing *is* the signal in event data; collapsing it
//!   to the frame rate discards the temporal precision that motivates using an event camera.
//!
//! # What is not modelled
//!
//! Motion blur and exposure time in the source footage (they arrive already baked in), arbiter and
//! bus-bandwidth saturation, hot and dead pixels (see [`EventStream::pixel_dropout`] for those), and
//! threshold dependence on illumination beyond the bandwidth term.

use std::sync::OnceLock;

use rand::{
    rngs::{SmallRng, StdRng},
    Rng, SeedableRng,
};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::viz::Rgb8Image;
use crate::{EventStream, EventStreamBuilder};

/// Intensity below which the log map is replaced by a linear one, in 8-bit levels.
///
/// v2e's value. Below this the sensor is photon-starved and `ln` would turn one-level quantisation
/// steps into large contrast changes.
const LIN_LOG_THRESHOLD: f32 = 20.0;

/// Floor on the photoreceptor bandwidth as a fraction of its maximum, so a black pixel still
/// tracks rather than freezing entirely.
const MIN_BANDWIDTH_FRACTION: f32 = 0.1;

/// How much quieter shot noise is in the brightest pixels than the darkest (v2e's `c`).
const SHOT_NOISE_BRIGHT_FACTOR: f32 = 0.25;

/// Default ceiling on adaptive upsampling, so a hard cut between two frames cannot ask for
/// thousands of sub-steps and stall the run. Overridable per run via
/// [`SimulatorConfig::max_upsample`] — the ceiling is a cost/accuracy trade, not a physical bound.
pub const MAX_UPSAMPLE: usize = 64;

/// Hard ceiling on [`SimulatorConfig::max_upsample`]. A sub-step costs a full pass over every
/// pixel, so an unbounded value turns a typo into an unkillable run.
const MAX_UPSAMPLE_CEILING: usize = 4096;

/// Pixels per parallel work block.
///
/// Fixed rather than derived from the core count, because each block seeds its own noise RNG: a
/// block size that varied with the machine would make the events depend on how many cores ran the
/// simulation. Sized so one block's [`PixelState`] (~192 KiB) stays resident across every sub-step
/// of a frame pair, which is why the sub-step loop lives *inside* the block rather than around it.
const PIXEL_BLOCK: usize = 8192;

/// Below this many pixels the rayon dispatch costs more than the work it splits, so the whole
/// interval runs inline. Mirrors `cmax`'s `PARALLEL_EVENT_THRESHOLD`.
const PARALLEL_PIXEL_THRESHOLD: usize = 1 << 16;

/// Above this many events in one interval the sort is worth handing to rayon.
const PARALLEL_SORT_THRESHOLD: usize = 1 << 16;

/// Entries in the per-sub-interval shot-noise probability table, indexed by quantised luma.
/// One 8-bit level of luma moves the probability by well under a percent of itself, which is far
/// below the noise the table is describing.
const NOISE_LUT_LEN: usize = 256;

/// How finely the interval between two source frames is subdivided before simulating.
///
/// Between two frames the true intensity path is unknown and is assumed linear. That assumption
/// degrades as more happens between them, so subdividing and interpolating recovers timing accuracy
/// — the axis no other event-vision library covers today.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Upsample {
    /// Simulate straight from the source frames.
    Off,
    /// Always insert `n - 1` interpolated frames between each pair.
    Fixed(usize),
    /// Subdivide until no pixel would emit more than `max_events_per_pixel` events per sub-interval.
    ///
    /// v2e picks its factor from optical flow, keeping motion under a pixel per sub-interval. This
    /// uses contrast instead: the quantity that actually bounds timestamp error is how many
    /// threshold crossings are being packed into one linear interpolation, and that is measured
    /// directly rather than inferred from displacement — no flow estimate required.
    Adaptive { max_events_per_pixel: f32 },
}

impl Default for Upsample {
    fn default() -> Self {
        Self::Adaptive {
            max_events_per_pixel: 1.0,
        }
    }
}

/// Pixel-model parameters. Defaults follow v2e's for a typical DVS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulatorConfig {
    /// Log-intensity increase that emits an ON event.
    pub pos_thres: f32,
    /// Log-intensity decrease that emits an OFF event.
    pub neg_thres: f32,
    /// Standard deviation of the per-pixel Gaussian threshold mismatch, in log units.
    pub sigma_thres: f32,
    /// Minimum time between two events from the same pixel, in µs.
    pub refractory_us: i64,
    /// Photoreceptor bandwidth for a white pixel, in Hz. `0` disables the lowpass.
    pub cutoff_hz: f32,
    /// Spontaneous ON events per pixel per second. `0` disables leak.
    pub leak_rate_hz: f32,
    /// Shot-noise events per pixel per second in the darkest pixels. `0` disables noise.
    pub shot_noise_rate_hz: f32,
    /// Seed for mismatch and noise, so a run is reproducible from its configuration.
    pub seed: u64,
    /// Frame subdivision — see [`Upsample`].
    pub upsample: Upsample,
    /// Ceiling on the sub-steps any one frame pair may be split into.
    ///
    /// Adaptive upsampling is driven by the *busiest* pixel, so one high-contrast edge can push a
    /// whole 1080p frame to the ceiling — 64 full-sensor passes for that pair. Lowering this
    /// bounds the worst case at some cost in timestamp accuracy where the scene really is that
    /// fast; raising it buys accuracy on hard cuts. Clamped to `1..=4096`.
    pub max_upsample: usize,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            pos_thres: 0.2,
            neg_thres: 0.2,
            sigma_thres: 0.03,
            refractory_us: 100,
            cutoff_hz: 200.0,
            leak_rate_hz: 1.0,
            shot_noise_rate_hz: 10.0,
            seed: 0,
            upsample: Upsample::default(),
            max_upsample: MAX_UPSAMPLE,
        }
    }
}

impl SimulatorConfig {
    /// A noiseless, ideal sensor: fixed thresholds, no mismatch, bandwidth, leak or shot noise.
    ///
    /// Not realistic, and not the default for that reason — but it is what makes the simulator
    /// *testable*, since an ideal pixel's event count is an analytic function of the contrast.
    pub fn ideal() -> Self {
        Self {
            sigma_thres: 0.0,
            cutoff_hz: 0.0,
            leak_rate_hz: 0.0,
            shot_noise_rate_hz: 0.0,
            refractory_us: 0,
            upsample: Upsample::Off,
            ..Self::default()
        }
    }
}

/// Everything the model remembers about one pixel.
///
/// Held as one array of these rather than five parallel arrays: the sub-step loop touches all of a
/// pixel's state together (so this is one cache line's worth of locality instead of five streams),
/// and a single slice is what [`par_chunks_mut`](rayon::slice::ParallelSliceMut::par_chunks_mut)
/// can hand to a worker without a five-way zip.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PixelState {
    /// Memorised log intensity — the level each event is measured against.
    pub(crate) log_ref: f32,
    /// Photoreceptor lowpass state.
    pub(crate) lowpass: f32,
    /// Thresholds, drawn once at construction: mismatch is a property of the silicon, not noise
    /// that resamples every frame.
    pub(crate) thres_pos: f32,
    pub(crate) thres_neg: f32,
    /// Last emission time, for the refractory check.
    pub(crate) last_t: i64,
}

/// One generated event, before it is sorted and handed to an [`EventStreamBuilder`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct SimEvent {
    pub(crate) t: i64,
    pub(crate) x: u16,
    pub(crate) y: u16,
    pub(crate) positive: bool,
}

/// A DVS pixel array, driven one frame at a time.
///
/// Frames are pushed in time order and each call returns the events generated since the previous
/// one, already sorted. Nothing accumulates across calls, so a recording of any length simulates in
/// memory proportional to one frame interval rather than to the whole output.
///
/// # Parallelism
///
/// A frame pair is split into fixed-size blocks of pixels and simulated with rayon. Pixels are
/// independent — nothing in the model couples one to its neighbours — so the only shared thing was
/// the random number generator, and each block draws from its own stream instead (see
/// [`block_seed`]). Both the block size and the seed derivation are fixed constants, so the output
/// is a function of the configuration alone: the same `seed` gives the same events on one core or
/// thirty-two.
pub struct Simulator {
    config: SimulatorConfig,
    width: usize,
    height: usize,
    /// Per-pixel model state.
    state: Vec<PixelState>,
    /// Working buffer for the incoming frame's lin-log intensity, reused across calls.
    log_now: Vec<f32>,
    /// Luma in `[0, 1]` for the incoming frame, reused across calls.
    luma: Vec<f32>,
    /// The previous frame's pair, swapped with the incoming buffers rather than cloned.
    prev_log: Vec<f32>,
    prev_luma: Vec<f32>,
    /// `None` until the first frame seeds the state.
    t_previous: Option<i64>,
    /// Counts pushed frames, so each interval's noise draws a different stream.
    frame: u64,
    /// One event buffer per pixel block, kept allocated across frames so a steady event rate stops
    /// reallocating. Concatenated in block order, which is what makes the result reproducible.
    blocks: Vec<Vec<SimEvent>>,
    /// The concatenated interval, sorted in place before being handed out.
    events: Vec<SimEvent>,
    /// Where the pixel model runs. The CPU is the reference and the default; see
    /// [`on_device`](Self::on_device).
    device: crate::accel::Device,
}

impl Simulator {
    pub fn new(width: usize, height: usize, config: SimulatorConfig) -> Self {
        let pixels = width * height;
        let mut rng = StdRng::seed_from_u64(config.seed);
        // Mismatch is sampled once and kept, from one serial stream: it is O(pixels) at
        // construction rather than per sub-step, so there is nothing to gain by splitting it, and
        // keeping it serial means the pattern of mismatch is unchanged by the parallel loop.
        // Thresholds are clamped well above zero: a threshold at or below zero would emit
        // unboundedly on any change at all.
        let sample = |rng: &mut StdRng, base: f32| -> Vec<f32> {
            match Normal::new(0.0_f32, config.sigma_thres.max(0.0)) {
                Ok(normal) if config.sigma_thres > 0.0 => (0..pixels)
                    .map(|_| (base + normal.sample(rng)).max(base * 0.1).max(1e-3))
                    .collect(),
                _ => vec![base.max(1e-3); pixels],
            }
        };
        let thres_pos = sample(&mut rng, config.pos_thres);
        let thres_neg = sample(&mut rng, config.neg_thres);
        let state = thres_pos
            .into_iter()
            .zip(thres_neg)
            .map(|(thres_pos, thres_neg)| PixelState {
                log_ref: 0.0,
                lowpass: 0.0,
                thres_pos,
                thres_neg,
                last_t: i64::MIN / 4,
            })
            .collect();
        Self {
            config,
            width,
            height,
            state,
            log_now: vec![0.0; pixels],
            luma: vec![0.0; pixels],
            prev_log: vec![0.0; pixels],
            prev_luma: vec![0.0; pixels],
            t_previous: None,
            frame: 0,
            blocks: vec![Vec::new(); pixels.div_ceil(PIXEL_BLOCK)],
            events: Vec::new(),
            device: crate::accel::Device::Cpu,
        }
    }

    /// Runs the pixel model on `device`.
    ///
    /// # What is and is not the same as the CPU
    ///
    /// The deterministic half is the same by construction. Threshold mismatch is drawn *here*, once
    /// at [`new`](Self::new), and uploaded — the GPU never regenerates it — so two sensors built
    /// from one seed are the same silicon whichever backend runs them. The sub-interval boundaries
    /// are computed on the host in `f64` and uploaded for the same reason. What is left is `exp`,
    /// `floor` and the arithmetic between them, where WGSL and Rust agree to a few ULP.
    ///
    /// The random half is **not** the same sample path. The CPU draws from a generator seeded per
    /// block of pixels and consumed in order; a kernel with one invocation per pixel cannot replay a
    /// sequential stream without serialising itself. The GPU uses a counter-based generator keyed on
    /// `(seed, frame, sub-step, pixel, polarity)`, which is reproducible from run to run and across
    /// devices — every operation in it is integer — but draws different numbers from the CPU's.
    ///
    /// So, measured rather than hoped for (see this module's `gpu_tests`):
    ///
    /// | configuration | how the backends compare |
    /// |---|---|
    /// | no shot noise, no leak, no mismatch | identical, event for event |
    /// | mismatch on, still noiseless | same events, same polarity; a couple of timestamps in ~25 000 land a microsecond apart, because the crossing fraction is `f64` on the CPU and WGSL has no `f64` |
    /// | noise on | same event *rate* to within a few per cent, different events — and bit-reproducible from run to run for a given seed |
    ///
    /// A run that needs to be compared against a stored CPU result should therefore either turn the
    /// random terms off or compare distributions. A run that just needs events does not care.
    pub fn on_device(mut self, device: crate::accel::Device) -> Self {
        self.device = device;
        self
    }

    pub fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Feeds one frame, returning every event generated since the previous frame.
    ///
    /// `t_us` must not go backwards. The first frame only seeds the pixel state and returns an empty
    /// stream — there is no interval to integrate over yet.
    pub fn push_frame(&mut self, frame: &[f32], t_us: i64) -> EventStream {
        assert_eq!(
            frame.len(),
            self.width * self.height,
            "frame does not match the simulator's sensor size"
        );
        for (index, &value) in frame.iter().enumerate() {
            let clamped = value.clamp(0.0, 1.0);
            self.luma[index] = clamped;
            self.log_now[index] = lin_log(clamped * 255.0);
        }

        let Some(t_previous) = self.t_previous else {
            for (pixel, &log) in self.state.iter_mut().zip(&self.log_now) {
                pixel.log_ref = log;
                pixel.lowpass = log;
                pixel.last_t = t_us;
            }
            self.advance(t_us);
            return self.empty_stream();
        };

        let span = (t_us - t_previous).max(0);
        let steps = self.upsample_steps(span);
        self.simulate_interval(t_previous, span, steps);
        self.advance(t_us);
        self.collect_interval()
    }

    /// Simulates one frame pair into `self.blocks`, in parallel over blocks of pixels.
    ///
    /// The sub-step loop runs *inside* each block rather than around all of them: pixels never
    /// interact, so a block can walk the whole interval on its own, which keeps its state hot in
    /// cache and costs one rayon dispatch per frame instead of one per sub-step.
    fn simulate_interval(&mut self, t_previous: i64, span: i64, steps: usize) {
        if self.device == crate::accel::Device::Gpu && self.simulate_interval_on_gpu(t_previous, span, steps) {
            return;
        }
        let width = self.width;
        let pixels = self.state.len();
        // Field-by-field borrows: the state is written, everything else only read.
        let config = &self.config;
        let frame = self.frame;
        let (prev_log, log_now) = (&self.prev_log, &self.log_now);
        let (prev_luma, luma) = (&self.prev_luma, &self.luma);

        let run = |block: usize, state: &mut [PixelState], out: &mut Vec<SimEvent>| {
            out.clear();
            let base = block * PIXEL_BLOCK;
            let mut rng = SmallRng::seed_from_u64(block_seed(config.seed, frame, block));
            for step in 1..=steps {
                // Linear interpolation in log space between the two frames. Sub-intervals are
                // walked in order, so events stay grouped in ascending time even before the sort.
                let alpha = step as f32 / steps as f32;
                let t_start = t_previous + (span as f64 * (step - 1) as f64 / steps as f64) as i64;
                let t_end = t_previous + (span as f64 * step as f64 / steps as f64) as i64;
                let dt_s = ((t_end - t_start) as f64 / 1e6) as f32;
                let noise = NoiseTable::new(config.shot_noise_rate_hz, dt_s);

                // Coordinates are walked rather than divided out per pixel: a block is a
                // contiguous index range, so this is two divisions per block instead of two per
                // pixel per sub-step (hundreds of millions of them on a 1080p clip).
                let (mut x, mut y) = ((base % width) as u16, (base / width) as u16);
                for (offset, pixel) in state.iter_mut().enumerate() {
                    let index = base + offset;
                    let target = prev_log[index] + (log_now[index] - prev_log[index]) * alpha;
                    let pixel_luma = prev_luma[index] + (luma[index] - prev_luma[index]) * alpha;
                    integrate_pixel(
                        pixel, x, y, target, pixel_luma, t_start, t_end, dt_s, config, &noise,
                        &mut rng, out,
                    );
                    x += 1;
                    if usize::from(x) == width {
                        x = 0;
                        y += 1;
                    }
                }
            }
        };

        if pixels < PARALLEL_PIXEL_THRESHOLD {
            for (block, (state, out)) in self
                .state
                .chunks_mut(PIXEL_BLOCK)
                .zip(self.blocks.iter_mut())
                .enumerate()
            {
                run(block, state, out);
            }
        } else {
            self.state
                .par_chunks_mut(PIXEL_BLOCK)
                .zip(self.blocks.par_iter_mut())
                .enumerate()
                .for_each(|(block, (state, out))| run(block, state, out));
        }
    }

    /// The same interval on the GPU, returning `false` when there is no adapter so the caller can
    /// fall through to the CPU loop.
    ///
    /// The kernel writes its events through one atomic cursor, so they arrive in whatever order the
    /// scheduler produced them. That costs nothing: [`collect_interval`](Self::collect_interval)
    /// sorts every interval by `(t, y, x, polarity)` anyway — the CPU path needs the same sort to
    /// stay independent of its core count — so the two backends hand back the same ordering.
    ///
    /// The events land in `blocks[0]` because that is what `collect_interval` concatenates; the
    /// remaining blocks are cleared so a run that switches backends mid-recording cannot replay a
    /// stale interval.
    #[cfg(feature = "gpu")]
    fn simulate_interval_on_gpu(&mut self, t_previous: i64, span: i64, steps: usize) -> bool {
        // Boundaries in `f64`, matching the CPU loop exactly; the shader has no `f64` to do this in.
        let bounds: Vec<i64> = (0..=steps)
            .map(|step| t_previous + (span as f64 * step as f64 / steps as f64) as i64)
            .collect();
        let Some(events) = crate::accel::sim::run_interval(
            self.width,
            self.height,
            &self.config,
            self.frame,
            &bounds,
            &self.prev_log,
            &self.log_now,
            &self.prev_luma,
            &self.luma,
            &mut self.state,
        ) else {
            return false;
        };
        for block in &mut self.blocks {
            block.clear();
        }
        if let Some(first) = self.blocks.first_mut() {
            *first = events;
        }
        true
    }

    #[cfg(not(feature = "gpu"))]
    fn simulate_interval_on_gpu(&mut self, _t_previous: i64, _span: i64, _steps: usize) -> bool {
        false
    }

    /// Concatenates the per-block buffers, sorts, and builds the interval's stream.
    fn collect_interval(&mut self) -> EventStream {
        self.events.clear();
        self.events.reserve(self.blocks.iter().map(Vec::len).sum());
        for block in &self.blocks {
            self.events.extend_from_slice(block);
        }

        // Sorting per interval rather than globally is what keeps this streaming: intervals are
        // produced in time order, so concatenating sorted blocks is already globally sorted, and no
        // sort ever sees more than one interval's events.
        //
        // The key is the whole event, not just `t`. An unstable sort leaves ties in an
        // implementation-defined order, and a *parallel* unstable sort's order also depends on how
        // the work was split — so with `t` alone the output would shift with the core count.
        // Ordering ties by position makes the result total and machine-independent.
        let key = |event: &SimEvent| (event.t, event.y, event.x, event.positive);
        if self.events.len() >= PARALLEL_SORT_THRESHOLD {
            self.events.par_sort_unstable_by_key(key);
        } else {
            self.events.sort_unstable_by_key(key);
        }

        let mut builder =
            EventStreamBuilder::with_capacity(self.width, self.height, 0.001, self.events.len());
        for event in &self.events {
            builder.push(event.x, event.y, event.t, event.positive);
        }
        builder.build()
    }

    /// Rolls the incoming frame into the "previous" slot. Swapping rather than cloning: the old
    /// code copied two full-frame `Vec<f32>`s per frame, which is 16 MB of memcpy per 1080p frame.
    fn advance(&mut self, t_us: i64) {
        std::mem::swap(&mut self.prev_log, &mut self.log_now);
        std::mem::swap(&mut self.prev_luma, &mut self.luma);
        self.t_previous = Some(t_us);
        self.frame += 1;
    }

    /// How many sub-intervals this frame pair needs.
    fn upsample_steps(&self, span_us: i64) -> usize {
        if span_us <= 0 {
            return 1;
        }
        let ceiling = self.config.max_upsample.clamp(1, MAX_UPSAMPLE_CEILING);
        match self.config.upsample {
            Upsample::Off => 1,
            Upsample::Fixed(n) => n.clamp(1, ceiling),
            Upsample::Adaptive {
                max_events_per_pixel,
            } => {
                let budget = max_events_per_pixel.max(0.1);
                ((self.worst_contrast() / budget).ceil() as usize).clamp(1, ceiling)
            }
        }
    }

    /// The most events any single pixel would emit over this frame pair.
    ///
    /// The busiest pixel decides the subdivision: if it would emit n events over the pair,
    /// subdivide until no sub-interval asks for more than the budget from it.
    fn worst_contrast(&self) -> f32 {
        let contrast = |(pixel, (before, after)): (&PixelState, (&f32, &f32))| {
            (after - before).abs() / pixel.thres_pos
        };
        let pairs = self.prev_log.iter().zip(&self.log_now);
        if self.state.len() < PARALLEL_PIXEL_THRESHOLD {
            self.state
                .iter()
                .zip(pairs)
                .map(contrast)
                .fold(0.0_f32, f32::max)
        } else {
            self.state
                .par_iter()
                .zip(self.prev_log.par_iter().zip(&self.log_now))
                .map(contrast)
                .reduce(|| 0.0_f32, f32::max)
        }
    }

    fn empty_stream(&self) -> EventStream {
        EventStreamBuilder::new(self.width, self.height, 0.001).build()
    }
}

/// Splits `seed` into an independent RNG stream per (frame, pixel block).
///
/// The noise draws happen inside a parallel loop, so they cannot share one generator. What keeps a
/// run reproducible is that each block's seed is derived from the *configuration* — never from the
/// thread or the core count — and that [`PIXEL_BLOCK`] is a constant, so the same `seed` always
/// partitions the sensor the same way. SplitMix64's finaliser does the mixing: cheap, and enough
/// that neighbouring blocks show no visible correlation in their noise.
fn block_seed(seed: u64, frame: u64, block: usize) -> u64 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(frame.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add((block as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The probability that a pixel emits a shot-noise event of one polarity in a sub-interval,
/// tabulated over luma.
///
/// The probability depends on the pixel only through its luma, so evaluating it directly meant one
/// `exp` per pixel per sub-step — a couple of hundred million of them on a second of 1080p.
/// Tabulating once per sub-interval turns that into an array lookup.
struct NoiseTable {
    table: [f32; NOISE_LUT_LEN],
    /// False when noise is switched off or the sub-interval has no duration, so the caller skips
    /// the RNG draws entirely rather than drawing and comparing against zero.
    enabled: bool,
}

impl NoiseTable {
    fn new(rate_hz: f32, dt_s: f32) -> Self {
        let enabled = rate_hz > 0.0 && dt_s > 0.0;
        let mut table = [0.0_f32; NOISE_LUT_LEN];
        if enabled {
            for (index, slot) in table.iter_mut().enumerate() {
                let luma = index as f32 / (NOISE_LUT_LEN - 1) as f32;
                let scale = 1.0 - (1.0 - SHOT_NOISE_BRIGHT_FACTOR) * luma;
                // `1 - exp(-λ)` is the Poisson probability of at least one event, not `λ` itself.
                // The distinction only shows up once λ approaches 1 — but there a raw `λ` exceeds
                // 1, the comparison is always true, and the intensity dependence silently
                // disappears into saturation. At most one event per polarity per sub-interval is
                // emitted either way, so a rate high relative to the interval still saturates:
                // that is a reason to subdivide (see `Upsample`), not something to paper over.
                let lambda = rate_hz * scale * dt_s / 2.0;
                *slot = 1.0 - (-lambda).exp();
            }
        }
        Self { table, enabled }
    }

    #[inline]
    fn probability(&self, luma: f32) -> f32 {
        let index = (luma.clamp(0.0, 1.0) * (NOISE_LUT_LEN - 1) as f32) as usize;
        self.table[index]
    }
}

/// One pixel over one sub-interval: bandwidth, threshold crossings, leak and noise.
///
/// A free function rather than a method because the parallel loop hands out one `&mut PixelState`
/// per pixel and one RNG per block; a `&mut self` method could not be called from inside it.
#[allow(clippy::too_many_arguments)]
#[inline]
fn integrate_pixel(
    pixel: &mut PixelState,
    x: u16,
    y: u16,
    target_log: f32,
    luma: f32,
    t_start: i64,
    t_end: i64,
    dt_s: f32,
    config: &SimulatorConfig,
    noise: &NoiseTable,
    rng: &mut SmallRng,
    events: &mut Vec<SimEvent>,
) {
    // Photoreceptor lowpass: cutoff falls with intensity, floored so dark pixels still track.
    let filtered = if config.cutoff_hz > 0.0 && dt_s > 0.0 {
        let bandwidth = MIN_BANDWIDTH_FRACTION + (1.0 - MIN_BANDWIDTH_FRACTION) * luma;
        let tau = 1.0 / (2.0 * std::f32::consts::PI * config.cutoff_hz * bandwidth);
        let epsilon = (dt_s / tau).clamp(0.0, 1.0);
        pixel.lowpass += epsilon * (target_log - pixel.lowpass);
        pixel.lowpass
    } else {
        pixel.lowpass = target_log;
        target_log
    };

    // Leak: the memorised level decays, so a still scene still emits the occasional ON event.
    // Per-pixel thresholds decorrelate them, which is what stops leak looking like a metronome.
    if config.leak_rate_hz > 0.0 && dt_s > 0.0 {
        pixel.log_ref -= pixel.thres_pos * config.leak_rate_hz * dt_s;
    }

    let delta = filtered - pixel.log_ref;
    let positive = delta > 0.0;
    let threshold = if positive {
        pixel.thres_pos
    } else {
        pixel.thres_neg
    };
    let crossings = (delta.abs() / threshold).floor() as i64;

    if crossings > 0 {
        let span = (t_end - t_start) as f64;
        for k in 1..=crossings {
            // The k-th crossing happens when the interpolated level reaches k thresholds away,
            // which under the linear assumption is a fixed fraction of the way through.
            let fraction = (k as f32 * threshold / delta.abs()).clamp(0.0, 1.0) as f64;
            let t = t_start + (span * fraction) as i64;
            if t - pixel.last_t < config.refractory_us {
                continue;
            }
            pixel.last_t = t;
            events.push(SimEvent {
                t: t.max(0),
                x,
                y,
                positive,
            });
        }
        // Carry the remainder rather than discarding it, so slow ramps still fire eventually.
        let signed = if positive { 1.0 } else { -1.0 };
        pixel.log_ref += signed * crossings as f32 * threshold;
    }

    // Shot noise: Poisson-thin, quieter where the scene is bright.
    if noise.enabled {
        let probability = noise.probability(luma);
        for polarity in [true, false] {
            if rng.gen::<f32>() < probability {
                let t = t_start + (rng.gen::<f64>() * (t_end - t_start) as f64) as i64;
                if t - pixel.last_t >= config.refractory_us {
                    pixel.last_t = t;
                    events.push(SimEvent {
                        t: t.max(0),
                        x,
                        y,
                        positive: polarity,
                    });
                }
            }
        }
    }
}

/// Lin-log intensity map over 8-bit levels: linear below [`LIN_LOG_THRESHOLD`], `ln` above.
///
/// The two segments meet continuously — the linear slope is chosen as `ln(threshold)/threshold`
/// precisely so there is no discontinuity to emit a spurious event.
fn lin_log(intensity_255: f32) -> f32 {
    let x = intensity_255.max(0.0);
    if x <= LIN_LOG_THRESHOLD {
        x * (LIN_LOG_THRESHOLD.ln() / LIN_LOG_THRESHOLD)
    } else {
        x.ln()
    }
}

/// The sRGB electro-optical transfer function over all 256 8-bit levels.
///
/// [`linear_luma`] runs for every pixel of every frame, and the EOTF costs a `powf` per channel —
/// three per pixel, a quarter of a billion for a couple of seconds of 1080p, and by some margin the
/// most expensive thing in the decode path. The input is 8-bit, so the whole function fits in a
/// 1 KiB table and the conversion becomes three loads.
fn srgb_to_linear() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|level| {
            let v = level as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
    })
}

/// Rec. 601 luma of an sRGB pixel, linearised, in `[0, 1]`.
///
/// Linearising first matters: contrast is a ratio of *light*, and sRGB stores a gamma-encoded
/// value. Measuring log contrast on the encoded value bakes the display transfer function into
/// every threshold.
pub fn linear_luma(r: u8, g: u8, b: u8) -> f32 {
    let table = srgb_to_linear();
    0.2126 * table[usize::from(r)] + 0.7152 * table[usize::from(g)] + 0.0722 * table[usize::from(b)]
}

/// Converts a decoded RGB frame into the linear luma the simulator consumes.
pub fn luma_from_rgb(image: &Rgb8Image) -> Vec<f32> {
    let convert = |pixel: &[u8]| linear_luma(pixel[0], pixel[1], pixel[2]);
    if image.pixels.len() / 3 < PARALLEL_PIXEL_THRESHOLD {
        return image.pixels.chunks_exact(3).map(convert).collect();
    }
    // `par_chunks_exact` is indexed, so collecting preserves pixel order.
    image.pixels.par_chunks_exact(3).map(convert).collect()
}

/// How far a [`simulate_video_with_progress`] run has got.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimulateProgress {
    /// Frames pushed so far.
    pub frames: usize,
    /// Frames the source is expected to hold, when `ffprobe` could say — `None` for a stream whose
    /// length it does not report.
    pub total_frames: Option<usize>,
    /// Events emitted so far.
    pub events: usize,
}

/// Simulates a whole video file, calling `on_events` with each interval's events as they are
/// produced.
///
/// Streaming by construction: nothing is held but the current frame pair, so a long recording costs
/// memory proportional to one interval rather than to the whole output. `scale` decodes at a
/// different resolution, which is much cheaper than decoding full-size and downsampling after.
/// `max_frames` stops early.
pub fn simulate_video(
    path: &std::path::Path,
    config: SimulatorConfig,
    scale: Option<(usize, usize)>,
    max_frames: Option<usize>,
    on_events: impl FnMut(EventStream) -> std::io::Result<()>,
) -> std::io::Result<(usize, usize)> {
    simulate_video_on(
        path,
        config,
        crate::accel::Device::Cpu,
        None,
        scale,
        max_frames,
        on_events,
        |_| Ok(()),
    )
}

/// [`simulate_video`], reporting after every frame.
///
/// Split out rather than folded in because a simulation is long enough that a caller needs to see
/// it moving, and there is nothing else in the loop that knows the frame count. `on_progress`
/// returns a `Result` so that the same hook can stop the run — a caller polling for `Ctrl+C` has
/// nowhere else to do it, since a whole simulation is one uninterruptible call otherwise.
pub fn simulate_video_with_progress(
    path: &std::path::Path,
    config: SimulatorConfig,
    scale: Option<(usize, usize)>,
    max_frames: Option<usize>,
    on_events: impl FnMut(EventStream) -> std::io::Result<()>,
    on_progress: impl FnMut(SimulateProgress) -> std::io::Result<()>,
) -> std::io::Result<(usize, usize)> {
    simulate_video_on(
        path,
        config,
        crate::accel::Device::Cpu,
        None,
        scale,
        max_frames,
        on_events,
        on_progress,
    )
}

/// [`simulate_video_with_progress`] on a chosen device, optionally interpolating first.
///
/// See [`Simulator::on_device`] for exactly how the two backends differ, and
/// [`crate::interp`] for what `interpolate` does and why it happens here rather than inside the
/// pixel model.
#[allow(clippy::too_many_arguments)]
pub fn simulate_video_on(
    path: &std::path::Path,
    config: SimulatorConfig,
    device: crate::accel::Device,
    mut interpolate: Option<crate::interp::Interpolation<'_>>,
    scale: Option<(usize, usize)>,
    max_frames: Option<usize>,
    mut on_events: impl FnMut(EventStream) -> std::io::Result<()>,
    mut on_progress: impl FnMut(SimulateProgress) -> std::io::Result<()>,
) -> std::io::Result<(usize, usize)> {
    let mut decoder = crate::video::FfmpegDecoder::open(path, scale)?;
    let info = decoder.info();
    let total_frames = match (info.frames, max_frames) {
        (Some(total), Some(limit)) => Some(total.min(limit)),
        (total, limit) => total.or(limit),
    };
    let mut simulator = Simulator::new(info.width, info.height, config).on_device(device);
    // Frame index drives the clock rather than any wall time: the source's own frame rate is what
    // the timestamps have to be consistent with.
    let us_per_frame = (1_000_000.0 / info.fps.max(1e-6)).round() as i64;
    let (mut frames, mut events) = (0usize, 0usize);
    // The previous source frame, kept only when there is something to interpolate between.
    let mut previous: Option<Vec<f32>> = None;
    while let Some(image) = decoder.next_frame()? {
        if max_frames.is_some_and(|limit| frames >= limit) {
            break;
        }
        let luma = luma_from_rgb(&image);
        let t = frames as i64 * us_per_frame;
        let mut push = |simulator: &mut Simulator,
                        frame: &[f32],
                        t: i64,
                        events: &mut usize|
         -> std::io::Result<()> {
            let stream = simulator.push_frame(frame, t);
            *events += stream.len();
            if !stream.is_empty() {
                on_events(stream)?;
            }
            Ok(())
        };

        // Interpolated frames are pushed as ordinary source frames at proportional timestamps, so
        // the simulator sees a denser recording and nothing else changes.
        if let (Some(plan), Some(before)) = (interpolate.as_mut(), previous.as_ref()) {
            let fractions = plan.fractions();
            let between = plan
                .interpolator
                .between(before, &luma, info.width, info.height, &fractions)
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            for (fraction, frame) in fractions.iter().zip(&between) {
                let at = t - us_per_frame + (f64::from(*fraction) * us_per_frame as f64) as i64;
                push(&mut simulator, frame, at, &mut events)?;
            }
        }
        push(&mut simulator, &luma, t, &mut events)?;
        if interpolate.is_some() {
            previous = Some(luma);
        }
        frames += 1;
        on_progress(SimulateProgress {
            frames,
            total_frames,
            events,
        })?;
    }
    Ok((frames, events))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant(width: usize, height: usize, value: f32) -> Vec<f32> {
        vec![value; width * height]
    }

    #[test]
    fn lin_log_is_continuous_at_the_join() {
        let below = lin_log(LIN_LOG_THRESHOLD - 0.001);
        let at = lin_log(LIN_LOG_THRESHOLD);
        let above = lin_log(LIN_LOG_THRESHOLD + 0.001);
        assert!((below - at).abs() < 1e-3, "{below} vs {at}");
        assert!((above - at).abs() < 1e-3, "{above} vs {at}");
        // The linear segment must not blow up at zero the way ln would.
        assert!(lin_log(0.0).is_finite());
    }

    #[test]
    fn linear_luma_matches_srgb_endpoints() {
        assert!((linear_luma(0, 0, 0) - 0.0).abs() < 1e-6);
        assert!((linear_luma(255, 255, 255) - 1.0).abs() < 1e-4);
        // Mid-grey in sRGB is ~0.216 in linear light, not 0.5 — this is the whole point of
        // linearising, and a regression here silently changes every threshold.
        assert!((linear_luma(128, 128, 128) - 0.216).abs() < 0.01);
    }

    #[test]
    fn the_first_frame_only_seeds_state() {
        let mut sim = Simulator::new(4, 4, SimulatorConfig::ideal());
        assert!(sim.push_frame(&constant(4, 4, 0.5), 0).is_empty());
    }

    #[test]
    fn a_static_scene_emits_nothing_when_ideal() {
        let mut sim = Simulator::new(8, 8, SimulatorConfig::ideal());
        sim.push_frame(&constant(8, 8, 0.5), 0);
        for step in 1..5 {
            assert!(sim.push_frame(&constant(8, 8, 0.5), step * 1000).is_empty());
        }
    }

    #[test]
    fn event_count_matches_the_analytic_prediction() {
        // An ideal pixel emits floor(|Δ log I| / threshold) events. With a known step, that is
        // arithmetic — the check evlib's simulator does not make.
        let config = SimulatorConfig {
            pos_thres: 0.2,
            ..SimulatorConfig::ideal()
        };
        let (before, after) = (0.2_f32, 0.8_f32);
        let expected =
            ((lin_log(after * 255.0) - lin_log(before * 255.0)).abs() / 0.2).floor() as usize;
        assert!(
            expected > 1,
            "test is only meaningful for multiple crossings"
        );

        let mut sim = Simulator::new(4, 4, config);
        sim.push_frame(&constant(4, 4, before), 0);
        let events = sim.push_frame(&constant(4, 4, after), 10_000);
        assert_eq!(events.len(), expected * 16);
        assert!(
            events.ps().iter().all(|&p| p),
            "a brightening emits ON only"
        );
    }

    #[test]
    fn timestamps_are_interpolated_across_the_interval() {
        // The property evlib gets wrong: several crossings at one pixel must be spread through the
        // interval, not all stamped with the frame time.
        let mut sim = Simulator::new(
            1,
            1,
            SimulatorConfig {
                pos_thres: 0.1,
                ..SimulatorConfig::ideal()
            },
        );
        sim.push_frame(&constant(1, 1, 0.1), 0);
        let events = sim.push_frame(&constant(1, 1, 0.9), 10_000);
        assert!(events.len() > 2);
        let ts = events.ts();
        assert!(ts.windows(2).all(|w| w[0] <= w[1]), "must be ascending");
        assert!(
            ts.first() != ts.last(),
            "all timestamps identical — interpolation is not happening"
        );
        assert!(ts.iter().all(|&t| (0..=10_000).contains(&t)));
    }

    #[test]
    fn output_is_globally_sorted_across_pixels() {
        // Per-interval sorting is what makes this true; a regression would show up as unsorted
        // output the moment more than one pixel fires.
        let mut sim = Simulator::new(16, 16, SimulatorConfig::default());
        sim.push_frame(&constant(16, 16, 0.2), 0);
        for step in 1..6 {
            let brightness = 0.2 + 0.1 * step as f32;
            let events = sim.push_frame(&constant(16, 16, brightness), step * 10_000);
            assert!(events.ts().windows(2).all(|w| w[0] <= w[1]));
        }
    }

    #[test]
    fn timestamps_are_never_negative() {
        // `EventStream::iter` casts to u64, so a negative timestamp silently becomes enormous and
        // corrupts every time-surface representation downstream.
        let mut sim = Simulator::new(4, 4, SimulatorConfig::default());
        sim.push_frame(&constant(4, 4, 0.5), 0);
        let events = sim.push_frame(&constant(4, 4, 0.9), 5_000);
        assert!(events.ts().iter().all(|&t| t >= 0));
    }

    #[test]
    fn same_seed_gives_identical_output() {
        let run = || {
            let mut sim = Simulator::new(8, 8, SimulatorConfig::default());
            sim.push_frame(&constant(8, 8, 0.3), 0);
            sim.push_frame(&constant(8, 8, 0.6), 20_000)
        };
        let (first, second) = (run(), run());
        assert_eq!(first.ts(), second.ts());
        assert_eq!(first.xs(), second.xs());
    }

    /// A sensor big enough to cross [`PARALLEL_PIXEL_THRESHOLD`], so the tests below exercise the
    /// rayon path rather than the serial fallback.
    const PARALLEL_SIDE: usize = 288;
    const _: () = assert!(
        PARALLEL_SIDE * PARALLEL_SIDE >= PARALLEL_PIXEL_THRESHOLD,
        "the test sensor must be large enough to take the parallel path"
    );

    fn run_parallel_sensor() -> EventStream {
        let mut sim = Simulator::new(
            PARALLEL_SIDE,
            PARALLEL_SIDE,
            SimulatorConfig {
                seed: 12345,
                ..SimulatorConfig::default()
            },
        );
        let mut last = sim.empty_stream();
        for step in 0..4 {
            let brightness = 0.2 + 0.15 * step as f32;
            last = sim.push_frame(
                &constant(PARALLEL_SIDE, PARALLEL_SIDE, brightness),
                step as i64 * 20_000,
            );
        }
        last
    }

    #[test]
    fn output_does_not_depend_on_the_thread_count() {
        // The whole reason the noise RNG is split per pixel block rather than shared: a run has to
        // be reproducible from its seed, and "reproducible" has to survive being run on a different
        // machine with a different number of cores. If this fails, `block_seed` or `PIXEL_BLOCK`
        // has picked up a dependency on the rayon pool.
        let in_pool = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("building a rayon pool")
                .install(run_parallel_sensor)
        };
        let (one, many) = (in_pool(1), in_pool(8));
        assert_eq!(one.len(), many.len(), "event counts differ");
        assert_eq!(one.ts(), many.ts());
        assert_eq!(one.xs(), many.xs());
        assert_eq!(one.ys(), many.ys());
        assert_eq!(one.ps(), many.ps());
    }

    #[test]
    fn parallel_output_is_sorted_and_in_bounds() {
        let events = run_parallel_sensor();
        assert!(!events.is_empty(), "a brightening sensor must emit");
        assert!(events.ts().windows(2).all(|w| w[0] <= w[1]));
        assert!(events
            .xs()
            .iter()
            .all(|&x| usize::from(x) < PARALLEL_SIDE));
        assert!(events
            .ys()
            .iter()
            .all(|&y| usize::from(y) < PARALLEL_SIDE));
    }

    #[test]
    fn every_pixel_block_is_reached() {
        // A coordinate walked incrementally across a block (rather than divided out per pixel) is
        // easy to get subtly wrong at a block boundary, and the symptom would be a whole band of
        // the sensor silently never firing. An ideal ramp fires every pixel exactly the same
        // number of times, so the coordinates must cover the grid exactly.
        let (width, height) = (PARALLEL_SIDE, PARALLEL_SIDE);
        let config = SimulatorConfig {
            pos_thres: 0.2,
            ..SimulatorConfig::ideal()
        };
        let mut sim = Simulator::new(width, height, config);
        sim.push_frame(&constant(width, height, 0.2), 0);
        let events = sim.push_frame(&constant(width, height, 0.8), 10_000);

        let mut seen = vec![0usize; width * height];
        for index in 0..events.len() {
            seen[usize::from(events.ys()[index]) * width + usize::from(events.xs()[index])] += 1;
        }
        let expected = seen[0];
        assert!(expected > 0, "an ideal ramp must fire every pixel");
        assert!(
            seen.iter().all(|&count| count == expected),
            "every pixel must fire the same number of times on a uniform ramp"
        );
    }

    #[test]
    fn max_upsample_caps_the_subdivision() {
        // The knob exists because adaptive upsampling is driven by the busiest pixel, so a single
        // hard edge can cost the ceiling in full-sensor passes. Asserted on the step count itself
        // rather than on the events, because a *linear* ramp interpolates to the same timestamps
        // however finely it is subdivided — the cost is real even where the output barely moves.
        let steps_for = |max_upsample: usize| {
            let mut sim = Simulator::new(
                4,
                4,
                SimulatorConfig {
                    pos_thres: 0.05,
                    max_upsample,
                    upsample: Upsample::default(),
                    ..SimulatorConfig::ideal()
                },
            );
            sim.prev_log.fill(lin_log(0.1 * 255.0));
            sim.log_now.fill(lin_log(0.9 * 255.0));
            sim.upsample_steps(10_000)
        };
        // The contrast here asks for ~44 sub-steps, so the ceiling is what binds below that.
        let uncapped = steps_for(MAX_UPSAMPLE);
        assert!(
            uncapped > 10,
            "this contrast should ask for a real subdivision, got {uncapped}"
        );
        assert_eq!(steps_for(1), 1, "a ceiling of 1 disables subdivision");
        assert_eq!(steps_for(10), 10, "the ceiling binds below what is asked");
        assert_eq!(
            steps_for(MAX_UPSAMPLE_CEILING),
            uncapped,
            "raising the ceiling past the demand changes nothing"
        );
    }

    #[test]
    fn leak_alone_fires_at_about_its_rate() {
        // Thresholds unreachable by the (static) scene, so every event must come from leak.
        let config = SimulatorConfig {
            leak_rate_hz: 10.0,
            shot_noise_rate_hz: 0.0,
            sigma_thres: 0.0,
            cutoff_hz: 0.0,
            refractory_us: 0,
            upsample: Upsample::Off,
            ..SimulatorConfig::default()
        };
        let (width, height) = (8, 8);
        let mut sim = Simulator::new(width, height, config);
        sim.push_frame(&constant(width, height, 0.5), 0);
        let mut total = 0;
        // One second in 100 ms steps.
        for step in 1..=10 {
            let events = sim.push_frame(&constant(width, height, 0.5), step * 100_000);
            assert!(events.ps().iter().all(|&p| p), "leak emits ON events");
            total += events.len();
        }
        let expected = 10.0 * (width * height) as f64;
        let ratio = total as f64 / expected;
        assert!(
            ratio > 0.5 && ratio < 1.5,
            "leak produced {total}, expected ~{expected}"
        );
    }

    #[test]
    fn shot_noise_alone_scales_with_its_rate() {
        let noisy = |rate: f32| {
            let config = SimulatorConfig {
                shot_noise_rate_hz: rate,
                leak_rate_hz: 0.0,
                cutoff_hz: 0.0,
                upsample: Upsample::Off,
                ..SimulatorConfig::default()
            };
            let mut sim = Simulator::new(16, 16, config);
            sim.push_frame(&constant(16, 16, 0.5), 0);
            (1..=10)
                .map(|step| sim.push_frame(&constant(16, 16, 0.5), step * 100_000).len())
                .sum::<usize>()
        };
        assert_eq!(noisy(0.0), 0);
        let (low, high) = (noisy(5.0), noisy(50.0));
        assert!(low > 0, "some noise expected at 5 Hz");
        assert!(
            high > low * 3,
            "10x the rate should give far more events: {low} vs {high}"
        );
    }

    #[test]
    fn threshold_mismatch_desynchronises_pixels() {
        // With identical thresholds a uniform ramp fires every pixel in lockstep; with mismatch the
        // firing times spread. That spread is the point of modelling mismatch at all.
        // Measured as distinct timestamps *per pixel's worth of events*: a pixel crossing its
        // threshold n times legitimately produces n distinct times even in lockstep, so the
        // signal of mismatch is that pixels stop agreeing with each other.
        let spread = |sigma: f32| {
            let config = SimulatorConfig {
                sigma_thres: sigma,
                leak_rate_hz: 0.0,
                shot_noise_rate_hz: 0.0,
                cutoff_hz: 0.0,
                refractory_us: 0,
                upsample: Upsample::Off,
                ..SimulatorConfig::default()
            };
            let pixels = 16 * 16;
            let mut sim = Simulator::new(16, 16, config);
            sim.push_frame(&constant(16, 16, 0.3), 0);
            let events = sim.push_frame(&constant(16, 16, 0.45), 10_000);
            let unique: std::collections::HashSet<i64> = events.ts().iter().copied().collect();
            // Events per pixel, and how many distinct times those landed on overall.
            (events.len() / pixels, unique.len())
        };
        let (per_pixel, unique) = spread(0.0);
        assert!(per_pixel > 0);
        assert_eq!(
            unique, per_pixel,
            "identical thresholds must put every pixel on the same {per_pixel} timestamps"
        );
        let (_, spread_unique) = spread(0.05);
        assert!(
            spread_unique > unique,
            "mismatch must spread the firing times: {spread_unique} vs {unique}"
        );
    }

    #[test]
    fn adaptive_upsampling_subdivides_high_contrast_pairs() {
        let steps_for = |upsample: Upsample, before: f32, after: f32| {
            let config = SimulatorConfig {
                pos_thres: 0.1,
                upsample,
                ..SimulatorConfig::ideal()
            };
            let mut sim = Simulator::new(4, 4, config);
            sim.push_frame(&constant(4, 4, before), 0);
            let events = sim.push_frame(&constant(4, 4, after), 10_000);
            let unique: std::collections::HashSet<i64> = events.ts().iter().copied().collect();
            unique.len()
        };
        // Off: every crossing is placed by interpolation within the single interval.
        let off = steps_for(Upsample::Off, 0.1, 0.9);
        // Adaptive: the pair is subdivided first, so crossings land on a finer time grid.
        let adaptive = steps_for(
            Upsample::Adaptive {
                max_events_per_pixel: 1.0,
            },
            0.1,
            0.9,
        );
        assert!(off > 1 && adaptive > 1);
        assert!(
            adaptive >= off,
            "subdividing must not coarsen timing: {adaptive} vs {off}"
        );
    }

    #[test]
    fn simulate_video_streams_a_real_clip() {
        // End to end through ffmpeg: synthesise a moving pattern, simulate it, and check events
        // arrive in ascending time across interval boundaries — the property per-interval sorting
        // is supposed to guarantee globally.
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_err()
        {
            return;
        }
        let mut path = std::env::temp_dir();
        path.push(format!("eventcv-sim-{}.mp4", std::process::id()));
        let made = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=1"])
            .args(["-pix_fmt", "yuv420p"])
            .arg(&path)
            .status();
        if !matches!(made, Ok(status) if status.success()) {
            return;
        }

        let mut last = -1_i64;
        let mut intervals = 0;
        let (frames, events) =
            simulate_video(&path, SimulatorConfig::default(), None, None, |stream| {
                for &t in stream.ts() {
                    assert!(t >= last, "events must not go backwards across intervals");
                    last = t;
                }
                intervals += 1;
                Ok(())
            })
            .expect("simulation should succeed");

        assert_eq!(frames, 30, "one second at 30 fps");
        assert!(events > 0, "a moving test pattern must generate events");
        assert!(
            intervals > 1,
            "events should arrive across several intervals"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_darkening_scene_emits_off_events() {
        let mut sim = Simulator::new(4, 4, SimulatorConfig::ideal());
        sim.push_frame(&constant(4, 4, 0.9), 0);
        let events = sim.push_frame(&constant(4, 4, 0.2), 10_000);
        assert!(!events.is_empty());
        assert!(
            events.ps().iter().all(|&p| !p),
            "a darkening emits OFF only"
        );
    }

    #[test]
    fn the_refractory_period_thins_a_burst() {
        let config = |refractory_us| SimulatorConfig {
            pos_thres: 0.05,
            refractory_us,
            ..SimulatorConfig::ideal()
        };
        let count = |refractory_us| {
            let mut sim = Simulator::new(1, 1, config(refractory_us));
            sim.push_frame(&constant(1, 1, 0.1), 0);
            sim.push_frame(&constant(1, 1, 0.9), 10_000).len()
        };
        let free = count(0);
        let limited = count(4_000);
        assert!(
            free > limited,
            "refractory must suppress: {free} vs {limited}"
        );
        assert!(limited > 0, "but not suppress everything");
    }
}

/// The GPU pixel model against the CPU one — the contract [`Simulator::on_device`] states.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::{Simulator, SimulatorConfig, Upsample};
    use crate::accel::Device;
    use crate::EventStream;

    fn skip_without_gpu() -> bool {
        if crate::accel::gpu_available() {
            return false;
        }
        assert!(
            std::env::var("EVENTCV_REQUIRE_GPU").is_err(),
            "EVENTCV_REQUIRE_GPU is set but no adapter was found"
        );
        true
    }

    /// A moving edge: the frames that make a DVS actually fire, rather than a flat field that
    /// exercises nothing but the leak term.
    fn frames(width: usize, height: usize, count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|index| {
                let edge = (index * width) / count.max(1);
                (0..width * height)
                    .map(|pixel| if pixel % width < edge { 0.85 } else { 0.15 })
                    .collect()
            })
            .collect()
    }

    fn run(config: SimulatorConfig, device: Device) -> Vec<EventStream> {
        let (width, height) = (64, 48);
        let mut simulator = Simulator::new(width, height, config).on_device(device);
        frames(width, height, 12)
            .iter()
            .enumerate()
            .map(|(index, frame)| simulator.push_frame(frame, index as i64 * 10_000))
            .collect()
    }

    fn columns(streams: &[EventStream]) -> (Vec<u16>, Vec<u16>, Vec<i64>, Vec<bool>) {
        let mut out = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for stream in streams {
            out.0.extend_from_slice(stream.xs());
            out.1.extend_from_slice(stream.ys());
            out.2.extend_from_slice(stream.ts());
            out.3.extend_from_slice(stream.ps());
        }
        out
    }

    /// The regression gate. With the random terms off, the only difference left between the
    /// backends is `exp`/`floor` rounding — and on a threshold model that either changes nothing or
    /// changes an event, so "the same events" is the right assertion, not "close".
    #[test]
    fn a_noiseless_sensor_produces_the_same_events_on_both_backends() {
        if skip_without_gpu() {
            return;
        }
        for upsample in [Upsample::Off, Upsample::Fixed(4)] {
            let config = SimulatorConfig {
                upsample,
                ..SimulatorConfig::ideal()
            };
            let cpu = columns(&run(config, Device::Cpu));
            let gpu = columns(&run(config, Device::Gpu));
            assert_eq!(cpu.2, gpu.2, "{upsample:?}: timestamps");
            assert_eq!(cpu.0, gpu.0, "{upsample:?}: x");
            assert_eq!(cpu.1, gpu.1, "{upsample:?}: y");
            assert_eq!(cpu.3, gpu.3, "{upsample:?}: polarity");
        }
    }

    /// Mismatch is drawn on the host and uploaded, so a sensor built from one seed is the same
    /// silicon on either backend: the same pixels fire, the same number of times, with the same
    /// polarity.
    ///
    /// Timestamps are where the two can part, and only just. The moment of the k-th crossing is
    /// `k * threshold / |delta|` of the way through the sub-interval — a fraction the CPU widens to
    /// `f64` before scaling, and the GPU cannot, because WGSL has no `f64`. On a run of ~25 000
    /// events that puts a couple of them one microsecond apart. The assertion is that bound, not a
    /// hopeful tolerance: anything larger would mean the model itself had diverged.
    #[test]
    fn threshold_mismatch_is_the_same_silicon_on_both_backends() {
        if skip_without_gpu() {
            return;
        }
        let config = SimulatorConfig {
            sigma_thres: 0.05,
            seed: 7,
            ..SimulatorConfig::ideal()
        };
        let (cpu_x, cpu_y, cpu_t, cpu_p) = columns(&run(config, Device::Cpu));
        let (gpu_x, gpu_y, gpu_t, gpu_p) = columns(&run(config, Device::Gpu));
        assert_eq!(cpu_t.len(), gpu_t.len(), "event count");
        assert_eq!(cpu_x, gpu_x, "x");
        assert_eq!(cpu_y, gpu_y, "y");
        assert_eq!(cpu_p, gpu_p, "polarity");

        let apart: Vec<i64> = cpu_t
            .iter()
            .zip(&gpu_t)
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .filter(|difference| *difference > 0)
            .collect();
        assert!(
            apart.iter().all(|difference| *difference <= 1),
            "timestamps should differ by at most a microsecond, got {:?}",
            apart.iter().max()
        );
        assert!(
            apart.len() * 100 < cpu_t.len(),
            "{} of {} timestamps differ; that is rounding turning into divergence",
            apart.len(),
            cpu_t.len()
        );
    }

    /// With noise on the two backends draw different numbers by design, so the claim is
    /// distributional: the same scene should produce a comparable event rate, not the same events.
    #[test]
    fn shot_noise_agrees_in_rate_rather_than_event_for_event() {
        if skip_without_gpu() {
            return;
        }
        let config = SimulatorConfig {
            shot_noise_rate_hz: 500.0,
            upsample: Upsample::Off,
            ..SimulatorConfig::ideal()
        };
        let cpu: usize = run(config, Device::Cpu).iter().map(EventStream::len).sum();
        let gpu: usize = run(config, Device::Gpu).iter().map(EventStream::len).sum();
        let ratio = gpu as f64 / cpu as f64;
        assert!(
            (0.9..1.1).contains(&ratio),
            "noise rates should agree to ~10%, got {cpu} vs {gpu} ({ratio:.3})"
        );
    }

    /// The counter-based generator's whole point: same seed, same events, every run.
    #[test]
    fn a_noisy_run_is_reproducible_on_the_gpu() {
        if skip_without_gpu() {
            return;
        }
        let config = SimulatorConfig {
            shot_noise_rate_hz: 500.0,
            leak_rate_hz: 5.0,
            ..SimulatorConfig::default()
        };
        let first = columns(&run(config, Device::Gpu));
        for _ in 0..2 {
            assert_eq!(first.2, columns(&run(config, Device::Gpu)).2);
        }
    }
}

/// Interpolation as a *preprocessing* stage — that it changes what the simulator sees, and that
/// leaving it out changes nothing.
#[cfg(test)]
mod interp_tests {
    use super::{simulate_video_on, SimulatorConfig, Upsample};
    use crate::interp::{Interpolation, LinearInterpolator};

    /// A short synthetic clip, or `None` when ffmpeg is not on `PATH` — the same shape the other
    /// video-backed test in this module uses.
    fn clip() -> Option<std::path::PathBuf> {
        std::process::Command::new("ffmpeg").arg("-version").output().ok()?;
        let path = std::env::temp_dir().join(format!("eventcv_interp_{}.mp4", std::process::id()));
        let made = std::process::Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=30:duration=1"])
            .args(["-pix_fmt", "yuv420p"])
            .arg(&path)
            .status()
            .ok()?;
        made.success().then_some(path)
    }

    fn count(path: &std::path::Path, factor: usize) -> usize {
        let mut linear = LinearInterpolator;
        let plan = (factor > 1).then_some(Interpolation {
            interpolator: &mut linear as &mut dyn crate::interp::FrameInterpolator,
            factor,
        });
        let mut events = 0;
        simulate_video_on(
            path,
            SimulatorConfig {
                upsample: Upsample::Off,
                ..SimulatorConfig::ideal()
            },
            crate::accel::Device::Cpu,
            plan,
            None,
            None,
            |stream| {
                events += stream.len();
                Ok(())
            },
            |_| Ok(()),
        )
        .expect("simulating the clip");
        events
    }

    /// The linear baseline is what the simulator's own subdivision already does, so inserting
    /// frames that way must not change the events — which is the strongest statement available that
    /// the plumbing is a *preprocessing* stage and not a change to the model.
    #[test]
    fn a_linear_interpolator_reproduces_the_uninterpolated_run() {
        let Some(path) = clip() else {
            return; // no ffmpeg here
        };
        let plain = count(&path, 1);
        assert!(plain > 0, "the clip should produce events at all");
        for factor in [2, 4] {
            let interpolated = count(&path, factor);
            // Not exactly equal: the extra frames give the threshold model more chances to cross,
            // so timestamps sharpen. The count is what must not move.
            let drift = (interpolated as f64 - plain as f64).abs() / plain as f64;
            assert!(
                drift < 0.02,
                "factor {factor}: {plain} events became {interpolated}, which is a change in the \
                 model rather than in the timing"
            );
        }
        std::fs::remove_file(&path).ok();
    }
}
