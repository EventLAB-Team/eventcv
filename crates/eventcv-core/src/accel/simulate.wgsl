// The DVS pixel model, one invocation per pixel — the GPU twin of `simulate::integrate_pixel`.
//
// The whole sub-step loop for a frame pair runs inside one dispatch, exactly as the CPU keeps it
// inside one rayon task: a pixel's state is sequential in time, so splitting the steps across
// dispatches would mean writing the state out and reading it back for every sub-interval.
//
// Two things differ from the CPU deliberately, and both are documented in `simulate::gpu`:
//
//   * Random draws come from a counter-based generator keyed on
//     `(seed, frame, step, pixel, stream)` rather than from a sequential stream per pixel block.
//     A per-thread kernel cannot cheaply replay a sequential generator, and a counter-based one is
//     reproducible across runs *and* across devices, which is the property that actually matters.
//     It is a different sample path from the CPU's, so noise is compared statistically.
//   * Events are appended through one atomic cursor, so their order within an interval is
//     whatever the scheduler produced. The host sorts every interval by `(t, y, x, polarity)`
//     regardless — the CPU path does the same, for the same reason — so the result is identical.

struct SimParams {
    width: u32,
    height: u32,
    // Sub-steps in *this* dispatch, and where they sit in the interval. A frame pair that would
    // need more event storage than one buffer binding can hold is split across dispatches; the
    // pixel state carries over, so splitting changes nothing about the result.
    steps: u32,
    step_offset: u32,
    total_steps: u32,
    // How many events the output buffer can take; the host grows it and retries on overflow.
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

/// Mirrors `simulate::PixelState`: memorised level, lowpass state, the two mismatched thresholds,
/// and the last emission time.
struct PixelState {
    log_ref: f32,
    lowpass: f32,
    thres_pos: f32,
    thres_neg: f32,
    last_t: i32,
}

/// Mirrors `simulate::SimEvent`, padded to a 16-byte stride.
struct SimEvent {
    t: i32,
    x: u32,
    y: u32,
    positive: u32,
}

@group(0) @binding(0) var<uniform> sim: SimParams;
/// The frame pair, as four consecutive planes of `width * height`: the previous and incoming
/// lin-log levels, then the previous and incoming luma. One buffer rather than four because the
/// conservative wgpu limits this device is opened with allow few storage bindings per stage, and
/// four planes that are always uploaded together do not need four of them.
@group(0) @binding(1) var<storage, read> planes: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<PixelState>;
@group(0) @binding(3) var<storage, read_write> events: array<SimEvent>;
/// `[0]` is the write cursor — which also reports how many events *would* have been written, so
/// the host can tell an exact fill from an overflow.
@group(0) @binding(4) var<storage, read_write> cursor: array<atomic<u32>>;
/// `steps + 1` sub-interval boundaries in microseconds, computed on the host.
@group(0) @binding(5) var<storage, read> bounds: array<i32>;

const WORKGROUP: u32 = 64u;
const MIN_BANDWIDTH_FRACTION: f32 = 0.1;
const SHOT_NOISE_BRIGHT_FACTOR: f32 = 0.25;
const TAU: f32 = 6.2831853;

/// One round of a counter-based mixer (SplitMix64's finaliser, on 32-bit halves).
///
/// Counter-based rather than sequential so that a pixel's k-th draw depends only on *which* draw it
/// is — no per-thread state to carry, identical from run to run, and identical on any device,
/// because every operation here is integer.
fn random(a: u32, b: u32, c: u32, d: u32) -> f32 {
    var h = a ^ (b * 0x9E3779B9u) ^ (c * 0x85EBCA6Bu) ^ (d * 0xC2B2AE35u);
    h = h ^ (h >> 16u);
    h = h * 0x7FEB352Du;
    h = h ^ (h >> 15u);
    h = h * 0x846CA68Bu;
    h = h ^ (h >> 16u);
    // 24 bits into [0,1), the same resolution `rand`'s `f32` sampling gives.
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

/// The probability that a pixel emits a shot-noise event of one polarity in this sub-interval.
/// `1 - exp(-λ)` rather than `λ`, matching `simulate::NoiseTable`.
fn noise_probability(l: f32, dt_s: f32) -> f32 {
    let scale = 1.0 - (1.0 - SHOT_NOISE_BRIGHT_FACTOR) * clamp(l, 0.0, 1.0);
    let lambda = sim.shot_noise_rate_hz * scale * dt_s / 2.0;
    return 1.0 - exp(-lambda);
}

fn emit(t: i32, x: u32, y: u32, positive: bool) {
    let slot = atomicAdd(&cursor[0], 1u);
    if (slot >= sim.capacity) { return; }
    events[slot] = SimEvent(max(t, 0), x, y, select(0u, 1u, positive));
}

@compute @workgroup_size(WORKGROUP)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let pixels = sim.width * sim.height;
    if (index >= pixels) { return; }

    let x = index % sim.width;
    let y = index / sim.width;
    var pixel = state[index];

    for (var local = 1u; local <= sim.steps; local = local + 1u) {
        // The interpolation and the noise key both use the step's place in the whole interval, not
        // its place in this dispatch, so a split interval produces exactly what an unsplit one does.
        let step = sim.step_offset + local;
        let alpha = f32(step) / f32(sim.total_steps);
        // Sub-interval bounds come from the host, which computes them in `f64` exactly as the
        // CPU path does. WGSL has no `f64`, and doing this division in `f32` would drift the
        // timestamps by a microsecond or so over a long interval — a difference in the output
        // rather than in the arithmetic.
        let t_start = bounds[local - 1u];
        let t_end = bounds[local];
        let dt_s = f32(t_end - t_start) / 1e6;

        let prev_log = planes[index];
        let log_now = planes[pixels + index];
        let prev_luma = planes[2u * pixels + index];
        let luma_now = planes[3u * pixels + index];
        let target_log = prev_log + (log_now - prev_log) * alpha;
        let pixel_luma = prev_luma + (luma_now - prev_luma) * alpha;

        // Photoreceptor lowpass: cutoff falls with intensity, floored so dark pixels still track.
        var filtered: f32;
        if (sim.cutoff_hz > 0.0 && dt_s > 0.0) {
            let bandwidth = MIN_BANDWIDTH_FRACTION + (1.0 - MIN_BANDWIDTH_FRACTION) * pixel_luma;
            let tau = 1.0 / (TAU * sim.cutoff_hz * bandwidth);
            let epsilon = clamp(dt_s / tau, 0.0, 1.0);
            pixel.lowpass = pixel.lowpass + epsilon * (target_log - pixel.lowpass);
            filtered = pixel.lowpass;
        } else {
            pixel.lowpass = target_log;
            filtered = target_log;
        }

        // Leak: the memorised level decays, so a still scene still emits the occasional ON event.
        if (sim.leak_rate_hz > 0.0 && dt_s > 0.0) {
            pixel.log_ref = pixel.log_ref - pixel.thres_pos * sim.leak_rate_hz * dt_s;
        }

        let delta = filtered - pixel.log_ref;
        let positive = delta > 0.0;
        let threshold = select(pixel.thres_neg, pixel.thres_pos, positive);
        let crossings = i32(floor(abs(delta) / threshold));

        if (crossings > 0) {
            let span = f32(t_end - t_start);
            for (var k = 1; k <= crossings; k = k + 1) {
                let fraction = clamp(f32(k) * threshold / abs(delta), 0.0, 1.0);
                let t = t_start + i32(span * fraction);
                if (t - pixel.last_t < sim.refractory_us) { continue; }
                pixel.last_t = t;
                emit(t, x, y, positive);
            }
            let direction = select(-1.0, 1.0, positive);
            pixel.log_ref = pixel.log_ref + direction * f32(crossings) * threshold;
        }

        // Shot noise: Poisson-thin, quieter where the scene is bright.
        if (sim.shot_noise_rate_hz > 0.0 && dt_s > 0.0) {
            let probability = noise_probability(pixel_luma, dt_s);
            for (var polarity = 0u; polarity < 2u; polarity = polarity + 1u) {
                if (random(sim.seed_lo ^ polarity, sim.seed_hi, sim.frame ^ (step << 16u), index)
                    < probability) {
                    let offset = random(sim.seed_hi ^ polarity, sim.seed_lo, step, index ^ 0x5bf03635u);
                    let t = t_start + i32(offset * f32(t_end - t_start));
                    if (t - pixel.last_t >= sim.refractory_us) {
                        pixel.last_t = t;
                        emit(t, x, y, polarity == 1u);
                    }
                }
            }
        }
    }

    state[index] = pixel;
}
