// Representation kernels — the GPU twins of `crate::representation`.
//
// One event per invocation, scattering into a per-cell accumulator. Every accumulator is an
// integer atomic, so a cell's result does not depend on the order the invocations reached it:
// integer addition commutes exactly where float addition does not, which is what makes these
// kernels reproducible run to run rather than merely close.
//
// Events arrive as two `u32`s: coordinates packed as `(y << 16) | x`, and the event's age in
// timestamp ticks measured back from the newest event in the stream (so `0` is the newest). Ages
// rather than timestamps because every representation here is a function of age, and because a
// `u32` of ticks covers any window a representation is asked about while an absolute Unix
// microsecond timestamp does not.

struct Params {
    width: u32,
    height: u32,
    n_events: u32,
    // Channel count for the voxel grid; unused elsewhere.
    bins: u32,
    // Milliseconds per timestamp tick, so ages convert without the host rescaling them.
    scale_ms: f32,
    // Window (voxel) or decay constant (time surfaces), in milliseconds.
    span_ms: f32,
    // Fixed-point scale: how many integer units one unit of the accumulated value is worth.
    fixed_one: f32,
    // The oldest age, in ticks, still inside the window.
    //
    // The host computes this in `f64` with the same expression the CPU implementation compares
    // against, and the kernel then makes an *integer* comparison. Deciding it here in `f32` instead
    // would put events whose age lands exactly on the window edge on one side for the CPU and the
    // other for the GPU — a discrete disagreement worth a whole event's contribution, which no
    // float tolerance can absorb.
    max_age_ticks: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coords: array<u32>;
@group(0) @binding(2) var<storage, read> ages: array<u32>;
@group(0) @binding(3) var<storage, read_write> cells: array<atomic<i32>>;
// Set to 1 when a fixed-point accumulator would have wrapped; the host turns that into an error
// rather than handing back a silently wrong frame.
@group(0) @binding(4) var<storage, read_write> saturated: atomic<u32>;

const WORKGROUP: u32 = 256u;
// Q16.16 leaves ±32768 of headroom on a cell, which no physical pixel reaches inside a window; a
// cell that would exceed it trips `saturated` instead of wrapping.
const FIXED_LIMIT: i32 = 2147483000;

fn plane() -> u32 {
    return params.width * params.height;
}

/// The event's cell within one plane, or `0xffffffff` when it falls outside the sensor.
///
/// The coordinate word is `(polarity << 31) | (y << 16) | x`, so `y` is masked to fifteen bits —
/// far more than any sensor, and it leaves the top bit for the polarity that would otherwise need
/// a third buffer.
fn cell_of(index: u32) -> u32 {
    let packed = coords[index];
    let x = packed & 0xffffu;
    let y = (packed >> 16u) & 0x7fffu;
    if (x >= params.width || y >= params.height) {
        return 0xffffffffu;
    }
    return y * params.width + x;
}

fn age_ms(index: u32) -> f32 {
    return f32(ages[index]) * params.scale_ms;
}

/// Polarity, carried in the coordinate word's top bit (see `cell_of`).
fn positive(index: u32) -> bool {
    return (coords[index] & 0x80000000u) != 0u;
}

/// Adds a fixed-point contribution, flagging rather than wrapping on overflow.
fn accumulate(cell: u32, value: f32) {
    let quantised = i32(round(value * params.fixed_one));
    let previous = atomicAdd(&cells[cell], quantised);
    if (abs(previous) > FIXED_LIMIT - abs(quantised)) {
        atomicStore(&saturated, 1u);
    }
}

// ---------------------------------------------------------------------------------------------

/// Event count per pixel, both polarities. Exact: every contribution is one.
@compute @workgroup_size(WORKGROUP)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.n_events) { return; }
    let cell = cell_of(index);
    if (cell == 0xffffffffu) { return; }
    atomicAdd(&cells[cell], 1);
}

/// Per-polarity counts — plane 0 positive, plane 1 negative. The count-mask image is built from
/// these on the host, where its percentile lives.
@compute @workgroup_size(WORKGROUP)
fn polarity_counts(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.n_events) { return; }
    let cell = cell_of(index);
    if (cell == 0xffffffffu) { return; }
    let offset = select(plane(), 0u, positive(index));
    atomicAdd(&cells[cell + offset], 1);
}

/// Voxel grid: each event's signed polarity split linearly between the two time bins its age falls
/// between, matching `representation::voxel`.
@compute @workgroup_size(WORKGROUP)
fn voxel(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.n_events) { return; }
    let cell = cell_of(index);
    if (cell == 0xffffffffu) { return; }
    if (ages[index] > params.max_age_ticks) { return; }
    let age = age_ms(index);

    var position = 0.0;
    if (params.bins > 1u) {
        position = (1.0 - age / params.span_ms) * f32(params.bins - 1u);
    }
    let lower = u32(floor(position));
    let upper = u32(ceil(position));
    let sign = select(-1.0, 1.0, positive(index));
    if (lower == upper) {
        accumulate(lower * plane() + cell, sign);
    } else {
        let weight = position - floor(position);
        accumulate(lower * plane() + cell, sign * (1.0 - weight));
        accumulate(upper * plane() + cell, sign * weight);
    }
}

/// Time surface: the smallest age seen at each (pixel, polarity), which the host maps through
/// `exp(-age/tau)`. `atomicMin` rather than a float max, so ties and ordering cannot matter.
@compute @workgroup_size(WORKGROUP)
fn time_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.n_events) { return; }
    let cell = cell_of(index);
    if (cell == 0xffffffffu) { return; }
    let offset = select(plane(), 0u, positive(index));
    // Ages are `u32` but the buffer is `atomic<i32>`; ages beyond 2^31 ticks (~35 minutes at
    // microsecond resolution) are older than any window and are dropped by the host's span check
    // anyway, so clamping here loses nothing.
    let age = i32(min(ages[index], 0x7fffffffu));
    atomicMin(&cells[cell + offset], age);
}

/// Averaged time surface: the running sum of `exp(-age/tau)` per (pixel, polarity) in the first
/// two planes, and the event count in the next two, which the host divides.
@compute @workgroup_size(WORKGROUP)
fn averaged_time_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.n_events) { return; }
    let cell = cell_of(index);
    if (cell == 0xffffffffu) { return; }
    let offset = select(plane(), 0u, positive(index));
    accumulate(cell + offset, exp(-age_ms(index) / params.span_ms));
    atomicAdd(&cells[cell + offset + 2u * plane()], 1);
}
