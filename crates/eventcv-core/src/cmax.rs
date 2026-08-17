//! Contrast maximisation — recovering motion by making the warped events sharp.
//!
//! Events from a moving camera are smeared along the motion's path. Warp them by a candidate motion
//! and accumulate them into an *image of warped events* (IWE): guess right and the smear collapses
//! into sharp edges, guess wrong and it stays blurred. Scoring that sharpness and maximising it
//! recovers the motion. Gallego et al., *A Unifying Contrast Maximization Framework for Event
//! Cameras* (CVPR 2018) is the canonical treatment.
//!
//! ```no_run
//! # use eventcv_core::cmax::{CmaxConfig, WarpModel};
//! # fn demo(stream: &eventcv_core::EventStream) -> Result<(), Box<dyn std::error::Error>> {
//! let result = stream.contrast_maximise(WarpModel::translation(), CmaxConfig::default())?;
//! println!("{:?} px/s, {:.1}x sharper", result.params, result.improvement());
//! let iwe = stream.iwe(WarpModel::Translation, &result.params)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Deliberate differences from the reference implementation
//!
//! `event_utils` (Stoffregen) is the reference every CMax paper cites. Three things here differ from
//! it on purpose:
//!
//! - **Events are warped to the interval midpoint**, not to the last event's timestamp. Warping to
//!   the end makes every `dt` negative and gives the earliest events the largest displacement;
//!   the midpoint halves the worst-case displacement, and with it the linearisation error.
//! - **Out-of-bounds events are dropped**, not folded onto pixel (0, 0). The reference multiplies
//!   coordinates by a 0/1 mask, which piles every escaped event into the corner and rewards warps
//!   that push events off the sensor — a bias directly against what the objective is measuring.
//! - **The optimiser is derivative-free.** The reference uses BFGS but its own documentation
//!   recommends numeric gradients as "more stable… less prone to noise", which is an argument for
//!   not needing gradients at all.

use std::fmt;

use crate::camera::Camera;
use crate::representation::{EventFrame, EventFrameData};
use crate::EventStream;

/// Ceiling on the Gaussian blur radius, so a large sigma cannot make one evaluation unbounded.
const MAX_BLUR_RADIUS: usize = 32;

/// Largest exponent [`Objective::SumOfExponentials`] will evaluate. `exp(32)` is ~8e13, far below
/// `f64`'s range, and an IWE pixel holding 32 events after blurring is already implausibly dense.
/// Fixed rather than derived per image so it cannot change the ordering between two images.
const SOE_MAX_EXPONENT: f64 = 32.0;

/// Event count above which the warp is parallelised. Below this, allocating and reducing a
/// per-thread image costs more than the scatter it saves. Measured with `cargo bench`, not guessed.
const PARALLEL_EVENT_THRESHOLD: usize = 20_000;

/// How a candidate motion displaces an event, as a function of how far it is from the reference
/// time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WarpModel {
    /// Constant image-plane velocity, `(vx, vy)` in pixels per second. Two parameters.
    ///
    /// The right model for a camera translating parallel to a flat scene, or for one small patch of
    /// any scene — and the only model the reference implementation has working.
    Translation,
    /// Camera rotation, `(wx, wy, wz)` in radians per second, about the optical centre. Three
    /// parameters, and needs intrinsics to map pixels onto rays.
    ///
    /// Rotation is the case where contrast maximisation is at its best: the warp is exact for *any*
    /// scene regardless of depth, because rotating a camera moves every ray the same way.
    Rotation { camera: Camera },
}

impl WarpModel {
    /// The 2-DoF translation model.
    pub fn translation() -> Self {
        Self::Translation
    }

    /// Number of free parameters.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Translation => 2,
            Self::Rotation { .. } => 3,
        }
    }

    /// Displaces one event to where it would have been at the reference time.
    ///
    /// `dt` is seconds from the reference time — negative for events before it. Returns float
    /// coordinates; quantising here is what would make the objective staircased and the optimiser
    /// blind to small improvements.
    fn warp(&self, x: f64, y: f64, dt: f64, params: &[f64]) -> (f64, f64) {
        match self {
            Self::Translation => (x - dt * params[0], y - dt * params[1]),
            Self::Rotation { camera } => {
                // Project to a normalised ray, apply the small-angle rotation, project back. Exact
                // for small `omega * dt`, which is the regime a single slice covers.
                let nx = (x - camera.cx) / camera.fx;
                let ny = (y - camera.cy) / camera.fy;
                let (wx, wy, wz) = (params[0] * dt, params[1] * dt, params[2] * dt);
                // First-order rotation of the ray (1, nx, ny) about the optical centre.
                let rx = nx - wz * ny + wy;
                let ry = ny + wz * nx - wx;
                (rx * camera.fx + camera.cx, ry * camera.fy + camera.cy)
            }
        }
    }
}

/// What "sharp" means when scoring an image of warped events. Every objective is *maximised*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Objective {
    /// Variance of the IWE — Gallego & Scaramuzza, RA-L 2017. The standard choice, and the default.
    #[default]
    Variance,
    /// Mean of the squared IWE — Stoffregen & Kleeman, CVPR 2019. Cheaper than variance and
    /// behaves similarly; slightly more sensitive to the total event count.
    SumOfSquares,
    /// Mean of `exp(IWE)` — Stoffregen & Kleeman, CVPR 2019. Rewards concentration much more
    /// aggressively, which sharpens the optimum but narrows the basin around it.
    SumOfExponentials,
}

impl Objective {
    /// Blur applied to the IWE before scoring, following the reference's per-objective table.
    ///
    /// Blur is not cosmetic: without it the objective is a field of isolated spikes with no gradient
    /// between them, and the optimiser has nothing to follow.
    pub fn default_blur(self) -> f64 {
        match self {
            Self::Variance | Self::SumOfSquares => 1.0,
            Self::SumOfExponentials => 2.5,
        }
    }

    /// Scores an IWE. Higher is sharper.
    pub fn score(self, iwe: &[f32]) -> f64 {
        if iwe.is_empty() {
            return 0.0;
        }
        let n = iwe.len() as f64;
        match self {
            Self::Variance => {
                let mean = iwe.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
                iwe.iter()
                    .map(|&v| {
                        let d = f64::from(v) - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / n
            }
            Self::SumOfSquares => {
                iwe.iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
                    / n
            }
            Self::SumOfExponentials => {
                // `exp` is convex, so for a fixed total mass this is maximised by concentrating it —
                // which is exactly the property being optimised for.
                //
                // The exponent is clamped to a *fixed* constant rather than shifted by each image's
                // own maximum. A per-image shift is not a shared monotonic transform: it rescales
                // every image to its own peak, which makes a uniform field score higher than a
                // sharp one and inverts the objective entirely. A fixed clamp is identical across
                // every image being compared, so it leaves the ordering intact while keeping
                // `exp` well away from overflow.
                iwe.iter()
                    .map(|&v| f64::from(v).min(SOE_MAX_EXPONENT).exp())
                    .sum::<f64>()
                    / n
            }
        }
    }
}

/// Where events are warped to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimeReference {
    /// The middle of the slice. Halves the worst-case displacement compared with either end, which
    /// halves the error from assuming motion is linear over the interval.
    #[default]
    Midpoint,
    /// The first event's timestamp.
    Start,
    /// The last event's timestamp — what the reference implementation uses.
    End,
}

/// Optimiser and scoring settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CmaxConfig {
    pub objective: Objective,
    pub time_reference: TimeReference,
    /// Gaussian sigma applied to the IWE before scoring. `None` takes the objective's default.
    pub blur_sigma: Option<f64>,
    /// Starting guess, in the model's units. Zero means "assume static and search outwards".
    pub initial: Option<[f64; 3]>,
    /// Initial simplex size — how far the first probes reach from `initial`. Too small and the
    /// optimiser cannot escape a flat region; too large and it steps over the optimum.
    pub initial_step: f64,
    pub max_iterations: usize,
    /// Stop once the simplex spans less than this in parameter units.
    pub tolerance: f64,
}

impl Default for CmaxConfig {
    fn default() -> Self {
        Self {
            objective: Objective::default(),
            time_reference: TimeReference::default(),
            blur_sigma: None,
            initial: None,
            // Pixels per second. A slice of a few tens of ms with motion of a few pixels lands here.
            initial_step: 50.0,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }
}

/// What the optimiser found.
#[derive(Clone, Debug, PartialEq)]
pub struct CmaxResult {
    /// Recovered motion, in the model's units — px/s for translation, rad/s for rotation.
    pub params: Vec<f64>,
    /// Objective value at `params`.
    pub score: f64,
    /// Objective value at zero motion, for comparison. A result whose `score` barely exceeds this
    /// means the optimiser found nothing — there was no coherent motion, or the slice was too short.
    pub score_at_rest: f64,
    pub iterations: usize,
}

impl CmaxResult {
    /// How much sharper the recovered motion is than assuming the camera was still.
    ///
    /// The number to check before trusting `params`: at or below 1.0 the optimiser did not find
    /// motion, and the parameters are wherever the search happened to stop.
    pub fn improvement(&self) -> f64 {
        if self.score_at_rest.abs() < f64::EPSILON {
            return 1.0;
        }
        self.score / self.score_at_rest
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CmaxError {
    EmptyStream,
    SizeOverflow,
    InvalidParameter(&'static str),
}

impl fmt::Display for CmaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStream => {
                formatter.write_str("contrast maximisation needs a non-empty stream")
            }
            Self::SizeOverflow => formatter.write_str("image dimensions are too large"),
            Self::InvalidParameter(name) => {
                write!(formatter, "{name} must be finite and positive")
            }
        }
    }
}

impl std::error::Error for CmaxError {}

impl EventStream {
    /// Accumulates the image of warped events for an explicit motion.
    ///
    /// The picture the objective actually scores, returned so it can be looked at — a blurred IWE
    /// with the "right" parameters is the clearest sign that a warp model does not fit the scene.
    pub fn iwe(&self, model: WarpModel, params: &[f64]) -> Result<EventFrame, CmaxError> {
        if params.len() < model.dimensions() {
            return Err(CmaxError::InvalidParameter("params"));
        }
        let (width, height) = self.sensor_size();
        let plane = width.checked_mul(height).ok_or(CmaxError::SizeOverflow)?;
        let mut image = vec![0.0_f32; plane];
        self.accumulate_warped(&model, params, TimeReference::Midpoint, &mut image);
        EventFrame::intensity(EventFrameData::F32(image), width, height)
            .map_err(|_| CmaxError::SizeOverflow)
    }

    /// Finds the motion that makes the warped events sharpest.
    ///
    /// Returns the recovered parameters along with the score at rest, so the caller can tell a real
    /// estimate from the optimiser wandering on a flat landscape — see [`CmaxResult::improvement`].
    pub fn contrast_maximise(
        &self,
        model: WarpModel,
        config: CmaxConfig,
    ) -> Result<CmaxResult, CmaxError> {
        if self.is_empty() {
            return Err(CmaxError::EmptyStream);
        }
        let sigma = config.blur_sigma.unwrap_or(config.objective.default_blur());
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(CmaxError::InvalidParameter("blur_sigma"));
        }
        if !(config.initial_step.is_finite() && config.initial_step > 0.0) {
            return Err(CmaxError::InvalidParameter("initial_step"));
        }
        let (width, height) = self.sensor_size();
        let plane = width.checked_mul(height).ok_or(CmaxError::SizeOverflow)?;
        let dims = model.dimensions();

        // One scratch buffer, reused for every evaluation — the optimiser runs hundreds of them and
        // reallocating a sensor-sized image each time would dominate the cost.
        let mut image = vec![0.0_f32; plane];
        let mut scratch = vec![0.0_f32; plane];
        let mut evaluate = |params: &[f64]| -> f64 {
            self.accumulate_warped(&model, params, config.time_reference, &mut image);
            if sigma > 0.0 {
                blur_in_place(&mut image, width, height, sigma, &mut scratch);
            }
            config.objective.score(&image)
        };

        let zero = vec![0.0; dims];
        let score_at_rest = evaluate(&zero);
        let start = match config.initial {
            Some(initial) => initial[..dims].to_vec(),
            None => zero,
        };
        let (params, score, iterations) = nelder_mead(
            &mut evaluate,
            start,
            config.initial_step,
            config.max_iterations,
            config.tolerance,
        );
        Ok(CmaxResult {
            params,
            score,
            score_at_rest,
            iterations,
        })
    }

    /// Warps every event and bilinearly scatters it into `image` (which is cleared first).
    fn accumulate_warped(
        &self,
        model: &WarpModel,
        params: &[f64],
        reference: TimeReference,
        image: &mut [f32],
    ) {
        image.fill(0.0);
        let (width, height) = self.sensor_size();
        let ts = self.ts();
        let (Some(&first), Some(&last)) = (ts.iter().min(), ts.iter().max()) else {
            return;
        };
        let reference_t = match reference {
            TimeReference::Midpoint => (first + last) / 2,
            TimeReference::Start => first,
            TimeReference::End => last,
        };
        // Timestamps are in units of `timestamp_scale_ms` milliseconds; the warp works in seconds
        // so its parameters read as px/s and rad/s rather than per-tick.
        let seconds_per_tick = self.timestamp_scale_ms() / 1000.0;
        let (xs, ys) = (self.xs(), self.ys());

        let warp_range = |range: std::ops::Range<usize>, target: &mut [f32]| {
            for index in range {
                let dt = (ts[index] - reference_t) as f64 * seconds_per_tick;
                let (wx, wy) = model.warp(f64::from(xs[index]), f64::from(ys[index]), dt, params);
                splat(target, width, height, wx, wy);
            }
        };

        // Scattering cannot be parallelised in place — several threads would write the same pixel —
        // so each chunk accumulates into its own image and the images are summed. That costs one
        // sensor-sized buffer per thread, which only pays off once there are enough events to
        // outweigh allocating and reducing them. Below the threshold the serial path is faster.
        if self.len() < PARALLEL_EVENT_THRESHOLD {
            warp_range(0..self.len(), image);
            return;
        }

        use rayon::prelude::*;
        let plane = image.len();
        let chunk = (self.len() / rayon::current_num_threads().max(1)).max(1);
        let partial = (0..self.len())
            .into_par_iter()
            .step_by(chunk)
            .map(|start| {
                let end = (start + chunk).min(self.len());
                let mut local = vec![0.0_f32; plane];
                warp_range(start..end, &mut local);
                local
            })
            .reduce(
                || vec![0.0_f32; plane],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(&b) {
                        *x += y;
                    }
                    a
                },
            );
        image.copy_from_slice(&partial);
    }
}

/// Adds one unit of mass at a float position, split across the four surrounding pixels.
///
/// Bilinear rather than nearest-neighbour because the objective has to respond to sub-pixel changes
/// in the warp; rounding to the nearest pixel makes the landscape a staircase with flat treads, and
/// a derivative-free optimiser stalls on the first one it lands on.
///
/// Events landing outside the sensor are dropped. The reference implementation instead folds them
/// onto pixel (0, 0), which rewards warps that push events off the image — the opposite of what the
/// objective is meant to measure.
fn splat(image: &mut [f32], width: usize, height: usize, x: f64, y: f64) {
    if !(x.is_finite() && y.is_finite()) {
        return;
    }
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = (x - x0) as f32;
    let fy = (y - y0) as f32;
    let x0 = x0 as i64;
    let y0 = y0 as i64;

    for (dx, dy, weight) in [
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let (px, py) = (x0 + dx, y0 + dy);
        if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
            image[py as usize * width + px as usize] += weight;
        }
    }
}

/// Separable Gaussian blur in place. `scratch` must be the same length as `image`.
fn blur_in_place(image: &mut [f32], width: usize, height: usize, sigma: f64, scratch: &mut [f32]) {
    let radius = ((sigma * 3.0).ceil() as usize).clamp(1, MAX_BLUR_RADIUS);
    let kernel: Vec<f32> = (0..=2 * radius)
        .map(|i| {
            let d = i as f64 - radius as f64;
            (-(d * d) / (2.0 * sigma * sigma)).exp() as f32
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    let kernel: Vec<f32> = kernel.iter().map(|k| k / sum).collect();

    // Horizontal into scratch, then vertical back into image. Edges clamp, which keeps the total
    // mass near constant rather than darkening the border.
    for y in 0..height {
        for x in 0..width {
            let mut total = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sx = (x as i64 + k as i64 - radius as i64).clamp(0, width as i64 - 1) as usize;
                total += image[y * width + sx] * weight;
            }
            scratch[y * width + x] = total;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut total = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sy = (y as i64 + k as i64 - radius as i64).clamp(0, height as i64 - 1) as usize;
                total += scratch[sy * width + x] * weight;
            }
            image[y * width + x] = total;
        }
    }
}

/// Nelder-Mead simplex search, maximising `evaluate`.
///
/// Derivative-free on purpose: the objective is a noisy function of a discrete event set, and the
/// reference implementation's own documentation prefers numeric gradients over analytic ones for
/// exactly that reason. If gradients are not trustworthy, a method that never asks for them is
/// simpler and no worse.
fn nelder_mead(
    evaluate: &mut impl FnMut(&[f64]) -> f64,
    start: Vec<f64>,
    step: f64,
    max_iterations: usize,
    tolerance: f64,
) -> (Vec<f64>, f64, usize) {
    let dims = start.len();
    if dims == 0 {
        return (start, 0.0, 0);
    }
    // The simplex: the start point plus one offset vertex per dimension.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(dims + 1);
    simplex.push(start.clone());
    for axis in 0..dims {
        let mut vertex = start.clone();
        vertex[axis] += step;
        simplex.push(vertex);
    }
    let mut scores: Vec<f64> = simplex.iter().map(|v| evaluate(v)).collect();

    let mut iterations = 0;
    while iterations < max_iterations {
        iterations += 1;
        // Sort best (highest score) first.
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        scores = order.iter().map(|&i| scores[i]).collect();

        // Converged once every vertex sits within `tolerance` of the best one.
        let spread = simplex[1..]
            .iter()
            .map(|v| {
                v.iter()
                    .zip(&simplex[0])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            })
            .fold(0.0_f64, f64::max);
        if spread < tolerance {
            break;
        }

        // Centroid of everything but the worst vertex.
        let worst = simplex.len() - 1;
        let mut centroid = vec![0.0; dims];
        for vertex in &simplex[..worst] {
            for (c, v) in centroid.iter_mut().zip(vertex) {
                *c += v / worst as f64;
            }
        }
        let combine = |a: &[f64], b: &[f64], t: f64| -> Vec<f64> {
            a.iter().zip(b).map(|(x, y)| x + t * (x - y)).collect()
        };

        let reflected = combine(&centroid, &simplex[worst], 1.0);
        let reflected_score = evaluate(&reflected);
        if reflected_score > scores[0] {
            // Better than the best: try stretching further in the same direction.
            let expanded = combine(&centroid, &simplex[worst], 2.0);
            let expanded_score = evaluate(&expanded);
            let (vertex, score) = if expanded_score > reflected_score {
                (expanded, expanded_score)
            } else {
                (reflected, reflected_score)
            };
            simplex[worst] = vertex;
            scores[worst] = score;
        } else if reflected_score > scores[worst - 1] {
            simplex[worst] = reflected;
            scores[worst] = reflected_score;
        } else {
            // Reflection did not help: pull the worst vertex toward the centroid instead.
            let contracted = combine(&centroid, &simplex[worst], -0.5);
            let contracted_score = evaluate(&contracted);
            if contracted_score > scores[worst] {
                simplex[worst] = contracted;
                scores[worst] = contracted_score;
            } else {
                // Nothing worked — shrink the whole simplex toward the best vertex.
                for index in 1..simplex.len() {
                    let best = simplex[0].clone();
                    for (v, b) in simplex[index].iter_mut().zip(&best) {
                        *v = b + 0.5 * (*v - b);
                    }
                    scores[index] = evaluate(&simplex[index]);
                }
            }
        }
    }

    let best = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    (simplex[best].clone(), scores[best], iterations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::RepresentationKind;
    use crate::EventStreamBuilder;

    /// A moving point: one event per step, travelling at a known velocity in px/s.
    fn moving_point(vx: f64, vy: f64, steps: usize, duration_us: i64) -> EventStream {
        let mut builder = EventStreamBuilder::new(64, 64, 0.001);
        for step in 0..steps {
            let t = step as i64 * duration_us / steps as i64;
            let seconds = t as f64 / 1e6;
            let x = 32.0 + vx * seconds;
            let y = 32.0 + vy * seconds;
            builder.push(x.round() as u16, y.round() as u16, t, true);
        }
        builder.build()
    }

    /// A moving edge: a column of events sweeping across, which is what a real scene looks like.
    fn moving_edge(vx: f64, steps: usize, duration_us: i64) -> EventStream {
        let mut builder = EventStreamBuilder::new(64, 64, 0.001);
        for step in 0..steps {
            let t = step as i64 * duration_us / steps as i64;
            let x = 16.0 + vx * (t as f64 / 1e6);
            for y in 8..56u16 {
                builder.push(x.round() as u16, y, t, true);
            }
        }
        builder.build()
    }

    #[test]
    fn splat_conserves_mass_and_splits_by_weight() {
        let mut image = vec![0.0_f32; 16];
        // Exactly between four pixels: a quarter each.
        splat(&mut image, 4, 4, 1.5, 1.5);
        for (index, expected) in [(5, 0.25), (6, 0.25), (9, 0.25), (10, 0.25)] {
            assert!((image[index] - expected).abs() < 1e-6, "pixel {index}");
        }
        assert!((image.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Exactly on a pixel: all of it there.
        let mut exact = vec![0.0_f32; 16];
        splat(&mut exact, 4, 4, 2.0, 1.0);
        assert!((exact[4 + 2] - 1.0).abs() < 1e-6); // row 1, column 2
        assert!((exact.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // A quarter of the way across splits 3:1.
        let mut fractional = vec![0.0_f32; 16];
        splat(&mut fractional, 4, 4, 1.25, 1.0);
        assert!((fractional[5] - 0.75).abs() < 1e-6);
        assert!((fractional[6] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn splat_drops_out_of_bounds_rather_than_folding_to_the_corner() {
        // The reference implementation's bug: escaped events pile up at (0, 0) and reward warps
        // that push events off the sensor.
        let mut image = vec![0.0_f32; 16];
        splat(&mut image, 4, 4, -50.0, -50.0);
        splat(&mut image, 4, 4, 500.0, 500.0);
        assert_eq!(image.iter().sum::<f32>(), 0.0);
        assert_eq!(image[0], 0.0, "nothing may accumulate at the origin");

        // A partly-outside event contributes only its inside weight.
        let mut edge = vec![0.0_f32; 16];
        splat(&mut edge, 4, 4, -0.5, 1.0);
        assert!(edge.iter().sum::<f32>() < 1.0);
    }

    #[test]
    fn splat_ignores_non_finite_coordinates() {
        let mut image = vec![0.0_f32; 16];
        splat(&mut image, 4, 4, f64::NAN, 1.0);
        splat(&mut image, 4, 4, 1.0, f64::INFINITY);
        assert_eq!(image.iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn blur_preserves_total_mass() {
        let mut image = vec![0.0_f32; 64 * 64];
        image[32 * 64 + 32] = 1.0;
        let mut scratch = vec![0.0_f32; 64 * 64];
        blur_in_place(&mut image, 64, 64, 1.5, &mut scratch);
        assert!((image.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        // And it actually spread.
        assert!(image[32 * 64 + 32] < 1.0);
        assert!(image[32 * 64 + 33] > 0.0);
    }

    #[test]
    fn objectives_prefer_a_concentrated_image() {
        // The minimum property: a sharp image must score above a smeared one with the same mass.
        let mut sharp = vec![0.0_f32; 100];
        sharp[50] = 10.0;
        let spread = vec![0.1_f32; 100];
        for objective in [
            Objective::Variance,
            Objective::SumOfSquares,
            Objective::SumOfExponentials,
        ] {
            assert!(
                objective.score(&sharp) > objective.score(&spread),
                "{objective:?} failed to prefer the concentrated image"
            );
        }
    }

    #[test]
    fn a_warp_at_the_true_velocity_beats_zero_and_a_wrong_guess() {
        let (vx, vy) = (300.0, 0.0);
        let stream = moving_point(vx, vy, 40, 40_000);
        let (width, height) = stream.sensor_size();
        let mut image = vec![0.0_f32; width * height];

        let score_for = |params: &[f64], image: &mut Vec<f32>| {
            stream.accumulate_warped(
                &WarpModel::Translation,
                params,
                TimeReference::Midpoint,
                image,
            );
            Objective::Variance.score(image)
        };
        let truth = score_for(&[vx, vy], &mut image);
        let rest = score_for(&[0.0, 0.0], &mut image);
        let wrong = score_for(&[-vx, 200.0], &mut image);
        assert!(truth > rest, "true warp {truth} must beat rest {rest}");
        assert!(
            truth > wrong,
            "true warp {truth} must beat a wrong one {wrong}"
        );
    }

    #[test]
    fn recovers_a_known_translation() {
        // The test the reference implementation does not have: assert the motion comes back.
        let (vx, vy) = (250.0, -150.0);
        let stream = moving_edge(vx, 30, 40_000);
        let result = stream
            .contrast_maximise(WarpModel::Translation, CmaxConfig::default())
            .expect("optimisation should succeed");
        assert!(
            (result.params[0] - vx).abs() < 60.0,
            "recovered vx {}, expected {vx}",
            result.params[0]
        );
        assert!(
            result.improvement() > 1.0,
            "should beat the static hypothesis"
        );
        assert!(result.iterations > 0);
        // vy is unconstrained for a vertical edge sweeping horizontally — the aperture problem —
        // so it is deliberately not asserted here. `recovers_translation_in_both_axes` covers it
        // with a stimulus that constrains both.
        let _ = vy;
    }

    #[test]
    fn recovers_translation_in_both_axes() {
        // A point track constrains both components, unlike an edge.
        let (vx, vy) = (200.0, 160.0);
        let stream = moving_point(vx, vy, 40, 50_000);
        let result = stream
            .contrast_maximise(WarpModel::Translation, CmaxConfig::default())
            .unwrap();
        assert!(
            (result.params[0] - vx).abs() < 80.0,
            "vx {}",
            result.params[0]
        );
        assert!(
            (result.params[1] - vy).abs() < 80.0,
            "vy {}",
            result.params[1]
        );
    }

    #[test]
    fn a_static_scene_recovers_no_motion() {
        let mut builder = EventStreamBuilder::new(64, 64, 0.001);
        for step in 0..40i64 {
            builder.push(20, 20, step * 1000, true);
            builder.push(40, 40, step * 1000, false);
        }
        let result = builder
            .build()
            .contrast_maximise(WarpModel::Translation, CmaxConfig::default())
            .unwrap();
        // Already sharp, so warping can only smear it: the optimum stays near zero.
        assert!(result.params[0].abs() < 60.0, "vx {}", result.params[0]);
        assert!(result.params[1].abs() < 60.0, "vy {}", result.params[1]);
    }

    #[test]
    fn every_objective_recovers_the_same_motion() {
        let vx = 250.0;
        let stream = moving_edge(vx, 30, 40_000);
        for objective in [
            Objective::Variance,
            Objective::SumOfSquares,
            Objective::SumOfExponentials,
        ] {
            let result = stream
                .contrast_maximise(
                    WarpModel::Translation,
                    CmaxConfig {
                        objective,
                        ..CmaxConfig::default()
                    },
                )
                .unwrap();
            assert!(
                (result.params[0] - vx).abs() < 100.0,
                "{objective:?} recovered {}",
                result.params[0]
            );
        }
    }

    #[test]
    fn the_time_reference_does_not_change_the_recovered_motion() {
        // Where events are warped to changes the IWE's position, not the velocity that sharpens it.
        let vx = 250.0;
        let stream = moving_edge(vx, 30, 40_000);
        for reference in [
            TimeReference::Midpoint,
            TimeReference::Start,
            TimeReference::End,
        ] {
            let result = stream
                .contrast_maximise(
                    WarpModel::Translation,
                    CmaxConfig {
                        time_reference: reference,
                        ..CmaxConfig::default()
                    },
                )
                .unwrap();
            assert!(
                (result.params[0] - vx).abs() < 100.0,
                "{reference:?} recovered {}",
                result.params[0]
            );
        }
    }

    #[test]
    fn the_iwe_is_inspectable_and_sharper_at_the_truth() {
        let vx = 300.0;
        let stream = moving_edge(vx, 30, 40_000);
        let sharp = stream.iwe(WarpModel::Translation, &[vx, 0.0]).unwrap();
        let smeared = stream.iwe(WarpModel::Translation, &[0.0, 0.0]).unwrap();
        assert_eq!(sharp.shape(), (1, 64, 64));
        assert_eq!(sharp.kind(), RepresentationKind::Intensity);

        let extent = |frame: &EventFrame| match frame.data() {
            EventFrameData::F32(values) => values.iter().filter(|&&v| v > 0.01).count(),
            _ => unreachable!("iwe is always f32"),
        };
        // Sharper means the same events occupy fewer pixels.
        assert!(
            extent(&sharp) < extent(&smeared),
            "warped {} vs unwarped {} lit pixels",
            extent(&sharp),
            extent(&smeared)
        );
    }

    #[test]
    fn an_empty_stream_is_rejected() {
        let empty = EventStreamBuilder::new(8, 8, 0.001).build();
        assert_eq!(
            empty.contrast_maximise(WarpModel::Translation, CmaxConfig::default()),
            Err(CmaxError::EmptyStream)
        );
    }

    #[test]
    fn bad_parameters_are_rejected() {
        let stream = moving_point(100.0, 0.0, 10, 10_000);
        for config in [
            CmaxConfig {
                blur_sigma: Some(f64::NAN),
                ..CmaxConfig::default()
            },
            CmaxConfig {
                initial_step: 0.0,
                ..CmaxConfig::default()
            },
        ] {
            assert!(stream
                .contrast_maximise(WarpModel::Translation, config)
                .is_err());
        }
        assert!(stream.iwe(WarpModel::Translation, &[1.0]).is_err());
    }

    #[test]
    fn rotation_has_three_parameters_and_runs() {
        let camera = Camera::new(100.0, 100.0, 32.0, 32.0);
        let model = WarpModel::Rotation { camera };
        assert_eq!(model.dimensions(), 3);
        let stream = moving_edge(200.0, 20, 30_000);
        let result = stream
            .contrast_maximise(
                model,
                CmaxConfig {
                    initial_step: 1.0, // rad/s, not px/s
                    ..CmaxConfig::default()
                },
            )
            .unwrap();
        assert_eq!(result.params.len(), 3);
        assert!(result.score.is_finite());
    }

    #[test]
    fn recovers_the_motion_the_simulator_was_given() {
        // The end-to-end claim: render a pattern moving at a known speed, simulate the events a DVS
        // would produce, and recover that speed from the events alone. Nothing in the chain is told
        // the answer — this is the validation a real recording cannot provide, because a recording
        // has no ground-truth motion attached.
        use crate::simulate::{Simulator, SimulatorConfig, Upsample};

        let (width, height) = (96usize, 64usize);
        let pixels_per_second = 200.0_f64;
        let fps = 500.0_f64;
        let frames = 24;

        let mut simulator = Simulator::new(
            width,
            height,
            SimulatorConfig {
                // A clean sensor: this test is about the geometry, not the noise model.
                sigma_thres: 0.0,
                leak_rate_hz: 0.0,
                shot_noise_rate_hz: 0.0,
                cutoff_hz: 0.0,
                refractory_us: 0,
                upsample: Upsample::Off,
                ..SimulatorConfig::default()
            },
        );

        let mut events = Vec::new();
        for frame in 0..frames {
            let seconds = frame as f64 / fps;
            // A vertical bar sweeping right at a known rate.
            let bar = 12.0 + pixels_per_second * seconds;
            let mut luma = vec![0.05_f32; width * height];
            for y in 0..height {
                for offset in 0..4 {
                    let x = (bar as usize).saturating_add(offset);
                    if x < width {
                        luma[y * width + x] = 0.9;
                    }
                }
            }
            let slice = simulator.push_frame(&luma, (seconds * 1e6) as i64);
            events.push(slice);
        }

        let stream = match events.split_first() {
            Some((first, rest)) => first.concat(&rest.iter().collect::<Vec<_>>()),
            None => unreachable!("frames were pushed"),
        };
        assert!(
            stream.len() > 100,
            "simulator produced {} events",
            stream.len()
        );

        let result = stream
            .contrast_maximise(WarpModel::Translation, CmaxConfig::default())
            .expect("optimisation should succeed");

        let recovered = result.params[0];
        assert!(
            (recovered - pixels_per_second).abs() < pixels_per_second * 0.35,
            "recovered {recovered:.1} px/s from simulated events, expected {pixels_per_second:.1}"
        );
        assert!(
            result.improvement() > 1.0,
            "the recovered motion must beat the static hypothesis"
        );
    }

    #[test]
    fn nelder_mead_finds_a_known_maximum() {
        // The optimiser in isolation: a smooth quadratic with a maximum at (3, -2).
        let mut evaluate = |p: &[f64]| -(p[0] - 3.0).powi(2) - (p[1] + 2.0).powi(2);
        let (params, score, iterations) =
            nelder_mead(&mut evaluate, vec![0.0, 0.0], 1.0, 500, 1e-6);
        assert!((params[0] - 3.0).abs() < 1e-3, "x {}", params[0]);
        assert!((params[1] + 2.0).abs() < 1e-3, "y {}", params[1]);
        assert!(score > -1e-5);
        assert!(iterations < 500, "should converge before the cap");
    }
}
