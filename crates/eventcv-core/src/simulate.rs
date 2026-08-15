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

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

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

/// Ceiling on adaptive upsampling, so a hard cut between two frames cannot ask for thousands of
/// sub-steps and stall the run.
const MAX_UPSAMPLE: usize = 64;

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

/// A DVS pixel array, driven one frame at a time.
///
/// Frames are pushed in time order and each call returns the events generated since the previous
/// one, already sorted. Nothing accumulates across calls, so a recording of any length simulates in
/// memory proportional to one frame interval rather than to the whole output.
pub struct Simulator {
    config: SimulatorConfig,
    width: usize,
    height: usize,
    /// Memorised log intensity per pixel — the level each event is measured against.
    log_ref: Vec<f32>,
    /// Photoreceptor lowpass state per pixel.
    lowpass: Vec<f32>,
    /// Per-pixel thresholds, drawn once at construction: mismatch is a property of the silicon, not
    /// noise that resamples every frame.
    thres_pos: Vec<f32>,
    thres_neg: Vec<f32>,
    /// Last emission time per pixel, for the refractory check.
    last_t: Vec<i64>,
    /// Working buffer for the incoming frame's lin-log intensity, reused across calls.
    log_now: Vec<f32>,
    /// Luma in `[0, 1]` for the incoming frame, reused across calls.
    luma: Vec<f32>,
    previous: Option<(Vec<f32>, Vec<f32>, i64)>,
    rng: StdRng,
}

impl Simulator {
    pub fn new(width: usize, height: usize, config: SimulatorConfig) -> Self {
        let pixels = width * height;
        let mut rng = StdRng::seed_from_u64(config.seed);
        // Mismatch is sampled once and kept. Thresholds are clamped well above zero: a threshold at
        // or below zero would emit unboundedly on any change at all.
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
        Self {
            config,
            width,
            height,
            log_ref: vec![0.0; pixels],
            lowpass: vec![0.0; pixels],
            thres_pos,
            thres_neg,
            last_t: vec![i64::MIN / 4; pixels],
            log_now: vec![0.0; pixels],
            luma: vec![0.0; pixels],
            previous: None,
            rng,
        }
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

        let Some((previous_log, previous_luma, t_previous)) = self.previous.take() else {
            self.log_ref.copy_from_slice(&self.log_now);
            self.lowpass.copy_from_slice(&self.log_now);
            self.last_t.fill(t_us);
            self.previous = Some((self.log_now.clone(), self.luma.clone(), t_us));
            return self.empty_stream();
        };

        let span = (t_us - t_previous).max(0);
        let steps = self.upsample_steps(&previous_log, span);
        let mut events: Vec<(u16, u16, i64, bool)> = Vec::new();

        for step in 1..=steps {
            // Linear interpolation in log space between the two frames. Sub-intervals are walked in
            // order, so events stay grouped in ascending time even before the sort below.
            let alpha = step as f32 / steps as f32;
            let t_start = t_previous + (span as f64 * (step - 1) as f64 / steps as f64) as i64;
            let t_end = t_previous + (span as f64 * step as f64 / steps as f64) as i64;
            let dt_s = ((t_end - t_start) as f64 / 1e6) as f32;

            for index in 0..self.width * self.height {
                let target =
                    previous_log[index] + (self.log_now[index] - previous_log[index]) * alpha;
                let luma = previous_luma[index] + (self.luma[index] - previous_luma[index]) * alpha;
                self.integrate_pixel(index, target, luma, t_start, t_end, dt_s, &mut events);
            }
        }

        // Sorting per interval rather than globally is what keeps this streaming: intervals are
        // produced in time order, so concatenating sorted blocks is already globally sorted, and no
        // sort ever sees more than one interval's events.
        events.sort_unstable_by_key(|&(_, _, t, _)| t);

        self.previous = Some((self.log_now.clone(), self.luma.clone(), t_us));
        let mut builder =
            EventStreamBuilder::with_capacity(self.width, self.height, 0.001, events.len());
        for (x, y, t, polarity) in events {
            builder.push(x, y, t, polarity);
        }
        builder.build()
    }

    /// One pixel over one sub-interval: bandwidth, threshold crossings, leak and noise.
    #[allow(clippy::too_many_arguments)]
    fn integrate_pixel(
        &mut self,
        index: usize,
        target_log: f32,
        luma: f32,
        t_start: i64,
        t_end: i64,
        dt_s: f32,
        events: &mut Vec<(u16, u16, i64, bool)>,
    ) {
        // Photoreceptor lowpass: cutoff falls with intensity, floored so dark pixels still track.
        let filtered = if self.config.cutoff_hz > 0.0 && dt_s > 0.0 {
            let bandwidth = MIN_BANDWIDTH_FRACTION + (1.0 - MIN_BANDWIDTH_FRACTION) * luma;
            let tau = 1.0 / (2.0 * std::f32::consts::PI * self.config.cutoff_hz * bandwidth);
            let epsilon = (dt_s / tau).clamp(0.0, 1.0);
            self.lowpass[index] += epsilon * (target_log - self.lowpass[index]);
            self.lowpass[index]
        } else {
            self.lowpass[index] = target_log;
            target_log
        };

        // Leak: the memorised level decays, so a still scene still emits the occasional ON event.
        // Per-pixel thresholds decorrelate them, which is what stops leak looking like a metronome.
        if self.config.leak_rate_hz > 0.0 && dt_s > 0.0 {
            self.log_ref[index] -= self.thres_pos[index] * self.config.leak_rate_hz * dt_s;
        }

        let delta = filtered - self.log_ref[index];
        let positive = delta > 0.0;
        let threshold = if positive {
            self.thres_pos[index]
        } else {
            self.thres_neg[index]
        };
        let crossings = (delta.abs() / threshold).floor() as i64;

        if crossings > 0 {
            let (x, y) = ((index % self.width) as u16, (index / self.width) as u16);
            let span = (t_end - t_start) as f64;
            for k in 1..=crossings {
                // The k-th crossing happens when the interpolated level reaches k thresholds away,
                // which under the linear assumption is a fixed fraction of the way through.
                let fraction = (k as f32 * threshold / delta.abs()).clamp(0.0, 1.0) as f64;
                let t = t_start + (span * fraction) as i64;
                if t - self.last_t[index] < self.config.refractory_us {
                    continue;
                }
                self.last_t[index] = t;
                events.push((x, y, t.max(0), positive));
            }
            // Carry the remainder rather than discarding it, so slow ramps still fire eventually.
            let signed = if positive { 1.0 } else { -1.0 };
            self.log_ref[index] += signed * crossings as f32 * threshold;
        }

        // Shot noise: Poisson-thin, quieter where the scene is bright.
        if self.config.shot_noise_rate_hz > 0.0 && dt_s > 0.0 {
            let scale = 1.0 - (1.0 - SHOT_NOISE_BRIGHT_FACTOR) * luma;
            // `1 - exp(-λ)` is the Poisson probability of at least one event, not `λ` itself.
            // The distinction only shows up once λ approaches 1 — but there a raw `λ` exceeds 1,
            // the comparison below is always true, and the intensity dependence silently
            // disappears into saturation. At most one event per polarity per sub-interval is
            // emitted either way, so a rate high relative to the interval still saturates: that is
            // a reason to subdivide (see `Upsample`), not something to paper over here.
            let lambda = self.config.shot_noise_rate_hz * scale * dt_s / 2.0;
            let probability = 1.0 - (-lambda).exp();
            for polarity in [true, false] {
                if self.rng.gen::<f32>() < probability {
                    let (x, y) = ((index % self.width) as u16, (index / self.width) as u16);
                    let t = t_start + (self.rng.gen::<f64>() * (t_end - t_start) as f64) as i64;
                    if t - self.last_t[index] >= self.config.refractory_us {
                        self.last_t[index] = t;
                        events.push((x, y, t.max(0), polarity));
                    }
                }
            }
        }
    }

    /// How many sub-intervals this frame pair needs.
    fn upsample_steps(&self, previous_log: &[f32], span_us: i64) -> usize {
        if span_us <= 0 {
            return 1;
        }
        match self.config.upsample {
            Upsample::Off => 1,
            Upsample::Fixed(n) => n.clamp(1, MAX_UPSAMPLE),
            Upsample::Adaptive {
                max_events_per_pixel,
            } => {
                let budget = max_events_per_pixel.max(0.1);
                // The busiest pixel decides: if it would emit n events over the pair, subdivide
                // until no sub-interval asks for more than `budget` from it.
                let worst = previous_log
                    .iter()
                    .zip(&self.log_now)
                    .zip(&self.thres_pos)
                    .map(|((before, after), threshold)| (after - before).abs() / threshold)
                    .fold(0.0_f32, f32::max);
                ((worst / budget).ceil() as usize).clamp(1, MAX_UPSAMPLE)
            }
        }
    }

    fn empty_stream(&self) -> EventStream {
        EventStreamBuilder::new(self.width, self.height, 0.001).build()
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

/// Rec. 601 luma of an sRGB pixel, linearised, in `[0, 1]`.
///
/// Linearising first matters: contrast is a ratio of *light*, and sRGB stores a gamma-encoded
/// value. Measuring log contrast on the encoded value bakes the display transfer function into
/// every threshold.
pub fn linear_luma(r: u8, g: u8, b: u8) -> f32 {
    let component = |value: u8| {
        let v = f32::from(value) / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * component(r) + 0.7152 * component(g) + 0.0722 * component(b)
}

/// Converts a decoded RGB frame into the linear luma the simulator consumes.
pub fn luma_from_rgb(image: &Rgb8Image) -> Vec<f32> {
    image
        .pixels
        .chunks_exact(3)
        .map(|pixel| linear_luma(pixel[0], pixel[1], pixel[2]))
        .collect()
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
    mut on_events: impl FnMut(EventStream) -> std::io::Result<()>,
) -> std::io::Result<(usize, usize)> {
    let mut decoder = crate::video::FfmpegDecoder::open(path, scale)?;
    let info = decoder.info();
    let mut simulator = Simulator::new(info.width, info.height, config);
    // Frame index drives the clock rather than any wall time: the source's own frame rate is what
    // the timestamps have to be consistent with.
    let us_per_frame = (1_000_000.0 / info.fps.max(1e-6)).round() as i64;
    let (mut frames, mut events) = (0usize, 0usize);
    while let Some(image) = decoder.next_frame()? {
        if max_frames.is_some_and(|limit| frames >= limit) {
            break;
        }
        let luma = luma_from_rgb(&image);
        let stream = simulator.push_frame(&luma, frames as i64 * us_per_frame);
        frames += 1;
        events += stream.len();
        if !stream.is_empty() {
            on_events(stream)?;
        }
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
