//! **FEAST** — Feature Extraction using Adaptive Selection Thresholds (Afshar et al.,
//! *Sensors* 2020, [arXiv:1907.07853](https://arxiv.org/abs/1907.07853)). An unsupervised,
//! online, event-by-event feature learner: the event-domain analogue of a learned descriptor /
//! bag-of-visual-words extractor. Unlike the stateless corner detectors in [`crate::features`],
//! a [`Feast`] carries a *trained model* — `N` feature prototypes (`w×w` weight patches on the
//! unit hypersphere) each with an adaptive selection threshold — that persists across calls.
//!
//! For every event it maintains a per-polarity Surface of Active Events (SAE, the latest
//! timestamp per pixel — the same machinery [`crate::features::EventStream::efast`] uses), reads
//! the exponentially-decayed `w×w` time-surface patch around the pixel, and L2-normalises it into
//! a descriptor `d`. During [`Feast::fit`] the closest feature *within its threshold* wins:
//! its weights move toward `d` and its threshold **contracts** (`−ΔI`); a **miss** (no feature in
//! range) **expands** every threshold in that population (`+ΔE`). This competition drives all
//! features to roughly equal firing rates. During [`Feast::transform`] the thresholds are
//! discarded and each event is simply assigned its nearest feature.
//!
//! Streams are assumed to arrive in **ascending time order** (what the readers produce); call
//! [`crate::EventStream::sort_by_time`] first otherwise.

use std::{error::Error, fmt};

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::EventStream;

/// Parameters for a [`Feast`] model. Defaults follow the paper (`patch = 11`, `eta = 0.001`,
/// `ΔI = 0.001`, `ΔE = 0.003`); `tau_ms` defaults to 30 ms to match [`crate::representation::
/// TimeSurface`] and is the exponential decay constant of the time-surface patch.
#[derive(Clone, Copy, Debug)]
pub struct FeastConfig {
    /// Number of feature prototypes **per polarity population** (`b` in the paper).
    pub n_features: usize,
    /// Side length `w` of the square ROI patch; must be odd so it centres on the event.
    pub patch: usize,
    /// Time-surface decay constant, milliseconds.
    pub tau_ms: f64,
    /// Weight mixing rate `η` in `w ← (1−η)w + η·d`.
    pub eta: f32,
    /// Threshold contraction on a match (`ΔI`).
    pub delta_i: f32,
    /// Threshold expansion on a miss (`ΔE`).
    pub delta_e: f32,
    /// When `true`, ON and OFF events train independent feature populations (as in the paper's
    /// N-MNIST setup); when `false`, a single population learns from both polarities on one merged
    /// surface. For the paper's ON-only mode, pass an ON-only stream with `per_polarity = false`.
    pub per_polarity: bool,
    /// Seed for the random on-hypersphere feature initialisation — fixes reproducibility.
    pub seed: u64,
}

impl Default for FeastConfig {
    fn default() -> Self {
        Self {
            n_features: 100,
            patch: 11,
            tau_ms: 30.0,
            eta: 0.001,
            delta_i: 0.001,
            delta_e: 0.003,
            per_polarity: true,
            seed: 0,
        }
    }
}

/// A FEAST feature-extractor model. Build with [`Feast::new`], adapt online with [`Feast::fit`]
/// (repeatable across recordings/epochs), then map events to feature ids with [`Feast::transform`].
#[derive(Clone, Debug)]
pub struct Feast {
    config: FeastConfig,
    /// 1 (merged) or 2 (per-polarity) independent feature populations.
    banks: usize,
    /// `patch * patch` — descriptor/feature dimensionality.
    dim: usize,
    /// Row-major `[bank][feature][dim]` unit-norm feature weights.
    weights: Vec<f32>,
    /// Row-major `[bank][feature]` selection thresholds (cosine-distance acceptance radii).
    thresholds: Vec<f32>,
    /// Miss rate of the most recent [`Feast::fit`] call — a convergence proxy.
    last_missed_rate: f64,
}

impl Feast {
    /// Builds a model with random unit-hypersphere feature weights and random thresholds in
    /// `[0, 1)`, seeded by `config.seed`. Fails if any parameter is out of range.
    pub fn new(config: FeastConfig) -> Result<Self, FeastError> {
        validate(&config)?;
        let banks = if config.per_polarity { 2 } else { 1 };
        let dim = config.patch * config.patch;
        let total = banks * config.n_features;

        let mut rng = StdRng::seed_from_u64(config.seed);
        let mut weights = vec![0.0_f32; total * dim];
        for feature in weights.chunks_mut(dim) {
            let mut norm = 0.0_f32;
            for value in feature.iter_mut() {
                let sample: f32 = rng.sample(StandardNormal);
                *value = sample;
                norm += sample * sample;
            }
            let inv = 1.0 / norm.sqrt();
            feature.iter_mut().for_each(|value| *value *= inv);
        }
        let thresholds = (0..total).map(|_| rng.gen::<f32>()).collect();

        Ok(Self {
            config,
            banks,
            dim,
            weights,
            thresholds,
            last_missed_rate: 0.0,
        })
    }

    /// Rehydrates a model from saved state (used by the Python `save`/`load` round-trip). `weights`
    /// is the flat `[bank][feature][dim]` array and `thresholds` the flat `[bank][feature]` array;
    /// both must match `config`'s implied sizes.
    pub fn from_state(
        config: FeastConfig,
        weights: Vec<f32>,
        thresholds: Vec<f32>,
    ) -> Result<Self, FeastError> {
        validate(&config)?;
        let banks = if config.per_polarity { 2 } else { 1 };
        let dim = config.patch * config.patch;
        let total = banks * config.n_features;
        if weights.len() != total * dim || thresholds.len() != total {
            return Err(FeastError::StateShapeMismatch);
        }
        Ok(Self {
            config,
            banks,
            dim,
            weights,
            thresholds,
            last_missed_rate: 0.0,
        })
    }

    /// Trains the model on `stream` for `epochs` passes (each a fresh time surface, weights and
    /// thresholds carried over), adapting weights and thresholds online. Returns the **miss rate**
    /// of the final epoch — the fraction of in-bounds events that matched no feature, which settles
    /// to a low steady state (a few percent in the paper) as the network converges.
    pub fn fit(&mut self, stream: &EventStream, epochs: usize) -> f64 {
        let (width, height) = stream.sensor_size();
        let scale = stream.timestamp_scale_ms();
        let plane = width * height;
        let n = self.config.n_features;
        let (eta, delta_i, delta_e) = (self.config.eta, self.config.delta_i, self.config.delta_e);

        let mut descriptor = vec![0.0_f32; self.dim];
        for _ in 0..epochs.max(1) {
            let mut sae = vec![i64::MIN; self.banks * plane];
            let (mut seen, mut missed) = (0_u64, 0_u64);

            for event in stream.iter() {
                let bank = self.bank_of(event.polarity);
                let surface = &mut sae[bank * plane..(bank + 1) * plane];
                surface[event.y * width + event.x] = event.timestamp as i64;

                if !self.fill_descriptor(surface, &event, width, height, scale, &mut descriptor) {
                    continue;
                }
                seen += 1;

                match self.best_within_threshold(bank, &descriptor) {
                    Some(feature) => {
                        let base = (bank * n + feature) * self.dim;
                        let weights = &mut self.weights[base..base + self.dim];
                        let mut norm = 0.0_f32;
                        for (weight, &input) in weights.iter_mut().zip(&descriptor) {
                            *weight = (1.0 - eta) * *weight + eta * input;
                            norm += *weight * *weight;
                        }
                        let inv = 1.0 / norm.sqrt();
                        weights.iter_mut().for_each(|weight| *weight *= inv);

                        let threshold = &mut self.thresholds[bank * n + feature];
                        *threshold = (*threshold - delta_i).max(0.0);
                    }
                    None => {
                        missed += 1;
                        for threshold in &mut self.thresholds[bank * n..(bank + 1) * n] {
                            *threshold += delta_e;
                        }
                    }
                }
            }
            self.last_missed_rate = if seen > 0 {
                missed as f64 / seen as f64
            } else {
                0.0
            };
        }
        self.last_missed_rate
    }

    /// Maps each event to the id of its **nearest** feature (smallest cosine distance), ignoring
    /// thresholds — the paper's inference rule. Returns one id per input event, aligned to the
    /// stream; events too close to the border to extract a patch get `-1`. Ids are global:
    /// population `b`, feature `f` → `b * n_features + f`, so per-polarity OFF ids start at
    /// `n_features`. Does not modify the model.
    pub fn transform(&self, stream: &EventStream) -> Vec<i32> {
        let (width, height) = stream.sensor_size();
        let scale = stream.timestamp_scale_ms();
        let plane = width * height;

        let mut ids = vec![-1_i32; stream.len()];
        let mut descriptor = vec![0.0_f32; self.dim];
        let mut sae = vec![i64::MIN; self.banks * plane];

        for (index, event) in stream.iter().enumerate() {
            let bank = self.bank_of(event.polarity);
            let surface = &mut sae[bank * plane..(bank + 1) * plane];
            surface[event.y * width + event.x] = event.timestamp as i64;

            if self.fill_descriptor(surface, &event, width, height, scale, &mut descriptor) {
                let feature = self.nearest(bank, &descriptor);
                ids[index] = (bank * self.config.n_features + feature) as i32;
            }
        }
        ids
    }

    /// Pooled feature-event counts over `stream` (the classifier input in the paper): a histogram
    /// of length `n_features_total` counting how many events each feature won under
    /// [`Self::transform`].
    pub fn histogram(&self, stream: &EventStream) -> Vec<u32> {
        let mut counts = vec![0_u32; self.n_features_total()];
        for id in self.transform(stream) {
            if id >= 0 {
                counts[id as usize] += 1;
            }
        }
        counts
    }

    /// The learned feature weights, flat `[bank][feature][dim]` (reshape to
    /// `(n_features_total, patch, patch)` for the per-feature patch images).
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// The current selection thresholds, flat `[bank][feature]`.
    pub fn thresholds(&self) -> &[f32] {
        &self.thresholds
    }

    /// Total feature count across populations (`banks * n_features`).
    pub fn n_features_total(&self) -> usize {
        self.banks * self.config.n_features
    }

    pub fn config(&self) -> &FeastConfig {
        &self.config
    }

    /// Miss rate recorded by the most recent [`Self::fit`] (0 before any training).
    pub fn missed_rate(&self) -> f64 {
        self.last_missed_rate
    }

    /// Which feature population an event of the given polarity trains: OFF → 1 only when
    /// `per_polarity` is set, else the single merged population 0.
    fn bank_of(&self, polarity: bool) -> usize {
        if self.config.per_polarity && !polarity {
            1
        } else {
            0
        }
    }

    /// Fills `descriptor` with the L2-normalised exponentially-decayed `w×w` patch centred on the
    /// event, reading `surface` (its own population's SAE, already updated at the centre pixel).
    /// Returns `false` — leaving `descriptor` untouched for the caller to skip — when the patch
    /// would cross a border. The centre pixel just fired, so the patch norm is always positive.
    fn fill_descriptor(
        &self,
        surface: &[i64],
        event: &crate::Event,
        width: usize,
        height: usize,
        scale: f64,
        descriptor: &mut [f32],
    ) -> bool {
        let radius = self.config.patch / 2;
        if event.x < radius
            || event.y < radius
            || event.x + radius >= width
            || event.y + radius >= height
        {
            return false;
        }
        let now = event.timestamp as i64;
        let mut norm = 0.0_f32;
        let mut slot = 0;
        for dy in -(radius as i32)..=radius as i32 {
            for dx in -(radius as i32)..=radius as i32 {
                let x = (event.x as i32 + dx) as usize;
                let y = (event.y as i32 + dy) as usize;
                let last = surface[y * width + x];
                let value = if last == i64::MIN {
                    0.0
                } else {
                    let age_ms = (now - last).max(0) as f64 * scale;
                    (-age_ms / self.config.tau_ms).exp() as f32
                };
                descriptor[slot] = value;
                norm += value * value;
                slot += 1;
            }
        }
        if norm <= 0.0 {
            return false;
        }
        let inv = 1.0 / norm.sqrt();
        descriptor.iter_mut().for_each(|value| *value *= inv);
        true
    }

    /// Best-matching feature in `bank` whose cosine distance is within its threshold (smallest
    /// distance wins), or `None` if the descriptor lands outside every threshold (a miss).
    fn best_within_threshold(&self, bank: usize, descriptor: &[f32]) -> Option<usize> {
        let n = self.config.n_features;
        let mut best = None;
        let mut best_distance = f32::INFINITY;
        for feature in 0..n {
            let distance = self.cosine_distance(bank * n + feature, descriptor);
            if distance <= self.thresholds[bank * n + feature] && distance < best_distance {
                best_distance = distance;
                best = Some(feature);
            }
        }
        best
    }

    /// Nearest feature in `bank` by cosine distance, ignoring thresholds (inference rule).
    fn nearest(&self, bank: usize, descriptor: &[f32]) -> usize {
        let n = self.config.n_features;
        (0..n)
            .min_by(|&a, &b| {
                let da = self.cosine_distance(bank * n + a, descriptor);
                let db = self.cosine_distance(bank * n + b, descriptor);
                da.total_cmp(&db)
            })
            .expect("n_features is at least 1")
    }

    /// Cosine distance `1 − w·d` between global feature `index` and a unit-norm descriptor.
    fn cosine_distance(&self, index: usize, descriptor: &[f32]) -> f32 {
        let base = index * self.dim;
        let dot: f32 = self.weights[base..base + self.dim]
            .iter()
            .zip(descriptor)
            .map(|(weight, input)| weight * input)
            .sum();
        1.0 - dot
    }
}

fn validate(config: &FeastConfig) -> Result<(), FeastError> {
    if config.n_features == 0 {
        return Err(FeastError::InvalidParameter("n_features must be at least 1"));
    }
    if config.patch == 0 || config.patch.is_multiple_of(2) {
        return Err(FeastError::InvalidParameter("patch must be odd and at least 1"));
    }
    if !config.tau_ms.is_finite() || config.tau_ms <= 0.0 {
        return Err(FeastError::InvalidParameter("tau_ms must be finite and positive"));
    }
    if !config.eta.is_finite() || config.eta <= 0.0 || config.eta > 1.0 {
        return Err(FeastError::InvalidParameter("eta must be in (0, 1]"));
    }
    if !config.delta_i.is_finite() || config.delta_i < 0.0 {
        return Err(FeastError::InvalidParameter("delta_i must be finite and non-negative"));
    }
    if !config.delta_e.is_finite() || config.delta_e < 0.0 {
        return Err(FeastError::InvalidParameter("delta_e must be finite and non-negative"));
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
pub enum FeastError {
    InvalidParameter(&'static str),
    StateShapeMismatch,
}

impl fmt::Display for FeastError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(message) => formatter.write_str(message),
            Self::StateShapeMismatch => {
                formatter.write_str("feature/threshold arrays do not match the configuration")
            }
        }
    }
}

impl Error for FeastError {}

#[cfg(test)]
mod tests {
    use super::{Feast, FeastConfig, FeastError};
    use crate::EventStream;

    /// A vertical bar sweeping left→right across a `size×size` sensor, repeated `sweeps` times.
    /// Column `x` fires (all rows) one step after column `x−1`, so the local time surface around
    /// each event is a clean spatial ramp — a learnable moving-edge feature.
    fn sweeping_bar(size: usize, sweeps: usize) -> EventStream {
        let mut rows = Vec::new();
        let mut t = 0_u64;
        for _ in 0..sweeps {
            for x in 0..size as u64 {
                for y in 0..size as u64 {
                    rows.push([x, y, t, 1]);
                }
                t += 1_000; // 1 ms/column
            }
        }
        let flat: Vec<u64> = rows.concat();
        let events = ndarray::Array2::from_shape_vec((rows.len(), 4), flat).unwrap();
        EventStream::from_array2(events, size, size, 0.001)
    }

    fn config(n_features: usize, patch: usize, per_polarity: bool) -> FeastConfig {
        FeastConfig {
            n_features,
            patch,
            per_polarity,
            ..FeastConfig::default()
        }
    }

    #[test]
    fn new_rejects_invalid_parameters() {
        assert_eq!(
            Feast::new(config(0, 5, false)).unwrap_err(),
            FeastError::InvalidParameter("n_features must be at least 1")
        );
        assert_eq!(
            Feast::new(config(4, 4, false)).unwrap_err(),
            FeastError::InvalidParameter("patch must be odd and at least 1")
        );
        assert!(matches!(
            Feast::new(FeastConfig {
                tau_ms: 0.0,
                ..config(4, 5, false)
            }),
            Err(FeastError::InvalidParameter(_))
        ));
    }

    #[test]
    fn init_is_deterministic_and_unit_norm() {
        let a = Feast::new(config(8, 5, false)).unwrap();
        let b = Feast::new(config(8, 5, false)).unwrap();
        assert_eq!(a.weights(), b.weights());
        assert_eq!(a.thresholds(), b.thresholds());
        // Every feature lies on the unit hypersphere.
        for feature in a.weights().chunks(5 * 5) {
            let norm: f32 = feature.iter().map(|value| value * value).sum();
            assert!((norm - 1.0).abs() < 1e-5, "‖w‖² = {norm}");
        }
    }

    #[test]
    fn fit_converges_to_a_low_miss_rate() {
        // Short epochs so the cold-start burst of misses (random features/thresholds reject most
        // early inputs) dominates epoch 1's average, then falls away as the network adapts.
        let stream = sweeping_bar(16, 4);
        let mut feast = Feast::new(config(8, 5, false)).unwrap();
        let first = feast.fit(&stream, 1);
        let mut last = first;
        for _ in 0..14 {
            last = feast.fit(&stream, 1);
        }
        // A cold network misses heavily; adaptation drives it to a low steady state (paper: a few
        // percent), so the converged rate is both far below the first epoch and small in absolute
        // terms.
        assert!(first > 0.1, "cold network should miss heavily, got {first}");
        assert!(last < first, "miss rate should fall with training: {first} → {last}");
        assert!(last < 0.05, "converged miss rate too high: {last}");
    }

    #[test]
    fn transform_labels_are_aligned_and_in_range() {
        let stream = sweeping_bar(16, 20);
        let mut feast = Feast::new(config(6, 5, false)).unwrap();
        feast.fit(&stream, 4);

        let ids = feast.transform(&stream);
        assert_eq!(ids.len(), stream.len());
        assert!(ids.iter().all(|&id| (-1..feast.n_features_total() as i32).contains(&id)));
        // Most events sit inside the border, so most are labelled.
        let labelled = ids.iter().filter(|&&id| id >= 0).count();
        assert!(labelled > stream.len() / 2);

        // The histogram is exactly the tally of the non-border labels.
        let histogram = feast.histogram(&stream);
        assert_eq!(histogram.len(), feast.n_features_total());
        assert_eq!(histogram.iter().sum::<u32>() as usize, labelled);
    }

    #[test]
    fn per_polarity_splits_populations() {
        let mut rows = Vec::new();
        let mut t = 0_u64;
        for x in 0..16_u64 {
            for y in 0..16_u64 {
                rows.push([x, y, t, (x % 2)]); // alternate ON/OFF columns
            }
            t += 1_000;
        }
        let flat: Vec<u64> = rows.concat();
        let events = ndarray::Array2::from_shape_vec((rows.len(), 4), flat).unwrap();
        let stream = EventStream::from_array2(events, 16, 16, 0.001);

        let mut feast = Feast::new(config(5, 5, true)).unwrap();
        feast.fit(&stream, 4);
        assert_eq!(feast.n_features_total(), 10);
        // OFF events must be able to land in the second population (ids ≥ n_features).
        assert!(feast.transform(&stream).iter().any(|&id| id >= 5));
    }

    #[test]
    fn from_state_round_trips_the_model() {
        let stream = sweeping_bar(16, 10);
        let mut feast = Feast::new(config(6, 5, false)).unwrap();
        feast.fit(&stream, 3);

        let rebuilt = Feast::from_state(
            *feast.config(),
            feast.weights().to_vec(),
            feast.thresholds().to_vec(),
        )
        .unwrap();
        assert_eq!(feast.transform(&stream), rebuilt.transform(&stream));

        assert_eq!(
            Feast::from_state(*feast.config(), vec![0.0; 3], vec![0.0; 2]).unwrap_err(),
            FeastError::StateShapeMismatch
        );
    }
}
