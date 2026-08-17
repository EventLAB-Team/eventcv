//! Random augmentations — the training-time counterparts of the deterministic ops in `transform`.
//!
//! Every augmentation takes an explicit `seed` and is a pure function of `(stream, seed)`: the same
//! seed always produces the same output. Nothing here reads the thread RNG or the clock, because an
//! augmentation whose result depends on *when* it ran cannot be reproduced from a training log.
//!
//! The ops that only sometimes fire (`random_flip_x`, `time_reversal`, …) take a probability `p` and
//! draw once per call, so the decision applies to the whole stream rather than per event. The ops
//! that perturb events (`event_drop`, `spatial_jitter`, …) draw per event.
//!
//! Where an augmentation is just "sometimes do an existing transform", it calls that transform rather
//! than reimplementing it — the geometry lives in one place. Note that the not-firing branch returns
//! `self.clone()`, which is a real deep copy of the column arrays; callers that augment large slices
//! should prefer chaining over calling an augmentation they expect to be a no-op.

use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::EventStream;

/// Derives an augmentation's RNG from its seed and the index of the slice being augmented.
///
/// Deferred reader ops see slices in whatever order the consumer asks for them — a shuffled
/// `DataLoader`, or several worker processes at once — so an RNG carried across calls would make the
/// augmentation depend on access order. Seeding per `(seed, index)` instead means slice `i` augments
/// identically however it is reached. The multiply-xorshift is splitmix64's finalizer, which
/// decorrelates the small, adjacent indices we actually pass in (0, 1, 2, …).
pub fn slice_rng(seed: u64, index: usize) -> StdRng {
    let mut z = seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    StdRng::seed_from_u64(z ^ (z >> 31))
}

impl EventStream {
    /// Mirrors the stream left-right with probability `p`.
    pub fn random_flip_x(&self, p: f64, seed: u64) -> EventStream {
        if fires(p, seed) {
            self.flip_x()
        } else {
            self.clone()
        }
    }

    /// Mirrors the stream top-bottom with probability `p`.
    pub fn random_flip_y(&self, p: f64, seed: u64) -> EventStream {
        if fires(p, seed) {
            self.flip_y()
        } else {
            self.clone()
        }
    }

    /// Inverts every polarity with probability `p`.
    ///
    /// Draws once for the stream, not once per event: flipping a random *subset* of polarities is
    /// label noise, whereas flipping all of them is the physically meaningful augmentation (the same
    /// scene with the contrast direction reversed).
    pub fn random_polarity_flip(&self, p: f64, seed: u64) -> EventStream {
        if fires(p, seed) {
            self.invert_polarity()
        } else {
            self.clone()
        }
    }

    /// Takes a random `width × height` crop. A window at least as large as the sensor is the
    /// identity, so this is safe to leave in a pipeline that also runs on smaller recordings.
    pub fn random_crop(&self, width: usize, height: usize, seed: u64) -> EventStream {
        let (sensor_w, sensor_h) = self.sensor_size();
        if width >= sensor_w && height >= sensor_h {
            return self.clone();
        }
        let mut rng = slice_rng(seed, 0);
        let x0 = rng.gen_range(0..=sensor_w.saturating_sub(width)) as i64;
        let y0 = rng.gen_range(0..=sensor_h.saturating_sub(height)) as i64;
        self.crop(x0, y0, width, height)
    }

    /// Drops each event independently with probability `p`, thinning the stream without changing its
    /// geometry or duration. `p <= 0` is the identity; `p >= 1` empties the stream.
    pub fn event_drop(&self, p: f64, seed: u64) -> EventStream {
        if p <= 0.0 {
            return self.clone();
        }
        let mut rng = slice_rng(seed, 0);
        let (width, height) = self.sensor_size();
        self.remap(width, height, move |x, y, t, polarity| {
            (rng.gen::<f64>() >= p).then_some((x, y, t, polarity))
        })
    }

    /// Silences a random `p` fraction of *pixels* for the whole stream.
    ///
    /// Unlike [`EventStream::event_drop`], which thins events independently, this removes every event
    /// from the chosen pixels — the failure mode of a real sensor with dead pixels, and a much harder
    /// augmentation for a model to average away.
    pub fn pixel_dropout(&self, p: f64, seed: u64) -> EventStream {
        if p <= 0.0 {
            return self.clone();
        }
        let (width, height) = self.sensor_size();
        let mut rng = slice_rng(seed, 0);
        // `drop_masked_pixels` reads `true` as *drop* — the opposite of `EventStream::mask`, where
        // `true` keeps. So this is the set to silence, and `p` is the fraction marked `true`.
        let drop: Vec<bool> = (0..width * height).map(|_| rng.gen::<f64>() < p).collect();
        self.drop_masked_pixels(&drop)
    }

    /// Jitters each event's position by a rounded gaussian offset with standard deviation `sigma`
    /// pixels. Events pushed off the sensor are dropped, so a large `sigma` also thins the stream.
    pub fn spatial_jitter(&self, sigma: f64, seed: u64) -> EventStream {
        if sigma <= 0.0 {
            return self.clone();
        }
        let normal = match Normal::new(0.0, sigma) {
            Ok(normal) => normal,
            Err(_) => return self.clone(),
        };
        let mut rng = slice_rng(seed, 0);
        let (width, height) = self.sensor_size();
        self.remap(width, height, move |x, y, t, polarity| {
            let dx = normal.sample(&mut rng).round() as i64;
            let dy = normal.sample(&mut rng).round() as i64;
            Some((x + dx, y + dy, t, polarity))
        })
    }

    /// Jitters each event's timestamp by a rounded gaussian offset with standard deviation `sigma`
    /// (same units as the stored timestamps, i.e. µs).
    ///
    /// Re-sorts afterwards: jitter can reorder neighbouring events, and the correlation-based filters
    /// (`background_activity_filter`, `refractory_filter`) require ascending time.
    pub fn time_jitter(&self, sigma: f64, seed: u64) -> EventStream {
        if sigma <= 0.0 {
            return self.clone();
        }
        let normal = match Normal::new(0.0, sigma) {
            Ok(normal) => normal,
            Err(_) => return self.clone(),
        };
        let mut rng = slice_rng(seed, 0);
        let (width, height) = self.sensor_size();
        let jittered = self.remap(width, height, move |x, y, t, polarity| {
            Some((x, y, t + normal.sample(&mut rng).round() as i64, polarity))
        });
        jittered.sort_by_time()
    }

    /// Plays the stream backwards with probability `p`, inverting polarity to match.
    ///
    /// Reversing time without inverting polarity would be physically wrong: an edge that brightened
    /// as it passed darkens when the same motion is run in reverse. Timestamps are mirrored within
    /// the stream's own span, so the result starts and ends where the original did.
    pub fn time_reversal(&self, p: f64, seed: u64) -> EventStream {
        if !fires(p, seed) || self.is_empty() {
            return self.clone();
        }
        let ts = self.ts();
        let (&t_min, &t_max) = match (ts.iter().min(), ts.iter().max()) {
            (Some(min), Some(max)) => (min, max),
            _ => return self.clone(),
        };
        let sum = t_min + t_max;
        let (width, height) = self.sensor_size();
        self.remap(width, height, |x, y, t, polarity| {
            Some((x, y, sum - t, !polarity))
        })
        .sort_by_time()
    }
}

/// Draws the single "does this augmentation apply?" decision. `p <= 0` never fires and `p >= 1`
/// always does, so the boundaries are exact rather than left to a float comparison.
fn fires(p: f64, seed: u64) -> bool {
    if p <= 0.0 {
        return false;
    }
    if p >= 1.0 {
        return true;
    }
    slice_rng(seed, 0).gen::<f64>() < p
}

#[cfg(test)]
mod tests {
    use crate::{EventStream, EventStreamBuilder};

    fn sample() -> EventStream {
        let mut builder = EventStreamBuilder::new(8, 6, 0.001);
        for i in 0..32u16 {
            builder.push(i % 8, i % 6, 100 + i64::from(i) * 10, i % 2 == 0);
        }
        builder.build()
    }

    fn coords(stream: &EventStream) -> Vec<(u16, u16)> {
        stream
            .xs()
            .iter()
            .copied()
            .zip(stream.ys().iter().copied())
            .collect()
    }

    #[test]
    fn probability_bounds_are_exact() {
        let s = sample();
        assert_eq!(coords(&s.random_flip_x(0.0, 7)), coords(&s));
        assert_eq!(coords(&s.random_flip_x(1.0, 7)), coords(&s.flip_x()));
        assert_eq!(s.event_drop(0.0, 7).len(), s.len());
        assert_eq!(s.event_drop(1.0, 7).len(), 0);
    }

    #[test]
    fn same_seed_gives_identical_output() {
        let s = sample();
        assert_eq!(s.event_drop(0.5, 42).ts(), s.event_drop(0.5, 42).ts());
        assert_eq!(
            coords(&s.spatial_jitter(1.5, 42)),
            coords(&s.spatial_jitter(1.5, 42))
        );
    }

    #[test]
    fn different_seeds_give_different_output() {
        let s = sample();
        // Not a guarantee for every pair of seeds, but with 32 events at p=0.5 a collision is ~2^-32.
        assert_ne!(s.event_drop(0.5, 1).len(), s.event_drop(0.5, 2).len());
    }

    #[test]
    fn slice_rng_decorrelates_adjacent_indices() {
        use rand::Rng;
        // Adjacent slice indices must not produce correlated draws — that is the whole point of
        // running the seed through the finalizer rather than adding the index to it.
        let draws: Vec<f64> = (0..8)
            .map(|index| super::slice_rng(0, index).gen::<f64>())
            .collect();
        for window in draws.windows(2) {
            assert!((window[0] - window[1]).abs() > 1e-6);
        }
    }

    #[test]
    fn event_drop_thins_without_moving_events() {
        let s = sample();
        let dropped = s.event_drop(0.5, 3);
        assert!(dropped.len() < s.len() && !dropped.is_empty());
        // Every surviving event must be one of the originals, unchanged.
        let original: Vec<_> = s
            .ts()
            .iter()
            .zip(coords(&s))
            .map(|(t, xy)| (*t, xy))
            .collect();
        for (t, xy) in dropped.ts().iter().zip(coords(&dropped)) {
            assert!(original.contains(&(*t, xy)));
        }
    }

    #[test]
    fn pixel_dropout_removes_whole_pixels() {
        let s = sample();
        let dropped = s.pixel_dropout(0.5, 5);
        let survivors: std::collections::HashSet<_> = coords(&dropped).into_iter().collect();
        let removed: std::collections::HashSet<_> = coords(&s)
            .into_iter()
            .filter(|xy| !survivors.contains(xy))
            .collect();
        // A pixel is either kept entirely or gone entirely — the two sets cannot overlap.
        assert!(removed.is_disjoint(&survivors));
        assert!(!removed.is_empty());
    }

    #[test]
    fn pixel_dropout_p_is_the_fraction_removed() {
        // `drop_masked_pixels` reads its mask as *drop*, the opposite of `EventStream::mask`.
        // Getting that backwards still produces a plausible-looking thinned stream, so pin the
        // direction: a small `p` must keep most pixels, not most of them vanish.
        let mut builder = EventStreamBuilder::new(40, 40, 0.001);
        for x in 0..40u16 {
            for y in 0..40u16 {
                builder.push(x, y, i64::from(x) * 40 + i64::from(y), true);
            }
        }
        // One event per pixel, so the surviving fraction of events *is* the surviving fraction
        // of pixels.
        let uniform = builder.build();
        let kept = uniform.pixel_dropout(0.1, 5).len() as f64 / uniform.len() as f64;
        assert!(kept > 0.8, "p=0.1 should keep ~90% of pixels, kept {kept}");
    }

    #[test]
    fn time_reversal_mirrors_span_and_inverts_polarity() {
        let s = sample();
        let reversed = s.time_reversal(1.0, 0);
        assert_eq!(reversed.len(), s.len());
        // Same span, and sorted ascending after the reversal.
        assert_eq!(reversed.ts().first(), s.ts().first());
        assert_eq!(reversed.ts().last(), s.ts().last());
        assert!(reversed.ts().windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(reversed.ps()[0], !s.ps()[s.len() - 1]);
    }

    #[test]
    fn time_jitter_leaves_the_stream_sorted() {
        let jittered = sample().time_jitter(500.0, 11);
        assert!(jittered.ts().windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn random_crop_larger_than_sensor_is_identity() {
        let s = sample();
        assert_eq!(coords(&s.random_crop(64, 64, 9)), coords(&s));
    }

    #[test]
    fn random_crop_bounds_the_result() {
        let cropped = sample().random_crop(3, 2, 9);
        assert_eq!(cropped.sensor_size(), (3, 2));
        assert!(cropped.xs().iter().all(|&x| (x as usize) < 3));
        assert!(cropped.ys().iter().all(|&y| (y as usize) < 2));
    }

    #[test]
    fn augmentations_handle_the_empty_stream() {
        let empty = EventStreamBuilder::new(8, 6, 0.001).build();
        assert!(empty.random_flip_x(1.0, 0).is_empty());
        assert!(empty.event_drop(0.5, 0).is_empty());
        assert!(empty.pixel_dropout(0.5, 0).is_empty());
        assert!(empty.spatial_jitter(2.0, 0).is_empty());
        assert!(empty.time_jitter(2.0, 0).is_empty());
        assert!(empty.time_reversal(1.0, 0).is_empty());
        assert!(empty.random_crop(3, 2, 0).is_empty());
    }
}
