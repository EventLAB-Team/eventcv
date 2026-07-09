//! Denoising filters — drop noise events from the sparse [`EventStream`] (OpenCV `imgproc`/`photo`
//! analogue). Each filter keeps `(x, y, p, t)` unchanged and only removes events, returning a
//! **new** stream so calls chain. The polarity filter lives with the transforms
//! ([`EventStream::filter_polarity`]).
//!
//! The neighbourhood/dead-time filters maintain a per-pixel surface of timestamps, so they assume
//! events arrive in **ascending time order** (what the readers produce); call
//! [`EventStream::sort_by_time`] first if a stream might be unordered.

use crate::{EventStream, EventStreamBuilder};

impl EventStream {
    /// Background-activity (nearest-neighbour) noise filter. Keeps an event only if some pixel in
    /// its 3×3 neighbourhood fired within `dt` (raw timestamp units, as [`Self::time_window`]):
    /// uncorrelated noise, which has no recent neighbours, is dropped. Assumes ascending time.
    pub fn background_activity_filter(&self, dt: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 {
            return self.clone();
        }
        let mut last = vec![i64::MIN; width * height]; // surface of active events
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        for index in 0..self.len() {
            let (x, y, t) = (xs[index] as usize, ys[index] as usize, ts[index]);
            let mut keep = false;
            for ny in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    if (nx, ny) != (x, y) && t.saturating_sub(last[ny * width + nx]) <= dt {
                        keep = true;
                    }
                }
            }
            last[y * width + x] = t; // seed the surface even for dropped noise
            if keep {
                builder.push(xs[index], ys[index], t, ps[index]);
            }
        }
        builder.build()
    }

    /// Refractory-period filter: after a pixel fires, suppress its events for `dt` (raw timestamp
    /// units). Keeps an event only when at least `dt` has elapsed since that pixel's last *kept*
    /// event; dropped events do not refresh the dead time. Assumes ascending time.
    pub fn refractory_filter(&self, dt: i64) -> EventStream {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 {
            return self.clone();
        }
        let mut last = vec![i64::MIN; width * height]; // last emitted timestamp per pixel
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        for index in 0..self.len() {
            let idx = ys[index] as usize * width + xs[index] as usize;
            if ts[index].saturating_sub(last[idx]) >= dt {
                last[idx] = ts[index];
                builder.push(xs[index], ys[index], ts[index], ps[index]);
            }
        }
        builder.build()
    }

    /// Hot-pixel removal: drops every event from stuck pixels whose total event count exceeds
    /// `mean + n_std·std`, with the mean and standard deviation taken over the **active** pixels
    /// (those with at least one event). A uniform or empty stream removes nothing.
    pub fn hot_pixel_filter(&self, n_std: f64) -> EventStream {
        self.drop_masked_pixels(&self.hot_pixel_mask(n_std))
    }

    /// The hot-pixel mask this stream would remove: `true` at each pixel whose total event count
    /// exceeds `mean + n_std·std` (statistics over the **active** pixels), row-major `width·height`.
    /// Degenerate sensors give an empty mask. Exposed on its own so a reader can compute the mask
    /// **once** over a whole recording and apply it to every slice with [`Self::drop_masked_pixels`]
    /// — a per-slice `hot_pixel_filter` instead re-thresholds each window, so hot pixels survive at
    /// long accumulation times.
    pub fn hot_pixel_mask(&self, n_std: f64) -> Vec<bool> {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 {
            return Vec::new();
        }
        let mut counts = vec![0u64; width * height];
        self.add_pixel_counts(&mut counts);
        Self::hot_pixel_mask_from_counts(&counts, n_std)
    }

    /// Adds this stream's per-pixel event counts into `counts` (row-major `width·height` for this
    /// stream's sensor); a mismatched buffer is left untouched. Lets a reader tally a whole
    /// recording chunk-by-chunk — feeding [`Self::hot_pixel_mask_from_counts`] — without ever
    /// materialising the full file, so the pre-scan stays within bounded memory.
    pub fn add_pixel_counts(&self, counts: &mut [u64]) {
        let (width, height) = self.sensor_size();
        if counts.len() != width * height {
            return;
        }
        let (xs, ys) = (self.xs(), self.ys());
        for index in 0..self.len() {
            counts[ys[index] as usize * width + xs[index] as usize] += 1;
        }
    }

    /// The hot-pixel mask for pre-tallied per-pixel `counts`: `true` where a count exceeds
    /// `mean + n_std·std` taken over the **active** (non-zero) pixels. All-zero counts flag nothing.
    /// The chunked-scan counterpart of [`Self::hot_pixel_mask`].
    pub fn hot_pixel_mask_from_counts(counts: &[u64], n_std: f64) -> Vec<bool> {
        let active: Vec<u64> = counts.iter().copied().filter(|&c| c > 0).collect();
        if active.is_empty() {
            return vec![false; counts.len()]; // no events -> nothing is hot
        }
        let n = active.len() as f64;
        let mean = active.iter().sum::<u64>() as f64 / n;
        let variance = active
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let threshold = mean + n_std * variance.sqrt();

        counts.iter().map(|&c| c as f64 > threshold).collect()
    }

    /// Drops every event whose pixel is flagged `true` in `mask` (row-major `width·height`, as
    /// [`Self::hot_pixel_mask`] returns), keeping `(x, y, p, t)` and order. A mask that does not
    /// match the sensor grid (e.g. the empty mask of a degenerate sensor) removes nothing, so a
    /// reader can carry one whole-recording mask and apply it to every slice unconditionally.
    pub fn drop_masked_pixels(&self, mask: &[bool]) -> EventStream {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 || mask.len() != width * height {
            return self.clone();
        }
        self.remap(width, height, |x, y, t, p| {
            (!mask[y as usize * width + x as usize]).then_some((x, y, t, p))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{EventStream, EventStreamBuilder};

    fn build(width: usize, height: usize, events: &[(u16, u16, i64, bool)]) -> EventStream {
        let mut builder = EventStreamBuilder::new(width, height, 0.001);
        for &(x, y, t, p) in events {
            builder.push(x, y, t, p);
        }
        builder.build()
    }

    #[test]
    fn background_activity_drops_isolated_noise_and_keeps_a_cluster() {
        // A tight correlated cluster around (5,5) fires within dt; one isolated event at (20,20)
        // has no recent neighbour and is noise.
        let stream = build(
            32,
            32,
            &[
                (5, 5, 100, true),
                (6, 5, 110, true),
                (5, 6, 120, false),
                (6, 6, 130, true),
                (20, 20, 200, true), // isolated noise
            ],
        );
        let filtered = stream.background_activity_filter(50);
        assert!(filtered.len() < stream.len());
        // The lone event survives nowhere; every kept event is in the cluster region.
        assert!(filtered.xs().iter().all(|&x| x <= 6));
        assert!(filtered
            .xs()
            .iter()
            .zip(filtered.ys())
            .all(|(&x, &y)| x != 20 && y != 20));
        assert_eq!(filtered.sensor_size(), (32, 32));
    }

    #[test]
    fn background_activity_respects_the_dt_window() {
        // Two neighbours far apart in time: with a small dt the second has no *recent* neighbour.
        let stream = build(8, 8, &[(2, 2, 0, true), (3, 2, 1_000, true)]);
        assert_eq!(stream.background_activity_filter(100).len(), 0); // 1000 > dt -> both drop
        assert_eq!(stream.background_activity_filter(2_000).len(), 1); // second sees the first
    }

    #[test]
    fn refractory_suppresses_rapid_repeats_on_one_pixel() {
        let stream = build(
            8,
            8,
            &[
                (3, 3, 0, true),
                (3, 3, 50, true),  // within dt -> dropped
                (3, 3, 500, true), // >= dt after the kept event at t=0 -> kept
                (4, 4, 60, false), // different pixel, unaffected
            ],
        );
        let filtered = stream.refractory_filter(100);
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.ts(), &[0, 500, 60]);
    }

    #[test]
    fn hot_pixel_removes_a_spamming_pixel() {
        // (1,1) fires far more than the rest; a scatter of single-event pixels forms the baseline.
        let mut events: Vec<(u16, u16, i64, bool)> =
            (0..30).map(|t| (1, 1, t as i64, true)).collect();
        for k in 0..20u16 {
            events.push((10 + k % 5, 5 + k / 5, 1_000 + i64::from(k), false));
        }
        let stream = build(32, 32, &events);
        let filtered = stream.hot_pixel_filter(3.0);
        assert!(filtered
            .xs()
            .iter()
            .zip(filtered.ys())
            .all(|(&x, &y)| (x, y) != (1, 1)));
        assert_eq!(filtered.len(), 20); // only the baseline survives
    }

    #[test]
    fn hot_pixel_mask_drops_the_pixel_from_a_window_where_it_would_survive() {
        // Global view: (1,1) is hot. Its mask must still remove (1,1) from a short window where
        // it fires just once — the window a per-slice hot_pixel_filter would leave untouched.
        let mut events: Vec<(u16, u16, i64, bool)> =
            (0..30).map(|t| (1, 1, t as i64, true)).collect();
        for k in 0..20u16 {
            events.push((10 + k % 5, 5 + k / 5, 1_000 + i64::from(k), false));
        }
        let mask = build(32, 32, &events).hot_pixel_mask(3.0);
        assert!(mask[33]); // (x=1, y=1) on a 32-wide grid is flagged

        let window = build(32, 32, &[(1, 1, 0, true), (10, 5, 1, false)]);
        assert_eq!(window.hot_pixel_filter(3.0).len(), 2); // per-window: nothing stands out
        let filtered = window.drop_masked_pixels(&mask); // global mask: (1,1) still goes
        assert_eq!(filtered.len(), 1);
        assert_eq!((filtered.xs()[0], filtered.ys()[0]), (10, 5));
    }

    #[test]
    fn hot_pixel_keeps_a_uniform_stream() {
        // Every active pixel has the same count -> std is 0, nothing exceeds the threshold.
        let events: Vec<(u16, u16, i64, bool)> = (0..16u16)
            .map(|i| (i % 4, i / 4, i64::from(i), true))
            .collect();
        let stream = build(4, 4, &events);
        assert_eq!(stream.hot_pixel_filter(3.0).len(), stream.len());
    }

    #[test]
    fn filters_handle_the_empty_stream() {
        let empty = EventStreamBuilder::new(8, 8, 0.001).build();
        assert!(empty.background_activity_filter(100).is_empty());
        assert!(empty.refractory_filter(100).is_empty());
        assert!(empty.hot_pixel_filter(3.0).is_empty());
    }
}
