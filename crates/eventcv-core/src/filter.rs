//! Denoising filters — drop noise events from the sparse [`EventStream`] (OpenCV `imgproc`/`photo`
//! analogue). Each filter keeps `(x, y, p, t)` unchanged and only removes events, returning a
//! **new** stream so calls chain. The polarity filter lives with the transforms
//! ([`EventStream::filter_polarity`]).
//!
//! The neighbourhood/dead-time filters maintain a per-pixel surface of timestamps, so they assume
//! events arrive in **ascending time order** (what the readers produce); call
//! [`EventStream::sort_by_time`] first if a stream might be unordered.

use crate::{EventStream, EventStreamBuilder};

/// Which pixels count as a neighbour for [`EventStream::background_activity_filter_with`].
///
/// The choice is a real accuracy trade-off, not a detail. A `Block` has eight chances to find a
/// recent neighbour where a `Cross` has four, so `Cross` is the stricter test: on uncorrelated shot
/// noise it retains measurably more signal at matched noise rejection (3–6.5 points of true-positive
/// rate against a per-event ground truth), which is why tonic and jAER use it. `Block` is the more
/// permissive reading and may suit correlated or textured noise better. `Block` is the default
/// because it is what this filter has always done.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Neighbourhood {
    /// The four edge-adjacent pixels.
    Cross,
    /// All eight surrounding pixels, diagonals included.
    #[default]
    Block,
}

impl Neighbourhood {
    /// Offsets from the centre pixel. `Cross` is the first four of `Block`, so one table serves
    /// both.
    fn offsets(self) -> &'static [(i32, i32)] {
        const BLOCK: [(i32, i32); 8] = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ];
        match self {
            Self::Cross => &BLOCK[..4],
            Self::Block => &BLOCK,
        }
    }

    /// Parses `"cross"` / `"block"` (case-insensitively), the spelling used by the Python
    /// `neighbourhood=` argument.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cross" => Some(Self::Cross),
            "block" => Some(Self::Block),
            _ => None,
        }
    }
}

/// Which events restart a pixel's dead time in [`EventStream::refractory_filter_with`].
///
/// `Kept` is the hardware reading — a silicon pixel's refractory period runs from the event it
/// *emitted* — and is what this filter has always done, so it is the default. `All` restarts the
/// clock on every arriving event, so a pixel must fall silent for `dt` before it emits again;
/// that is what tonic and evlib do, and under a noise burst it suppresses more (worth about 7
/// points of true-positive rate at matched noise rejection in a ground-truth comparison). The two
/// are indistinguishable on clean data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RefreshOn {
    /// Only emitted events restart the dead time.
    #[default]
    Kept,
    /// Every event restarts it, suppressed ones included.
    All,
}

impl RefreshOn {
    /// Parses `"kept"` / `"all"` (case-insensitively), the spelling used by the Python `refresh=`
    /// argument.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "kept" => Some(Self::Kept),
            "all" => Some(Self::All),
            _ => None,
        }
    }
}

impl EventStream {
    /// Background-activity (nearest-neighbour) noise filter. Keeps an event only if some pixel in
    /// its 3×3 neighbourhood fired within `dt` (raw timestamp units, as [`Self::time_window`]):
    /// uncorrelated noise, which has no recent neighbours, is dropped. Assumes ascending time.
    ///
    /// See [`Self::background_activity_filter_with`] to use the 4-neighbour cross instead, which
    /// rejects uncorrelated noise more cleanly.
    pub fn background_activity_filter(&self, dt: i64) -> EventStream {
        self.background_activity_filter_with(dt, Neighbourhood::Block)
    }

    /// [`Self::background_activity_filter`] over a chosen [`Neighbourhood`].
    pub fn background_activity_filter_with(
        &self,
        dt: i64,
        neighbourhood: Neighbourhood,
    ) -> EventStream {
        let (width, height) = self.sensor_size();
        self.background_activity_on(&mut vec![i64::MIN; width * height], dt, neighbourhood)
    }

    /// [`Self::background_activity_filter_with`] against `last`, each pixel's latest timestamp
    /// (row-major `width·height`, `i64::MIN` for none), which it updates. Carried from one batch to
    /// the next, it filters a stream batch by batch exactly as in a single call.
    fn background_activity_on(
        &self,
        last: &mut [i64],
        dt: i64,
        neighbourhood: Neighbourhood,
    ) -> EventStream {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 {
            return self.clone();
        }
        let offsets = neighbourhood.offsets();
        // The same offsets as steps through the row-major surface, for pixels off the border.
        let steps: Vec<isize> = offsets
            .iter()
            .map(|&(dx, dy)| dy as isize * width as isize + dx as isize)
            .collect();
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        for index in 0..self.len() {
            let (x, y, t) = (xs[index] as usize, ys[index] as usize, ts[index]);
            let pixel = y * width + x;
            let keep = if x > 0 && y > 0 && x + 1 < width && y + 1 < height {
                steps
                    .iter()
                    .any(|&step| t.saturating_sub(last[pixel.wrapping_add_signed(step)]) <= dt)
            } else {
                offsets.iter().any(|&(dx, dy)| {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    (0..width as i32).contains(&nx)
                        && (0..height as i32).contains(&ny)
                        && t.saturating_sub(last[ny as usize * width + nx as usize]) <= dt
                })
            };
            last[pixel] = t; // seed the surface even for dropped noise
            if keep {
                builder.push(xs[index], ys[index], t, ps[index]);
            }
        }
        builder.build()
    }

    /// Refractory-period filter: after a pixel fires, suppress its events for `dt` (raw timestamp
    /// units). Keeps an event only when at least `dt` has elapsed since that pixel's last *kept*
    /// event; dropped events do not refresh the dead time. Assumes ascending time.
    ///
    /// That rule is [`RefreshOn::Kept`], the hardware definition of a dead time. tonic and evlib
    /// instead restart the clock on every arriving event — see
    /// [`Self::refractory_filter_with`] to reproduce them.
    pub fn refractory_filter(&self, dt: i64) -> EventStream {
        self.refractory_filter_with(dt, RefreshOn::Kept)
    }

    /// [`Self::refractory_filter`] under a chosen [`RefreshOn`] rule.
    pub fn refractory_filter_with(&self, dt: i64, refresh: RefreshOn) -> EventStream {
        let (width, height) = self.sensor_size();
        self.refractory_on(&mut vec![i64::MIN; width * height], dt, refresh)
    }

    /// [`Self::refractory_filter_with`] against `last`, each pixel's dead-time start (row-major
    /// `width·height`, `i64::MIN` for none), which it updates — so carried across batches it
    /// filters as one call over them all.
    fn refractory_on(&self, last: &mut [i64], dt: i64, refresh: RefreshOn) -> EventStream {
        let (width, height) = self.sensor_size();
        if width == 0 || height == 0 {
            return self.clone();
        }
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        // Branching on the variant only where the event is *dropped* keeps the kept path exactly
        // the shape it had before the rule became a parameter.
        let refresh_all = refresh == RefreshOn::All;
        for index in 0..self.len() {
            let idx = ys[index] as usize * width + xs[index] as usize;
            if ts[index].saturating_sub(last[idx]) >= dt {
                last[idx] = ts[index];
                builder.push(xs[index], ys[index], ts[index], ps[index]);
            } else if refresh_all {
                last[idx] = ts[index];
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
        match hot_pixel_threshold(counts, n_std) {
            Some(threshold) => counts.iter().map(|&c| c as f64 > threshold).collect(),
            None => vec![false; counts.len()], // no events -> nothing is hot
        }
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

/// The count above which a pixel is hot: `mean + n_std·std` over the active (non-zero) `counts`, or
/// `None` when there are none.
fn hot_pixel_threshold(counts: &[u64], n_std: f64) -> Option<f64> {
    let active = || counts.iter().filter(|&&c| c > 0);
    let (n, sum) = active().fold((0_u64, 0_u64), |(n, sum), &c| (n + 1, sum + c));
    if n == 0 {
        return None;
    }
    let n = n as f64;
    let mean = sum as f64 / n;
    let variance = active().map(|&c| (c as f64 - mean).powi(2)).sum::<f64>() / n;
    Some(mean + n_std * variance.sqrt())
}

// Streaming pipeline, shared by the GUI viewer and the TUI.

/// One stage of a [`Pipeline`]. Stages see a stream in consecutive batches (playback windows or
/// camera packets) and carry their per-pixel state from one to the next, so the filters above
/// behave continuously across batch boundaries. A caller going back over time it has already fed
/// (a seek) must [`reset`](Self::reset) first, then warm the stage on [`lookback_us`] of input
/// before the window it wants.
///
/// [`lookback_us`]: Self::lookback_us
pub trait Processor: Send {
    fn lookback_us(&self) -> i64;
    /// Feeds `stream` through and returns its events inside `window`, `[start, end)`.
    fn process(&mut self, window: (i64, i64), stream: EventStream) -> EventStream;
    fn reset(&mut self);
}

/// `surface` sized for `stream`'s sensor; a new (or resized) one starts with no events.
fn fit_surface(surface: &mut Vec<i64>, stream: &EventStream) {
    let (width, height) = stream.sensor_size();
    if surface.len() != width * height {
        *surface = vec![i64::MIN; width * height];
    }
}

/// Streaming [`EventStream::background_activity_filter`]. The timestamp surface persists, rather
/// than being rebuilt each batch from a re-filtered `dt` of history: a 1280×720 surface is 7 MB
/// to refill, and at sensor rates the history is as many events again as the batch.
pub struct BackgroundProcessor {
    dt_us: i64,
    last: Vec<i64>,
}

impl BackgroundProcessor {
    pub fn new(dt_us: i64) -> Self {
        Self {
            dt_us: dt_us.max(1),
            last: Vec::new(),
        }
    }
}

impl Processor for BackgroundProcessor {
    fn lookback_us(&self) -> i64 {
        self.dt_us
    }

    fn process(&mut self, window: (i64, i64), stream: EventStream) -> EventStream {
        fit_surface(&mut self.last, &stream);
        stream
            .background_activity_on(&mut self.last, self.dt_us, Neighbourhood::Block)
            .time_window(window.0, window.1)
    }

    fn reset(&mut self) {
        self.last.fill(i64::MIN);
    }
}

/// Streaming [`EventStream::refractory_filter`], its dead-time surface carried across batches.
pub struct RefractoryProcessor {
    dt_us: i64,
    last: Vec<i64>,
}

impl RefractoryProcessor {
    pub fn new(dt_us: i64) -> Self {
        Self {
            dt_us: dt_us.max(1),
            last: Vec::new(),
        }
    }
}

impl Processor for RefractoryProcessor {
    fn lookback_us(&self) -> i64 {
        self.dt_us
    }

    fn process(&mut self, window: (i64, i64), stream: EventStream) -> EventStream {
        fit_surface(&mut self.last, &stream);
        stream
            .refractory_on(&mut self.last, self.dt_us, RefreshOn::Kept)
            .time_window(window.0, window.1)
    }

    fn reset(&mut self) {
        self.last.fill(i64::MIN);
    }
}

/// Streaming [`EventStream::hot_pixel_filter`]: pixel counts add up across batches, so the mask
/// firms up as the stream runs instead of being re-thresholded on each batch alone.
pub struct HotPixelProcessor {
    n_std: f64,
    counts: Vec<u64>,
}

impl HotPixelProcessor {
    pub fn new(n_std: f64) -> Self {
        Self {
            n_std,
            counts: Vec::new(),
        }
    }
}

impl Processor for HotPixelProcessor {
    fn lookback_us(&self) -> i64 {
        0
    }

    fn process(&mut self, _window: (i64, i64), stream: EventStream) -> EventStream {
        let (width, height) = stream.sensor_size();
        self.counts.resize(width * height, 0);
        stream.add_pixel_counts(&mut self.counts);
        let Some(threshold) = hot_pixel_threshold(&self.counts, self.n_std) else {
            return stream;
        };
        // Tested against the counts directly: a whole-sensor mask each batch costs more than the
        // events it removes.
        let counts = &self.counts;
        stream.remap(width, height, |x, y, t, p| {
            (counts[y as usize * width + x as usize] as f64 <= threshold).then_some((x, y, t, p))
        })
    }

    fn reset(&mut self) {
        self.counts.clear();
    }
}

/// [`Processor`]s applied in order.
#[derive(Default)]
pub struct Pipeline {
    processors: Vec<Box<dyn Processor>>,
}

impl Pipeline {
    pub fn new(processors: Vec<Box<dyn Processor>>) -> Self {
        Self { processors }
    }

    pub fn lookback_us(&self) -> i64 {
        // Chained filters can extend one another's required input history, so sum is the safe bound.
        self.processors.iter().fold(0_i64, |total, processor| {
            total.saturating_add(processor.lookback_us())
        })
    }

    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
    }

    pub fn process(&mut self, window: (i64, i64), mut stream: EventStream) -> EventStream {
        for processor in &mut self.processors {
            stream = processor.process(window, stream);
        }
        stream
    }
}

#[cfg(test)]
mod tests {
    use super::{HotPixelProcessor, Processor, RefractoryProcessor};
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

    /// The cross is the block minus its diagonals, so it can only ever keep fewer events — and on
    /// a pixel whose only recent neighbour *is* diagonal, it keeps none.
    #[test]
    fn cross_is_stricter_than_block() {
        use crate::filter::Neighbourhood;

        // (5,5) fires, then (6,6) — diagonally adjacent, so a neighbour under Block but not Cross.
        let stream = build(16, 16, &[(5, 5, 0, true), (6, 6, 10, true)]);
        assert_eq!(
            stream
                .background_activity_filter_with(100, Neighbourhood::Block)
                .len(),
            1
        );
        assert_eq!(
            stream
                .background_activity_filter_with(100, Neighbourhood::Cross)
                .len(),
            0
        );

        // The public entry point is Block, unchanged.
        let cluster = build(
            32,
            32,
            &[
                (5, 5, 100, true),
                (6, 5, 110, true),
                (5, 6, 120, false),
                (6, 6, 130, true),
                (20, 20, 200, true),
            ],
        );
        assert_eq!(
            cluster.background_activity_filter(50).ts(),
            cluster
                .background_activity_filter_with(50, Neighbourhood::Block)
                .ts()
        );
    }

    /// `All` restarts the dead time on suppressed events too, so a pixel under a burst has to fall
    /// silent for `dt` before it emits again.
    #[test]
    fn refresh_on_all_suppresses_more_than_refresh_on_kept() {
        use crate::filter::RefreshOn;

        // A burst at one pixel every 50 units. Under `Kept` the clock runs from t=0, so t=100 is
        // emitted; under `All` every arrival pushes it out and nothing after t=0 survives.
        let stream = build(
            8,
            8,
            &[
                (3, 3, 0, true),
                (3, 3, 50, true),
                (3, 3, 100, true),
                (3, 3, 150, true),
            ],
        );
        assert_eq!(
            stream.refractory_filter_with(100, RefreshOn::Kept).ts(),
            &[0, 100]
        );
        assert_eq!(
            stream.refractory_filter_with(100, RefreshOn::All).ts(),
            &[0]
        );

        // The public entry point is `Kept`, unchanged.
        assert_eq!(
            stream.refractory_filter(100).ts(),
            stream.refractory_filter_with(100, RefreshOn::Kept).ts()
        );
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

    /// Pseudo-random events in time order, dense enough that neighbours and dead times interact.
    fn noise(n: usize) -> EventStream {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let events: Vec<_> = (0..n as i64)
            .map(|t| {
                let r = next();
                (
                    (r % 24) as u16,
                    (r >> 8) as u16 % 16,
                    t * 7,
                    r >> 20 & 1 == 1,
                )
            })
            .collect();
        build(24, 16, &events)
    }

    fn assert_same(a: &EventStream, b: &EventStream) {
        assert_eq!(
            (a.xs(), a.ys(), a.ts(), a.ps()),
            (b.xs(), b.ys(), b.ts(), b.ps())
        );
    }

    /// `processor` fed `stream` in `batch`-µs windows, the pieces joined back up.
    fn in_batches(mut processor: impl Processor, stream: &EventStream, batch: i64) -> EventStream {
        let end = stream.ts().last().unwrap() + 1;
        let pieces: Vec<_> = (0..end)
            .step_by(batch as usize)
            .map(|start| {
                let window = (start, (start + batch).min(end));
                processor.process(window, stream.time_window(window.0, window.1))
            })
            .collect();
        let empty = EventStreamBuilder::new(24, 16, 0.001).build();
        empty.concat(&pieces.iter().collect::<Vec<_>>())
    }

    #[test]
    fn streaming_matches_one_call_over_the_whole_stream() {
        use super::BackgroundProcessor;

        let stream = noise(5_000);
        for batch in [1, 97, 1_000, 40_000] {
            let background = in_batches(BackgroundProcessor::new(60), &stream, batch);
            assert_same(&background, &stream.background_activity_filter(60));
            let refractory = in_batches(RefractoryProcessor::new(900), &stream, batch);
            assert_same(&refractory, &stream.refractory_filter(900));
        }
        // Neither the filters nor the batching make this vacuous.
        let kept = stream.background_activity_filter(60).len();
        assert!(0 < kept && kept < stream.len());
    }

    #[test]
    fn reset_forgets_what_was_fed() {
        use super::BackgroundProcessor;

        let mut processor = BackgroundProcessor::new(100);
        let pair = build(4, 4, &[(1, 1, 10, true)]);
        processor.process((0, 20), build(4, 4, &[(0, 1, 5, true)]));
        assert_eq!(processor.process((0, 20), pair.clone()).len(), 1);
        processor.reset();
        assert!(processor.process((0, 20), pair).is_empty());
    }

    #[test]
    fn hot_pixel_processor_matches_the_mask() {
        let stream = noise(3_000).concat(&[&build(24, 16, &[(3, 3, 30_000, true); 400])]);
        let filtered = HotPixelProcessor::new(2.0).process((0, i64::MAX), stream.clone());
        assert_same(&filtered, &stream.hot_pixel_filter(2.0));
        assert!(filtered.len() < stream.len());
    }

    #[test]
    fn refractory_history_crosses_window_boundary() {
        let mut processor = RefractoryProcessor::new(1_000);
        let first = processor.process((0, 1_000), build(4, 4, &[(0, 0, 900, true)]));
        let second = processor.process((1_000, 2_000), build(4, 4, &[(0, 0, 1_100, true)]));
        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
    }

    #[test]
    fn hot_pixel_counts_carry_across_batches() {
        // Alone, the second batch is uniform and would keep everything; the first batch's tally
        // still marks (0, 0) as hot.
        let mut processor = HotPixelProcessor::new(1.0);
        let mut first: Vec<_> = (0..50).map(|t| (0, 0, t, true)).collect();
        first.extend((1..4).map(|x| (x, 1, 60, true)));
        assert_eq!(processor.process((0, 100), build(4, 4, &first)).len(), 3);
        let second = [(0, 0, 150, true), (1, 1, 150, true), (2, 1, 150, true)];
        let second = processor.process((100, 200), build(4, 4, &second));
        assert_eq!(second.xs(), &[1, 2]);
    }
}
