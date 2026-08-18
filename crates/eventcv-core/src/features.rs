//! Event feature detection (OpenCV `features2d` analogue). Both detectors maintain a per-polarity
//! Surface of Active Events (SAE — the latest timestamp seen at each pixel) and return a **new**
//! [`EventStream`] holding only the events that sit on a moving corner, so detection **chains**
//! like a denoising filter. Both assume events arrive in **ascending time order** (what the
//! readers produce); call [`EventStream::sort_by_time`] first if a stream might be unordered.

use crate::{EventStream, EventStreamBuilder};

/// Bresenham circle of radius 3 (16 pixels), in contiguous clockwise order — the FAST ring.
const INNER_CIRCLE: [(i32, i32); 16] = [
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 0),
    (3, -1),
    (2, -2),
    (1, -3),
    (0, -3),
    (-1, -3),
    (-2, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-2, 2),
    (-1, 3),
];

/// Bresenham circle of radius 4 (20 pixels), contiguous clockwise — eFAST's outer ring.
const OUTER_CIRCLE: [(i32, i32); 20] = [
    (0, 4),
    (1, 4),
    (2, 3),
    (3, 2),
    (4, 1),
    (4, 0),
    (4, -1),
    (3, -2),
    (2, -3),
    (1, -4),
    (0, -4),
    (-1, -4),
    (-2, -3),
    (-3, -2),
    (-4, -1),
    (-4, 0),
    (-4, 1),
    (-3, 2),
    (-2, 3),
    (-1, 4),
];

/// eFAST arc-length bounds: a corner's most-recent pixels form a contiguous arc this many pixels
/// long. A straight edge fills ~half the ring (too long); noise fills too little.
const INNER_ARC: (usize, usize) = (3, 6);
const OUTER_ARC: (usize, usize) = (4, 8);

/// Harris uses a fixed 9×9 window (radius 4) of the time surface around each event.
const HARRIS_RADIUS: usize = 4;
/// Harris' empirical sensitivity constant `k` in `det - k·trace²`.
const HARRIS_K: f64 = 0.04;

impl EventStream {
    /// **eFAST** event corner detector (Mueggler et al., *Fast Event-based Corner Detection*,
    /// BMVC 2017). For each event it updates its polarity's SAE, then tests two Bresenham rings
    /// (radius 3 and 4) around the pixel: an event is a corner when, on **both** rings, the most
    /// recent timestamps form a contiguous arc within the `INNER_ARC`/`OUTER_ARC` bounds — the
    /// signature of a moving corner rather than a straight edge. Events too close to the border to
    /// evaluate the outer ring are dropped. Returns the corner events as a new stream.
    pub fn efast(&self) -> EventStream {
        let (width, height) = self.sensor_size();
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        if width == 0 || height == 0 {
            return builder.build();
        }
        // Separate surfaces per polarity, as in the paper.
        let mut sae_on = vec![i64::MIN; width * height];
        let mut sae_off = vec![i64::MIN; width * height];

        for index in 0..self.len() {
            let (x, y, t, p) = (xs[index] as usize, ys[index] as usize, ts[index], ps[index]);
            let sae = if p { &mut sae_on } else { &mut sae_off };
            sae[y * width + x] = t; // the current pixel is the newest by construction

            // Need radius 4 clear of every border to sample the outer ring.
            if x < 4 || y < 4 || x + 4 >= width || y + 4 >= height {
                continue;
            }
            if ring_is_corner(sae, x, y, width, &INNER_CIRCLE, INNER_ARC)
                && ring_is_corner(sae, x, y, width, &OUTER_CIRCLE, OUTER_ARC)
            {
                builder.push(xs[index], ys[index], t, p);
            }
        }
        builder.build()
    }

    /// **Harris corner score on the Surface of Active Events.** For each event it updates a
    /// merged SAE of raw latest timestamps, then computes the normalised Harris response
    /// `det(M)/trace(M)² - k` of the structure tensor `M = Σ ∇T ∇Tᵀ` of the SAE's spatial
    /// gradient over a 9×9 window. Because the SAE is a local time *ramp*, a straight moving edge
    /// has a constant gradient direction (rank-1 `M`, `R < 0`) while a corner mixes gradient
    /// directions (rank-2 `M`, `R > 0`) — so the default `threshold = 0` keeps corners and rejects
    /// edges. The score is bounded to `[-k, 0.25 - k]` = `[-0.04, 0.21]`, so raising `threshold`
    /// within that range is what makes it stricter; anything above `0.21` keeps nothing. Returns
    /// the corner events as a new stream; a score-based complement to [`Self::efast`].
    pub fn harris_corners(&self, threshold: f64) -> EventStream {
        let (width, height) = self.sensor_size();
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let scale = self.timestamp_scale_ms();
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        let margin = HARRIS_RADIUS + 1; // central differences need one pixel beyond the window
        if width < 2 * margin + 1 || height < 2 * margin + 1 {
            return builder.build();
        }
        // One merged ramp — flow/corner geometry is independent of contrast polarity.
        let mut sae = vec![i64::MIN; width * height];

        for index in 0..self.len() {
            let (x, y, t, p) = (xs[index] as usize, ys[index] as usize, ts[index], ps[index]);
            sae[y * width + x] = t;

            if x < margin || y < margin || x + margin >= width || y + margin >= height {
                continue;
            }
            if harris_response(&sae, x, y, width, scale) > threshold {
                builder.push(xs[index], ys[index], t, p);
            }
        }
        builder.build()
    }
}

/// Tests whether the ring's most-recent timestamps form a contiguous arc whose length lies in
/// `[min_arc, max_arc]` — i.e. some rotation of the ring has a run of pixels all strictly newer
/// than every pixel outside the run. `circle` offsets are contiguous around the ring so wrap-around
/// runs are handled by indexing modulo the ring length.
fn ring_is_corner(
    sae: &[i64],
    cx: usize,
    cy: usize,
    width: usize,
    circle: &[(i32, i32)],
    (min_arc, max_arc): (usize, usize),
) -> bool {
    let n = circle.len();
    let mut times = [i64::MIN; 20]; // 20 = the largest ring
    for (slot, &(dx, dy)) in times.iter_mut().zip(circle) {
        let px = (cx as i32 + dx) as usize;
        let py = (cy as i32 + dy) as usize;
        *slot = sae[py * width + px];
    }
    let times = &times[..n];

    for length in min_arc..=max_arc {
        for start in 0..n {
            let arc_min = (0..length)
                .map(|k| times[(start + k) % n])
                .min()
                .expect("arc length is at least 1");
            let rest_max = (length..n)
                .map(|k| times[(start + k) % n])
                .max()
                .expect("ring is longer than the arc");
            if arc_min > rest_max {
                return true;
            }
        }
    }
    false
}

/// Minimum number of valid gradient samples in the window for a Harris score to be meaningful.
const HARRIS_MIN_SAMPLES: usize = 3;

/// Harris response of the raw SAE ramp over the `(2R+1)²` window around `(cx, cy)`. Reads the SAE
/// as a time surface `T` (in ms), takes central-difference gradients wherever both neighbours have
/// fired, and returns `det(M)/trace(M)² - k` for the structure tensor `M = Σ ∇T ∇Tᵀ`. Unfired
/// pixels are skipped (their gradient is undefined), so a window with too little structure returns
/// `-∞` (never a corner). The caller guarantees the window lies `HARRIS_RADIUS + 1` inside every
/// border, so the `x ± 1` / `y ± 1` reads are in bounds.
///
/// Dividing by `trace²` is what makes the score a usable *knob* rather than only a sign. `M` is a
/// plain sum over up to 81 samples of a gradient measured in milliseconds, so `det - k·trace²` is
/// quartic in that gradient and unbounded — in practice `1e3`..`1e7`, which no caller can guess.
/// `det/trace²` is `λ₁λ₂/(λ₁+λ₂)²`: dimensionless, independent of how many pixels in the window
/// happened to have fired, and bounded by `0.25`. The score therefore lands in `[-k, 0.25 - k]`.
/// The sign is unchanged — for `trace > 0`, `det - k·trace² > 0` iff `det/trace² - k > 0` — so the
/// default `threshold = 0` selects exactly the same events it always did.
fn harris_response(sae: &[i64], cx: usize, cy: usize, width: usize, scale: f64) -> f64 {
    let t_at = |x: usize, y: usize| -> Option<f64> {
        let t = sae[y * width + x];
        (t != i64::MIN).then_some(t as f64 * scale)
    };
    // Gradient at a fired pixel from central differences, needing both neighbours on each axis.
    let gradient = |x: usize, y: usize| -> Option<(f64, f64)> {
        t_at(x, y)?; // the pixel itself must have fired
        let gx = (t_at(x + 1, y)? - t_at(x - 1, y)?) / 2.0;
        let gy = (t_at(x, y + 1)? - t_at(x, y - 1)?) / 2.0;
        Some((gx, gy))
    };

    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    let mut samples = 0;
    for y in cy - HARRIS_RADIUS..=cy + HARRIS_RADIUS {
        for x in cx - HARRIS_RADIUS..=cx + HARRIS_RADIUS {
            if let Some((ix, iy)) = gradient(x, y) {
                sxx += ix * ix;
                syy += iy * iy;
                sxy += ix * iy;
                samples += 1;
            }
        }
    }
    if samples < HARRIS_MIN_SAMPLES {
        return f64::NEG_INFINITY;
    }
    let det = sxx * syy - sxy * sxy;
    let trace = sxx + syy;
    if trace <= 0.0 {
        return f64::NEG_INFINITY; // every gradient in the window was zero
    }
    det / (trace * trace) - HARRIS_K
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::{ring_is_corner, INNER_ARC, INNER_CIRCLE};
    use crate::EventStream;

    fn empty(width: usize, height: usize) -> EventStream {
        EventStream::from_array2(Array2::zeros((0, 4)), width, height, 0.001)
    }

    // A tiny SAE laid out so a 4-pixel recent arc appears on the 16-pixel inner ring.
    fn sae_with_arc(recent: &[usize]) -> Vec<i64> {
        // 7×7 grid, centre at (3,3); ring pixels default old, `recent` indices set newest.
        let width = 7;
        let mut sae = vec![0_i64; width * width];
        for (k, &(dx, dy)) in INNER_CIRCLE.iter().enumerate() {
            let x = (3 + dx) as usize;
            let y = (3 + dy) as usize;
            sae[y * width + x] = if recent.contains(&k) { 100 } else { 1 };
        }
        sae
    }

    #[test]
    fn ring_detects_contiguous_arc_and_rejects_scattered_or_long() {
        let width = 7;
        // All ring pixels equal → flat, no corner.
        let flat = sae_with_arc(&[]);
        assert!(!ring_is_corner(
            &flat,
            3,
            3,
            width,
            &INNER_CIRCLE,
            INNER_ARC
        ));
        // 4 contiguous newest pixels → corner.
        let corner = sae_with_arc(&[2, 3, 4, 5]);
        assert!(ring_is_corner(
            &corner,
            3,
            3,
            width,
            &INNER_CIRCLE,
            INNER_ARC
        ));
        // Half the ring newest (8 pixels) → straight edge, exceeds max arc.
        let edge = sae_with_arc(&[0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(!ring_is_corner(
            &edge,
            3,
            3,
            width,
            &INNER_CIRCLE,
            INNER_ARC
        ));
        // Two isolated newest pixels on opposite sides → not contiguous.
        let scattered = sae_with_arc(&[0, 8]);
        assert!(!ring_is_corner(
            &scattered,
            3,
            3,
            width,
            &INNER_CIRCLE,
            INNER_ARC
        ));
    }

    #[test]
    fn efast_empty_stream_is_empty() {
        assert_eq!(empty(20, 20).efast().len(), 0);
        assert_eq!(empty(0, 0).efast().len(), 0);
    }

    #[test]
    fn efast_and_harris_return_a_subset() {
        // A moving L-corner: a horizontal then vertical arm sweeping across time.
        let mut rows = Vec::new();
        let mut t = 0_u64;
        for x in 0..20u64 {
            rows.push([x, 10, t, 1]);
            t += 10;
        }
        for y in 0..20u64 {
            rows.push([10, y, t, 1]);
            t += 10;
        }
        let events = Array2::from_shape_vec((rows.len(), 4), rows.concat()).unwrap();
        let stream = EventStream::from_array2(events, 20, 20, 0.001);

        let corners = stream.efast();
        assert!(corners.len() <= stream.len());
        let harris = stream.harris_corners(0.0);
        assert!(harris.len() <= stream.len());
    }

    #[test]
    fn harris_empty_and_tiny_sensor_are_empty() {
        assert_eq!(empty(20, 20).harris_corners(0.0).len(), 0);
        // Sensor smaller than the Harris window → nothing evaluable.
        assert_eq!(
            EventStream::from_array2(array![[1, 1, 5, 1]], 4, 4, 0.001)
                .harris_corners(0.0)
                .len(),
            0
        );
    }

    /// An L-shaped corner sweeping across the sensor, as `(events, sensor_size)`.
    fn moving_corner() -> (Array2<u64>, usize) {
        let mut rows = Vec::new();
        let mut t = 0_u64;
        for step in 0..24u64 {
            let c = 6 + step;
            for offset in 0..10u64 {
                rows.push([c, c + offset, t, 1]); // vertical arm
                rows.push([c + offset, c, t, 1]); // horizontal arm
            }
            t += 100;
        }
        let len = rows.len();
        (Array2::from_shape_vec((len, 4), rows.concat()).unwrap(), 48)
    }

    /// The response is `det/trace² - k`, which is a ratio of two quantities that scale together —
    /// so re-expressing the same recording in different time units must not move the score. Before
    /// it was normalised, the raw `det - k·trace²` differed by `10¹²` between these two, and any
    /// non-zero threshold meant something different for each.
    #[test]
    fn harris_is_invariant_to_the_timestamp_unit() {
        let (events, size) = moving_corner();
        let in_ms = EventStream::from_array2(events.clone(), size, size, 1.0);
        // The same instants in microseconds: 1000× the raw ticks, each worth a thousandth as much.
        let in_us = EventStream::from_array2(events, size, size, 0.001).time_scale(1000.0);

        for threshold in [0.0, 0.05, 0.1] {
            assert_eq!(
                in_ms.harris_corners(threshold).len(),
                in_us.harris_corners(threshold).len(),
                "threshold {threshold} disagreed between ms and µs timestamps"
            );
        }

        // And the knob has to actually discriminate somewhere inside its range.
        let all = in_ms.harris_corners(-0.04).len();
        let some = in_ms.harris_corners(0.0).len();
        let none = in_ms.harris_corners(0.25).len();
        assert!(some > 0 && some < all, "got {some} of {all}");
        assert_eq!(none, 0);
    }

    /// A straight edge sweeping across the sensor is rank-1 on the SAE ramp (`R < 0`), so the
    /// `threshold = 0` Harris keeps almost nothing — the property that makes it a corner detector
    /// rather than an edge detector.
    #[test]
    fn harris_rejects_a_straight_moving_edge() {
        let mut rows = Vec::new();
        let mut t = 0_u64;
        for step in 0..30u64 {
            for y in 2..28u64 {
                rows.push([5 + step, y, t, 1]); // vertical edge column advancing in +x
            }
            t += 100;
        }
        let events = Array2::from_shape_vec((rows.len(), 4), rows.concat()).unwrap();
        let stream = EventStream::from_array2(events, 40, 30, 0.001);
        let kept = stream.harris_corners(0.0).len();
        // Far below 5% of the edge events survive (no true corner is present).
        assert!(
            kept * 20 < stream.len(),
            "straight edge should yield ~no corners, got {kept}/{}",
            stream.len()
        );
    }
}
