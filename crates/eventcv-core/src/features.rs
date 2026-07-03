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
    /// recent timestamps form a contiguous arc within the [`INNER_ARC`]/[`OUTER_ARC`] bounds — the
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

    /// **Harris corner score on the time surface.** Maintains a per-polarity SAE, and for each
    /// event forms a 9×9 exponential time-surface patch (`exp(-age/τ)`, `τ = tau_ms`) from that
    /// polarity's SAE, then computes the Harris response `det(M) - k·trace(M)²` of the local
    /// structure tensor `M`. Events whose response exceeds `threshold` are kept. `tau_ms` must be
    /// finite and positive (falls back to 30 ms otherwise). Returns the corner events as a new
    /// stream. A nice-to-have complement to [`Self::efast`] that yields a tunable score.
    pub fn harris_corners(&self, threshold: f64, tau_ms: f64) -> EventStream {
        let tau_ms = if tau_ms.is_finite() && tau_ms > 0.0 {
            tau_ms
        } else {
            30.0
        };
        let (width, height) = self.sensor_size();
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        let scale = self.timestamp_scale_ms();
        let mut builder =
            EventStreamBuilder::with_capacity(width, height, self.timestamp_scale_ms(), self.len());
        let margin = HARRIS_RADIUS + 1; // central differences need one pixel beyond the window
        if width < 2 * margin + 1 || height < 2 * margin + 1 {
            return builder.build();
        }
        let mut sae_on = vec![i64::MIN; width * height];
        let mut sae_off = vec![i64::MIN; width * height];

        for index in 0..self.len() {
            let (x, y, t, p) = (xs[index] as usize, ys[index] as usize, ts[index], ps[index]);
            let sae = if p { &mut sae_on } else { &mut sae_off };
            sae[y * width + x] = t;

            if x < margin || y < margin || x + margin >= width || y + margin >= height {
                continue;
            }
            if harris_response(sae, x, y, width, t, scale, tau_ms) > threshold {
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

/// Harris response of the local time surface. Builds the exponential surface `exp(-age/τ)` over the
/// `(2R+1)²` window around `(cx, cy)`, then sums the structure tensor of its central-difference
/// gradients and returns `det - k·trace²`.
fn harris_response(
    sae: &[i64],
    cx: usize,
    cy: usize,
    width: usize,
    now: i64,
    scale: f64,
    tau_ms: f64,
) -> f64 {
    let side = 2 * HARRIS_RADIUS + 1;
    let surface = |x: usize, y: usize| -> f64 {
        let t = sae[y * width + x];
        if t == i64::MIN {
            0.0
        } else {
            (-(now.saturating_sub(t) as f64) * scale / tau_ms).exp()
        }
    };
    // Sample the window plus a one-pixel border so central differences are defined everywhere.
    let mut patch = vec![0.0_f64; (side + 2) * (side + 2)];
    for j in 0..side + 2 {
        for i in 0..side + 2 {
            let x = cx + i - (HARRIS_RADIUS + 1);
            let y = cy + j - (HARRIS_RADIUS + 1);
            patch[j * (side + 2) + i] = surface(x, y);
        }
    }

    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for j in 1..=side {
        for i in 1..=side {
            let ix = (patch[j * (side + 2) + i + 1] - patch[j * (side + 2) + i - 1]) / 2.0;
            let iy = (patch[(j + 1) * (side + 2) + i] - patch[(j - 1) * (side + 2) + i]) / 2.0;
            sxx += ix * ix;
            syy += iy * iy;
            sxy += ix * iy;
        }
    }
    let det = sxx * syy - sxy * sxy;
    let trace = sxx + syy;
    det - HARRIS_K * trace * trace
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
        let harris = stream.harris_corners(0.0, 30.0);
        assert!(harris.len() <= stream.len());
    }

    #[test]
    fn harris_empty_and_tiny_sensor_are_empty() {
        assert_eq!(empty(20, 20).harris_corners(0.0, 30.0).len(), 0);
        // Sensor smaller than the Harris window → nothing evaluable.
        assert_eq!(
            EventStream::from_array2(array![[1, 1, 5, 1]], 4, 4, 0.001)
                .harris_corners(-1.0, 30.0)
                .len(),
            0
        );
    }
}
