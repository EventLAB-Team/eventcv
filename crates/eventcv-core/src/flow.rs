//! Optical flow (OpenCV `video` analogue). **Lucas-Kanade on the time surface**: the Surface of
//! Active Events `T(x, y)` (the latest event time at each pixel, in milliseconds) is a local ramp
//! whose gradient encodes edge velocity — an edge moving with pixel velocity `v` satisfies
//! `∇T · v = 1`. We fit one constant `v` per pixel over a window by least squares, exactly like
//! image Lucas-Kanade, and return a two-channel `(flow_x, flow_y)` [`EventFrame`] in **pixels per
//! millisecond**. Where the window holds only a single edge the structure tensor is rank-deficient
//! (the aperture problem), so we fall back to **normal flow** along the gradient; genuinely flat,
//! event-free windows stay zero. Flow is emitted only at pixels that saw an event. Assumes
//! ascending time order.

use std::{error::Error, fmt};

use crate::representation::{EventFrame, EventFrameData, RepresentationKind};
use crate::EventStream;

/// Conditioning threshold: when `det(M) > COND_RATIO · trace(M)²` the structure tensor is well
/// enough conditioned (a corner) to recover full 2-D flow; below it the window is a single edge
/// (aperture problem) and we fall back to **normal flow** along the gradient. Scale-free, since
/// both `det` and `trace²` scale as the fourth power of the gradient.
const COND_RATIO: f64 = 1e-3;
/// `trace(M)` (summed squared gradient) below this is a flat, event-free window — no flow.
const FLAT_EPSILON: f64 = 1e-12;
/// Minimum gradient coherence (`|Σ∇T|² / (n·Σ|∇T|²)`, in `[0, 1]`) to trust a normal-flow estimate.
/// Below this the window's gradients disagree (noise), and normal flow would blow up.
const NORMAL_FLOW_MIN_COHERENCE: f64 = 0.35;

impl EventStream {
    /// Estimates dense optical flow by Lucas-Kanade on the time surface. `window` is the
    /// half-width of the least-squares neighbourhood (a `(2·window+1)²` patch); it must be at
    /// least 1. Returns a two-channel `f32` frame — channel 0 `flow_x`, channel 1 `flow_y`, in
    /// pixels/ms — zero wherever flow is undefined.
    pub fn optical_flow(&self, window: usize) -> Result<EventFrame, FlowError> {
        if window == 0 {
            return Err(FlowError::InvalidParameter("window"));
        }
        let (width, height) = self.sensor_size();
        let plane = width
            .checked_mul(height)
            .ok_or(FlowError::SizeOverflow)?
            .checked_mul(2)
            .ok_or(FlowError::SizeOverflow)?;
        let mut flow = vec![0.0_f32; plane];
        if width < 3 || height < 3 {
            return Ok(flow_frame(flow, width, height));
        }

        // Merge polarities: flow tracks edges regardless of contrast sign. `t_ms` holds the latest
        // event time per pixel in milliseconds; `NaN` marks pixels that never fired.
        let scale = self.timestamp_scale_ms();
        let (xs, ys, ts) = (self.xs(), self.ys(), self.ts());
        let mut latest = vec![i64::MIN; width * height];
        for index in 0..self.len() {
            let i = ys[index] as usize * width + xs[index] as usize;
            if ts[index] > latest[i] {
                latest[i] = ts[index];
            }
        }
        let t_ms: Vec<f64> = latest
            .iter()
            .map(|&t| {
                if t == i64::MIN {
                    f64::NAN
                } else {
                    t as f64 * scale
                }
            })
            .collect();

        // Central-difference gradients of the time surface, only where both neighbours fired.
        let plane_len = width * height;
        let (mut gx, mut gy) = (vec![f64::NAN; plane_len], vec![f64::NAN; plane_len]);
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let i = y * width + x;
                if t_ms[i].is_nan() {
                    continue;
                }
                let (l, r) = (t_ms[i - 1], t_ms[i + 1]);
                let (u, d) = (t_ms[i - width], t_ms[i + width]);
                if !l.is_nan() && !r.is_nan() {
                    gx[i] = (r - l) / 2.0;
                }
                if !u.is_nan() && !d.is_nan() {
                    gy[i] = (d - u) / 2.0;
                }
            }
        }

        let w = window as isize;
        for y in 0..height {
            for x in 0..width {
                if t_ms[y * width + x].is_nan() {
                    continue; // only emit flow at pixels that saw an event
                }
                let (mut sxx, mut syy, mut sxy, mut bx, mut by) = (0.0, 0.0, 0.0, 0.0, 0.0);
                let mut count = 0.0;
                for dy in -w..=w {
                    let ny = y as isize + dy;
                    if ny < 0 || ny >= height as isize {
                        continue;
                    }
                    for dx in -w..=w {
                        let nx = x as isize + dx;
                        if nx < 0 || nx >= width as isize {
                            continue;
                        }
                        let i = ny as usize * width + nx as usize;
                        let (ix, iy) = (gx[i], gy[i]);
                        if ix.is_nan() || iy.is_nan() {
                            continue;
                        }
                        // Constraint per sample: ix·u + iy·v = 1.
                        sxx += ix * ix;
                        syy += iy * iy;
                        sxy += ix * iy;
                        bx += ix;
                        by += iy;
                        count += 1.0;
                    }
                }
                let trace = sxx + syy;
                if trace <= FLAT_EPSILON {
                    continue; // flat window — no edge to track
                }
                let det = sxx * syy - sxy * sxy;
                let (u, v) = if det > COND_RATIO * trace * trace {
                    // Well-conditioned corner — full Lucas-Kanade.
                    ((syy * bx - sxy * by) / det, (sxx * by - sxy * bx) / det)
                } else {
                    // Single edge (aperture problem) — normal flow along the mean gradient. Guard
                    // against near-cancelling (incoherent) gradients: `coherence = |Σ∇T|² /
                    // (n·Σ|∇T|²)` is 1 for a clean edge and →0 for noise, and `‖normal flow‖ =
                    // 1/|mean ∇T|` blows up as the mean shrinks. Skip low-coherence windows so
                    // noisy patches don't emit huge spurious vectors.
                    let coherence = (bx * bx + by * by) / (count * trace);
                    if coherence < NORMAL_FLOW_MIN_COHERENCE {
                        continue;
                    }
                    let (gx_bar, gy_bar) = (bx / count, by / count);
                    let mag2 = gx_bar * gx_bar + gy_bar * gy_bar;
                    (gx_bar / mag2, gy_bar / mag2)
                };
                let i = y * width + x;
                flow[i] = u as f32;
                flow[plane_len + i] = v as f32;
            }
        }

        Ok(flow_frame(flow, width, height))
    }
}

fn flow_frame(flow: Vec<f32>, width: usize, height: usize) -> EventFrame {
    EventFrame::from_parts(
        EventFrameData::F32(flow),
        width,
        height,
        RepresentationKind::Flow,
        vec!["flow_x".to_owned(), "flow_y".to_owned()],
    )
}

#[derive(Debug, PartialEq, Eq)]
pub enum FlowError {
    SizeOverflow,
    InvalidParameter(&'static str),
}

impl fmt::Display for FlowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => formatter.write_str("flow field dimensions are too large"),
            Self::InvalidParameter("window") => formatter.write_str("window must be at least 1"),
            Self::InvalidParameter(name) => write!(formatter, "{name} is invalid"),
        }
    }
}

impl Error for FlowError {}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{EventFrameData, EventStream, FlowError};

    fn stream(rows: Vec<[u64; 4]>, width: usize, height: usize) -> EventStream {
        let flat: Vec<u64> = rows.iter().flatten().copied().collect();
        EventStream::from_array2(
            Array2::from_shape_vec((rows.len(), 4), flat).unwrap(),
            width,
            height,
            0.001,
        )
    }

    #[test]
    fn rejects_zero_window() {
        assert_eq!(
            stream(vec![], 8, 8).optical_flow(0).unwrap_err(),
            FlowError::InvalidParameter("window")
        );
    }

    #[test]
    fn empty_stream_is_all_zero_two_channel() {
        let frame = stream(vec![], 8, 8).optical_flow(2).unwrap();
        assert_eq!(frame.shape(), (2, 8, 8));
        let EventFrameData::F32(values) = frame.data() else {
            panic!("flow frames are float32");
        };
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn horizontal_edge_flow_points_along_x() {
        // A vertical bar sweeping left→right: column x fires at time 10·x, so T increases with x.
        // The edge moves in +x, so flow_x should be positive and flow_y ≈ 0.
        let mut rows = Vec::new();
        for x in 0..8u64 {
            for y in 0..8u64 {
                rows.push([x, y, 10 * x, 1]);
            }
        }
        let frame = stream(rows, 8, 8).optical_flow(2).unwrap();
        let EventFrameData::F32(values) = frame.data() else {
            panic!("flow frames are float32");
        };
        let plane = 8 * 8;
        // Sample an interior pixel with full support.
        let i = 4 * 8 + 4;
        let (fx, fy) = (values[i], values[plane + i]);
        // T = 10·x in raw units × 0.001 ms/unit ⇒ dT/dx = 0.01 ms/px ⇒ v_x = 100 px/ms.
        assert!(fx > 0.0, "flow_x should be positive, got {fx}");
        assert!(fy.abs() < fx.abs() * 0.1, "flow_y should be ~0, got {fy}");
    }
}
