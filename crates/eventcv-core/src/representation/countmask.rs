use super::{
    frame_len, polarity::polarity_counts, EventFrame, EventFrameData, Representation,
    RepresentationError, RepresentationKind,
};
use crate::EventStream;

/// Count-mask image (GEPT, Sec. 3.2 "Event Accumulation", Eq. 2): a three-channel RGB encoding
/// where red and blue hold the per-pixel positive and negative event counts and green is a binary
/// activity mask — full scale wherever an event of either polarity landed. Timestamps are not
/// used at all, which is what separates it from [`Tencode`](super::Tencode) (latest polarity plus
/// event age) and from [`EventCount`](super::EventCount) (one plane of raw totals).
///
/// Both count planes are clipped and rescaled by a **single** `alpha`: the `pct`-th percentile of
/// the non-zero counts of the two planes **pooled together**. A per-channel percentile would let a
/// quiet polarity saturate against a busy one, so the joint scale is what keeps red and blue
/// comparable.
///
/// `white_frame` inverts the image to a white background. The black-background default is the form
/// downstream descriptor models are trained on; the inverted one is offered for parity with the
/// reference renderer, not for feeding a model.
#[derive(Clone, Copy, Debug)]
pub struct CountMask {
    pct: f64,
    white_frame: bool,
}

impl CountMask {
    pub fn new(pct: f64, white_frame: bool) -> Self {
        Self { pct, white_frame }
    }

    /// The clip bound and divisor for both count planes, as `f32` — see [`percentile_linear`] for
    /// why the percentile itself is computed in `f64` and narrowed only at the end.
    fn alpha(&self, counts: &[u64]) -> f32 {
        // Counts convert to `f32` exactly below 2^24, so pooling them through `f64` here matches
        // the reference's float32 accumulator for any physically plausible slice. It would only
        // diverge if a single pixel saw more than 16.7M events in one slice.
        let mut nonzero: Vec<f64> = counts
            .iter()
            .copied()
            .filter(|&count| count > 0)
            .map(|count| count as f64)
            .collect();
        if nonzero.is_empty() {
            return 1.0;
        }

        let alpha = percentile_linear(&mut nonzero, self.pct);
        if alpha > 0.0 {
            alpha as f32
        } else {
            // Unreachable while every pooled sample is at least 1, but kept so the scale can never
            // become zero and blow up the division below.
            counts.iter().copied().max().unwrap_or(0).max(1) as f32
        }
    }
}

impl Default for CountMask {
    fn default() -> Self {
        Self::new(99.0, false)
    }
}

impl Representation for CountMask {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        if !self.pct.is_finite() || !(0.0..=100.0).contains(&self.pct) {
            return Err(RepresentationError::InvalidParameter("pct"));
        }
        let (width, height, length) = frame_len(stream, 3)?;
        let plane_len = width * height;
        let (_, _, counts) = polarity_counts(stream)?;

        let alpha = self.alpha(&counts);
        // Everything from here on runs in `f32`, matching the reference NumPy pipeline: a Python
        // float scalar never upcasts a float32 array, so the clip, the divide and the `* 255` are
        // all single precision. Computing them in `f64` shifts one grey level on ~0.02% of alphas.
        let level = |count: u64| {
            let value = (count as f32).min(alpha) / alpha;
            if self.white_frame {
                1.0 - value
            } else {
                value
            }
        };
        // `as u8` truncates toward zero, which is what `.astype(np.uint8)` does — a count of 1 at
        // `alpha = 2` is 127.5 and must land on 127, not 128.
        let byte = |value: f32| (value * 255.0).clamp(0.0, 255.0) as u8;

        let mut values = vec![0_u8; length];
        for index in 0..plane_len {
            let positive = counts[index];
            let negative = counts[plane_len + index];
            let active = f32::from(u8::from(positive + negative > 0));
            values[index] = byte(level(positive));
            values[plane_len + index] = byte(if self.white_frame { 1.0 - active } else { active });
            values[2 * plane_len + index] = byte(level(negative));
        }

        Ok(EventFrame {
            data: EventFrameData::U8(values),
            channels: 3,
            width,
            height,
            kind: RepresentationKind::CountMask,
            channel_names: vec![
                "positive".to_owned(),
                "activity".to_owned(),
                "negative".to_owned(),
            ],
        })
    }
}

/// NumPy's default `linear` percentile (`np.percentile(values, pct)`), reproduced exactly.
///
/// Byte-for-byte agreement with the reference renderer depends on two details that a
/// "close enough" percentile gets wrong. The first is interpolation: the result sits *between*
/// two order statistics, so `percentile([1, 2, 3, 4, 7], 99)` is `6.88` — nearest-rank would say
/// `7`. The second is the endpoint switch in NumPy's `_lerp`, which the `gamma >= 0.5` branch
/// below mirrors; the naive `lower + delta * gamma` form disagrees on roughly 1 input in 50 000.
///
/// Reorders `values` in place (two partial selections rather than a full sort).
fn percentile_linear(values: &mut [f64], pct: f64) -> f64 {
    let last = values.len() - 1;
    let virtual_index = (pct / 100.0) * last as f64;

    if virtual_index >= last as f64 {
        // NumPy pins both order statistics to the maximum here, making the interpolation a no-op.
        let (_, maximum, _) = values.select_nth_unstable_by(last, f64::total_cmp);
        return *maximum;
    }

    let gamma = virtual_index - virtual_index.floor();
    let previous = virtual_index.floor() as usize;
    let (_, lower, rest) = values.select_nth_unstable_by(previous, f64::total_cmp);
    // Everything after `previous` is `>= lower`, so its minimum is the next order statistic.
    let (lower, upper) = (*lower, rest.iter().copied().fold(f64::INFINITY, f64::min));
    let delta = upper - lower;

    if gamma >= 0.5 {
        upper - delta * (1.0 - gamma)
    } else {
        lower + delta * gamma
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::{CountMask, Representation};
    use crate::{
        representation::{EventFrameData, RepresentationError},
        EventStream,
    };

    /// 40 pseudo-random events on a 6x8 sensor, from the reference renderer's
    /// `np.random.default_rng(12345)` fixture. Every non-zero count here is 1 or 2 and
    /// `alpha == 2`, so a count of 1 must land on **127** — the rounding-vs-truncation check.
    fn golden_stream() -> EventStream {
        EventStream::from_array2(
            array![
                [5, 4, 0, 1],
                [1, 0, 1, 0],
                [6, 2, 2, 0],
                [2, 0, 3, 0],
                [1, 4, 4, 1],
                [6, 2, 5, 0],
                [5, 2, 6, 1],
                [5, 2, 7, 0],
                [7, 2, 8, 1],
                [3, 1, 9, 0],
                [6, 3, 10, 1],
                [2, 4, 11, 0],
                [4, 2, 12, 1],
                [4, 1, 13, 0],
                [1, 0, 14, 1],
                [1, 0, 15, 0],
                [1, 0, 16, 1],
                [5, 0, 17, 1],
                [4, 0, 18, 0],
                [7, 3, 19, 0],
                [5, 4, 20, 0],
                [1, 5, 21, 1],
                [7, 3, 22, 1],
                [7, 3, 23, 0],
                [5, 1, 24, 0],
                [5, 5, 25, 1],
                [1, 3, 26, 0],
                [0, 4, 27, 1],
                [2, 4, 28, 1],
                [3, 5, 29, 0],
                [0, 4, 30, 1],
                [7, 5, 31, 1],
                [3, 3, 32, 0],
                [5, 3, 33, 0],
                [1, 1, 34, 0],
                [2, 5, 35, 0],
                [0, 3, 36, 0],
                [5, 2, 37, 1],
                [6, 1, 38, 1],
                [1, 1, 39, 1],
            ],
            8,
            6,
            0.001,
        )
    }

    #[rustfmt::skip]
    const GOLDEN: [u8; 144] = [
        // red — positive counts, clipped and normalised by alpha = 2
        0, 255,   0,   0,   0, 127,   0,   0,
        0, 127,   0,   0,   0,   0, 127,   0,
        0,   0,   0,   0, 127, 255,   0, 127,
        0,   0,   0,   0,   0,   0, 127, 127,
      255, 127, 127,   0,   0, 127,   0,   0,
        0, 127,   0,   0,   0, 127,   0, 127,
        // green — binary activity mask, either polarity
        0, 255, 255,   0, 255, 255,   0,   0,
        0, 255,   0, 255, 255, 255, 255,   0,
        0,   0,   0,   0, 255, 255, 255, 255,
      255, 255,   0, 255,   0, 255, 255, 255,
      255, 255, 255,   0,   0, 255,   0,   0,
        0, 255, 255, 255,   0, 255,   0, 255,
        // blue — negative counts, same alpha
        0, 255, 127,   0, 127,   0,   0,   0,
        0, 127,   0, 127, 127, 127,   0,   0,
        0,   0,   0,   0,   0, 127, 255,   0,
      127, 127,   0, 127,   0, 127,   0, 255,
        0,   0, 127,   0,   0, 127,   0,   0,
        0,   0, 127, 127,   0,   0,   0,   0,
    ];

    #[test]
    fn matches_the_reference_renderer() {
        let frame = CountMask::default().generate(&golden_stream()).unwrap();

        assert_eq!(frame.shape(), (3, 6, 8));
        assert_eq!(frame.data(), &EventFrameData::U8(GOLDEN.to_vec()));
    }

    /// Timestamps are not part of the encoding, so rescaling them must not move a single byte.
    #[test]
    fn ignores_timestamps() {
        let mut rows = golden_stream().to_array2();
        rows.column_mut(2).map_inplace(|t| *t = *t * 1_000_000 + 7);
        let shifted = EventStream::from_array2(rows, 8, 6, 0.001);

        let frame = CountMask::default().generate(&shifted).unwrap();

        assert_eq!(frame.data(), &EventFrameData::U8(GOLDEN.to_vec()));
    }

    /// `alpha` pools the non-zero counts of **both** planes before taking the percentile. Here the
    /// pooled 99th percentile is 11.8, while a per-channel one would be 1.0 for red (turning its
    /// three pixels into 255 instead of 21) and 11.92 for blue. Including the zero-valued pixels
    /// would give 11.56, which is different again. The fixture also exercises a fractional alpha
    /// and NumPy's `gamma >= 0.5` interpolation branch, neither of which the 6x8 golden reaches.
    #[test]
    fn normalizes_both_planes_by_one_pooled_percentile() {
        // Positive: one event each at x = 0, 1, 2. Negative: 4 at x = 3, 8 at x = 4, 12 at x = 5.
        let mut rows = Vec::new();
        let pixels = [(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 4, 0), (4, 8, 0), (5, 12, 0)];
        for (x, count, polarity) in pixels {
            for _ in 0..count {
                let timestamp = rows.len() as u64;
                rows.push([x, 0, timestamp, polarity]);
            }
        }
        let stream = EventStream::from_array2(
            Array2::from_shape_fn((rows.len(), 4), |(row, column)| rows[row][column]),
            6,
            1,
            0.001,
        );

        let frame = CountMask::default().generate(&stream).unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::U8(vec![
                21, 21, 21, 0, 0, 0, // red: 1 / 11.8 -> 21, not 255
                255, 255, 255, 255, 255, 255, // green: every pixel saw an event
                0, 0, 0, 86, 172, 255, // blue: 8 / 11.8 * 255 = 172.88 -> 172
            ])
        );

        let inverted = CountMask::new(99.0, true).generate(&stream).unwrap();

        assert_eq!(
            inverted.data(),
            &EventFrameData::U8(vec![
                233, 233, 233, 255, 255, 255, //
                0, 0, 0, 0, 0, 0, //
                255, 255, 255, 168, 82, 0,
            ])
        );
    }

    #[test]
    fn rejects_out_of_bounds_events() {
        let stream = EventStream::from_array2(array![[2, 0, 10, 1]], 2, 2, 0.001);

        let error = CountMask::default().generate(&stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "event coordinate (2, 0) exceeds sensor size 2x2"
        );
    }

    #[test]
    fn rejects_percentiles_outside_zero_to_one_hundred() {
        let stream = golden_stream();

        for pct in [-1.0, 100.5, f64::NAN] {
            assert_eq!(
                CountMask::new(pct, false).generate(&stream).unwrap_err(),
                RepresentationError::InvalidParameter("pct")
            );
        }
        assert_eq!(
            CountMask::new(150.0, false)
                .generate(&stream)
                .unwrap_err()
                .to_string(),
            "pct must be between 0 and 100"
        );
    }
}
