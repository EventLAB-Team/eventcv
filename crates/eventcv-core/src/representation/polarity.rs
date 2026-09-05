use super::{
    event_index, frame_len, EventFrame, EventFrameData, Representation, RepresentationError,
    RepresentationKind,
};
use crate::io::EventConsumer;
use crate::EventStream;

#[derive(Clone, Copy, Debug)]
pub struct Polarity {
    normalize: bool,
    downsample: usize,
}

impl Default for Polarity {
    fn default() -> Self {
        Self::new(false)
    }
}

impl Polarity {
    pub fn new(normalize: bool) -> Self {
        Self {
            normalize,
            downsample: 1,
        }
    }

    /// Bins `factor`×`factor` sensor pixels into one cell (`x / factor`, `y / factor`), so the
    /// frame is `ceil(width / factor)` × `ceil(height / factor)`. The same integers as
    /// `stream.resize(width / factor, height / factor)` followed by the full-resolution
    /// histogram, without building the resized stream. `1` (the default) is the sensor grid.
    pub fn with_downsample(mut self, factor: usize) -> Self {
        self.downsample = factor.max(1);
        self
    }

    pub fn is_normalized(&self) -> bool {
        self.normalize
    }

    pub fn downsample(&self) -> usize {
        self.downsample
    }
}

/// Per-pixel, per-polarity counts accumulated one event at a time — the [`EventConsumer`] behind
/// [`Polarity`], so a lazy reader can feed it straight from its decoder
/// (`SliceSource::slice_time_into`) instead of building an `EventStream` first.
///
/// Cells are `(x / factor, y / factor)` on a `ceil(width / factor)` × `ceil(height / factor)`
/// grid; the positive plane comes first, matching [`Polarity`]'s frame layout exactly.
#[derive(Clone, Debug)]
pub struct PolarityAccumulator {
    counts: Vec<u64>,
    width: usize,
    height: usize,
    plane: usize,
    factor: usize,
    /// `Some(bits)` when `factor` is a power of two, so the division is a shift.
    shift: Option<u32>,
}

impl PolarityAccumulator {
    /// An empty accumulator for a `sensor_width` × `sensor_height` sensor binned by `factor`.
    pub fn new(
        sensor_width: usize,
        sensor_height: usize,
        factor: usize,
    ) -> Result<Self, RepresentationError> {
        let factor = factor.max(1);
        let width = sensor_width.div_ceil(factor);
        let height = sensor_height.div_ceil(factor);
        let plane = width
            .checked_mul(height)
            .ok_or(RepresentationError::SizeOverflow)?;
        let length = plane
            .checked_mul(2)
            .ok_or(RepresentationError::SizeOverflow)?;
        Ok(Self {
            counts: vec![0; length],
            width,
            height,
            plane,
            factor,
            shift: factor.is_power_of_two().then(|| factor.trailing_zeros()),
        })
    }

    /// Zeroes the counts so the buffer can take the next window.
    pub fn reset(&mut self) {
        self.counts.iter_mut().for_each(|count| *count = 0);
    }

    /// The accumulated frame, raw `u64` counts or `u8` rescaled to the busiest cell.
    pub fn into_frame(self, normalize: bool) -> EventFrame {
        EventFrame {
            data: if normalize {
                EventFrameData::U8(normalize_u8(self.counts))
            } else {
                EventFrameData::U64(self.counts)
            },
            channels: 2,
            width: self.width,
            height: self.height,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        }
    }
}

impl EventConsumer for PolarityAccumulator {
    #[inline]
    fn push(&mut self, x: u16, y: u16, _t: i64, p: bool) {
        let (cx, cy) = match self.shift {
            Some(shift) => (usize::from(x) >> shift, usize::from(y) >> shift),
            None => (usize::from(x) / self.factor, usize::from(y) / self.factor),
        };
        let offset = if p { 0 } else { self.plane };
        self.counts[offset + cy * self.width + cx] += 1;
    }
}

impl Representation for Polarity {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        self.generate_on(stream, crate::accel::Device::Cpu)
    }

    fn generate_on(
        &self,
        stream: &EventStream,
        device: crate::accel::Device,
    ) -> Result<EventFrame, RepresentationError> {
        if self.downsample > 1 {
            // Binned frames go through the accumulator (CPU): the same integers a resized
            // stream would give, in one pass and without the resized stream.
            let (width, height) = stream.sensor_size();
            let mut accumulator = PolarityAccumulator::new(width, height, self.downsample)?;
            for event in stream.iter() {
                event_index(event, width, height)?;
                accumulator.push(event.x as u16, event.y as u16, event.timestamp as i64, event.polarity);
            }
            return Ok(accumulator.into_frame(self.normalize));
        }
        let (width, height, counts) = polarity_counts_on(stream, device)?;
        Ok(EventFrame {
            data: if self.normalize {
                EventFrameData::U8(normalize_u8(counts))
            } else {
                EventFrameData::U64(counts)
            },
            channels: 2,
            width,
            height,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        })
    }
}

/// [`polarity_counts`] on `device`. Counting is integer work, so the kernel's answer is not merely
/// close to the CPU's — it is the same numbers, whatever order the invocations landed in.
///
/// Both [`Polarity`] and the count-mask image are built from these two planes, so one kernel gives
/// both a GPU path.
pub(super) fn polarity_counts_on(
    stream: &EventStream,
    device: crate::accel::Device,
) -> Result<(usize, usize, Vec<u64>), RepresentationError> {
    if device == crate::accel::Device::Cpu {
        return polarity_counts(stream);
    }
    let (width, height, length) = frame_len(stream, 2)?;
    let cells = super::on_gpu(
        stream,
        &crate::accel::GpuDispatch {
            entry: "polarity_counts",
            cells: length,
            initial: 0,
            bins: 2,
            span_ms: 0.0,
            fixed_one: 1.0,
            window_ms: None,
            needs_ages: false,
        },
    )?;
    Ok((width, height, cells.iter().map(|cell| *cell as u64).collect()))
}

/// Per-pixel event counts split into a positive plane (offset `0`) and a negative one (offset
/// `width * height`), row-major within each. Shared with the count-mask representation, which
/// needs the same two planes before it normalises them differently.
///
/// Counts are accumulated as `u64`: a hot pixel on a multi-second recording (or a busy slice of
/// one) easily exceeds `u16` — see TASKS.md known issues.
pub(super) fn polarity_counts(
    stream: &EventStream,
) -> Result<(usize, usize, Vec<u64>), RepresentationError> {
    let (width, height, frame_len) = frame_len(stream, 2)?;
    let plane_len = width * height;
    let mut counts = vec![0_u64; frame_len];

    for event in stream.iter() {
        let index = event_index(event, width, height)?;
        let channel_offset = if event.polarity { 0 } else { plane_len };
        counts[channel_offset + index] += 1;
    }

    Ok((width, height, counts))
}

fn normalize_u8(counts: Vec<u64>) -> Vec<u8> {
    let maximum = counts.iter().copied().max().unwrap_or(0);
    if maximum == 0 {
        return vec![0; counts.len()];
    }

    counts
        .into_iter()
        .map(|count| {
            let scaled = count * u64::from(u8::MAX);
            ((scaled + maximum / 2) / maximum) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::PolarityAccumulator;
    use crate::io::EventConsumer;
    use ndarray::array;

    use super::{Polarity, Representation};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn downsample_matches_resize_then_histogram() {
        // 8x6 sensor, factor 2 -> 4x3; events on every pixel plus a pile-up, both polarities
        let mut rows = Vec::new();
        for y in 0..6u64 {
            for x in 0..8u64 {
                rows.push([x, y, x + 8 * y, (x + y) % 2]);
            }
        }
        rows.extend([[7, 5, 100, 1], [7, 5, 101, 1], [6, 4, 102, 0]]);
        let stream = EventStream::from_array2(
            ndarray::Array2::from_shape_vec((rows.len(), 4), rows.concat()).unwrap(), 8, 6, 0.001,
        );
        let fused = Polarity::new(false).with_downsample(2).generate(&stream).unwrap();
        let two_pass = Polarity::new(false).generate(&stream.resize(4, 3)).unwrap();
        assert_eq!(fused.shape(), (2, 3, 4));
        assert_eq!(fused.data(), two_pass.data());
        // and the sink path gives the same frame as the stream path
        let mut acc = PolarityAccumulator::new(8, 6, 2).unwrap();
        for event in stream.iter() {
            acc.push(event.x as u16, event.y as u16, event.timestamp as i64, event.polarity);
        }
        assert_eq!(acc.into_frame(false).data(), fused.data());
        // non-integer ratio: 8x6 by 3 -> 3x2 (ceil), every event still lands in the frame
        let odd = Polarity::new(false).with_downsample(3).generate(&stream).unwrap();
        assert_eq!(odd.shape(), (2, 2, 3));
        assert_eq!(
            match odd.data() { EventFrameData::U64(v) => v.iter().sum::<u64>(), _ => 0 },
            rows.len() as u64
        );
    }

    #[test]
    fn accumulates_and_normalizes_polarities() {
        let stream = EventStream::from_array2(
            array![[0, 0, 10, 1], [0, 0, 11, 1], [1, 0, 12, 0]],
            2,
            1,
            0.001,
        );

        let raw = Polarity::default().generate(&stream).unwrap();
        let normalized = Polarity::new(true).generate(&stream).unwrap();

        assert_eq!(raw.data(), &EventFrameData::U64(vec![2, 0, 0, 1]));
        assert_eq!(normalized.data(), &EventFrameData::U8(vec![255, 0, 0, 128]));
    }

    #[test]
    fn rejects_out_of_bounds_events() {
        let stream = EventStream::from_array2(array![[2, 0, 10, 1]], 2, 2, 0.001);

        let error = Polarity::default().generate(&stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "event coordinate (2, 0) exceeds sensor size 2x2"
        );
    }

    #[test]
    fn counts_above_uint16_are_preserved() {
        let event_count = usize::from(u16::MAX) + 1; // 65536 — would overflow a u16 counter
        let stream = EventStream::from_array2(
            ndarray::Array2::from_shape_fn((event_count, 4), |(index, column)| match column {
                2 => index as u64,
                3 => 1,
                _ => 0,
            }),
            1,
            1,
            0.001,
        );

        let raw = Polarity::default().generate(&stream).unwrap();

        assert_eq!(
            raw.data(),
            &EventFrameData::U64(vec![event_count as u64, 0])
        );
    }
}
