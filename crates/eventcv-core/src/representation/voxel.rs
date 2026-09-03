use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

#[derive(Clone, Copy, Debug)]
pub struct VoxelGrid {
    bins: usize,
    window_ms: f64,
}

impl VoxelGrid {
    pub fn new(bins: usize, window_ms: f64) -> Self {
        Self { bins, window_ms }
    }
}

impl Default for VoxelGrid {
    fn default() -> Self {
        Self::new(9, 30.0)
    }
}

impl Representation for VoxelGrid {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        if self.bins == 0 {
            return Err(RepresentationError::InvalidParameter("bins"));
        }
        validate_positive(self.window_ms, "window_ms")?;
        let (width, height, length) = frame_len(stream, self.bins)?;
        let plane_len = width * height;
        let mut values = vec![0_f32; length];

        if let Some(reference) = reference_time(stream) {
            for event in stream.iter() {
                let age = age_ms(stream, reference, event.timestamp);
                if age > self.window_ms {
                    continue;
                }
                let spatial_index = event_index(event, width, height)?;
                let position = if self.bins == 1 {
                    0.0
                } else {
                    (1.0 - age / self.window_ms) * (self.bins - 1) as f64
                };
                // Truncation rather than `floor` + `ceil`, which is the same answer for a
                // fraction of the cost. `age` is filtered to `[0, window_ms]` just above, so
                // `position` lands in `[0, bins - 1]` and never goes negative --- and for a
                // non-negative float `as usize` *is* the floor. The fraction then says
                // everything the second call did: it is zero exactly when floor and ceil
                // agreed, and when it is not, the upper bin is the next one up. That leaves
                // one saturating float-to-int conversion per event where there were two,
                // which measures at 1.5-1.8x over the whole kernel on a 1280x720 grid.
                let lower = position as usize;
                let upper_weight = position - lower as f64;
                let polarity = if event.polarity { 1.0 } else { -1.0 };
                if upper_weight == 0.0 {
                    values[lower * plane_len + spatial_index] += polarity;
                } else {
                    // `position` is not an integer here, so it is strictly below `bins - 1`
                    // and `lower + 1` is in range.
                    let upper_weight = upper_weight as f32;
                    values[lower * plane_len + spatial_index] += polarity * (1.0 - upper_weight);
                    values[(lower + 1) * plane_len + spatial_index] += polarity * upper_weight;
                }
            }
        }

        Ok(self.frame(values, width, height))
    }

    /// The GPU twin scatters the same signed, linearly split contributions, but accumulates them in
    /// Q16.16 integers rather than `f32`. That makes the result independent of the order the
    /// invocations landed — where the CPU's float sum is order-dependent in its last bits — at the
    /// cost of a per-event quantisation below `f32`'s own resolution here. The two agree to about
    /// 1e-4 on a cell, which `tests/test_gpu.py` pins.
    fn generate_on(
        &self,
        stream: &EventStream,
        device: crate::accel::Device,
    ) -> Result<EventFrame, RepresentationError> {
        if device == crate::accel::Device::Cpu {
            return self.generate(stream);
        }
        if self.bins == 0 {
            return Err(RepresentationError::InvalidParameter("bins"));
        }
        validate_positive(self.window_ms, "window_ms")?;
        let (width, height, length) = frame_len(stream, self.bins)?;
        let cells = super::on_gpu(
            stream,
            &crate::accel::GpuDispatch {
                entry: "voxel",
                cells: length,
                initial: 0,
                bins: self.bins as u32,
                span_ms: self.window_ms as f32,
                fixed_one: crate::accel::FIXED_ONE,
                window_ms: Some(self.window_ms),
                needs_ages: true,
            },
        )?;
        let values = cells
            .iter()
            .map(|cell| *cell as f32 / crate::accel::FIXED_ONE)
            .collect();
        Ok(self.frame(values, width, height))
    }
}

impl VoxelGrid {
    fn frame(&self, values: Vec<f32>, width: usize, height: usize) -> EventFrame {
        EventFrame {
            data: EventFrameData::F32(values),
            channels: self.bins,
            width,
            height,
            kind: RepresentationKind::Voxel,
            channel_names: (0..self.bins).map(|bin| format!("bin_{bin}")).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Representation, VoxelGrid};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn interpolates_signed_events_between_time_bins() {
        let stream = EventStream::from_array2(
            array![[0, 0, 0, 1], [0, 0, 15_000, 0], [1, 0, 30_000, 1]],
            2,
            1,
            0.001,
        );

        let frame = VoxelGrid::new(3, 30.0).generate(&stream).unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::F32(vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn equal_timestamps_use_the_final_bin() {
        let stream = EventStream::from_array2(array![[0, 0, 10, 1], [0, 0, 10, 1]], 1, 1, 0.001);

        let frame = VoxelGrid::new(3, 30.0).generate(&stream).unwrap();

        assert_eq!(frame.data(), &EventFrameData::F32(vec![0.0, 0.0, 2.0]));
    }

    #[test]
    fn opposing_events_cancel_in_the_same_bins() {
        let stream = EventStream::from_array2(array![[0, 0, 10, 1], [0, 0, 10, 0]], 1, 1, 0.001);

        let frame = VoxelGrid::new(3, 30.0).generate(&stream).unwrap();

        assert_eq!(frame.data(), &EventFrameData::F32(vec![0.0; 3]));
    }
}
