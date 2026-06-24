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
                let lower = position.floor() as usize;
                let upper = position.ceil() as usize;
                let polarity = if event.polarity { 1.0 } else { -1.0 };
                if lower == upper {
                    values[lower * plane_len + spatial_index] += polarity;
                } else {
                    let upper_weight = (position - lower as f64) as f32;
                    values[lower * plane_len + spatial_index] += polarity * (1.0 - upper_weight);
                    values[upper * plane_len + spatial_index] += polarity * upper_weight;
                }
            }
        }

        Ok(EventFrame {
            data: EventFrameData::F32(values),
            channels: self.bins,
            width,
            height,
            kind: RepresentationKind::Voxel,
            channel_names: (0..self.bins).map(|bin| format!("bin_{bin}")).collect(),
        })
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
