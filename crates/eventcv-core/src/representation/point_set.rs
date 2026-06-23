use super::{event_index, Representation, RepresentationError};
use crate::EventStream;

#[derive(Clone, Copy, Debug, Default)]
pub struct PointSet;

#[derive(Clone, Debug, PartialEq)]
pub struct EventPointSet {
    data: Vec<f32>,
    len: usize,
}

impl EventPointSet {
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.len, 4)
    }

    pub fn columns(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        ("x", "y", "t", "p")
    }
}

impl Representation for PointSet {
    type Output = EventPointSet;

    fn generate(&self, stream: &EventStream) -> Result<EventPointSet, RepresentationError> {
        let (width, height) = stream.sensor_size();
        let capacity = stream
            .len()
            .checked_mul(4)
            .ok_or(RepresentationError::SizeOverflow)?;
        let mut data = Vec::with_capacity(capacity);
        let minimum_time = stream.iter().map(|event| event.timestamp).min().unwrap_or(0);
        let maximum_time = stream.iter().map(|event| event.timestamp).max().unwrap_or(0);
        let duration = maximum_time - minimum_time;

        for event in stream.iter() {
            event_index(event, width, height)?;
            data.extend([
                normalize_coordinate(event.x, width),
                normalize_coordinate(event.y, height),
                if duration == 0 {
                    0.0
                } else {
                    (event.timestamp - minimum_time) as f32 / duration as f32
                },
                if event.polarity { 1.0 } else { -1.0 },
            ]);
        }

        Ok(EventPointSet {
            data,
            len: stream.len(),
        })
    }
}

fn normalize_coordinate(coordinate: usize, length: usize) -> f32 {
    if length <= 1 {
        0.0
    } else {
        coordinate as f32 / (length - 1) as f32
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{PointSet, Representation};
    use crate::EventStream;

    #[test]
    fn normalizes_points_without_reordering_events() {
        let stream = EventStream {
            events: array![[1, 0, 30, 0], [0, 2, 10, 1], [1, 1, 20, 1]],
            width: 2,
            height: 3,
            timestamp_scale_ms: 0.001,
        };

        let points = PointSet.generate(&stream).unwrap();

        assert_eq!(points.shape(), (3, 4));
        assert_eq!(
            points.data(),
            &[1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5, 1.0]
        );
    }

    #[test]
    fn zero_duration_streams_have_zero_normalized_time() {
        let stream = EventStream {
            events: array![[0, 0, 10, 1], [1, 0, 10, 0]],
            width: 2,
            height: 1,
            timestamp_scale_ms: 0.001,
        };

        let points = PointSet.generate(&stream).unwrap();

        assert_eq!(points.data()[2], 0.0);
        assert_eq!(points.data()[6], 0.0);
    }
}
