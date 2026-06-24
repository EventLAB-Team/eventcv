use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

const WINDOW_COUNT: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct Mcts {
    max_window_ms: f64,
}

impl Mcts {
    pub fn new(max_window_ms: f64) -> Self {
        Self { max_window_ms }
    }

    fn windows(&self) -> [f64; WINDOW_COUNT] {
        let ratio = self.max_window_ms.powf(1.0 / (WINDOW_COUNT - 1) as f64);
        std::array::from_fn(|index| ratio.powi(index as i32))
    }
}

impl Default for Mcts {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Representation for Mcts {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        validate_positive(self.max_window_ms, "max_window_ms")?;
        if self.max_window_ms < 1.0 {
            return Err(RepresentationError::InvalidParameter("max_window_ms"));
        }
        let (width, height, length) = frame_len(stream, WINDOW_COUNT * 2)?;
        let plane_len = width * height;
        let windows = self.windows();
        let mut values = vec![0_f32; length];

        if let Some(reference) = reference_time(stream) {
            for event in stream.iter() {
                let index = event_index(event, width, height)?;
                let age = age_ms(stream, reference, event.timestamp);
                let channel_offset = if event.polarity { WINDOW_COUNT } else { 0 };
                for (window_index, window) in windows.iter().copied().enumerate() {
                    if age <= window {
                        let value = (1.0 - age / window) as f32;
                        let output_index = (channel_offset + window_index) * plane_len + index;
                        values[output_index] = values[output_index].max(value);
                    }
                }
            }
        }

        let channel_names = ["negative", "positive"]
            .into_iter()
            .flat_map(|polarity| {
                windows
                    .iter()
                    .map(move |window| format!("{polarity}_{window:.3}ms"))
            })
            .collect();

        Ok(EventFrame {
            data: EventFrameData::F32(values),
            channels: WINDOW_COUNT * 2,
            width,
            height,
            kind: RepresentationKind::Mcts,
            channel_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Mcts, Representation};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn builds_logarithmic_windows_for_each_polarity() {
        let stream =
            EventStream::from_array2(array![[0, 0, 14_000, 0], [0, 0, 16_000, 1]], 1, 1, 0.001);

        let frame = Mcts::new(16.0).generate(&stream).unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::F32(vec![0.0, 0.0, 0.5, 0.75, 0.875, 1.0, 1.0, 1.0, 1.0, 1.0])
        );
    }
}
