use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

const WINDOW_COUNT: usize = 5;

#[derive(Clone, Debug)]
pub struct Mcts {
    windows: Windows,
}

#[derive(Clone, Debug)]
enum Windows {
    /// `WINDOW_COUNT` log-spaced windows from 1 ms up to the max (the default form).
    LogSpaced { max_window_ms: f64 },
    /// Caller-supplied windows (ms), used in the order given.
    Explicit(Vec<f64>),
}

impl Mcts {
    pub fn new(max_window_ms: f64) -> Self {
        Self {
            windows: Windows::LogSpaced { max_window_ms },
        }
    }

    pub fn with_windows(windows_ms: Vec<f64>) -> Self {
        Self {
            windows: Windows::Explicit(windows_ms),
        }
    }

    fn windows(&self) -> Result<Vec<f64>, RepresentationError> {
        match &self.windows {
            Windows::LogSpaced { max_window_ms } => {
                validate_positive(*max_window_ms, "max_window_ms")?;
                if *max_window_ms < 1.0 {
                    return Err(RepresentationError::InvalidParameter("max_window_ms"));
                }
                let ratio = max_window_ms.powf(1.0 / (WINDOW_COUNT - 1) as f64);
                Ok((0..WINDOW_COUNT)
                    .map(|index| ratio.powi(index as i32))
                    .collect())
            }
            Windows::Explicit(windows) => {
                if windows.is_empty()
                    || windows
                        .iter()
                        .any(|window| !window.is_finite() || *window <= 0.0)
                {
                    return Err(RepresentationError::InvalidParameter("windows"));
                }
                Ok(windows.clone())
            }
        }
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
        let windows = self.windows()?;
        let (width, height, length) = frame_len(stream, windows.len() * 2)?;
        let plane_len = width * height;
        let mut values = vec![0_f32; length];

        if let Some(reference) = reference_time(stream) {
            for event in stream.iter() {
                let index = event_index(event, width, height)?;
                let age = age_ms(stream, reference, event.timestamp);
                let channel_offset = if event.polarity { windows.len() } else { 0 };
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
            channels: windows.len() * 2,
            width,
            height,
            kind: RepresentationKind::Mcts,
            channel_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::{Mcts, Representation};
    use crate::{
        representation::{EventFrameData, RepresentationError},
        EventStream,
    };

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

    #[test]
    fn builds_explicit_windows_in_the_order_given() {
        let stream =
            EventStream::from_array2(array![[0, 0, 14_000, 0], [0, 0, 16_000, 1]], 1, 1, 0.001);

        let frame = Mcts::with_windows(vec![4.0, 8.0])
            .generate(&stream)
            .unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::F32(vec![0.5, 0.75, 1.0, 1.0])
        );
        assert_eq!(
            frame.channel_names(),
            &[
                "negative_4.000ms",
                "negative_8.000ms",
                "positive_4.000ms",
                "positive_8.000ms"
            ]
        );
    }

    #[test]
    fn rejects_invalid_explicit_windows() {
        let stream = EventStream::from_array2(Array2::zeros((0, 4)), 1, 1, 0.001);
        for windows in [vec![], vec![0.0], vec![-1.0], vec![4.0, f64::NAN]] {
            assert!(matches!(
                Mcts::with_windows(windows).generate(&stream),
                Err(RepresentationError::InvalidParameter("windows"))
            ));
        }
    }

    #[test]
    fn explicit_windows_on_an_empty_stream_produce_zeros() {
        let stream = EventStream::from_array2(Array2::zeros((0, 4)), 2, 1, 0.001);

        let frame = Mcts::with_windows(vec![1.0, 5.0, 20.0])
            .generate(&stream)
            .unwrap();

        assert_eq!(frame.shape(), (6, 1, 2));
        assert_eq!(frame.data(), &EventFrameData::F32(vec![0.0; 12]));
    }
}
