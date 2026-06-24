use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

#[derive(Clone, Copy, Debug)]
pub struct TimeSurface {
    tau_ms: f64,
}

impl TimeSurface {
    pub fn new(tau_ms: f64) -> Self {
        Self { tau_ms }
    }
}

impl Default for TimeSurface {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Representation for TimeSurface {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        validate_positive(self.tau_ms, "tau_ms")?;
        let (width, height, length) = frame_len(stream, 2)?;
        let plane_len = width * height;
        let mut latest = vec![None; length];

        for event in stream.iter() {
            let index =
                event_index(event, width, height)? + if event.polarity { 0 } else { plane_len };
            if latest[index].is_none_or(|timestamp| event.timestamp > timestamp) {
                latest[index] = Some(event.timestamp);
            }
        }

        let mut values = vec![0_f32; length];
        if let Some(reference) = reference_time(stream) {
            for (value, timestamp) in values.iter_mut().zip(latest) {
                if let Some(timestamp) = timestamp {
                    *value = (-age_ms(stream, reference, timestamp) / self.tau_ms).exp() as f32;
                }
            }
        }

        Ok(EventFrame {
            data: EventFrameData::F32(values),
            channels: 2,
            width,
            height,
            kind: RepresentationKind::TimeSurface,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Representation, TimeSurface};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn uses_latest_event_per_pixel_and_polarity() {
        let stream = EventStream::from_array2(
            array![[0, 0, 30_000, 1], [1, 0, 20_000, 0], [0, 0, 10_000, 1]],
            2,
            1,
            0.001,
        );

        let frame = TimeSurface::new(10.0).generate(&stream).unwrap();
        let EventFrameData::F32(values) = frame.data() else {
            panic!("time surfaces must use float32 data");
        };

        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        assert!((values[3] - (-1.0_f32).exp()).abs() < 1e-6);
    }
}
