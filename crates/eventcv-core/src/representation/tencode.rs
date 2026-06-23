use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

#[derive(Clone, Copy, Debug)]
pub struct Tencode {
    window_ms: f64,
}

impl Tencode {
    pub fn new(window_ms: f64) -> Self {
        Self { window_ms }
    }
}

impl Default for Tencode {
    fn default() -> Self {
        Self::new(30.0)
    }
}

impl Representation for Tencode {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        validate_positive(self.window_ms, "window_ms")?;
        let (width, height, length) = frame_len(stream, 3)?;
        let plane_len = width * height;
        let mut latest = vec![None; plane_len];
        let mut values = vec![0_u8; length];

        if let Some(reference) = reference_time(stream) {
            for (order, event) in stream.iter().enumerate() {
                if age_ms(stream, reference, event.timestamp) > self.window_ms {
                    continue;
                }
                let index = event_index(event, width, height)?;
                if latest[index].is_none_or(|(timestamp, previous_order, _)| {
                    event.timestamp > timestamp
                        || (event.timestamp == timestamp && order > previous_order)
                }) {
                    latest[index] = Some((event.timestamp, order, event.polarity));
                }
            }

            for (index, event) in latest.into_iter().enumerate() {
                if let Some((timestamp, _, polarity)) = event {
                    if polarity {
                        values[index] = u8::MAX;
                    } else {
                        values[2 * plane_len + index] = u8::MAX;
                    }
                    values[plane_len + index] =
                        (255.0 * age_ms(stream, reference, timestamp) / self.window_ms)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(EventFrame {
            data: EventFrameData::U8(values),
            channels: 3,
            width,
            height,
            kind: RepresentationKind::Tencode,
            channel_names: vec![
                "positive".to_owned(),
                "age".to_owned(),
                "negative".to_owned(),
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Representation, Tencode};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn encodes_latest_polarity_and_age() {
        let stream = EventStream {
            events: array![[0, 0, 10_000, 1], [1, 0, 30_000, 1], [1, 0, 30_000, 0]],
            width: 2,
            height: 1,
            timestamp_scale_ms: 0.001,
        };

        let frame = Tencode::new(30.0).generate(&stream).unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::U8(vec![255, 0, 170, 0, 0, 255])
        );
    }
}
