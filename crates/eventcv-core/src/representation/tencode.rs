use super::{
    age_ms, event_index, frame_len, reference_time, validate_positive, EventFrame, EventFrameData,
    Representation, RepresentationError, RepresentationKind,
};
use crate::EventStream;

/// Per-pixel winner states for the scratch column: no event in the window, or the polarity of the
/// latest one.
const NOTHING: u8 = 0;
const POSITIVE: u8 = 1;
const NEGATIVE: u8 = 2;

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
        // Per-pixel winner, split into two lean columns rather than one `Vec<Option<(u64, usize,
        // bool)>>`: at 1280×720 that packing zeroed 22 MB of scratch per frame, which dominated the
        // cost of a live window. `state` doubles as the "seen" flag the `Option` used to carry.
        let mut latest_t = vec![0_u64; plane_len];
        let mut latest_state = vec![NOTHING; plane_len];
        let mut values = vec![0_u8; length];

        if let Some(reference) = reference_time(stream) {
            for event in stream.iter() {
                if age_ms(stream, reference, event.timestamp) > self.window_ms {
                    continue;
                }
                let index = event_index(event, width, height)?;
                // Events are visited in stream order, so `>=` keeps the last event of a timestamp
                // tie — the same winner an explicit order comparison picked.
                if latest_state[index] == NOTHING || event.timestamp >= latest_t[index] {
                    latest_t[index] = event.timestamp;
                    latest_state[index] = if event.polarity { POSITIVE } else { NEGATIVE };
                }
            }

            for index in 0..plane_len {
                match latest_state[index] {
                    POSITIVE => values[index] = u8::MAX,
                    NEGATIVE => values[2 * plane_len + index] = u8::MAX,
                    _ => continue,
                }
                values[plane_len + index] = (255.0
                    * age_ms(stream, reference, latest_t[index])
                    / self.window_ms)
                    .round()
                    .clamp(0.0, 255.0) as u8;
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
        let stream = EventStream::from_array2(
            array![[0, 0, 10_000, 1], [1, 0, 30_000, 1], [1, 0, 30_000, 0]],
            2,
            1,
            0.001,
        );

        let frame = Tencode::new(30.0).generate(&stream).unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::U8(vec![255, 0, 170, 0, 0, 255])
        );
    }
}
