use super::{
    event_index, frame_len, EventFrame, EventFrameData, Representation, RepresentationError,
    RepresentationKind,
};
use crate::EventStream;

/// Single-channel event-count image: the number of events at each pixel, both polarities
/// summed. Unlike [`super::Polarity`] (per-polarity planes) this collapses polarity into
/// one intensity map — the plainest "how much happened here" frame. Counts accumulate as
/// `u64`; `normalize` rescales the plane into `u8` for display.
#[derive(Clone, Copy, Debug, Default)]
pub struct EventCount {
    normalize: bool,
}

impl EventCount {
    pub fn new(normalize: bool) -> Self {
        Self { normalize }
    }

    pub fn is_normalized(&self) -> bool {
        self.normalize
    }
}

impl Representation for EventCount {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        let (width, height, length) = frame_len(stream, 1)?;
        let mut counts = vec![0_u64; length];

        for event in stream.iter() {
            counts[event_index(event, width, height)?] += 1;
        }

        let data = if self.normalize {
            EventFrameData::U8(normalize_u8(&counts))
        } else {
            EventFrameData::U64(counts)
        };

        Ok(EventFrame {
            data,
            channels: 1,
            width,
            height,
            kind: RepresentationKind::Count,
            channel_names: vec!["count".to_owned()],
        })
    }
}

/// Linearly rescales the counts so the busiest pixel maps to `u8::MAX` (rounded).
fn normalize_u8(counts: &[u64]) -> Vec<u8> {
    let maximum = counts.iter().copied().max().unwrap_or(0);
    if maximum == 0 {
        return vec![0; counts.len()];
    }
    counts
        .iter()
        .map(|&count| {
            let scaled = count * u64::from(u8::MAX);
            ((scaled + maximum / 2) / maximum) as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{EventCount, Representation};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn sums_both_polarities_into_one_plane() {
        let stream = EventStream::from_array2(
            array![[0, 0, 1, 1], [0, 0, 2, 0], [1, 1, 3, 1]],
            2,
            2,
            0.001,
        );

        let frame = EventCount::default().generate(&stream).unwrap();

        assert_eq!(frame.shape(), (1, 2, 2));
        assert_eq!(frame.data(), &EventFrameData::U64(vec![2, 0, 0, 1]));
    }

    #[test]
    fn normalizes_the_busiest_pixel_to_full_scale() {
        let stream = EventStream::from_array2(
            array![[0, 0, 1, 1], [0, 0, 2, 0], [1, 0, 3, 1]],
            2,
            1,
            0.001,
        );

        let frame = EventCount::new(true).generate(&stream).unwrap();

        assert_eq!(frame.data(), &EventFrameData::U8(vec![255, 128]));
    }

    #[test]
    fn rejects_out_of_bounds_events() {
        let stream = EventStream::from_array2(array![[2, 0, 10, 1]], 2, 2, 0.001);

        let error = EventCount::default().generate(&stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "event coordinate (2, 0) exceeds sensor size 2x2"
        );
    }
}
