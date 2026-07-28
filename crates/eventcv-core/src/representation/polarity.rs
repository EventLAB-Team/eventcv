use super::{
    event_index, frame_len, EventFrame, EventFrameData, Representation, RepresentationError,
    RepresentationKind,
};
use crate::EventStream;

#[derive(Clone, Copy, Debug, Default)]
pub struct Polarity {
    normalize: bool,
}

impl Polarity {
    pub fn new(normalize: bool) -> Self {
        Self { normalize }
    }

    pub fn is_normalized(&self) -> bool {
        self.normalize
    }
}

impl Representation for Polarity {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        let (width, height, counts) = polarity_counts(stream)?;

        let data = if self.normalize {
            EventFrameData::U8(normalize_u8(counts))
        } else {
            EventFrameData::U64(counts)
        };

        Ok(EventFrame {
            data,
            channels: 2,
            width,
            height,
            kind: RepresentationKind::Polarity,
            channel_names: vec!["positive".to_owned(), "negative".to_owned()],
        })
    }
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
    use ndarray::array;

    use super::{Polarity, Representation};
    use crate::{representation::EventFrameData, EventStream};

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
