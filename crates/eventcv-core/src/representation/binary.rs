use super::{
    event_index, frame_len, EventFrame, EventFrameData, Representation, RepresentationError,
    RepresentationKind,
};
use crate::EventStream;

#[derive(Clone, Copy, Debug, Default)]
pub struct Binary;

impl Representation for Binary {
    type Output = EventFrame;

    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        let (width, height, length) = frame_len(stream, 1)?;
        let mut occupancy = vec![0_u8; length];
        for event in stream.iter() {
            occupancy[event_index(event, width, height)?] = 1;
        }

        Ok(EventFrame {
            data: EventFrameData::U8(occupancy),
            channels: 1,
            width,
            height,
            kind: RepresentationKind::Binary,
            channel_names: vec!["event".to_owned()],
        })
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{Binary, Representation};
    use crate::{representation::EventFrameData, EventStream};

    #[test]
    fn records_single_channel_occupancy() {
        let stream = EventStream {
            events: array![[0, 0, 1, 1], [0, 0, 2, 0], [1, 1, 3, 1]],
            width: 2,
            height: 2,
            timestamp_scale_ms: 0.001,
        };

        let frame = Binary.generate(&stream).unwrap();

        assert_eq!(frame.shape(), (1, 2, 2));
        assert_eq!(frame.data(), &EventFrameData::U8(vec![1, 0, 0, 1]));
    }
}
