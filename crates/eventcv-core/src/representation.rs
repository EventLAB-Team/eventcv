use std::{error::Error, fmt};

use crate::{Event, EventStream};

mod binary;
mod mcts;
mod point_set;
mod polarity;
mod tencode;
mod time_surface;
mod voxel;

pub use binary::Binary;
pub use mcts::Mcts;
pub use point_set::{EventPointSet, PointSet};
pub use polarity::Polarity;
pub use tencode::Tencode;
pub use time_surface::TimeSurface;
pub use voxel::VoxelGrid;

pub trait Representation {
    type Output;

    fn generate(&self, stream: &EventStream) -> Result<Self::Output, RepresentationError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationKind {
    Binary,
    Mcts,
    Polarity,
    Tencode,
    TimeSurface,
    Voxel,
}

impl RepresentationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Mcts => "mcts",
            Self::Polarity => "polarity",
            Self::Tencode => "tencode",
            Self::TimeSurface => "tsurf",
            Self::Voxel => "voxel",
        }
    }
}

#[derive(Clone, Debug)]
pub struct EventFrame {
    pub(crate) data: EventFrameData,
    pub(crate) channels: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) kind: RepresentationKind,
    pub(crate) channel_names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EventFrameData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U64(Vec<u64>),
    F32(Vec<f32>),
}

impl EventFrame {
    pub fn data(&self) -> &EventFrameData {
        &self.data
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }

    pub fn kind(&self) -> RepresentationKind {
        self.kind
    }
}

pub(crate) fn frame_len(
    stream: &EventStream,
    channels: usize,
) -> Result<(usize, usize, usize), RepresentationError> {
    let (width, height) = stream.sensor_size();
    let plane_len = width
        .checked_mul(height)
        .ok_or(RepresentationError::SizeOverflow)?;
    let length = plane_len
        .checked_mul(channels)
        .ok_or(RepresentationError::SizeOverflow)?;
    Ok((width, height, length))
}

pub(crate) fn event_index(
    event: Event,
    width: usize,
    height: usize,
) -> Result<usize, RepresentationError> {
    if event.x >= width || event.y >= height {
        return Err(RepresentationError::EventOutOfBounds {
            x: event.x,
            y: event.y,
            width,
            height,
        });
    }
    Ok(event.y * width + event.x)
}

pub(crate) fn validate_positive(value: f64, name: &'static str) -> Result<(), RepresentationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RepresentationError::InvalidParameter(name));
    }
    Ok(())
}

pub(crate) fn reference_time(stream: &EventStream) -> Option<u64> {
    stream.iter().map(|event| event.timestamp).max()
}

pub(crate) fn age_ms(stream: &EventStream, reference: u64, timestamp: u64) -> f64 {
    reference.saturating_sub(timestamp) as f64 * stream.timestamp_scale_ms()
}

#[derive(Debug, PartialEq, Eq)]
pub enum RepresentationError {
    SizeOverflow,
    CountOverflow {
        x: usize,
        y: usize,
    },
    EventOutOfBounds {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    },
    InvalidParameter(&'static str),
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => formatter.write_str("representation dimensions are too large"),
            Self::CountOverflow { x, y } => {
                write!(formatter, "event count at ({x}, {y}) exceeds uint16 capacity")
            }
            Self::EventOutOfBounds {
                x,
                y,
                width,
                height,
            } => write!(
                formatter,
                "event coordinate ({x}, {y}) exceeds sensor size {width}x{height}"
            ),
            Self::InvalidParameter(name) => {
                match *name {
                    "bins" => formatter.write_str("bins must be at least 1"),
                    "max_window_ms" => {
                        formatter.write_str("max_window_ms must be finite and at least 1")
                    }
                    _ => write!(formatter, "{name} must be finite and positive"),
                }
            }
        }
    }
}

impl Error for RepresentationError {}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{
        Binary, EventFrameData, Mcts, PointSet, Representation, RepresentationError, Tencode,
        TimeSurface, VoxelGrid,
    };
    use crate::EventStream;

    fn empty_stream(width: usize, height: usize) -> EventStream {
        EventStream {
            events: Array2::zeros((0, 4)),
            width,
            height,
            timestamp_scale_ms: 0.001,
        }
    }

    #[test]
    fn empty_streams_produce_zero_outputs() {
        let stream = empty_stream(2, 3);

        for frame in [
            Binary.generate(&stream).unwrap(),
            VoxelGrid::default().generate(&stream).unwrap(),
            TimeSurface::default().generate(&stream).unwrap(),
            Tencode::default().generate(&stream).unwrap(),
            Mcts::default().generate(&stream).unwrap(),
        ] {
            match frame.data() {
                EventFrameData::U8(values) => assert!(values.iter().all(|&value| value == 0)),
                EventFrameData::F32(values) => assert!(values.iter().all(|&value| value == 0.0)),
                _ => panic!("unexpected empty representation dtype"),
            }
        }
        assert_eq!(PointSet.generate(&stream).unwrap().shape(), (0, 4));
    }

    #[test]
    fn rejects_invalid_parameters_and_size_overflow() {
        let stream = empty_stream(2, 3);

        assert_eq!(
            VoxelGrid::new(0, 30.0).generate(&stream).unwrap_err(),
            RepresentationError::InvalidParameter("bins")
        );
        assert_eq!(
            TimeSurface::new(f64::NAN).generate(&stream).unwrap_err(),
            RepresentationError::InvalidParameter("tau_ms")
        );
        assert_eq!(
            Tencode::new(0.0).generate(&stream).unwrap_err(),
            RepresentationError::InvalidParameter("window_ms")
        );
        assert_eq!(
            Mcts::new(0.5).generate(&stream).unwrap_err(),
            RepresentationError::InvalidParameter("max_window_ms")
        );

        let oversized = empty_stream(usize::MAX, 2);
        assert_eq!(
            Binary.generate(&oversized).unwrap_err(),
            RepresentationError::SizeOverflow
        );
    }
}
