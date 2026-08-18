use std::{error::Error, fmt};

use crate::{Event, EventStream};

mod averaged_time_surface;
mod binary;
mod count;
mod countmask;
mod mcts;
mod point_set;
mod polarity;
mod tencode;
mod time_surface;
mod voxel;

pub use averaged_time_surface::AveragedTimeSurface;
pub use binary::Binary;
pub use count::EventCount;
pub use countmask::CountMask;
pub use mcts::Mcts;
pub use point_set::{EventPointSet, PointSet};
pub use polarity::Polarity;
pub use tencode::Tencode;
pub use time_surface::TimeSurface;
pub use voxel::VoxelGrid;

pub trait Representation {
    type Output;

    /// The reference implementation, on the CPU. Always available, always the definition of what
    /// this representation *is*: the GPU kernels are checked against it, not the other way round.
    fn generate(&self, stream: &EventStream) -> Result<Self::Output, RepresentationError>;

    /// Runs on `device`.
    ///
    /// The default ignores it and runs [`generate`](Self::generate) — a representation with no GPU
    /// kernel is not an error to ask for on the GPU, it is simply computed on the CPU, and a
    /// pipeline that sets `device="gpu"` once should not have to know which of its steps have
    /// kernels. Asking for a GPU that does not *exist* is still an error; that is decided by the
    /// implementations that do have one.
    fn generate_on(
        &self,
        stream: &EventStream,
        _device: crate::accel::Device,
    ) -> Result<Self::Output, RepresentationError> {
        self.generate(stream)
    }
}

/// Runs a kernel over `stream`, or says why it could not.
///
/// Every GPU-capable representation funnels through here, so "no adapter", "this build has no GPU
/// support" and "an accumulator would have overflowed" are worded once. The `gpu` feature being
/// off collapses this to the error arm, which is why the body is not itself behind a `cfg`.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
pub(crate) fn on_gpu(
    stream: &EventStream,
    dispatch: &crate::accel::GpuDispatch,
) -> Result<Vec<i32>, RepresentationError> {
    #[cfg(feature = "gpu")]
    {
        crate::accel::gpu::with_context(|context| {
            crate::accel::gpu::run(context, stream, dispatch)
        })
        .ok_or_else(|| RepresentationError::Device(crate::accel::unavailable_reason()))?
        .map_err(|error| match error {
            crate::accel::gpu::GpuError::Saturated => RepresentationError::Device(
                "a GPU accumulator overflowed: the kernels sum in fixed point so that the result \
                 does not depend on the order the events arrived, and a cell of this frame exceeds \
                 what that can hold. Narrow the window, or use device=\"cpu\"."
                    .to_owned(),
            ),
            crate::accel::gpu::GpuError::Driver(message) => {
                RepresentationError::Device(format!("the GPU driver refused the work: {message}"))
            }
        })
    }
    #[cfg(not(feature = "gpu"))]
    {
        Err(RepresentationError::Device(
            crate::accel::unavailable_reason(),
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationKind {
    AveragedTimeSurface,
    Binary,
    Count,
    CountMask,
    Flow,
    /// A plain greyscale image — not derived from events at all. Produced by the frame readers and
    /// by reconstruction, and consumed by the simulator. Like `Flow` and `Labels` it has no
    /// `Representation` impl, because there is no stream to generate it from.
    Intensity,
    Labels,
    Mcts,
    Polarity,
    Tencode,
    TimeSurface,
    Voxel,
}

impl RepresentationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AveragedTimeSurface => "atsurf",
            Self::Binary => "binary",
            Self::Count => "count",
            Self::CountMask => "countmask",
            Self::Flow => "flow",
            Self::Intensity => "intensity",
            Self::Labels => "labels",
            Self::Mcts => "mcts",
            Self::Polarity => "polarity",
            Self::Tencode => "tencode",
            Self::TimeSurface => "tsurf",
            Self::Voxel => "voxel",
        }
    }

    /// The inverse of [`Self::as_str`] — recovers the kind tag stored by the frame writers.
    pub fn from_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "atsurf" => Self::AveragedTimeSurface,
            "binary" => Self::Binary,
            "count" => Self::Count,
            "countmask" => Self::CountMask,
            "flow" => Self::Flow,
            "intensity" => Self::Intensity,
            "labels" => Self::Labels,
            "mcts" => Self::Mcts,
            "polarity" => Self::Polarity,
            "tencode" => Self::Tencode,
            "tsurf" => Self::TimeSurface,
            "voxel" => Self::Voxel,
            _ => return None,
        })
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
    /// Reassembles a frame from its stored parts — used by the IO frame readers. Channels
    /// is `channel_names.len()`, and the data length must equal `channels * width * height`.
    pub(crate) fn from_parts(
        data: EventFrameData,
        width: usize,
        height: usize,
        kind: RepresentationKind,
        channel_names: Vec<String>,
    ) -> Self {
        Self {
            data,
            channels: channel_names.len(),
            width,
            height,
            kind,
            channel_names,
        }
    }

    /// Builds a single-channel greyscale frame from raw samples, row-major.
    ///
    /// The public counterpart to [`EventFrame::from_parts`], which is crate-private and validates
    /// nothing — so this is the only way anything outside the core (the bindings, the video decoder)
    /// can construct a frame, and it checks the length rather than trusting the caller. Everything
    /// downstream indexes `c * width * height + y * width + x` and would read out of bounds on a
    /// mismatch.
    pub fn intensity(
        data: EventFrameData,
        width: usize,
        height: usize,
    ) -> Result<Self, RepresentationError> {
        let expected = width
            .checked_mul(height)
            .ok_or(RepresentationError::SizeOverflow)?;
        let actual = match &data {
            EventFrameData::U8(values) => values.len(),
            EventFrameData::U16(values) => values.len(),
            EventFrameData::U64(values) => values.len(),
            EventFrameData::F32(values) => values.len(),
        };
        if actual != expected {
            return Err(RepresentationError::ShapeMismatch {
                samples: actual,
                width,
                height,
            });
        }
        Ok(Self::from_parts(
            data,
            width,
            height,
            RepresentationKind::Intensity,
            vec!["intensity".to_owned()],
        ))
    }

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

impl EventFrameData {
    /// Number of scalar elements (`channels * width * height` for a well-formed frame).
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::F32(values) => values.len(),
        }
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
    /// A frame's sample count does not match its dimensions. Caught at construction because every
    /// consumer indexes `c * width * height + y * width + x` and would read out of bounds.
    ShapeMismatch {
        samples: usize,
        width: usize,
        height: usize,
    },
    /// The requested device could not run this — no adapter, no GPU support in the build, or a
    /// kernel that could not represent the answer. Carries the sentence explaining which.
    Device(String),
}

impl fmt::Display for RepresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => formatter.write_str("representation dimensions are too large"),
            Self::CountOverflow { x, y } => {
                write!(
                    formatter,
                    "event count at ({x}, {y}) exceeds uint16 capacity"
                )
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
            Self::InvalidParameter(name) => match *name {
                "bins" => formatter.write_str("bins must be at least 1"),
                "max_window_ms" => {
                    formatter.write_str("max_window_ms must be finite and at least 1")
                }
                "pct" => formatter.write_str("pct must be between 0 and 100"),
                _ => write!(formatter, "{name} must be finite and positive"),
            },
            Self::ShapeMismatch {
                samples,
                width,
                height,
            } => write!(
                formatter,
                "frame has {samples} samples but {width}x{height} needs {}",
                width * height
            ),
            Self::Device(message) => formatter.write_str(message),
        }
    }
}

impl Error for RepresentationError {}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{
        AveragedTimeSurface, Binary, CountMask, EventCount, EventFrameData, Mcts, PointSet,
        Representation, RepresentationError, Tencode, TimeSurface, VoxelGrid,
    };
    use crate::EventStream;

    fn empty_stream(width: usize, height: usize) -> EventStream {
        EventStream::from_array2(Array2::zeros((0, 4)), width, height, 0.001)
    }

    #[test]
    fn empty_streams_produce_zero_outputs() {
        let stream = empty_stream(2, 3);

        for frame in [
            Binary.generate(&stream).unwrap(),
            EventCount::new(true).generate(&stream).unwrap(),
            VoxelGrid::default().generate(&stream).unwrap(),
            TimeSurface::default().generate(&stream).unwrap(),
            AveragedTimeSurface::default().generate(&stream).unwrap(),
            Tencode::default().generate(&stream).unwrap(),
            Mcts::default().generate(&stream).unwrap(),
            CountMask::default().generate(&stream).unwrap(),
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
        assert_eq!(
            CountMask::new(150.0, false).generate(&stream).unwrap_err(),
            RepresentationError::InvalidParameter("pct")
        );

        let oversized = empty_stream(usize::MAX, 2);
        assert_eq!(
            Binary.generate(&oversized).unwrap_err(),
            RepresentationError::SizeOverflow
        );
    }
}
