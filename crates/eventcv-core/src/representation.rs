use std::{error::Error, fmt};

use crate::EventStream;

const POLARITY_CHANNELS: [&str; 2] = ["positive", "negative"];

pub trait Representation {
    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentationKind {
    Polarity,
}

#[derive(Clone, Debug)]
pub struct EventFrame {
    data: EventFrameData,
    channels: usize,
    width: usize,
    height: usize,
    kind: RepresentationKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventFrameData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl EventFrame {
    pub fn data(&self) -> &EventFrameData {
        &self.data
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    pub fn channel_names(&self) -> &'static [&'static str] {
        match self.kind {
            RepresentationKind::Polarity => &POLARITY_CHANNELS,
        }
    }

    pub fn kind(&self) -> RepresentationKind {
        self.kind
    }
}

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
    fn generate(&self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        let (width, height) = stream.sensor_size();
        let plane_len = width
            .checked_mul(height)
            .ok_or(RepresentationError::SizeOverflow)?;
        let frame_len = plane_len
            .checked_mul(POLARITY_CHANNELS.len())
            .ok_or(RepresentationError::SizeOverflow)?;
        let mut counts = vec![0_u16; frame_len];

        for event in stream.iter() {
            if event.x >= width || event.y >= height {
                return Err(RepresentationError::EventOutOfBounds {
                    x: event.x,
                    y: event.y,
                    width,
                    height,
                });
            }
            let channel_offset = if event.polarity { 0 } else { plane_len };
            let count = &mut counts[channel_offset + event.y * width + event.x];
            *count = count
                .checked_add(1)
                .ok_or(RepresentationError::CountOverflow {
                    x: event.x,
                    y: event.y,
                })?;
        }

        let data = if self.normalize {
            EventFrameData::U8(normalize_u8(counts))
        } else {
            EventFrameData::U16(counts)
        };

        Ok(EventFrame {
            data,
            channels: POLARITY_CHANNELS.len(),
            width,
            height,
            kind: RepresentationKind::Polarity,
        })
    }
}

fn normalize_u8(counts: Vec<u16>) -> Vec<u8> {
    let maximum = counts.iter().copied().max().unwrap_or(0);
    if maximum == 0 {
        return vec![0; counts.len()];
    }

    counts
        .into_iter()
        .map(|count| {
            let scaled = u32::from(count) * u32::from(u8::MAX);
            ((scaled + u32::from(maximum) / 2) / u32::from(maximum)) as u8
        })
        .collect()
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
        }
    }
}

impl Error for RepresentationError {}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, time::Instant};

    use ndarray::array;

    use super::{EventFrameData, Polarity, Representation};
    use crate::EventStream;

    #[test]
    fn accumulates_positive_and_negative_events() {
        let stream = EventStream {
            events: array![[1, 0, 10, 1], [1, 0, 11, 1], [0, 1, 12, 0]],
            width: 2,
            height: 2,
        };

        let frame = Polarity::default().generate(&stream).unwrap();

        assert_eq!(frame.shape(), (2, 2, 2));
        assert_eq!(frame.channel_names(), ["positive", "negative"]);
        assert_eq!(
            frame.data(),
            &EventFrameData::U16(vec![0, 2, 0, 0, 0, 0, 1, 0])
        );
    }

    #[test]
    fn normalizes_both_polarities_with_one_scale() {
        let stream = EventStream {
            events: array![[0, 0, 10, 1], [0, 0, 11, 1], [1, 0, 12, 0]],
            width: 2,
            height: 1,
        };

        let frame = Polarity::new(true).generate(&stream).unwrap();

        assert_eq!(
            frame.data(),
            &EventFrameData::U8(vec![255, 0, 0, 128])
        );
    }

    #[test]
    fn empty_stream_produces_an_empty_histogram() {
        let stream = EventStream {
            events: ndarray::Array2::zeros((0, 4)),
            width: 2,
            height: 2,
        };

        let frame = Polarity::default().generate(&stream).unwrap();

        assert_eq!(frame.data(), &EventFrameData::U16(vec![0; 8]));
    }

    #[test]
    fn rejects_out_of_bounds_events() {
        let stream = EventStream {
            events: array![[2, 0, 10, 1]],
            width: 2,
            height: 2,
        };

        let error = Polarity::default().generate(&stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "event coordinate (2, 0) exceeds sensor size 2x2"
        );
    }

    #[test]
    fn rejects_counts_that_exceed_uint16() {
        let event_count = usize::from(u16::MAX) + 1;
        let stream = EventStream {
            events: ndarray::Array2::from_shape_fn((event_count, 4), |(index, column)| {
                match column {
                    2 => index as u64,
                    3 => 1,
                    _ => 0,
                }
            }),
            width: 1,
            height: 1,
        };

        let error = Polarity::default().generate(&stream).unwrap_err();

        assert_eq!(
            error.to_string(),
            "event count at (0, 0) exceeds uint16 capacity"
        );
    }

    #[test]
    #[ignore = "manual performance benchmark"]
    fn benchmark_polarity_generation() {
        let sample_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/test/example.npz");
        let sample = crate::load(sample_path).unwrap();
        let sample_iterations = 1_000;
        black_box(Polarity::default().generate(&sample).unwrap());
        let sample_start = Instant::now();

        for _ in 0..sample_iterations {
            black_box(Polarity::default().generate(black_box(&sample)).unwrap());
        }

        eprintln!(
            "sample polarity generation: {:?} per frame",
            sample_start.elapsed() / sample_iterations
        );

        let normalized = Polarity::new(true);
        black_box(normalized.generate(&sample).unwrap());
        let normalized_start = Instant::now();

        for _ in 0..sample_iterations {
            black_box(normalized.generate(black_box(&sample)).unwrap());
        }

        eprintln!(
            "normalized sample polarity generation: {:?} per frame",
            normalized_start.elapsed() / sample_iterations
        );

        let event_count = 1_000_000;
        let stream = EventStream {
            events: ndarray::Array2::from_shape_fn((event_count, 4), |(index, column)| {
                match column {
                    0 => (index % 640) as u64,
                    1 => (index / 640 % 480) as u64,
                    2 => index as u64,
                    _ => (index % 2) as u64,
                }
            }),
            width: 640,
            height: 480,
        };
        let iterations = 10;
        black_box(Polarity::default().generate(&stream).unwrap());
        let start = Instant::now();

        for _ in 0..iterations {
            black_box(Polarity::default().generate(black_box(&stream)).unwrap());
        }

        let elapsed = start.elapsed();
        eprintln!(
            "one-million-event polarity generation: {:.1} million events/s",
            event_count as f64 * iterations as f64 / elapsed.as_secs_f64() / 1_000_000.0
        );
    }
}
