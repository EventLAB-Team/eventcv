use std::{error::Error, fmt, io, path::Path};

use crate::{EventStream, EventStreamBuilder};

mod bag;
#[cfg(feature = "hdf5")]
mod h5;
mod npz;
mod text;

pub use bag::read_bag;
#[cfg(feature = "hdf5")]
pub use h5::{open_hdf5_slice, read_hdf5, Hdf5SliceSource};
pub use npz::read_npz;
pub use text::{read_text, ColumnOrder, TextOptions, TextReader, TimeUnit};

/// A single event as produced by a reader, before it is placed on the sensor grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawEvent {
    pub x: u16,
    pub y: u16,
    pub t: i64,
    pub p: bool,
}

/// A bounded-memory source of events. Readers parse on demand so multi-gigabyte
/// files never need to be resident; the consumer decides what to accumulate.
pub trait EventSource {
    fn sensor_size(&self) -> (usize, usize);
    fn timestamp_scale_ms(&self) -> f64;
    /// Returns the next event, or `None` at end of stream.
    fn next_event(&mut self) -> Result<Option<RawEvent>, IoError>;
}

/// Drains a source into an in-memory [`EventStream`], dropping out-of-bounds events.
pub fn read_all(source: impl EventSource) -> Result<EventStream, IoError> {
    read_capped(source, None)
}

/// Like [`read_all`] but stops after `max` kept events (for previewing huge files).
pub fn read_capped(
    mut source: impl EventSource,
    max: Option<usize>,
) -> Result<EventStream, IoError> {
    let (width, height) = source.sensor_size();
    let mut builder = EventStreamBuilder::new(width, height, source.timestamp_scale_ms());
    while let Some(event) = source.next_event()? {
        builder.push(event.x, event.y, event.t, event.p);
        if max.is_some_and(|max| builder.len() >= max) {
            break;
        }
    }
    Ok(builder.build())
}

/// Options for the unified [`load`] entry point. Most fields apply to a single
/// format; readers ignore the ones they do not need.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// `(width, height)`. Required for text files; an optional override elsewhere.
    pub sensor_size: Option<(usize, usize)>,
    /// Text timestamp unit.
    pub time_unit: TimeUnit,
    /// Text column order.
    pub order: ColumnOrder,
    /// Rosbag topic to read (defaults to `/davis/left/events`).
    pub topic: Option<String>,
    /// Cap on the number of events to read.
    pub max_events: Option<usize>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            sensor_size: None,
            time_unit: TimeUnit::Seconds,
            order: ColumnOrder::Txyp,
            topic: None,
            max_events: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Npz,
    Text,
    Hdf5,
    Rosbag,
    Aedat4,
    Prophesee,
}

fn detect_format(path: &Path) -> Result<Format, IoError> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("npz") => Ok(Format::Npz),
        Some("txt") | Some("csv") => Ok(Format::Text),
        Some("h5") | Some("hdf5") => Ok(Format::Hdf5),
        Some("bag") => Ok(Format::Rosbag),
        Some("aedat4") | Some("aedat") => Ok(Format::Aedat4),
        Some("dat") | Some("raw") => Ok(Format::Prophesee),
        Some(other) => Err(IoError::Unsupported(format!(
            "unrecognised file extension: .{other}"
        ))),
        None => Err(IoError::Unsupported(
            "file has no extension to detect its format".to_owned(),
        )),
    }
}

/// Loads events from any supported file, detected by extension — the OpenCV-style
/// single entry point. Supported today: `.npz`, `.txt`/`.csv`, `.bag`.
pub fn load(path: impl AsRef<Path>, options: LoadOptions) -> Result<EventStream, IoError> {
    let path = path.as_ref();
    match detect_format(path)? {
        Format::Npz => npz::read_npz(path, options.sensor_size),
        Format::Text => {
            let (width, height) = options.sensor_size.ok_or_else(|| {
                IoError::Format("text files require sensor_size = (width, height)".to_owned())
            })?;
            let text_options = TextOptions {
                width,
                height,
                time_unit: options.time_unit,
                order: options.order,
            };
            read_capped(text::open(path, text_options)?, options.max_events)
        }
        Format::Rosbag => bag::read_bag(path, &options),
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                h5::read_hdf5(path, &options)
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        Format::Aedat4 => Err(IoError::Unsupported(
            "AEDAT4 (.aedat4) reading is not implemented yet".to_owned(),
        )),
        Format::Prophesee => Err(IoError::Unsupported(
            "Prophesee (.dat/.raw) reading is not implemented yet".to_owned(),
        )),
    }
}

/// Random-access view over a file's events: fetch an arbitrary time or count range
/// without materialising the whole stream. This backs the lazy [`open`] handle — the
/// OpenCV `VideoCapture` to [`load`]'s `imread`. Each call returns a new [`EventStream`].
pub trait SliceSource: Send {
    fn sensor_size(&self) -> (usize, usize);
    fn timestamp_scale_ms(&self) -> f64;
    /// Total events in the file.
    fn n_events(&self) -> usize;
    /// `(t_min, t_max)` in microseconds across the whole file; `(0, 0)` when empty.
    fn time_span(&self) -> (i64, i64);
    /// Events whose index lies in `[i0, i1)` (clamped to the file).
    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError>;
    /// Events whose timestamp (µs) lies in the half-open window `[t0, t1)`.
    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError>;
}

/// A [`SliceSource`] backed by an already-loaded stream — the universal fallback for
/// formats without native random access (npz/txt/bag today). Slicing is entirely
/// in-RAM, so `open()` returns a working handle for every supported format from day one.
pub struct MemorySliceSource {
    stream: EventStream,
}

impl MemorySliceSource {
    pub fn new(stream: EventStream) -> Self {
        Self { stream }
    }

    fn rebuild(&self, indices: impl Iterator<Item = usize>) -> EventStream {
        let (width, height) = self.stream.sensor_size();
        let mut builder = EventStreamBuilder::new(width, height, self.stream.timestamp_scale_ms());
        let (xs, ys, ts, ps) = (
            self.stream.xs(),
            self.stream.ys(),
            self.stream.ts(),
            self.stream.ps(),
        );
        for index in indices {
            builder.push(xs[index], ys[index], ts[index], ps[index]);
        }
        builder.build()
    }
}

impl SliceSource for MemorySliceSource {
    fn sensor_size(&self) -> (usize, usize) {
        self.stream.sensor_size()
    }

    fn timestamp_scale_ms(&self) -> f64 {
        self.stream.timestamp_scale_ms()
    }

    fn n_events(&self) -> usize {
        self.stream.len()
    }

    fn time_span(&self) -> (i64, i64) {
        let ts = self.stream.ts();
        match (ts.iter().min(), ts.iter().max()) {
            (Some(&min), Some(&max)) => (min, max),
            _ => (0, 0),
        }
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let i0 = i0.min(self.stream.len());
        let i1 = i1.clamp(i0, self.stream.len());
        Ok(self.rebuild(i0..i1))
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let ts = self.stream.ts();
        Ok(self.rebuild((0..ts.len()).filter(|&index| ts[index] >= t0 && ts[index] < t1)))
    }
}

/// A boxed [`SliceSource`] — the handle [`open`] returns.
pub type Reader = Box<dyn SliceSource>;

/// Opens a file for lazy slicing, detected by extension (the `VideoCapture` analogue to
/// [`load`]'s `imread`). HDF5 is sliced in place by binary-searching its timestamp
/// dataset; every other format is loaded once and sliced in memory. Same `LoadOptions`
/// as [`load`] (`max_events` is ignored — slicing supersedes it).
pub fn open(path: impl AsRef<Path>, options: LoadOptions) -> Result<Reader, IoError> {
    let path = path.as_ref();
    match detect_format(path)? {
        Format::Hdf5 => {
            #[cfg(feature = "hdf5")]
            {
                Ok(Box::new(h5::open_hdf5_slice(path, &options)?))
            }
            #[cfg(not(feature = "hdf5"))]
            {
                Err(IoError::Unsupported(
                    "HDF5 support is not built in; rebuild with --features hdf5".to_owned(),
                ))
            }
        }
        _ => Ok(Box::new(MemorySliceSource::new(load(path, options)?))),
    }
}

#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    Parse { line: usize, message: String },
    Format(String),
    InvalidSensorSize,
    Unsupported(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Parse { line, message } => write!(formatter, "line {line}: {message}"),
            Self::Format(message) => formatter.write_str(message),
            Self::InvalidSensorSize => {
                formatter.write_str("sensor width and height must be positive")
            }
            Self::Unsupported(message) => formatter.write_str(message),
        }
    }
}

impl Error for IoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for IoError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(test)]
mod tests {
    use super::{load, open, IoError, LoadOptions, MemorySliceSource, SliceSource};
    use crate::EventStreamBuilder;

    #[test]
    fn unknown_extension_is_unsupported() {
        let error = load("recording.mp4", LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Unsupported(_)));
    }

    #[test]
    fn hdf5_extension_dispatches_per_feature() {
        // `.h5` is always recognised; with the feature it reaches the reader (which
        // needs sensor_size), without it the dispatch reports missing support.
        let error = load("recording.h5", LoadOptions::default()).unwrap_err();
        #[cfg(feature = "hdf5")]
        assert!(matches!(error, IoError::Format(_)));
        #[cfg(not(feature = "hdf5"))]
        match error {
            IoError::Unsupported(message) => assert!(message.contains("HDF5")),
            other => panic!("expected unsupported error, got {other:?}"),
        }
    }

    #[test]
    fn text_without_sensor_size_is_rejected() {
        let error = load("events.txt", LoadOptions::default()).unwrap_err();
        match error {
            IoError::Format(message) => assert!(message.contains("sensor_size")),
            other => panic!("expected format error, got {other:?}"),
        }
    }

    fn sample_source() -> MemorySliceSource {
        let mut builder = EventStreamBuilder::new(4, 4, 0.001);
        builder.push(0, 0, 0, true);
        builder.push(1, 1, 10, false);
        builder.push(2, 2, 20, true);
        builder.push(0, 1, 30, false);
        MemorySliceSource::new(builder.build())
    }

    #[test]
    fn memory_source_reports_span_and_count() {
        let source = sample_source();
        assert_eq!(source.n_events(), 4);
        assert_eq!(source.time_span(), (0, 30));
    }

    #[test]
    fn memory_source_slices_by_time_and_index() {
        let source = sample_source();

        // Half-open [10, 30) keeps t = 10 and 20.
        assert_eq!(source.slice_time(10, 30).unwrap().ts(), &[10, 20]);
        assert_eq!(source.slice_index(1, 3).unwrap().ts(), &[10, 20]);
        assert_eq!(source.slice_index(2, 100).unwrap().len(), 2); // hi clamped to len
        assert!(source.slice_time(100, 200).unwrap().is_empty());
    }

    #[test]
    fn open_rejects_unknown_extension() {
        // `Reader` is a trait object (not `Debug`), so match rather than `unwrap_err`.
        match open("recording.mp4", LoadOptions::default()) {
            Err(IoError::Unsupported(_)) => {}
            Err(other) => panic!("expected unsupported error, got {other:?}"),
            Ok(_) => panic!("expected an error for an unknown extension"),
        }
    }
}
