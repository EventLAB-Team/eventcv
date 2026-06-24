use std::{error::Error, fmt, io, path::Path};

use crate::{EventStream, EventStreamBuilder};

mod bag;
#[cfg(feature = "hdf5")]
mod h5;
mod npz;
mod text;

pub use bag::read_bag;
#[cfg(feature = "hdf5")]
pub use h5::read_hdf5;
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
    use super::{load, IoError, LoadOptions};

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
}
