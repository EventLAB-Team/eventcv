//! E2VID interchange — the text layout the reference event-to-video reconstruction expects.
//!
//! [E2VID](https://github.com/uzh-rpg/rpg_e2vid) (Rebecq et al., *High Speed and High Dynamic
//! Range Video with an Event Camera*) reads events from a `.txt` file, or a `.zip` containing one:
//! a header line, then one whitespace-separated `t x y p` row per event with `t` in **float
//! seconds** and `p` as `0`/`1`, ascending in time. That differs from eventcv's own text writer
//! (raw microseconds, no header), so it gets its own writer rather than an option on that one.
//!
//! The header line carries `width height` — E2VID skips it, but the convention is what every
//! published dataset uses, so a file written here also loads in the tools that do read it.
//!
//! [`E2vidWriter`] appends stream by stream, so a multi-gigabyte recording converts through
//! `EventReader::windows()` without ever being fully resident.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

use zip::write::FileOptions;
use zip::{CompressionMethod, ZipWriter};

use super::IoError;
use crate::EventStream;

/// Decimal places on the seconds column: eventcv stores microseconds, so six is exact and any
/// more would only write zeros.
const SECONDS_DECIMALS: usize = 6;

/// Where the rows are going: a plain `.txt`, or the single entry inside a `.zip`.
enum Sink {
    Text(BufWriter<File>),
    Zip(Box<ZipWriter<BufWriter<File>>>),
}

impl Write for Sink {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        match self {
            Self::Text(writer) => writer.write(buffer),
            Self::Zip(writer) => writer.write(buffer),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Self::Text(writer) => writer.flush(),
            Self::Zip(writer) => writer.flush(),
        }
    }
}

/// Writes events in E2VID's layout, one appended stream at a time.
///
/// The sensor size goes in the header, so it is fixed by the first append; a later stream from a
/// different sensor is an error rather than a silently mislabelled file. Timestamps must not go
/// backwards — E2VID's windowing assumes they rise — so an out-of-order append is rejected too.
pub struct E2vidWriter {
    sink: Sink,
    sensor_size: Option<(usize, usize)>,
    last_t_us: Option<i64>,
    n_events: usize,
}

impl E2vidWriter {
    /// Creates `path`, writing a zip container for `.zip` and a bare text file otherwise. The zip
    /// entry is named after the file (`events.zip` → `events.txt`), which is what E2VID's reader
    /// expects to find inside.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref();
        let file = BufWriter::new(File::create(path).map_err(IoError::Io)?);
        let zipped = path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"));
        let sink = if zipped {
            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("events");
            let mut writer = ZipWriter::new(file);
            writer
                .start_file(
                    format!("{stem}.txt"),
                    FileOptions::default().compression_method(CompressionMethod::Deflated),
                )
                .map_err(zip_error)?;
            Sink::Zip(Box::new(writer))
        } else {
            Sink::Text(file)
        };
        Ok(Self {
            sink,
            sensor_size: None,
            last_t_us: None,
            n_events: 0,
        })
    }

    /// Appends one stream's events. The first call also writes the header.
    pub fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        let sensor_size = stream.sensor_size();
        match self.sensor_size {
            None => {
                let (width, height) = sensor_size;
                writeln!(self.sink, "{width} {height}").map_err(IoError::Io)?;
                self.sensor_size = Some(sensor_size);
            }
            Some(first) if first != sensor_size => {
                return Err(IoError::Unsupported(format!(
                    "an E2VID file holds one recording: this append is {}x{} but the file was \
                     started as {}x{}",
                    sensor_size.0, sensor_size.1, first.0, first.1
                )))
            }
            Some(_) => {}
        }
        // Timestamps are stored in whatever unit the stream carries; E2VID wants seconds.
        let seconds_per_unit = stream.timestamp_scale_ms() / 1000.0;
        let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
        for index in 0..stream.len() {
            let t = ts[index];
            if self.last_t_us.is_some_and(|last| t < last) {
                return Err(IoError::Unsupported(format!(
                    "E2VID needs events in time order, but timestamp {t} follows {}; sort the \
                     stream first (stream.sort_by_time())",
                    self.last_t_us.unwrap_or_default()
                )));
            }
            self.last_t_us = Some(t);
            writeln!(
                self.sink,
                "{:.*} {} {} {}",
                SECONDS_DECIMALS,
                t as f64 * seconds_per_unit,
                xs[index],
                ys[index],
                u8::from(ps[index])
            )
            .map_err(IoError::Io)?;
        }
        self.n_events += stream.len();
        Ok(())
    }

    /// Events written so far.
    pub fn n_events(&self) -> usize {
        self.n_events
    }

    /// Flushes and closes the file (finalising the zip's central directory, when zipped).
    pub fn finish(mut self) -> Result<(), IoError> {
        // A recording with no events still gets its header, so the file is loadable rather than
        // empty — the same shape E2VID would produce for an idle sensor.
        if self.sensor_size.is_none() {
            writeln!(self.sink, "0 0").map_err(IoError::Io)?;
        }
        match self.sink {
            Sink::Text(mut writer) => writer.flush().map_err(IoError::Io),
            Sink::Zip(mut writer) => {
                let mut file = writer.finish().map_err(zip_error)?;
                file.flush().map_err(IoError::Io)
            }
        }
    }
}

/// Writes a whole stream in one call — the eager form of [`E2vidWriter`].
pub fn write_e2vid(path: impl AsRef<Path>, stream: &EventStream) -> Result<(), IoError> {
    let mut writer = E2vidWriter::create(path)?;
    writer.append(stream)?;
    writer.finish()
}

fn zip_error(error: zip::result::ZipError) -> IoError {
    match error {
        zip::result::ZipError::Io(error) => IoError::Io(error),
        other => IoError::Format(other.to_string()),
    }
}

/// `Seek` is needed by `ZipWriter`, and `BufWriter<File>` has it; the impl just forwards.
impl Seek for Sink {
    fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::Text(writer) => writer.seek(position),
            Self::Zip(_) => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "a zipped E2VID entry is written sequentially",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventStreamBuilder;

    fn stream() -> EventStream {
        let mut builder = EventStreamBuilder::new(640, 480, 0.001);
        builder.push(10, 20, 1_500_000, true);
        builder.push(11, 21, 2_250_000, false);
        builder.build()
    }

    fn written(name: &str, stream: &EventStream) -> String {
        let path = std::env::temp_dir().join(name);
        write_e2vid(&path, stream).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        text
    }

    #[test]
    fn writes_the_header_then_seconds_x_y_polarity() {
        let text = written("eventcv_e2vid_basic.txt", &stream());
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("640 480"));
        assert_eq!(lines.next(), Some("1.500000 10 20 1"));
        assert_eq!(lines.next(), Some("2.250000 11 21 0"));
        assert_eq!(lines.next(), None);
    }

    #[test]
    fn appends_keep_one_header_and_count_events() {
        let path = std::env::temp_dir().join("eventcv_e2vid_append.txt");
        let mut writer = E2vidWriter::create(&path).unwrap();
        writer.append(&stream()).unwrap();
        // A later window, as `EventReader::windows()` would hand them over.
        writer.append(&stream().time_shift(10_000_000)).unwrap();
        assert_eq!(writer.n_events(), 4);
        writer.finish().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(text.lines().filter(|line| *line == "640 480").count(), 1);
        assert_eq!(text.lines().count(), 5);
    }

    #[test]
    fn an_empty_stream_still_writes_a_loadable_file() {
        let empty = EventStreamBuilder::new(346, 260, 0.001).build();
        let text = written("eventcv_e2vid_empty.txt", &empty);
        assert_eq!(text, "346 260\n");
    }

    #[test]
    fn out_of_order_timestamps_are_rejected() {
        let mut builder = EventStreamBuilder::new(64, 64, 0.001);
        builder.push(1, 1, 2_000_000, true);
        builder.push(1, 1, 1_000_000, true);
        let path = std::env::temp_dir().join("eventcv_e2vid_unsorted.txt");
        let error = write_e2vid(&path, &builder.build()).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(error.to_string().contains("time order"), "{error}");
    }

    #[test]
    fn a_second_sensor_size_is_rejected() {
        let path = std::env::temp_dir().join("eventcv_e2vid_mixed.txt");
        let mut writer = E2vidWriter::create(&path).unwrap();
        writer.append(&stream()).unwrap();
        let other = EventStreamBuilder::new(346, 260, 0.001).build();
        let error = writer.append(&other).unwrap_err();
        writer.finish().ok();
        std::fs::remove_file(&path).ok();
        assert!(error.to_string().contains("one recording"), "{error}");
    }

    #[test]
    fn a_zip_target_holds_one_named_text_entry() {
        let path = std::env::temp_dir().join("eventcv_e2vid_zipped.zip");
        write_e2vid(&path, &stream()).unwrap();
        let file = File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 1);
        let mut entry = archive.by_index(0).unwrap();
        assert_eq!(entry.name(), "eventcv_e2vid_zipped.txt");
        let mut text = String::new();
        std::io::Read::read_to_string(&mut entry, &mut text).unwrap();
        drop(entry);
        drop(archive);
        std::fs::remove_file(&path).ok();
        assert!(text.starts_with("640 480\n1.500000 10 20 1\n"), "{text}");
    }

    #[test]
    fn a_non_microsecond_stream_still_writes_seconds() {
        // Timestamps carry their own scale, so a millisecond-based stream converts by its scale
        // rather than by assuming microseconds.
        let mut builder = EventStreamBuilder::new(64, 64, 1.0); // 1 ms per unit
        builder.push(1, 1, 1500, true);
        let text = written("eventcv_e2vid_scale.txt", &builder.build());
        assert!(text.contains("1.500000 1 1 1"), "{text}");
    }
}
