use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::{FromStr, SplitWhitespace};

use super::{read_all, read_capped, EventSource, IoError, LoadOptions, RawEvent};
use crate::{EventStream, EventStreamBuilder};

/// Unit of the timestamp column. Events are stored internally in microseconds, so
/// [`TextReader`] always reports `timestamp_scale_ms() == 0.001`; sub-microsecond
/// precision is rounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl TimeUnit {
    fn to_microseconds(self, value: f64) -> i64 {
        let microseconds = match self {
            Self::Seconds => value * 1e6,
            Self::Milliseconds => value * 1e3,
            Self::Microseconds => value,
            Self::Nanoseconds => value / 1e3,
        };
        microseconds.round() as i64
    }

    /// Guesses the unit of an integer timestamp column from the recording's raw span
    /// (`max - min`). Event recordings run ~seconds to hours, so we pick the *finest*
    /// unit whose total duration is at least one second — e.g. a span of 6.5e11 reads as
    /// nanoseconds (651 s), not microseconds (7.5 days). Assumes a recording ≥ ~1 s;
    /// callers pass an explicit unit to override. A fractional text value means seconds.
    pub(crate) fn infer_from_span(span: i64) -> TimeUnit {
        let span = span.max(0) as f64;
        if span * 1e-9 >= 1.0 {
            TimeUnit::Nanoseconds
        } else if span * 1e-6 >= 1.0 {
            TimeUnit::Microseconds
        } else if span * 1e-3 >= 1.0 {
            TimeUnit::Milliseconds
        } else {
            TimeUnit::Seconds
        }
    }

    /// Converts an integer timestamp column (e.g. from HDF5) to microseconds,
    /// saturating rather than overflowing if the wrong unit is supplied.
    #[cfg(feature = "hdf5")]
    pub(crate) fn microseconds_from_int(self, value: i64) -> i64 {
        let value = i128::from(value);
        let microseconds = match self {
            Self::Seconds => value * 1_000_000,
            Self::Milliseconds => value * 1_000,
            Self::Microseconds => value,
            Self::Nanoseconds => value / 1_000,
        };
        microseconds.clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }
}

/// Column order of each whitespace-separated line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColumnOrder {
    /// `t x y p` (e.g. EV-IMO, RPG datasets).
    #[default]
    Txyp,
    /// `x y t p`.
    Xytp,
}

#[derive(Clone, Copy, Debug)]
pub struct TextOptions {
    pub width: usize,
    pub height: usize,
    pub time_unit: TimeUnit,
    pub order: ColumnOrder,
}

impl TextOptions {
    /// Defaults to seconds timestamps in `t x y p` order (the EV-IMO layout).
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            time_unit: TimeUnit::Seconds,
            order: ColumnOrder::Txyp,
        }
    }
}

/// Streams events from whitespace-separated text, one event per line. Blank lines
/// and `#` comments are skipped; polarity is positive when its value is greater
/// than zero (handles both `0/1` and `-1/1` conventions). Extra columns are ignored.
#[derive(Debug)]
pub struct TextReader<R> {
    reader: R,
    options: TextOptions,
    buffer: String,
    line: usize,
}

impl<R: BufRead> TextReader<R> {
    pub fn new(reader: R, options: TextOptions) -> Result<Self, IoError> {
        if options.width == 0 || options.height == 0 {
            return Err(IoError::InvalidSensorSize);
        }
        Ok(Self {
            reader,
            options,
            buffer: String::new(),
            line: 0,
        })
    }

    fn parse_line(&self, line: &str) -> Result<RawEvent, IoError> {
        let mut fields = line.split_whitespace();
        let (t, x, y, p) = match self.options.order {
            ColumnOrder::Txyp => {
                let t = self.field(&mut fields, "t")?;
                (
                    t,
                    self.field(&mut fields, "x")?,
                    self.field(&mut fields, "y")?,
                    self.field(&mut fields, "p")?,
                )
            }
            ColumnOrder::Xytp => {
                let x = self.field(&mut fields, "x")?;
                let y = self.field(&mut fields, "y")?;
                (
                    self.field(&mut fields, "t")?,
                    x,
                    y,
                    self.field(&mut fields, "p")?,
                )
            }
        };
        Ok(RawEvent {
            x: self.parse(x, "x")?,
            y: self.parse(y, "y")?,
            t: self
                .options
                .time_unit
                .to_microseconds(self.parse::<f64>(t, "t")?),
            p: self.parse::<i32>(p, "p")? > 0,
        })
    }

    fn field<'a>(&self, fields: &mut SplitWhitespace<'a>, name: &str) -> Result<&'a str, IoError> {
        fields.next().ok_or_else(|| IoError::Parse {
            line: self.line,
            message: format!("missing {name}"),
        })
    }

    fn parse<T: FromStr>(&self, value: &str, field: &str) -> Result<T, IoError> {
        value.parse().map_err(|_| IoError::Parse {
            line: self.line,
            message: format!("invalid {field}: {value:?}"),
        })
    }
}

impl<R: BufRead> EventSource for TextReader<R> {
    fn sensor_size(&self) -> (usize, usize) {
        (self.options.width, self.options.height)
    }

    fn timestamp_scale_ms(&self) -> f64 {
        0.001
    }

    fn next_event(&mut self) -> Result<Option<RawEvent>, IoError> {
        loop {
            self.buffer.clear();
            self.line += 1;
            if self.reader.read_line(&mut self.buffer)? == 0 {
                return Ok(None);
            }
            let trimmed = self.buffer.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            return self.parse_line(trimmed).map(Some);
        }
    }
}

/// Opens a text file as a streaming [`TextReader`].
pub fn open(
    path: impl AsRef<Path>,
    options: TextOptions,
) -> Result<TextReader<BufReader<File>>, IoError> {
    TextReader::new(BufReader::new(File::open(path)?), options)
}

/// Reads an entire text file into an [`EventStream`].
pub fn read_text(path: impl AsRef<Path>, options: TextOptions) -> Result<EventStream, IoError> {
    read_all(open(path, options)?)
}

/// One parsed row before unit conversion / bounds filtering. The inference path needs
/// the *raw* timestamp (to detect the unit), so it can't go through [`TextReader`].
struct RawRow {
    x: u16,
    y: u16,
    t: f64,
    p: bool,
}

/// Loads a text file, inferring whichever of `sensor_size` (from the coordinate range)
/// and `time_unit` (fractional value ⇒ seconds, else the span magnitude) the caller
/// left unset. A fully-specified load streams without buffering; inference reads the
/// rows once into memory.
pub fn load_text(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    if let (Some((width, height)), Some(time_unit)) = (options.sensor_size, options.time_unit) {
        let text_options = TextOptions {
            width,
            height,
            time_unit,
            order: options.order,
        };
        return read_capped(open(path.as_ref(), text_options)?, options.max_events);
    }

    let rows = read_raw_rows(path.as_ref(), options.order)?;
    let (width, height) = options
        .sensor_size
        .unwrap_or_else(|| infer_sensor_size(&rows));
    let time_unit = options.time_unit.unwrap_or_else(|| infer_time_unit(&rows));
    if width == 0 || height == 0 {
        return Err(IoError::InvalidSensorSize);
    }

    let mut builder = EventStreamBuilder::new(width, height, 0.001);
    for row in &rows {
        builder.push(row.x, row.y, time_unit.to_microseconds(row.t), row.p);
        if options.max_events.is_some_and(|max| builder.len() >= max) {
            break;
        }
    }
    Ok(builder.build())
}

/// Smallest sensor that holds every event: `(max_x + 1, max_y + 1)`, or `(1, 1)` when
/// there are no events.
fn infer_sensor_size(rows: &[RawRow]) -> (usize, usize) {
    let width = rows.iter().map(|row| usize::from(row.x)).max();
    let height = rows.iter().map(|row| usize::from(row.y)).max();
    match (width, height) {
        (Some(width), Some(height)) => (width + 1, height + 1),
        _ => (1, 1),
    }
}

/// A fractional timestamp means seconds; otherwise pick the unit from the span.
fn infer_time_unit(rows: &[RawRow]) -> TimeUnit {
    if rows.iter().any(|row| row.t.fract() != 0.0) {
        return TimeUnit::Seconds;
    }
    let min = rows.iter().map(|row| row.t).fold(f64::INFINITY, f64::min);
    let max = rows
        .iter()
        .map(|row| row.t)
        .fold(f64::NEG_INFINITY, f64::max);
    if min.is_finite() {
        TimeUnit::infer_from_span((max - min) as i64)
    } else {
        TimeUnit::Seconds
    }
}

fn read_raw_rows(path: &Path, order: ColumnOrder) -> Result<Vec<RawRow>, IoError> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let number = index + 1;
        let mut fields = trimmed.split_whitespace();
        let mut next = |name: &str| {
            fields.next().ok_or(IoError::Parse {
                line: number,
                message: format!("missing {name}"),
            })
        };
        let (t, x, y, p) = match order {
            ColumnOrder::Txyp => {
                let t = next("t")?;
                (t, next("x")?, next("y")?, next("p")?)
            }
            ColumnOrder::Xytp => {
                let x = next("x")?;
                let y = next("y")?;
                (next("t")?, x, y, next("p")?)
            }
        };
        let parse = |value: &str, field: &str| {
            value.parse::<f64>().map_err(|_| IoError::Parse {
                line: number,
                message: format!("invalid {field}: {value:?}"),
            })
        };
        rows.push(RawRow {
            x: parse(x, "x")? as u16,
            y: parse(y, "y")? as u16,
            t: parse(t, "t")?,
            p: parse(p, "p")? > 0.0,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{load_text, ColumnOrder, TextOptions, TextReader, TimeUnit};
    use crate::io::{read_all, IoError, LoadOptions};
    use crate::EventStream;

    fn read(data: &str, options: TextOptions) -> Result<EventStream, IoError> {
        read_all(TextReader::new(Cursor::new(data), options).unwrap())
    }

    #[test]
    fn parses_txyp_seconds_skips_noise_and_drops_out_of_bounds() {
        let data = "0.0 1 2 1\n0.000002 3 0 0\n\n# comment\n0.00001 0 4 1\n0.00002 4 0 1\n";
        let stream = read(data, TextOptions::new(4, 5)).unwrap();

        assert_eq!(stream.len(), 3); // (4, 0) dropped: x == width
        assert_eq!(stream.xs(), &[1, 3, 0]);
        assert_eq!(stream.ys(), &[2, 0, 4]);
        assert_eq!(stream.ts(), &[0, 2, 10]);
        assert_eq!(stream.ps(), &[true, false, true]);
        assert_eq!(stream.sensor_size(), (4, 5));
        assert_eq!(stream.timestamp_scale_ms(), 0.001);
    }

    #[test]
    fn supports_xytp_order_and_negative_polarity() {
        let options = TextOptions {
            order: ColumnOrder::Xytp,
            ..TextOptions::new(8, 8)
        };
        let stream = read("1 2 0.5 -1\n", options).unwrap();

        assert_eq!(stream.xs(), &[1]);
        assert_eq!(stream.ys(), &[2]);
        assert_eq!(stream.ts(), &[500_000]); // 0.5 s -> 500000 us
        assert_eq!(stream.ps(), &[false]); // -1 -> negative
    }

    #[test]
    fn converts_time_units_to_microseconds() {
        for (unit, raw, expected) in [
            (TimeUnit::Microseconds, "7", 7_i64),
            (TimeUnit::Milliseconds, "2", 2_000),
            (TimeUnit::Nanoseconds, "2400", 2),
        ] {
            let data = format!("{raw} 0 0 1\n");
            let options = TextOptions {
                time_unit: unit,
                ..TextOptions::new(4, 4)
            };
            assert_eq!(
                read(&data, options).unwrap().ts(),
                &[expected],
                "unit {unit:?}"
            );
        }
    }

    #[test]
    fn reports_parse_errors_with_line_numbers() {
        let error = read("0.0 1 2 1\n0.0 nope 2 1\n", TextOptions::new(4, 4)).unwrap_err();
        match error {
            IoError::Parse { line, .. } => assert_eq!(line, 2),
            other => panic!("expected parse error, got {other:?}"),
        }
    }

    #[test]
    fn reports_missing_fields() {
        let error = read("0.0 1 2\n", TextOptions::new(4, 4)).unwrap_err();
        assert!(matches!(error, IoError::Parse { line: 1, .. }));
    }

    #[test]
    fn rejects_zero_sensor_size() {
        let error = TextReader::new(Cursor::new(""), TextOptions::new(0, 4)).unwrap_err();
        assert!(matches!(error, IoError::InvalidSensorSize));
    }

    #[test]
    fn empty_input_yields_empty_stream() {
        let stream = read("\n\n# only comments\n", TextOptions::new(4, 4)).unwrap();
        assert!(stream.is_empty());
        assert_eq!(stream.sensor_size(), (4, 4));
    }

    fn write_temp(tag: &str, contents: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("eventcv-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.txt");
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn load_text_infers_size_and_microseconds() {
        // Integer µs, txyp; coords up to (3, 2) -> 4x3; span 5e6 -> microseconds.
        let path = write_temp("txtus", "1000000 0 0 1\n3000000 3 1 0\n6000000 1 2 1\n");
        let stream = load_text(&path, &LoadOptions::default()).unwrap();

        assert_eq!(stream.sensor_size(), (4, 3));
        assert_eq!(stream.len(), 3); // nothing dropped: size came from the data
        assert_eq!(stream.ts(), &[1_000_000, 3_000_000, 6_000_000]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn load_text_infers_seconds_from_a_fractional_value() {
        let path = write_temp("txtsec", "0.0 0 0 1\n0.5 1 1 0\n");
        let stream = load_text(&path, &LoadOptions::default()).unwrap();

        assert_eq!(stream.ts(), &[0, 500_000]); // 0.5 s -> 500000 µs
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn explicit_options_override_inference() {
        let path = write_temp("txtexp", "7 0 0 1\n");
        let options = LoadOptions {
            sensor_size: Some((4, 4)),
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let stream = load_text(&path, &options).unwrap();

        assert_eq!(stream.sensor_size(), (4, 4));
        assert_eq!(stream.ts(), &[7]); // explicit µs, not inferred
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
