use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use super::{
    read_all, read_capped, role_of, EventKeys, EventSource, IoError, LoadOptions, RawEvent,
    SliceSource, P, T, X, Y,
};
use crate::{EventStream, EventStreamBuilder};

/// Writes a stream as whitespace-separated `t x y p` lines (the reader's default
/// [`ColumnOrder::Txyp`]), `t` in raw microseconds and `p` as `0`/`1`. Loading it back with
/// `time_unit="us"` reproduces the events exactly; sensor size is inferred or passed as an
/// option (txt carries no metadata header). The frame-domain counterpart lives in npz/HDF5.
pub fn write_text_stream(path: impl AsRef<Path>, stream: &EventStream) -> Result<(), IoError> {
    let mut writer = BufWriter::new(File::create(path).map_err(IoError::Io)?);
    let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
    for index in 0..stream.len() {
        writeln!(
            writer,
            "{} {} {} {}",
            ts[index],
            xs[index],
            ys[index],
            u8::from(ps[index])
        )
        .map_err(IoError::Io)?;
    }
    writer.flush().map_err(IoError::Io)
}

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

    /// Maps a stored `timestamp_scale_ms` (milliseconds per raw unit) back to the matching
    /// unit, when it is one of the standard powers of 1000 — lets the HDF5 reader honour the
    /// scale a stream was saved with instead of re-inferring it. `None` for other scales.
    #[cfg_attr(not(feature = "hdf5"), allow(dead_code))]
    pub(crate) fn from_scale_ms(scale_ms: f64) -> Option<TimeUnit> {
        for (unit, expected) in [
            (TimeUnit::Nanoseconds, 1e-6),
            (TimeUnit::Microseconds, 1e-3),
            (TimeUnit::Milliseconds, 1.0),
            (TimeUnit::Seconds, 1e3),
        ] {
            if (scale_ms - expected).abs() <= expected * 1e-9 {
                return Some(unit);
            }
        }
        None
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

/// The field index of each event column within a row (0-based). A [`ColumnOrder`] is the
/// fixed-position case; a detected header or explicit `keys` produce arbitrary indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnMap {
    pub t: usize,
    pub x: usize,
    pub y: usize,
    pub p: usize,
}

impl ColumnMap {
    /// The layout implied by a positional [`ColumnOrder`].
    fn from_order(order: ColumnOrder) -> Self {
        match order {
            ColumnOrder::Txyp => Self {
                t: 0,
                x: 1,
                y: 2,
                p: 3,
            },
            ColumnOrder::Xytp => Self {
                x: 0,
                y: 1,
                t: 2,
                p: 3,
            },
        }
    }
}

/// Splits a text line into fields on commas or whitespace, so both `.txt` (space/tab) and
/// `.csv` (comma) parse. Empty fields (e.g. from doubled delimiters) are skipped.
fn split_fields(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| c == ',' || c.is_whitespace())
        .filter(|field| !field.is_empty())
}

/// Whether a line is a column header rather than data — true when any field isn't a number.
fn is_header_line(line: &str) -> bool {
    split_fields(line).any(|field| field.parse::<f64>().is_err())
}

/// Builds a [`ColumnMap`] from a header line by matching each name to an x/y/t/p synonym;
/// `None` if the header doesn't name all four.
fn header_map(line: &str) -> Option<ColumnMap> {
    let mut indices = [None; 4];
    for (index, field) in split_fields(line).enumerate() {
        if let Some(role) = role_of(field) {
            indices[role].get_or_insert(index);
        }
    }
    Some(ColumnMap {
        x: indices[X]?,
        y: indices[Y]?,
        t: indices[T]?,
        p: indices[P]?,
    })
}

/// Builds a [`ColumnMap`] from explicit `keys`, each a 0-based column index or a header name.
fn keys_map(keys: &EventKeys, header: Option<&str>) -> Result<ColumnMap, IoError> {
    let resolve = |value: &str, field: &str| -> Result<usize, IoError> {
        if let Ok(index) = value.parse::<usize>() {
            return Ok(index);
        }
        if let Some(names) = header {
            if let Some(index) =
                split_fields(names).position(|name| name.eq_ignore_ascii_case(value))
            {
                return Ok(index);
            }
        }
        Err(IoError::Format(format!(
            "text key for '{field}' ({value:?}) is neither a column index nor a header name"
        )))
    };
    Ok(ColumnMap {
        x: resolve(&keys.x, "x")?,
        y: resolve(&keys.y, "y")?,
        t: resolve(&keys.t, "t")?,
        p: resolve(&keys.p, "p")?,
    })
}

/// Resolves the column layout and whether the first data-bearing line is a header to skip.
/// `keys` win; otherwise a non-numeric first line is a header matched by synonym; otherwise
/// the positional `order`. `first_line` is the first non-blank, non-`#` line (or `None`).
fn resolve_column_map(
    first_line: Option<&str>,
    order: ColumnOrder,
    keys: Option<&EventKeys>,
) -> Result<(ColumnMap, bool), IoError> {
    let has_header = first_line.is_some_and(is_header_line);
    let header = has_header.then(|| first_line.unwrap());
    let map = match keys {
        Some(keys) => keys_map(keys, header)?,
        None if has_header => header_map(header.unwrap()).ok_or_else(|| {
            IoError::Format(format!(
                "text header {:?} does not name all of x/y/t/p; pass order= or keys= to map the \
                 columns",
                header.unwrap()
            ))
        })?,
        None => ColumnMap::from_order(order),
    };
    Ok((map, has_header))
}

#[derive(Clone, Copy, Debug)]
pub struct TextOptions {
    pub width: usize,
    pub height: usize,
    pub time_unit: TimeUnit,
    /// Which field is x/y/t/p in each row.
    pub map: ColumnMap,
    /// Skip the first non-blank, non-`#` line (a column header) before parsing data.
    pub has_header: bool,
}

impl TextOptions {
    /// Defaults to seconds timestamps in `t x y p` order (the EV-IMO layout), no header.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            time_unit: TimeUnit::Seconds,
            map: ColumnMap::from_order(ColumnOrder::Txyp),
            has_header: false,
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
    /// Set until the first data line has been reached, at which point a leading header
    /// (`options.has_header`) is skipped.
    header_pending: bool,
}

impl<R: BufRead> TextReader<R> {
    pub fn new(reader: R, options: TextOptions) -> Result<Self, IoError> {
        if options.width == 0 || options.height == 0 {
            return Err(IoError::InvalidSensorSize);
        }
        Ok(Self {
            reader,
            header_pending: options.has_header,
            options,
            buffer: String::new(),
            line: 0,
        })
    }

    fn parse_line(&self, line: &str) -> Result<RawEvent, IoError> {
        let fields: Vec<&str> = split_fields(line).collect();
        let map = self.options.map;
        let pick = |index: usize, name: &str| -> Result<&str, IoError> {
            fields.get(index).copied().ok_or_else(|| IoError::Parse {
                line: self.line,
                message: format!("missing {name}"),
            })
        };
        Ok(RawEvent {
            x: self.parse(pick(map.x, "x")?, "x")?,
            y: self.parse(pick(map.y, "y")?, "y")?,
            t: self
                .options
                .time_unit
                .to_microseconds(self.parse::<f64>(pick(map.t, "t")?, "t")?),
            p: self.parse::<i32>(pick(map.p, "p")?, "p")? > 0,
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
            if self.header_pending {
                self.header_pending = false;
                continue; // the first data line is a column header — skip it
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
/// Also the input to [`load_rows`], the in-memory (`from_numpy`) loader.
pub struct RawRow {
    pub x: u16,
    pub y: u16,
    pub t: f64,
    pub p: bool,
}

/// Loads a text file, inferring whichever of `sensor_size` (from the coordinate range)
/// and `time_unit` (fractional value ⇒ seconds, else the span magnitude) the caller
/// left unset. A fully-specified load streams without buffering; inference reads the
/// rows once into memory.
pub fn load_text(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let path = path.as_ref();
    let first_line = peek_first_line(path)?;
    let (map, has_header) =
        resolve_column_map(first_line.as_deref(), options.order, options.keys.as_ref())?;
    if let (Some((width, height)), Some(time_unit)) = (options.sensor_size, options.time_unit) {
        let text_options = TextOptions {
            width,
            height,
            time_unit,
            map,
            has_header,
        };
        return read_capped(open(path, text_options)?, options.max_events);
    }

    let rows = read_raw_rows(path, map, has_header)?;
    load_rows(&rows, options)
}

/// Reads the first non-blank, non-`#` line (trimmed), or `None` for an empty/comment-only file
/// — enough to decide the column layout (header vs positional) before the full parse.
fn peek_first_line(path: &Path) -> Result<Option<String>, IoError> {
    let reader = BufReader::new(File::open(path)?);
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            return Ok(Some(trimmed.to_owned()));
        }
    }
    Ok(None)
}

/// Builds an [`EventStream`] from already-parsed rows, inferring whichever of
/// `sensor_size`/`time_unit` the caller left unset (the in-memory twin of
/// [`load_text`], shared with `eventcv.from_numpy`).
pub fn load_rows(rows: &[RawRow], options: &LoadOptions) -> Result<EventStream, IoError> {
    let (width, height) = options
        .sensor_size
        .unwrap_or_else(|| infer_sensor_size(rows));
    let time_unit = options.time_unit.unwrap_or_else(|| infer_time_unit(rows));
    if width == 0 || height == 0 {
        return Err(IoError::InvalidSensorSize);
    }

    let mut builder = EventStreamBuilder::new(width, height, 0.001);
    for row in rows {
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

/// Parses one non-blank line into a [`RawRow`] (no unit conversion or bounds check).
/// Shared by the buffered [`load_text`] and the [`TextSliceSource`] index scan.
fn parse_raw_row(trimmed: &str, map: ColumnMap, number: usize) -> Result<RawRow, IoError> {
    let fields: Vec<&str> = split_fields(trimmed).collect();
    let pick = |index: usize, name: &str| -> Result<&str, IoError> {
        fields.get(index).copied().ok_or(IoError::Parse {
            line: number,
            message: format!("missing {name}"),
        })
    };
    let parse = |value: &str, field: &str| {
        value.parse::<f64>().map_err(|_| IoError::Parse {
            line: number,
            message: format!("invalid {field}: {value:?}"),
        })
    };
    Ok(RawRow {
        x: parse(pick(map.x, "x")?, "x")? as u16,
        y: parse(pick(map.y, "y")?, "y")? as u16,
        t: parse(pick(map.t, "t")?, "t")?,
        p: parse(pick(map.p, "p")?, "p")? > 0.0,
    })
}

fn read_raw_rows(path: &Path, map: ColumnMap, has_header: bool) -> Result<Vec<RawRow>, IoError> {
    let reader = BufReader::new(File::open(path)?);
    let mut rows = Vec::new();
    let mut header_pending = has_header;
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if header_pending {
            header_pending = false;
            continue;
        }
        rows.push(parse_raw_row(trimmed, map, index + 1)?);
    }
    Ok(rows)
}

/// Number of events between sparse index samples — a slice reads at most this many extra
/// rows past a sample before reaching its window.
const TEXT_INDEX_STRIDE: usize = 4096;

#[derive(Clone, Copy)]
struct TextIndexEntry {
    offset: u64,
    count: usize,
    t_us: i64,
}

/// In-place [`SliceSource`] for text files. Text isn't seekable by content, so `open`
/// scans once to build a sparse `(byte offset, event count, timestamp)` index (one entry
/// per [`TEXT_INDEX_STRIDE`] events); slices binary-search it, seek the file, and parse
/// forward. Bounded memory; assumes events are time-ordered (errors otherwise).
pub struct TextSliceSource {
    path: PathBuf,
    map: ColumnMap,
    time_unit: TimeUnit,
    sensor: (usize, usize),
    total: usize,
    span_us: (i64, i64),
    index: Vec<TextIndexEntry>,
}

/// Scans the file once to build a `TextSliceSource`, inferring `sensor_size`/`time_unit`
/// when unset exactly as `load_text` does, and dropping out-of-bounds events (when the
/// size is explicit) so the index counts the same events `load` would keep.
pub fn open_text_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<TextSliceSource, IoError> {
    let path = path.as_ref();
    let (map, has_header) = resolve_column_map(
        peek_first_line(path)?.as_deref(),
        options.order,
        options.keys.as_ref(),
    )?;
    let mut reader = BufReader::new(File::open(path)?);
    let mut buffer = String::new();
    let mut offset = 0u64;
    let mut line_no = 0usize;
    let mut kept = 0usize;
    let mut header_pending = has_header;
    let mut samples: Vec<(u64, usize, f64)> = Vec::new();
    let (mut max_x, mut max_y) = (0u16, 0u16);
    let (mut min_t, mut max_t) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut fractional = false;
    let mut sorted = true;
    let mut previous_t = f64::NEG_INFINITY;

    loop {
        buffer.clear();
        let line_start = offset;
        let bytes = reader.read_line(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        offset += bytes as u64;
        line_no += 1;
        let trimmed = buffer.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if header_pending {
            header_pending = false;
            continue; // the header line is not indexed and not counted
        }
        let row = parse_raw_row(trimmed, map, line_no)?;
        if let Some((width, height)) = options.sensor_size {
            if usize::from(row.x) >= width || usize::from(row.y) >= height {
                continue; // matches the OOB drop a load with this size would do
            }
        }
        if kept.is_multiple_of(TEXT_INDEX_STRIDE) {
            samples.push((line_start, kept, row.t));
        }
        max_x = max_x.max(row.x);
        max_y = max_y.max(row.y);
        min_t = min_t.min(row.t);
        max_t = max_t.max(row.t);
        fractional |= row.t.fract() != 0.0;
        sorted &= row.t >= previous_t;
        previous_t = row.t;
        kept += 1;
    }

    let sensor = options
        .sensor_size
        .unwrap_or((usize::from(max_x) + 1, usize::from(max_y) + 1));
    if sensor.0 == 0 || sensor.1 == 0 {
        return Err(IoError::InvalidSensorSize);
    }
    if !sorted {
        return Err(IoError::Format(
            "text timestamps are not sorted; in-place slicing requires time-ordered events"
                .to_owned(),
        ));
    }
    let time_unit = options.time_unit.unwrap_or_else(|| {
        if fractional || !min_t.is_finite() {
            TimeUnit::Seconds
        } else {
            TimeUnit::infer_from_span((max_t - min_t) as i64)
        }
    });

    let index = samples
        .into_iter()
        .map(|(offset, count, t)| TextIndexEntry {
            offset,
            count,
            t_us: time_unit.to_microseconds(t),
        })
        .collect();
    let span_us = if kept == 0 {
        (0, 0)
    } else {
        (
            time_unit.to_microseconds(min_t),
            time_unit.to_microseconds(max_t),
        )
    };
    Ok(TextSliceSource {
        path: path.to_path_buf(),
        map,
        time_unit,
        sensor,
        total: kept,
        span_us,
        index,
    })
}

impl TextSliceSource {
    /// Opens the file again seeked to `offset`, wrapped in a [`TextReader`] for parsing.
    fn reader_at(&self, offset: u64) -> Result<TextReader<BufReader<File>>, IoError> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(offset))?;
        TextReader::new(
            BufReader::new(file),
            TextOptions {
                width: self.sensor.0,
                height: self.sensor.1,
                time_unit: self.time_unit,
                map: self.map,
                has_header: false, // seeks land on data lines; the header was skipped at open
            },
        )
    }

    fn keeps(&self, x: u16, y: u16) -> bool {
        usize::from(x) < self.sensor.0 && usize::from(y) < self.sensor.1
    }
}

impl SliceSource for TextSliceSource {
    fn sensor_size(&self) -> (usize, usize) {
        self.sensor
    }

    fn timestamp_scale_ms(&self) -> f64 {
        0.001
    }

    fn n_events(&self) -> usize {
        self.total
    }

    fn time_span(&self) -> (i64, i64) {
        self.span_us
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let i0 = i0.min(self.total);
        let i1 = i1.clamp(i0, self.total);
        let mut builder = EventStreamBuilder::new(self.sensor.0, self.sensor.1, 0.001);
        if i0 == i1 || self.index.is_empty() {
            return Ok(builder.build());
        }
        // index[0].count == 0, so partition_point is >= 1 and the subtraction is safe.
        let entry = self.index[self.index.partition_point(|e| e.count <= i0) - 1];
        let mut reader = self.reader_at(entry.offset)?;
        let mut index = entry.count;
        while index < i1 {
            let Some(event) = reader.next_event()? else {
                break;
            };
            if !self.keeps(event.x, event.y) {
                continue; // dropped, not counted — keeps indices aligned with `load`
            }
            if index >= i0 {
                builder.push(event.x, event.y, event.t, event.p);
            }
            index += 1;
        }
        Ok(builder.build())
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let mut builder = EventStreamBuilder::new(self.sensor.0, self.sensor.1, 0.001);
        if self.index.is_empty() {
            return Ok(builder.build());
        }
        // Start strictly before t0: when many events share t0 and span more than
        // TEXT_INDEX_STRIDE, several index entries carry t_us == t0, so `<= t0` would
        // seek past the earliest ones and drop them. `< t0` lands before them all.
        let entry = self.index[self
            .index
            .partition_point(|e| e.t_us < t0)
            .saturating_sub(1)];
        let mut reader = self.reader_at(entry.offset)?;
        while let Some(event) = reader.next_event()? {
            if event.t >= t1 {
                break; // events are time-ordered (checked at open)
            }
            if event.t >= t0 {
                builder.push(event.x, event.y, event.t, event.p);
            }
        }
        Ok(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        load_text, open_text_slice, ColumnMap, ColumnOrder, TextOptions, TextReader, TimeUnit,
    };
    use crate::io::{read_all, IoError, LoadOptions, SliceSource};
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
            map: ColumnMap::from_order(ColumnOrder::Xytp),
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

    #[test]
    fn text_slice_source_matches_load() {
        // 10 events, integer µs t = 0..9000 at distinct pixels (x = 0..9, y = i % 5).
        let mut data = String::new();
        for i in 0..10 {
            data.push_str(&format!("{} {} {} {}\n", i * 1000, i, i % 5, i % 2));
        }
        let path = write_temp("txtslice", &data);
        // Explicit µs so the small integer timestamps aren't inferred as milliseconds.
        let options = LoadOptions {
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let source = open_text_slice(&path, &options).unwrap();
        let full = load_text(&path, &options).unwrap();

        assert_eq!(source.n_events(), full.len());
        assert_eq!(source.sensor_size(), full.sensor_size()); // (10, 5)
        assert_eq!(source.time_span(), (0, 9000));

        assert_eq!(
            source.slice_time(2000, 6000).unwrap().ts(),
            &[2000, 3000, 4000, 5000]
        );
        assert_eq!(
            source.slice_index(3, 7).unwrap().ts(),
            &[3000, 4000, 5000, 6000]
        );

        let whole = source.slice_index(0, source.n_events()).unwrap();
        assert_eq!(whole.xs(), full.xs());
        assert_eq!(whole.ts(), full.ts());
        assert_eq!(whole.ps(), full.ps());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn text_slice_rejects_unsorted_timestamps() {
        let path = write_temp("txtunsorted", "0 0 0 1\n5000 1 1 0\n2000 2 2 1\n");
        match open_text_slice(&path, &LoadOptions::default()) {
            Err(IoError::Format(message)) => assert!(message.contains("not sorted")),
            Err(other) => panic!("expected a not-sorted error, got {other:?}"),
            Ok(_) => panic!("expected unsorted text to be rejected"),
        }
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn text_slice_keeps_same_timestamp_events_spanning_the_index_stride() {
        // A single timestamp t = 1000 carries more events than TEXT_INDEX_STRIDE, so the
        // sparse index holds several entries with t_us == 1000. slice_time(1000, ..) must
        // seek *before* all of them and keep every event at t == 1000, not just the tail.
        let dense = super::TEXT_INDEX_STRIDE * 2 + 100;
        let mut data = String::from("0 0 0 1\n"); // one earlier event
        for _ in 0..dense {
            data.push_str("1000 1 1 1\n");
        }
        data.push_str("2000 2 2 0\n"); // one later event
        let path = write_temp("txtdense", &data);
        let options = LoadOptions {
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let source = open_text_slice(&path, &options).unwrap();

        assert_eq!(source.slice_time(1000, 2000).unwrap().len(), dense);
        // Full tiling must recover every event with no drops or duplicates.
        let tiled = source.slice_time(0, 1000).unwrap().len()
            + source.slice_time(1000, 2000).unwrap().len()
            + source.slice_time(2000, 3000).unwrap().len();
        assert_eq!(tiled, source.n_events());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn text_slice_empty_file() {
        let path = write_temp("txtsliceempty", "\n# only a comment\n");
        let source = open_text_slice(&path, &LoadOptions::default()).unwrap();

        assert_eq!(source.n_events(), 0);
        assert_eq!(source.time_span(), (0, 0));
        assert!(source.slice_time(0, 1000).unwrap().is_empty());
        assert!(source.slice_index(0, 10).unwrap().is_empty());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn write_text_stream_round_trips_at_event_level() {
        let mut builder = crate::EventStreamBuilder::new(16, 12, 0.001);
        for &(x, y, t, p) in &[(0u16, 0u16, 5i64, true), (15, 11, 2_500_000, false)] {
            builder.push(x, y, t, p);
        }
        let stream = builder.build();

        let dir = std::env::temp_dir().join(format!("eventcv-txtrt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.txt");
        super::write_text_stream(&path, &stream).unwrap();

        // Loaded back as microseconds with an explicit grid, the events match exactly (txt
        // carries no metadata header, so size/unit come from options or inference).
        let options = LoadOptions {
            sensor_size: Some((16, 12)),
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let loaded = load_text(&path, &options).unwrap();
        assert_eq!(loaded.xs(), stream.xs());
        assert_eq!(loaded.ys(), stream.ys());
        assert_eq!(loaded.ts(), stream.ts());
        assert_eq!(loaded.ps(), stream.ps());
        assert_eq!(loaded.sensor_size(), (16, 12));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_text_reads_csv_with_header_in_any_order() {
        // A `.csv` with a header naming the columns out of the default order and commas.
        let path = write_temp("txtcsvhdr", "x,y,t,p\n1,2,1000,1\n3,0,2000,0\n0,1,3000,1\n");
        let options = LoadOptions {
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let stream = load_text(&path, &options).unwrap();

        assert_eq!(stream.len(), 3); // header skipped, three data rows
        assert_eq!(stream.xs(), &[1, 3, 0]);
        assert_eq!(stream.ys(), &[2, 0, 1]);
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        assert_eq!(stream.ps(), &[true, false, true]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn load_text_matches_synonym_header_names() {
        // Whitespace-separated with synonym header names (timestamp/polarity), reordered.
        let path = write_temp(
            "txtsynhdr",
            "timestamp x y polarity\n1000 1 2 1\n2000 3 0 0\n",
        );
        let options = LoadOptions {
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let stream = load_text(&path, &options).unwrap();
        assert_eq!(stream.xs(), &[1, 3]);
        assert_eq!(stream.ts(), &[1000, 2000]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn text_keys_override_selects_columns_by_index() {
        // No header; a non-default column order named explicitly by 0-based index.
        let path = write_temp("txtkeys", "1 2 1000 1\n3 0 2000 0\n"); // x y t p
        let options = LoadOptions {
            time_unit: Some(TimeUnit::Microseconds),
            keys: Some(crate::io::EventKeys {
                x: "0".to_owned(),
                y: "1".to_owned(),
                t: "2".to_owned(),
                p: "3".to_owned(),
            }),
            ..LoadOptions::default()
        };
        let stream = load_text(&path, &options).unwrap();
        assert_eq!(stream.xs(), &[1, 3]);
        assert_eq!(stream.ys(), &[2, 0]);
        assert_eq!(stream.ts(), &[1000, 2000]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn text_slice_over_headed_csv_matches_load() {
        let mut data = String::from("t,x,y,p\n");
        for i in 0..10 {
            data.push_str(&format!("{},{},{},{}\n", i * 1000, i, i % 5, i % 2));
        }
        let path = write_temp("txtslicehdr", &data);
        let options = LoadOptions {
            time_unit: Some(TimeUnit::Microseconds),
            ..LoadOptions::default()
        };
        let source = open_text_slice(&path, &options).unwrap();
        let full = load_text(&path, &options).unwrap();

        assert_eq!(source.n_events(), full.len()); // header excluded from the count
        assert_eq!(source.time_span(), (0, 9000));
        assert_eq!(
            source.slice_time(2000, 5000).unwrap().ts(),
            &[2000, 3000, 4000]
        );
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }
}
