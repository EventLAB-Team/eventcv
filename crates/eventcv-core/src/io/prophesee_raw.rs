//! Prophesee EVT3 `.raw` reader.
//!
//! EVT3 is a stateful stream of little-endian 16-bit words.  A word either updates the
//! current address/time state or emits one or more CD events.  The recordings this reader
//! targets are routinely tens of gigabytes, so [`RawSliceSource`] builds a sparse byte/time
//! index once and replays from the nearest saved decoder state for each slice.  The event
//! payload is never materialised as a whole.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::{EventStream, EventStreamBuilder, IoError, LoadOptions, RawEvent, SliceSource};

const TIMESTAMP_SCALE_MS: f64 = 0.001;
const TIME_LOOP: i64 = 1 << 24;
const MAX_TIMESTAMP_BASE: i64 = ((1 << 12) - 1) << 12;
const LOOP_THRESHOLD: i64 = 10 << 12;
const CHECKPOINT_BYTES: u64 = 1 << 20;

#[derive(Clone, Copy, Debug, Default)]
struct Decoder {
    initialised: bool,
    time_base: i64,
    time: i64,
    y: u16,
    base_x: u16,
    polarity: bool,
    high_loops: i64,
}

impl Decoder {
    fn decode(&mut self, word: u16, mut emit: impl FnMut(RawEvent)) {
        let kind = word >> 12;
        let payload = word & 0x0fff;

        // Coordinate words before the first TIME_HIGH do not have a valid timestamp.
        if !self.initialised {
            if kind == 0x8 {
                self.time_base = i64::from(payload) << 12;
                self.time = self.time_base;
                self.initialised = true;
            }
            return;
        }

        match kind {
            0x0 => self.y = word & 0x07ff, // EVT_ADDR_Y
            0x2 => emit(RawEvent { // EVT_ADDR_X
                x: word & 0x07ff,
                y: self.y,
                t: self.time,
                p: word & 0x0800 != 0,
            }),
            0x3 => { // VECT_BASE_X
                self.base_x = word & 0x07ff;
                self.polarity = word & 0x0800 != 0;
            }
            0x4 => { // VECT_12
                for bit in 0..12 {
                    if payload & (1 << bit) != 0 {
                        emit(RawEvent {
                            x: self.base_x.saturating_add(bit),
                            y: self.y,
                            t: self.time,
                            p: self.polarity,
                        });
                    }
                }
                self.base_x = self.base_x.saturating_add(12);
            }
            0x5 => { // VECT_8
                for bit in 0..8 {
                    if payload & (1 << bit) != 0 {
                        emit(RawEvent {
                            x: self.base_x.saturating_add(bit),
                            y: self.y,
                            t: self.time,
                            p: self.polarity,
                        });
                    }
                }
                self.base_x = self.base_x.saturating_add(8);
            }
            0x6 => self.time = self.time_base + i64::from(payload), // EVT_TIME_LOW
            0x8 => { // EVT_TIME_HIGH, including the 24-bit wrap every 16.777216 s
                let mut next = (i64::from(payload) << 12) + self.high_loops * TIME_LOOP;
                if self.time_base > next
                    && self.time_base - next >= MAX_TIMESTAMP_BASE - LOOP_THRESHOLD
                {
                    self.high_loops += 1;
                    next += TIME_LOOP;
                }
                self.time_base = next;
                self.time = next;
            }
            // EXT_TRIGGER, OTHERS, CONTINUED and reserved packet types do not emit CD events.
            _ => {}
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Checkpoint {
    byte_offset: u64,
    event_index: usize,
    time: i64,
    decoder: Decoder,
}

#[derive(Debug)]
struct Header {
    data_offset: u64,
    sensor_size: (usize, usize),
}

fn parse_header(reader: &mut BufReader<File>, override_size: Option<(usize, usize)>)
    -> Result<Header, IoError>
{
    let mut lines = Vec::new();
    loop {
        let is_comment = match reader.fill_buf()? {
            [] => break,
            bytes => bytes[0] == b'%',
        };
        if !is_comment {
            break;
        }
        let mut line = Vec::new();
        reader.read_until(b'\n', &mut line)?;
        lines.push(String::from_utf8_lossy(&line).trim().to_owned());
    }

    let mut geometry = None;
    let mut format = None;
    let mut evt = None;
    for line in &lines {
        let body = line.trim_start_matches('%').trim();
        let (key, value) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
        match key.to_ascii_lowercase().as_str() {
            "geometry" => {
                let value = value.trim();
                let parts = value.split_once('x').or_else(|| value.split_once('X'));
                if let Some((width, height)) = parts {
                    geometry = width.trim().parse().ok().zip(height.trim().parse().ok());
                }
            }
            "format" => format = Some(value.trim().to_ascii_lowercase()),
            "evt" => evt = Some(value.trim().to_ascii_lowercase()),
            _ => {}
        }
    }
    let is_evt3 = format.as_deref().is_some_and(|value| value.starts_with("evt3"))
        || evt.as_deref().is_some_and(|value| value.starts_with('3'));
    if !is_evt3 {
        return Err(IoError::Unsupported(
            "Prophesee RAW reader currently supports EVT3 recordings only".to_owned(),
        ));
    }
    let sensor_size = override_size.or(geometry).ok_or_else(|| {
        IoError::Unsupported(
            "could not determine sensor size from the Prophesee RAW header; pass sensor_size"
                .to_owned(),
        )
    })?;
    if sensor_size.0 == 0 || sensor_size.1 == 0 {
        return Err(IoError::InvalidSensorSize);
    }
    Ok(Header {
        data_offset: reader.stream_position()?,
        sensor_size,
    })
}

/// Lazy, indexed source for a Prophesee EVT3 RAW recording.
pub struct RawSliceSource {
    path: PathBuf,
    width: usize,
    height: usize,
    checkpoints: Vec<Checkpoint>,
    n_events: usize,
    time_span: (i64, i64),
}

impl RawSliceSource {
    fn open(path: &Path, options: &LoadOptions) -> Result<Self, IoError> {
        let mut reader = BufReader::new(File::open(path)?);
        let header = parse_header(&mut reader, options.sensor_size)?;
        let mut decoder = Decoder::default();
        let mut checkpoints = vec![Checkpoint {
            byte_offset: header.data_offset,
            event_index: 0,
            time: 0,
            decoder,
        }];
        let mut event_index = 0usize;
        let mut minimum = i64::MAX;
        let mut maximum = i64::MIN;
        let mut byte_offset = header.data_offset;
        let mut chunk = Vec::with_capacity(CHECKPOINT_BYTES as usize);

        loop {
            chunk.clear();
            (&mut reader)
                .take(CHECKPOINT_BYTES)
                .read_to_end(&mut chunk)?;
            if chunk.is_empty() {
                break;
            }
            if byte_offset != header.data_offset {
                checkpoints.push(Checkpoint {
                    byte_offset,
                    event_index,
                    time: decoder.time,
                    decoder,
                });
            }
            for bytes in chunk.chunks_exact(2) {
                decoder.decode(u16::from_le_bytes([bytes[0], bytes[1]]), |event| {
                    if usize::from(event.x) < header.sensor_size.0
                        && usize::from(event.y) < header.sensor_size.1
                    {
                        event_index += 1;
                        minimum = minimum.min(event.t);
                        maximum = maximum.max(event.t);
                    }
                });
            }
            byte_offset += chunk.len() as u64;
        }

        Ok(Self {
            path: path.to_owned(),
            width: header.sensor_size.0,
            height: header.sensor_size.1,
            checkpoints,
            n_events: event_index,
            time_span: if event_index == 0 { (0, 0) } else { (minimum, maximum) },
        })
    }

    fn checkpoint_for_index(&self, index: usize) -> Checkpoint {
        let position = self
            .checkpoints
            .partition_point(|checkpoint| checkpoint.event_index <= index)
            .saturating_sub(1);
        self.checkpoints[position]
    }

    fn checkpoint_for_time(&self, time: i64) -> Checkpoint {
        let position = self
            .checkpoints
            .partition_point(|checkpoint| checkpoint.time <= time)
            .saturating_sub(1);
        // A checkpoint timestamp is the state before its next word. Replaying one checkpoint
        // earlier also covers streams whose repeated TIME_LOW values briefly move backwards.
        self.checkpoints[position.saturating_sub(1)]
    }

    fn reader_at(&self, checkpoint: Checkpoint) -> Result<BufReader<File>, IoError> {
        let mut reader = BufReader::new(File::open(&self.path)?);
        reader.seek(SeekFrom::Start(checkpoint.byte_offset))?;
        Ok(reader)
    }
}

impl SliceSource for RawSliceSource {
    fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn timestamp_scale_ms(&self) -> f64 {
        TIMESTAMP_SCALE_MS
    }

    fn n_events(&self) -> usize {
        self.n_events
    }

    fn time_span(&self) -> (i64, i64) {
        self.time_span
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let i0 = i0.min(self.n_events);
        let i1 = i1.clamp(i0, self.n_events);
        let checkpoint = self.checkpoint_for_index(i0);
        let mut reader = self.reader_at(checkpoint)?;
        let mut decoder = checkpoint.decoder;
        let mut index = checkpoint.event_index;
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        let mut bytes = [0u8; 2];
        while index < i1 {
            match reader.read_exact(&mut bytes) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(IoError::Io(error)),
            }
            decoder.decode(u16::from_le_bytes(bytes), |event| {
                if usize::from(event.x) < self.width && usize::from(event.y) < self.height {
                    if index >= i0 && index < i1 {
                        builder.push(event.x, event.y, event.t, event.p);
                    }
                    index += 1;
                }
            });
        }
        Ok(builder.build())
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let checkpoint = self.checkpoint_for_time(t0);
        let mut reader = self.reader_at(checkpoint)?;
        let mut decoder = checkpoint.decoder;
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        let mut bytes = [0u8; 2];
        loop {
            match reader.read_exact(&mut bytes) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(IoError::Io(error)),
            }
            decoder.decode(u16::from_le_bytes(bytes), |event| {
                if event.t >= t0
                    && event.t < t1
                    && usize::from(event.x) < self.width
                    && usize::from(event.y) < self.height
                {
                    builder.push(event.x, event.y, event.t, event.p);
                }
            });
            if decoder.initialised && decoder.time >= t1 {
                break;
            }
        }
        Ok(builder.build())
    }
}

/// Opens a Prophesee EVT3 RAW file for bounded-memory random slicing.
pub fn open_raw_slice(path: impl AsRef<Path>, options: &LoadOptions)
    -> Result<RawSliceSource, IoError>
{
    RawSliceSource::open(path.as_ref(), options)
}

/// Eagerly reads a Prophesee EVT3 RAW file. Prefer [`open_raw_slice`] for large recordings.
pub fn read_raw(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let source = open_raw_slice(path, options)?;
    let limit = options.max_events.unwrap_or(source.n_events());
    source.slice_index(0, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn word(kind: u16, payload: u16) -> [u8; 2] {
        ((kind << 12) | (payload & 0x0fff)).to_le_bytes()
    }

    fn raw(words: &[[u8; 2]]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "eventcv_evt3_{}_{}.raw",
            std::process::id(),
            words.len()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(b"% format EVT3\n% geometry 1280x720\n").unwrap();
        for bytes in words {
            file.write_all(bytes).unwrap();
        }
        path
    }

    #[test]
    fn scalar_and_vector_events_decode_and_slice() {
        let path = raw(&[
            word(0x8, 1),
            word(0x6, 10),
            word(0x0, 20),
            word(0x2, 30 | 0x0800),
            word(0x3, 40),
            word(0x4, 0b1000_0000_0101),
            word(0x6, 20),
            word(0x2, 31),
        ]);
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(source.sensor_size(), (1280, 720));
        assert_eq!(source.n_events(), 5);
        assert_eq!(source.time_span(), (4106, 4116));

        let all = source.slice_index(0, 10).unwrap();
        assert_eq!(all.xs(), &[30, 40, 42, 51, 31]);
        assert_eq!(all.ys(), &[20, 20, 20, 20, 20]);
        assert_eq!(all.ps(), &[true, false, false, false, false]);
        assert_eq!(all.ts(), &[4106, 4106, 4106, 4106, 4116]);
        assert_eq!(source.slice_index(1, 3).unwrap().xs(), &[40, 42]);
        assert_eq!(source.slice_time(4110, 4120).unwrap().xs(), &[31]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn timestamp_high_wrap_is_monotonic() {
        let path = raw(&[
            word(0x8, 0x0fff),
            word(0x6, 0x0fff),
            word(0x0, 1),
            word(0x2, 1),
            word(0x8, 0),
            word(0x6, 2),
            word(0x2, 2),
        ]);
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        let events = source.slice_index(0, 2).unwrap();
        assert_eq!(events.ts(), &[TIME_LOOP - 1, TIME_LOOP + 2]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn slices_from_a_sparse_checkpoint() {
        let mut words = vec![word(0x8, 0), word(0x6, 10), word(0x0, 4)];
        while words.len() < (CHECKPOINT_BYTES as usize / 2) - 1 {
            words.push(word(0x9, 0));
        }
        words.push(word(0x2, 7));
        words.extend([word(0x6, 20), word(0x2, 8)]);

        let path = raw(&words);
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(source.checkpoints.len(), 2);
        assert_eq!(source.slice_index(1, 2).unwrap().xs(), &[8]);
        assert_eq!(source.slice_time(15, 25).unwrap().xs(), &[8]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_non_evt3_and_accepts_explicit_geometry() {
        let path = std::env::temp_dir().join(format!("eventcv_evt2_{}.raw", std::process::id()));
        std::fs::write(&path, b"% format EVT2\n% geometry 4x3\n").unwrap();
        assert!(matches!(
            open_raw_slice(&path, &LoadOptions::default()),
            Err(IoError::Unsupported(_))
        ));
        std::fs::write(&path, b"% format EVT3\n").unwrap();
        let options = LoadOptions {
            sensor_size: Some((4, 3)),
            ..LoadOptions::default()
        };
        assert_eq!(open_raw_slice(&path, &options).unwrap().sensor_size(), (4, 3));
        std::fs::remove_file(path).ok();
    }
}
