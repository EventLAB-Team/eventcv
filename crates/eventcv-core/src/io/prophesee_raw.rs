//! Prophesee EVT2 and EVT3 `.raw` reader.
//!
//! Both encodings share the extension and an ASCII `% key value` header, and the header's `evt` or
//! `format` line says which one follows.  They are otherwise quite different:
//!
//! - **EVT3** is a stateful stream of little-endian 16-bit words.  A word either updates the current
//!   address/time state or emits one or more CD events, so a decoder carries `y`, `base_x`, polarity
//!   and a timestamp base between words.
//! - **EVT2** is a stream of 32-bit words that are almost self-contained: a CD word carries its own
//!   `x`, `y`, polarity and the low 6 bits of its timestamp, and the only state is the 28-bit
//!   timestamp high half set by the most recent `EVT_TIME_HIGH` word.
//!
//! The recordings this reader targets are routinely tens of gigabytes, so [`RawSliceSource`] builds a
//! sparse byte/time index once and replays from the nearest saved decoder state for each slice.  The
//! event payload is never materialised as a whole.  That machinery is shared: only the word size and
//! the decoder state differ, which is what [`Codec`] abstracts.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::{EventStream, EventStreamBuilder, IoError, LoadOptions, RawEvent, SliceSource};

const TIMESTAMP_SCALE_MS: f64 = 0.001;
const TIME_LOOP: i64 = 1 << 24;
const MAX_TIMESTAMP_BASE: i64 = ((1 << 12) - 1) << 12;
const LOOP_THRESHOLD: i64 = 10 << 12;
const CHECKPOINT_BYTES: u64 = 1 << 20;

/// EVT2 splits a timestamp into a 28-bit high half (its own word) and a 6-bit low half carried by
/// each CD event, so the high half is stored pre-shifted and the wrap period is `1 << 34` µs — about
/// 4.9 hours. Long enough that no realistic recording reaches it, handled anyway because a silently
/// backwards timestamp is far worse than a handled one.
const EVT2_TIME_SHIFT: u32 = 6;
const EVT2_TIME_LOOP: i64 = 1 << (28 + EVT2_TIME_SHIFT);
/// How far a new high half may fall below the current one before it is read as a wrap rather than
/// as the small non-monotonicity a stream can legitimately show.
const EVT2_LOOP_THRESHOLD: i64 = EVT2_TIME_LOOP / 16;

/// Which encoding a file uses. Both are `.raw`, so this comes from the header rather than the path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    Evt2,
    Evt3,
}

/// Decoder state for whichever encoding is in play.
///
/// `Copy` and small, because a checkpoint stores one every megabyte — that is what makes replaying
/// from the middle of a 30 GB file cheap.
#[derive(Clone, Copy, Debug)]
enum Codec {
    Evt2(Evt2Decoder),
    Evt3(Decoder),
}

impl Codec {
    fn new(encoding: Encoding) -> Self {
        match encoding {
            Encoding::Evt2 => Self::Evt2(Evt2Decoder::default()),
            Encoding::Evt3 => Self::Evt3(Decoder::default()),
        }
    }

    /// Bytes per word. The index pass and both slice methods step by this rather than by a literal,
    /// which is the only place the two encodings differ structurally.
    fn word_size(self) -> usize {
        match self {
            Self::Evt2(_) => 4,
            Self::Evt3(_) => 2,
        }
    }

    fn decode(&mut self, bytes: &[u8], emit: impl FnMut(RawEvent)) {
        match self {
            Self::Evt2(decoder) => decoder.decode(
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                emit,
            ),
            Self::Evt3(decoder) => decoder.decode(u16::from_le_bytes([bytes[0], bytes[1]]), emit),
        }
    }

    /// The decoder's current timestamp, for checkpointing and for the early exit in `slice_time`.
    fn time(self) -> i64 {
        match self {
            Self::Evt2(decoder) => decoder.time,
            Self::Evt3(decoder) => decoder.time,
        }
    }

    /// Whether a timestamp has been established yet — before the first TIME_HIGH there is none.
    fn initialised(self) -> bool {
        match self {
            Self::Evt2(decoder) => decoder.initialised,
            Self::Evt3(decoder) => decoder.initialised,
        }
    }
}

/// EVT2: one 32-bit word per event, with a shared timestamp high half.
///
/// | Bits | Meaning |
/// |------|---------|
/// | `31:28` | type — `0x0` CD_OFF, `0x1` CD_ON, `0x8` EVT_TIME_HIGH |
/// | `27:22` | timestamp low 6 bits (CD words) |
/// | `21:11` | x |
/// | `10:0`  | y |
///
/// `EXT_TRIGGER` (`0xA`), `OTHERS` (`0xE`) and `CONTINUED` (`0xF`) carry no CD events and are
/// skipped.
#[derive(Clone, Copy, Debug, Default)]
struct Evt2Decoder {
    initialised: bool,
    /// Timestamp high half, already shifted left by [`EVT2_TIME_SHIFT`] and including any wraps.
    time_high: i64,
    time: i64,
    high_loops: i64,
}

impl Evt2Decoder {
    fn decode(&mut self, word: u32, mut emit: impl FnMut(RawEvent)) {
        match word >> 28 {
            0x8 => {
                // EVT_TIME_HIGH. Wrap accounting mirrors the EVT3 path: a high half that drops a
                // long way rather than a little is a counter rollover, not jitter.
                let mut next = ((i64::from(word & 0x0FFF_FFFF)) << EVT2_TIME_SHIFT)
                    + self.high_loops * EVT2_TIME_LOOP;
                if self.time_high > next
                    && self.time_high - next >= EVT2_TIME_LOOP - EVT2_LOOP_THRESHOLD
                {
                    self.high_loops += 1;
                    next += EVT2_TIME_LOOP;
                }
                self.time_high = next;
                self.time = next;
                self.initialised = true;
            }
            kind @ (0x0 | 0x1) => {
                // A CD word before the first TIME_HIGH has no valid timestamp, so it is dropped —
                // the same rule the EVT3 decoder applies to coordinate words.
                if !self.initialised {
                    return;
                }
                self.time = self.time_high + i64::from((word >> 22) & 0x3F);
                emit(RawEvent {
                    x: ((word >> 11) & 0x7FF) as u16,
                    y: (word & 0x7FF) as u16,
                    t: self.time,
                    p: kind == 0x1,
                });
            }
            _ => {}
        }
    }
}

/// EVT3: a stateful stream of 16-bit words.
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
            0x2 => emit(RawEvent {
                // EVT_ADDR_X
                x: word & 0x07ff,
                y: self.y,
                t: self.time,
                p: word & 0x0800 != 0,
            }),
            0x3 => {
                // VECT_BASE_X
                self.base_x = word & 0x07ff;
                self.polarity = word & 0x0800 != 0;
            }
            0x4 => {
                // VECT_12
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
            0x5 => {
                // VECT_8
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
            0x8 => {
                // EVT_TIME_HIGH, including the 24-bit wrap every 16.777216 s
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
    codec: Codec,
}

#[derive(Debug)]
struct Header {
    data_offset: u64,
    /// `None` when the header has no `% geometry` line and the caller gave no override — some
    /// recorders omit it entirely. The size is then derived from the events themselves; see
    /// [`RawSliceSource::open`].
    sensor_size: Option<(usize, usize)>,
    encoding: Encoding,
}

fn parse_header(
    reader: &mut BufReader<File>,
    override_size: Option<(usize, usize)>,
) -> Result<Header, IoError> {
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
    // Recorders spell the version either way round: `% format EVT3` or `% evt 2.0`.
    let says = |needle: char, prefix: &str| {
        format
            .as_deref()
            .is_some_and(|value| value.starts_with(prefix))
            || evt
                .as_deref()
                .is_some_and(|value| value.starts_with(needle))
    };
    let encoding = if says('3', "evt3") {
        Encoding::Evt3
    } else if says('2', "evt2") {
        Encoding::Evt2
    } else {
        return Err(IoError::Unsupported(format!(
            "unrecognised Prophesee RAW encoding (format={:?}, evt={:?}); this reader handles \
             EVT2 and EVT3",
            format.as_deref().unwrap_or("-"),
            evt.as_deref().unwrap_or("-")
        )));
    };
    // Geometry may be absent — `spinner.raw` from a Gen3 has no such line. Deriving it from the
    // events is handled by the caller, which has already decoded them all for the index.
    let sensor_size = override_size.or(geometry);
    if sensor_size.is_some_and(|(width, height)| width == 0 || height == 0) {
        return Err(IoError::InvalidSensorSize);
    }
    Ok(Header {
        data_offset: reader.stream_position()?,
        sensor_size,
        encoding,
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
        let mut codec = Codec::new(header.encoding);
        let word_size = codec.word_size();
        let mut checkpoints = vec![Checkpoint {
            byte_offset: header.data_offset,
            event_index: 0,
            time: 0,
            codec,
        }];
        let mut event_index = 0usize;
        let mut minimum = i64::MAX;
        let mut maximum = i64::MIN;
        // Largest coordinates seen, used only when the header declared no geometry. Taking the
        // bound from the events themselves means nothing is ever dropped for being out of bounds —
        // which matters because `EventStreamBuilder::push` drops silently.
        let mut extent = (0usize, 0usize);
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
                    time: codec.time(),
                    codec,
                });
            }
            for bytes in chunk.chunks_exact(word_size) {
                codec.decode(bytes, |event| {
                    // With a declared size, out-of-bounds events are not counted, so the index
                    // matches what the slice methods will actually emit. Without one, everything
                    // counts and the size is derived below.
                    let inside = match header.sensor_size {
                        Some((width, height)) => {
                            usize::from(event.x) < width && usize::from(event.y) < height
                        }
                        None => true,
                    };
                    if inside {
                        extent.0 = extent.0.max(usize::from(event.x));
                        extent.1 = extent.1.max(usize::from(event.y));
                        event_index += 1;
                        minimum = minimum.min(event.t);
                        maximum = maximum.max(event.t);
                    }
                });
            }
            byte_offset += chunk.len() as u64;
        }

        // A derived size is a tight bound on the events present, not the physical sensor: a 640x480
        // recording whose rightmost column never fired reports 639 wide. That is the honest answer —
        // rounding up to a guessed "known" sensor would be inventing a number.
        let (width, height) = match header.sensor_size {
            Some(size) => size,
            None if event_index > 0 => (extent.0 + 1, extent.1 + 1),
            None => return Err(IoError::InvalidSensorSize),
        };

        Ok(Self {
            path: path.to_owned(),
            width,
            height,
            checkpoints,
            n_events: event_index,
            time_span: if event_index == 0 {
                (0, 0)
            } else {
                (minimum, maximum)
            },
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
        let mut codec = checkpoint.codec;
        let mut index = checkpoint.event_index;
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        let mut bytes = [0u8; 4];
        let word_size = codec.word_size();
        while index < i1 {
            match reader.read_exact(&mut bytes[..word_size]) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(IoError::Io(error)),
            }
            codec.decode(&bytes[..word_size], |event| {
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
        let mut codec = checkpoint.codec;
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        let mut bytes = [0u8; 4];
        let word_size = codec.word_size();
        loop {
            match reader.read_exact(&mut bytes[..word_size]) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => return Err(IoError::Io(error)),
            }
            codec.decode(&bytes[..word_size], |event| {
                if event.t >= t0
                    && event.t < t1
                    && usize::from(event.x) < self.width
                    && usize::from(event.y) < self.height
                {
                    builder.push(event.x, event.y, event.t, event.p);
                }
            });
            if codec.initialised() && codec.time() >= t1 {
                break;
            }
        }
        Ok(builder.build())
    }
}

/// Opens a Prophesee EVT3 RAW file for bounded-memory random slicing.
pub fn open_raw_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<RawSliceSource, IoError> {
    RawSliceSource::open(path.as_ref(), options)
}

/// Eagerly reads a Prophesee EVT3 RAW file. Prefer [`open_raw_slice`] for large recordings.
pub fn read_raw(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let source = open_raw_slice(path, options)?;
    let limit = options.max_events.unwrap_or(source.n_events());
    source.slice_index(0, limit)
}

/// Which of the two `.raw` encodings [`RawEventSink`] writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvtVersion {
    /// One self-contained 32-bit word per event. Four bytes an event, and a decoder needs no state
    /// beyond the timestamp high half — the default for that reason.
    #[default]
    Evt2,
    /// Stateful 16-bit words. Two bytes an event before any vectorisation, and less than one for a
    /// dense row, at the cost of a decoder that has to carry `y`, `x` base and time across words.
    Evt3,
}

/// Writes a Prophesee `.raw` recording a window at a time — the inverse of [`read_raw`].
///
/// The `%` header goes out on the first non-empty append (it carries `% geometry`, which comes from
/// the stream's sensor size); after that both encodings are pure appends, so the file is readable at
/// any point.
///
/// **Events must arrive in non-decreasing time order.** Both encodings carry the timestamp as a
/// coarse half that is emitted only when it changes, so a backwards step would be encoded as a
/// counter wrap and silently move the event forwards by hours. Out-of-order input is rejected
/// rather than mis-encoded.
pub struct RawEventSink {
    writer: BufWriter<File>,
    version: EvtVersion,
    header_written: bool,
    n_events: usize,
    last_t: Option<i64>,
    /// EVT2: the timestamp high half last emitted. EVT3: `(time_base, time_low)`.
    state: EncoderState,
}

#[derive(Clone, Copy, Debug, Default)]
struct EncoderState {
    /// `t >> EVT2_TIME_SHIFT` (EVT2) or `t >> 12` (EVT3), as last written.
    time_high: Option<i64>,
    /// EVT3 only: the low 12 bits, and the row and polarity the decoder currently holds.
    time_low: Option<i64>,
    y: Option<u16>,
}

impl RawEventSink {
    pub fn create(path: impl AsRef<Path>, version: EvtVersion) -> Result<Self, IoError> {
        Ok(Self {
            writer: BufWriter::new(File::create(path).map_err(IoError::Io)?),
            version,
            header_written: false,
            n_events: 0,
            last_t: None,
            state: EncoderState::default(),
        })
    }

    fn word16(&mut self, kind: u16, payload: u16) -> Result<(), IoError> {
        self.writer
            .write_all(&(((kind & 0xF) << 12) | (payload & 0x0FFF)).to_le_bytes())
            .map_err(IoError::Io)
    }

    fn word32(&mut self, word: u32) -> Result<(), IoError> {
        self.writer.write_all(&word.to_le_bytes()).map_err(IoError::Io)
    }

    /// EVT2: an `EVT_TIME_HIGH` whenever the 28-bit high half moves, then one CD word per event.
    fn append_evt2(&mut self, stream: &EventStream) -> Result<(), IoError> {
        let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
        for index in 0..stream.len() {
            let t = ts[index];
            let high = t >> EVT2_TIME_SHIFT;
            if self.state.time_high != Some(high) {
                // Masked to 28 bits: past that the counter wraps, which is exactly what the
                // decoder's `high_loops` accounting reconstructs.
                self.word32((0x8 << 28) | (high as u32 & 0x0FFF_FFFF))?;
                self.state.time_high = Some(high);
            }
            let low = (t - (high << EVT2_TIME_SHIFT)) as u32 & 0x3F;
            self.word32(
                ((ps[index] as u32) << 28)
                    | (low << 22)
                    | ((u32::from(xs[index]) & 0x7FF) << 11)
                    | (u32::from(ys[index]) & 0x7FF),
            )?;
        }
        Ok(())
    }

    /// EVT3: `EVT_TIME_HIGH` / `EVT_TIME_LOW` when the clock moves, `EVT_ADDR_Y` when the row
    /// changes, then the events themselves.
    ///
    /// A run of events sharing row, time and polarity with ascending `x` is written as a
    /// `VECT_BASE_X` plus one `VECT_12` per 12-pixel window — one word for up to twelve events
    /// instead of one word each. That is what EVT3 is *for*, and such runs are exactly what a
    /// recording decoded from EVT3 in the first place still contains. Anything shorter goes out as
    /// plain `EVT_ADDR_X`, which costs fewer words than a vector's two-word preamble.
    fn append_evt3(&mut self, stream: &EventStream) -> Result<(), IoError> {
        let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
        let mut index = 0;
        while index < stream.len() {
            let (t, y, p) = (ts[index], ys[index], ps[index]);
            let high = t >> 12;
            if self.state.time_high != Some(high) {
                self.word16(0x8, (high & 0xFFF) as u16)?;
                self.state.time_high = Some(high);
                // The decoder resets `time` to the base on a TIME_HIGH, so the low half it holds
                // is no longer whatever we last sent.
                self.state.time_low = Some(0);
            }
            let low = t - (high << 12);
            if self.state.time_low != Some(low) {
                self.word16(0x6, low as u16)?;
                self.state.time_low = Some(low);
            }
            if self.state.y != Some(y) {
                self.word16(0x0, y & 0x07FF)?;
                self.state.y = Some(y);
            }
            // How many events from here share this row, time and polarity with ascending x.
            let mut run = 1;
            while index + run < stream.len()
                && ts[index + run] == t
                && ys[index + run] == y
                && ps[index + run] == p
                && xs[index + run] > xs[index + run - 1]
            {
                run += 1;
            }
            let span = usize::from(xs[index + run - 1] - xs[index]) + 1;
            let vector_words = 1 + span.div_ceil(12);
            if run > 2 && vector_words < run {
                let mut base = xs[index];
                self.word16(0x3, (base & 0x07FF) | ((p as u16) << 11))?;
                let end = index + run;
                let mut cursor = index;
                while cursor < end {
                    let mut mask = 0u16;
                    while cursor < end && xs[cursor] < base + 12 {
                        mask |= 1 << (xs[cursor] - base);
                        cursor += 1;
                    }
                    self.word16(0x4, mask)?;
                    base += 12;
                }
                index = end;
            } else {
                for offset in 0..run {
                    self.word16(
                        0x2,
                        (xs[index + offset] & 0x07FF) | ((p as u16) << 11),
                    )?;
                }
                index += run;
            }
        }
        Ok(())
    }
}

impl super::EventSink for RawEventSink {
    fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        if stream.is_empty() {
            return Ok(());
        }
        let ts = stream.ts();
        for index in 0..stream.len() {
            let previous = if index == 0 { self.last_t } else { Some(ts[index - 1]) };
            if previous.is_some_and(|previous| ts[index] < previous) {
                return Err(IoError::Unsupported(format!(
                    "Prophesee .raw carries the timestamp as a coarse half emitted only when it \
                     changes, so events must be in time order: {} follows {}. Sort first \
                     (stream.sort_by_time()).",
                    ts[index],
                    previous.unwrap_or_default()
                )));
            }
        }
        if !self.header_written {
            let (width, height) = stream.sensor_size();
            let format = match self.version {
                EvtVersion::Evt2 => "EVT2",
                EvtVersion::Evt3 => "EVT3",
            };
            write!(
                self.writer,
                "% format {format}\n% geometry {width}x{height}\n% generator eventcv\n% end\n"
            )
            .map_err(IoError::Io)?;
            self.header_written = true;
        }
        match self.version {
            EvtVersion::Evt2 => self.append_evt2(stream)?,
            EvtVersion::Evt3 => self.append_evt3(stream)?,
        }
        self.last_t = ts.last().copied();
        self.n_events += stream.len();
        Ok(())
    }

    fn n_events(&self) -> usize {
        self.n_events
    }

    fn flush(&mut self) -> Result<(), IoError> {
        self.writer.flush().map_err(IoError::Io)
    }

    fn finish(mut self: Box<Self>) -> Result<(), IoError> {
        self.writer.flush().map_err(IoError::Io)
    }
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
        file.write_all(b"% format EVT3\n% geometry 1280x720\n")
            .unwrap();
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
    fn accepts_both_encodings_and_an_explicit_geometry() {
        // Was `rejects_non_evt3_...`: EVT2 used to be turned away here, and now it is read.
        let path = std::env::temp_dir().join(format!("eventcv_hdr_{}.raw", std::process::id()));

        // EVT2 with a declared geometry.
        std::fs::write(&path, b"% format EVT2\n% geometry 4x3\n").unwrap();
        assert_eq!(
            open_raw_slice(&path, &LoadOptions::default())
                .unwrap()
                .sensor_size(),
            (4, 3)
        );
        // The `% evt 2.0` spelling real recorders use, which carries no `format` line at all.
        std::fs::write(&path, b"% evt 2.0\n% geometry 8x6\n").unwrap();
        assert_eq!(
            open_raw_slice(&path, &LoadOptions::default())
                .unwrap()
                .sensor_size(),
            (8, 6)
        );
        // EVT3 with the size supplied by the caller rather than the header.
        std::fs::write(&path, b"% format EVT3\n").unwrap();
        let options = LoadOptions {
            sensor_size: Some((4, 3)),
            ..LoadOptions::default()
        };
        assert_eq!(
            open_raw_slice(&path, &options).unwrap().sensor_size(),
            (4, 3)
        );

        // An encoding this reader does not know is still refused, and says what it saw.
        std::fs::write(&path, b"% format EVT4\n% geometry 4x3\n").unwrap();
        let error = open_raw_slice(&path, &LoadOptions::default())
            .err()
            .expect("an unknown encoding must be refused");
        assert!(matches!(error, IoError::Unsupported(ref message) if message.contains("EVT2")));
        std::fs::remove_file(path).ok();
    }

    /// An EVT2 word: 4-bit type, then the payload already positioned by the caller.
    fn evt2_word(kind: u32, payload: u32) -> [u8; 4] {
        ((kind << 28) | (payload & 0x0FFF_FFFF)).to_le_bytes()
    }

    /// An EVT2 CD word: 6-bit timestamp low, 11-bit x, 11-bit y.
    fn evt2_cd(polarity: bool, t_low: u32, x: u32, y: u32) -> [u8; 4] {
        evt2_word(
            u32::from(polarity),
            ((t_low & 0x3F) << 22) | ((x & 0x7FF) << 11) | (y & 0x7FF),
        )
    }

    fn evt2_raw(header: &str, words: &[[u8; 4]]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "eventcv_evt2_{}_{}.raw",
            std::process::id(),
            words.len()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(header.as_bytes()).unwrap();
        for bytes in words {
            file.write_all(bytes).unwrap();
        }
        path
    }

    #[test]
    fn evt2_decodes_coordinates_polarity_and_time() {
        let path = evt2_raw(
            "% evt 2.0\n% geometry 640x480\n",
            &[
                evt2_word(0x8, 100),         // TIME_HIGH -> 100 << 6 = 6400 us
                evt2_cd(true, 5, 10, 20),    // ON  at t = 6405
                evt2_cd(false, 9, 300, 400), // OFF at t = 6409
                evt2_word(0xA, 0),           // EXT_TRIGGER — emits nothing
                evt2_word(0xE, 0),           // OTHERS — emits nothing
                evt2_cd(true, 63, 639, 479), // the far corner, at the top of the low range
            ],
        );
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(source.sensor_size(), (640, 480));
        assert_eq!(
            source.n_events(),
            3,
            "trigger and OTHERS words must not become events"
        );

        let stream = source.slice_index(0, 3).unwrap();
        assert_eq!(stream.xs(), &[10, 300, 639]);
        assert_eq!(stream.ys(), &[20, 400, 479]);
        assert_eq!(stream.ts(), &[6405, 6409, 6463]);
        assert_eq!(stream.ps(), &[true, false, true]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn evt2_drops_events_before_the_first_time_high() {
        // Their timestamp is unknowable, so they are discarded rather than stamped with zero —
        // the same rule the EVT3 decoder applies to coordinate words.
        let path = evt2_raw(
            "% evt 2.0\n% geometry 64x48\n",
            &[
                evt2_cd(true, 1, 1, 1),
                evt2_cd(false, 2, 2, 2),
                evt2_word(0x8, 10),
                evt2_cd(true, 3, 3, 3),
            ],
        );
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(source.n_events(), 1);
        assert_eq!(source.slice_index(0, 1).unwrap().xs(), &[3]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn evt2_timestamps_stay_monotonic_across_time_high_words() {
        let mut words = Vec::new();
        for high in 1..=6u32 {
            words.push(evt2_word(0x8, high));
            for low in [0u32, 20, 63] {
                words.push(evt2_cd(true, low, high, low));
            }
        }
        let path = evt2_raw("% evt 2.0\n% geometry 64x64\n", &words);
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        let stream = source.slice_index(0, source.n_events()).unwrap();
        assert_eq!(stream.len(), 18);
        assert!(
            stream.ts().windows(2).all(|w| w[0] <= w[1]),
            "timestamps must not go backwards: {:?}",
            stream.ts()
        );
        // The high half is shifted by six bits, so each TIME_HIGH step is 64 us.
        assert_eq!(stream.ts()[0], 64);
        assert_eq!(stream.ts()[1], 84);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn evt2_derives_the_sensor_size_when_the_header_omits_it() {
        // Real Gen3 recordings (`spinner.raw`) carry no `% geometry` line. The size then comes from
        // the events, which is a tight bound on what is present — so nothing is silently dropped.
        let path = evt2_raw(
            "% evt 2.0\n",
            &[
                evt2_word(0x8, 1),
                evt2_cd(true, 0, 100, 50),
                evt2_cd(false, 1, 42, 90),
            ],
        );
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(
            source.sensor_size(),
            (101, 91),
            "one past the largest coordinate"
        );
        assert_eq!(
            source.n_events(),
            2,
            "no event may be lost to a derived bound"
        );

        // An explicit size still wins over the derivation.
        let options = LoadOptions {
            sensor_size: Some((640, 480)),
            ..LoadOptions::default()
        };
        assert_eq!(
            open_raw_slice(&path, &options).unwrap().sensor_size(),
            (640, 480)
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn evt2_slices_by_index_and_by_time() {
        let mut words = vec![evt2_word(0x8, 1)];
        for i in 0..10u32 {
            words.push(evt2_cd(i % 2 == 0, i, i, i));
        }
        let path = evt2_raw("% evt 2.0\n% geometry 64x64\n", &words);
        let source = open_raw_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(source.n_events(), 10);

        let middle = source.slice_index(3, 6).unwrap();
        assert_eq!(middle.xs(), &[3, 4, 5]);

        // TIME_HIGH 1 -> 64 us, plus the low half, so events land at 64..73.
        let windowed = source.slice_time(66, 69).unwrap();
        assert_eq!(windowed.ts(), &[66, 67, 68]);
        std::fs::remove_file(path).ok();
    }
}
