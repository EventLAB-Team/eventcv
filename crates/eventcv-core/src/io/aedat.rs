//! AEDAT 2.0 reader (jAER format, e.g. DAVIS recordings). The file is an ASCII comment
//! header (lines starting with `#`) followed by big-endian 8-byte records — a 32-bit
//! address and a 32-bit microsecond timestamp. The raw stream interleaves DVS events with
//! APS (frame) and IMU samples; this module decodes the DVS events.
//!
//! DAVIS address layout (verified against a real DAVIS346 recording): a record is a DVS
//! event when bit 31 (APS/ADC sample) and bit 10 (IMU/type) are both clear; then
//! `y = bits 22..30 (9b)`, `x = bits 12..21 (10b)`, `polarity = bit 11`. jAER's y origin is
//! the bottom-left, so we flip rows (`y = height-1 - y`) to the top-left image convention
//! the other readers use — otherwise frames render upside down.
//!
//! Records are a fixed eight bytes, so a byte offset is a record offset and the index that
//! backs [`AedatSliceSource`] can be built in parallel: each span of the body is decodable
//! on its own, unlike the stateful Prophesee EVT3 stream. Only the timestamp needs stitching
//! afterwards — jAER's clock is 32 bits and wraps every ~71 minutes, so each span reports the
//! wraps it saw and a serial pass turns those into an absolute microsecond offset.

use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use super::{EventSource, ImuSample, IoError, LoadOptions, RawEvent, SliceSource};
use crate::representation::{EventFrame, EventFrameData};
use crate::{EventStream, EventStreamBuilder};

/// AEDAT 2.0 timestamps tick in microseconds.
const TIMESTAMP_SCALE_MS: f64 = 0.001;

/// Address plus timestamp, big-endian.
const RECORD: usize = 8;

const APS_SAMPLE_FLAG: u32 = 0x8000_0000; // bit 31: APS/ADC read, not a DVS event
const TYPE_FLAG: u32 = 0x0000_0400; // bit 10: IMU/other, not a DVS event

/// Bits 11..10 of an APS record say which of the two array reads the sample belongs to; the
/// fourth code is not a read at all but the marker jAER uses for an IMU sample.
const READOUT_RESET: u32 = 0;
const READOUT_SIGNAL: u32 = 1;
const READOUT_IMU: u32 = 3;

/// The APS ADC is 10 bits.
const ADC_MASK: u32 = 0x0000_03FF;

/// An IMU sample is seven consecutive records sharing one timestamp: accelerometer x/y/z,
/// temperature, then gyroscope x/y/z, each a 16-bit reading in bits 27..12 tagged by bits 30..28.
const IMU_FIELDS: usize = 7;

/// Full-scale settings of the DAVIS's MPU-6150, indexed by the header's `IMU.AccelFullScale` /
/// `IMU.GyroFullScale` value. Index 0 is also the chip's reset default, so it stands in when the
/// header does not say.
const ACCEL_FULL_SCALE_G: [f64; 4] = [2.0, 4.0, 8.0, 16.0];
const GYRO_FULL_SCALE_DEG: [f64; 4] = [250.0, 500.0, 1000.0, 2000.0];
/// The readings are signed 16-bit, so full scale sits at 2^15.
const IMU_FULL_SCALE: f64 = 32_768.0;
const STANDARD_GRAVITY: f64 = 9.806_65;

/// One index entry per megabyte of records, the stride `prophesee_raw` settled on: fine enough
/// that a slice replays a few thousand records at most, coarse enough that a multi-gigabyte
/// recording indexes into a few thousand entries.
const CHECKPOINT_BYTES: u64 = 1 << 20;

/// jAER writes the timestamp as a 32-bit microsecond counter. A step *backwards* of more than
/// half its range is the counter rolling over (or being reset), not jitter.
const WRAP_THRESHOLD: u32 = 1 << 31;
const WRAP_PERIOD: i64 = 1 << 32;

/// Reads the ASCII comment header, leaving `reader` positioned at the first binary record.
/// Header lines begin with `#`; the first line that does not marks the start of the body.
/// Also returns how many bytes the header occupied, which is where the records begin.
fn read_header<R: BufRead>(reader: &mut R) -> Result<(Vec<String>, u64), IoError> {
    let mut lines = Vec::new();
    let mut consumed = 0u64;
    loop {
        let is_comment = match reader.fill_buf()? {
            [] => break, // EOF: header-only file
            buf => buf[0] == b'#',
        };
        if !is_comment {
            break;
        }
        let mut line = Vec::new();
        reader.read_until(b'\n', &mut line)?;
        consumed += line.len() as u64;
        lines.push(String::from_utf8_lossy(&line).trim_end().to_owned());
    }
    Ok((lines, consumed))
}

fn check_version(header: &[String]) -> Result<(), IoError> {
    match header.first().map(String::as_str) {
        Some(line) if line.contains("AER-DAT2.0") => Ok(()),
        Some(line) if line.contains("AER-DAT") => Err(IoError::Unsupported(format!(
            "unsupported AEDAT version ({line:?}); only AEDAT 2.0 (.aedat) is implemented"
        ))),
        _ => Err(IoError::Format(
            "not an AEDAT file (missing #!AER-DAT header)".to_owned(),
        )),
    }
}

/// DAVIS-family sensors share the DVS bit layout above; resolve the grid from the chip named in
/// the header (or an explicit override).
///
/// The chip is named on the `# AEChip:` / `# HardwareInterface:` lines, so those are searched
/// first: jAER also dumps its entire preferences tree into the header, and that mentions every
/// chip it knows about, which would make a whole-header search answer with the wrong geometry.
fn resolve_sensor(
    header: &[String],
    sensor: Option<(usize, usize)>,
) -> Result<(usize, usize), IoError> {
    if let Some(size) = sensor {
        return Ok(size);
    }
    let names: String = header
        .iter()
        .filter(|line| line.contains("AEChip:") || line.contains("HardwareInterface:"))
        .map(|line| line.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    // DVS128 and the other pre-DAVIS chips encode x/y/polarity in different bits entirely, so
    // they are refused by name rather than silently decoded as noise.
    if names.contains("dvs128") || names.contains("dvs 128") {
        return Err(IoError::Unsupported(
            "this AEDAT 2.0 file is from a DVS128, whose address layout this reader does not \
             decode; only the DAVIS family is supported"
                .to_owned(),
        ));
    }
    for (needles, size) in CHIPS {
        if needles.iter().any(|needle| names.contains(needle)) {
            return Ok(*size);
        }
    }
    Err(IoError::Unsupported(
        "could not determine the DAVIS sensor geometry from the AEDAT header; pass sensor_size"
            .to_owned(),
    ))
}

/// DAVIS chips sharing the address layout, longest name first so `davis346` is not matched by
/// the `davis34` prefix of another entry.
const CHIPS: &[(&[&str], (usize, usize))] = &[
    (&["davis346", "davis 346"], (346, 260)),
    (&["davis240", "davis 240"], (240, 180)),
    (&["davis208", "davis 208"], (208, 192)),
    (&["davis128", "davis 128"], (128, 128)),
];

/// What one record's address encodes. `y` is the raw jAER row (bottom-left origin); the caller
/// flips it, so events and frames stay registered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Record {
    Event {
        x: u16,
        y: u16,
        polarity: bool,
    },
    Aps {
        x: u16,
        y: u16,
        reset: bool,
        adc: u16,
    },
    Imu {
        field: usize,
        reading: i16,
    },
    /// An ADC read type this reader does not use (jAER's "C" cycle), or an IMU field past the
    /// seven it defines.
    Other,
}

fn classify(address: u32) -> Record {
    let x = ((address >> 12) & 0x3FF) as u16;
    let y = ((address >> 22) & 0x1FF) as u16;
    if address & APS_SAMPLE_FLAG == 0 {
        // A DVS record carries nothing below the polarity bit, so bit 10 set without bit 31 is
        // not an event this layout describes.
        return match address & TYPE_FLAG {
            0 => Record::Event {
                x,
                y,
                polarity: address & (1 << 11) != 0,
            },
            _ => Record::Other,
        };
    }
    match (address >> 10) & 3 {
        readout @ (READOUT_RESET | READOUT_SIGNAL) => Record::Aps {
            x,
            y,
            reset: readout == READOUT_RESET,
            adc: (address & ADC_MASK) as u16,
        },
        READOUT_IMU => match ((address >> 28) & 0x7) as usize {
            field if field < IMU_FIELDS => Record::Imu {
                field,
                reading: ((address >> 12) & 0xFFFF) as u16 as i16,
            },
            _ => Record::Other,
        },
        _ => Record::Other,
    }
}

/// Turns the file's 32-bit timestamps into a monotonic microsecond clock by accumulating a
/// wrap offset. Carried in every checkpoint so a slice can resume mid-file.
#[derive(Clone, Copy, Debug, Default)]
struct Clock {
    wrap: i64,
    previous: Option<u32>,
}

impl Clock {
    fn new(wrap: i64) -> Self {
        Self {
            wrap,
            previous: None,
        }
    }

    /// Splits one record into its address and its unwrapped timestamp.
    fn read(&mut self, record: &[u8]) -> (u32, i64) {
        let address = u32::from_be_bytes([record[0], record[1], record[2], record[3]]);
        let raw = u32::from_be_bytes([record[4], record[5], record[6], record[7]]);
        if self
            .previous
            .and_then(|previous| previous.checked_sub(raw))
            .is_some_and(|step| step > WRAP_THRESHOLD)
        {
            self.wrap += WRAP_PERIOD;
        }
        self.previous = Some(raw);
        (address, self.wrap + i64::from(raw))
    }
}

/// Pull-based source over the big-endian record stream, yielding DVS events only.
struct AedatSource<R: BufRead> {
    reader: R,
    clock: Clock,
    width: usize,
    height: usize,
}

impl<R: BufRead> EventSource for AedatSource<R> {
    fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn timestamp_scale_ms(&self) -> f64 {
        TIMESTAMP_SCALE_MS
    }

    fn next_event(&mut self) -> Result<Option<RawEvent>, IoError> {
        let mut record = [0u8; RECORD];
        loop {
            match self.reader.read_exact(&mut record) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
                Err(error) => return Err(IoError::Io(error)),
            }
            let (address, t) = self.clock.read(&record);
            let Record::Event { x, y, polarity } = classify(address) else {
                continue; // APS / IMU sample
            };
            // jAER stores y with the origin at the bottom-left; flip to the top-left image
            // convention the other readers use (out-of-range rows are left for the builder
            // to drop). x is unchanged.
            let y = self
                .height
                .checked_sub(1 + usize::from(y))
                .unwrap_or(self.height) as u16;
            return Ok(Some(RawEvent {
                x,
                y,
                t,
                p: polarity,
            }));
        }
    }
}

fn read_aedat_from<R: BufRead>(
    mut reader: R,
    options: &LoadOptions,
) -> Result<EventStream, IoError> {
    let (header, _) = read_header(&mut reader)?;
    check_version(&header)?;
    let (width, height) = resolve_sensor(&header, options.sensor_size)?;
    super::read_capped(
        AedatSource {
            reader,
            clock: Clock::default(),
            width,
            height,
        },
        options.max_events,
    )
}

/// Reads an AEDAT 2.0 (`.aedat`) recording. `sensor_size` overrides the chip geometry; the
/// timestamp unit is always microseconds. `max_events` caps how many events are kept.
/// Prefer [`open_aedat_slice`] for large recordings — this materialises the whole stream.
pub fn read_aedat(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    read_aedat_from(BufReader::new(File::open(path)?), options)
}

/// One array read, filled sample by sample as it arrives.
struct Plane {
    samples: Vec<u16>,
    written: usize,
    t_first: i64,
}

impl Plane {
    fn new(pixels: usize) -> Self {
        Self {
            samples: vec![0; pixels],
            written: 0,
            t_first: 0,
        }
    }

    fn full(&self) -> bool {
        self.written >= self.samples.len()
    }
}

/// Reassembles APS frames from the ADC samples interleaved into the event stream.
///
/// A DAVIS reads the whole array twice per frame — the sampled signal level, then the reset
/// level — and the pixel value is the difference between them (correlated double sampling).
/// The two reads arrive as separate complete blocks, so a frame is a signal block and the reset
/// block that follows it; a block left unpaired at either end of a slice is dropped. Rows are
/// flipped on the way in, the same way events are, so frames and events stay registered.
struct FrameDecoder {
    width: usize,
    height: usize,
    signal: Plane,
    reset: Plane,
    readout: Option<bool>,
}

impl FrameDecoder {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            signal: Plane::new(width * height),
            reset: Plane::new(width * height),
            readout: None,
        }
    }

    /// Files an APS sample, returning the frame it completed.
    fn push(&mut self, x: u16, y: u16, reset: bool, adc: u16, t: i64) -> Option<(i64, EventFrame)> {
        if self.readout != Some(reset) {
            // A block boundary. A signal block starts a new frame, so anything half-filled from
            // before it — the tail of a pair that began before this slice — is abandoned.
            if reset {
                self.reset.written = 0;
                self.reset.t_first = t;
            } else {
                self.signal.written = 0;
                self.signal.t_first = t;
                self.reset.written = 0;
            }
            self.readout = Some(reset);
        }
        let (x, y) = (usize::from(x), usize::from(y));
        if x < self.width && y < self.height {
            let plane = if reset { &mut self.reset } else { &mut self.signal };
            plane.samples[(self.height - 1 - y) * self.width + x] = adc;
            plane.written += 1;
        }
        (reset && self.signal.full() && self.reset.full())
            .then(|| self.take())
            .flatten()
    }

    fn take(&mut self) -> Option<(i64, EventFrame)> {
        let pixels: Vec<u16> = self
            .reset
            .samples
            .iter()
            .zip(&self.signal.samples)
            // Both reads are 10-bit and the reset level is the higher of the two, so the
            // saturating difference is the clamp to `[0, 1023]` without a second pass.
            .map(|(reset, signal)| reset.saturating_sub(*signal))
            .collect();
        let t = self.signal.t_first;
        self.signal.written = 0;
        self.reset.written = 0;
        self.readout = None;
        EventFrame::intensity(EventFrameData::U16(pixels), self.width, self.height)
            .ok()
            .map(|frame| (t, frame))
    }
}

/// Reassembles IMU samples from the seven records each one is split across.
struct ImuDecoder {
    /// Metres per second squared, and radians per second, per ADC count.
    accel_scale: f64,
    gyro_scale: f64,
    readings: [i16; IMU_FIELDS],
    seen: u8,
    t: i64,
}

impl ImuDecoder {
    /// Scales come from the header's full-scale settings; index 0 (±2 g, ±250 °/s) is the
    /// sensor's reset default and stands in when the header does not say.
    fn new(header: &[String]) -> Self {
        let chip = chip_name(header);
        let scale = |setting: &str, table: [f64; 4]| {
            let setting = chip
                .and_then(|chip| header_key(header, &format!("{chip}.IMU.{setting}")))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            table[setting.min(table.len() - 1)] / IMU_FULL_SCALE
        };
        Self {
            accel_scale: scale("AccelFullScale", ACCEL_FULL_SCALE_G) * STANDARD_GRAVITY,
            gyro_scale: scale("GyroFullScale", GYRO_FULL_SCALE_DEG) * std::f64::consts::PI / 180.0,
            readings: [0; IMU_FIELDS],
            seen: 0,
            t: 0,
        }
    }

    /// Files one of the seven records, returning the sample it completed. The temperature
    /// reading (field 3) is decoded but dropped — [`ImuSample`] carries motion only.
    fn push(&mut self, field: usize, reading: i16, t: i64) -> Option<ImuSample> {
        let bit = 1 << field;
        if field == 0 || self.seen & bit != 0 {
            self.seen = 0; // a repeat means the previous group was cut short
            self.t = t;
        }
        self.readings[field] = reading;
        self.seen |= bit;
        if self.seen != (1 << IMU_FIELDS) - 1 {
            return None;
        }
        self.seen = 0;
        let axes = |base: usize, scale: f64| {
            [0, 1, 2].map(|offset| f64::from(self.readings[base + offset]) * scale)
        };
        Some(ImuSample {
            t_us: self.t,
            linear_acceleration: axes(0, self.accel_scale),
            angular_velocity: axes(4, self.gyro_scale),
        })
    }
}

/// The chip's jAER class name — the last segment of the `# AEChip:` line's class path, which is
/// also the prefix its preference keys carry.
fn chip_name(header: &[String]) -> Option<&str> {
    let line = header.iter().find(|line| line.contains("AEChip:"))?;
    let path = line.split("AEChip:").nth(1)?.trim();
    Some(path.rsplit('.').next().unwrap_or(path))
}

/// The value of the `key="…" value="…"` pair naming exactly `key` in the header.
///
/// jAER writes its entire preferences tree into the header — over half a megabyte of XML on a
/// real recording — so the settings this reader needs are picked out by scanning rather than by
/// parsing the document. The key must match in full: that tree holds an entry for *every* chip
/// jAER knows about, so a suffix match would answer with some other camera's settings.
fn header_key(header: &[String], key: &str) -> Option<String> {
    for line in header {
        let mut rest = line.as_str();
        while let Some((name, tail)) = quoted_after(rest, "key=\"") {
            rest = tail;
            if name == key {
                return quoted_after(tail, "value=\"").map(|(value, _)| value.to_owned());
            }
        }
    }
    None
}

/// The quoted string following the next `marker` in `text`, and what comes after it.
fn quoted_after<'a>(text: &'a str, marker: &str) -> Option<(&'a str, &'a str)> {
    let after = &text[text.find(marker)? + marker.len()..];
    let end = after.find('"')?;
    Some((&after[..end], &after[end + 1..]))
}

/// Where a slice can start replaying: a record boundary, the event count and clock state
/// reached there.
#[derive(Clone, Copy, Debug)]
struct Checkpoint {
    byte_offset: u64,
    event_index: usize,
    /// Unwrapped timestamp of the first record at `byte_offset`.
    time: i64,
    /// Wrap offset in force at `byte_offset`, so replay resumes on the same clock.
    wrap: i64,
}

/// What one parallel scan of a checkpoint-sized span found. Timestamps are relative to the
/// span's own start; the serial pass in [`AedatSliceSource::open`] rebases them.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    /// In-bounds DVS events, matching what `EventStreamBuilder::push` would keep.
    events: usize,
    first_raw: u32,
    last_raw: u32,
    /// Wraps observed inside the span (its boundary wrap is decided by the serial pass).
    wraps: i64,
    first_event: Option<i64>,
    last_event: Option<i64>,
}

/// Scans one span of records without any cross-span state — the reason the index parallelises.
/// `bytes` is a non-zero multiple of [`RECORD`].
fn scan_span(
    path: &Path,
    offset: u64,
    bytes: usize,
    sensor: (usize, usize),
) -> Result<Span, IoError> {
    let (width, height) = sensor;
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut buffer = vec![0u8; bytes];
    file.read_exact(&mut buffer)?;

    let raw_at = |at: usize| u32::from_be_bytes(buffer[at + 4..at + 8].try_into().unwrap());
    let mut span = Span {
        first_raw: raw_at(0),
        last_raw: raw_at(bytes - RECORD),
        ..Span::default()
    };
    let mut clock = Clock::default();
    for record in buffer.chunks_exact(RECORD) {
        let (address, t) = clock.read(record);
        let Record::Event { x, y, .. } = classify(address) else {
            continue;
        };
        if usize::from(x) < width && usize::from(y) < height {
            span.events += 1;
            span.first_event.get_or_insert(t);
            span.last_event = Some(t);
        }
    }
    span.wraps = clock.wrap / WRAP_PERIOD;
    Ok(span)
}

/// Lazy, indexed source for an AEDAT 2.0 recording.
pub struct AedatSliceSource {
    path: PathBuf,
    width: usize,
    height: usize,
    checkpoints: Vec<Checkpoint>,
    n_events: usize,
    time_span: (i64, i64),
    /// The header lines, kept for the IMU full-scale settings the samples are decoded against.
    header: Vec<String>,
}

impl AedatSliceSource {
    fn open(path: &Path, options: &LoadOptions) -> Result<Self, IoError> {
        let mut reader = BufReader::new(File::open(path)?);
        let (header, data_offset) = read_header(&mut reader)?;
        check_version(&header)?;
        let (width, height) = resolve_sensor(&header, options.sensor_size)?;
        drop(reader);

        let body = File::open(path)?.metadata()?.len().saturating_sub(data_offset);
        // A trailing partial record is truncation, not data; ignore it as the sequential
        // reader's `UnexpectedEof` already does.
        let body = body - body % RECORD as u64;
        let spans = span_bounds(data_offset, body);
        let scans = spans
            .par_iter()
            .map(|&(offset, bytes)| scan_span(path, offset, bytes, (width, height)))
            .collect::<Result<Vec<_>, _>>()?;

        let mut checkpoints = Vec::with_capacity(scans.len());
        let mut n_events = 0usize;
        let mut wrap = 0i64;
        let mut previous: Option<u32> = None;
        let mut first = None;
        let mut last = None;
        for (&(byte_offset, _), scan) in spans.iter().zip(&scans) {
            if previous
                .and_then(|previous: u32| previous.checked_sub(scan.first_raw))
                .is_some_and(|step| step > WRAP_THRESHOLD)
            {
                wrap += WRAP_PERIOD;
            }
            checkpoints.push(Checkpoint {
                byte_offset,
                event_index: n_events,
                time: wrap + i64::from(scan.first_raw),
                wrap,
            });
            if let Some(t) = scan.first_event {
                first.get_or_insert(wrap + t);
            }
            if let Some(t) = scan.last_event {
                last = Some(wrap + t);
            }
            n_events += scan.events;
            wrap += scan.wraps * WRAP_PERIOD;
            previous = Some(scan.last_raw);
        }

        Ok(Self {
            path: path.to_path_buf(),
            width,
            height,
            checkpoints,
            n_events,
            time_span: first.zip(last).unwrap_or((0, 0)),
            header,
        })
    }

    fn checkpoint_for_index(&self, index: usize) -> Option<Checkpoint> {
        self.checkpoint(
            self.checkpoints
                .partition_point(|checkpoint| checkpoint.event_index <= index),
        )
    }

    fn checkpoint_for_time(&self, time: i64) -> Option<Checkpoint> {
        self.checkpoint(
            self.checkpoints
                .partition_point(|checkpoint| checkpoint.time <= time),
        )
    }

    /// The checkpoint one before `position`, or `None` for a file with no records.
    fn checkpoint(&self, position: usize) -> Option<Checkpoint> {
        self.checkpoints.get(position.saturating_sub(1)).copied()
    }

    /// Replays records from `checkpoint`, handing each one to `visit` with its unwrapped
    /// timestamp. `visit` returns `false` to stop.
    fn replay(
        &self,
        checkpoint: Option<Checkpoint>,
        mut visit: impl FnMut(Record, i64) -> bool,
    ) -> Result<(), IoError> {
        let Some(checkpoint) = checkpoint else {
            return Ok(());
        };
        let mut reader = BufReader::new(File::open(&self.path)?);
        reader.seek(SeekFrom::Start(checkpoint.byte_offset))?;
        let mut clock = Clock::new(checkpoint.wrap);
        let mut record = [0u8; RECORD];
        loop {
            match reader.read_exact(&mut record) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => return Err(IoError::Io(error)),
            }
            let (address, t) = clock.read(&record);
            if !visit(classify(address), t) {
                return Ok(());
            }
        }
    }

    /// Replays in-bounds DVS events from `checkpoint`, flipped to the image convention and
    /// numbered as they are in [`SliceSource::slice_index`].
    fn replay_events(
        &self,
        checkpoint: Option<Checkpoint>,
        mut visit: impl FnMut(usize, u16, u16, i64, bool) -> bool,
    ) -> Result<(), IoError> {
        let mut index = checkpoint.map_or(0, |checkpoint| checkpoint.event_index);
        self.replay(checkpoint, |record, t| {
            let Record::Event { x, y, polarity } = record else {
                return true;
            };
            if usize::from(x) >= self.width || usize::from(y) >= self.height {
                return true;
            }
            let y = (self.height - 1 - usize::from(y)) as u16;
            if !visit(index, x, y, t, polarity) {
                return false;
            }
            index += 1;
            true
        })
    }
}

impl SliceSource for AedatSliceSource {
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

    fn frames(&self, t0: i64, t1: i64) -> Result<Vec<(i64, EventFrame)>, IoError> {
        let mut decoder = FrameDecoder::new(self.width, self.height);
        let mut frames = Vec::new();
        let mut reading_signal = false;
        // A frame is stamped with the first sample of its signal block, so every frame in the
        // window begins at or after `t0` — starting from `t0`'s checkpoint cannot miss one.
        self.replay(self.checkpoint_for_time(t0), |record, t| {
            let Record::Aps { x, y, reset, adc } = record else {
                return true;
            };
            // Stop once a *new* signal block opens at or after `t1`: that is the first frame the
            // window cannot contain. Testing every signal sample instead would abandon a block
            // that started inside the window and runs past its end, losing that frame.
            if !reset && !reading_signal && t >= t1 {
                return false;
            }
            reading_signal = !reset;
            if let Some((stamp, frame)) = decoder.push(x, y, reset, adc, t) {
                if stamp >= t0 && stamp < t1 {
                    frames.push((stamp, frame));
                }
            }
            true
        })?;
        Ok(frames)
    }

    fn imu(&self, t0: i64, t1: i64) -> Result<Vec<ImuSample>, IoError> {
        let mut decoder = ImuDecoder::new(&self.header);
        let mut samples = Vec::new();
        self.replay(self.checkpoint_for_time(t0), |record, t| {
            if t >= t1 {
                return false;
            }
            if let Record::Imu { field, reading } = record {
                if let Some(sample) = decoder.push(field, reading, t) {
                    if sample.t_us >= t0 {
                        samples.push(sample);
                    }
                }
            }
            true
        })?;
        Ok(samples)
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let i0 = i0.min(self.n_events);
        let i1 = i1.clamp(i0, self.n_events);
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        self.replay_events(self.checkpoint_for_index(i0), |index, x, y, t, p| {
            if index >= i1 {
                return false;
            }
            if index >= i0 {
                builder.push(x, y, t, p);
            }
            true
        })?;
        Ok(builder.build())
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        self.replay_events(self.checkpoint_for_time(t0), |_, x, y, t, p| {
            if t >= t1 {
                return false;
            }
            if t >= t0 {
                builder.push(x, y, t, p);
            }
            true
        })?;
        Ok(builder.build())
    }
}

/// Splits the body into checkpoint-sized spans of whole records: `(byte offset, byte length)`.
fn span_bounds(data_offset: u64, body: u64) -> Vec<(u64, usize)> {
    let mut spans = Vec::new();
    let mut offset = 0;
    while offset < body {
        let bytes = CHECKPOINT_BYTES.min(body - offset);
        spans.push((data_offset + offset, bytes as usize));
        offset += bytes;
    }
    spans
}

/// Opens an AEDAT 2.0 file for bounded-memory random slicing. Building the index costs one
/// parallel pass over the file; slicing afterwards touches only the records it needs.
pub fn open_aedat_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<AedatSliceSource, IoError> {
    AedatSliceSource::open(path.as_ref(), options)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn dvs_address(x: u32, y: u32, polarity: bool) -> u32 {
        (y << 22) | (x << 12) | ((polarity as u32) << 11)
    }

    fn aps_address(x: u32, y: u32, reset: bool, adc: u32) -> u32 {
        let readout = if reset { READOUT_RESET } else { READOUT_SIGNAL };
        APS_SAMPLE_FLAG | (y << 22) | (x << 12) | (readout << 10) | adc
    }

    fn imu_address(field: u32, reading: i16) -> u32 {
        APS_SAMPLE_FLAG | (field << 28) | (u32::from(reading as u16) << 12) | (READOUT_IMU << 10)
    }

    /// The seven records one IMU sample is split across: accel x/y/z, temperature, gyro x/y/z.
    fn imu_records(readings: [i16; 7], timestamp: u32) -> Vec<[u8; 8]> {
        (0..7)
            .map(|field| record(imu_address(field as u32, readings[field]), timestamp))
            .collect()
    }

    /// One array read of a 2×2 sensor, in the order the sensor scans it.
    fn aps_block(reset: bool, adc: [u32; 4], timestamp: u32) -> Vec<[u8; 8]> {
        [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .zip(adc)
            .map(|(&(x, y), adc)| record(aps_address(x, y, reset, adc), timestamp))
            .collect()
    }

    fn record(address: u32, timestamp: u32) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&address.to_be_bytes());
        bytes[4..8].copy_from_slice(&timestamp.to_be_bytes());
        bytes
    }

    fn file_with(records: &[[u8; 8]]) -> Vec<u8> {
        let mut data =
            b"#!AER-DAT2.0\r\n# AEChip: eu.seebetter.ini.chips.davis.Davis346red\r\n".to_vec();
        for record in records {
            data.extend_from_slice(record);
        }
        data
    }

    fn read(data: &[u8], options: &LoadOptions) -> Result<EventStream, IoError> {
        read_aedat_from(Cursor::new(data.to_vec()), options)
    }

    /// Writes `data` to a temporary `.aedat` and opens it for slicing.
    fn slice_source(data: &[u8], options: &LoadOptions) -> (PathBuf, AedatSliceSource) {
        use std::io::Write;
        use std::sync::atomic::{AtomicUsize, Ordering};
        // Tests run in parallel and several build the same bytes, so the name needs a counter,
        // not a hash of the contents.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "eventcv_aedat_{}_{}.aedat",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        File::create(&path).unwrap().write_all(data).unwrap();
        let source = open_aedat_slice(&path, options).unwrap();
        (path, source)
    }

    #[test]
    fn decodes_dvs_events_and_skips_aps_imu_samples() {
        let data = file_with(&[
            record(dvs_address(100, 50, true), 1_000),
            record(0x8000_0000, 1_001), // APS sample (bit 31) -> skipped
            record(TYPE_FLAG | 0x1234, 1_002), // IMU/type (bit 10) -> skipped
            record(dvs_address(200, 10, false), 1_005),
        ]);
        let stream = read(&data, &LoadOptions::default()).unwrap();

        assert_eq!(stream.sensor_size(), (346, 260)); // inferred from the Davis346 chip line
        assert_eq!(stream.len(), 2);
        assert_eq!(stream.xs(), &[100, 200]);
        assert_eq!(stream.ys(), &[209, 249]); // raw 50/10 flipped to top-left (259 - y)
        assert_eq!(stream.ts(), &[1_000, 1_005]);
        assert_eq!(stream.ps(), &[true, false]);
    }

    #[test]
    fn header_only_file_is_empty() {
        let stream = read(&file_with(&[]), &LoadOptions::default()).unwrap();
        assert!(stream.is_empty());
        assert_eq!(stream.sensor_size(), (346, 260));
    }

    #[test]
    fn out_of_bounds_events_are_dropped_by_the_builder() {
        let data = file_with(&[
            record(dvs_address(1, 1, true), 1),
            record(dvs_address(300, 5, true), 2), // x >= width -> dropped
        ]);
        let options = LoadOptions {
            sensor_size: Some((4, 4)),
            ..LoadOptions::default()
        };
        let stream = read(&data, &options).unwrap();
        assert_eq!(stream.len(), 1);
        assert_eq!(stream.xs(), &[1]);
    }

    #[test]
    fn rows_are_flipped_to_the_top_left_image_convention() {
        // Raw bottom row (y=0) maps to the top (height-1); raw top row maps to 0.
        let data = file_with(&[
            record(dvs_address(0, 0, true), 1),
            record(dvs_address(1, 3, true), 2),
        ]);
        let options = LoadOptions {
            sensor_size: Some((4, 4)),
            ..LoadOptions::default()
        };
        let stream = read(&data, &options).unwrap();
        assert_eq!(stream.ys(), &[3, 0]);
    }

    #[test]
    fn max_events_caps_the_read() {
        let data = file_with(&[
            record(dvs_address(1, 1, true), 1),
            record(dvs_address(2, 2, true), 2),
            record(dvs_address(3, 3, true), 3),
        ]);
        let options = LoadOptions {
            max_events: Some(2),
            ..LoadOptions::default()
        };
        assert_eq!(read(&data, &options).unwrap().len(), 2);
    }

    #[test]
    fn other_aedat_versions_are_unsupported() {
        let data = b"#!AER-DAT3.1\r\n".to_vec();
        match read(&data, &LoadOptions::default()) {
            Err(IoError::Unsupported(message)) => assert!(message.contains("AEDAT 2.0")),
            other => panic!("expected unsupported version error, got {other:?}"),
        }
    }

    #[test]
    fn missing_header_is_a_format_error() {
        match read(b"not an aedat file", &LoadOptions::default()) {
            Err(IoError::Format(message)) => assert!(message.contains("AER-DAT")),
            other => panic!("expected a format error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_chip_without_override_is_unsupported() {
        let mut data = b"#!AER-DAT2.0\r\n# AEChip: SomeUnknownChip\r\n".to_vec();
        data.extend_from_slice(&record(dvs_address(1, 1, true), 1));
        match read(&data, &LoadOptions::default()) {
            Err(IoError::Unsupported(message)) => assert!(message.contains("sensor_size")),
            other => panic!("expected unsupported geometry error, got {other:?}"),
        }
    }

    #[test]
    fn a_preferences_dump_naming_other_chips_does_not_decide_the_geometry() {
        // jAER writes every chip's preferences into the header; only the AEChip line counts.
        let mut data = b"#!AER-DAT2.0\r\n# AEChip: eu.seebetter.ini.chips.davis.Davis240C\r\n"
            .to_vec();
        data.extend_from_slice(b"#  <entry key=\"Davis346red.APS.Run\" value=\"true\"/>\r\n");
        data.extend_from_slice(&record(dvs_address(1, 1, true), 1));
        assert_eq!(
            read(&data, &LoadOptions::default()).unwrap().sensor_size(),
            (240, 180)
        );
    }

    #[test]
    fn a_dvs128_recording_is_refused_rather_than_mis_decoded() {
        let mut data = b"#!AER-DAT2.0\r\n# AEChip: ch.unizh.ini.caviar.chip.retina.DVS128\r\n"
            .to_vec();
        data.extend_from_slice(&record(dvs_address(1, 1, true), 1));
        match read(&data, &LoadOptions::default()) {
            Err(IoError::Unsupported(message)) => assert!(message.contains("DVS128")),
            other => panic!("expected an unsupported-chip error, got {other:?}"),
        }
    }

    #[test]
    fn the_32_bit_timestamp_clock_is_unwrapped() {
        let data = file_with(&[
            record(dvs_address(1, 1, true), u32::MAX - 1),
            record(dvs_address(2, 2, true), 5), // counter rolled over
            record(dvs_address(3, 3, true), 9),
        ]);
        let stream = read(&data, &LoadOptions::default()).unwrap();
        assert_eq!(
            stream.ts(),
            &[
                i64::from(u32::MAX - 1),
                WRAP_PERIOD + 5,
                WRAP_PERIOD + 9
            ]
        );
    }

    #[test]
    fn slices_agree_with_a_whole_file_read() {
        let records: Vec<_> = (0..4_000u32)
            .map(|i| record(dvs_address(i % 300, i % 200, i % 2 == 0), i * 10))
            .collect();
        let data = file_with(&records);
        let (path, source) = slice_source(&data, &LoadOptions::default());
        let whole = read(&data, &LoadOptions::default()).unwrap();

        assert_eq!(source.n_events(), whole.len());
        assert_eq!(source.time_span(), (0, 39_990));
        assert_eq!(source.sensor_size(), (346, 260));

        let head = source.slice_index(0, 10).unwrap();
        assert_eq!(head.xs(), &whole.xs()[..10]);
        assert_eq!(head.ts(), &whole.ts()[..10]);

        // A window that starts past the first checkpoint exercises the index.
        let middle = source.slice_index(3_000, 3_010).unwrap();
        assert_eq!(middle.xs(), &whole.xs()[3_000..3_010]);
        assert_eq!(middle.ys(), &whole.ys()[3_000..3_010]);

        let window = source.slice_time(1_000, 1_100).unwrap();
        assert_eq!(window.ts(), &[1_000, 1_010, 1_020, 1_030, 1_040, 1_050, 1_060, 1_070, 1_080, 1_090]);

        std::fs::remove_file(&path).ok();
    }

    /// A file with a header naming `chip`, one frame's worth of APS blocks and one IMU sample.
    fn aps_and_imu_file(settings: &str) -> Vec<u8> {
        let mut data =
            b"#!AER-DAT2.0\r\n# AEChip: eu.seebetter.ini.chips.davis.Davis346red\r\n".to_vec();
        data.extend_from_slice(settings.as_bytes());
        let mut records = Vec::new();
        // A truncated reset block first: the tail of a pair that began before this file, which
        // must not be merged into the frame that follows.
        records.push(record(aps_address(0, 0, true, 999), 5));
        records.extend(aps_block(false, [10, 20, 30, 40], 10)); // signal
        records.push(record(dvs_address(1, 1, true), 11)); // events interleave with the blocks
        records.extend(aps_block(true, [100, 100, 100, 100], 20)); // reset
        records.extend(imu_records([-16_384, 0, 0, 8_000, 16_384, 0, 0], 30));
        for record in &records {
            data.extend_from_slice(record);
        }
        data
    }

    fn two_by_two() -> LoadOptions {
        LoadOptions {
            sensor_size: Some((2, 2)),
            ..LoadOptions::default()
        }
    }

    #[test]
    fn aps_blocks_pair_into_a_correlated_double_sampled_frame() {
        let (path, source) = slice_source(&aps_and_imu_file(""), &two_by_two());
        let frames = source.frames(0, i64::MAX).unwrap();
        assert_eq!(frames.len(), 1);

        let (t, frame) = &frames[0];
        assert_eq!(*t, 10); // stamped with the start of the signal block
        assert_eq!(frame.shape(), (1, 2, 2));
        // reset - signal, with jAER's bottom-left rows flipped to the top-left convention.
        assert_eq!(
            frame.data(),
            &EventFrameData::U16(vec![70, 60, 90, 80]),
            "rows flipped and differenced"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn frames_outside_the_window_are_not_returned() {
        let (path, source) = slice_source(&aps_and_imu_file(""), &two_by_two());
        assert!(source.frames(11, i64::MAX).unwrap().is_empty());
        assert!(source.frames(0, 10).unwrap().is_empty());
        assert_eq!(source.frames(10, 11).unwrap().len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_frame_whose_readout_runs_past_the_window_is_still_returned() {
        // The signal block starts at 10 and the pair completes at 20. A window ending at 15 cuts
        // through the readout, but the frame belongs to it: its timestamp is inside.
        let (path, source) = slice_source(&aps_and_imu_file(""), &two_by_two());
        let frames = source.frames(0, 15).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].0, 10);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn imu_records_decode_to_si_units() {
        let (path, source) = slice_source(&aps_and_imu_file(""), &two_by_two());
        let samples = source.imu(0, i64::MAX).unwrap();
        assert_eq!(samples.len(), 1);

        let sample = samples[0];
        assert_eq!(sample.t_us, 30);
        // Half of the default ±2 g scale is one g, and half of ±250 °/s is 125 °/s.
        assert!((sample.linear_acceleration[0] + STANDARD_GRAVITY).abs() < 1e-9);
        assert_eq!(sample.linear_acceleration[1..], [0.0, 0.0]);
        assert!((sample.angular_velocity[0] - 125_f64.to_radians()).abs() < 1e-9);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_headers_full_scale_settings_scale_the_imu() {
        // jAER dumps every chip's settings, so the entry that counts is the recording chip's —
        // reading any other camera's would scale the samples by the wrong factor.
        let settings = "#  <entry key=\"DAVIS240A.IMU.AccelFullScale\" value=\"3\"/>\r\n\
                        #  <entry key=\"Davis346red.IMU.AccelFullScale\" value=\"2\"/>\r\n";
        let (path, source) = slice_source(&aps_and_imu_file(settings), &two_by_two());
        let sample = source.imu(0, i64::MAX).unwrap()[0];
        // Setting 2 is ±8 g, so the same reading is four times the acceleration.
        assert!((sample.linear_acceleration[0] + 4.0 * STANDARD_GRAVITY).abs() < 1e-9);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn formats_without_auxiliary_streams_report_none() {
        // The trait defaults are what every other reader inherits.
        let (path, source) = slice_source(&file_with(&[record(dvs_address(1, 1, true), 1)]), &two_by_two());
        assert!(source.frames(0, i64::MAX).unwrap().is_empty());
        assert!(source.imu(0, i64::MAX).unwrap().is_empty());
        assert!(source.camera().unwrap().is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn slicing_an_empty_body_yields_nothing() {
        let (path, source) = slice_source(&file_with(&[]), &LoadOptions::default());
        assert_eq!(source.n_events(), 0);
        assert_eq!(source.time_span(), (0, 0));
        assert!(source.slice_index(0, 10).unwrap().is_empty());
        assert!(source.slice_time(0, 1_000).unwrap().is_empty());
        std::fs::remove_file(&path).ok();
    }
}
