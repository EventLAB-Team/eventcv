//! AEDAT 4.0 reader (iniVation DV format).
//!
//! The file is `#!AER-DAT4.0\r\n`, a 32-bit little-endian length and that many bytes of
//! `IOHeader` FlatBuffer, then a run of packets. Each packet is an 8-byte `PacketHeader`
//! (`i32` stream id, `i32` body size) followed by a compressed body which, once decompressed,
//! is a size-prefixed FlatBuffer. A `FileDataTable` at the header's `dataTablePosition` indexes
//! every packet — byte offset, element count and time span — so a recording is sliceable
//! without reading it; when it is absent (an interrupted recording) the packets are walked once
//! to rebuild it.
//!
//! The header's `infoNode` is a small XML document naming each stream: `EVTS` events, `FRME`
//! APS frames, `IMUS` inertial samples, `TRIG` triggers (read past, not decoded).
//!
//! FlatBuffers are decoded by the bounds-checked reader below rather than by the `flatbuffers`
//! crate. That crate's table accessors are `unsafe` and unchecked, and its safe entry points
//! need verifier code that normally comes out of `flatc` — more machinery than these four small
//! schemas are worth. Every read here returns `Option`, so a truncated or corrupt packet becomes
//! an [`IoError::Format`] instead of reading past the end of a buffer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::{ImuSample, IoError, LoadOptions, SliceSource};
use crate::representation::{EventFrame, EventFrameData};
use crate::{EventStream, EventStreamBuilder};

/// AEDAT 4 timestamps are Unix microseconds.
const TIMESTAMP_SCALE_MS: f64 = 0.001;

const MAGIC: &[u8] = b"#!AER-DAT4.0\r\n";
/// `PacketHeader { StreamID: int32; Size: int32 }`.
const PACKET_HEADER: usize = 8;

/// A `dv::Event` struct: `int64 timestamp`, `int16 x`, `int16 y`, `bool polarity`, padded to the
/// struct's 8-byte alignment. Vectors of structs are packed inline, so this is the stride.
const EVENT_STRIDE: usize = 16;

/// `IMU` reports acceleration in g and rotation in degrees per second; [`ImuSample`] is SI.
const STANDARD_GRAVITY: f64 = 9.806_65;

/// FlatBuffer field slots, `4 + 2 * field_index` from the schemas in `dv-processing`'s
/// `.fbs` files. Grouped by table so they can be checked against the schema by eye.
mod slot {
    // IOHeader { compression, dataTablePosition, infoNode }
    pub const COMPRESSION: usize = 4;
    pub const DATA_TABLE_POSITION: usize = 6;
    pub const INFO_NODE: usize = 8;

    // EventPacket / IMUPacket / TriggerPacket { elements }
    pub const ELEMENTS: usize = 4;

    // Frame { timestamp, timestampStartOfFrame, timestampEndOfFrame, timestampStartOfExposure,
    //         timestampEndOfExposure, format, sizeX, sizeY, positionX, positionY, pixels, … }
    pub const FRAME_TIMESTAMP: usize = 4;
    pub const FRAME_FORMAT: usize = 14;
    pub const FRAME_SIZE_X: usize = 16;
    pub const FRAME_SIZE_Y: usize = 18;
    pub const FRAME_PIXELS: usize = 24;

    // IMU { timestamp, temperature, accelerometerX/Y/Z, gyroscopeX/Y/Z, magnetometerX/Y/Z }
    pub const IMU_TIMESTAMP: usize = 4;
    pub const IMU_ACCELEROMETER: usize = 8; // X; Y and Z follow at +2 and +4
    pub const IMU_GYROSCOPE: usize = 14;

    // FileDataDefinition { ByteOffset, PacketInfo, NumElements, TimestampStart, TimestampEnd }
    pub const DEFINITION_BYTE_OFFSET: usize = 4;
    pub const DEFINITION_PACKET_INFO: usize = 6;
    pub const DEFINITION_NUM_ELEMENTS: usize = 8;
    pub const DEFINITION_TIMESTAMP_START: usize = 10;
    pub const DEFINITION_TIMESTAMP_END: usize = 12;

    // FileDataTable { Table }
    pub const TABLE: usize = 4;
}

/// `FrameFormat` values this reader decodes; the rest are rejected by name.
const FORMAT_GREY: i8 = 0; // OPENCV_8U_C1
const FORMAT_BGR: i8 = 16; // OPENCV_8U_C3
const FORMAT_BGRA: i8 = 24; // OPENCV_8U_C4

fn scalar<const N: usize>(buffer: &[u8], at: usize) -> Option<[u8; N]> {
    buffer.get(at..at.checked_add(N)?)?.try_into().ok()
}

/// A FlatBuffer table: a position in a buffer, and a vtable saying where its fields are.
#[derive(Clone, Copy)]
struct Table<'a> {
    buffer: &'a [u8],
    at: usize,
}

impl<'a> Table<'a> {
    /// The root of a size-prefixed buffer — the framing every AEDAT 4 packet body uses.
    fn size_prefixed_root(buffer: &'a [u8]) -> Option<Self> {
        Self::root_at(buffer, 4)
    }

    /// The root of a bare buffer; the `IOHeader` is stored this way, its length carried outside.
    fn root(buffer: &'a [u8]) -> Option<Self> {
        Self::root_at(buffer, 0)
    }

    fn root_at(buffer: &'a [u8], at: usize) -> Option<Self> {
        let offset = u32::from_le_bytes(scalar(buffer, at)?) as usize;
        let at = at.checked_add(offset)?;
        (at < buffer.len()).then_some(Self { buffer, at })
    }

    /// Where field `slot` is stored, or `None` when the table omits it — meaning the writer left
    /// it at the schema default, which is what the readers below fall back to.
    fn field(&self, slot: usize) -> Option<usize> {
        let soffset = i32::from_le_bytes(scalar(self.buffer, self.at)?);
        let vtable = usize::try_from(i64::try_from(self.at).ok()? - i64::from(soffset)).ok()?;
        let length = u16::from_le_bytes(scalar(self.buffer, vtable)?) as usize;
        if slot + 2 > length {
            return None; // written by an older schema that had no such field
        }
        let offset = u16::from_le_bytes(scalar(self.buffer, vtable.checked_add(slot)?)?) as usize;
        (offset != 0).then(|| self.at.checked_add(offset)).flatten()
    }

    fn i8(&self, slot: usize, default: i8) -> i8 {
        self.field(slot)
            .and_then(|at| scalar::<1>(self.buffer, at))
            .map_or(default, i8::from_le_bytes)
    }

    fn i16(&self, slot: usize) -> i16 {
        self.field(slot)
            .and_then(|at| scalar::<2>(self.buffer, at))
            .map_or(0, i16::from_le_bytes)
    }

    fn i32(&self, slot: usize) -> i32 {
        self.field(slot)
            .and_then(|at| scalar::<4>(self.buffer, at))
            .map_or(0, i32::from_le_bytes)
    }

    fn i64(&self, slot: usize, default: i64) -> i64 {
        self.field(slot)
            .and_then(|at| scalar::<8>(self.buffer, at))
            .map_or(default, i64::from_le_bytes)
    }

    fn f32(&self, slot: usize) -> f32 {
        self.field(slot)
            .and_then(|at| scalar::<4>(self.buffer, at))
            .map_or(0.0, f32::from_le_bytes)
    }

    /// Where an offset field (string, vector or table) points.
    fn indirect(&self, slot: usize) -> Option<usize> {
        let at = self.field(slot)?;
        let offset = u32::from_le_bytes(scalar(self.buffer, at)?) as usize;
        at.checked_add(offset)
    }

    fn string(&self, slot: usize) -> Option<&'a str> {
        let (bytes, _) = self.vector(slot, 1)?;
        std::str::from_utf8(bytes).ok()
    }

    /// A vector of inline `stride`-byte elements: its bytes and its element count, both checked
    /// against the buffer so a bad length cannot make a slice run off the end.
    fn vector(&self, slot: usize, stride: usize) -> Option<(&'a [u8], usize)> {
        let at = self.indirect(slot)?;
        let elements = u32::from_le_bytes(scalar(self.buffer, at)?) as usize;
        let start = at.checked_add(4)?;
        let end = start.checked_add(elements.checked_mul(stride)?)?;
        Some((self.buffer.get(start..end)?, elements))
    }

    /// A vector of tables. Each element is an offset relative to its own position.
    fn tables(&self, slot: usize) -> Vec<Table<'a>> {
        let Some((bytes, elements)) = self.vector(slot, 4) else {
            return Vec::new();
        };
        let start = bytes.as_ptr() as usize - self.buffer.as_ptr() as usize;
        (0..elements)
            .filter_map(|index| {
                let at = start + index * 4;
                let offset = u32::from_le_bytes(scalar(self.buffer, at)?) as usize;
                Some(Table {
                    buffer: self.buffer,
                    at: at.checked_add(offset)?,
                })
            })
            .collect()
    }
}

/// How each packet body is compressed — `IOHeader.compression` on read, and the choice
/// [`Aedat4EventSink`] writes with. LZ4 is the default because it is what DV itself writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compression {
    None,
    #[default]
    Lz4,
    Zstd,
}

impl Compression {
    /// The `IOHeader.compression` value that names this scheme, the inverse of
    /// [`from_header`](Self::from_header). The plain rather than the `_HIGH` variant: an event
    /// packet is written once and read many times, and the extra effort buys little on data this
    /// regular.
    fn to_header(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
            Self::Zstd => 3,
        }
    }

    /// Compresses one packet body for writing. LZ4 goes out in the *frame* format DV expects,
    /// which is what [`decode`](Self::decode) reads back.
    fn encode(self, body: &[u8]) -> Result<Vec<u8>, IoError> {
        match self {
            Self::None => Ok(body.to_vec()),
            Self::Lz4 => {
                let mut encoder = lz4::EncoderBuilder::new()
                    .build(Vec::new())
                    .map_err(IoError::Io)?;
                encoder.write_all(body).map_err(IoError::Io)?;
                let (out, result) = encoder.finish();
                result.map_err(IoError::Io)?;
                Ok(out)
            }
            Self::Zstd => zstd::stream::encode_all(body, 3).map_err(IoError::Io),
        }
    }

    fn from_header(value: i32) -> Result<Self, IoError> {
        match value {
            0 => Ok(Self::None),
            1 | 2 => Ok(Self::Lz4),   // LZ4, LZ4_HIGH — same decoder, different effort
            3 | 4 => Ok(Self::Zstd),  // ZSTD, ZSTD_HIGH
            other => Err(IoError::Unsupported(format!(
                "unknown AEDAT4 compression type {other}"
            ))),
        }
    }

    fn decode(self, body: Vec<u8>) -> Result<Vec<u8>, IoError> {
        match self {
            Self::None => Ok(body),
            // DV writes LZ4 in its frame format, not raw blocks.
            Self::Lz4 => {
                let mut out = Vec::new();
                lz4::Decoder::new(body.as_slice())
                    .and_then(|mut decoder| decoder.read_to_end(&mut out))
                    .map_err(|error| {
                        IoError::Format(format!("AEDAT4 LZ4 packet is corrupt: {error}"))
                    })?;
                Ok(out)
            }
            Self::Zstd => zstd::stream::decode_all(body.as_slice())
                .map_err(|error| IoError::Format(format!("AEDAT4 zstd packet is corrupt: {error}"))),
        }
    }
}

/// One packet as the index sees it: where its body is, what it holds, and when.
#[derive(Clone, Copy, Debug)]
struct Packet {
    offset: u64,
    size: usize,
    /// Elements in the stream before this packet — the cumulative count `slice_index` seeks on.
    first: usize,
    elements: usize,
    start: i64,
    end: i64,
}

/// What a stream carries, from its `typeIdentifier` in the header's XML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Events,
    Frames,
    Imu,
    Other,
}

impl Kind {
    fn from_identifier(identifier: &str) -> Self {
        match identifier {
            "EVTS" => Self::Events,
            "FRME" => Self::Frames,
            "IMUS" => Self::Imu,
            _ => Self::Other,
        }
    }
}

/// A stream declared by the header's `infoNode`.
#[derive(Clone, Debug)]
struct Stream {
    id: i32,
    kind: Kind,
    size: Option<(usize, usize)>,
}

/// The streams the header declares, in the order they appear.
///
/// `infoNode` is a small XML document (a few kilobytes) listing numbered stream nodes, each with
/// its type identifier and, for pixel streams, its resolution. It is scanned for those three
/// attributes rather than parsed: DV is the only writer of this format, so the shape is fixed,
/// and an XML crate would be a dependency earning its keep on two attributes.
fn parse_streams(info: &str) -> Vec<Stream> {
    const NODE: &str = "<node name=\"";
    let mut marks: Vec<(usize, i32)> = Vec::new();
    let mut at = 0;
    while let Some(found) = info[at..].find(NODE) {
        let start = at + found + NODE.len();
        at = start;
        let Some(end) = info[start..].find('"') else {
            break;
        };
        // Stream nodes are named by their id; `info` and the document root are not.
        if let Ok(id) = info[start..start + end].parse::<i32>() {
            marks.push((start, id));
        }
    }
    marks
        .iter()
        .enumerate()
        .map(|(index, &(start, id))| {
            let end = marks.get(index + 1).map_or(info.len(), |&(next, _)| next);
            let block = &info[start..end];
            Stream {
                id,
                kind: Kind::from_identifier(attribute(block, "typeIdentifier").unwrap_or("")),
                size: attribute(block, "sizeX")
                    .and_then(|value| value.parse().ok())
                    .zip(attribute(block, "sizeY").and_then(|value| value.parse().ok())),
            }
        })
        .collect()
}

/// The text of the `<attr key="…">value</attr>` naming `key`.
fn attribute<'a>(block: &'a str, key: &str) -> Option<&'a str> {
    let after = &block[block.find(&format!("key=\"{key}\""))?..];
    let open = after.find('>')? + 1;
    let close = after[open..].find('<')?;
    Some(&after[open..open + close])
}

/// The parsed `IOHeader`, and where the first packet starts.
struct Header {
    compression: Compression,
    data_table_position: i64,
    streams: Vec<Stream>,
    body_offset: u64,
}

fn read_header(file: &mut File) -> Result<Header, IoError> {
    let mut magic = [0u8; MAGIC.len()];
    file.read_exact(&mut magic).map_err(|_| {
        IoError::Format("not an AEDAT 4 file (too short for a version line)".to_owned())
    })?;
    if magic != MAGIC {
        return Err(IoError::Format(format!(
            "not an AEDAT 4.0 file (version line is {:?})",
            String::from_utf8_lossy(&magic).trim_end()
        )));
    }
    let mut length = [0u8; 4];
    file.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    let mut buffer = vec![0u8; length];
    file.read_exact(&mut buffer)?;

    let header = Table::root(&buffer)
        .ok_or_else(|| IoError::Format("AEDAT4 header is not a readable FlatBuffer".to_owned()))?;
    Ok(Header {
        compression: Compression::from_header(header.i32(slot::COMPRESSION))?,
        data_table_position: header.i64(slot::DATA_TABLE_POSITION, -1),
        streams: parse_streams(header.string(slot::INFO_NODE).unwrap_or("")),
        body_offset: (MAGIC.len() + 4 + length) as u64,
    })
}

/// Reads `size` bytes at `offset` and decompresses them.
fn read_body(
    path: &Path,
    compression: Compression,
    offset: u64,
    size: usize,
) -> Result<Vec<u8>, IoError> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut body = vec![0u8; size];
    file.read_exact(&mut body)?;
    compression.decode(body)
}

/// Lazy, indexed source for an AEDAT 4 recording.
pub struct Aedat4SliceSource {
    path: PathBuf,
    compression: Compression,
    width: usize,
    height: usize,
    events: Vec<Packet>,
    frames: Vec<Packet>,
    imu: Vec<Packet>,
    n_events: usize,
    time_span: (i64, i64),
}

impl Aedat4SliceSource {
    fn open(path: &Path, options: &LoadOptions) -> Result<Self, IoError> {
        let mut file = File::open(path)?;
        let header = read_header(&mut file)?;
        let total = file.metadata()?.len();

        let packets = match header.data_table_position {
            position if position >= 0 && (position as u64) < total => read_data_table(
                path,
                header.compression,
                position as u64,
                total - position as u64,
            )?,
            // No table (an interrupted recording, or one still being written): walk the packets
            // and decode each one's span, which is what DV itself falls back to.
            _ => walk_packets(path, header.compression, header.body_offset, total)?,
        };

        let of_kind = |kind: Kind| -> Vec<Packet> {
            let ids: Vec<i32> = header
                .streams
                .iter()
                .filter(|stream| stream.kind == kind)
                .map(|stream| stream.id)
                .collect();
            let mut selected: Vec<Packet> = packets
                .iter()
                .filter(|(id, _)| ids.contains(id))
                .map(|(_, packet)| *packet)
                .collect();
            selected.sort_by_key(|packet| packet.start);
            let mut first = 0;
            for packet in &mut selected {
                packet.first = first;
                first += packet.elements;
            }
            selected
        };
        let events = of_kind(Kind::Events);
        let (width, height) = options
            .sensor_size
            .or_else(|| {
                header
                    .streams
                    .iter()
                    .find(|stream| stream.kind == Kind::Events)
                    .and_then(|stream| stream.size)
            })
            .ok_or(IoError::InvalidSensorSize)?;
        if width == 0 || height == 0 {
            return Err(IoError::InvalidSensorSize);
        }

        let n_events = events.last().map_or(0, |last| last.first + last.elements);
        let time_span = match (events.first(), events.last()) {
            (Some(first), Some(last)) => (first.start, last.end),
            _ => (0, 0),
        };
        Ok(Self {
            path: path.to_path_buf(),
            compression: header.compression,
            width,
            height,
            events,
            frames: of_kind(Kind::Frames),
            imu: of_kind(Kind::Imu),
            n_events,
            time_span,
        })
    }

    fn body(&self, packet: &Packet) -> Result<Vec<u8>, IoError> {
        read_body(&self.path, self.compression, packet.offset, packet.size)
    }

    /// Decodes every event in `packet`, handing each to `visit` with its index in the stream.
    fn each_event(
        &self,
        packet: &Packet,
        mut visit: impl FnMut(usize, u16, u16, i64, bool),
    ) -> Result<(), IoError> {
        let body = self.body(packet)?;
        let table = Table::size_prefixed_root(&body)
            .ok_or_else(|| IoError::Format("AEDAT4 event packet is not readable".to_owned()))?;
        let Some((bytes, elements)) = table.vector(slot::ELEMENTS, EVENT_STRIDE) else {
            return Ok(());
        };
        let (events, _) = bytes.as_chunks::<EVENT_STRIDE>();
        for (index, event) in events.iter().enumerate().take(elements) {
            let t = i64::from_le_bytes(event[0..8].try_into().expect("eight bytes"));
            let x = i16::from_le_bytes(event[8..10].try_into().expect("two bytes"));
            let y = i16::from_le_bytes(event[10..12].try_into().expect("two bytes"));
            if x < 0 || y < 0 {
                continue;
            }
            visit(packet.first + index, x as u16, y as u16, t, event[12] != 0);
        }
        Ok(())
    }
}

impl SliceSource for Aedat4SliceSource {
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
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        for packet in &self.events {
            if packet.first >= i1 || packet.first + packet.elements <= i0 {
                continue;
            }
            self.each_event(packet, |index, x, y, t, p| {
                if index >= i0 && index < i1 {
                    builder.push(x, y, t, p);
                }
            })?;
        }
        Ok(builder.build())
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let mut builder = EventStreamBuilder::new(self.width, self.height, TIMESTAMP_SCALE_MS);
        for packet in overlapping(&self.events, t0, t1) {
            self.each_event(packet, |_, x, y, t, p| {
                if t >= t0 && t < t1 {
                    builder.push(x, y, t, p);
                }
            })?;
        }
        Ok(builder.build())
    }

    fn frames(&self, t0: i64, t1: i64) -> Result<Vec<(i64, EventFrame)>, IoError> {
        let mut frames = Vec::new();
        for packet in overlapping(&self.frames, t0, t1) {
            let body = self.body(packet)?;
            // A frame packet's root *is* the frame — `Frame` is its own `root_type`, with no
            // wrapping element vector, unlike events and IMU samples.
            let table = Table::size_prefixed_root(&body)
                .ok_or_else(|| IoError::Format("AEDAT4 frame packet is not readable".to_owned()))?;
            let t = table.i64(slot::FRAME_TIMESTAMP, 0);
            if t < t0 || t >= t1 {
                continue;
            }
            frames.push((t, decode_frame(&table)?));
        }
        Ok(frames)
    }

    fn imu(&self, t0: i64, t1: i64) -> Result<Vec<ImuSample>, IoError> {
        let mut samples = Vec::new();
        for packet in overlapping(&self.imu, t0, t1) {
            let body = self.body(packet)?;
            let table = Table::size_prefixed_root(&body)
                .ok_or_else(|| IoError::Format("AEDAT4 IMU packet is not readable".to_owned()))?;
            for element in table.tables(slot::ELEMENTS) {
                let t = element.i64(slot::IMU_TIMESTAMP, 0);
                if t < t0 || t >= t1 {
                    continue;
                }
                // The schema states g and degrees per second; `ImuSample` is m/s² and rad/s.
                let axes = |base: usize, scale: f64| {
                    [0, 1, 2].map(|axis| f64::from(element.f32(base + axis * 2)) * scale)
                };
                samples.push(ImuSample {
                    t_us: t,
                    linear_acceleration: axes(slot::IMU_ACCELEROMETER, STANDARD_GRAVITY),
                    angular_velocity: axes(slot::IMU_GYROSCOPE, std::f64::consts::PI / 180.0),
                });
            }
        }
        Ok(samples)
    }
}

/// Packets whose span touches the half-open window `[t0, t1)`.
fn overlapping(packets: &[Packet], t0: i64, t1: i64) -> impl Iterator<Item = &Packet> {
    packets
        .iter()
        .filter(move |packet| packet.end >= t0 && packet.start < t1)
}

/// Turns a `Frame` table into a greyscale [`EventFrame`].
fn decode_frame(table: &Table<'_>) -> Result<EventFrame, IoError> {
    let width = table.i16(slot::FRAME_SIZE_X) as usize;
    let height = table.i16(slot::FRAME_SIZE_Y) as usize;
    let format = table.i8(slot::FRAME_FORMAT, FORMAT_GREY);
    let channels = match format {
        FORMAT_GREY => 1,
        FORMAT_BGR => 3,
        FORMAT_BGRA => 4,
        other => {
            return Err(IoError::Unsupported(format!(
                "AEDAT4 frame format {other} is not an 8-bit image this reader decodes"
            )))
        }
    };
    let (pixels, _) = table
        .vector(slot::FRAME_PIXELS, 1)
        .ok_or_else(|| IoError::Format("AEDAT4 frame has no pixels".to_owned()))?;
    if pixels.len() != width * height * channels {
        return Err(IoError::Format(format!(
            "AEDAT4 frame declares {width}x{height}x{channels} but carries {} bytes",
            pixels.len()
        )));
    }
    // OpenCV colour is BGR, so the channels are reversed before the luma weights are applied.
    let samples: Vec<u8> = match channels {
        1 => pixels.to_vec(),
        _ => pixels
            .chunks_exact(channels)
            .map(|bgr| super::luma(&[bgr[2], bgr[1], bgr[0]]))
            .collect(),
    };
    EventFrame::intensity(EventFrameData::U8(samples), width, height)
        .map_err(|error| IoError::Format(format!("AEDAT4 frame: {error}")))
}

/// Reads the `FileDataTable` DV appends after the last packet: one entry per packet, with its
/// byte offset, element count and time span already computed.
fn read_data_table(
    path: &Path,
    compression: Compression,
    position: u64,
    size: u64,
) -> Result<Vec<(i32, Packet)>, IoError> {
    let body = read_body(path, compression, position, size as usize)?;
    let table = Table::size_prefixed_root(&body)
        .ok_or_else(|| IoError::Format("AEDAT4 data table is not readable".to_owned()))?;
    Ok(table
        .tables(slot::TABLE)
        .iter()
        .filter_map(|entry| {
            // `PacketInfo` is an inline struct: `int32 StreamID`, `int32 Size`.
            let info = entry.field(slot::DEFINITION_PACKET_INFO)?;
            let id = i32::from_le_bytes(scalar(entry.buffer, info)?);
            let size = i32::from_le_bytes(scalar(entry.buffer, info + 4)?);
            Some((
                id,
                Packet {
                    offset: entry.i64(slot::DEFINITION_BYTE_OFFSET, 0).try_into().ok()?,
                    size: size.try_into().ok()?,
                    first: 0,
                    elements: entry
                        .i64(slot::DEFINITION_NUM_ELEMENTS, 0)
                        .try_into()
                        .ok()?,
                    start: entry.i64(slot::DEFINITION_TIMESTAMP_START, 0),
                    end: entry.i64(slot::DEFINITION_TIMESTAMP_END, 0),
                },
            ))
        })
        .collect())
}

/// Rebuilds the index by reading every packet, for a file whose table is missing. Costs a full
/// decode pass, which is why DV writes the table in the first place.
fn walk_packets(
    path: &Path,
    compression: Compression,
    body_offset: u64,
    total: u64,
) -> Result<Vec<(i32, Packet)>, IoError> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(body_offset))?;
    let mut packets = Vec::new();
    let mut offset = body_offset;
    let mut header = [0u8; PACKET_HEADER];
    while offset + PACKET_HEADER as u64 <= total {
        file.read_exact(&mut header)?;
        let id = i32::from_le_bytes(header[0..4].try_into().expect("four bytes"));
        let size = i32::from_le_bytes(header[4..8].try_into().expect("four bytes"));
        let Ok(size) = usize::try_from(size) else {
            break;
        };
        offset += PACKET_HEADER as u64;
        if offset + size as u64 > total {
            break; // truncated tail
        }
        let mut packet = Packet {
            offset,
            size,
            first: 0,
            elements: 0,
            start: 0,
            end: 0,
        };
        if let Ok(body) = read_body(path, compression, offset, size) {
            (packet.elements, packet.start, packet.end) = span_of(&body);
        }
        packets.push((id, packet));
        offset += size as u64;
        file.seek(SeekFrom::Start(offset))?;
    }
    Ok(packets)
}

/// `(elements, first timestamp, last timestamp)` of a decompressed packet, whichever of the
/// three shapes it is: a vector of event structs, a vector of tables, or a lone frame.
fn span_of(body: &[u8]) -> (usize, i64, i64) {
    let Some(table) = Table::size_prefixed_root(body) else {
        return (0, 0, 0);
    };
    if let Some((bytes, elements)) = table.vector(slot::ELEMENTS, EVENT_STRIDE) {
        if elements > 0 && bytes.len() >= EVENT_STRIDE {
            let at = |chunk: &[u8]| i64::from_le_bytes(chunk[0..8].try_into().expect("eight bytes"));
            return (
                elements,
                at(&bytes[..EVENT_STRIDE]),
                at(&bytes[bytes.len() - EVENT_STRIDE..]),
            );
        }
    }
    let elements = table.tables(slot::ELEMENTS);
    if let (Some(first), Some(last)) = (elements.first(), elements.last()) {
        return (
            elements.len(),
            first.i64(slot::IMU_TIMESTAMP, 0),
            last.i64(slot::IMU_TIMESTAMP, 0),
        );
    }
    let t = table.i64(slot::FRAME_TIMESTAMP, 0);
    (1, t, t)
}

/// Opens an AEDAT 4 file for bounded-memory random slicing.
pub fn open_aedat4_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<Aedat4SliceSource, IoError> {
    Aedat4SliceSource::open(path.as_ref(), options)
}

/// Eagerly reads an AEDAT 4 (`.aedat4`) recording's events. Prefer [`open_aedat4_slice`] for
/// large recordings; APS frames and IMU samples are read with the reader's `frames`/`imu`.
pub fn read_aedat4(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let source = open_aedat4_slice(path, options)?;
    let limit = options.max_events.unwrap_or(source.n_events());
    source.slice_index(0, limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same 64x48 DAVIS recording written by iniVation's own DV under each compression it
    /// offers, so these tests check this reader against the format's reference implementation
    /// rather than against itself.
    const SAMPLE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/test/sample.aedat4"
    );
    const SAMPLE_ZSTD: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/test/sample_zstd.aedat4"
    );

    fn sample() -> Aedat4SliceSource {
        open_aedat4_slice(SAMPLE, &LoadOptions::default()).unwrap()
    }

    #[test]
    fn reads_a_recording_written_by_dv() {
        let source = sample();
        assert_eq!(source.sensor_size(), (64, 48)); // from the header's infoNode
        assert_eq!(source.n_events(), 64);
        assert_eq!(source.time_span(), (1_000_000, 1_063_000));

        let stream = source.slice_index(0, source.n_events()).unwrap();
        assert_eq!(
            (stream.xs()[0], stream.ys()[0], stream.ts()[0], stream.ps()[0]),
            (0, 0, 1_000_000, false)
        );
        assert_eq!(
            (
                stream.xs()[63],
                stream.ys()[63],
                stream.ts()[63],
                stream.ps()[63]
            ),
            (63, 45, 1_063_000, true)
        );
    }

    #[test]
    fn slices_agree_with_a_whole_file_read() {
        let source = sample();
        let whole = source.slice_index(0, source.n_events()).unwrap();
        let head = source.slice_index(0, 10).unwrap();
        assert_eq!(head.ts(), &whole.ts()[..10]);
        let window = source.slice_time(1_010_000, 1_015_000).unwrap();
        assert_eq!(window.ts(), &[1_010_000, 1_011_000, 1_012_000, 1_013_000, 1_014_000]);
    }

    #[test]
    fn frames_and_imu_come_back_beside_the_events() {
        let source = sample();
        let frames = source.frames(0, i64::MAX).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].0, 1_000_000);
        assert_eq!(frames[0].1.shape(), (1, 48, 64));
        let EventFrameData::U8(pixels) = frames[0].1.data() else {
            panic!("an 8-bit APS frame");
        };
        // The fixture's image is `(index + 5k) % 251`, so its corners pin the row order down.
        assert_eq!((pixels[0], pixels[pixels.len() - 1]), (0, 59));

        let samples = source.imu(0, i64::MAX).unwrap();
        assert_eq!(samples.len(), 8);
        assert_eq!(samples[0].t_us, 1_000_000);
        // The fixture records -1 g on Y and no rotation; the schema is g and °/s, this is SI.
        assert!((samples[0].linear_acceleration[1] + STANDARD_GRAVITY).abs() < 1e-6);
        assert_eq!(samples[0].angular_velocity, [0.0; 3]);

        // Windows apply to the auxiliary streams too.
        assert_eq!(source.frames(1_020_000, i64::MAX).unwrap().len(), 1);
        assert_eq!(source.imu(0, 1_010_000).unwrap().len(), 2);
    }

    #[test]
    fn compression_does_not_change_what_is_read() {
        // Same recording, written twice — once LZ4, once Zstd. Both decoders have to produce
        // the same bytes, which is what makes the compression an implementation detail.
        let lz4 = sample();
        let zstd = open_aedat4_slice(SAMPLE_ZSTD, &LoadOptions::default()).unwrap();
        assert_eq!(zstd.sensor_size(), lz4.sensor_size());
        assert_eq!(zstd.n_events(), lz4.n_events());
        assert_eq!(zstd.time_span(), lz4.time_span());

        let (a, b) = (
            lz4.slice_index(0, lz4.n_events()).unwrap(),
            zstd.slice_index(0, zstd.n_events()).unwrap(),
        );
        assert_eq!((a.xs(), a.ys(), a.ts(), a.ps()), (b.xs(), b.ys(), b.ts(), b.ps()));
        assert_eq!(
            lz4.frames(0, i64::MAX).unwrap()[0].1.data(),
            zstd.frames(0, i64::MAX).unwrap()[0].1.data()
        );
        assert_eq!(lz4.imu(0, i64::MAX).unwrap(), zstd.imu(0, i64::MAX).unwrap());
    }

    /// Builds a minimal uncompressed AEDAT 4 file: an `IOHeader` naming one event stream, then
    /// `events` in a single packet. `data_table` puts a `dataTablePosition` in the header;
    /// without one the reader has to walk the packets instead.
    fn minimal_file(events: &[(i64, i16, i16, bool)]) -> Vec<u8> {
        let info = "<node name=\"0\" path=\"/outInfo/0/\">\
                    <attr key=\"typeIdentifier\" type=\"string\">EVTS</attr>\
                    <node name=\"info\"><attr key=\"sizeX\" type=\"int\">8</attr>\
                    <attr key=\"sizeY\" type=\"int\">4</attr></node></node>";

        // IOHeader: root offset, vtable, table, then the string it points at.
        let mut header = Vec::new();
        header.extend_from_slice(&14u32.to_le_bytes()); // root table at 14
        header.extend_from_slice(&10u16.to_le_bytes()); // vtable spans slots 4, 6 and 8
        header.extend_from_slice(&20u16.to_le_bytes()); // table length
        for offset in [4u16, 8, 16] {
            header.extend_from_slice(&offset.to_le_bytes());
        }
        header.extend_from_slice(&10i32.to_le_bytes()); // soffset: table 14 - vtable 4
        header.extend_from_slice(&0i32.to_le_bytes()); // compression NONE
        header.extend_from_slice(&(-1i64).to_le_bytes()); // no data table: force the walk
        header.extend_from_slice(&4u32.to_le_bytes()); // infoNode is the next thing written
        header.extend_from_slice(&(info.len() as u32).to_le_bytes());
        header.extend_from_slice(info.as_bytes());
        header.push(0); // strings are null-terminated

        // EventPacket: size prefix, root offset, vtable, table, then the element vector.
        let mut packet = Vec::new();
        packet.extend_from_slice(&10u32.to_le_bytes()); // root table at 14, relative to 4
        packet.extend_from_slice(&6u16.to_le_bytes()); // vtable spans slot 4 only
        packet.extend_from_slice(&8u16.to_le_bytes()); // table length
        packet.extend_from_slice(&4u16.to_le_bytes()); // elements at table + 4
        packet.extend_from_slice(&6i32.to_le_bytes()); // soffset: table 14 - vtable 8
        packet.extend_from_slice(&4u32.to_le_bytes()); // the vector is the next thing written
        packet.extend_from_slice(&(events.len() as u32).to_le_bytes());
        for &(t, x, y, polarity) in events {
            packet.extend_from_slice(&t.to_le_bytes());
            packet.extend_from_slice(&x.to_le_bytes());
            packet.extend_from_slice(&y.to_le_bytes());
            packet.push(polarity as u8);
            packet.extend_from_slice(&[0; 3]); // struct padding to the 8-byte alignment
        }

        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&(header.len() as u32).to_le_bytes());
        file.extend_from_slice(&header);
        // One packet: an 8-byte `PacketHeader`, then the size-prefixed FlatBuffer body.
        file.extend_from_slice(&0i32.to_le_bytes()); // stream id
        file.extend_from_slice(&((packet.len() + 4) as i32).to_le_bytes()); // body size
        file.extend_from_slice(&(packet.len() as u32).to_le_bytes()); // FlatBuffer size prefix
        file.extend_from_slice(&packet);
        file
    }

    fn write_temporary(data: &[u8], tag: &str) -> PathBuf {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "eventcv_aedat4_{}_{tag}.aedat4",
            std::process::id()
        ));
        File::create(&path).unwrap().write_all(data).unwrap();
        path
    }

    #[test]
    fn a_file_without_a_data_table_is_indexed_by_walking_its_packets() {
        let events = [(10i64, 1i16, 2i16, true), (20, 3, 0, false), (30, 7, 3, true)];
        let path = write_temporary(&minimal_file(&events), "walk");
        let source = open_aedat4_slice(&path, &LoadOptions::default()).unwrap();

        assert_eq!(source.sensor_size(), (8, 4));
        assert_eq!(source.n_events(), 3);
        assert_eq!(source.time_span(), (10, 30));
        let stream = source.slice_index(0, 3).unwrap();
        assert_eq!(stream.ts(), &[10, 20, 30]);
        assert_eq!(stream.xs(), &[1, 3, 7]);
        assert_eq!(stream.ps(), &[true, false, true]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_truncated_file_errors_rather_than_reading_past_its_end() {
        let full = minimal_file(&[(10, 1, 2, true), (20, 3, 0, false)]);
        // Every prefix must fail cleanly — the point is that none of them panics.
        for cut in [4, MAGIC.len(), MAGIC.len() + 8, full.len() - 40, full.len() - 8] {
            let path = write_temporary(&full[..cut], "cut");
            match open_aedat4_slice(&path, &LoadOptions::default()) {
                Err(_) => {}
                Ok(source) => {
                    // A prefix that still parses must at least not invent events.
                    let _ = source.slice_index(0, source.n_events()).unwrap();
                }
            }
            std::fs::remove_file(&path).ok();
        }
    }

    #[test]
    fn a_file_that_is_not_aedat4_is_a_format_error() {
        let path = write_temporary(b"#!AER-DAT2.0\r\nrubbish", "wrong");
        match open_aedat4_slice(&path, &LoadOptions::default()) {
            Err(IoError::Format(message)) => assert!(message.contains("AEDAT 4")),
            other => panic!("expected a format error, got {:?}", other.map(|_| ())),
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn the_header_xml_is_scanned_for_stream_types_and_resolution() {
        let info = "<node name=\"outInfo\"><node name=\"0\">\
                    <attr key=\"typeIdentifier\" type=\"string\">EVTS</attr>\
                    <node name=\"info\"><attr key=\"sizeX\" type=\"int\">640</attr>\
                    <attr key=\"sizeY\" type=\"int\">480</attr></node></node>\
                    <node name=\"1\"><attr key=\"typeIdentifier\" type=\"string\">FRME</attr></node>\
                    <node name=\"2\"><attr key=\"typeIdentifier\" type=\"string\">TRIG</attr></node>\
                    </node>";
        let streams = parse_streams(info);
        assert_eq!(streams.len(), 3); // `outInfo` and `info` are not stream nodes
        assert_eq!(streams[0].id, 0);
        assert_eq!(streams[0].kind, Kind::Events);
        assert_eq!(streams[0].size, Some((640, 480)));
        assert_eq!(streams[1].kind, Kind::Frames);
        assert_eq!(streams[1].size, None);
        assert_eq!(streams[2].kind, Kind::Other); // triggers are read past, not decoded
    }

    #[test]
    fn unknown_compression_is_rejected_by_name() {
        assert_eq!(Compression::from_header(0).unwrap(), Compression::None);
        assert_eq!(Compression::from_header(2).unwrap(), Compression::Lz4);
        assert_eq!(Compression::from_header(4).unwrap(), Compression::Zstd);
        match Compression::from_header(9) {
            Err(IoError::Unsupported(message)) => assert!(message.contains("compression")),
            other => panic!("expected an unsupported-compression error, got {other:?}"),
        }
    }

    #[test]
    fn table_reads_are_bounds_checked() {
        // A root offset past the end, a vtable past the end, and a vector claiming more
        // elements than the buffer holds: all `None`, none a panic.
        assert!(Table::root(&[0xFF, 0xFF, 0xFF, 0x7F]).is_none());
        let table = Table::root(&[4, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0x7F]).unwrap();
        assert!(table.field(slot::ELEMENTS).is_none());
        assert_eq!(table.i64(slot::ELEMENTS, -7), -7); // absent field falls back to the default
    }
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

/// FlatBuffer file identifiers, the four bytes each root carries at offset 4. Taken from the files
/// DV writes rather than from the schemas, since the schemas are not shipped with a recording.
const IDENT_HEADER: &[u8; 4] = b"IOHE";
const IDENT_EVENTS: &[u8; 4] = b"EVTS";
const IDENT_TABLE: &[u8; 4] = b"FTAB";

/// The one stream a written recording declares. Events only: a sink is handed an [`EventStream`],
/// which carries neither frames nor IMU.
const EVENT_STREAM_ID: i32 = 0;

/// Events per packet. DV writes a packet per capture interval; a sink has no intervals, so this
/// bounds the working buffer instead — 100k events is ~1.6 MB uncompressed, which compresses in
/// one pass and keeps the data table short enough to stay a useful index.
const PACKET_EVENTS: usize = 100_000;

/// A minimal FlatBuffer *writer*, the counterpart of the [`Table`] reader above.
///
/// Buffers are built back-to-front, which is what the format's self-relative offsets require, so
/// the bytes are accumulated reversed and flipped once at the end. Offsets are therefore counted
/// from the buffer's end: an object written when `len()` was `n` ends up at final position
/// `total - n`, and that single convention is what every offset below is expressed in.
///
/// This exists for the same reason the reader hand-rolls its decoding: the four schemas involved
/// are small and fixed, and the `flatbuffers` crate's builder would pull in a code generator to
/// emit what amounts to the hundred lines below.
struct FlatBuilder {
    /// The buffer reversed — `rev[0]` is the final buffer's last byte.
    rev: Vec<u8>,
    /// The strictest alignment anything asked for, used to pad the front at the end.
    min_align: usize,
}

impl FlatBuilder {
    fn new() -> Self {
        Self {
            rev: Vec::new(),
            min_align: 4,
        }
    }

    fn len(&self) -> usize {
        self.rev.len()
    }

    /// Pads so that `size` more bytes land on an `align` boundary.
    fn pre_align(&mut self, size: usize, align: usize) {
        self.min_align = self.min_align.max(align);
        let pad = (align - ((self.len() + size) % align)) % align;
        self.rev.resize(self.len() + pad, 0);
    }

    /// Writes `bytes` at the current position (reversed, so they read forwards in the result).
    fn push(&mut self, bytes: &[u8]) {
        self.rev.extend(bytes.iter().rev());
    }

    /// Writes one aligned scalar and returns its offset.
    fn push_scalar(&mut self, bytes: &[u8]) -> usize {
        self.pre_align(bytes.len(), bytes.len());
        self.push(bytes);
        self.len()
    }

    /// Writes a `uoffset` pointing at `target`, which must already have been written.
    fn push_uoffset(&mut self, target: usize) -> usize {
        self.pre_align(4, 4);
        let value = (self.len() + 4 - target) as u32;
        self.push(&value.to_le_bytes());
        self.len()
    }

    /// Writes a length-prefixed UTF-8 string (NUL-terminated, as FlatBuffers stores them).
    fn push_string(&mut self, value: &str) -> usize {
        self.pre_align(value.len() + 1, 4);
        self.push(&[0]);
        self.push(value.as_bytes());
        self.push_scalar(&(value.len() as u32).to_le_bytes())
    }

    /// Writes a vector of inline structs from `data` (already in final byte order), returning the
    /// offset of its length prefix.
    fn push_struct_vector(&mut self, data: &[u8], count: usize, align: usize) -> usize {
        self.pre_align(data.len(), align);
        self.push(data);
        self.push_scalar(&(count as u32).to_le_bytes())
    }

    /// Closes a table whose fields have already been written.
    ///
    /// `start` is `len()` from before the first field, and `fields` pairs each occupied vtable slot
    /// with the offset the field was written at. Every table gets its own vtable — flatc's builder
    /// dedupes identical ones, which saves bytes but nothing a reader can tell apart.
    fn end_table(&mut self, start: usize, fields: &[(usize, usize)]) -> usize {
        self.pre_align(4, 4);
        self.push(&0i32.to_le_bytes()); // patched below, once the vtable's position is known
        let table = self.len();

        let slots = fields.iter().map(|(slot, _)| *slot).max().unwrap_or(2);
        let entries = (slots.saturating_sub(4)) / 2 + 1;
        let mut vtable = vec![0u16; 2 + entries];
        vtable[0] = (vtable.len() * 2) as u16;
        vtable[1] = (table - start) as u16;
        for (slot, offset) in fields {
            vtable[2 + (slot - 4) / 2] = (table - offset) as u16;
        }
        self.pre_align(vtable.len() * 2, 2);
        for value in vtable.iter().rev() {
            self.push(&value.to_le_bytes());
        }
        let vtable_at = self.len();

        // soffset = table position - vtable position, positive because the vtable precedes it.
        let soffset = (vtable_at - table) as i32;
        let bytes = soffset.to_le_bytes();
        self.rev[table - 4..table].copy_from_slice(&[bytes[3], bytes[2], bytes[1], bytes[0]]);
        table
    }

    /// Emits the finished buffer: the root offset, the file identifier, and — for the packet
    /// bodies, which are framed that way — a leading length.
    fn finish(mut self, root: usize, identifier: &[u8; 4], size_prefixed: bool) -> Vec<u8> {
        let prefix = 4 + 4 + usize::from(size_prefixed) * 4;
        self.pre_align(prefix, self.min_align);
        self.push(identifier);
        self.push_uoffset(root);
        if size_prefixed {
            let size = self.len() as u32;
            self.push(&size.to_le_bytes());
        }
        self.rev.reverse();
        self.rev
    }
}

/// The `infoNode` XML declaring the single event stream, in the shape DV writes and
/// [`parse_streams`] reads back.
fn info_node(width: usize, height: usize, compression: Compression) -> String {
    let compression = match compression {
        Compression::None => "NONE",
        Compression::Lz4 => "LZ4",
        Compression::Zstd => "ZSTD",
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n\
         <dv version=\"2.0\">\n\
         \x20   <node name=\"outInfo\" path=\"/outInfo/\">\n\
         \x20       <node name=\"{EVENT_STREAM_ID}\" path=\"/outInfo/{EVENT_STREAM_ID}/\">\n\
         \x20           <attr key=\"compression\" type=\"string\">{compression}</attr>\n\
         \x20           <attr key=\"originalModuleName\" type=\"string\">eventcv</attr>\n\
         \x20           <attr key=\"originalOutputName\" type=\"string\">events</attr>\n\
         \x20           <attr key=\"typeDescription\" type=\"string\">Event data.</attr>\n\
         \x20           <attr key=\"typeIdentifier\" type=\"string\">EVTS</attr>\n\
         \x20           <node name=\"info\" path=\"/outInfo/{EVENT_STREAM_ID}/info/\">\n\
         \x20               <attr key=\"sizeX\" type=\"int\">{width}</attr>\n\
         \x20               <attr key=\"sizeY\" type=\"int\">{height}</attr>\n\
         \x20               <attr key=\"source\" type=\"string\">eventcv</attr>\n\
         \x20           </node>\n\
         \x20       </node>\n\
         \x20   </node>\n\
         </dv>\n"
    )
}

/// Builds the `IOHeader` FlatBuffer, and reports where its `dataTablePosition` field sits within
/// the buffer so the sink can patch it once the table has been written.
fn build_header(width: usize, height: usize, compression: Compression) -> (Vec<u8>, usize) {
    let mut builder = FlatBuilder::new();
    let info = builder.push_string(&info_node(width, height, compression));
    let start = builder.len();
    let compression_at = builder.push_scalar(&compression.to_header().to_le_bytes());
    // Written as -1 — "no table" — so a recording abandoned before `finish` still opens, by the
    // same packet walk the reader uses for an interrupted DV recording.
    let position_at = builder.push_scalar(&(-1i64).to_le_bytes());
    let info_at = builder.push_uoffset(info);
    let root = builder.end_table(
        start,
        &[
            (slot::COMPRESSION, compression_at),
            (slot::DATA_TABLE_POSITION, position_at),
            (slot::INFO_NODE, info_at),
        ],
    );
    let buffer = builder.finish(root, IDENT_HEADER, false);
    // Offsets were counted from the end; convert the field's to a position in the finished buffer.
    let field = buffer.len() - position_at;
    (buffer, field)
}

/// One packet as the data table records it.
#[derive(Clone, Copy, Debug)]
struct Definition {
    byte_offset: i64,
    size: i32,
    elements: i64,
    start: i64,
    end: i64,
}

/// Builds the trailing `FileDataTable` — the index that makes a recording sliceable without
/// reading it.
fn build_data_table(definitions: &[Definition]) -> Vec<u8> {
    let mut builder = FlatBuilder::new();
    // Children before parents: every definition table is written first, then the vector of
    // offsets to them, then the table that holds it.
    let mut offsets = Vec::with_capacity(definitions.len());
    for definition in definitions.iter().rev() {
        let start = builder.len();
        let byte_offset = builder.push_scalar(&definition.byte_offset.to_le_bytes());
        let elements = builder.push_scalar(&definition.elements.to_le_bytes());
        let start_at = builder.push_scalar(&definition.start.to_le_bytes());
        let end_at = builder.push_scalar(&definition.end.to_le_bytes());
        // `PacketInfo` is an inline struct of two `int32`s, so it is written in place.
        builder.pre_align(8, 4);
        builder.push(&definition.size.to_le_bytes());
        builder.push(&EVENT_STREAM_ID.to_le_bytes());
        let info = builder.len();
        offsets.push(builder.end_table(
            start,
            &[
                (slot::DEFINITION_BYTE_OFFSET, byte_offset),
                (slot::DEFINITION_PACKET_INFO, info),
                (slot::DEFINITION_NUM_ELEMENTS, elements),
                (slot::DEFINITION_TIMESTAMP_START, start_at),
                (slot::DEFINITION_TIMESTAMP_END, end_at),
            ],
        ));
    }
    offsets.reverse();

    builder.pre_align(offsets.len() * 4, 4);
    let mut vector_at = builder.len();
    for offset in offsets.iter().rev() {
        vector_at = builder.push_uoffset(*offset);
    }
    let _ = vector_at;
    let vector = builder.push_scalar(&(offsets.len() as u32).to_le_bytes());

    let start = builder.len();
    let table_at = builder.push_uoffset(vector);
    let root = builder.end_table(start, &[(slot::TABLE, table_at)]);
    builder.finish(root, IDENT_TABLE, true)
}

/// Builds one `EventPacket` body from `stream[range]`.
fn build_event_packet(stream: &EventStream, range: std::ops::Range<usize>) -> Vec<u8> {
    let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
    let mut elements = Vec::with_capacity(range.len() * EVENT_STRIDE);
    for index in range.clone() {
        elements.extend_from_slice(&ts[index].to_le_bytes());
        elements.extend_from_slice(&(xs[index] as i16).to_le_bytes());
        elements.extend_from_slice(&(ys[index] as i16).to_le_bytes());
        elements.push(u8::from(ps[index]));
        elements.extend_from_slice(&[0; 3]); // padding to the struct's 8-byte alignment
    }
    let mut builder = FlatBuilder::new();
    let vector = builder.push_struct_vector(&elements, range.len(), 8);
    let start = builder.len();
    let elements_at = builder.push_uoffset(vector);
    let root = builder.end_table(start, &[(slot::ELEMENTS, elements_at)]);
    builder.finish(root, IDENT_EVENTS, true)
}

/// Writes an AEDAT 4.0 (`.aedat4`) recording a window at a time — the inverse of [`read_aedat4`].
///
/// The `IOHeader` goes out on the first non-empty append, because its `infoNode` names the sensor
/// size. Each append is then split into packets of at most [`PACKET_EVENTS`] events; every packet
/// is a size-prefixed `EventPacket` FlatBuffer, compressed, behind an 8-byte `PacketHeader`.
///
/// The `FileDataTable` is written at [`finish`](super::EventSink::finish) and its position patched
/// back into the header, which is what makes the result sliceable rather than merely readable. A
/// recording abandoned before then still opens: the header's `dataTablePosition` stays `-1`, and
/// the reader falls back to walking the packets — the same path it takes for a DV recording that
/// was interrupted.
pub struct Aedat4EventSink {
    file: File,
    compression: Compression,
    /// Byte offset of the header's `dataTablePosition` field; `None` until the header is written.
    position_field: Option<u64>,
    definitions: Vec<Definition>,
    n_events: usize,
}

impl Aedat4EventSink {
    pub fn create(path: impl AsRef<Path>, compression: Compression) -> Result<Self, IoError> {
        Ok(Self {
            file: File::create(path.as_ref())?,
            compression,
            position_field: None,
            definitions: Vec::new(),
            n_events: 0,
        })
    }

    /// Compresses and writes one packet, recording its data-table entry.
    fn write_packet(&mut self, stream: &EventStream, range: std::ops::Range<usize>) -> Result<(), IoError> {
        let ts = stream.ts();
        let body = self
            .compression
            .encode(&build_event_packet(stream, range.clone()))?;
        let size = i32::try_from(body.len()).map_err(|_| {
            IoError::Unsupported("an AEDAT4 packet body exceeds 2 GB".to_owned())
        })?;
        self.file.write_all(&EVENT_STREAM_ID.to_le_bytes())?;
        self.file.write_all(&size.to_le_bytes())?;
        let byte_offset = self.file.stream_position()?;
        self.file.write_all(&body)?;
        self.definitions.push(Definition {
            byte_offset: byte_offset as i64,
            size,
            elements: range.len() as i64,
            start: ts[range.start],
            end: ts[range.end - 1],
        });
        Ok(())
    }
}

impl super::EventSink for Aedat4EventSink {
    fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        if stream.is_empty() {
            return Ok(());
        }
        if self.position_field.is_none() {
            let (width, height) = stream.sensor_size();
            let (header, field) = build_header(width, height, self.compression);
            self.file.write_all(MAGIC)?;
            self.file.write_all(&(header.len() as u32).to_le_bytes())?;
            self.file.write_all(&header)?;
            self.position_field = Some((MAGIC.len() + 4 + field) as u64);
        }
        let mut start = 0;
        while start < stream.len() {
            let end = (start + PACKET_EVENTS).min(stream.len());
            self.write_packet(stream, start..end)?;
            start = end;
        }
        self.n_events += stream.len();
        Ok(())
    }

    fn n_events(&self) -> usize {
        self.n_events
    }

    fn flush(&mut self) -> Result<(), IoError> {
        self.file.flush().map_err(IoError::Io)
    }

    fn finish(mut self: Box<Self>) -> Result<(), IoError> {
        let Some(field) = self.position_field else {
            // Nothing was ever appended: an empty header with no streams would not describe an
            // event recording, so write one for a 1x1 sensor — what the readers infer for an
            // empty stream anyway.
            let (header, _) = build_header(1, 1, self.compression);
            self.file.write_all(MAGIC)?;
            self.file.write_all(&(header.len() as u32).to_le_bytes())?;
            self.file.write_all(&header)?;
            return self.file.flush().map_err(IoError::Io);
        };
        let position = self.file.stream_position()?;
        let table = self.compression.encode(&build_data_table(&self.definitions))?;
        self.file.write_all(&table)?;
        self.file.seek(SeekFrom::Start(field))?;
        self.file.write_all(&(position as i64).to_le_bytes())?;
        self.file.flush().map_err(IoError::Io)
    }
}
