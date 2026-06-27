use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

use rosbag::{ChunkRecord, IndexRecord, MessageRecord, RosBag};

use super::{IoError, LoadOptions, SliceSource};
use crate::{EventStream, EventStreamBuilder};

const DEFAULT_TOPIC: &str = "/davis/left/events";
const EVENT_ARRAY_TYPE: &str = "dvs_msgs/EventArray";
/// `dvs_msgs/EventArray` md5sum (32 lowercase hex). The reader matches on topic + type,
/// not this value, but the rosbag parser requires a well-formed md5sum on every connection.
const EVENT_ARRAY_MD5: &str = "5e8beee5759d85e9f5a8c1f5ad1cd00b";
/// Message definition stored on the connection (informational; any non-empty text parses).
const EVENT_ARRAY_DEF: &str =
    "std_msgs/Header header\nuint32 height\nuint32 width\ndvs_msgs/Event[] events\n";
/// Events per `dvs_msgs/EventArray` message — batches keep individual message buffers bounded.
const MESSAGE_EVENTS: usize = 1_000_000;
/// ROS bag 2.0 magic line.
const BAG_MAGIC: &[u8] = b"#ROSBAG V2.0\n";

/// Reads a ROS1 bag, decoding `dvs_msgs/EventArray` messages on a single topic
/// (`options.topic`, default `/davis/left/events`). Sensor size comes from the
/// messages unless `options.sensor_size` overrides it; timestamps become microseconds.
pub fn read_bag(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let topic = options.topic.as_deref().unwrap_or(DEFAULT_TOPIC);
    let bag = RosBag::new(path).map_err(IoError::Io)?;

    let mut wanted: HashMap<u32, bool> = HashMap::new();
    let mut builder: Option<EventStreamBuilder> = None;

    'outer: for record in bag.chunk_records() {
        let ChunkRecord::Chunk(chunk) = record.map_err(map_bag_error)? else {
            continue;
        };
        for message in chunk.messages() {
            match message.map_err(map_bag_error)? {
                MessageRecord::Connection(connection) => {
                    let matches = connection.topic == topic && connection.tp == EVENT_ARRAY_TYPE;
                    wanted.insert(connection.id, matches);
                }
                MessageRecord::MessageData(data) => {
                    if !wanted.get(&data.conn_id).copied().unwrap_or(false) {
                        continue;
                    }
                    let (width, height) = match options.sensor_size {
                        Some(size) => size,
                        None => read_event_array_header(data.data)?,
                    };
                    let builder = builder
                        .get_or_insert_with(|| EventStreamBuilder::new(width, height, 0.001));
                    let stopped = decode_event_array(data.data, &mut |x, y, t, p| {
                        builder.push(x, y, t, p);
                        options.max_events.is_some_and(|max| builder.len() >= max)
                    })?;
                    if stopped {
                        break 'outer;
                    }
                }
            }
        }
    }

    builder.map(EventStreamBuilder::build).ok_or_else(|| {
        IoError::Format(format!(
            "no {EVENT_ARRAY_TYPE} messages found on topic {topic}"
        ))
    })
}

/// Reads the `(width, height)` from a `dvs_msgs/EventArray` header (the fields precede
/// the events), without decoding the events.
fn read_event_array_header(bytes: &[u8]) -> Result<(usize, usize), IoError> {
    let mut reader = ByteReader::new(bytes);
    reader.skip(4 + 8)?; // Header: seq (u32) + stamp (sec u32, nsec u32)
    let frame_id_len = reader.u32()? as usize;
    reader.skip(frame_id_len)?; // Header: frame_id
    let height = reader.u32()? as usize;
    let width = reader.u32()? as usize;
    Ok((width, height))
}

/// Decodes one serialized `dvs_msgs/EventArray`, calling `on_event(x, y, t_us, polarity)`
/// per event. Stops early (returning `Ok(true)`) once `on_event` returns `true`. The
/// single decode path shared by `read_bag` (eager), time slicing, and the count pass.
fn decode_event_array(
    bytes: &[u8],
    on_event: &mut dyn FnMut(u16, u16, i64, bool) -> bool,
) -> Result<bool, IoError> {
    let mut reader = ByteReader::new(bytes);
    reader.skip(4 + 8)?;
    let frame_id_len = reader.u32()? as usize;
    reader.skip(frame_id_len)?;
    reader.skip(4 + 4)?; // height + width (read by `read_event_array_header`)
    let count = reader.u32()? as usize;
    for _ in 0..count {
        let x = reader.u16()?;
        let y = reader.u16()?;
        let seconds = i64::from(reader.u32()?);
        let nanoseconds = i64::from(reader.u32()?);
        let polarity = reader.u8()? != 0;
        if on_event(x, y, seconds * 1_000_000 + nanoseconds / 1000, polarity) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Writes a stream as a ROS1 bag containing one connection (`topic`, default
/// `/davis/left/events`) and a single chunk of `dvs_msgs/EventArray` messages, round-tripping
/// through [`read_bag`] and [`open_bag_slice`]. Events are batched into messages and the
/// timestamps split into ROS `sec`/`nsec` (so microseconds are preserved exactly). The whole
/// chunk is buffered, so very large streams use proportional memory.
pub fn write_bag(
    path: impl AsRef<Path>,
    stream: &EventStream,
    topic: Option<&str>,
) -> Result<(), IoError> {
    let topic = topic.unwrap_or(DEFAULT_TOPIC);
    let conn_id: u32 = 0;
    let ts = stream.ts();

    // Build the chunk payload: one connection record, then the batched message records.
    // `index_entries` records each message's (start time, byte offset within the payload).
    let mut payload: Vec<u8> =
        make_record(&connection_fields(conn_id, topic), &connection_data(topic));
    let mut index_entries: Vec<(i64, u32)> = Vec::new();
    let mut start = 0;
    while start < stream.len() {
        let end = (start + MESSAGE_EVENTS).min(stream.len());
        let offset = payload.len() as u32;
        let message = serialize_event_array(stream, start, end);
        write_record(&mut payload, &message_fields(conn_id, ts[start]), &message)
            .expect("writing to a Vec never fails");
        index_entries.push((ts[start], offset));
        start = end;
    }
    if stream.is_empty() {
        // Emit one empty message so `read_bag` finds a connection + data and rebuilds an
        // empty stream (with the correct sensor size from the message header).
        let offset = payload.len() as u32;
        let message = serialize_event_array(stream, 0, 0);
        write_record(&mut payload, &message_fields(conn_id, 0), &message)
            .expect("writing to a Vec never fails");
        index_entries.push((0, offset));
    }
    let message_count = index_entries.len() as u32;
    let (start_us, end_us) = match (ts.iter().min(), ts.iter().max()) {
        (Some(&lo), Some(&hi)) => (lo, hi),
        _ => (0, 0),
    };

    // The bag header has a fixed length (its fields are fixed-size), so the chunk position is
    // known before the index position is, and the header can be built once index_pos is set.
    let bag_header_len = make_record(&bag_header_fields(0, 1, 1), &[]).len();
    let chunk_pos = BAG_MAGIC.len() + bag_header_len;
    let chunk_fields = chunk_fields(payload.len() as u32);
    let mut index_data: Vec<u8> = Vec::with_capacity(index_entries.len() * 12);
    for (time_us, offset) in &index_entries {
        index_data.extend_from_slice(&ros_time(*time_us));
        index_data.extend_from_slice(&offset.to_le_bytes());
    }
    let index_record = make_record(&index_data_fields(conn_id, message_count), &index_data);
    let index_pos = chunk_pos + record_len(&chunk_fields, payload.len()) + index_record.len();

    let mut writer = BufWriter::new(File::create(path).map_err(IoError::Io)?);
    writer.write_all(BAG_MAGIC).map_err(IoError::Io)?;
    write_record(&mut writer, &bag_header_fields(index_pos as u64, 1, 1), &[])
        .map_err(IoError::Io)?;
    write_record(&mut writer, &chunk_fields, &payload).map_err(IoError::Io)?;
    writer.write_all(&index_record).map_err(IoError::Io)?;
    // Footer (read by `index_records`): the connection, then the chunk info.
    write_record(
        &mut writer,
        &connection_fields(conn_id, topic),
        &connection_data(topic),
    )
    .map_err(IoError::Io)?;
    let chunk_info_data = {
        let mut data = Vec::with_capacity(8);
        data.extend_from_slice(&conn_id.to_le_bytes());
        data.extend_from_slice(&message_count.to_le_bytes());
        data
    };
    write_record(
        &mut writer,
        &chunk_info_fields(chunk_pos as u64, start_us, end_us),
        &chunk_info_data,
    )
    .map_err(IoError::Io)?;
    writer.flush().map_err(IoError::Io)
}

/// Splits a microsecond timestamp into the ROS `sec`/`nsec` pair (8 bytes, little-endian) the
/// reader recombines into nanoseconds — exact for microsecond data. Negative times clamp to 0.
fn ros_time(t_us: i64) -> [u8; 8] {
    let t_us = t_us.max(0);
    let seconds = (t_us / 1_000_000) as u32;
    let nanoseconds = ((t_us % 1_000_000) * 1000) as u32;
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&seconds.to_le_bytes());
    bytes[4..].copy_from_slice(&nanoseconds.to_le_bytes());
    bytes
}

/// Serializes events `[start, end)` as a `dvs_msgs/EventArray` body matching [`decode_event_array`].
fn serialize_event_array(stream: &EventStream, start: usize, end: usize) -> Vec<u8> {
    let (width, height) = stream.sensor_size();
    let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
    let stamp = ts.get(start).copied().unwrap_or(0);
    let mut message = Vec::with_capacity(24 + (end - start) * 13);
    message.extend_from_slice(&0u32.to_le_bytes()); // header.seq
    message.extend_from_slice(&ros_time(stamp)); // header.stamp
    message.extend_from_slice(&0u32.to_le_bytes()); // header.frame_id (empty string)
    message.extend_from_slice(&(height as u32).to_le_bytes());
    message.extend_from_slice(&(width as u32).to_le_bytes());
    message.extend_from_slice(&((end - start) as u32).to_le_bytes());
    for index in start..end {
        message.extend_from_slice(&xs[index].to_le_bytes());
        message.extend_from_slice(&ys[index].to_le_bytes());
        message.extend_from_slice(&ros_time(ts[index]));
        message.push(u8::from(ps[index]));
    }
    message
}

/// One `<len:u32><name=value>` record-header field.
fn field(name: &str, value: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + name.len() + 1 + value.len());
    let body_len = (name.len() + 1 + value.len()) as u32;
    bytes.extend_from_slice(&body_len.to_le_bytes());
    bytes.extend_from_slice(name.as_bytes());
    bytes.push(b'=');
    bytes.extend_from_slice(value);
    bytes
}

/// Byte length of a record: `<header_len:u32><header><data_len:u32><data>`.
fn record_len(fields: &[Vec<u8>], data_len: usize) -> usize {
    4 + fields.iter().map(Vec::len).sum::<usize>() + 4 + data_len
}

/// Builds a complete record in memory (for the small records whose bytes/length are reused).
fn make_record(fields: &[Vec<u8>], data: &[u8]) -> Vec<u8> {
    let mut record = Vec::with_capacity(record_len(fields, data.len()));
    write_record(&mut record, fields, data).expect("writing to a Vec never fails");
    record
}

/// Streams one record to `writer` (used for the large chunk record to avoid copying the payload).
fn write_record<W: Write>(writer: &mut W, fields: &[Vec<u8>], data: &[u8]) -> std::io::Result<()> {
    let header_len: usize = fields.iter().map(Vec::len).sum();
    writer.write_all(&(header_len as u32).to_le_bytes())?;
    for field in fields {
        writer.write_all(field)?;
    }
    writer.write_all(&(data.len() as u32).to_le_bytes())?;
    writer.write_all(data)
}

fn bag_header_fields(index_pos: u64, conn_count: u32, chunk_count: u32) -> Vec<Vec<u8>> {
    vec![
        field("op", &[0x03]),
        field("index_pos", &index_pos.to_le_bytes()),
        field("conn_count", &conn_count.to_le_bytes()),
        field("chunk_count", &chunk_count.to_le_bytes()),
    ]
}

fn chunk_fields(uncompressed_size: u32) -> Vec<Vec<u8>> {
    vec![
        field("op", &[0x05]),
        field("compression", b"none"),
        field("size", &uncompressed_size.to_le_bytes()),
    ]
}

fn connection_fields(conn_id: u32, topic: &str) -> Vec<Vec<u8>> {
    vec![
        field("op", &[0x07]),
        field("conn", &conn_id.to_le_bytes()),
        field("topic", topic.as_bytes()),
    ]
}

fn connection_data(topic: &str) -> Vec<u8> {
    let fields = [
        field("topic", topic.as_bytes()),
        field("type", EVENT_ARRAY_TYPE.as_bytes()),
        field("md5sum", EVENT_ARRAY_MD5.as_bytes()),
        field("message_definition", EVENT_ARRAY_DEF.as_bytes()),
    ];
    fields.concat()
}

fn message_fields(conn_id: u32, time_us: i64) -> Vec<Vec<u8>> {
    vec![
        field("op", &[0x02]),
        field("conn", &conn_id.to_le_bytes()),
        field("time", &ros_time(time_us)),
    ]
}

fn index_data_fields(conn_id: u32, count: u32) -> Vec<Vec<u8>> {
    vec![
        field("op", &[0x04]),
        field("ver", &1u32.to_le_bytes()),
        field("conn", &conn_id.to_le_bytes()),
        field("count", &count.to_le_bytes()),
    ]
}

fn chunk_info_fields(chunk_pos: u64, start_us: i64, end_us: i64) -> Vec<Vec<u8>> {
    vec![
        field("op", &[0x06]),
        field("ver", &1u32.to_le_bytes()),
        field("chunk_pos", &chunk_pos.to_le_bytes()),
        field("start_time", &ros_time(start_us)),
        field("end_time", &ros_time(end_us)),
        field("count", &1u32.to_le_bytes()),
    ]
}

/// One chunk's byte offset and (microsecond) time range, copied out of the bag's index
/// so the [`BagSliceSource`] holds no borrows of the mmapped `RosBag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ChunkMeta {
    pos: u64,
    start_us: i64,
    end_us: i64,
}

/// In-place [`SliceSource`] for rosbags. The bag's own chunk index (`ChunkInfo`) gives
/// each chunk's byte offset and time range, so time slicing seeks straight to the
/// overlapping chunks and decompresses only those — no full read, bounded memory. Event
/// **counts** aren't in the index, so `n_events`/`slice_index` decode the target chunks
/// once and cache the per-chunk cumulative counts; time slicing never triggers that.
pub struct BagSliceSource {
    bag: RosBag,
    conn_ids: Vec<u32>,
    chunks: Vec<ChunkMeta>,
    sensor: (usize, usize),
    span_us: (i64, i64),
    counts: Mutex<Option<Vec<usize>>>,
}

/// Opens a bag for lazy slicing: reads the index for the target topic's connection ids,
/// the chunks containing them (offset + time range), the span, and the sensor size
/// (override, else the first message header). No chunk data is decompressed.
pub fn open_bag_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<BagSliceSource, IoError> {
    let topic = options.topic.as_deref().unwrap_or(DEFAULT_TOPIC);
    let bag = RosBag::new(path).map_err(IoError::Io)?;

    let mut conn_ids: Vec<u32> = Vec::new();
    for record in bag.index_records() {
        if let IndexRecord::Connection(connection) = record.map_err(map_bag_error)? {
            if connection.topic == topic && connection.tp == EVENT_ARRAY_TYPE {
                conn_ids.push(connection.id);
            }
        }
    }
    if conn_ids.is_empty() {
        return Err(IoError::Format(format!(
            "no {EVENT_ARRAY_TYPE} connection on topic {topic}"
        )));
    }

    let mut chunks: Vec<ChunkMeta> = Vec::new();
    for record in bag.index_records() {
        if let IndexRecord::ChunkInfo(info) = record.map_err(map_bag_error)? {
            if info
                .entries()
                .any(|entry| conn_ids.contains(&entry.conn_id))
            {
                chunks.push(ChunkMeta {
                    pos: info.chunk_pos,
                    start_us: (info.start_time / 1000) as i64,
                    end_us: (info.end_time / 1000) as i64,
                });
            }
        }
    }
    chunks.sort_by_key(|chunk| chunk.start_us);

    let span_us = match (
        chunks.iter().map(|c| c.start_us).min(),
        chunks.iter().map(|c| c.end_us).max(),
    ) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (0, 0),
    };
    let sensor = match options.sensor_size {
        Some(size) => size,
        None => detect_sensor(&bag, &chunks, &conn_ids)?,
    };

    Ok(BagSliceSource {
        bag,
        conn_ids,
        chunks,
        sensor,
        span_us,
        counts: Mutex::new(None),
    })
}

/// Reads the sensor size from the first event message of the first chunk.
fn detect_sensor(
    bag: &RosBag,
    chunks: &[ChunkMeta],
    conn_ids: &[u32],
) -> Result<(usize, usize), IoError> {
    let Some(first) = chunks.first() else {
        return Ok((1, 1));
    };
    let mut iterator = bag.chunk_records();
    iterator.seek(first.pos).map_err(map_bag_error)?;
    if let Some(record) = iterator.next() {
        if let ChunkRecord::Chunk(chunk) = record.map_err(map_bag_error)? {
            for message in chunk.messages() {
                if let MessageRecord::MessageData(data) = message.map_err(map_bag_error)? {
                    if conn_ids.contains(&data.conn_id) {
                        return read_event_array_header(data.data);
                    }
                }
            }
        }
    }
    Ok((1, 1))
}

/// Chunks whose time range overlaps the half-open window `[t0, t1)` (µs).
fn select_chunks(chunks: &[ChunkMeta], t0: i64, t1: i64) -> impl Iterator<Item = &ChunkMeta> {
    chunks
        .iter()
        .filter(move |chunk| chunk.start_us < t1 && chunk.end_us >= t0)
}

/// Index of the chunk containing global event `i`, given cumulative per-chunk counts
/// (`counts[k]` = events before chunk `k`, `counts.len() == chunks + 1`).
fn locate_chunk(counts: &[usize], i: usize) -> usize {
    counts
        .partition_point(|&count| count <= i)
        .saturating_sub(1)
}

impl BagSliceSource {
    /// Seeks to `chunk` and calls `on_event` for every event on a target connection,
    /// stopping the chunk early once `on_event` returns `true`.
    fn for_each_event(
        &self,
        chunk: &ChunkMeta,
        on_event: &mut dyn FnMut(u16, u16, i64, bool) -> bool,
    ) -> Result<(), IoError> {
        let mut iterator = self.bag.chunk_records();
        iterator.seek(chunk.pos).map_err(map_bag_error)?;
        let Some(record) = iterator.next() else {
            return Ok(());
        };
        let ChunkRecord::Chunk(decoded) = record.map_err(map_bag_error)? else {
            return Ok(());
        };
        for message in decoded.messages() {
            if let MessageRecord::MessageData(data) = message.map_err(map_bag_error)? {
                if self.conn_ids.contains(&data.conn_id) && decode_event_array(data.data, on_event)?
                {
                    break;
                }
            }
        }
        Ok(())
    }

    /// Cumulative event count per chunk, decoded once and cached.
    fn cumulative_counts(&self) -> Result<Vec<usize>, IoError> {
        let mut guard = self.counts.lock().unwrap();
        if let Some(counts) = guard.as_ref() {
            return Ok(counts.clone());
        }
        let mut counts = Vec::with_capacity(self.chunks.len() + 1);
        counts.push(0);
        for chunk in &self.chunks {
            let mut events = 0usize;
            self.for_each_event(chunk, &mut |_, _, _, _| {
                events += 1;
                false
            })?;
            counts.push(counts.last().unwrap() + events);
        }
        *guard = Some(counts.clone());
        Ok(counts)
    }
}

impl SliceSource for BagSliceSource {
    fn sensor_size(&self) -> (usize, usize) {
        self.sensor
    }

    fn timestamp_scale_ms(&self) -> f64 {
        0.001
    }

    fn n_events(&self) -> usize {
        // The count requires decoding; treat a decode error as "unknown" (0).
        self.cumulative_counts()
            .ok()
            .and_then(|counts| counts.last().copied())
            .unwrap_or(0)
    }

    fn time_span(&self) -> (i64, i64) {
        self.span_us
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let mut builder = EventStreamBuilder::new(self.sensor.0, self.sensor.1, 0.001);
        for chunk in select_chunks(&self.chunks, t0, t1) {
            self.for_each_event(chunk, &mut |x, y, t, p| {
                if t >= t1 {
                    return true; // events are time-ordered within a chunk
                }
                if t >= t0 {
                    builder.push(x, y, t, p);
                }
                false
            })?;
        }
        Ok(builder.build())
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let counts = self.cumulative_counts()?;
        let total = counts.last().copied().unwrap_or(0);
        let i0 = i0.min(total);
        let i1 = i1.clamp(i0, total);
        let mut builder = EventStreamBuilder::new(self.sensor.0, self.sensor.1, 0.001);
        if i0 == i1 {
            return Ok(builder.build());
        }
        for (offset, chunk) in self
            .chunks
            .iter()
            .enumerate()
            .skip(locate_chunk(&counts, i0))
        {
            if counts[offset] >= i1 {
                break;
            }
            let mut index = counts[offset];
            self.for_each_event(chunk, &mut |x, y, t, p| {
                if (i0..i1).contains(&index) {
                    builder.push(x, y, t, p);
                }
                index += 1;
                index >= i1
            })?;
        }
        Ok(builder.build())
    }
}

fn map_bag_error(error: rosbag::Error) -> IoError {
    IoError::Format(format!("rosbag: {error}"))
}

/// Little-endian cursor over a ROS message payload.
struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], IoError> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| IoError::Format("truncated dvs_msgs/EventArray message".to_owned()))?;
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn skip(&mut self, count: usize) -> Result<(), IoError> {
        self.take(count).map(|_| ())
    }

    fn u8(&mut self) -> Result<u8, IoError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, IoError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, IoError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::{locate_chunk, open_bag_slice, read_bag, select_chunks, write_bag, ChunkMeta};
    use crate::io::{LoadOptions, SliceSource};
    use crate::{EventStream, EventStreamBuilder};

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("eventcv_{tag}_{nanos}.bag"))
    }

    fn sample_stream() -> EventStream {
        let mut builder = EventStreamBuilder::new(16, 12, 0.001);
        for &(x, y, t, p) in &[
            (0u16, 0u16, 5i64, true),
            (3, 4, 1_000_001, false),
            (15, 11, 2_500_000, true),
            (7, 8, 2_500_000, false),
        ] {
            builder.push(x, y, t, p);
        }
        builder.build()
    }

    #[test]
    fn bag_round_trips_through_the_reader() {
        let stream = sample_stream();
        let path = temp_path("bag_rt");
        write_bag(&path, &stream, None).unwrap();

        let loaded = read_bag(&path, &LoadOptions::default()).unwrap();
        assert_eq!(loaded.sensor_size(), (16, 12));
        assert_eq!(loaded.xs(), stream.xs());
        assert_eq!(loaded.ys(), stream.ys());
        assert_eq!(loaded.ts(), stream.ts()); // microseconds preserved exactly
        assert_eq!(loaded.ps(), stream.ps());

        // The lazy slicer reads the same file in place.
        let reader = open_bag_slice(&path, &LoadOptions::default()).unwrap();
        assert_eq!(reader.sensor_size(), (16, 12));
        assert_eq!(reader.n_events(), stream.len());
        assert_eq!(reader.time_span(), (5, 2_500_000));
        let window = reader.slice_time(1_000_000, 2_000_000).unwrap();
        assert_eq!(window.ts(), &[1_000_001]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bag_round_trips_an_empty_stream() {
        let stream = EventStreamBuilder::new(8, 6, 0.001).build();
        let path = temp_path("bag_empty");
        write_bag(&path, &stream, Some("/cam/events")).unwrap();

        let options = LoadOptions {
            topic: Some("/cam/events".to_owned()),
            ..LoadOptions::default()
        };
        let loaded = read_bag(&path, &options).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.sensor_size(), (8, 6));

        std::fs::remove_file(&path).ok();
    }

    fn meta(pos: u64, start_us: i64, end_us: i64) -> ChunkMeta {
        ChunkMeta {
            pos,
            start_us,
            end_us,
        }
    }

    #[test]
    fn select_chunks_picks_overlapping_windows() {
        let chunks = [meta(0, 0, 100), meta(1, 100, 200), meta(2, 200, 300)];

        let picked: Vec<u64> = select_chunks(&chunks, 150, 250).map(|c| c.pos).collect();
        assert_eq!(picked, [1, 2]); // [150, 250) overlaps chunks 1 and 2

        // Half-open: a window ending exactly at a chunk's start excludes it.
        let picked: Vec<u64> = select_chunks(&chunks, 0, 100).map(|c| c.pos).collect();
        assert_eq!(picked, [0]);

        assert!(select_chunks(&chunks, 1000, 2000).next().is_none());
    }

    #[test]
    fn locate_chunk_finds_the_containing_chunk() {
        let counts = [0usize, 10, 25, 40]; // cumulative counts for 3 chunks

        assert_eq!(locate_chunk(&counts, 0), 0);
        assert_eq!(locate_chunk(&counts, 9), 0);
        assert_eq!(locate_chunk(&counts, 10), 1);
        assert_eq!(locate_chunk(&counts, 24), 1);
        assert_eq!(locate_chunk(&counts, 39), 2);
    }
}
