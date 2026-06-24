use std::collections::HashMap;
use std::path::Path;

use rosbag::{ChunkRecord, MessageRecord, RosBag};

use super::{IoError, LoadOptions};
use crate::{EventStream, EventStreamBuilder};

const DEFAULT_TOPIC: &str = "/davis/left/events";
const EVENT_ARRAY_TYPE: &str = "dvs_msgs/EventArray";

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
                    if wanted.get(&data.conn_id).copied().unwrap_or(false)
                        && decode_event_array(
                            data.data,
                            &mut builder,
                            options.sensor_size,
                            options.max_events,
                        )?
                    {
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

/// Decodes one serialized `dvs_msgs/EventArray`, appending events to `builder`
/// (created on first call). Returns `true` once `max` events have been collected.
fn decode_event_array(
    bytes: &[u8],
    builder: &mut Option<EventStreamBuilder>,
    sensor: Option<(usize, usize)>,
    max: Option<usize>,
) -> Result<bool, IoError> {
    let mut reader = ByteReader::new(bytes);
    reader.skip(4 + 8)?; // Header: seq (u32) + stamp (sec u32, nsec u32)
    let frame_id_len = reader.u32()? as usize;
    reader.skip(frame_id_len)?; // Header: frame_id
    let height = reader.u32()? as usize;
    let width = reader.u32()? as usize;

    let builder = builder.get_or_insert_with(|| {
        let (width, height) = sensor.unwrap_or((width, height));
        EventStreamBuilder::new(width, height, 0.001)
    });

    let count = reader.u32()? as usize;
    for _ in 0..count {
        let x = reader.u16()?;
        let y = reader.u16()?;
        let seconds = i64::from(reader.u32()?);
        let nanoseconds = i64::from(reader.u32()?);
        let polarity = reader.u8()? != 0;
        builder.push(x, y, seconds * 1_000_000 + nanoseconds / 1000, polarity);
        if max.is_some_and(|max| builder.len() >= max) {
            return Ok(true);
        }
    }
    Ok(false)
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
