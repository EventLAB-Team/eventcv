//! Prophesee `.dat` CD (change-detection) event reader. The file is an ASCII comment
//! header (lines starting with `%`), then two bytes giving the event type and event size,
//! then little-endian 8-byte records: a 32-bit microsecond timestamp and a 32-bit packed
//! word `x = bits 0..13`, `y = bits 14..27`, `polarity = bit 28`.
//!
//! Implemented from the published `.dat` layout (used by Prophesee/Metavision and Tonic).
//! It has not yet been checked against a real `.dat` file — the unit tests cover the byte
//! layout exactly. The EVT2/EVT3 `.raw` encodings are a separate, unimplemented format.

use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::Path;

use super::{EventSource, IoError, LoadOptions, RawEvent};
use crate::{EventStream, EventStreamBuilder};

/// Prophesee timestamps tick in microseconds.
const TIMESTAMP_SCALE_MS: f64 = 0.001;

/// Only the 8-byte CD event layout is implemented.
const CD_EVENT_SIZE: u8 = 8;

/// Reads the ASCII comment header (lines starting with `%`), leaving `reader` positioned at
/// the event-type/size bytes.
fn read_header<R: BufRead>(reader: &mut R) -> Result<Vec<String>, IoError> {
    let mut lines = Vec::new();
    loop {
        let is_comment = match reader.fill_buf()? {
            [] => break,
            buf => buf[0] == b'%',
        };
        if !is_comment {
            break;
        }
        let mut line = Vec::new();
        reader.read_until(b'\n', &mut line)?;
        lines.push(String::from_utf8_lossy(&line).trim_end().to_owned());
    }
    Ok(lines)
}

/// Pulls the sensor size from `% Width`/`% Height` header lines, or an explicit override.
fn resolve_sensor(
    header: &[String],
    sensor: Option<(usize, usize)>,
) -> Result<(usize, usize), IoError> {
    if let Some(size) = sensor {
        return Ok(size);
    }
    let (mut width, mut height) = (None, None);
    for line in header {
        let body = line.trim_start_matches('%').trim().to_ascii_lowercase();
        let value = |rest: &str| rest.split_whitespace().next().and_then(|v| v.parse().ok());
        if let Some(rest) = body.strip_prefix("width") {
            width = value(rest);
        } else if let Some(rest) = body.strip_prefix("height") {
            height = value(rest);
        }
    }
    match (width, height) {
        (Some(width), Some(height)) => Ok((width, height)),
        _ => Err(IoError::Unsupported(
            "could not determine sensor size from the Prophesee .dat header; pass sensor_size"
                .to_owned(),
        )),
    }
}

struct DatSource<R: BufRead> {
    reader: R,
    width: usize,
    height: usize,
}

impl<R: BufRead> EventSource for DatSource<R> {
    fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn timestamp_scale_ms(&self) -> f64 {
        TIMESTAMP_SCALE_MS
    }

    fn next_event(&mut self) -> Result<Option<RawEvent>, IoError> {
        let mut record = [0u8; 8];
        match self.reader.read_exact(&mut record) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(IoError::Io(error)),
        }
        let timestamp = u32::from_le_bytes([record[0], record[1], record[2], record[3]]);
        let data = u32::from_le_bytes([record[4], record[5], record[6], record[7]]);
        Ok(Some(RawEvent {
            x: (data & 0x3FFF) as u16,
            y: ((data >> 14) & 0x3FFF) as u16,
            t: i64::from(timestamp),
            p: (data >> 28) & 1 != 0,
        }))
    }
}

fn read_dat_from<R: BufRead>(mut reader: R, options: &LoadOptions) -> Result<EventStream, IoError> {
    let header = read_header(&mut reader)?;
    let (width, height) = resolve_sensor(&header, options.sensor_size)?;

    let mut type_size = [0u8; 2];
    match reader.read_exact(&mut type_size) {
        Ok(()) => {
            let event_size = type_size[1];
            if event_size != CD_EVENT_SIZE {
                return Err(IoError::Unsupported(format!(
                    "unsupported Prophesee .dat event size {event_size} (only 8-byte CD events are implemented)"
                )));
            }
        }
        // Header with no events: a valid, empty recording.
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
            return Ok(EventStreamBuilder::new(width, height, TIMESTAMP_SCALE_MS).build());
        }
        Err(error) => return Err(IoError::Io(error)),
    }

    super::read_capped(
        DatSource {
            reader,
            width,
            height,
        },
        options.max_events,
    )
}

/// Reads a Prophesee `.dat` recording. `sensor_size` overrides the header geometry; the
/// timestamp unit is always microseconds. `max_events` caps how many events are kept.
pub fn read_dat(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    read_dat_from(BufReader::new(File::open(path)?), options)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn cd_word(x: u32, y: u32, polarity: bool) -> u32 {
        (x & 0x3FFF) | ((y & 0x3FFF) << 14) | ((polarity as u32) << 28)
    }

    fn record(timestamp: u32, data: u32) -> [u8; 8] {
        let mut bytes = [0u8; 8];
        bytes[0..4].copy_from_slice(&timestamp.to_le_bytes());
        bytes[4..8].copy_from_slice(&data.to_le_bytes());
        bytes
    }

    fn file_with(records: &[[u8; 8]]) -> Vec<u8> {
        let mut data = b"% Date 2024-01-01\n% Width 640\n% Height 480\n% Version 2\n".to_vec();
        data.extend_from_slice(&[0x00, CD_EVENT_SIZE]); // event type + size
        for record in records {
            data.extend_from_slice(record);
        }
        data
    }

    fn read(data: &[u8], options: &LoadOptions) -> Result<EventStream, IoError> {
        read_dat_from(Cursor::new(data.to_vec()), options)
    }

    #[test]
    fn decodes_cd_events() {
        let data = file_with(&[
            record(1_000, cd_word(10, 20, true)),
            record(1_005, cd_word(600, 400, false)),
        ]);
        let stream = read(&data, &LoadOptions::default()).unwrap();

        assert_eq!(stream.sensor_size(), (640, 480));
        assert_eq!(stream.len(), 2);
        assert_eq!(stream.xs(), &[10, 600]);
        assert_eq!(stream.ys(), &[20, 400]);
        assert_eq!(stream.ts(), &[1_000, 1_005]);
        assert_eq!(stream.ps(), &[true, false]);
    }

    #[test]
    fn header_only_file_is_empty() {
        let data = b"% Width 640\n% Height 480\n".to_vec();
        let stream = read(&data, &LoadOptions::default()).unwrap();
        assert!(stream.is_empty());
        assert_eq!(stream.sensor_size(), (640, 480));
    }

    #[test]
    fn out_of_bounds_events_are_dropped() {
        let data = file_with(&[
            record(1, cd_word(1, 1, true)),
            record(2, cd_word(700, 1, true)), // x >= width -> dropped
        ]);
        let options = LoadOptions {
            sensor_size: Some((4, 4)),
            ..LoadOptions::default()
        };
        assert_eq!(read(&data, &options).unwrap().len(), 1);
    }

    #[test]
    fn non_cd_event_size_is_unsupported() {
        let mut data = b"% Width 640\n% Height 480\n".to_vec();
        data.extend_from_slice(&[0x0C, 16]); // 16-byte events -> unsupported
        match read(&data, &LoadOptions::default()) {
            Err(IoError::Unsupported(message)) => assert!(message.contains("event size")),
            other => panic!("expected unsupported event-size error, got {other:?}"),
        }
    }

    #[test]
    fn missing_geometry_without_override_is_unsupported() {
        let mut data = b"% Date 2024-01-01\n% Version 2\n".to_vec();
        data.extend_from_slice(&[0x00, CD_EVENT_SIZE]);
        data.extend_from_slice(&record(1, cd_word(1, 1, true)));
        match read(&data, &LoadOptions::default()) {
            Err(IoError::Unsupported(message)) => assert!(message.contains("sensor_size")),
            other => panic!("expected unsupported geometry error, got {other:?}"),
        }
    }

    #[test]
    fn explicit_sensor_size_overrides_the_header() {
        let data = file_with(&[record(1, cd_word(1, 1, true))]);
        let options = LoadOptions {
            sensor_size: Some((1280, 720)),
            ..LoadOptions::default()
        };
        assert_eq!(read(&data, &options).unwrap().sensor_size(), (1280, 720));
    }
}
