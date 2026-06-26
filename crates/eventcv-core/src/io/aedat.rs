//! AEDAT 2.0 reader (jAER format, e.g. DAVIS recordings). The file is an ASCII comment
//! header (lines starting with `#`) followed by big-endian 8-byte records — a 32-bit
//! address and a 32-bit microsecond timestamp. The raw stream interleaves DVS events with
//! APS (frame) and IMU samples; we keep only the DVS events.
//!
//! DAVIS address layout (verified against a real DAVIS346 recording): a record is a DVS
//! event when bit 31 (APS/ADC sample) and bit 10 (IMU/type) are both clear; then
//! `y = bits 22..30 (9b)`, `x = bits 12..21 (10b)`, `polarity = bit 11`. jAER's y origin is
//! the bottom-left, so we flip rows (`y = height-1 - y`) to the top-left image convention
//! the other readers use — otherwise frames render upside down.

use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::Path;

use super::{EventSource, IoError, LoadOptions, RawEvent};
use crate::EventStream;

/// AEDAT 2.0 timestamps tick in microseconds.
const TIMESTAMP_SCALE_MS: f64 = 0.001;

const APS_SAMPLE_FLAG: u32 = 0x8000_0000; // bit 31: APS/ADC read, not a DVS event
const TYPE_FLAG: u32 = 0x0000_0400; // bit 10: IMU/other, not a DVS event

/// Reads the ASCII comment header, leaving `reader` positioned at the first binary record.
/// Header lines begin with `#`; the first line that does not marks the start of the body.
fn read_header<R: BufRead>(reader: &mut R) -> Result<Vec<String>, IoError> {
    let mut lines = Vec::new();
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
        lines.push(String::from_utf8_lossy(&line).trim_end().to_owned());
    }
    Ok(lines)
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

/// DAVIS-family sensors share the DVS bit layout; resolve the grid from the chip named in
/// the header (or an explicit override). DAVIS346 → 346×260, DAVIS240 → 240×180.
fn resolve_sensor(
    header: &[String],
    sensor: Option<(usize, usize)>,
) -> Result<(usize, usize), IoError> {
    if let Some(size) = sensor {
        return Ok(size);
    }
    let text = header.join("\n").to_ascii_lowercase();
    if text.contains("davis346") || text.contains("davis 346") {
        Ok((346, 260))
    } else if text.contains("davis240") || text.contains("davis 240") {
        Ok((240, 180))
    } else {
        Err(IoError::Unsupported(
            "could not determine the DAVIS sensor geometry from the AEDAT header; pass sensor_size"
                .to_owned(),
        ))
    }
}

/// Pull-based source over the big-endian record stream, yielding DVS events only.
struct AedatSource<R: BufRead> {
    reader: R,
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
        let mut record = [0u8; 8];
        loop {
            match self.reader.read_exact(&mut record) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
                Err(error) => return Err(IoError::Io(error)),
            }
            let address = u32::from_be_bytes([record[0], record[1], record[2], record[3]]);
            let timestamp = u32::from_be_bytes([record[4], record[5], record[6], record[7]]);
            if address & (APS_SAMPLE_FLAG | TYPE_FLAG) != 0 {
                continue; // skip APS / IMU samples
            }
            // jAER stores y with the origin at the bottom-left; flip to the top-left image
            // convention the other readers use (out-of-range rows are left for the builder
            // to drop). x is unchanged.
            let y_raw = ((address >> 22) & 0x1FF) as usize;
            let y = self.height.checked_sub(1 + y_raw).unwrap_or(self.height) as u16;
            return Ok(Some(RawEvent {
                x: ((address >> 12) & 0x3FF) as u16,
                y,
                t: i64::from(timestamp),
                p: address & (1 << 11) != 0,
            }));
        }
    }
}

fn read_aedat_from<R: BufRead>(
    mut reader: R,
    options: &LoadOptions,
) -> Result<EventStream, IoError> {
    let header = read_header(&mut reader)?;
    check_version(&header)?;
    let (width, height) = resolve_sensor(&header, options.sensor_size)?;
    super::read_capped(
        AedatSource {
            reader,
            width,
            height,
        },
        options.max_events,
    )
}

/// Reads an AEDAT 2.0 (`.aedat`) recording. `sensor_size` overrides the chip geometry; the
/// timestamp unit is always microseconds. `max_events` caps how many events are kept.
pub fn read_aedat(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    read_aedat_from(BufReader::new(File::open(path)?), options)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn dvs_address(x: u32, y: u32, polarity: bool) -> u32 {
        (y << 22) | (x << 12) | ((polarity as u32) << 11)
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
}
