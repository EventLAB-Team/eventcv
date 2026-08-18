use std::fs::File;
use std::io::{BufReader, Seek, Write};
use std::path::Path;

use npyz::npz::{NpzArchive, NpzWriter};
use npyz::{AutoSerialize, Deserialize, Serialize, WriterBuilder};
use zip::write::FileOptions;
use zip::CompressionMethod;

use super::IoError;
use crate::representation::{EventFrame, EventFrameData, RepresentationKind};
use crate::{EventStream, EventStreamBuilder};

const EVENT_DATA_KEY: &str = "event_data";
const N_IMAGENET_WIDTH: usize = 640;
const N_IMAGENET_HEIGHT: usize = 480;
const DEFAULT_SCALE_MS: f64 = 0.001;

#[derive(npyz::Deserialize)]
struct NImageNetEvent {
    x: u16,
    y: u16,
    t: u16,
    p: bool,
}

/// Reads a `.npz` archive of events, auto-detecting the layout: an N-ImageNet
/// `event_data` structured array, or EventCV's native per-column layout written by
/// [`write_npz_stream`] (`x`/`y`/`t`/`p` arrays + `width`/`height`/`timestamp_scale_ms`
/// metadata). `sensor` overrides the grid; out-of-bounds coordinates are dropped.
pub fn read_npz(
    path: impl AsRef<Path>,
    sensor: Option<(usize, usize)>,
) -> Result<EventStream, IoError> {
    let mut archive = open_archive(path)?;
    if archive
        .by_name(EVENT_DATA_KEY)
        .map_err(IoError::Io)?
        .is_some()
    {
        read_n_imagenet(&mut archive, sensor)
    } else {
        read_native_stream(&mut archive, sensor)
    }
}

/// Reads the N-ImageNet `event_data` structured array (`t` is a `u16`); the historical
/// default layout. `sensor` overrides the default 640x480 grid.
fn read_n_imagenet(
    archive: &mut NpzArchive<BufReader<File>>,
    sensor: Option<(usize, usize)>,
) -> Result<EventStream, IoError> {
    let (width, height) = sensor.unwrap_or((N_IMAGENET_WIDTH, N_IMAGENET_HEIGHT));
    let npy = archive
        .by_name(EVENT_DATA_KEY)
        .map_err(IoError::Io)?
        .ok_or_else(|| IoError::Format("missing event_data array".to_owned()))?;

    let [event_count] = npy.shape() else {
        return Err(IoError::Format(
            "event_data must be one-dimensional".to_owned(),
        ));
    };
    let event_count = usize::try_from(*event_count)
        .map_err(|_| IoError::Format("event_data is too large".to_owned()))?;
    let records = npy
        .data::<NImageNetEvent>()
        .map_err(|error| IoError::Format(error.to_string()))?;
    let mut builder =
        EventStreamBuilder::with_capacity(width, height, DEFAULT_SCALE_MS, event_count);

    for record in records {
        let record = record.map_err(|error| IoError::Format(error.to_string()))?;
        if !builder.push(record.x, record.y, i64::from(record.t), record.p) {
            return Err(IoError::Format(format!(
                "event coordinate ({}, {}) exceeds sensor size {}x{}",
                record.x, record.y, width, height
            )));
        }
    }

    Ok(builder.build())
}

/// Reads EventCV's native per-column layout. Metadata is honoured when present; missing
/// `sensor`/metadata falls back to inference (size from the coordinate range, default scale).
fn read_native_stream(
    archive: &mut NpzArchive<BufReader<File>>,
    sensor: Option<(usize, usize)>,
) -> Result<EventStream, IoError> {
    let xs = read_array::<u16>(archive, "x")?;
    let ys = read_array::<u16>(archive, "y")?;
    let ts = read_array::<i64>(archive, "t")?;
    let ps = read_array::<u8>(archive, "p")?;
    let count = xs.len();
    if ys.len() != count || ts.len() != count || ps.len() != count {
        return Err(IoError::Format(
            "event columns x/y/t/p have mismatched lengths".to_owned(),
        ));
    }

    let scale = read_scalar::<f64>(archive, "timestamp_scale_ms")?.unwrap_or(DEFAULT_SCALE_MS);
    let (width, height) = match sensor {
        Some(size) => size,
        None => match (
            read_scalar::<u64>(archive, "width")?,
            read_scalar::<u64>(archive, "height")?,
        ) {
            (Some(width), Some(height)) => (width as usize, height as usize),
            _ => infer_size(&xs, &ys),
        },
    };

    let mut builder = EventStreamBuilder::with_capacity(width, height, scale, count);
    for index in 0..count {
        builder.push(xs[index], ys[index], ts[index], ps[index] != 0);
    }
    Ok(builder.build())
}

/// Persists a stream as a native EventCV `.npz`: the four columns plus `width`, `height`,
/// and `timestamp_scale_ms`, so [`read_npz`] reconstructs it exactly. Uncompressed (STORED),
/// matching `numpy.savez`.
pub fn write_npz_stream(path: impl AsRef<Path>, stream: &EventStream) -> Result<(), IoError> {
    let mut npz = NpzWriter::create(path).map_err(IoError::Io)?;
    write_array(&mut npz, "t", stream.ts())?;
    write_array(&mut npz, "x", stream.xs())?;
    write_array(&mut npz, "y", stream.ys())?;
    let polarities: Vec<u8> = stream.ps().iter().map(|&p| u8::from(p)).collect();
    write_array(&mut npz, "p", &polarities)?;
    let (width, height) = stream.sensor_size();
    write_array(&mut npz, "width", &[width as u64])?;
    write_array(&mut npz, "height", &[height as u64])?;
    write_array(
        &mut npz,
        "timestamp_scale_ms",
        &[stream.timestamp_scale_ms()],
    )?;
    finish(npz)
}

/// The `.npy` preamble: magic, version, and the `u16` header length.
const NPY_PREAMBLE: usize = 10;

/// numpy pads the header dict so the array data starts on a 64-byte boundary.
const NPY_ALIGN: usize = 64;

/// Builds a `.npy` header for a 1-D array of `len` elements with numpy dtype string `descr`.
fn npy_header(descr: &str, len: usize) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({len},), }}");
    // The dict is padded with spaces and newline-terminated so `NPY_PREAMBLE + header` lands on a
    // 64-byte boundary, which is what numpy itself writes.
    let padded = (NPY_PREAMBLE + dict.len() + 1).next_multiple_of(NPY_ALIGN) - NPY_PREAMBLE;
    let mut header = Vec::with_capacity(NPY_PREAMBLE + padded);
    header.extend_from_slice(b"\x93NUMPY\x01\x00");
    header.extend_from_slice(&(padded as u16).to_le_bytes());
    header.extend_from_slice(dict.as_bytes());
    header.resize(NPY_PREAMBLE + padded - 1, b' ');
    header.push(b'\n');
    header
}

/// One event column being spilled to a scratch file while the recording is still open.
struct Column {
    descr: &'static str,
    name: &'static str,
    path: std::path::PathBuf,
    writer: std::io::BufWriter<File>,
}

/// Writes a native EventCV `.npz` a window at a time — the streaming form of
/// [`write_npz_stream`].
///
/// A `.npz` is a zip of `.npy` members, and a `.npy` states its length in a header written *before*
/// the data, so the length has to be known before the first byte goes out. Rather than hold the
/// recording in memory to learn it, each column is spilled to a scratch file beside the target and
/// copied into the archive at [`finish`](super::EventSink::finish), when the count is finally
/// known. Memory stays flat; the cost is that the `.npz` does not exist until the recording is
/// closed, and that the scratch files transiently need the recording's own size on disk.
pub struct NpzEventSink {
    path: std::path::PathBuf,
    columns: Vec<Column>,
    /// `None` until the first non-empty append fixes the sensor size and time base.
    metadata: Option<((usize, usize), f64)>,
    n_events: usize,
}

impl NpzEventSink {
    pub fn create(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let path = path.as_ref().to_path_buf();
        // Scratch files sit beside the target so the copy at `finish` stays on one filesystem.
        let columns = [("t", "<i8"), ("x", "<u2"), ("y", "<u2"), ("p", "|u1")]
            .into_iter()
            .map(|(name, descr)| {
                let part = path.with_extension(format!("{name}.part"));
                Ok(Column {
                    descr,
                    name,
                    writer: std::io::BufWriter::new(File::create(&part).map_err(IoError::Io)?),
                    path: part,
                })
            })
            .collect::<Result<Vec<_>, IoError>>()?;
        Ok(Self {
            path,
            columns,
            metadata: None,
            n_events: 0,
        })
    }

    /// Streams one spilled column into the archive as a STORED `.npy` member.
    fn store_column(
        zip: &mut zip::ZipWriter<File>,
        column: &Column,
        len: usize,
    ) -> Result<(), IoError> {
        let options = FileOptions::default().compression_method(CompressionMethod::Stored);
        zip.start_file(format!("{}.npy", column.name), options)
            .map_err(|error| IoError::Format(format!("zip: {error}")))?;
        zip.write_all(&npy_header(column.descr, len))
            .map_err(IoError::Io)?;
        let mut spilled = BufReader::new(File::open(&column.path).map_err(IoError::Io)?);
        std::io::copy(&mut spilled, zip).map_err(IoError::Io)?;
        Ok(())
    }
}

impl Drop for NpzEventSink {
    fn drop(&mut self) {
        // A sink dropped without `finish` leaves no archive; the scratch files would otherwise
        // outlive it as several gigabytes of nothing.
        for column in &self.columns {
            let _ = std::fs::remove_file(&column.path);
        }
    }
}

impl super::EventSink for NpzEventSink {
    fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        if stream.is_empty() {
            return Ok(());
        }
        self.metadata
            .get_or_insert_with(|| (stream.sensor_size(), stream.timestamp_scale_ms()));
        let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
        for value in ts {
            self.columns[0]
                .writer
                .write_all(&value.to_le_bytes())
                .map_err(IoError::Io)?;
        }
        for (column, values) in self.columns[1..3].iter_mut().zip([xs, ys]) {
            for value in values {
                column
                    .writer
                    .write_all(&value.to_le_bytes())
                    .map_err(IoError::Io)?;
            }
        }
        for &value in ps {
            self.columns[3]
                .writer
                .write_all(&[u8::from(value)])
                .map_err(IoError::Io)?;
        }
        self.n_events += stream.len();
        Ok(())
    }

    fn n_events(&self) -> usize {
        self.n_events
    }

    fn flush(&mut self) -> Result<(), IoError> {
        for column in &mut self.columns {
            column.writer.flush().map_err(IoError::Io)?;
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<(), IoError> {
        for column in &mut self.columns {
            column.writer.flush().map_err(IoError::Io)?;
        }
        let mut zip = zip::ZipWriter::new(File::create(&self.path).map_err(IoError::Io)?);
        for column in &self.columns {
            Self::store_column(&mut zip, column, self.n_events)?;
        }
        // An empty recording never learned a sensor size; fall back to the same 1x1 the readers'
        // `infer_size` produces for one, so the file still round-trips.
        let ((width, height), scale) = self.metadata.unwrap_or(((1, 1), DEFAULT_SCALE_MS));
        let mut scalar = |name: &str, descr: &str, bytes: &[u8]| -> Result<(), IoError> {
            let options = FileOptions::default().compression_method(CompressionMethod::Stored);
            zip.start_file(format!("{name}.npy"), options)
                .map_err(|error| IoError::Format(format!("zip: {error}")))?;
            zip.write_all(&npy_header(descr, 1)).map_err(IoError::Io)?;
            zip.write_all(bytes).map_err(IoError::Io)
        };
        scalar("width", "<u8", &(width as u64).to_le_bytes())?;
        scalar("height", "<u8", &(height as u64).to_le_bytes())?;
        scalar("timestamp_scale_ms", "<f8", &scale.to_le_bytes())?;
        zip.finish()
            .map_err(|error| IoError::Format(format!("zip: {error}")))?;
        Ok(())
    }
}

/// Persists an [`EventFrame`] as `.npz`: the flat `data` (native dtype), the `[C, H, W]`
/// `shape`, a `dtype` code, and the `kind` / `channel_names` tags so [`read_npz_frame`]
/// recovers the frame verbatim.
pub fn write_npz_frame(path: impl AsRef<Path>, frame: &EventFrame) -> Result<(), IoError> {
    let (channels, height, width) = frame.shape();
    let mut npz = NpzWriter::create(path).map_err(IoError::Io)?;
    write_array(
        &mut npz,
        "shape",
        &[channels as u64, height as u64, width as u64],
    )?;
    write_array(&mut npz, "kind", frame.kind().as_str().as_bytes())?;
    write_array(
        &mut npz,
        "channel_names",
        frame.channel_names().join("\n").as_bytes(),
    )?;
    let code = match frame.data() {
        EventFrameData::U8(values) => {
            write_array(&mut npz, "data", values)?;
            0u8
        }
        EventFrameData::U16(values) => {
            write_array(&mut npz, "data", values)?;
            1
        }
        EventFrameData::U64(values) => {
            write_array(&mut npz, "data", values)?;
            2
        }
        EventFrameData::F32(values) => {
            write_array(&mut npz, "data", values)?;
            3
        }
    };
    write_array(&mut npz, "dtype", &[code])?;
    finish(npz)
}

/// Reads an [`EventFrame`] written by [`write_npz_frame`].
pub fn read_npz_frame(path: impl AsRef<Path>) -> Result<EventFrame, IoError> {
    let mut archive = open_archive(path)?;
    let shape = read_array::<u64>(&mut archive, "shape")?;
    let &[channels, height, width] = shape.as_slice() else {
        return Err(IoError::Format(
            "frame 'shape' must have exactly three elements".to_owned(),
        ));
    };
    let (channels, height, width) = (channels as usize, height as usize, width as usize);

    let kind = String::from_utf8(read_array::<u8>(&mut archive, "kind")?)
        .map_err(|_| IoError::Format("frame 'kind' is not valid UTF-8".to_owned()))?;
    let kind = RepresentationKind::from_tag(&kind)
        .ok_or_else(|| IoError::Format(format!("unknown representation kind '{kind}'")))?;
    let names = String::from_utf8(read_array::<u8>(&mut archive, "channel_names")?)
        .map_err(|_| IoError::Format("frame 'channel_names' is not valid UTF-8".to_owned()))?;
    let channel_names: Vec<String> = if names.is_empty() {
        Vec::new()
    } else {
        names.split('\n').map(str::to_owned).collect()
    };

    let code = read_scalar::<u8>(&mut archive, "dtype")?
        .ok_or_else(|| IoError::Format("missing frame 'dtype' code".to_owned()))?;
    let data = match code {
        0 => EventFrameData::U8(read_array::<u8>(&mut archive, "data")?),
        1 => EventFrameData::U16(read_array::<u16>(&mut archive, "data")?),
        2 => EventFrameData::U64(read_array::<u64>(&mut archive, "data")?),
        3 => EventFrameData::F32(read_array::<f32>(&mut archive, "data")?),
        other => return Err(IoError::Format(format!("unknown frame dtype code {other}"))),
    };
    if data.len() != channels * height * width {
        return Err(IoError::Format(
            "frame 'data' length does not match its shape".to_owned(),
        ));
    }
    Ok(EventFrame::from_parts(
        data,
        width,
        height,
        kind,
        channel_names,
    ))
}

/// Opens the archive, mapping a non-archive file to a `Format` error (matching the reader's
/// historical behaviour).
fn open_archive(path: impl AsRef<Path>) -> Result<NpzArchive<BufReader<File>>, IoError> {
    NpzArchive::open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::InvalidData {
            IoError::Format(error.to_string())
        } else {
            IoError::Io(error)
        }
    })
}

/// Reads a required array by name into an owned `Vec`.
fn read_array<T: Deserialize>(
    archive: &mut NpzArchive<BufReader<File>>,
    name: &str,
) -> Result<Vec<T>, IoError> {
    archive
        .by_name(name)
        .map_err(IoError::Io)?
        .ok_or_else(|| IoError::Format(format!("missing array '{name}'")))?
        .into_vec::<T>()
        .map_err(IoError::Io)
}

/// Reads the first element of an optional 1-element metadata array, or `None` if absent.
fn read_scalar<T: Deserialize>(
    archive: &mut NpzArchive<BufReader<File>>,
    name: &str,
) -> Result<Option<T>, IoError> {
    match archive.by_name(name).map_err(IoError::Io)? {
        Some(npy) => Ok(npy.into_vec::<T>().map_err(IoError::Io)?.into_iter().next()),
        None => Ok(None),
    }
}

/// Writes one named 1-D array as an uncompressed (STORED) `.npy` entry in the archive.
fn write_array<T, W>(npz: &mut NpzWriter<W>, name: &str, data: &[T]) -> Result<(), IoError>
where
    T: Serialize + AutoSerialize + Clone,
    W: Write + Seek,
{
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    let mut writer = npz
        .array::<T>(name, options)
        .map_err(IoError::Io)?
        .default_dtype()
        .shape(&[data.len() as u64])
        .begin_nd()
        .map_err(IoError::Io)?;
    writer.extend(data.iter().cloned()).map_err(IoError::Io)?;
    writer.finish().map_err(IoError::Io)
}

/// Finalises the zip central directory (otherwise the archive is truncated).
fn finish<W: Write + Seek>(mut npz: NpzWriter<W>) -> Result<(), IoError> {
    npz.zip_writer()
        .finish()
        .map(|_| ())
        .map_err(|error| IoError::Format(format!("zip: {error}")))
}

/// `(max x + 1, max y + 1)`, or `(1, 1)` for an empty stream — the size-inference fallback
/// shared with the other readers.
fn infer_size(xs: &[u16], ys: &[u16]) -> (usize, usize) {
    let width = xs.iter().copied().max().map_or(1, |x| usize::from(x) + 1);
    let height = ys.iter().copied().max().map_or(1, |y| usize::from(y) + 1);
    (width, height)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{read_npz, read_npz_frame, write_npz_frame, write_npz_stream};
    use crate::representation::{Representation, VoxelGrid};
    use crate::{EventStream, EventStreamBuilder};

    fn temp_path(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("eventcv_{tag}_{nanos}.npz"))
    }

    fn sample_stream() -> EventStream {
        let mut builder = EventStreamBuilder::new(20, 15, 0.5); // non-default scale
        for &(x, y, t, p) in &[
            (0u16, 0u16, 10i64, true),
            (19, 14, 2_000, false),
            (5, 7, 3_000, true),
        ] {
            builder.push(x, y, t, p);
        }
        builder.build()
    }

    fn assert_streams_eq(a: &EventStream, b: &EventStream) {
        assert_eq!(a.xs(), b.xs());
        assert_eq!(a.ys(), b.ys());
        assert_eq!(a.ts(), b.ts());
        assert_eq!(a.ps(), b.ps());
        assert_eq!(a.sensor_size(), b.sensor_size());
        assert_eq!(a.timestamp_scale_ms(), b.timestamp_scale_ms());
    }

    #[test]
    fn native_stream_round_trips_exactly() {
        let stream = sample_stream();
        let path = temp_path("npz_stream");
        write_npz_stream(&path, &stream).unwrap();
        assert_streams_eq(&read_npz(&path, None).unwrap(), &stream);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_stream_round_trips() {
        let stream = EventStreamBuilder::new(8, 6, 0.001).build();
        let path = temp_path("npz_empty");
        write_npz_stream(&path, &stream).unwrap();
        let loaded = read_npz(&path, None).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.sensor_size(), (8, 6));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn frame_round_trips_with_metadata() {
        let frame = VoxelGrid::new(3, 30.0).generate(&sample_stream()).unwrap();
        let path = temp_path("npz_frame");
        write_npz_frame(&path, &frame).unwrap();

        let loaded = read_npz_frame(&path).unwrap();
        assert_eq!(loaded.shape(), frame.shape());
        assert_eq!(loaded.kind(), frame.kind());
        assert_eq!(loaded.channel_names(), frame.channel_names());
        assert_eq!(loaded.data(), frame.data());
        std::fs::remove_file(&path).ok();
    }
}
