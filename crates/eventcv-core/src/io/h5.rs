use std::ops::Range;
use std::path::Path;

use std::str::FromStr;

use hdf5::types::{IntSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Dataset, File, Group, H5Type};
use ndarray::Array1;

use super::{IoError, LoadOptions, SliceSource, TimeUnit};
use crate::representation::{EventFrame, EventFrameData, RepresentationKind};
use crate::{EventStream, EventStreamBuilder};

/// Events are read in blocks of this many so capping with `max_events` and reading
/// huge files both stay bounded in memory.
const BLOCK: usize = 1_000_000;

/// Reads event datasets `x`, `y`, `t`, `p` from an HDF5 file (looked up under an
/// `events/` group, then at the root). `sensor_size` and `time_unit` are inferred when
/// left `None` — the unit from the timestamp span, the size from the coordinate range.
pub fn read_hdf5(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let opened = open_validated(path.as_ref())?;
    let ((width, height), time_unit) = resolve_params(&opened, options)?;
    let target = options
        .max_events
        .map_or(opened.total, |max| max.min(opened.total));
    read_range(&opened.datasets, 0..target, width, height, time_unit)
}

/// The open file, its `x`/`y`/`t`/`p` datasets, and the event count — the shared
/// result of opening an HDF5 file. The `file` must outlive use of the datasets.
struct OpenedFile {
    file: File,
    datasets: [Dataset; 4],
    total: usize,
}

/// Per-dataset raw chunk cache (bytes / slots) applied when opening event files. The
/// default HDF5 cache (1 MiB) holds under one LZF chunk, so every `slice_time` binary
/// search re-decompresses ~all its probed chunks. Sizing the cache to the binary-search
/// working set lets repeated and adjacent slices (batching, sequential readers) reuse
/// decompressed chunks instead of paying the decompress each time. `nslots` is prime per
/// the HDF5 guidance that it comfortably exceed the number of cached chunks.
const CHUNK_CACHE_BYTES: usize = 64 * 1024 * 1024;
const CHUNK_CACHE_SLOTS: usize = 8009;

/// Opens the file, locates the `x`/`y`/`t`/`p` datasets, and checks their lengths
/// agree. Shared by the eager [`read_hdf5`] and the lazy [`Hdf5SliceSource`].
fn open_validated(path: &Path) -> Result<OpenedFile, IoError> {
    if !path.exists() {
        return Err(IoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            path.display().to_string(),
        )));
    }

    let file = File::with_options()
        .with_fapl(|fapl| fapl.chunk_cache(CHUNK_CACHE_SLOTS, CHUNK_CACHE_BYTES, 0.75))
        .open(path)
        .map_err(map_hdf5_error)?;
    let datasets = open_event_datasets(&file)?;
    let total = datasets[0].shape().first().copied().unwrap_or(0);
    for (axis, dataset) in [
        ("y", &datasets[1]),
        ("t", &datasets[2]),
        ("p", &datasets[3]),
    ] {
        if dataset.shape().first().copied().unwrap_or(0) != total {
            return Err(IoError::Format(format!(
                "event column '{axis}' length does not match 'x'"
            )));
        }
    }
    Ok(OpenedFile {
        file,
        datasets,
        total,
    })
}

/// Resolves `(sensor_size, time_unit)`, inferring whichever the caller left `None`: the
/// time unit from the raw timestamp span (two cheap reads), and the sensor size from the
/// coordinate range — a full `x`/`y` scan (seconds on a multi-GB file), so passing an
/// explicit `sensor_size` skips it.
fn resolve_params(
    opened: &OpenedFile,
    options: &LoadOptions,
) -> Result<((usize, usize), TimeUnit), IoError> {
    let time_unit = match options.time_unit {
        Some(unit) => unit,
        // Honour a saved `timestamp_scale_ms` attribute (written by `write_hdf5_stream`);
        // otherwise fall back to inferring the unit from the span.
        None => match read_scalar_attr::<f64>(&opened.file, "timestamp_scale_ms")?
            .and_then(TimeUnit::from_scale_ms)
        {
            Some(unit) => unit,
            None if opened.total == 0 => TimeUnit::Microseconds,
            None => {
                let first = read_integers(&opened.datasets[2], 0..1)?[0];
                let last = read_integers(&opened.datasets[2], opened.total - 1..opened.total)?[0];
                TimeUnit::infer_from_span(last - first)
            }
        },
    };
    let sensor = match options.sensor_size {
        Some(size) => size,
        // Prefer saved `width`/`height` attributes over a coordinate-range scan.
        None => match (
            read_scalar_attr::<u64>(&opened.file, "width")?,
            read_scalar_attr::<u64>(&opened.file, "height")?,
        ) {
            (Some(width), Some(height)) => (width as usize, height as usize),
            _ => infer_sensor_size(&opened.datasets, opened.total)?,
        },
    };
    Ok((sensor, time_unit))
}

/// Infers the sensor (`max coordinate + 1`) from the first and last [`BLOCK`] events.
/// A full scan of `x`/`y` is slow on a multi-GB LZF file, and an active sensor exercises
/// every row/column within the first/last second, so head+tail matches the true
/// resolution in practice; for files up to `2 * BLOCK` events it covers everything
/// exactly. Pass `sensor_size` if a border pixel is silent at both ends of the recording.
fn infer_sensor_size(datasets: &[Dataset; 4], total: usize) -> Result<(usize, usize), IoError> {
    if total == 0 {
        return Ok((1, 1));
    }
    let [x, y, ..] = datasets;
    let head = 0..BLOCK.min(total);
    let tail = total.saturating_sub(BLOCK)..total;
    let max_x = column_max(x, head.clone())?.max(column_max(x, tail.clone())?);
    let max_y = column_max(y, head)?.max(column_max(y, tail)?);
    Ok((max_x + 1, max_y + 1))
}

/// Largest value in `range` of a coordinate column, reading the dataset's *native*
/// small-int type (no widening to `i64`).
fn column_max(dataset: &Dataset, range: Range<usize>) -> Result<usize, IoError> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(map_hdf5_error)?;
    let max = match descriptor {
        TypeDescriptor::Unsigned(IntSize::U1) => read_block::<u8>(dataset, range)?
            .iter()
            .map(|&v| usize::from(v))
            .max(),
        TypeDescriptor::Unsigned(IntSize::U2) => read_block::<u16>(dataset, range)?
            .iter()
            .map(|&v| usize::from(v))
            .max(),
        TypeDescriptor::Unsigned(IntSize::U4) => read_block::<u32>(dataset, range)?
            .iter()
            .map(|&v| v as usize)
            .max(),
        _ => read_integers(dataset, range)?
            .iter()
            .map(|&v| v.max(0) as usize)
            .max(),
    };
    Ok(max.unwrap_or(0))
}

/// Block-reads the `[x, y, t, p]` datasets over `range`, converting timestamps to
/// microseconds and dropping out-of-bounds events. The single read path for both a
/// whole-file load (`0..total`) and a slice (`[i0, i1)`); blocks bound peak memory.
fn read_range(
    datasets: &[Dataset; 4],
    range: Range<usize>,
    width: usize,
    height: usize,
    time_unit: TimeUnit,
) -> Result<EventStream, IoError> {
    let [x, y, t, p] = datasets;
    let mut builder = EventStreamBuilder::with_capacity(width, height, 0.001, range.len());
    let mut start = range.start;
    while start < range.end {
        let end = (start + BLOCK).min(range.end);
        let xs = read_integers(x, start..end)?;
        let ys = read_integers(y, start..end)?;
        let ts = read_integers(t, start..end)?;
        let ps = read_polarities(p, start..end)?;
        for index in 0..(end - start) {
            builder.push(
                xs[index] as u16,
                ys[index] as u16,
                time_unit.microseconds_from_int(ts[index]),
                ps[index],
            );
        }
        start = end;
    }
    Ok(builder.build())
}

/// A [`SliceSource`] that reads time/count ranges straight from the original HDF5
/// file. Because the `t` column is monotone, `slice_time` binary-searches it on disk
/// (a handful of one-element reads) and only the bracketed events are materialised —
/// no precomputed index, no rewrite, bounded memory on multi-GB files.
pub struct Hdf5SliceSource {
    // The open file keeps the dataset handles valid; it is not read from directly.
    _file: File,
    datasets: [Dataset; 4],
    width: usize,
    height: usize,
    time_unit: TimeUnit,
    total: usize,
    span: (i64, i64),
}

/// Opens an HDF5 file for lazy slicing: confirms the timestamps are time-ordered,
/// resolves the sensor size and time unit (inferring whichever was left unset), and
/// reads the first/last timestamps for the time span.
pub fn open_hdf5_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<Hdf5SliceSource, IoError> {
    let opened = open_validated(path.as_ref())?;
    assert_sorted(&opened.datasets[2], opened.total)?;
    let ((width, height), time_unit) = resolve_params(&opened, options)?;
    let span = if opened.total == 0 {
        (0, 0)
    } else {
        let first = read_integers(&opened.datasets[2], 0..1)?[0];
        let last = read_integers(&opened.datasets[2], opened.total - 1..opened.total)?[0];
        (
            time_unit.microseconds_from_int(first),
            time_unit.microseconds_from_int(last),
        )
    };
    let OpenedFile {
        file,
        datasets,
        total,
    } = opened;
    Ok(Hdf5SliceSource {
        _file: file,
        datasets,
        width,
        height,
        time_unit,
        total,
        span,
    })
}

/// Events read forward per block when locating a window end; small enough that the
/// overshoot past the end is at most one block, large enough to amortise read calls.
const SCAN_BLOCK: usize = 1 << 16;

impl Hdf5SliceSource {
    /// First index whose timestamp (µs) is `>= target_us` (`total` if none). Each probe
    /// decompresses a 100k-element LZF chunk, so the probe count *is* the latency — and it
    /// dominates a slice (the window read past it is comparatively cheap). Rather than
    /// bisecting, this interpolation-searches the monotone `t` column: each probe is seeded
    /// by a linear guess from the bracketing (index, time) pairs, which for the roughly
    /// steady event rate of a recording converges in a handful of chunk reads instead of
    /// ~log2(total). The `hi - 1` clamp guarantees the range shrinks every step, so uneven
    /// event density degrades gracefully rather than looping.
    fn lower_bound_time(&self, target_us: i64) -> Result<usize, IoError> {
        let t = &self.datasets[2];
        if self.total == 0 || target_us <= self.span.0 {
            return Ok(0);
        }
        if target_us > self.span.1 {
            return Ok(self.total);
        }
        // Endpoints of the search come free from the cached time span (t[0], t[total-1]).
        let (mut lo, mut lo_t) = (0usize, self.span.0);
        let (mut hi, mut hi_t) = (self.total - 1, self.span.1);
        while lo < hi {
            let frac = (target_us - lo_t) as f64 / (hi_t - lo_t).max(1) as f64;
            let mid = (lo + (frac.clamp(0.0, 1.0) * (hi - lo) as f64) as usize).clamp(lo, hi - 1);
            let mid_t = self
                .time_unit
                .microseconds_from_int(read_integers(t, mid..mid + 1)?[0]);
            if mid_t < target_us {
                (lo, lo_t) = (mid + 1, mid_t);
            } else {
                (hi, hi_t) = (mid, mid_t);
            }
        }
        Ok(lo)
    }

    /// Reads events from `start` onward, stopping at the first timestamp `>= t1_us`. The
    /// window end is discovered while reading the events we return, so `slice_time` needs
    /// only one binary search (for the start) instead of two.
    fn read_window(&self, start: usize, t1_us: i64) -> Result<EventStream, IoError> {
        let [x, y, t, p] = &self.datasets;
        let mut builder = EventStreamBuilder::with_capacity(self.width, self.height, 0.001, 0);
        let mut s = start;
        'scan: while s < self.total {
            let end = (s + SCAN_BLOCK).min(self.total);
            let xs = read_integers(x, s..end)?;
            let ys = read_integers(y, s..end)?;
            let ts = read_integers(t, s..end)?;
            let ps = read_polarities(p, s..end)?;
            for index in 0..(end - s) {
                let t_us = self.time_unit.microseconds_from_int(ts[index]);
                if t_us >= t1_us {
                    break 'scan;
                }
                builder.push(xs[index] as u16, ys[index] as u16, t_us, ps[index]);
            }
            s = end;
        }
        Ok(builder.build())
    }
}

impl SliceSource for Hdf5SliceSource {
    fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn timestamp_scale_ms(&self) -> f64 {
        0.001
    }

    fn n_events(&self) -> usize {
        self.total
    }

    fn time_span(&self) -> (i64, i64) {
        self.span
    }

    fn slice_index(&self, i0: usize, i1: usize) -> Result<EventStream, IoError> {
        let i0 = i0.min(self.total);
        let i1 = i1.clamp(i0, self.total);
        read_range(
            &self.datasets,
            i0..i1,
            self.width,
            self.height,
            self.time_unit,
        )
    }

    fn slice_time(&self, t0: i64, t1: i64) -> Result<EventStream, IoError> {
        let start = self.lower_bound_time(t0)?;
        self.read_window(start, t1)
    }
}

/// Cheaply confirms the timestamp column is non-decreasing by sampling, so in-place
/// time slicing (which assumes a sorted `t`) cannot silently return wrong events.
fn assert_sorted(t: &Dataset, total: usize) -> Result<(), IoError> {
    if total < 2 {
        return Ok(());
    }
    const SAMPLES: usize = 64;
    let step = (total / SAMPLES).max(1);
    let mut previous = i64::MIN;
    let mut index = 0;
    while index < total {
        let value = read_integers(t, index..index + 1)?[0];
        if value < previous {
            return Err(IoError::Format(
                "HDF5 't' is not sorted; in-place time slicing requires a time-ordered \
                 timestamp column"
                    .to_owned(),
            ));
        }
        previous = value;
        index += step;
    }
    Ok(())
}

fn open_event_datasets(file: &File) -> Result<[Dataset; 4], IoError> {
    for prefix in ["events/", ""] {
        let names = ["x", "y", "t", "p"].map(|axis| format!("{prefix}{axis}"));
        if let (Ok(x), Ok(y), Ok(t), Ok(p)) = (
            file.dataset(&names[0]),
            file.dataset(&names[1]),
            file.dataset(&names[2]),
            file.dataset(&names[3]),
        ) {
            return Ok([x, y, t, p]);
        }
    }
    Err(IoError::Format(
        "could not find event datasets x/y/t/p (looked under 'events/' and the root)".to_owned(),
    ))
}

/// Reads an integer (or unsigned) column as `i64`, dispatching on its on-disk width.
fn read_integers(dataset: &Dataset, range: Range<usize>) -> Result<Vec<i64>, IoError> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(map_hdf5_error)?;
    match descriptor {
        TypeDescriptor::Unsigned(IntSize::U1) => widen::<u8>(dataset, range),
        TypeDescriptor::Unsigned(IntSize::U2) => widen::<u16>(dataset, range),
        TypeDescriptor::Unsigned(IntSize::U4) => widen::<u32>(dataset, range),
        TypeDescriptor::Unsigned(IntSize::U8) => widen::<u64>(dataset, range),
        TypeDescriptor::Integer(IntSize::U1) => widen::<i8>(dataset, range),
        TypeDescriptor::Integer(IntSize::U2) => widen::<i16>(dataset, range),
        TypeDescriptor::Integer(IntSize::U4) => widen::<i32>(dataset, range),
        TypeDescriptor::Integer(IntSize::U8) => widen::<i64>(dataset, range),
        other => Err(IoError::Format(format!(
            "unsupported HDF5 integer column type: {other:?}"
        ))),
    }
}

/// Reads a polarity column as `bool`, accepting HDF5 boolean, enum, or 1-byte int.
fn read_polarities(dataset: &Dataset, range: Range<usize>) -> Result<Vec<bool>, IoError> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(map_hdf5_error)?;
    match descriptor {
        TypeDescriptor::Boolean => read_block::<bool>(dataset, range),
        TypeDescriptor::Enum(_) | TypeDescriptor::Integer(IntSize::U1) => {
            Ok(read_block::<i8>(dataset, range)?
                .into_iter()
                .map(|value| value != 0)
                .collect())
        }
        TypeDescriptor::Unsigned(IntSize::U1) => Ok(read_block::<u8>(dataset, range)?
            .into_iter()
            .map(|value| value != 0)
            .collect()),
        other => Err(IoError::Format(format!(
            "unsupported HDF5 polarity column type: {other:?}"
        ))),
    }
}

fn widen<T: H5Type + Clone + IntoI64>(
    dataset: &Dataset,
    range: Range<usize>,
) -> Result<Vec<i64>, IoError> {
    Ok(read_block::<T>(dataset, range)?
        .into_iter()
        .map(IntoI64::into_i64)
        .collect())
}

fn read_block<T: H5Type + Clone>(
    dataset: &Dataset,
    range: Range<usize>,
) -> Result<Vec<T>, IoError> {
    dataset
        .read_slice_1d::<T, _>(range)
        .map(|array| array.to_vec())
        .map_err(map_hdf5_error)
}

fn map_hdf5_error(error: hdf5::Error) -> IoError {
    IoError::Format(format!("hdf5: {error}"))
}

trait IntoI64 {
    fn into_i64(self) -> i64;
}

macro_rules! impl_into_i64 {
    ($($type:ty),*) => {
        $(impl IntoI64 for $type {
            fn into_i64(self) -> i64 {
                self as i64
            }
        })*
    };
}

impl_into_i64!(u8, u16, u32, u64, i8, i16, i32, i64);

// ---- Writers (symmetric with the readers above) ----

/// Persists a stream as `events/{x,y,t,p}` datasets plus `width`/`height`/`timestamp_scale_ms`
/// root attributes, so the reader reconstructs the sensor size and time scale exactly (the `t`
/// column is stored in the stream's own raw units — microseconds for the usual `0.001` scale).
pub fn write_hdf5_stream(path: impl AsRef<Path>, stream: &EventStream) -> Result<(), IoError> {
    let file = File::create(path.as_ref()).map_err(map_hdf5_error)?;
    let events = file.create_group("events").map_err(map_hdf5_error)?;
    write_dataset(&events, "x", stream.xs())?;
    write_dataset(&events, "y", stream.ys())?;
    write_dataset(&events, "t", stream.ts())?;
    let polarities: Vec<u8> = stream.ps().iter().map(|&p| u8::from(p)).collect();
    write_dataset(&events, "p", &polarities)?;
    let (width, height) = stream.sensor_size();
    write_scalar_attr(&file, "width", width as u64)?;
    write_scalar_attr(&file, "height", height as u64)?;
    write_scalar_attr(&file, "timestamp_scale_ms", stream.timestamp_scale_ms())?;
    Ok(())
}

/// Persists an [`EventFrame`] as a `[C, H, W]` `frame` dataset plus `dtype`/`kind`/
/// `channel_names`/`width`/`height` attributes, recovered by [`read_hdf5_frame`].
pub fn write_hdf5_frame(path: impl AsRef<Path>, frame: &EventFrame) -> Result<(), IoError> {
    let (channels, height, width) = frame.shape();
    let file = File::create(path.as_ref()).map_err(map_hdf5_error)?;
    match frame.data() {
        EventFrameData::U8(values) => write_frame_dataset(&file, channels, height, width, values)?,
        EventFrameData::U16(values) => write_frame_dataset(&file, channels, height, width, values)?,
        EventFrameData::U64(values) => write_frame_dataset(&file, channels, height, width, values)?,
        EventFrameData::F32(values) => write_frame_dataset(&file, channels, height, width, values)?,
    }
    write_string_attr(&file, "dtype", dtype_tag(frame.data()))?;
    write_string_attr(&file, "kind", frame.kind().as_str())?;
    write_string_attr(&file, "channel_names", &frame.channel_names().join("\n"))?;
    write_scalar_attr(&file, "width", width as u64)?;
    write_scalar_attr(&file, "height", height as u64)?;
    Ok(())
}

/// Reads an [`EventFrame`] written by [`write_hdf5_frame`].
pub fn read_hdf5_frame(path: impl AsRef<Path>) -> Result<EventFrame, IoError> {
    let file = File::open(path.as_ref()).map_err(map_hdf5_error)?;
    let dataset = file.dataset("frame").map_err(map_hdf5_error)?;
    let [_, height, width] = <[usize; 3]>::try_from(dataset.shape())
        .map_err(|_| IoError::Format("frame dataset must be 3-D [C, H, W]".to_owned()))?;
    let kind = read_string_attr(&file, "kind")?;
    let kind = RepresentationKind::from_tag(&kind)
        .ok_or_else(|| IoError::Format(format!("unknown representation kind '{kind}'")))?;
    let names = read_string_attr(&file, "channel_names")?;
    let channel_names = if names.is_empty() {
        Vec::new()
    } else {
        names.split('\n').map(str::to_owned).collect()
    };
    let data = match read_string_attr(&file, "dtype")?.as_str() {
        "u8" => EventFrameData::U8(dataset.read_raw::<u8>().map_err(map_hdf5_error)?),
        "u16" => EventFrameData::U16(dataset.read_raw::<u16>().map_err(map_hdf5_error)?),
        "u64" => EventFrameData::U64(dataset.read_raw::<u64>().map_err(map_hdf5_error)?),
        "f32" => EventFrameData::F32(dataset.read_raw::<f32>().map_err(map_hdf5_error)?),
        other => return Err(IoError::Format(format!("unknown frame dtype '{other}'"))),
    };
    Ok(EventFrame::from_parts(
        data,
        width,
        height,
        kind,
        channel_names,
    ))
}

/// Streams [`EventFrame`]s into one extendable `[N, C, H, W]` HDF5 dataset — for computing a
/// representation window-by-window over a whole recording without buffering the result. The
/// dataset dtype and `[C, H, W]` shape are fixed by the first appended frame.
pub struct Hdf5FrameSink {
    file: File,
    dataset: Option<Dataset>,
    code: u8,
    shape: (usize, usize, usize),
    count: usize,
}

impl Hdf5FrameSink {
    /// Creates the output file; the dataset is created lazily on the first [`append`](Self::append).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, IoError> {
        let file = File::create(path.as_ref()).map_err(map_hdf5_error)?;
        Ok(Self {
            file,
            dataset: None,
            code: 0,
            shape: (0, 0, 0),
            count: 0,
        })
    }

    /// Appends one frame as the next slice along axis 0. Every frame must share the first
    /// frame's dtype and `[C, H, W]` shape.
    pub fn append(&mut self, frame: &EventFrame) -> Result<(), IoError> {
        let (channels, height, width) = frame.shape();
        let code = dtype_code(frame.data());
        let first = self.dataset.is_none();
        if !first && ((channels, height, width) != self.shape || code != self.code) {
            return Err(IoError::Format(
                "appended frame shape/dtype differs from the first frame".to_owned(),
            ));
        }
        let n = self.count;
        macro_rules! append_typed {
            ($values:expr, $type:ty) => {{
                if first {
                    // Resizable extents (axis 0 unlimited) auto-chunk in the hdf5 builder.
                    let dataset = self
                        .file
                        .new_dataset::<$type>()
                        .shape((0usize.., channels, height, width))
                        .create("frames")
                        .map_err(map_hdf5_error)?;
                    write_string_attr(&self.file, "dtype", dtype_tag(frame.data()))?;
                    write_string_attr(&self.file, "kind", frame.kind().as_str())?;
                    write_string_attr(
                        &self.file,
                        "channel_names",
                        &frame.channel_names().join("\n"),
                    )?;
                    self.dataset = Some(dataset);
                    self.shape = (channels, height, width);
                    self.code = code;
                }
                let dataset = self.dataset.as_ref().expect("created above");
                dataset
                    .resize((n + 1, channels, height, width))
                    .map_err(map_hdf5_error)?;
                let array =
                    ndarray::Array::from_shape_vec((1, channels, height, width), $values.to_vec())
                        .map_err(|error| IoError::Format(error.to_string()))?;
                dataset
                    .write_slice(&array, ndarray::s![n..n + 1, .., .., ..])
                    .map_err(map_hdf5_error)?;
            }};
        }
        match frame.data() {
            EventFrameData::U8(values) => append_typed!(values, u8),
            EventFrameData::U16(values) => append_typed!(values, u16),
            EventFrameData::U64(values) => append_typed!(values, u64),
            EventFrameData::F32(values) => append_typed!(values, f32),
        }
        self.count += 1;
        Ok(())
    }

    /// Number of frames appended so far (the size of axis 0).
    pub fn n_frames(&self) -> usize {
        self.count
    }

    /// Flushes and closes the file (dropping the sink also closes it).
    pub fn finish(self) -> Result<(), IoError> {
        self.file.flush().map_err(map_hdf5_error)
    }
}

fn write_dataset<T: H5Type + Clone>(group: &Group, name: &str, data: &[T]) -> Result<(), IoError> {
    let array = Array1::from_vec(data.to_vec());
    group
        .new_dataset_builder()
        .with_data(&array)
        .create(name)
        .map(|_| ())
        .map_err(map_hdf5_error)
}

fn write_frame_dataset<T: H5Type + Clone>(
    file: &File,
    channels: usize,
    height: usize,
    width: usize,
    data: &[T],
) -> Result<(), IoError> {
    let array = ndarray::Array::from_shape_vec((channels, height, width), data.to_vec())
        .map_err(|error| IoError::Format(error.to_string()))?;
    file.new_dataset_builder()
        .with_data(&array)
        .create("frame")
        .map(|_| ())
        .map_err(map_hdf5_error)
}

fn write_scalar_attr<T: H5Type>(file: &File, name: &str, value: T) -> Result<(), IoError> {
    let attr = file
        .new_attr::<T>()
        .shape(())
        .create(name)
        .map_err(map_hdf5_error)?;
    attr.write_scalar(&value).map_err(map_hdf5_error)
}

fn write_string_attr(file: &File, name: &str, value: &str) -> Result<(), IoError> {
    let text =
        VarLenUnicode::from_str(value).map_err(|error| IoError::Format(error.to_string()))?;
    let attr = file
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .map_err(map_hdf5_error)?;
    attr.write_scalar(&text).map_err(map_hdf5_error)
}

fn read_scalar_attr<T: H5Type>(file: &File, name: &str) -> Result<Option<T>, IoError> {
    if !file
        .attr_names()
        .map_err(map_hdf5_error)?
        .iter()
        .any(|attr| attr == name)
    {
        return Ok(None);
    }
    let value = file
        .attr(name)
        .map_err(map_hdf5_error)?
        .as_reader()
        .read_scalar::<T>()
        .map_err(map_hdf5_error)?;
    Ok(Some(value))
}

fn read_string_attr(file: &File, name: &str) -> Result<String, IoError> {
    let text: VarLenUnicode = file
        .attr(name)
        .map_err(map_hdf5_error)?
        .as_reader()
        .read_scalar()
        .map_err(map_hdf5_error)?;
    Ok(text.as_str().to_owned())
}

fn dtype_tag(data: &EventFrameData) -> &'static str {
    match data {
        EventFrameData::U8(_) => "u8",
        EventFrameData::U16(_) => "u16",
        EventFrameData::U64(_) => "u64",
        EventFrameData::F32(_) => "f32",
    }
}

fn dtype_code(data: &EventFrameData) -> u8 {
    match data {
        EventFrameData::U8(_) => 0,
        EventFrameData::U16(_) => 1,
        EventFrameData::U64(_) => 2,
        EventFrameData::F32(_) => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceSource, TimeUnit};

    fn options(width: usize, height: usize, time_unit: TimeUnit) -> LoadOptions {
        LoadOptions {
            sensor_size: Some((width, height)),
            time_unit: Some(time_unit),
            ..LoadOptions::default()
        }
    }

    #[test]
    fn reads_grouped_event_datasets_and_drops_out_of_bounds() {
        let dir = std::env::temp_dir().join(format!("eventcv-h5-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            let group = file.create_group("events").unwrap();
            group
                .new_dataset_builder()
                .with_data(&[1u16, 3, 0, 4][..])
                .create("x")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[2u16, 0, 1, 0][..])
                .create("y")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[1000u64, 2000, 3000, 4000][..])
                .create("t")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[true, false, true, false][..])
                .create("p")
                .unwrap();
        }

        let stream = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();

        assert_eq!(stream.len(), 3); // (4, 0) dropped: x == width
        assert_eq!(stream.xs(), &[1, 3, 0]);
        assert_eq!(stream.ys(), &[2, 0, 1]);
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        assert_eq!(stream.ps(), &[true, false, true]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nanosecond_timestamps_convert_to_microseconds() {
        let dir = std::env::temp_dir().join(format!("eventcv-h5ns-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ns.h5");
        {
            let file = File::create(&path).unwrap();
            file.new_dataset_builder()
                .with_data(&[0u16, 1][..])
                .create("x")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[0u16, 1][..])
                .create("y")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[1_000_000u64, 2_500_000][..])
                .create("t")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[true, false][..])
                .create("p")
                .unwrap();
        }

        let stream = read_hdf5(&path, &options(8, 8, TimeUnit::Nanoseconds)).unwrap();

        assert_eq!(stream.ts(), &[1000, 2500]); // ns -> us
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_file_is_reported() {
        let error = read_hdf5("missing.h5", &LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Io(_)));
    }

    #[test]
    fn infers_sensor_size_and_time_unit_when_unset() {
        // Realistic magnitudes: x,y up to 5 -> 6x6; t span 5e6 (5 s) -> microseconds.
        let dir = std::env::temp_dir().join(format!("eventcv-h5infer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            file.new_dataset_builder()
                .with_data(&[0u64, 1, 2, 3, 4, 5][..])
                .create("x")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[0u64, 1, 2, 3, 4, 5][..])
                .create("y")
                .unwrap();
            file.new_dataset_builder()
                .with_data(
                    &[
                        1_000_000u64,
                        2_000_000,
                        3_000_000,
                        4_000_000,
                        5_000_000,
                        6_000_000,
                    ][..],
                )
                .create("t")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[true, false, true, false, true, false][..])
                .create("p")
                .unwrap();
        }

        let stream = read_hdf5(&path, &LoadOptions::default()).unwrap();

        assert_eq!(stream.sensor_size(), (6, 6)); // max coord 5 -> 6, nothing dropped
        assert_eq!(stream.len(), 6);
        // span 5e6 -> microseconds (µs leaves the values unchanged)
        assert_eq!(
            stream.ts(),
            &[1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Writes six in-bounds events with strictly increasing microsecond timestamps
    /// `1000..=6000` to root-level `x`/`y`/`t`/`p`, returning the file path.
    fn write_sorted(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("eventcv-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.h5");
        let file = File::create(&path).unwrap();
        for (name, data) in [
            ("x", vec![0u64, 1, 2, 3, 4, 5]),
            ("y", vec![0u64, 1, 2, 3, 4, 5]),
            ("t", vec![1000u64, 2000, 3000, 4000, 5000, 6000]),
        ] {
            file.new_dataset_builder()
                .with_data(&data[..])
                .create(name)
                .unwrap();
        }
        file.new_dataset_builder()
            .with_data(&[true, false, true, false, true, false][..])
            .create("p")
            .unwrap();
        path
    }

    #[test]
    fn slice_source_reports_span_and_count() {
        let path = write_sorted("h5span");
        let source = open_hdf5_slice(&path, &options(8, 8, TimeUnit::Microseconds)).unwrap();

        assert_eq!(source.n_events(), 6);
        assert_eq!(source.time_span(), (1000, 6000));
        assert_eq!(source.sensor_size(), (8, 8));
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn slice_time_is_half_open_and_binary_searched() {
        let path = write_sorted("h5time");
        let source = open_hdf5_slice(&path, &options(8, 8, TimeUnit::Microseconds)).unwrap();

        let slice = source.slice_time(2000, 5000).unwrap();
        assert_eq!(slice.ts(), &[2000, 3000, 4000]); // 5000 excluded (half-open)
        assert_eq!(slice.xs(), &[1, 2, 3]);

        // A window past the end is empty; one spanning everything keeps all six.
        assert!(source.slice_time(7000, 8000).unwrap().is_empty());
        assert_eq!(source.slice_time(0, 10_000).unwrap().len(), 6);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn slice_index_clamps_out_of_range() {
        let path = write_sorted("h5index");
        let source = open_hdf5_slice(&path, &options(8, 8, TimeUnit::Microseconds)).unwrap();

        assert_eq!(source.slice_index(1, 4).unwrap().ts(), &[2000, 3000, 4000]);
        assert_eq!(source.slice_index(4, 100).unwrap().ts(), &[5000, 6000]); // hi clamped
        assert!(source.slice_index(10, 20).unwrap().is_empty()); // wholly past the end
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn slice_matches_full_load() {
        let path = write_sorted("h5parity");
        let opts = options(8, 8, TimeUnit::Microseconds);
        let source = open_hdf5_slice(&path, &opts).unwrap();
        let full = read_hdf5(&path, &opts).unwrap();

        let whole = source.slice_index(0, source.n_events()).unwrap();
        assert_eq!(whole.xs(), full.xs());
        assert_eq!(whole.ts(), full.ts());
        assert_eq!(whole.ps(), full.ps());
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn slice_time_handles_nonuniform_timestamps() {
        // Geometric gaps: the window end (found by reading forward) must land exactly.
        let dir = std::env::temp_dir().join(format!("eventcv-h5nu-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            file.new_dataset_builder()
                .with_data(&[0u64, 1, 2, 3, 4, 5, 6, 7][..])
                .create("x")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[0u64; 8][..])
                .create("y")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[1u64, 2, 4, 8, 16, 32, 64, 1000][..])
                .create("t")
                .unwrap();
            file.new_dataset_builder()
                .with_data(&[true; 8][..])
                .create("p")
                .unwrap();
        }
        let source = open_hdf5_slice(&path, &options(8, 8, TimeUnit::Microseconds)).unwrap();

        assert_eq!(source.slice_time(4, 33).unwrap().ts(), &[4, 8, 16, 32]);
        assert_eq!(source.slice_time(0, 2).unwrap().ts(), &[1]);
        assert_eq!(source.slice_time(64, 2000).unwrap().ts(), &[64, 1000]);
        assert!(source.slice_time(65, 1000).unwrap().is_empty()); // 64 < 65, 1000 excluded
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_unsorted_timestamps() {
        let dir = std::env::temp_dir().join(format!("eventcv-h5unsorted-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            for (name, data) in [
                ("x", vec![0u64, 1, 2]),
                ("y", vec![0u64, 1, 2]),
                ("t", vec![10u64, 20, 5]), // not monotone
            ] {
                file.new_dataset_builder()
                    .with_data(&data[..])
                    .create(name)
                    .unwrap();
            }
            file.new_dataset_builder()
                .with_data(&[true, false, true][..])
                .create("p")
                .unwrap();
        }

        // `Hdf5SliceSource` is not `Debug`, so match rather than `unwrap_err`.
        match open_hdf5_slice(&path, &options(8, 8, TimeUnit::Microseconds)) {
            Err(IoError::Format(message)) => assert!(message.contains("not sorted")),
            Err(other) => panic!("expected a format error, got {other:?}"),
            Ok(_) => panic!("expected unsorted timestamps to be rejected"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("eventcv-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn stream_round_trips_with_metadata_attrs() {
        let mut builder = EventStreamBuilder::new(20, 15, 0.001);
        for &(x, y, t, p) in &[(0u16, 0u16, 7i64, true), (19, 14, 2_500_000, false)] {
            builder.push(x, y, t, p);
        }
        let stream = builder.build();

        let dir = temp_dir("h5streamrt");
        let path = dir.join("stream.h5");
        write_hdf5_stream(&path, &stream).unwrap();

        // No options: the size and unit come from the saved attributes, not inference.
        let loaded = read_hdf5(&path, &LoadOptions::default()).unwrap();
        assert_eq!(loaded.sensor_size(), (20, 15));
        assert_eq!(loaded.xs(), stream.xs());
        assert_eq!(loaded.ys(), stream.ys());
        assert_eq!(loaded.ts(), stream.ts());
        assert_eq!(loaded.ps(), stream.ps());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frame_round_trips_with_metadata() {
        use crate::representation::{Representation, VoxelGrid};
        let mut builder = EventStreamBuilder::new(8, 6, 0.001);
        builder.push(1, 2, 100, true);
        builder.push(7, 5, 5_000, false);
        let frame = VoxelGrid::new(3, 30.0).generate(&builder.build()).unwrap();

        let dir = temp_dir("h5framert");
        let path = dir.join("frame.h5");
        write_hdf5_frame(&path, &frame).unwrap();

        let loaded = read_hdf5_frame(&path).unwrap();
        assert_eq!(loaded.shape(), frame.shape());
        assert_eq!(loaded.kind(), frame.kind());
        assert_eq!(loaded.channel_names(), frame.channel_names());
        assert_eq!(loaded.data(), frame.data());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frame_sink_appends_a_stack() {
        use crate::representation::{Binary, Representation};
        let mut builder = EventStreamBuilder::new(4, 3, 0.001);
        builder.push(1, 1, 0, true);
        builder.push(2, 2, 10, false);
        let frame = Binary.generate(&builder.build()).unwrap();
        let (channels, height, width) = frame.shape();

        let dir = temp_dir("h5sink");
        let path = dir.join("stack.h5");
        let mut sink = Hdf5FrameSink::open(&path).unwrap();
        sink.append(&frame).unwrap();
        sink.append(&frame).unwrap();
        sink.append(&frame).unwrap();
        assert_eq!(sink.n_frames(), 3);
        sink.finish().unwrap();

        // The stack is a [N, C, H, W] dataset readable straight back.
        let file = File::open(&path).unwrap();
        let dataset = file.dataset("frames").unwrap();
        assert_eq!(dataset.shape(), vec![3, channels, height, width]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
