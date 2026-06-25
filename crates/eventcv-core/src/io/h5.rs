use std::ops::Range;
use std::path::Path;

use hdf5::types::{IntSize, TypeDescriptor};
use hdf5::{Dataset, File, H5Type};

use super::{IoError, LoadOptions, SliceSource, TimeUnit};
use crate::{EventStream, EventStreamBuilder};

/// Events are read in blocks of this many so capping with `max_events` and reading
/// huge files both stay bounded in memory.
const BLOCK: usize = 1_000_000;

/// Reads event datasets `x`, `y`, `t`, `p` from an HDF5 file (looked up under an
/// `events/` group, then at the root). Timestamps are converted to microseconds via
/// `options.time_unit` — Brisbane-Event-VPR stores nanoseconds, so pass `time_unit="ns"`.
pub fn read_hdf5(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    // `file` is held (not read) to keep the dataset handles valid through `read_range`.
    let OpenedFile {
        sensor: (width, height),
        file: _file,
        datasets,
        total,
    } = open_validated(path.as_ref(), options)?;
    let target = options.max_events.map_or(total, |max| max.min(total));
    read_range(&datasets, 0..target, width, height, options.time_unit)
}

/// The open file, its `x`/`y`/`t`/`p` datasets, and the event count — the shared
/// result of opening an HDF5 file. The `file` must outlive use of the datasets.
struct OpenedFile {
    sensor: (usize, usize),
    file: File,
    datasets: [Dataset; 4],
    total: usize,
}

/// Opens the file, locates the `x`/`y`/`t`/`p` datasets, and checks their lengths
/// agree. Shared by the eager [`read_hdf5`] and the lazy [`Hdf5SliceSource`].
fn open_validated(path: &Path, options: &LoadOptions) -> Result<OpenedFile, IoError> {
    let (width, height) = options.sensor_size.ok_or_else(|| {
        IoError::Format("HDF5 files require sensor_size = (width, height)".to_owned())
    })?;
    if !path.exists() {
        return Err(IoError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            path.display().to_string(),
        )));
    }

    let file = File::open(path).map_err(map_hdf5_error)?;
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
        sensor: (width, height),
        file,
        datasets,
        total,
    })
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

/// Opens an HDF5 file for lazy slicing, reading just the first/last timestamps for the
/// time span and sampling the column to confirm it is time-ordered.
pub fn open_hdf5_slice(
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<Hdf5SliceSource, IoError> {
    let OpenedFile {
        sensor: (width, height),
        file,
        datasets,
        total,
    } = open_validated(path.as_ref(), options)?;
    assert_sorted(&datasets[2], total)?;
    let span = if total == 0 {
        (0, 0)
    } else {
        let first = read_integers(&datasets[2], 0..1)?[0];
        let last = read_integers(&datasets[2], total - 1..total)?[0];
        (
            options.time_unit.microseconds_from_int(first),
            options.time_unit.microseconds_from_int(last),
        )
    };
    Ok(Hdf5SliceSource {
        _file: file,
        datasets,
        width,
        height,
        time_unit: options.time_unit,
        total,
        span,
    })
}

/// Events read forward per block when locating a window end; small enough that the
/// overshoot past the end is at most one block, large enough to amortise read calls.
const SCAN_BLOCK: usize = 1 << 16;

impl Hdf5SliceSource {
    /// First index whose timestamp (µs) is `>= target_us` (`total` if none), by binary
    /// search over the on-disk `t` dataset. Each probe decompresses a 100k-element LZF
    /// chunk, so the probe count *is* the latency — only one binary search runs per
    /// slice (the window end is found while reading the events, see [`read_window`]).
    fn lower_bound_time(&self, target_us: i64) -> Result<usize, IoError> {
        let t = &self.datasets[2];
        let mut lo = 0;
        let mut hi = self.total;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let value = self
                .time_unit
                .microseconds_from_int(read_integers(t, mid..mid + 1)?[0]);
            if value < target_us {
                lo = mid + 1;
            } else {
                hi = mid;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{SliceSource, TimeUnit};

    fn options(width: usize, height: usize, time_unit: TimeUnit) -> LoadOptions {
        LoadOptions {
            sensor_size: Some((width, height)),
            time_unit,
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
    fn requires_sensor_size() {
        let error = read_hdf5("missing.h5", &LoadOptions::default()).unwrap_err();
        assert!(matches!(error, IoError::Format(_)));
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
}
