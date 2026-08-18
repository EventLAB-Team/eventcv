use std::ops::Range;
use std::path::Path;

use std::str::FromStr;

use hdf5::datatype::{ByteOrder, Datatype};
use hdf5::types::{CompoundType, FloatSize, IntSize, TypeDescriptor, VarLenUnicode};
use hdf5::{Dataset, File, Group, H5Type};
use hdf5_sys::h5::hsize_t;
use hdf5_sys::h5d::H5Dread;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::H5S_seloper_t::H5S_SELECT_SET;
use hdf5_sys::h5s::{H5Sclose, H5Screate_simple, H5Sselect_hyperslab};
use ndarray::Array1;

use super::{
    role_of, role_rank, Compression, EventKeys, IoError, LoadOptions, SliceSource, TimeUnit, P, T,
    X, Y,
};
use crate::representation::{EventFrame, EventFrameData, RepresentationKind};
use crate::{EventStream, EventStreamBuilder};

/// Events are read in blocks of this many so capping with `max_events` and reading
/// huge files both stay bounded in memory.
const BLOCK: usize = 1_000_000;

/// Reads event datasets `x`, `y`, `t`, `p` from an HDF5 file (looked up under an
/// `events/` group, then at the root). `sensor_size` and `time_unit` are inferred when
/// left `None` — the unit from the timestamp span, the size from the coordinate range.
pub fn read_hdf5(path: impl AsRef<Path>, options: &LoadOptions) -> Result<EventStream, IoError> {
    let opened = open_validated(path.as_ref(), options.keys.as_ref())?;
    let ((width, height), time_unit) = resolve_params(&opened, options)?;
    let target = options
        .max_events
        .map_or(opened.total, |max| max.min(opened.total));
    read_range(&opened.columns, 0..target, width, height, time_unit)
}

/// The open file, its resolved x/y/t/p [`EventColumns`], and the event count — the shared
/// result of opening an HDF5 file. The `file` must outlive use of the columns.
struct OpenedFile {
    file: File,
    columns: EventColumns,
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

/// Opens the file and resolves its x/y/t/p [`EventColumns`] — auto-detected, or named
/// explicitly via `keys`. Shared by the eager [`read_hdf5`] and the lazy [`Hdf5SliceSource`].
fn open_validated(path: &Path, keys: Option<&EventKeys>) -> Result<OpenedFile, IoError> {
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
    let columns = resolve_columns(&file, keys)?;
    let total = columns.len();
    Ok(OpenedFile {
        file,
        columns,
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
                let first = opened.columns.read_ints(T, 0..1)?[0];
                let last = opened
                    .columns
                    .read_ints(T, opened.total - 1..opened.total)?[0];
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
            _ => infer_sensor_size(&opened.columns, opened.total)?,
        },
    };
    Ok((sensor, time_unit))
}

/// Infers the sensor (`max coordinate + 1`) from the first and last [`BLOCK`] events.
/// A full scan of `x`/`y` is slow on a multi-GB LZF file, and an active sensor exercises
/// every row/column within the first/last second, so head+tail matches the true
/// resolution in practice; for files up to `2 * BLOCK` events it covers everything
/// exactly. Pass `sensor_size` if a border pixel is silent at both ends of the recording.
fn infer_sensor_size(columns: &EventColumns, total: usize) -> Result<(usize, usize), IoError> {
    if total == 0 {
        return Ok((1, 1));
    }
    let head = 0..BLOCK.min(total);
    let tail = total.saturating_sub(BLOCK)..total;
    let max_x = columns
        .column_max(X, head.clone())?
        .max(columns.column_max(X, tail.clone())?);
    let max_y = columns
        .column_max(Y, head)?
        .max(columns.column_max(Y, tail)?);
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
    columns: &EventColumns,
    range: Range<usize>,
    width: usize,
    height: usize,
    time_unit: TimeUnit,
) -> Result<EventStream, IoError> {
    let mut builder = EventStreamBuilder::with_capacity(width, height, 0.001, range.len());
    let mut start = range.start;
    while start < range.end {
        let end = (start + BLOCK).min(range.end);
        let xs = columns.read_ints(X, start..end)?;
        let ys = columns.read_ints(Y, start..end)?;
        let ts = columns.read_ints(T, start..end)?;
        let ps = columns.read_polarity(start..end)?;
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
    columns: EventColumns,
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
    let opened = open_validated(path.as_ref(), options.keys.as_ref())?;
    assert_sorted(&opened.columns, opened.total)?;
    let ((width, height), time_unit) = resolve_params(&opened, options)?;
    let span = if opened.total == 0 {
        (0, 0)
    } else {
        let first = opened.columns.read_ints(T, 0..1)?[0];
        let last = opened
            .columns
            .read_ints(T, opened.total - 1..opened.total)?[0];
        (
            time_unit.microseconds_from_int(first),
            time_unit.microseconds_from_int(last),
        )
    };
    let OpenedFile {
        file,
        columns,
        total,
    } = opened;
    Ok(Hdf5SliceSource {
        _file: file,
        columns,
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
                .microseconds_from_int(self.columns.read_ints(T, mid..mid + 1)?[0]);
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
        let mut builder = EventStreamBuilder::with_capacity(self.width, self.height, 0.001, 0);
        let mut s = start;
        'scan: while s < self.total {
            let end = (s + SCAN_BLOCK).min(self.total);
            let xs = self.columns.read_ints(X, s..end)?;
            let ys = self.columns.read_ints(Y, s..end)?;
            let ts = self.columns.read_ints(T, s..end)?;
            let ps = self.columns.read_polarity(s..end)?;
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
            &self.columns,
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

    /// Tallies per-pixel counts reading only the `x`/`y` columns — skipping the `t` column (by
    /// far the largest on disk) and `p`, plus timestamp conversion and stream construction, which
    /// are the bulk of a slice's cost. Above [`SCAN_BUDGET`] events it further reads only
    /// evenly-spaced segments rather than every event: the hot-pixel mask is an outlier test, and
    /// stuck pixels fire throughout the recording, so they dominate any representative sample while
    /// the threshold (`mean + n·std`) stays ~scale-invariant. This keeps a multi-GB scan bounded.
    fn pixel_counts(&self) -> Result<Vec<u64>, IoError> {
        let mut counts = vec![0u64; self.width * self.height];
        if self.total <= SCAN_BUDGET {
            let mut start = 0;
            while start < self.total {
                let end = (start + BLOCK).min(self.total);
                add_xy_counts(
                    &self.columns,
                    start..end,
                    self.width,
                    self.height,
                    &mut counts,
                )?;
                start = end;
            }
        } else {
            // Read SCAN_SEGMENT-event windows spread evenly across the file (~SCAN_BUDGET total).
            let segments = SCAN_BUDGET / SCAN_SEGMENT;
            let stride = self.total / segments;
            for i in 0..segments {
                let start = i * stride;
                let end = (start + SCAN_SEGMENT).min(self.total);
                add_xy_counts(
                    &self.columns,
                    start..end,
                    self.width,
                    self.height,
                    &mut counts,
                )?;
            }
        }
        Ok(counts)
    }
}

/// Target events for the hot-pixel pre-scan. A recording larger than this is *sampled* as
/// [`SCAN_SEGMENT`]-event windows spread evenly across it rather than read in full — enough to
/// pin the count distribution's shape (all a stuck-pixel outlier test needs) while keeping the
/// scan of a multi-GB file well under a second.
const SCAN_BUDGET: usize = 32_000_000;
/// Events per evenly-spaced sample window. Kept small (≈ one on-disk chunk) so the budget buys
/// *many* windows: fine temporal coverage is what catches a hot pixel whose bursts of activity
/// would fall between coarser, wider-spaced samples.
const SCAN_SEGMENT: usize = 100_000;

/// Adds the `x`/`y` columns over `range` into `counts`, dropping out-of-bounds events (matching
/// [`read_range`]). Reads coordinates in their native small-int type — no widening to `i64`.
fn add_xy_counts(
    columns: &EventColumns,
    range: Range<usize>,
    width: usize,
    height: usize,
    counts: &mut [u64],
) -> Result<(), IoError> {
    let xs = columns.read_coords(X, range.clone())?;
    let ys = columns.read_coords(Y, range)?;
    for (&xi, &yi) in xs.iter().zip(&ys) {
        if xi < width && yi < height {
            counts[yi * width + xi] += 1;
        }
    }
    Ok(())
}

/// Reads a coordinate column over `range` as `usize`, dispatching on its on-disk width like
/// [`column_max`] so a `u16` file is not widened through `i64`.
fn read_coords(dataset: &Dataset, range: Range<usize>) -> Result<Vec<usize>, IoError> {
    let descriptor = dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(map_hdf5_error)?;
    Ok(match descriptor {
        TypeDescriptor::Unsigned(IntSize::U1) => read_block::<u8>(dataset, range)?
            .iter()
            .map(|&v| usize::from(v))
            .collect(),
        TypeDescriptor::Unsigned(IntSize::U2) => read_block::<u16>(dataset, range)?
            .iter()
            .map(|&v| usize::from(v))
            .collect(),
        TypeDescriptor::Unsigned(IntSize::U4) => read_block::<u32>(dataset, range)?
            .iter()
            .map(|&v| v as usize)
            .collect(),
        // Wider or signed columns: reuse the general reader, clamping negatives out of bounds.
        _ => read_integers(dataset, range)?
            .iter()
            .map(|&v| if v < 0 { usize::MAX } else { v as usize })
            .collect(),
    })
}

/// Cheaply confirms the timestamp column is non-decreasing by sampling, so in-place
/// time slicing (which assumes a sorted `t`) cannot silently return wrong events.
fn assert_sorted(columns: &EventColumns, total: usize) -> Result<(), IoError> {
    if total < 2 {
        return Ok(());
    }
    const SAMPLES: usize = 64;
    let step = (total / SAMPLES).max(1);
    let mut previous = i64::MIN;
    let mut index = 0;
    while index < total {
        let value = columns.read_ints(T, index..index + 1)?[0];
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

/// The four event columns, however the file stores them: as four separate 1-D datasets
/// (the common layout) or as fields of one compound dataset. Presents a uniform
/// read/inference interface so the rest of the reader is agnostic to the on-disk shape.
enum EventColumns {
    /// `[x, y, t, p]` as four separate datasets.
    Separate([Dataset; 4]),
    /// One compound dataset whose `[x, y, t, p]` fields are read by hyperslab.
    Compound(CompoundColumns),
}

impl EventColumns {
    /// Number of events (the first dimension shared by all four columns).
    fn len(&self) -> usize {
        match self {
            Self::Separate(datasets) => datasets[X].shape().first().copied().unwrap_or(0),
            Self::Compound(compound) => compound.total,
        }
    }

    /// Reads column `col` (`X`/`Y`/`T`/`P`) over `range` as `i64` — the shared path for
    /// timestamps and coordinates.
    fn read_ints(&self, col: usize, range: Range<usize>) -> Result<Vec<i64>, IoError> {
        match self {
            Self::Separate(datasets) => read_integers(&datasets[col], range),
            Self::Compound(compound) => compound.read_ints(col, range),
        }
    }

    /// Reads coordinate column `col` (`X`/`Y`) over `range` as `usize`, negatives mapped to
    /// `usize::MAX` so they drop out of bounds (matching the separate-dataset reader).
    fn read_coords(&self, col: usize, range: Range<usize>) -> Result<Vec<usize>, IoError> {
        match self {
            Self::Separate(datasets) => read_coords(&datasets[col], range),
            Self::Compound(compound) => Ok(compound
                .read_ints(col, range)?
                .into_iter()
                .map(|v| if v < 0 { usize::MAX } else { v as usize })
                .collect()),
        }
    }

    /// Reads the polarity column over `range` as `bool` (non-zero is positive).
    fn read_polarity(&self, range: Range<usize>) -> Result<Vec<bool>, IoError> {
        match self {
            Self::Separate(datasets) => read_polarities(&datasets[P], range),
            Self::Compound(compound) => Ok(compound
                .read_ints(P, range)?
                .into_iter()
                .map(|v| v != 0)
                .collect()),
        }
    }

    /// Largest value in `range` of coordinate column `col`, negatives clamped to 0.
    fn column_max(&self, col: usize, range: Range<usize>) -> Result<usize, IoError> {
        match self {
            Self::Separate(datasets) => column_max(&datasets[col], range),
            Self::Compound(compound) => Ok(compound
                .read_ints(col, range)?
                .into_iter()
                .map(|v| v.max(0) as usize)
                .max()
                .unwrap_or(0)),
        }
    }
}

/// Locates the x/y/t/p columns. With `keys` the caller names them explicitly; otherwise
/// they are auto-detected across arbitrary dataset names (synonyms) and nesting, and a
/// single compound dataset is recognised too. Replaces the old fixed `x`/`y`/`t`/`p` lookup.
fn resolve_columns(file: &File, keys: Option<&EventKeys>) -> Result<EventColumns, IoError> {
    if let Some(keys) = keys {
        return resolve_named_columns(file, keys);
    }
    let mut datasets = Vec::new();
    collect_datasets(file, &mut datasets)?;

    // Prefer four separate arrays (the common, FFI-free path); fall back to a compound dataset.
    if let Some(columns) = detect_separate(&datasets)? {
        return Ok(columns);
    }
    if let Some(columns) = detect_compound(&datasets)? {
        return Ok(columns);
    }
    Err(IoError::Format(unresolved_message(&datasets)))
}

/// Recursively collects every dataset in the file (depth-first from the root group).
fn collect_datasets(group: &Group, out: &mut Vec<Dataset>) -> Result<(), IoError> {
    for dataset in group.datasets().map_err(map_hdf5_error)? {
        out.push(dataset);
    }
    for subgroup in group.groups().map_err(map_hdf5_error)? {
        collect_datasets(&subgroup, out)?;
    }
    Ok(())
}

/// The base name (final path component) and parent path of a dataset, e.g.
/// `/events/x_coordinates` → (`x_coordinates`, `/events`).
fn split_path(name: &str) -> (&str, &str) {
    match name.rsplit_once('/') {
        Some((parent, base)) => (base, parent),
        None => (name, ""),
    }
}

/// Finds four separate 1-D datasets whose names match the x/y/t/p synonyms and that share a
/// parent group and length. Among qualifying groups the one with the most events wins (so a
/// full stream beats a decimated preview); ties break toward the shallower, then earlier path.
fn detect_separate(datasets: &[Dataset]) -> Result<Option<EventColumns>, IoError> {
    use std::collections::HashMap;
    // parent path -> [best (rank, dataset index) per role]
    let mut groups: HashMap<String, [Option<(usize, usize)>; 4]> = HashMap::new();
    for (index, dataset) in datasets.iter().enumerate() {
        // Skip compound datasets and anything not 1-D here — they are not separate columns.
        if dataset.shape().len() != 1 || is_compound(dataset) {
            continue;
        }
        let name = dataset.name();
        let (base, parent) = split_path(&name);
        if let Some(role) = role_of(base) {
            let rank = role_rank(role, base).unwrap_or(usize::MAX);
            let slot = &mut groups.entry(parent.to_owned()).or_default()[role];
            if slot.is_none_or(|(best, _)| rank < best) {
                *slot = Some((rank, index));
            }
        }
    }

    let mut best: Option<(usize, usize, [usize; 4])> = None; // (len, shallowness, indices)
    for (parent, roles) in &groups {
        let Some(indices) = all_roles(roles) else {
            continue;
        };
        let len = datasets[indices[0]].shape().first().copied().unwrap_or(0);
        if indices[1..]
            .iter()
            .any(|&i| datasets[i].shape().first().copied().unwrap_or(0) != len)
        {
            continue; // columns of a group must agree in length to be one event stream
        }
        let shallowness = usize::MAX - parent.matches('/').count();
        if best
            .is_none_or(|(best_len, best_shallow, _)| (len, shallowness) > (best_len, best_shallow))
        {
            best = Some((len, shallowness, indices));
        }
    }

    Ok(best.map(|(_, _, indices)| EventColumns::Separate(indices.map(|i| datasets[i].clone()))))
}

/// Returns the four dataset indices if every role slot is filled, else `None`.
fn all_roles(roles: &[Option<(usize, usize)>; 4]) -> Option<[usize; 4]> {
    Some([roles[X]?.1, roles[Y]?.1, roles[T]?.1, roles[P]?.1])
}

/// Finds a compound dataset whose member names cover x/y/t/p and reads its fields by
/// hyperslab (see [`CompoundColumns`]).
fn detect_compound(datasets: &[Dataset]) -> Result<Option<EventColumns>, IoError> {
    for dataset in datasets {
        if let Some(columns) = CompoundColumns::try_new(dataset)? {
            return Ok(Some(EventColumns::Compound(columns)));
        }
    }
    Ok(None)
}

/// Whether a dataset stores a compound (structured) type.
fn is_compound(dataset: &Dataset) -> bool {
    matches!(
        dataset.dtype().and_then(|dtype| dtype.to_descriptor()),
        Ok(TypeDescriptor::Compound(_))
    )
}

/// The "couldn't identify the columns" error, listing what the file actually contains so the
/// user can see what to rename or name via `keys`.
fn unresolved_message(datasets: &[Dataset]) -> String {
    let mut listing = String::new();
    for dataset in datasets {
        let dtype = dataset
            .dtype()
            .and_then(|dtype| dtype.to_descriptor())
            .map(|descriptor| format!("{descriptor:?}"))
            .unwrap_or_else(|_| "?".to_owned());
        listing.push_str(&format!(
            "\n  {} ({dtype}, shape {:?})",
            dataset.name(),
            dataset.shape()
        ));
    }
    if listing.is_empty() {
        listing.push_str(" (none)");
    }
    format!(
        "could not identify the x/y/t/p event columns. Datasets present:{listing}\nPass \
         keys={{\"x\": …, \"y\": …, \"t\": …, \"p\": …}} to name them explicitly (an HDF5 value \
         is a dataset path, or `dataset/field` for a compound dataset)."
    )
}

/// Resolves user-named columns: either four separate dataset paths, or four fields of one
/// compound dataset (`dataset/field`). Mixing the two is rejected.
fn resolve_named_columns(file: &File, keys: &EventKeys) -> Result<EventColumns, IoError> {
    let paths = [&keys.x, &keys.y, &keys.t, &keys.p];
    // Try four separate datasets first.
    let separate: Vec<Option<Dataset>> = paths.iter().map(|path| file.dataset(path).ok()).collect();
    if separate.iter().all(Option::is_some) {
        let datasets = [
            separate[X].clone().unwrap(),
            separate[Y].clone().unwrap(),
            separate[T].clone().unwrap(),
            separate[P].clone().unwrap(),
        ];
        return Ok(EventColumns::Separate(datasets));
    }

    // Otherwise interpret each as `dataset/field` into a single compound dataset.
    let mut fields = [""; 4];
    let mut dataset_path = None;
    for (role, path) in paths.iter().enumerate() {
        let (field, parent) = split_path(path);
        fields[role] = field;
        match dataset_path {
            None => dataset_path = Some(parent.to_owned()),
            Some(ref existing) if existing == parent => {}
            Some(_) => {
                return Err(IoError::Format(
                    "keys must be four separate dataset paths or four fields of the same \
                     compound dataset"
                        .to_owned(),
                ))
            }
        }
    }
    let dataset_path = dataset_path.unwrap_or_default();
    let dataset = file
        .dataset(&dataset_path)
        .map_err(|_| IoError::Format(format!("no dataset '{dataset_path}' for the named keys")))?;
    let named = fields.map(str::to_owned);
    CompoundColumns::with_fields(&dataset, &named)?
        .map(EventColumns::Compound)
        .ok_or_else(|| {
            IoError::Format(format!(
                "compound dataset '{dataset_path}' has no fields {named:?}"
            ))
        })
}

/// One field of a compound event dataset: where it sits in each packed record and how to
/// decode it.
#[derive(Clone)]
struct FieldDecode {
    offset: usize,
    descriptor: TypeDescriptor,
}

/// Reads x/y/t/p from a single compound (structured) dataset. hdf5-metno's safe API can't
/// decode an arbitrary compound generically, so a slice reads the packed records for its range
/// with a raw `H5Dread` (memory type = the file datatype, so no conversion — records land in the
/// file's byte order at the offsets [`to_descriptor`](Datatype::to_descriptor) reports) and
/// decodes each field in place.
struct CompoundColumns {
    dataset: Dataset,
    // Held so its id stays valid as the `H5Dread` memory datatype for the lifetime of the reads.
    dtype: Datatype,
    record_size: usize,
    order: ByteOrder,
    fields: [FieldDecode; 4],
    total: usize,
}

impl CompoundColumns {
    /// Auto-detects x/y/t/p among a compound dataset's member names (synonym match); `Ok(None)`
    /// if the dataset isn't compound or doesn't carry all four.
    fn try_new(dataset: &Dataset) -> Result<Option<Self>, IoError> {
        let Some(compound) = compound_of(dataset) else {
            return Ok(None);
        };
        let mut chosen: [Option<(usize, FieldDecode)>; 4] = [None, None, None, None];
        for field in &compound.fields {
            if let Some(role) = role_of(&field.name) {
                let rank = role_rank(role, &field.name).unwrap_or(usize::MAX);
                if chosen[role].as_ref().is_none_or(|(best, _)| rank < *best) {
                    chosen[role] = Some((
                        rank,
                        FieldDecode {
                            offset: field.offset,
                            descriptor: field.ty.clone(),
                        },
                    ));
                }
            }
        }
        Self::from_chosen(dataset, chosen)
    }

    /// Resolves explicitly named compound fields (the `keys` path).
    fn with_fields(dataset: &Dataset, names: &[String; 4]) -> Result<Option<Self>, IoError> {
        let Some(compound) = compound_of(dataset) else {
            return Ok(None);
        };
        let mut chosen: [Option<(usize, FieldDecode)>; 4] = [None, None, None, None];
        for (role, wanted) in names.iter().enumerate() {
            if let Some(field) = compound.fields.iter().find(|field| field.name == *wanted) {
                chosen[role] = Some((
                    0,
                    FieldDecode {
                        offset: field.offset,
                        descriptor: field.ty.clone(),
                    },
                ));
            }
        }
        Self::from_chosen(dataset, chosen)
    }

    fn from_chosen(
        dataset: &Dataset,
        chosen: [Option<(usize, FieldDecode)>; 4],
    ) -> Result<Option<Self>, IoError> {
        let [x, y, t, p] = chosen;
        let (Some((_, x)), Some((_, y)), Some((_, t)), Some((_, p))) = (x, y, t, p) else {
            return Ok(None);
        };
        let dtype = dataset.dtype().map_err(map_hdf5_error)?;
        let record_size = dtype.size();
        let order = dtype.byte_order();
        let total = dataset.shape().first().copied().unwrap_or(0);
        Ok(Some(Self {
            dataset: dataset.clone(),
            dtype,
            record_size,
            order,
            fields: [x, y, t, p],
            total,
        }))
    }

    /// Reads column `col` (`X`/`Y`/`T`/`P`) over `range` as `i64` by decoding the packed records.
    fn read_ints(&self, col: usize, range: Range<usize>) -> Result<Vec<i64>, IoError> {
        let n = range.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.read_records(range)?;
        let field = &self.fields[col];
        Ok(bytes
            .chunks_exact(self.record_size)
            .map(|record| decode_scalar(&record[field.offset..], &field.descriptor, self.order))
            .collect())
    }

    /// Reads the packed compound records for `range` into a byte buffer via `H5Dread`.
    fn read_records(&self, range: Range<usize>) -> Result<Vec<u8>, IoError> {
        let n = range.len();
        let mut buffer = vec![0u8; n * self.record_size];
        // Fresh copy of the dataset's dataspace (mutating its selection is local to this read),
        // with a [start, start+n) hyperslab; a matching 1-D memory space receives the records.
        let file_space = self.dataset.space().map_err(map_hdf5_error)?;
        let start = [range.start as hsize_t];
        let count = [n as hsize_t];
        let dims = [n as hsize_t];
        // The static libhdf5 build is not thread-safe; every call must hold the library's global
        // lock. hdf5-metno wraps its own calls in `sync`, so the raw FFI here must too, or it races
        // the safe API on other threads (corrupting internal state).
        // SAFETY: `file_space`/`self.dtype`/`self.dataset` are live HDF5 handles; the buffer holds
        // exactly `n * record_size` bytes, matching the selected element count and record stride.
        hdf5::sync::sync(|| unsafe {
            if H5Sselect_hyperslab(
                file_space.id(),
                H5S_SELECT_SET,
                start.as_ptr(),
                std::ptr::null(),
                count.as_ptr(),
                std::ptr::null(),
            ) < 0
            {
                return Err(IoError::Format(
                    "hdf5: hyperslab selection failed on compound dataset".to_owned(),
                ));
            }
            let mem_space = H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
            if mem_space < 0 {
                return Err(IoError::Format(
                    "hdf5: could not create memory dataspace for compound read".to_owned(),
                ));
            }
            let status = H5Dread(
                self.dataset.id(),
                self.dtype.id(),
                mem_space,
                file_space.id(),
                H5P_DEFAULT,
                buffer.as_mut_ptr().cast(),
            );
            H5Sclose(mem_space);
            if status < 0 {
                return Err(IoError::Format(
                    "hdf5: reading compound dataset records failed".to_owned(),
                ));
            }
            Ok(())
        })?;
        Ok(buffer)
    }
}

/// The compound layout of a dataset, or `None` if it isn't a compound type.
fn compound_of(dataset: &Dataset) -> Option<CompoundType> {
    match dataset.dtype().and_then(|dtype| dtype.to_descriptor()) {
        Ok(TypeDescriptor::Compound(compound)) => Some(compound),
        _ => None,
    }
}

/// Decodes one scalar as `i64`; `b` starts at the field's byte offset within a packed record.
/// Covers the integer / float / boolean / enum member types an event column may use, honouring
/// the compound's byte order.
fn decode_scalar(b: &[u8], descriptor: &TypeDescriptor, order: ByteOrder) -> i64 {
    use TypeDescriptor as TD;
    let be = matches!(order, ByteOrder::BigEndian);
    macro_rules! int {
        ($t:ty, $n:literal) => {{
            let mut bytes = [0u8; $n];
            bytes.copy_from_slice(&b[..$n]);
            (if be {
                <$t>::from_be_bytes(bytes)
            } else {
                <$t>::from_le_bytes(bytes)
            }) as i64
        }};
    }
    match descriptor {
        TD::Unsigned(IntSize::U1) => b[0] as i64,
        TD::Unsigned(IntSize::U2) => int!(u16, 2),
        TD::Unsigned(IntSize::U4) => int!(u32, 4),
        TD::Unsigned(IntSize::U8) => int!(u64, 8),
        TD::Integer(IntSize::U1) => b[0] as i8 as i64,
        TD::Integer(IntSize::U2) => int!(i16, 2),
        TD::Integer(IntSize::U4) => int!(i32, 4),
        TD::Integer(IntSize::U8) => int!(i64, 8),
        TD::Float(FloatSize::U4) => int!(f32, 4),
        TD::Float(FloatSize::U8) => int!(f64, 8),
        TD::Boolean => i64::from(b[0] != 0),
        // An enum stores an integer of its base width/signedness (e.g. an HDF5 boolean polarity).
        TD::Enum(enumeration) => match (enumeration.size, enumeration.signed) {
            (IntSize::U1, true) => b[0] as i8 as i64,
            (IntSize::U1, false) => b[0] as i64,
            (IntSize::U2, true) => int!(i16, 2),
            (IntSize::U2, false) => int!(u16, 2),
            (IntSize::U4, true) => int!(i32, 4),
            (IntSize::U4, false) => int!(u32, 4),
            (IntSize::U8, true) => int!(i64, 8),
            (IntSize::U8, false) => int!(u64, 8),
        },
        _ => 0,
    }
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
///
/// The columns are chunked at [`EVENT_CHUNK`] and, by default, byte-shuffled and deflated. That
/// costs write throughput and buys a great deal of disk: the raw layout is 13 bytes per event
/// (`u16` + `u16` + `i64` + `u8`), and a monotonic `t` under a byte shuffle is mostly runs of equal
/// high bytes. Chunking matters on its own too — the slicing reader is built around chunk-sized
/// reads, and a contiguous dataset gives it nothing to cache.
pub fn write_hdf5_stream(
    path: impl AsRef<Path>,
    stream: &EventStream,
    compression: Compression,
) -> Result<(), IoError> {
    let file = File::create(path.as_ref()).map_err(map_hdf5_error)?;
    let events = file.create_group("events").map_err(map_hdf5_error)?;
    write_event_column(&events, "x", stream.xs(), compression)?;
    write_event_column(&events, "y", stream.ys(), compression)?;
    write_event_column(&events, "t", stream.ts(), compression)?;
    let polarities: Vec<u8> = stream.ps().iter().map(|&p| u8::from(p)).collect();
    write_event_column(&events, "p", &polarities, compression)?;
    let (width, height) = stream.sensor_size();
    write_scalar_attr(&file, "width", width as u64)?;
    write_scalar_attr(&file, "height", height as u64)?;
    write_scalar_attr(&file, "timestamp_scale_ms", stream.timestamp_scale_ms())?;
    Ok(())
}

/// Writes one event column, chunked and optionally shuffled + deflated.
///
/// Borrows the data rather than copying it: the previous `to_vec()` doubled peak memory for the
/// duration of the save, which on a multi-gigabyte recording is the difference between saving and
/// not. An empty column keeps a chunked, resizable shape so the file still round-trips.
fn write_event_column<T: H5Type>(
    group: &Group,
    name: &str,
    data: &[T],
    compression: Compression,
) -> Result<(), IoError> {
    let mut builder = group.new_dataset::<T>().chunk(EVENT_CHUNK.min(data.len().max(1)));
    if let Some(level) = compression.level() {
        // Shuffle before deflate, not instead of it: on its own the shuffle changes nothing about
        // the size, but grouping like-significance bytes together is what gives deflate the long
        // runs it needs on `t` and `p`.
        builder = builder.shuffle().deflate(level);
    }
    let dataset = builder
        .shape((data.len(),))
        .create(name)
        .map_err(map_hdf5_error)?;
    if !data.is_empty() {
        dataset
            .write_raw(data)
            .map_err(map_hdf5_error)?;
    }
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

/// Events per chunk in the extendable `events/*` datasets. Large enough that appending many small
/// live windows (a 30 ms camera window may hold only a few thousand events) doesn't litter the file
/// with tiny chunks, small enough that the write cache for four columns stays modest.
const EVENT_CHUNK: usize = 1 << 16;

/// The four extendable column datasets, created lazily once the first window fixes the sensor grid.
struct SinkColumns {
    x: Dataset,
    y: Dataset,
    t: Dataset,
    p: Dataset,
}

/// Streams [`EventStream`] windows into extendable `events/{x,y,t,p}` datasets — the event-level
/// counterpart of [`Hdf5FrameSink`]. Where that sink appends computed frames, this appends the raw
/// events themselves, so a live camera can be recorded to disk window-by-window without holding the
/// whole session in memory. The file it produces is read back by [`read_hdf5`] / [`open_hdf5_slice`]
/// exactly like a recording written by [`write_hdf5_stream`]: the same `events` group, the same
/// `width` / `height` / `timestamp_scale_ms` root attributes, and the `t` column in the stream's own
/// raw units (microseconds for the usual `0.001` scale).
///
/// The sensor size and time base are taken from the first appended window (mirroring how
/// [`Hdf5FrameSink`] takes its shape from the first frame), so no dimensions are needed up front.
/// The datasets are always chunked (required for the unlimited append axis); `compression` decides
/// whether they are also shuffled and deflated, trading write throughput for size exactly as
/// [`write_hdf5_stream`] does.
pub struct Hdf5EventSink {
    file: File,
    compression: Compression,
    // `None` until the first non-empty append creates the group, datasets, and metadata attributes.
    columns: Option<SinkColumns>,
    count: usize,
}

impl Hdf5EventSink {
    /// Creates the output file; the `events` group and its column datasets are created lazily on the
    /// first non-empty [`append`](Self::append), which also stamps the sensor size and time base.
    pub fn open(path: impl AsRef<Path>, compression: Compression) -> Result<Self, IoError> {
        let file = File::create(path.as_ref()).map_err(map_hdf5_error)?;
        Ok(Self {
            file,
            compression,
            columns: None,
            count: 0,
        })
    }

    /// Appends one window's events to the end of every column. Empty windows are a no-op. The first
    /// non-empty window creates the datasets and writes `width` / `height` / `timestamp_scale_ms`
    /// from the stream. The `t` values are written verbatim (the stream's raw units), matching
    /// [`write_hdf5_stream`].
    pub fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        let n = stream.len();
        if n == 0 {
            return Ok(());
        }
        if self.columns.is_none() {
            let (width, height) = stream.sensor_size();
            let events = self.file.create_group("events").map_err(map_hdf5_error)?;
            let columns = SinkColumns {
                x: new_event_column::<u16>(&events, "x", self.compression)?,
                y: new_event_column::<u16>(&events, "y", self.compression)?,
                t: new_event_column::<i64>(&events, "t", self.compression)?,
                p: new_event_column::<u8>(&events, "p", self.compression)?,
            };
            write_scalar_attr(&self.file, "width", width as u64)?;
            write_scalar_attr(&self.file, "height", height as u64)?;
            write_scalar_attr(&self.file, "timestamp_scale_ms", stream.timestamp_scale_ms())?;
            self.columns = Some(columns);
        }
        let columns = self.columns.as_ref().expect("created above");
        let start = self.count;
        let end = start + n;
        append_column(&columns.x, start, end, stream.xs())?;
        append_column(&columns.y, start, end, stream.ys())?;
        append_column(&columns.t, start, end, stream.ts())?;
        let polarities: Vec<u8> = stream.ps().iter().map(|&p| u8::from(p)).collect();
        append_column(&columns.p, start, end, &polarities)?;
        self.count = end;
        Ok(())
    }

    /// Total events written so far (the shared length of every column).
    pub fn n_events(&self) -> usize {
        self.count
    }

    /// Forces buffered data out to the file without closing it, so a crash mid-recording keeps
    /// everything appended so far. Call periodically during a long capture.
    pub fn flush(&self) -> Result<(), IoError> {
        self.file.flush().map_err(map_hdf5_error)
    }

    /// Flushes and closes the file (dropping the sink also closes it).
    pub fn finish(self) -> Result<(), IoError> {
        self.file.flush().map_err(map_hdf5_error)
    }
}

/// HDF5 was the first appendable target and the one the trait's shape was taken from, so every
/// method here forwards to the inherent one.
impl super::EventSink for Hdf5EventSink {
    fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        Hdf5EventSink::append(self, stream)
    }

    fn n_events(&self) -> usize {
        Hdf5EventSink::n_events(self)
    }

    fn flush(&mut self) -> Result<(), IoError> {
        Hdf5EventSink::flush(self)
    }

    fn finish(self: Box<Self>) -> Result<(), IoError> {
        Hdf5EventSink::finish(*self)
    }
}

/// Creates one empty, extendable (`0..` on the append axis), chunked column dataset, optionally
/// shuffled and gzip-compressed.
fn new_event_column<T: H5Type>(
    group: &Group,
    name: &str,
    compression: Compression,
) -> Result<Dataset, IoError> {
    let mut builder = group.new_dataset::<T>().chunk(EVENT_CHUNK);
    if let Some(level) = compression.level() {
        builder = builder.shuffle().deflate(level);
    }
    builder
        .shape((0usize..,))
        .create(name)
        .map_err(map_hdf5_error)
}

/// Resizes `dataset` to `end` and writes `values` into the freshly opened `start..end` tail.
fn append_column<T: H5Type + Clone>(
    dataset: &Dataset,
    start: usize,
    end: usize,
    values: &[T],
) -> Result<(), IoError> {
    dataset.resize(end).map_err(map_hdf5_error)?;
    let array = Array1::from_vec(values.to_vec());
    dataset
        .write_slice(&array, ndarray::s![start..end])
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

    /// Writes four separate columns with the given dataset names under `group_path` (empty =
    /// root), microsecond timestamps `1000..`, returning the file path.
    fn write_named(tag: &str, group_path: &str, names: [&str; 4]) -> std::path::PathBuf {
        let dir = temp_dir(tag);
        let path = dir.join("events.h5");
        let file = File::create(&path).unwrap();
        let group = if group_path.is_empty() {
            None
        } else {
            Some(file.create_group(group_path).unwrap())
        };
        let container: &Group = group.as_ref().unwrap_or(&file);
        container
            .new_dataset_builder()
            .with_data(&[1u16, 3, 0][..])
            .create(names[0])
            .unwrap();
        container
            .new_dataset_builder()
            .with_data(&[2u16, 0, 1][..])
            .create(names[1])
            .unwrap();
        container
            .new_dataset_builder()
            .with_data(&[1000i64, 2000, 3000][..])
            .create(names[2])
            .unwrap();
        container
            .new_dataset_builder()
            .with_data(&[1u8, 0, 1][..])
            .create(names[3])
            .unwrap();
        path
    }

    #[test]
    fn detects_synonym_columns_under_a_nested_group() {
        // The QCR Fast-Slow / ROS dvs_msgs layout: events/{x_coordinates, …}, not x/y/t/p.
        let path = write_named(
            "h5syn",
            "events",
            ["x_coordinates", "y_coordinates", "timestamps", "polarities"],
        );
        let stream = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();
        assert_eq!(stream.xs(), &[1, 3, 0]);
        assert_eq!(stream.ys(), &[2, 0, 1]);
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        assert_eq!(stream.ps(), &[true, false, true]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn detects_root_level_synonyms() {
        let path = write_named("h5rootsyn", "", ["xs", "ys", "ts", "ps"]);
        let stream = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();
        assert_eq!(stream.len(), 3);
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[test]
    fn keys_override_names_arbitrary_datasets() {
        // Names that match no synonym: only an explicit `keys` mapping can resolve them.
        let path = write_named("h5keys", "raw", ["aa", "bb", "cc", "dd"]);
        let keys = EventKeys {
            x: "raw/aa".to_owned(),
            y: "raw/bb".to_owned(),
            t: "raw/cc".to_owned(),
            p: "raw/dd".to_owned(),
        };
        // Auto-detection fails and lists the datasets present.
        let error = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap_err();
        match error {
            IoError::Format(message) => {
                assert!(message.contains("could not identify"));
                assert!(message.contains("raw/aa"));
            }
            other => panic!("expected a format error, got {other:?}"),
        }
        // With keys it loads.
        let opts = LoadOptions {
            keys: Some(keys),
            ..options(4, 4, TimeUnit::Microseconds)
        };
        let stream = read_hdf5(&path, &opts).unwrap();
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        std::fs::remove_dir_all(path.parent().unwrap()).ok();
    }

    #[derive(hdf5::H5Type, Clone, Copy)]
    #[repr(C)]
    struct CdEvent {
        x: u16,
        y: u16,
        p: u8,
        t: i64,
    }

    #[test]
    fn detects_and_reads_a_compound_dataset() {
        let dir = temp_dir("h5compound");
        let path = dir.join("events.h5");
        let events = [
            CdEvent {
                x: 1,
                y: 2,
                p: 1,
                t: 1000,
            },
            CdEvent {
                x: 3,
                y: 0,
                p: 0,
                t: 2000,
            },
            CdEvent {
                x: 0,
                y: 1,
                p: 1,
                t: 3000,
            },
            CdEvent {
                x: 4,
                y: 0,
                p: 1,
                t: 4000,
            }, // out of a 4x4 sensor -> dropped
        ];
        {
            let file = File::create(&path).unwrap();
            file.new_dataset_builder()
                .with_data(&events[..])
                .create("CD")
                .unwrap();
        }
        // Auto-detected via the compound member names x/y/t/p, read by hyperslab + decode.
        let stream = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();
        assert_eq!(stream.len(), 3); // (4, 0) dropped
        assert_eq!(stream.xs(), &[1, 3, 0]);
        assert_eq!(stream.ys(), &[2, 0, 1]);
        assert_eq!(stream.ts(), &[1000, 2000, 3000]);
        assert_eq!(stream.ps(), &[true, false, true]);

        // Slicing over the same compound dataset agrees with the full load.
        let source = open_hdf5_slice(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();
        assert_eq!(source.n_events(), 4);
        assert_eq!(source.slice_time(2000, 4000).unwrap().ts(), &[2000, 3000]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn empty_compound_dataset_reads_empty() {
        let dir = temp_dir("h5compoundempty");
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            file.new_dataset_builder()
                .with_data(&[] as &[CdEvent])
                .create("CD")
                .unwrap();
        }
        let stream = read_hdf5(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();
        assert!(stream.is_empty());
        std::fs::remove_dir_all(&dir).ok();
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
    fn pixel_counts_reads_xy_only_and_drops_out_of_bounds() {
        // The hot-pixel pre-scan tallies coordinates without the t/p columns; it must still drop
        // out-of-bounds events exactly as read_range does. Small file -> the exact (non-sampled)
        // branch. Events: (1,2), (3,0), (0,1), and (4,0) which is out of a 4x4 sensor.
        let dir = temp_dir("h5counts");
        let path = dir.join("events.h5");
        {
            let file = File::create(&path).unwrap();
            let group = file.create_group("events").unwrap();
            for (name, data) in [("x", vec![1u16, 3, 0, 4]), ("y", vec![2u16, 0, 1, 0])] {
                group
                    .new_dataset_builder()
                    .with_data(&data[..])
                    .create(name)
                    .unwrap();
            }
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
        let source = open_hdf5_slice(&path, &options(4, 4, TimeUnit::Microseconds)).unwrap();

        let counts = source.pixel_counts().unwrap();
        assert_eq!(counts.len(), 16);
        assert_eq!(counts.iter().sum::<u64>(), 3); // (4, 0) dropped: x == width
        assert_eq!(counts[9], 1); // (x=1, y=2) -> 2*4 + 1
        assert_eq!(counts[3], 1); // (x=3, y=0)
        assert_eq!(counts[4], 1); // (x=0, y=1) -> 1*4 + 0
        std::fs::remove_dir_all(&dir).ok();
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
        write_hdf5_stream(&path, &stream, Compression::default()).unwrap();

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

    #[test]
    fn event_sink_appends_windows_and_round_trips() {
        // Two windows appended one after another must read back as one concatenated recording,
        // in order, with the sensor size and time base recovered from the attributes alone.
        let window = |events: &[(u16, u16, i64, bool)]| {
            let mut builder = EventStreamBuilder::new(12, 9, 0.001);
            for &(x, y, t, p) in events {
                builder.push(x, y, t, p);
            }
            builder.build()
        };
        let first = window(&[(1, 2, 100, true), (3, 4, 200, false)]);
        let second = window(&[(5, 6, 300, true), (11, 8, 400, false), (0, 0, 500, true)]);

        let dir = temp_dir("h5eventsink");
        let path = dir.join("live.h5");
        let mut sink = Hdf5EventSink::open(&path, Compression::None).unwrap();
        sink.append(&first).unwrap();
        sink.append(&window(&[])).unwrap(); // empty window is a no-op
        sink.flush().unwrap();
        sink.append(&second).unwrap();
        assert_eq!(sink.n_events(), 5);
        sink.finish().unwrap();

        // No options: size and unit come from the saved attributes, not inference.
        let loaded = read_hdf5(&path, &LoadOptions::default()).unwrap();
        assert_eq!(loaded.sensor_size(), (12, 9));
        assert_eq!(loaded.xs(), &[1, 3, 5, 11, 0]);
        assert_eq!(loaded.ys(), &[2, 4, 6, 8, 0]);
        assert_eq!(loaded.ts(), &[100, 200, 300, 400, 500]);
        assert_eq!(loaded.ps(), &[true, false, true, false, true]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compression_shrinks_a_stream_and_still_round_trips() {
        // The reason the writer defaults to compressed at all: uncompressed events cost 13 bytes
        // each, which is how a couple of seconds of 1080p simulation reached 2.2 GB. The reader is
        // filter-agnostic, so the only thing that could break is the writer itself.
        let mut builder = EventStreamBuilder::new(640, 480, 0.001);
        for index in 0..200_000i64 {
            builder.push(
                (index % 640) as u16,
                ((index / 640) % 480) as u16,
                index * 7,
                index % 3 == 0,
            );
        }
        let stream = builder.build();

        let dir = temp_dir("h5compress");
        let (plain, packed) = (dir.join("plain.h5"), dir.join("packed.h5"));
        write_hdf5_stream(&plain, &stream, Compression::None).unwrap();
        write_hdf5_stream(&packed, &stream, Compression::default()).unwrap();

        let size = |path: &std::path::Path| std::fs::metadata(path).unwrap().len();
        assert!(
            size(&packed) * 2 < size(&plain),
            "compression should more than halve this: {} vs {}",
            size(&packed),
            size(&plain)
        );

        for path in [&plain, &packed] {
            let loaded = read_hdf5(path, &LoadOptions::default()).unwrap();
            assert_eq!(loaded.sensor_size(), (640, 480));
            assert_eq!(loaded.xs(), stream.xs());
            assert_eq!(loaded.ys(), stream.ys());
            assert_eq!(loaded.ts(), stream.ts());
            assert_eq!(loaded.ps(), stream.ps());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_empty_stream_still_round_trips() {
        // The chunk size is clamped to the data length, so a zero-length column is the edge case
        // that clamp has to survive.
        let stream = EventStreamBuilder::new(4, 4, 0.001).build();
        let dir = temp_dir("h5empty");
        let path = dir.join("empty.h5");
        write_hdf5_stream(&path, &stream, Compression::default()).unwrap();
        let loaded = read_hdf5(&path, &LoadOptions::default()).unwrap();
        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.sensor_size(), (4, 4));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn event_sink_compressed_round_trips() {
        let mut builder = EventStreamBuilder::new(6, 6, 0.001);
        for t in 0..1000i64 {
            builder.push((t % 6) as u16, ((t / 6) % 6) as u16, t * 10, t % 2 == 0);
        }
        let stream = builder.build();

        let dir = temp_dir("h5eventsinkgz");
        let path = dir.join("live-gz.h5");
        let mut sink = Hdf5EventSink::open(&path, Compression::Gzip(4)).unwrap();
        sink.append(&stream).unwrap();
        sink.finish().unwrap();

        let loaded = read_hdf5(&path, &LoadOptions::default()).unwrap();
        assert_eq!(loaded.len(), 1000);
        assert_eq!(loaded.xs(), stream.xs());
        assert_eq!(loaded.ts(), stream.ts());
        assert_eq!(loaded.ps(), stream.ps());
        std::fs::remove_dir_all(&dir).ok();
    }
}
