//! # eventcv-core
//!
//! The Rust core of **EventCV** — "OpenCV for event-based vision".
//!
//! Everything is built around [`EventStream`], a struct-of-arrays container of events
//! (`xs`, `ys`, `ts`, `ps` columns plus sensor size and timestamp scale). Streams are
//! constructed only through [`EventStreamBuilder`], which drops out-of-bounds events.
//!
//! The crate is organised into focused modules:
//!
//! - [`io`] — readers/writers for `.npz`, `.txt`, `.bag`, `.h5`, `.aedat`, `.dat`, plus the
//!   [`io::load`] extension dispatcher and lazy [`io::SliceSource`] indexing for large files.
//! - [`representation`] — event → dense tensor ([`representation::Representation`], e.g. voxel
//!   grids and time surfaces).
//! - [`transform`] — chainable event-domain geometry/temporal/polarity ops on streams.
//! - [`camera`] — intrinsics and `undistort`.
//! - [`features`], [`flow`], [`cluster`] — corner detection, optical flow, connected components.
//! - [`feast`] — unsupervised online feature learning (FEAST adaptive-threshold clustering).
//! - [`filter`], [`image`], [`viz`] — hot-pixel filtering, frame-domain resize, colormapped export.
//! - [`mask`] — region-of-interest shapes (rectangle, ellipse, polygon) for [`EventStream::mask`].
//! - [`bias`] — the adaptive-biasing control law that holds a live camera's event rate steady.
//! - `device` — live USB event-camera capture into [`EventStream`] windows (`camera` feature).
//!
//! The `hdf5` feature (off by default to keep `cargo test` fast) enables the `.h5`/`.hdf5`
//! reader by building libhdf5 from source. The `camera` feature (also off by default) enables the
//! `device` module, pulling in the `neuromorphic-drivers` crate and a vendored libusb.

use ndarray::Array2;

pub mod accel;
pub mod analytics;
pub mod augment;
pub mod bias;
pub mod camera;
pub mod cluster;
pub mod cmax;
#[cfg(feature = "camera")]
pub mod device;
pub mod feast;
pub mod features;
pub mod filter;
pub mod flow;
pub mod image;
pub mod interp;
pub mod io;
pub mod mask;
/// ONNX inference. Requires the `onnx` feature (on in the published wheels).
#[cfg(feature = "onnx")]
pub mod model;
pub mod net;
#[cfg(feature = "ros2")]
pub mod ros2;

pub mod representation;
pub mod simulate;
pub mod track;
pub mod transform;
pub mod video;
pub mod viz;

const COLUMN_COUNT: usize = 4;

/// A stream of events stored column-wise (struct-of-arrays). Columns compress and
/// transform far better than interleaved rows, and timestamps use `i64` (µs) so
/// real multi-second recordings fit. See `TASKS.md` §3.
///
/// The columns are shared, not owned: a transform that rewrites one column hands the other three
/// on untouched, and `clone` is four refcount bumps rather than a copy of the whole recording.
/// `Arc<Vec<T>>` rather than `Arc<[T]>` because the latter cannot reuse a `Vec`'s allocation (the
/// refcount header sits in the same block), so every `EventStreamBuilder::build` — every reader
/// slice, every subsetting transform — would copy all four columns; and because `Arc::make_mut`
/// gives copy-on-write for free on a `Vec` and is not available on an unsized `[T]`.
#[derive(Clone, Debug)]
pub struct EventStream {
    xs: Vec<u16>,
    ys: Vec<u16>,
    ts: Vec<i64>,
    ps: Vec<bool>,
    width: usize,
    height: usize,
    timestamp_scale_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub x: usize,
    pub y: usize,
    pub timestamp: u64,
    pub polarity: bool,
}

impl EventStream {
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    pub fn sensor_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn timestamp_scale_ms(&self) -> f64 {
        self.timestamp_scale_ms
    }

    pub fn xs(&self) -> &[u16] {
        &self.xs
    }

    pub fn ys(&self) -> &[u16] {
        &self.ys
    }

    pub fn ts(&self) -> &[i64] {
        &self.ts
    }

    pub fn ps(&self) -> &[bool] {
        &self.ps
    }

    pub fn iter(&self) -> impl Iterator<Item = Event> + '_ {
        // Bind the columns once rather than re-projecting them per event: every representation
        // consumes the stream through here.
        let (xs, ys, ts, ps) = (self.xs(), self.ys(), self.ts(), self.ps());
        (0..self.len()).map(move |index| Event {
            x: xs[index] as usize,
            y: ys[index] as usize,
            timestamp: ts[index] as u64,
            polarity: ps[index],
        })
    }

    /// Materialises an owned `(N, 4)` array of `[x, y, t, p]` rows for numpy interop.
    pub fn to_array2(&self) -> Array2<u64> {
        let mut values = Vec::with_capacity(self.len() * COLUMN_COUNT);
        for index in 0..self.len() {
            values.push(u64::from(self.xs[index]));
            values.push(u64::from(self.ys[index]));
            values.push(self.ts[index] as u64);
            values.push(u64::from(self.ps[index]));
        }
        Array2::from_shape_vec((self.len(), COLUMN_COUNT), values)
            .expect("columns share a length by construction")
    }

    /// Test-only constructor from an `(N, 4)` `[x, y, t, p]` array. Preserves every
    /// row verbatim (no bounds filtering) so fixtures can exercise error paths.
    #[cfg(test)]
    pub(crate) fn from_array2(
        events: Array2<u64>,
        width: usize,
        height: usize,
        timestamp_scale_ms: f64,
    ) -> Self {
        Self {
            xs: events.column(0).iter().map(|&value| value as u16).collect(),
            ys: events.column(1).iter().map(|&value| value as u16).collect(),
            ts: events.column(2).iter().map(|&value| value as i64).collect(),
            ps: events.column(3).iter().map(|&value| value != 0).collect(),
            width,
            height,
            timestamp_scale_ms,
        }
    }
}

/// Builds an [`EventStream`] one event at a time, dropping events outside the
/// sensor. The single construction path shared by readers and (future) transforms.
#[derive(Clone, Debug)]
pub struct EventStreamBuilder {
    xs: Vec<u16>,
    ys: Vec<u16>,
    ts: Vec<i64>,
    ps: Vec<bool>,
    width: usize,
    height: usize,
    timestamp_scale_ms: f64,
}

impl EventStreamBuilder {
    pub fn new(width: usize, height: usize, timestamp_scale_ms: f64) -> Self {
        Self::with_capacity(width, height, timestamp_scale_ms, 0)
    }

    pub fn with_capacity(
        width: usize,
        height: usize,
        timestamp_scale_ms: f64,
        capacity: usize,
    ) -> Self {
        Self {
            xs: Vec::with_capacity(capacity),
            ys: Vec::with_capacity(capacity),
            ts: Vec::with_capacity(capacity),
            ps: Vec::with_capacity(capacity),
            width,
            height,
            timestamp_scale_ms,
        }
    }

    /// Appends an event, returning `false` if it lies outside the sensor and was
    /// dropped. Callers that treat out-of-bounds events as errors inspect the result.
    pub fn push(&mut self, x: u16, y: u16, timestamp: i64, polarity: bool) -> bool {
        if usize::from(x) >= self.width || usize::from(y) >= self.height {
            return false;
        }
        self.xs.push(x);
        self.ys.push(y);
        self.ts.push(timestamp);
        self.ps.push(polarity);
        true
    }

    /// Appends an event already known to lie on this builder's sensor, skipping the test
    /// [`push`](Self::push) makes. For the callers that have just made that test themselves —
    /// `EventStream::remap`, which has to range-check in `i64` before it can cast to `u16` — and
    /// would otherwise pay for it twice on every surviving event.
    ///
    /// The four pushes are spelled out again rather than shared with [`push`](Self::push): having
    /// `push` delegate here cost its callers about 25% on a per-event loop (measured on
    /// `decimate`), `#[inline]` included, and `push` is on the hot path of every reader. Inlined
    /// here so that `remap`'s call site pays nothing for the split.
    #[inline]
    pub(crate) fn push_in_bounds(&mut self, x: u16, y: u16, timestamp: i64, polarity: bool) {
        self.xs.push(x);
        self.ys.push(y);
        self.ts.push(timestamp);
        self.ps.push(polarity);
    }

    /// Appends every event of `stream`, dropping any that fall outside this builder's sensor.
    ///
    /// The bulk counterpart of [`push`](Self::push), for the callers that join streams rather than
    /// generate them — concatenation, and the streaming writers that hand on a window at a time.
    /// When every event fits (the case for anything produced by a reader or the simulator, which
    /// cannot emit a coordinate its own sensor does not have) this is four `extend_from_slice`
    /// calls instead of a bounds check and four pushes per event; the scan that establishes that is
    /// two comparisons per event and vectorises.
    pub fn extend_from_stream(&mut self, stream: &EventStream) {
        let fits = stream
            .xs()
            .iter()
            .zip(stream.ys())
            .all(|(&x, &y)| usize::from(x) < self.width && usize::from(y) < self.height);
        if fits {
            self.extend_from_columns(stream.xs(), stream.ys(), stream.ts(), stream.ps());
            return;
        }
        for index in 0..stream.len() {
            self.push(
                stream.xs()[index],
                stream.ys()[index],
                stream.ts()[index],
                stream.ps()[index],
            );
        }
    }

    /// Appends events whose coordinates are already known to lie on this builder's sensor —
    /// four `extend_from_slice` calls and no per-event check. The four slices must share a length.
    /// Callers that cannot make that guarantee want [`push`](Self::push) or
    /// [`extend_from_stream`](Self::extend_from_stream) instead.
    pub(crate) fn extend_from_columns(&mut self, xs: &[u16], ys: &[u16], ts: &[i64], ps: &[bool]) {
        self.xs.extend_from_slice(xs);
        self.ys.extend_from_slice(ys);
        self.ts.extend_from_slice(ts);
        self.ps.extend_from_slice(ps);
    }

    /// Reserves room for `additional` more events across every column.
    pub fn reserve(&mut self, additional: usize) {
        self.xs.reserve(additional);
        self.ys.reserve(additional);
        self.ts.reserve(additional);
        self.ps.reserve(additional);
    }

    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    pub fn build(self) -> EventStream {
        EventStream {
            xs: self.xs,
            ys: self.ys,
            ts: self.ts,
            ps: self.ps,
            width: self.width,
            height: self.height,
            timestamp_scale_ms: self.timestamp_scale_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::io::{load, LoadOptions};

    use super::{EventStream, EventStreamBuilder};

    #[test]
    fn loads_n_imagenet_events() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/test/example.npz");
        let stream = load(path, LoadOptions::default()).unwrap();
        let events = stream.to_array2();

        assert!(!stream.is_empty());
        assert_eq!(stream.sensor_size(), (640, 480));
        assert_eq!(events.dim(), (stream.len(), 4));
        assert!(events.column(0).iter().all(|&x| x < 640));
        assert!(events.column(1).iter().all(|&y| y < 480));
        assert!(events.column(3).iter().all(|&polarity| polarity <= 1));
    }

    #[test]
    fn builder_drops_out_of_bounds_events_and_keeps_columns_aligned() {
        let mut builder = EventStreamBuilder::new(4, 3, 0.001);

        assert!(builder.push(1, 2, 10, true));
        assert!(!builder.push(4, 0, 20, false)); // x == width -> dropped
        assert!(!builder.push(0, 3, 30, true)); // y == height -> dropped
        assert!(builder.push(3, 0, 40, false));

        let stream = builder.build();
        assert_eq!(stream.len(), 2);
        assert_eq!(stream.xs(), &[1, 3]);
        assert_eq!(stream.ys(), &[2, 0]);
        assert_eq!(stream.ts(), &[10, 40]);
        assert_eq!(stream.ps(), &[true, false]);
        assert_eq!(stream.sensor_size(), (4, 3));
    }

    #[test]
    fn to_array2_round_trips_columns_in_xytp_order() {
        let stream =
            EventStream::from_array2(ndarray::array![[1, 2, 100, 1], [3, 0, 250, 0]], 4, 3, 0.001);
        let events = stream.to_array2();

        assert_eq!(events.dim(), (2, 4));
        assert_eq!(events.row(0).to_vec(), vec![1, 2, 100, 1]);
        assert_eq!(events.row(1).to_vec(), vec![3, 0, 250, 0]);
    }
}
