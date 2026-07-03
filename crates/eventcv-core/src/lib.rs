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
//! - [`filter`], [`image`], [`viz`] — hot-pixel filtering, frame-domain resize, colormapped export.
//!
//! The `hdf5` feature (off by default to keep `cargo test` fast) enables the `.h5`/`.hdf5`
//! reader by building libhdf5 from source.

use ndarray::Array2;

pub mod camera;
pub mod cluster;
pub mod features;
pub mod filter;
pub mod flow;
pub mod image;
pub mod io;
pub mod representation;
pub mod transform;
pub mod viz;

const COLUMN_COUNT: usize = 4;

/// A stream of events stored column-wise (struct-of-arrays). Columns compress and
/// transform far better than interleaved rows, and timestamps use `i64` (µs) so
/// real multi-second recordings fit. See `TASKS.md` §3.
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
        (0..self.len()).map(move |index| Event {
            x: self.xs[index] as usize,
            y: self.ys[index] as usize,
            timestamp: self.ts[index] as u64,
            polarity: self.ps[index],
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
