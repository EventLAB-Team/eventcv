#[cfg(feature = "camera")]
mod capture;
mod viewer;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use eventcv_core::{
    camera::Camera,
    cluster::ClusterError,
    feast::{Feast, FeastConfig, FeastError},
    flow::FlowError,
    image::{PoolingMethod, ResizeError},
    io::{
        load_rows, open as open_reader, ColumnOrder, EventKeys, IoError, LoadOptions, RawRow,
        Reader, SaveOptions, TimeUnit,
    },
    representation::{
        AveragedTimeSurface, Binary, CountMask, EventCount, EventFrame, EventFrameData,
        EventPointSet, Mcts, PointSet, Polarity, Representation, RepresentationError, Tencode,
        TimeSurface, VoxelGrid,
    },
    viz::Colormap,
    EventStream, EventStreamBuilder,
};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
// Only the ONNX path returns a list (a multi-output graph), so the import carries the same gate
// as the code that uses it — otherwise a default build warns about it being unused.
#[cfg(feature = "onnx")]
use pyo3::types::PyList;

/// Default time span for the representations that have one (`window_ms`, `tau_ms`,
/// `max_window_ms`) when none of its unit forms is given.
const DEFAULT_SPAN_MS: f64 = 30.0;

/// The optional features compiled into this extension, in a fixed order so the string is stable.
/// Empty when neither is on — a plain `cargo build` of the bindings.
const FEATURES: &[&str] = &[
    #[cfg(feature = "hdf5")]
    "hdf5",
    #[cfg(feature = "camera")]
    "camera",
    #[cfg(feature = "onnx")]
    "onnx",
];

/// Events read per pass when converting a whole `EventReader` to another format. Large enough
/// that the per-chunk overhead disappears, small enough that the decoded columns stay well under
/// a gigabyte on any sensor.
const EXPORT_CHUNK: usize = 4_000_000;

/// Slices sampled to pick one display scale for an exported animation. A robust percentile over a
/// few frames tracks the recording closely enough, and bounding it keeps `save_video` from reading
/// a long file twice just to decide how bright to draw it.
const CLIM_CALIBRATION_FRAMES: usize = 8;

#[pyclass(name = "EventStream", frozen)]
struct PyEventStream {
    inner: eventcv_core::EventStream,
    /// Default representation carried from `open(repr=…)` / `with_repr` — what `view()` and
    /// `flatten()` render when no explicit representation is passed. `None` for raw streams
    /// (e.g. from `load`), which fall back to the polarity image.
    repr: Option<ReprSpec>,
}

#[pymethods]
impl PyEventStream {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.len(), 4)
    }

    #[getter]
    fn columns(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        ("x", "y", "t", "p")
    }

    #[getter]
    fn sensor_size(&self) -> (usize, usize) {
        self.inner.sensor_size()
    }

    #[getter]
    fn timestamp_scale_ms(&self) -> f64 {
        self.inner.timestamp_scale_ms()
    }

    /// The default representation name carried from `open(repr=…)` / `with_repr` (what
    /// `view()`/`flatten()` render), or `None` for a raw stream.
    #[getter]
    fn repr(&self) -> Option<&'static str> {
        self.repr.map(ReprSpec::name)
    }

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        self.inner.to_array2().into_pyarray(py)
    }

    /// Saves the stream to `path`, format chosen by extension (`.npz`/`.txt`/`.h5`/`.bag`, or
    /// `.zip` for E2VID) — the counterpart of `eventcv.load`. npz/HDF5/rosbag round-trip exactly;
    /// `topic` names the rosbag connection; `format` overrides the extension.
    #[pyo3(signature = (path, *, topic=None, format=None))]
    fn save(
        &self,
        py: Python<'_>,
        path: &str,
        topic: Option<String>,
        format: Option<String>,
    ) -> PyResult<()> {
        let options = SaveOptions {
            topic,
            format,
            ..SaveOptions::default()
        };
        py.detach(|| eventcv_core::io::save_stream(path, &self.inner, &options))
            .map_err(map_io_error)
    }

    // ---- Event-domain transforms (Workstream B). Each returns a new EventStream so they
    // chain; geometry is on the sparse stream itself (frame-domain resize is EventFrame.resize).

    /// Keeps events inside the `w`×`h` window at `(x0, y0)`, shifted to a new origin.
    fn crop(&self, py: Python<'_>, x0: i64, y0: i64, w: usize, h: usize) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.crop(x0, y0, w, h)))
    }

    /// Mirrors horizontally (`x → width-1-x`).
    fn flip_x(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.flip_x()))
    }

    /// Mirrors vertically (`y → height-1-y`).
    fn flip_y(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.flip_y()))
    }

    /// Rotates by `k * 90°` clockwise (quarter turns swap the sensor dims).
    fn rotate90(&self, py: Python<'_>, k: i32) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.rotate90(k)))
    }

    /// Reflects across the main diagonal (`(x, y) → (y, x)`); swaps the sensor dims.
    fn transpose(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.transpose()))
    }

    /// Translates by `(dx, dy)`; events shifted off the sensor are dropped.
    fn translate(&self, py: Python<'_>, dx: i64, dy: i64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.translate(dx, dy)))
    }

    /// Event-domain resize to a `width`×`height` grid (rebinned, not interpolated).
    fn resize(&self, py: Python<'_>, width: usize, height: usize) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.resize(width, height)))
    }

    /// Scales the sensor by `(sx, sy)`.
    fn scale(&self, py: Python<'_>, sx: f64, sy: f64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.scale(sx, sy)))
    }

    /// Applies a 2×3 affine matrix `[[a,b,c],[d,e,f]]` (rounded, no interpolation).
    fn warp_affine(&self, py: Python<'_>, matrix: [[f64; 3]; 2]) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.warp_affine(matrix)))
    }

    /// Applies a 3×3 perspective (homography) matrix.
    fn warp_perspective(&self, py: Python<'_>, matrix: [[f64; 3]; 3]) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.warp_perspective(matrix)))
    }

    /// Rectifies events with a `Camera`'s intrinsics + distortion (lens undistortion).
    fn undistort(&self, py: Python<'_>, camera: PyRef<'_, PyCamera>) -> PyEventStream {
        let camera = camera.inner;
        self.wrap(py.detach(|| self.inner.undistort(&camera)))
    }

    /// Keeps events inside the region of interest: a `(H, W)` boolean mask (or 8-bit map, where
    /// non-zero keeps the pixel) covering this stream's sensor. Build one with
    /// `eventcv.circle_mask` / `ellipse_mask` / `rect_mask` / `polygon_mask`, draw one with
    /// `draw_mask()`, or load one with `eventcv.load_mask`.
    fn mask(&self, py: Python<'_>, mask: &Bound<'_, PyAny>) -> PyResult<PyEventStream> {
        let (width, height) = self.inner.sensor_size();
        let flat = parse_mask(mask, (width, height))?;
        Ok(self.wrap(py.detach(|| self.inner.mask(&flat, width, height))))
    }

    /// Keeps events whose timestamp lies in the half-open window `[t0, t1)`. Positional bounds are
    /// microseconds (the unit timestamps are stored in); `t0_s`/`t0_ms`/`t0_us`/`t0_ns` and the
    /// `t1` equivalents take any unit.
    #[pyo3(signature = (t0=None, t1=None, *, t0_s=None, t0_ms=None, t0_us=None, t0_ns=None, t1_s=None, t1_ms=None, t1_us=None, t1_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn time_window(
        &self,
        py: Python<'_>,
        t0: Option<i64>,
        t1: Option<i64>,
        t0_s: Option<f64>,
        t0_ms: Option<f64>,
        t0_us: Option<f64>,
        t0_ns: Option<f64>,
        t1_s: Option<f64>,
        t1_ms: Option<f64>,
        t1_us: Option<f64>,
        t1_ns: Option<f64>,
    ) -> PyResult<PyEventStream> {
        let t0 = resolve_bound("t0", t0, t0_s, t0_ms, t0_us, t0_ns)?.unwrap_or(i64::MIN);
        let t1 = resolve_bound("t1", t1, t1_s, t1_ms, t1_us, t1_ns)?.unwrap_or(i64::MAX);
        Ok(self.wrap(py.detach(|| self.inner.time_window(t0, t1))))
    }

    /// Shifts every timestamp by `dt`. The positional form is microseconds; `dt_s`/`dt_ms`/`dt_us`/
    /// `dt_ns` take any unit, and unlike a window duration this one may be negative.
    #[pyo3(signature = (dt=None, *, dt_s=None, dt_ms=None, dt_us=None, dt_ns=None))]
    fn time_shift(
        &self,
        py: Python<'_>,
        dt: Option<i64>,
        dt_s: Option<f64>,
        dt_ms: Option<f64>,
        dt_us: Option<f64>,
        dt_ns: Option<f64>,
    ) -> PyResult<PyEventStream> {
        let dt = resolve_bound("dt", dt, dt_s, dt_ms, dt_us, dt_ns)?.unwrap_or(0);
        Ok(self.wrap(py.detach(|| self.inner.time_shift(dt))))
    }

    /// Scales every timestamp by `factor` (rounded).
    fn time_scale(&self, py: Python<'_>, factor: f64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.time_scale(factor)))
    }

    /// Shifts timestamps so the earliest event starts at zero.
    fn normalize_time(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.normalize_time()))
    }

    /// Keeps every `k`-th event by index.
    fn decimate(&self, py: Python<'_>, k: usize) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.decimate(k)))
    }

    /// Keeps only events of the given polarity (nonzero / `True` = ON, `0` / `False` = OFF).
    fn filter_polarity(&self, py: Python<'_>, polarity: i64) -> PyEventStream {
        let polarity = polarity != 0;
        self.wrap(py.detach(|| self.inner.filter_polarity(polarity)))
    }


    /// Recovers the camera motion that makes the warped events sharpest.
    ///
    /// Returns a dict with `params` (px/s for translation, rad/s for rotation), `score`,
    /// `score_at_rest` and `improvement` — check `improvement` before trusting `params`, since at
    /// or below 1.0 the optimiser found no motion and stopped wherever it happened to be.
    #[pyo3(signature = (
        *, model="translation", objective="variance", camera=None, blur_sigma=None,
        initial=None, initial_step=50.0, max_iterations=200, tolerance=1e-3,
        time_reference="midpoint"
    ))]
    #[allow(clippy::too_many_arguments)]
    fn contrast_maximise<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        objective: &str,
        camera: Option<PyRef<'_, PyCamera>>,
        blur_sigma: Option<f64>,
        initial: Option<Vec<f64>>,
        initial_step: f64,
        max_iterations: usize,
        tolerance: f64,
        time_reference: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        use eventcv_core::cmax::{CmaxConfig, TimeReference};

        let warp = parse_warp_model(model, camera.as_deref())?;
        let mut initial_params = [0.0_f64; 3];
        if let Some(values) = initial {
            if values.len() > 3 {
                return Err(PyValueError::new_err("initial takes at most three values"));
            }
            initial_params[..values.len()].copy_from_slice(&values);
        }
        let config = CmaxConfig {
            objective: parse_objective(objective)?,
            time_reference: match time_reference {
                "midpoint" => TimeReference::Midpoint,
                "start" => TimeReference::Start,
                "end" => TimeReference::End,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "time_reference must be \"midpoint\", \"start\" or \"end\", not {other:?}"
                    )))
                }
            },
            blur_sigma,
            initial: Some(initial_params),
            initial_step,
            max_iterations,
            tolerance,
        };
        let result = py
            .detach(|| self.inner.contrast_maximise(warp, config))
            .map_err(map_cmax_error)?;
        let dict = PyDict::new(py);
        dict.set_item("params", result.params.clone())?;
        dict.set_item("score", result.score)?;
        dict.set_item("score_at_rest", result.score_at_rest)?;
        dict.set_item("improvement", result.improvement())?;
        dict.set_item("iterations", result.iterations)?;
        Ok(dict)
    }

    /// Builds the image of warped events for an explicit motion — the picture the objective scores.
    #[pyo3(signature = (params, *, model="translation", camera=None))]
    fn iwe(
        &self,
        py: Python<'_>,
        params: Vec<f64>,
        model: &str,
        camera: Option<PyRef<'_, PyCamera>>,
    ) -> PyResult<PyEventFrame> {
        let warp = parse_warp_model(model, camera.as_deref())?;
        py.detach(|| self.inner.iwe(warp, &params))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_cmax_error)
    }

    /// Activity over time, binned into fixed intervals.
    ///
    /// Returns a dict of numpy arrays — `t` (each bin's left edge, µs), `count`, `positive`,
    /// `negative`, and `rate` (events/second) — so it plots directly:
    /// `plt.plot(*r["t"] / 1e6, r["rate"])`. The spatial counterpart is `pixel_counts()` on a
    /// reader, which totals events per pixel over a whole file.
    #[pyo3(signature = (bin_ms=None, *, bin_s=None, bin_us=None, bin_ns=None))]
    fn event_rate<'py>(
        &self,
        py: Python<'py>,
        bin_ms: Option<f64>,
        bin_s: Option<f64>,
        bin_us: Option<f64>,
        bin_ns: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let bin_us = resolve_duration_us("bin", bin_s, bin_ms, bin_us, bin_ns)?.unwrap_or(1_000);
        let rate = py.detach(|| self.inner.event_rate(bin_us));
        let dict = PyDict::new(py);
        dict.set_item("t", rate.starts.clone().into_pyarray(py))?;
        dict.set_item("count", rate.counts.clone().into_pyarray(py))?;
        dict.set_item("positive", rate.positive.clone().into_pyarray(py))?;
        dict.set_item("negative", rate.negative.clone().into_pyarray(py))?;
        dict.set_item("rate", rate.per_second().into_pyarray(py))?;
        dict.set_item("bin_us", rate.bin_us)?;
        Ok(dict)
    }

    // --- Random augmentations -------------------------------------------------------------
    // Every one is a pure function of its inputs and `seed`: the same seed gives the same result,
    // so a training run is reproducible from its config. On an `EventReader` the same names seed
    // per slice instead, so a shuffled DataLoader still augments each slice identically.

    /// Mirrors left-right with probability `p`.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn random_flip_x(&self, py: Python<'_>, p: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.random_flip_x(p, seed)))
    }

    /// Mirrors top-bottom with probability `p`.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn random_flip_y(&self, py: Python<'_>, p: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.random_flip_y(p, seed)))
    }

    /// Inverts every polarity with probability `p` (all events, not a random subset).
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn random_polarity_flip(&self, py: Python<'_>, p: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.random_polarity_flip(p, seed)))
    }

    /// Takes a random `width` × `height` crop; a window at least as large as the sensor is a no-op.
    #[pyo3(signature = (width, height, *, seed=0))]
    fn random_crop(&self, py: Python<'_>, width: usize, height: usize, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.random_crop(width, height, seed)))
    }

    /// Drops each event independently with probability `p`.
    #[pyo3(signature = (p=0.1, *, seed=0))]
    fn event_drop(&self, py: Python<'_>, p: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.event_drop(p, seed)))
    }

    /// Silences a random `p` fraction of pixels for the whole stream (dead-pixel simulation).
    #[pyo3(signature = (p=0.1, *, seed=0))]
    fn pixel_dropout(&self, py: Python<'_>, p: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.pixel_dropout(p, seed)))
    }

    /// Jitters each event's position by a gaussian offset of standard deviation `sigma` pixels.
    #[pyo3(signature = (sigma=1.0, *, seed=0))]
    fn spatial_jitter(&self, py: Python<'_>, sigma: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.spatial_jitter(sigma, seed)))
    }

    /// Jitters each event's timestamp by a gaussian offset, re-sorting afterwards.
    #[pyo3(signature = (sigma_ms=None, *, sigma_s=None, sigma_us=None, sigma_ns=None, seed=0))]
    fn time_jitter(
        &self,
        py: Python<'_>,
        sigma_ms: Option<f64>,
        sigma_s: Option<f64>,
        sigma_us: Option<f64>,
        sigma_ns: Option<f64>,
        seed: u64,
    ) -> PyResult<PyEventStream> {
        let sigma = resolve_us("sigma", sigma_s, sigma_ms, sigma_us, sigma_ns)?.unwrap_or(1000.0);
        Ok(self.wrap(py.detach(|| self.inner.time_jitter(sigma, seed))))
    }

    /// Plays the stream backwards with probability `p`, inverting polarity to match.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn time_reversal(&self, py: Python<'_>, p: f64, seed: u64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.time_reversal(p, seed)))
    }

    /// Flips every event's polarity.
    fn invert_polarity(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.invert_polarity()))
    }

    /// Returns a copy reordered by ascending timestamp (stable).
    fn sort_by_time(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.sort_by_time()))
    }

    /// Concatenates this stream with `others` (argument order; sensor = element-wise max).
    fn concat(&self, py: Python<'_>, others: Vec<PyRef<'_, PyEventStream>>) -> PyEventStream {
        let refs: Vec<&EventStream> = others.iter().map(|other| &other.inner).collect();
        self.wrap(py.detach(|| self.inner.concat(&refs)))
    }

    // ---- Denoising filters (Phase 3). Each drops noise events and returns a new EventStream.
    // The neighbourhood/dead-time filters assume ascending time (call sort_by_time first if not).

    /// Background-activity (nearest-neighbour) noise filter: keeps an event only if a 3×3
    /// neighbour fired within `dt` (raw timestamp units, e.g. microseconds).
    fn background_activity_filter(&self, py: Python<'_>, dt: i64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.background_activity_filter(dt)))
    }

    /// Refractory-period filter: suppresses a pixel's events for `dt` after it fires.
    fn refractory_filter(&self, py: Python<'_>, dt: i64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.refractory_filter(dt)))
    }

    /// Hot-pixel removal: drops pixels whose event count exceeds `mean + n_std·std` over the
    /// active pixels (default `n_std=3.0`).
    #[pyo3(signature = (n_std=3.0))]
    fn hot_pixel_filter(&self, py: Python<'_>, n_std: f64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.hot_pixel_filter(n_std)))
    }

    // ---- Feature detection (Phase 5). Corner detectors return the corner events as a new
    // EventStream (so they chain); both assume ascending time (call sort_by_time first if not).

    /// eFAST event corner detector (Mueggler et al., BMVC 2017). Keeps the events sitting on a
    /// moving corner, tested on two Bresenham rings over the per-polarity surface of active events.
    fn efast(&self, py: Python<'_>) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.efast()))
    }

    /// Harris corner score on the Surface of Active Events: keeps events whose Harris response
    /// `det - k·trace²` of the SAE-ramp structure tensor exceeds `threshold`. The default
    /// `threshold=0` keeps corners (rank-2, R>0) and rejects straight edges (rank-1, R<0); raise
    /// it to be stricter.
    #[pyo3(signature = (threshold=0.0))]
    fn harris_corners(&self, py: Python<'_>, threshold: f64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.harris_corners(threshold)))
    }

    /// Dense Lucas-Kanade optical flow on the time surface.
    ///
    /// Returns a two-channel `(flow_x, flow_y)` frame in pixels/ms; `window` is the half-width of
    /// the least-squares neighbourhood.
    #[pyo3(signature = (*, window=3))]
    fn optical_flow(&self, py: Python<'_>, window: usize) -> PyResult<PyEventFrame> {
        py.detach(|| self.inner.optical_flow(window))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_flow_error)
    }

    #[pyo3(signature = (representation=None, *, normalize=None, binary=false))]
    fn flatten(
        &self,
        py: Python<'_>,
        representation: Option<&Bound<'_, PyAny>>,
        normalize: Option<bool>,
        binary: bool,
    ) -> PyResult<PyEventFrame> {
        if binary {
            if representation.is_some() {
                return Err(PyValueError::new_err(
                    "binary=True cannot be combined with an explicit representation",
                ));
            }
            return py
                .detach(|| Binary.generate(&self.inner))
                .map(|inner| PyEventFrame { inner })
                .map_err(map_representation_error);
        }
        match representation {
            Some(representation) => self.generate(py, representation, normalize),
            None => self.default_frame(py, normalize),
        }
    }

    #[pyo3(signature = (*, bins=9, window_ms=None, window_s=None, window_us=None, window_ns=None))]
    fn voxel(
        &self,
        py: Python<'_>,
        bins: i64,
        window_ms: Option<f64>,
        window_s: Option<f64>,
        window_us: Option<f64>,
        window_ns: Option<f64>,
    ) -> PyResult<PyEventFrame> {
        let bins =
            usize::try_from(bins).map_err(|_| PyValueError::new_err("bins must be at least 1"))?;
        let window_ms = resolve_ms("window", window_s, window_ms, window_us, window_ns)?
            .unwrap_or(DEFAULT_SPAN_MS);
        py.detach(|| VoxelGrid::new(bins, window_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    #[pyo3(signature = (*, tau_ms=None, tau_s=None, tau_us=None, tau_ns=None))]
    fn tsurf(
        &self,
        py: Python<'_>,
        tau_ms: Option<f64>,
        tau_s: Option<f64>,
        tau_us: Option<f64>,
        tau_ns: Option<f64>,
    ) -> PyResult<PyEventFrame> {
        let tau_ms = resolve_ms("tau", tau_s, tau_ms, tau_us, tau_ns)?.unwrap_or(DEFAULT_SPAN_MS);
        py.detach(|| TimeSurface::new(tau_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Averaged time surface — the per-pixel mean of `exp(-age/tau_ms)` over all events
    /// (two polarity channels, float32). Brighter where activity recurs; see `tsurf`.
    #[pyo3(signature = (*, tau_ms=None, tau_s=None, tau_us=None, tau_ns=None))]
    fn atsurf(
        &self,
        py: Python<'_>,
        tau_ms: Option<f64>,
        tau_s: Option<f64>,
        tau_us: Option<f64>,
        tau_ns: Option<f64>,
    ) -> PyResult<PyEventFrame> {
        let tau_ms = resolve_ms("tau", tau_s, tau_ms, tau_us, tau_ns)?.unwrap_or(DEFAULT_SPAN_MS);
        py.detach(|| AveragedTimeSurface::new(tau_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Event-count image — one channel of total events per pixel (both polarities). Raw
    /// `uint64` counts by default; `uint8` rescaled to the busiest pixel when `normalize=True`.
    #[pyo3(signature = (*, normalize=false))]
    fn count(&self, py: Python<'_>, normalize: bool) -> PyResult<PyEventFrame> {
        py.detach(|| EventCount::new(normalize).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    fn pset(&self, py: Python<'_>) -> PyResult<PyEventPointSet> {
        py.detach(|| PointSet.generate(&self.inner))
            .map(|inner| PyEventPointSet { inner })
            .map_err(map_representation_error)
    }

    #[pyo3(signature = (*, window_ms=None, window_s=None, window_us=None, window_ns=None))]
    fn tencode(
        &self,
        py: Python<'_>,
        window_ms: Option<f64>,
        window_s: Option<f64>,
        window_us: Option<f64>,
        window_ns: Option<f64>,
    ) -> PyResult<PyEventFrame> {
        let window_ms = resolve_ms("window", window_s, window_ms, window_us, window_ns)?
            .unwrap_or(DEFAULT_SPAN_MS);
        py.detach(|| Tencode::new(window_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Count-mask image (GEPT, Sec. 3.2) — red holds the per-pixel positive-event count, blue the
    /// negative one, and green is a binary activity mask at full scale wherever any event landed.
    /// Both count planes are clipped and normalized by a single scale: the `pct`-th percentile of
    /// the non-zero counts of the two planes pooled together. Timestamps are not used.
    /// `white_frame` inverts to a white background — the black-background default is the form
    /// downstream descriptor models expect.
    #[pyo3(signature = (*, pct=99.0, white_frame=false))]
    fn countmask(&self, py: Python<'_>, pct: f64, white_frame: bool) -> PyResult<PyEventFrame> {
        py.detach(|| CountMask::new(pct, white_frame).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    #[pyo3(signature = (*, max_window_ms=None, max_window_s=None, max_window_us=None, max_window_ns=None))]
    fn mcts(
        &self,
        py: Python<'_>,
        max_window_ms: Option<f64>,
        max_window_s: Option<f64>,
        max_window_us: Option<f64>,
        max_window_ns: Option<f64>,
    ) -> PyResult<PyEventFrame> {
        let max_window_ms = resolve_ms(
            "max_window",
            max_window_s,
            max_window_ms,
            max_window_us,
            max_window_ns,
        )?
        .unwrap_or(DEFAULT_SPAN_MS);
        py.detach(|| Mcts::new(max_window_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Opens the interactive viewer on this stream. Pass a representation **name** to choose what
    /// to show — `stream.view("flow")`, `stream.view("count")`, `stream.view("voxel")`, … — or
    /// omit it to use the stream's stored representation (from `open(repr=…)`), falling back to
    /// the polarity image. (Equivalent to `stream.<repr>().view()`.)
    #[pyo3(signature = (representation=None, *, colormap="viridis", normalize=None))]
    fn view(
        &self,
        py: Python<'_>,
        representation: Option<&Bound<'_, PyAny>>,
        colormap: &str,
        normalize: Option<bool>,
    ) -> PyResult<()> {
        let frame = match representation {
            Some(representation) => self.generate(py, representation, normalize)?,
            None => self.default_frame(py, normalize)?,
        };
        // Rendering always auto-contrasts unless the caller opts out explicitly.
        frame.view(py, colormap, normalize.unwrap_or(true))
    }

    /// Draws a region of interest over this stream and returns it as an `(H, W)` boolean mask —
    /// the interactive way to build the argument `mask()` takes. Shows the same view `view()`
    /// would (pass a representation name to choose another), then: drag to keep an area,
    /// shift+drag to drop one, `e`/`r`/`f` to switch between ellipse, rectangle, and freehand,
    /// `a`/`c` to select all or clear, `z` to undo. Whatever stays bright is what the mask keeps.
    ///
    /// `Enter` accepts and returns the mask; closing the window or pressing `Esc` returns `None`.
    #[pyo3(signature = (representation=None, *, colormap="viridis", normalize=None))]
    fn draw_mask<'py>(
        &self,
        py: Python<'py>,
        representation: Option<&Bound<'_, PyAny>>,
        colormap: &str,
        normalize: Option<bool>,
    ) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        let frame = match representation {
            Some(representation) => self.generate(py, representation, normalize)?,
            None => self.default_frame(py, normalize)?,
        };
        frame.draw_mask(py, colormap, normalize.unwrap_or(true))
    }
}

impl PyEventStream {
    /// Wraps a core stream as a Python `EventStream`, carrying this stream's stored
    /// representation forward (used by the chainable transforms, so `open(repr=…)` survives
    /// `stream.flip_x().view()` etc.).
    fn wrap(&self, inner: EventStream) -> PyEventStream {
        PyEventStream {
            inner,
            repr: self.repr,
        }
    }

    /// The frame rendered when no explicit representation is passed: the stream's stored
    /// representation (from `open(repr=…)`) if set, else the default polarity image.
    fn default_frame(&self, py: Python<'_>, normalize: Option<bool>) -> PyResult<PyEventFrame> {
        match self.repr {
            Some(spec) => py
                .detach(|| spec.generate(&self.inner))
                .map(|inner| PyEventFrame { inner })
                .map_err(map_representation_error),
            None => self.generate_polarity(py, Polarity::new(normalize.unwrap_or(true))),
        }
    }

    fn generate(
        &self,
        py: Python<'_>,
        representation: &Bound<'_, PyAny>,
        normalize: Option<bool>,
    ) -> PyResult<PyEventFrame> {
        if representation.extract::<PyRef<'_, PyPolarity>>().is_ok() {
            return self.generate_polarity(py, Polarity::new(normalize.unwrap_or(true)));
        }
        // A representation *name* ("count", "flow", "voxel", …) renders that repr with its
        // defaults — so `stream.view("flow")` / `stream.flatten("count")` read naturally.
        if let Ok(name) = representation.extract::<String>() {
            let spec = ReprSpec::named(&name, normalize)?;
            return py
                .detach(|| spec.generate(&self.inner))
                .map(|inner| PyEventFrame { inner })
                .map_err(map_representation_error);
        }
        let name = representation.get_type().name()?.extract::<String>()?;
        Err(PyTypeError::new_err(format!(
            "unsupported representation: {name}"
        )))
    }

    fn generate_polarity(&self, py: Python<'_>, polarity: Polarity) -> PyResult<PyEventFrame> {
        py.detach(|| polarity.generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }
}

#[pyclass(name = "Polarity", frozen)]
struct PyPolarity;

#[pymethods]
impl PyPolarity {
    #[new]
    fn new() -> Self {
        Self
    }
}

/// Pinhole intrinsics + Brown–Conrady distortion, e.g. an EV-IMO `calib.txt`
/// (`fx fy cx cy k1 k2 p1 p2`). Pass to `stream.undistort(camera)`.
#[pyclass(name = "Camera", frozen)]
struct PyCamera {
    inner: Camera,
}

#[pymethods]
impl PyCamera {
    #[new]
    #[pyo3(signature = (fx, fy, cx, cy, *, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    ) -> Self {
        Self {
            inner: Camera::with_distortion(fx, fy, cx, cy, k1, k2, p1, p2, k3),
        }
    }

    #[getter]
    fn fx(&self) -> f64 {
        self.inner.fx
    }
    #[getter]
    fn fy(&self) -> f64 {
        self.inner.fy
    }
    #[getter]
    fn cx(&self) -> f64 {
        self.inner.cx
    }
    #[getter]
    fn cy(&self) -> f64 {
        self.inner.cy
    }
    #[getter]
    fn distortion(&self) -> (f64, f64, f64, f64, f64) {
        let c = &self.inner;
        (c.k1, c.k2, c.p1, c.p2, c.k3)
    }

    /// Maps a distorted pixel `(u, v)` to its undistorted location.
    fn undistort_point(&self, u: f64, v: f64) -> (f64, f64) {
        self.inner.undistort_point(u, v)
    }

    fn __repr__(&self) -> String {
        let c = &self.inner;
        format!(
            "Camera(fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, p1={}, p2={}, k3={})",
            c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.p1, c.p2, c.k3
        )
    }
}

#[pyclass(name = "EventFrame", frozen)]
struct PyEventFrame {
    inner: EventFrame,
}

#[pymethods]
impl PyEventFrame {
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn channel_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.channel_names().iter().map(String::as_str))
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind().as_str()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        frame_numpy(py, &self.inner)
    }

    /// Saves the frame to `path`. `.npz`/`.h5` store the raw array (shape, dtype, `kind`,
    /// `channel_names`) for `eventcv.load_frame`; `.png` writes a colormapped 2-D **view**
    /// (`colormap` = `viridis`/`turbo`/`grayscale`/`redblue`; `normalize` auto-contrasts).
    #[pyo3(signature = (path, *, colormap="viridis", normalize=true))]
    fn save(&self, py: Python<'_>, path: &str, colormap: &str, normalize: bool) -> PyResult<()> {
        let options = SaveOptions {
            colormap: parse_colormap(colormap)?,
            normalize: Some(normalize),
            ..SaveOptions::default()
        };
        py.detach(|| eventcv_core::io::save_frame(path, &self.inner, &options))
            .map_err(map_io_error)
    }

    /// Resize spatial dimensions using average or sum pooling on shrinking axes.
    #[pyo3(signature = (width, height, *, pooling="average"))]
    fn resize(
        &self,
        py: Python<'_>,
        width: i64,
        height: i64,
        pooling: &str,
    ) -> PyResult<PyEventFrame> {
        let width = usize::try_from(width)
            .map_err(|_| PyValueError::new_err("resize dimensions must be positive"))?;
        let height = usize::try_from(height)
            .map_err(|_| PyValueError::new_err("resize dimensions must be positive"))?;
        let pooling = match pooling {
            "average" => PoolingMethod::Average,
            "sum" => PoolingMethod::Sum,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported pooling method: {pooling}"
                )))
            }
        };

        py.detach(|| self.inner.resize(width, height, pooling))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_resize_error)
    }

    /// Connected-component labelling (Phase 5): treats any non-zero pixel as foreground and labels
    /// each 4- or 8-connected blob `1..=k`, background `0`. Returns a single-channel `u64` frame.
    #[pyo3(signature = (*, connectivity=8))]
    fn connected_components(&self, py: Python<'_>, connectivity: u8) -> PyResult<PyEventFrame> {
        py.detach(|| self.inner.connected_components(connectivity))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_cluster_error)
    }

    /// Opens the interactive GPU viewer. Image reprs are shown colour-mapped (`colormap`:
    /// `viridis`/`turbo`/`grayscale`/`redblue`; `normalize` auto-contrasts); volumetric
    /// reprs become an orbitable 3-D point cloud (drag to rotate, Esc to close).
    #[pyo3(signature = (*, colormap="viridis", normalize=true))]
    fn view(&self, py: Python<'_>, colormap: &str, normalize: bool) -> PyResult<()> {
        let colormap = parse_colormap(colormap)?;
        py.detach(|| viewer::view(&self.inner, colormap, normalize))
            .map_err(PyRuntimeError::new_err)
    }

    /// Draws a region of interest over this frame and returns it as an `(H, W)` boolean mask.
    /// See `EventStream.draw_mask` for the controls; `Esc` (or closing the window) returns `None`.
    #[pyo3(signature = (*, colormap="viridis", normalize=true))]
    fn draw_mask<'py>(
        &self,
        py: Python<'py>,
        colormap: &str,
        normalize: bool,
    ) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        let colormap = parse_colormap(colormap)?;
        let image = eventcv_core::viz::render_frame(&self.inner, colormap, normalize);
        let (width, height) = (image.width, image.height);
        let title = self.inner.kind().as_str().to_owned();
        // A still background is a producer that hands over its one frame and then has nothing new.
        let mut once = Some(image);
        let mask = py
            .detach(|| viewer::draw_mask(move || Ok(once.take()), width, height, title))
            .map_err(PyRuntimeError::new_err)?;
        mask.map(|mask| mask_to_py(py, mask, width, height))
            .transpose()
    }
}

#[pyclass(name = "EventPointSet", frozen)]
struct PyEventPointSet {
    inner: EventPointSet,
}

#[pymethods]
impl PyEventPointSet {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn columns(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        self.inner.columns()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        numpy::ndarray::Array2::from_shape_vec(self.inner.shape(), self.inner.data().to_vec())
            .expect("EventPointSet shape must match its data")
            .into_pyarray(py)
    }

    fn view(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| viewer::view_point_set(&self.inner))
            .map_err(PyRuntimeError::new_err)
    }
}

fn parse_sensor_size(sensor_size: Option<(i64, i64)>) -> PyResult<Option<(usize, usize)>> {
    sensor_size
        .map(|(width, height)| {
            Ok::<_, PyErr>((
                usize::try_from(width)
                    .map_err(|_| PyValueError::new_err("sensor width must be positive"))?,
                usize::try_from(height)
                    .map_err(|_| PyValueError::new_err("sensor height must be positive"))?,
            ))
        })
        .transpose()
}

/// Reads an ROI mask array as `(mask, width, height)`, row-major with `true` = keep.
///
/// Accepts a `(H, W)` boolean array or an 8-bit map where **any non-zero value keeps the pixel**,
/// so a mask binarised in another tool (or loaded from a PNG) is passed straight in. Any layout
/// works — the array is read through its strides, not its buffer.
fn mask_array(mask: &Bound<'_, PyAny>) -> PyResult<(Vec<bool>, usize, usize)> {
    if let Ok(array) = mask.extract::<PyReadonlyArray2<bool>>() {
        let view = array.as_array();
        Ok((
            view.iter().copied().collect(),
            view.shape()[1],
            view.shape()[0],
        ))
    } else if let Ok(array) = mask.extract::<PyReadonlyArray2<u8>>() {
        let view = array.as_array();
        Ok((
            view.iter().map(|&value| value != 0).collect(),
            view.shape()[1],
            view.shape()[0],
        ))
    } else {
        Err(PyTypeError::new_err(
            "mask must be a (H, W) numpy array of bool or uint8 (non-zero keeps the pixel); \
             for any other dtype pass `mask > 0`",
        ))
    }
}

/// As [`mask_array`], but rejects a mask that isn't the size of the `(width, height)` sensor it is
/// about to be applied to — silently dropping every event is a far worse failure than raising.
fn parse_mask(mask: &Bound<'_, PyAny>, sensor: (usize, usize)) -> PyResult<Vec<bool>> {
    let (flat, width, height) = mask_array(mask)?;
    check_mask_sensor((width, height), sensor)?;
    Ok(flat)
}

fn check_mask_sensor(mask: (usize, usize), sensor: (usize, usize)) -> PyResult<()> {
    if mask != sensor {
        return Err(PyValueError::new_err(format!(
            "mask is for a {}x{} sensor but this one is {}x{} — build it for this sensor, e.g. \
             eventcv.circle_mask(({}, {}), …)",
            mask.0, mask.1, sensor.0, sensor.1, sensor.0, sensor.1,
        )));
    }
    Ok(())
}

/// Hands a row-major mask back to Python as an `(H, W)` boolean array.
fn mask_to_py(
    py: Python<'_>,
    mask: Vec<bool>,
    width: usize,
    height: usize,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    Array2::from_shape_vec((height, width), mask)
        .map(|array| array.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// `None` or `"auto"` means infer the unit from the data; anything else is explicit.
fn parse_time_unit(time_unit: Option<&str>) -> PyResult<Option<TimeUnit>> {
    match time_unit {
        None | Some("auto") => Ok(None),
        Some(name) => name.parse::<TimeUnit>().map(Some).map_err(|_| {
            PyValueError::new_err(format!(
                "unsupported time_unit: {name} (expected s, ms, us, ns, or auto)"
            ))
        }),
    }
}

/// Microseconds per `unit`, for the getters that *report* a time (`dt`, `duration`, `time_span`)
/// rather than take one. `"auto"` has no meaning here, so only real units are accepted.
fn parse_unit_scale(unit: &str) -> PyResult<f64> {
    unit.parse::<TimeUnit>()
        .map(TimeUnit::scale_us)
        .map_err(PyValueError::new_err)
}

/// Resolves one time argument given as a set of mutually-exclusive `<name>_{s,ms,us,ns}` siblings,
/// returning microseconds — the unit every timestamp in eventcv is stored in.
///
/// Every time-valued argument in the Python API comes in these four forms, so `dt_us=500` and
/// `dt_ms=0.5` are the same request. Passing two of them for the same quantity is an error rather
/// than a precedence rule to remember.
fn resolve_us(
    name: &str,
    seconds: Option<f64>,
    milliseconds: Option<f64>,
    microseconds: Option<f64>,
    nanoseconds: Option<f64>,
) -> PyResult<Option<f64>> {
    let given: Vec<(TimeUnit, f64)> = [
        (TimeUnit::Seconds, seconds),
        (TimeUnit::Milliseconds, milliseconds),
        (TimeUnit::Microseconds, microseconds),
        (TimeUnit::Nanoseconds, nanoseconds),
    ]
    .into_iter()
    .filter_map(|(unit, value)| value.map(|value| (unit, value)))
    .collect();
    if given.len() > 1 {
        let names: Vec<String> = given
            .iter()
            .map(|(unit, _)| format!("{name}_{unit}"))
            .collect();
        return Err(PyValueError::new_err(format!(
            "pass one of {name}_s / {name}_ms / {name}_us / {name}_ns, not {}",
            names.join(" and "),
        )));
    }
    match given.first() {
        None => Ok(None),
        Some((_, value)) if value.is_nan() => {
            Err(PyValueError::new_err(format!("{name} must be a number")))
        }
        Some((unit, value)) => Ok(Some(value * unit.scale_us())),
    }
}

/// [`resolve_us`] for a signed point or shift on the timestamp axis (`time_window`'s bounds,
/// `time_shift`'s offset), which also accept a bare positional value in stored microseconds.
fn resolve_bound(
    name: &str,
    raw_us: Option<i64>,
    seconds: Option<f64>,
    milliseconds: Option<f64>,
    microseconds: Option<f64>,
    nanoseconds: Option<f64>,
) -> PyResult<Option<i64>> {
    let resolved = resolve_us(name, seconds, milliseconds, microseconds, nanoseconds)?;
    match (raw_us, resolved) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
            "pass {name} positionally (microseconds) or as {name}_s/{name}_ms/{name}_us/{name}_ns, \
             not both"
        ))),
        (Some(raw), None) => Ok(Some(raw)),
        (None, resolved) => Ok(resolved.map(|us| us.round() as i64)),
    }
}

/// [`resolve_us`] for the representation spans, which the core takes in milliseconds.
fn resolve_ms(
    name: &str,
    seconds: Option<f64>,
    milliseconds: Option<f64>,
    microseconds: Option<f64>,
    nanoseconds: Option<f64>,
) -> PyResult<Option<f64>> {
    Ok(resolve_us(name, seconds, milliseconds, microseconds, nanoseconds)?.map(|us| us / 1000.0))
}

/// [`resolve_us`] for a *duration* — something that has to be positive and land on the microsecond
/// grid, like `dt` or a window `step`. A value below half a microsecond is rejected rather than
/// rounded up to 1 µs, so `dt_ns=100` says so instead of silently becoming ten times itself.
fn resolve_duration_us(
    name: &str,
    seconds: Option<f64>,
    milliseconds: Option<f64>,
    microseconds: Option<f64>,
    nanoseconds: Option<f64>,
) -> PyResult<Option<i64>> {
    let Some(us) = resolve_us(name, seconds, milliseconds, microseconds, nanoseconds)? else {
        return Ok(None);
    };
    if us <= 0.0 {
        return Err(PyValueError::new_err(format!("{name} must be positive")));
    }
    let rounded = us.round() as i64;
    if rounded < 1 {
        return Err(PyValueError::new_err(format!(
            "{name} of {us} us is below eventcv's one-microsecond timestamp resolution"
        )));
    }
    Ok(Some(rounded))
}

fn parse_colormap(name: &str) -> PyResult<Colormap> {
    Colormap::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported colormap: {name}")))
}

fn parse_order(order: &str) -> PyResult<ColumnOrder> {
    Ok(match order {
        "txyp" => ColumnOrder::Txyp,
        "xytp" => ColumnOrder::Xytp,
        other => return Err(PyValueError::new_err(format!("unsupported order: {other}"))),
    })
}

/// Turns a `keys={"x": …, "y": …, "t": …, "p": …}` mapping into [`EventKeys`], requiring all
/// four fields. Values are format-specific column names (HDF5 dataset paths / `dataset/field`;
/// text header names or indices) — see `eventcv.load`.
fn parse_keys(keys: Option<HashMap<String, String>>) -> PyResult<Option<EventKeys>> {
    let Some(mut keys) = keys else {
        return Ok(None);
    };
    let mut take = |field: &str| -> PyResult<String> {
        keys.remove(field)
            .ok_or_else(|| PyValueError::new_err(format!("keys is missing '{field}'")))
    };
    let parsed = EventKeys {
        x: take("x")?,
        y: take("y")?,
        t: take("t")?,
        p: take("p")?,
    };
    if let Some(extra) = keys.keys().next() {
        return Err(PyValueError::new_err(format!(
            "keys has an unexpected entry '{extra}' (only x/y/t/p are used)"
        )));
    }
    Ok(Some(parsed))
}

fn us_to_ms(us: i64) -> f64 {
    us as f64 / 1000.0
}

/// Converts an `offset` argument — an absolute timestamp in the file's own time base — into
/// microseconds. `None` means "read from the start"; the bare `offset` is milliseconds.
fn parse_offset(
    offset: Option<f64>,
    offset_s: Option<f64>,
    offset_ms: Option<f64>,
    offset_us: Option<f64>,
    offset_ns: Option<f64>,
) -> PyResult<Option<i64>> {
    // `offset` predates the unit siblings and means milliseconds, so it is `offset_ms` by another
    // name — accepted, but not alongside it.
    let milliseconds = match (offset, offset_ms) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "pass offset or offset_ms, not both (they are the same argument)",
            ))
        }
        (bare, suffixed) => bare.or(suffixed),
    };
    match resolve_us("offset", offset_s, milliseconds, offset_us, offset_ns)? {
        None => Ok(None),
        Some(us) if us < 0.0 => Err(PyValueError::new_err("offset must be non-negative")),
        Some(us) => Ok(Some(us.round() as i64)),
    }
}

#[pyfunction]
#[pyo3(signature = (path, *, sensor_size=None, time_unit=None, order="txyp", topic=None, max_events=None, offset=None, offset_s=None, offset_ms=None, offset_us=None, offset_ns=None, keys=None))]
#[allow(clippy::too_many_arguments)]
fn load(
    py: Python<'_>,
    path: &str,
    sensor_size: Option<(i64, i64)>,
    time_unit: Option<&str>,
    order: &str,
    topic: Option<String>,
    max_events: Option<i64>,
    offset: Option<f64>,
    offset_s: Option<f64>,
    offset_ms: Option<f64>,
    offset_us: Option<f64>,
    offset_ns: Option<f64>,
    keys: Option<HashMap<String, String>>,
) -> PyResult<PyEventStream> {
    let max_events = max_events
        .map(|count| {
            usize::try_from(count)
                .map_err(|_| PyValueError::new_err("max_events must be non-negative"))
        })
        .transpose()?;
    let options = LoadOptions {
        sensor_size: parse_sensor_size(sensor_size)?,
        time_unit: parse_time_unit(time_unit)?,
        order: parse_order(order)?,
        topic,
        max_events,
        offset: parse_offset(offset, offset_s, offset_ms, offset_us, offset_ns)?,
        keys: parse_keys(keys)?,
    };
    py.detach(|| eventcv_core::io::load(path, options))
        .map(|inner| PyEventStream { inner, repr: None })
        .map_err(map_io_error)
}

/// Builds an `EventStream` from an in-memory `(N, 4)` numpy array — the constructor
/// mirror of `EventStream.numpy()`. Columns follow `order` (default `xytp`, matching
/// `numpy()`'s output); `sensor_size` and `time_unit` are inferred exactly like `load`
/// when omitted. Any integer or float dtype is accepted.
#[pyfunction]
#[pyo3(signature = (events, *, sensor_size=None, time_unit=None, order="xytp"))]
fn from_numpy(
    py: Python<'_>,
    events: &Bound<'_, PyAny>,
    sensor_size: Option<(i64, i64)>,
    time_unit: Option<&str>,
    order: &str,
) -> PyResult<PyEventStream> {
    let options = LoadOptions {
        sensor_size: parse_sensor_size(sensor_size)?,
        time_unit: parse_time_unit(time_unit)?,
        order: parse_order(order)?,
        topic: None,
        max_events: None,
        offset: None,
        keys: None,
    };
    let rows = numpy_rows(events, options.order)?;
    py.detach(|| load_rows(&rows, &options))
        .map(|inner| PyEventStream { inner, repr: None })
        .map_err(map_io_error)
}

/// Converts an `(N, ≥4)` numpy array of any int/float dtype into raw event rows laid
/// out per `order` (extra columns are ignored, like the text loader).
fn numpy_rows(events: &Bound<'_, PyAny>, order: ColumnOrder) -> PyResult<Vec<RawRow>> {
    macro_rules! try_dtype {
        ($($ty:ty),*) => {$(
            if let Ok(array) = events.extract::<PyReadonlyArray2<$ty>>() {
                return rows_from_view(&array.as_array().mapv(|value| value as f64), order);
            }
        )*};
    }
    try_dtype!(u64, i64, f64, u32, i32, f32, u16, i16, u8);
    Err(PyTypeError::new_err(
        "events must be an (N, 4) numpy array of an integer or float dtype",
    ))
}

fn rows_from_view(view: &Array2<f64>, order: ColumnOrder) -> PyResult<Vec<RawRow>> {
    if view.ncols() < 4 {
        return Err(PyValueError::new_err(format!(
            "expected an (N, 4) event array, got {} column(s)",
            view.ncols()
        )));
    }
    let (tc, xc, yc, pc) = match order {
        ColumnOrder::Txyp => (0, 1, 2, 3),
        ColumnOrder::Xytp => (2, 0, 1, 3),
    };
    let mut rows = Vec::with_capacity(view.nrows());
    for (index, row) in view.rows().into_iter().enumerate() {
        let coord = |value: f64, name: &str| -> PyResult<u16> {
            let rounded = value.round();
            if !(0.0..=f64::from(u16::MAX)).contains(&rounded) {
                return Err(PyValueError::new_err(format!(
                    "row {index}: {name} = {value} is not a valid sensor coordinate"
                )));
            }
            Ok(rounded as u16)
        };
        rows.push(RawRow {
            x: coord(row[xc], "x")?,
            y: coord(row[yc], "y")?,
            t: row[tc],
            p: row[pc] > 0.0,
        });
    }
    Ok(rows)
}

fn parse_dt(
    dt_s: Option<f64>,
    dt_ms: Option<f64>,
    dt_us: Option<f64>,
    dt_ns: Option<f64>,
) -> PyResult<Option<i64>> {
    resolve_duration_us("dt", dt_s, dt_ms, dt_us, dt_ns)
}

#[pyfunction]
#[pyo3(signature = (path, *, dt_ms=None, dt_s=None, dt_us=None, dt_ns=None, max_events=None, offset=None, offset_s=None, offset_ms=None, offset_us=None, offset_ns=None, repr=None, sensor_size=None, time_unit=None, order="txyp", topic=None, hot_pixel_filter=false, hot_pixel_std=3.0, keys=None))]
#[allow(clippy::too_many_arguments)]
fn open(
    py: Python<'_>,
    path: &str,
    dt_ms: Option<f64>,
    dt_s: Option<f64>,
    dt_us: Option<f64>,
    dt_ns: Option<f64>,
    max_events: Option<i64>,
    offset: Option<f64>,
    offset_s: Option<f64>,
    offset_ms: Option<f64>,
    offset_us: Option<f64>,
    offset_ns: Option<f64>,
    repr: Option<&str>,
    sensor_size: Option<(i64, i64)>,
    time_unit: Option<&str>,
    order: &str,
    topic: Option<String>,
    hot_pixel_filter: bool,
    hot_pixel_std: f64,
    keys: Option<HashMap<String, String>>,
) -> PyResult<PyEventReader> {
    let mode = match (parse_dt(dt_s, dt_ms, dt_us, dt_ns)?, max_events) {
        (Some(_), Some(_)) => return Err(PyValueError::new_err(
            "pass either dt (fixed-duration slices) or max_events (fixed-count slices), not both",
        )),
        (Some(dt), None) => Some(SliceMode::Duration(dt)),
        (None, Some(len)) => Some(SliceMode::Count(
            usize::try_from(len)
                .ok()
                .filter(|&len| len >= 1)
                .ok_or_else(|| PyValueError::new_err("max_events must be at least 1"))?,
        )),
        (None, None) => None,
    };
    let origin_us = parse_offset(offset, offset_s, offset_ms, offset_us, offset_ns)?;
    let repr = repr.map(|name| ReprSpec::named(name, None)).transpose()?;
    let options = LoadOptions {
        sensor_size: parse_sensor_size(sensor_size)?,
        time_unit: parse_time_unit(time_unit)?,
        order: parse_order(order)?,
        topic,
        max_events: None,
        offset: None,
        keys: parse_keys(keys)?,
    };
    let reader = py
        .detach(|| open_reader(path, options))
        .map_err(map_io_error)?;
    // Translate the absolute offset timestamp into a base event index for fixed-count framing:
    // the events before it are the prefix `slice(0)` must skip. Duration framing needs no index
    // (its windows start at the origin timestamp), so this read only runs for count mode.
    let base_index = match (origin_us, mode) {
        (Some(origin), Some(SliceMode::Count(_))) => {
            let (lo, _) = reader.time_span();
            let origin = origin.max(lo);
            if origin > lo {
                reader.slice_time(lo, origin).map_err(map_io_error)?.len()
            } else {
                0
            }
        }
        _ => 0,
    };
    // Whole-recording hot-pixel mask: tally per-pixel counts once now (the accepted pre-processing
    // cost) so every slice drops the same stuck pixels. A per-slice filter can't — its threshold
    // shifts with each window, letting hot pixels survive at long dt_ms. `pixel_counts` reads only
    // the coordinate columns (HDF5), so the scan stays lazy and cheap. `None` unless requested.
    let hot_pixel_mask = if hot_pixel_filter {
        let counts = reader.pixel_counts().map_err(map_io_error)?;
        Some(Arc::from(EventStream::hot_pixel_mask_from_counts(
            &counts,
            hot_pixel_std,
        )))
    } else {
        None
    };
    Ok(PyEventReader {
        inner: Arc::new(Mutex::new(reader)),
        mode,
        repr,
        slice_ops: no_slice_ops(),
        offset_us: origin_us,
        base_index,
        hot_pixel_mask,
    })
}

/// Lazy, seekable handle over a file's events — the `VideoCapture` to `load`'s
/// `imread`. Slices are fetched on demand (HDF5 by binary-searching its timestamps on
/// disk), so multi-GB files never need to be fully resident. Opening with `dt_ms` fixes
/// a frame duration so `slice(n)` returns the `n`-th frame (like seeking a video).
#[pyclass(name = "EventReader", frozen)]
struct PyEventReader {
    inner: Arc<Mutex<Reader>>,
    /// How the recording is partitioned into frames from `open(dt_ms=…)` /
    /// `open(max_events=…)`; enables integer `slice(n)`.
    mode: Option<SliceMode>,
    /// Per-slice dense rendering set by `open(repr=…)` / `with_repr` — makes the reader a
    /// map-style dataset (`reader[i]` → `[C, H, W]`, `batch` → `[B, C, H, W]`).
    repr: Option<ReprSpec>,
    /// Deferred per-slice stream ops set by applying a stream op to the reader
    /// (`reader.hot_pixel_filter()`, `efast`, …) — applied in order to every slice before any
    /// `repr`, so they compose with `slice`/`windows`/`with_repr`.
    slice_ops: SliceOps,
    /// `open(offset=…)`: the absolute framing origin (µs, the file's own time base). `slice(0)` /
    /// `windows()` / `n_slices` all start here (clamped up to `t_min`); events before it are
    /// outside every indexed frame. `None` frames the whole recording from `t_min`.
    offset_us: Option<i64>,
    /// The event index of the first event at/after the offset origin — the fixed-count framing
    /// counterpart of `offset_us` (0 for duration mode, no offset, or an offset past the end).
    base_index: usize,
    /// Whole-recording hot-pixel mask from `open(hot_pixel_filter=True)` (row-major
    /// `width·height`, `true` = drop). Computed once at `open` and applied to every fetched slice
    /// before any per-slice op, so stuck pixels are removed consistently across frames; `None`
    /// when hot-pixel filtering was not requested.
    hot_pixel_mask: Option<Arc<[bool]>>,
}

/// A per-slice `EventStream` → `EventStream` transform a reader defers onto every fetched
/// slice — the machinery behind `reader.hot_pixel_filter()`, `reader.flip_x()`, `efast`, … .
/// Boxed so any stream op composes, and `Send + Sync` so it runs off-GIL in `py.detach`.
/// The `usize` is the index of the slice being processed. Deterministic ops ignore it; the random
/// augmentations seed from it, so slice `i` augments identically however it is reached — a shuffled
/// `DataLoader`, several workers at once, or plain iteration all agree. Threading the index is what
/// lets the RNG stay a pure function of `(seed, index)` instead of state carried between calls.
type SliceOp = Arc<dyn Fn(usize, EventStream) -> EventStream + Send + Sync>;

/// A reader's chain of deferred per-slice ops (empty = none), applied in push order.
type SliceOps = Arc<[SliceOp]>;

/// The empty op chain (a reader with no deferred per-slice transforms).
fn no_slice_ops() -> SliceOps {
    Arc::from(Vec::new())
}

/// Applies a reader's deferred per-slice ops (in order) to a freshly fetched slice.
fn apply_slice_ops(ops: &[SliceOp], index: usize, stream: EventStream) -> EventStream {
    ops.iter().fold(stream, |stream, op| op(index, stream))
}

/// Combines a user's `seed` with a slice index into the seed that slice's augmentation uses.
///
/// Adding them would make `seed=0, slice=1` and `seed=1, slice=0` the same draw, which would tie
/// together runs a user expects to be independent; the odd multiplier keeps those apart, and
/// `augment::slice_rng` runs the result through splitmix64's finalizer before use.
fn mix_seed(seed: u64, index: usize) -> u64 {
    seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// How `slice(n)` / `reader[n]` / `windows()` partition the recording: fixed-duration
/// frames (`open(dt_ms=…)`, µs) or fixed-count frames (`open(max_events=…)`).
#[derive(Clone, Copy)]
enum SliceMode {
    Duration(i64),
    Count(usize),
}

/// One resolved slice: a half-open time window (µs) or event-index range.
#[derive(Clone, Copy)]
enum SliceWindow {
    Time(i64, i64),
    Index(usize, usize),
}

/// Reads `window` from the shared reader off-GIL, dropping the whole-recording hot pixels (if
/// any) before the deferred per-slice ops, so corner detectors never fire on stuck pixels.
///
/// `index` identifies which slice this is, and is handed to every deferred op — see [`SliceOp`].
fn fetch_window(
    py: Python<'_>,
    reader: &Arc<Mutex<Reader>>,
    window: SliceWindow,
    index: usize,
    ops: SliceOps,
    hot_pixel_mask: Option<Arc<[bool]>>,
) -> PyResult<EventStream> {
    let reader = Arc::clone(reader);
    py.detach(move || {
        let guard = reader.lock().unwrap();
        match window {
            SliceWindow::Time(t0, t1) => guard.slice_time(t0, t1),
            SliceWindow::Index(i0, i1) => guard.slice_index(i0, i1),
        }
        .map(|stream| {
            let stream = match &hot_pixel_mask {
                Some(mask) => stream.drop_masked_pixels(mask),
                None => stream,
            };
            apply_slice_ops(&ops, index, stream)
        })
    })
    .map_err(map_io_error)
}

/// Wraps a freshly fetched slice for a public slice API. When the reader carries a
/// representation (`open(repr=…)` / `with_repr`) the slice is rendered eagerly and returned as
/// an `EventFrame` (so `open(repr="mcts").slice(0)` == the frame `open().slice(0).mcts()`
/// yields); otherwise the raw `EventStream` is returned.
fn render_slice(
    py: Python<'_>,
    repr: Option<ReprSpec>,
    stream: EventStream,
) -> PyResult<Py<PyAny>> {
    match repr {
        Some(spec) => {
            let frame = py
                .detach(|| spec.generate(&stream))
                .map_err(map_representation_error)?;
            Ok(Bound::new(py, PyEventFrame { inner: frame })?
                .into_any()
                .unbind())
        }
        None => Ok(Bound::new(
            py,
            PyEventStream {
                inner: stream,
                repr: None,
            },
        )?
        .into_any()
        .unbind()),
    }
}

#[pymethods]
impl PyEventReader {
    /// `n_slices` in the `dt_ms` / `max_events` frame-dataset modes (so `len(reader) ==
    /// len(reader[:])`), else the raw event count. Lets `DataLoader` iterate the reader directly.
    fn __len__(&self) -> PyResult<usize> {
        match self.mode {
            Some(mode) => Ok(self.frame_count(mode)),
            None => Ok(self.inner.lock().unwrap().n_events()),
        }
    }

    #[getter]
    fn n_events(&self) -> usize {
        self.inner.lock().unwrap().n_events()
    }

    /// The per-slice representation name set at `open`/`with_repr`, or `None` (raw streams).
    #[getter]
    fn repr(&self) -> Option<&'static str> {
        self.repr.map(ReprSpec::name)
    }

    #[getter]
    fn sensor_size(&self) -> (usize, usize) {
        self.inner.lock().unwrap().sensor_size()
    }

    #[getter]
    fn time_span_ms(&self) -> (f64, f64) {
        let (lo, hi) = self.inner.lock().unwrap().time_span();
        (us_to_ms(lo), us_to_ms(hi))
    }

    /// The recording's `(first, last)` timestamp in `unit` — `"s"`, `"ms"` (the default), `"us"`,
    /// or `"ns"`. The unit-flexible form of `time_span_ms`.
    #[pyo3(signature = (unit="ms"))]
    fn time_span(&self, unit: &str) -> PyResult<(f64, f64)> {
        let scale = parse_unit_scale(unit)?;
        let (lo, hi) = self.inner.lock().unwrap().time_span();
        Ok((lo as f64 / scale, hi as f64 / scale))
    }

    #[getter]
    fn duration_ms(&self) -> f64 {
        let (lo, hi) = self.inner.lock().unwrap().time_span();
        us_to_ms(hi - lo)
    }

    /// How long the recording runs, in `unit` (`"s"`, `"ms"`, `"us"`, `"ns"`).
    #[pyo3(signature = (unit="ms"))]
    fn duration(&self, unit: &str) -> PyResult<f64> {
        let scale = parse_unit_scale(unit)?;
        let (lo, hi) = self.inner.lock().unwrap().time_span();
        Ok((hi - lo) as f64 / scale)
    }

    /// Number of fixed slices spanning the recording (requires `open(dt_ms=…)` or
    /// `open(max_events=…)`).
    #[getter]
    fn n_slices(&self) -> PyResult<usize> {
        let mode = self.require_mode()?;
        Ok(self.frame_count(mode))
    }

    /// The fixed slice duration set at `open`, or `None` if it was not given.
    #[getter]
    fn dt_ms(&self) -> Option<f64> {
        match self.mode {
            Some(SliceMode::Duration(dt)) => Some(us_to_ms(dt)),
            _ => None,
        }
    }

    /// The fixed slice duration set at `open`, in `unit` (`"s"`, `"ms"`, `"us"`, `"ns"`), or
    /// `None` if the reader is in `max_events` mode.
    #[pyo3(signature = (unit="ms"))]
    fn dt(&self, unit: &str) -> PyResult<Option<f64>> {
        let scale = parse_unit_scale(unit)?;
        Ok(match self.mode {
            Some(SliceMode::Duration(dt)) => Some(dt as f64 / scale),
            _ => None,
        })
    }

    /// The fixed events-per-slice set at `open(max_events=…)`, or `None` if it was not given.
    #[getter]
    fn max_events(&self) -> Option<usize> {
        match self.mode {
            Some(SliceMode::Count(len)) => Some(len),
            _ => None,
        }
    }

    /// The absolute framing origin (ms, the file's time base) set at `open(offset=…)`, or
    /// `None` if it was not given.
    #[getter]
    fn offset(&self) -> Option<f64> {
        self.offset_us.map(us_to_ms)
    }

    /// One slice. With a positional index `n` (requires `open(dt_ms=…)` or
    /// `open(max_events=…)`), returns the `n`-th fixed frame — the `dt_ms`-long window
    /// `[t_min + n·dt, t_min + (n+1)·dt)`, or the `max_events`-sized event chunk
    /// `[n·N, (n+1)·N)`; negative `n` counts from the end. Otherwise returns the half-open
    /// time window `[t0_ms, t1_ms)`, with omitted bounds extending to the recording's start /
    /// end. Bounds take any unit — `t0_s` / `t0_ms` / `t0_us` / `t0_ns`, and the same for `t1`.
    /// When the reader carries a representation (`open(repr=…)` / `with_repr`) the slice is
    /// rendered to that `EventFrame`; otherwise it is a raw `EventStream`.
    #[pyo3(signature = (index=None, *, t0_ms=None, t0_s=None, t0_us=None, t0_ns=None, t1_ms=None, t1_s=None, t1_us=None, t1_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn slice(
        &self,
        py: Python<'_>,
        index: Option<i64>,
        t0_ms: Option<f64>,
        t0_s: Option<f64>,
        t0_us: Option<f64>,
        t0_ns: Option<f64>,
        t1_ms: Option<f64>,
        t1_s: Option<f64>,
        t1_us: Option<f64>,
        t1_ns: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let t0 = resolve_us("t0", t0_s, t0_ms, t0_us, t0_ns)?;
        let t1 = resolve_us("t1", t1_s, t1_ms, t1_us, t1_ns)?;
        let window = if let Some(index) = index {
            if t0.is_some() || t1.is_some() {
                return Err(PyValueError::new_err(
                    "pass either a slice index or t0/t1 bounds, not both",
                ));
            }
            self.window_for_index(index)?
        } else {
            let (lo, hi) = self.inner.lock().unwrap().time_span();
            SliceWindow::Time(
                t0.map_or(lo, |us| us.round() as i64),
                // +1 keeps the last event
                t1.map_or(hi.saturating_add(1), |us| us.round() as i64),
            )
        };
        // An ad-hoc t0/t1 window is not a numbered slice, so it seeds from 0; an explicit index
        // seeds from that slice, matching `reader[n]`.
        self.fetch_slice(py, window, index.map_or(0, |index| self.slice_seed(index)))
    }

    /// `reader[n]` — the `n`-th fixed frame (requires `open(dt_ms=…)` or
    /// `open(max_events=…)`). Without a representation it returns the raw `EventStream`
    /// (like `slice(n)`); with one set (`open(repr=…)` / `with_repr`) it returns the dense
    /// `[C, H, W]` array, so a `torch.utils.data.DataLoader` can collate the reader directly.
    fn __getitem__(&self, py: Python<'_>, index: i64) -> PyResult<Py<PyAny>> {
        let window = self.window_for_index(index)?;
        let stream = self.fetch(py, window, self.slice_seed(index))?;
        match self.repr {
            Some(spec) => {
                let frame = py
                    .detach(|| spec.generate(&stream.inner))
                    .map_err(map_representation_error)?;
                Ok(frame_numpy(py, &frame).unbind())
            }
            None => Ok(Bound::new(py, stream)?.into_any().unbind()),
        }
    }

    /// Returns a new reader over the same file that renders each slice with `repr` (unset
    /// params take their method defaults). The dataset-mode counterpart of `open(repr=…)`,
    /// but with per-representation options: e.g. `reader.with_repr("voxel", bins=5)`.
    #[pyo3(signature = (repr, *, bins=None, window_ms=None, window_s=None, window_us=None, window_ns=None, tau_ms=None, tau_s=None, tau_us=None, tau_ns=None, max_window_ms=None, max_window_s=None, max_window_us=None, max_window_ns=None, window=None, normalize=None, pct=None, white_frame=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_repr(
        &self,
        repr: &str,
        bins: Option<i64>,
        window_ms: Option<f64>,
        window_s: Option<f64>,
        window_us: Option<f64>,
        window_ns: Option<f64>,
        tau_ms: Option<f64>,
        tau_s: Option<f64>,
        tau_us: Option<f64>,
        tau_ns: Option<f64>,
        max_window_ms: Option<f64>,
        max_window_s: Option<f64>,
        max_window_us: Option<f64>,
        max_window_ns: Option<f64>,
        window: Option<i64>,
        normalize: Option<bool>,
        pct: Option<f64>,
        white_frame: Option<bool>,
    ) -> PyResult<PyEventReader> {
        let spec = ReprSpec::new(
            repr,
            bins,
            resolve_ms("window", window_s, window_ms, window_us, window_ns)?,
            resolve_ms("tau", tau_s, tau_ms, tau_us, tau_ns)?,
            resolve_ms(
                "max_window",
                max_window_s,
                max_window_ms,
                max_window_us,
                max_window_ns,
            )?,
            window,
            normalize,
            pct,
            white_frame,
        )?;
        Ok(PyEventReader {
            inner: Arc::clone(&self.inner),
            mode: self.mode,
            repr: Some(spec),
            slice_ops: self.slice_ops.clone(),
            offset_us: self.offset_us,
            base_index: self.base_index,
            hot_pixel_mask: self.hot_pixel_mask.clone(),
        })
    }

    // ---- Deferred per-slice stream ops. Each returns a new reader that applies the matching
    // `EventStream` op to every slice (in chain order, before any `repr`), so a stream op used
    // on a reader stays lazy and composes with slice/windows/with_repr/batch:
    //   filtered = reader.hot_pixel_filter()   # reader whose slices drop hot pixels
    //   filtered.slice(500).count()            # count image for frame 500, hot pixels removed
    //   reader.flip_x().efast().with_repr("count")[500]   # chained, as a dense [C, H, W] array
    // These mirror the identically named `EventStream` methods 1:1 (see that block for details).

    /// Crops each slice to the `w`×`h` window at `(x0, y0)` (see [`EventStream::crop`]).
    fn crop(&self, x0: i64, y0: i64, w: usize, h: usize) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.crop(x0, y0, w, h)))
    }

    /// Mirrors each slice horizontally.
    fn flip_x(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.flip_x()))
    }

    /// Mirrors each slice vertically.
    fn flip_y(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.flip_y()))
    }

    /// Rotates each slice by `k * 90°` clockwise.
    fn rotate90(&self, k: i32) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.rotate90(k)))
    }

    /// Reflects each slice across the main diagonal.
    fn transpose(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.transpose()))
    }

    /// Translates each slice by `(dx, dy)`.
    fn translate(&self, dx: i64, dy: i64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.translate(dx, dy)))
    }

    /// Event-domain resize of each slice to a `width`×`height` grid.
    fn resize(&self, width: usize, height: usize) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.resize(width, height)))
    }

    /// Scales each slice's sensor by `(sx, sy)`.
    fn scale(&self, sx: f64, sy: f64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.scale(sx, sy)))
    }

    /// Applies a 2×3 affine matrix to each slice.
    fn warp_affine(&self, matrix: [[f64; 3]; 2]) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.warp_affine(matrix)))
    }

    /// Applies a 3×3 perspective matrix to each slice.
    fn warp_perspective(&self, matrix: [[f64; 3]; 3]) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.warp_perspective(matrix)))
    }

    /// Undistorts each slice with a `Camera`'s intrinsics + distortion.
    fn undistort(&self, camera: PyRef<'_, PyCamera>) -> PyEventReader {
        let camera = camera.inner;
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.undistort(&camera)))
    }

    /// Keeps each slice's events inside the region of interest — a `(H, W)` boolean mask (or 8-bit
    /// map, where non-zero keeps the pixel) covering this recording's sensor. Applied lazily to
    /// every slice, so `open(...).mask(roi)` reads a whole recording through the ROI.
    fn mask(&self, mask: &Bound<'_, PyAny>) -> PyResult<PyEventReader> {
        let (width, height) = self.sensor_size();
        let flat = parse_mask(mask, (width, height))?;
        Ok(self.with_slice_op(Arc::new(move |_, s: EventStream| s.mask(&flat, width, height))))
    }

    /// Keeps each slice's events whose timestamp lies in `[t0, t1)` — positionally in
    /// microseconds, or in any unit via `t0_s`/`t0_ms`/`t0_us`/`t0_ns` and the `t1` equivalents.
    #[pyo3(signature = (t0=None, t1=None, *, t0_s=None, t0_ms=None, t0_us=None, t0_ns=None, t1_s=None, t1_ms=None, t1_us=None, t1_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn time_window(
        &self,
        t0: Option<i64>,
        t1: Option<i64>,
        t0_s: Option<f64>,
        t0_ms: Option<f64>,
        t0_us: Option<f64>,
        t0_ns: Option<f64>,
        t1_s: Option<f64>,
        t1_ms: Option<f64>,
        t1_us: Option<f64>,
        t1_ns: Option<f64>,
    ) -> PyResult<PyEventReader> {
        let t0 = resolve_bound("t0", t0, t0_s, t0_ms, t0_us, t0_ns)?.unwrap_or(i64::MIN);
        let t1 = resolve_bound("t1", t1, t1_s, t1_ms, t1_us, t1_ns)?.unwrap_or(i64::MAX);
        Ok(self.with_slice_op(Arc::new(move |_, s: EventStream| s.time_window(t0, t1))))
    }

    /// Shifts each slice's timestamps by `dt` — positionally in microseconds, or in any unit via
    /// `dt_s`/`dt_ms`/`dt_us`/`dt_ns`.
    #[pyo3(signature = (dt=None, *, dt_s=None, dt_ms=None, dt_us=None, dt_ns=None))]
    fn time_shift(
        &self,
        dt: Option<i64>,
        dt_s: Option<f64>,
        dt_ms: Option<f64>,
        dt_us: Option<f64>,
        dt_ns: Option<f64>,
    ) -> PyResult<PyEventReader> {
        let dt = resolve_bound("dt", dt, dt_s, dt_ms, dt_us, dt_ns)?.unwrap_or(0);
        Ok(self.with_slice_op(Arc::new(move |_, s: EventStream| s.time_shift(dt))))
    }

    /// Scales each slice's timestamps by `factor`.
    fn time_scale(&self, factor: f64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.time_scale(factor)))
    }

    /// Shifts each slice's timestamps so its earliest event starts at zero.
    fn normalize_time(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.normalize_time()))
    }

    /// Renders every slice and writes them as one animation. Format comes from `path`'s
    /// extension: `.gif`, `.apng`/`.png`, or `.mp4` (which pipes to a system `ffmpeg`).
    ///
    /// Needs a representation — `open(repr=…)` or `with_repr(…)` — since a video is a sequence of
    /// rendered frames.
    ///
    /// `clim` fixes the value that maps to the top of the colormap, held across every frame. Left
    /// as `None`, it is calibrated from the first few slices and then held, because auto-contrast
    /// computed per frame makes the brightness pump visibly over a sequence. Pass `clim=0` to get
    /// the per-frame behaviour back, or an explicit value to match two videos to the same scale.
    #[pyo3(signature = (path, *, fps=30.0, colormap="viridis", clim=None, max_frames=None))]
    fn save_video(
        &self,
        py: Python<'_>,
        path: &str,
        fps: f64,
        colormap: &str,
        clim: Option<f64>,
        max_frames: Option<usize>,
    ) -> PyResult<usize> {
        use eventcv_core::video::{encoder_for, Fps};
        use eventcv_core::viz::{frame_extent, render_frame_scaled, Scale};

        let spec = self.repr.ok_or_else(|| {
            PyValueError::new_err(
                "save_video needs a representation; open(repr=…) or reader.with_repr(…)",
            )
        })?;
        let colormap = parse_colormap(colormap)?;
        let mode = self.require_mode()?;
        let total = self.frame_count(mode);
        let frames = max_frames.map_or(total, |limit| limit.min(total));
        if frames == 0 {
            return Err(PyValueError::new_err(
                "nothing to render: the reader has no slices",
            ));
        }

        // Calibrate one display scale for the whole sequence. Bounded to a handful of slices so a
        // multi-GB recording doesn't get read twice; the extent is a robust percentile, so a short
        // sample is representative unless the scene changes completely.
        let scale = match clim {
            Some(value) if value > 0.0 => Scale::Fixed(value),
            Some(_) => Scale::Auto, // clim=0 opts back in to per-frame auto-contrast
            None => {
                let mut extent = 0.0_f64;
                for index in 0..frames.min(CLIM_CALIBRATION_FRAMES) {
                    let stream = fetch_window(
                        py,
                        &self.inner,
                        self.window_for_index(index as i64)?,
                        index,
                        self.slice_ops.clone(),
                        self.hot_pixel_mask.clone(),
                    )?;
                    let frame = py
                        .detach(|| spec.generate(&stream))
                        .map_err(map_representation_error)?;
                    extent = extent.max(py.detach(|| frame_extent(&frame)));
                }
                if extent > 0.0 {
                    Scale::Fixed(extent)
                } else {
                    Scale::Auto
                }
            }
        };

        let mut encoder: Option<Box<dyn eventcv_core::video::AnimationEncoder + Send>> = None;
        for index in 0..frames {
            let stream = fetch_window(
                py,
                &self.inner,
                self.window_for_index(index as i64)?,
                index,
                self.slice_ops.clone(),
                self.hot_pixel_mask.clone(),
            )?;
            let image = py
                .detach(|| {
                    spec.generate(&stream)
                        .map(|frame| render_frame_scaled(&frame, colormap, scale))
                })
                .map_err(map_representation_error)?;
            // Opened on the first frame so the encoder learns the dimensions from real output
            // rather than from the sensor size, which a crop or resize op may have changed.
            let encoder = match encoder.as_mut() {
                Some(encoder) => encoder,
                None => encoder.insert(
                    encoder_for(
                        std::path::Path::new(path),
                        frames as u32,
                        image.width,
                        image.height,
                        Fps::new(fps),
                    )
                    .map_err(map_std_io_error)?,
                ),
            };
            encoder.write_frame(&image).map_err(map_std_io_error)?;
            py.check_signals()?; // a long render has to stay interruptible
        }
        if let Some(encoder) = encoder.take() {
            py.detach(|| encoder.finish()).map_err(map_std_io_error)?;
        }
        Ok(frames)
    }

    // --- Random augmentations -------------------------------------------------------------
    // Each op mixes `seed` with the index of the slice it is given, so slice `i` always augments
    // the same way — whether it is reached by `reader[i]`, by `batch()`, by iteration, or by
    // several DataLoader workers at once — while different slices still augment differently.

    /// Mirrors each slice left-right with probability `p`.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn random_flip_x(&self, p: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.random_flip_x(p, mix_seed(seed, i))
        }))
    }

    /// Mirrors each slice top-bottom with probability `p`.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn random_flip_y(&self, p: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.random_flip_y(p, mix_seed(seed, i))
        }))
    }

    /// Inverts every polarity in a slice with probability `p`.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn random_polarity_flip(&self, p: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.random_polarity_flip(p, mix_seed(seed, i))
        }))
    }

    /// Takes a random `width` × `height` crop of each slice.
    #[pyo3(signature = (width, height, *, seed=0))]
    fn random_crop(&self, width: usize, height: usize, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.random_crop(width, height, mix_seed(seed, i))
        }))
    }

    /// Drops each event in a slice independently with probability `p`.
    #[pyo3(signature = (p=0.1, *, seed=0))]
    fn event_drop(&self, p: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.event_drop(p, mix_seed(seed, i))
        }))
    }

    /// Silences a random `p` fraction of pixels in each slice.
    #[pyo3(signature = (p=0.1, *, seed=0))]
    fn pixel_dropout(&self, p: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.pixel_dropout(p, mix_seed(seed, i))
        }))
    }

    /// Jitters event positions in each slice by a gaussian of standard deviation `sigma` pixels.
    #[pyo3(signature = (sigma=1.0, *, seed=0))]
    fn spatial_jitter(&self, sigma: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.spatial_jitter(sigma, mix_seed(seed, i))
        }))
    }

    /// Jitters event timestamps in each slice, re-sorting afterwards.
    #[pyo3(signature = (sigma_ms=None, *, sigma_s=None, sigma_us=None, sigma_ns=None, seed=0))]
    fn time_jitter(
        &self,
        sigma_ms: Option<f64>,
        sigma_s: Option<f64>,
        sigma_us: Option<f64>,
        sigma_ns: Option<f64>,
        seed: u64,
    ) -> PyResult<PyEventReader> {
        let sigma = resolve_us("sigma", sigma_s, sigma_ms, sigma_us, sigma_ns)?.unwrap_or(1000.0);
        Ok(self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.time_jitter(sigma, mix_seed(seed, i))
        })))
    }

    /// Plays each slice backwards with probability `p`, inverting polarity to match.
    #[pyo3(signature = (p=0.5, *, seed=0))]
    fn time_reversal(&self, p: f64, seed: u64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |i, s: EventStream| {
            s.time_reversal(p, mix_seed(seed, i))
        }))
    }

    /// Keeps every `k`-th event of each slice.
    fn decimate(&self, k: usize) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.decimate(k)))
    }

    /// Keeps only events of the given polarity per slice (nonzero / `True` = ON).
    fn filter_polarity(&self, polarity: i64) -> PyEventReader {
        let polarity = polarity != 0;
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.filter_polarity(polarity)))
    }

    /// Flips every event's polarity, per slice.
    fn invert_polarity(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.invert_polarity()))
    }

    /// Reorders each slice by ascending timestamp (stable).
    fn sort_by_time(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.sort_by_time()))
    }

    /// Background-activity (nearest-neighbour) noise filter, per slice.
    fn background_activity_filter(&self, dt: i64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| {
            s.background_activity_filter(dt)
        }))
    }

    /// Refractory-period filter, per slice.
    fn refractory_filter(&self, dt: i64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.refractory_filter(dt)))
    }

    /// Hot-pixel removal applied to each slice (per-slice statistics; distinct from
    /// `open(hot_pixel_filter=True)`, which computes one mask over the whole recording).
    #[pyo3(signature = (n_std=3.0))]
    fn hot_pixel_filter(&self, n_std: f64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.hot_pixel_filter(n_std)))
    }

    /// Returns a new reader whose every slice is passed through [`EventStream::efast`], so the
    /// reader yields corner sub-streams (chain `.count()` / `with_repr` to visualise them).
    fn efast(&self) -> PyEventReader {
        self.with_slice_op(Arc::new(|_, s: EventStream| s.efast()))
    }

    /// Returns a new reader whose every slice is passed through [`EventStream::harris_corners`].
    #[pyo3(signature = (threshold=0.0))]
    fn harris_corners(&self, threshold: f64) -> PyEventReader {
        self.with_slice_op(Arc::new(move |_, s: EventStream| s.harris_corners(threshold)))
    }

    /// Renders slice `indices` into one dense `[B, C, H, W]` array — the explicit-batch path
    /// for training. Requires a representation (`open(repr=…)` / `with_repr`). `indices` is any
    /// int sequence (list / range / …); each is a `dt_ms` frame index.
    fn batch<'py>(&self, py: Python<'py>, indices: Vec<i64>) -> PyResult<Bound<'py, PyAny>> {
        let spec = self.repr.ok_or_else(|| {
            PyValueError::new_err(
                "batch needs a representation; open(repr=…) or reader.with_repr(…)",
            )
        })?;
        let mut frames = Vec::with_capacity(indices.len());
        for index in indices {
            let window = self.window_for_index(index)?;
            let stream = fetch_window(
                py,
                &self.inner,
                window,
                self.slice_seed(index),
                self.slice_ops.clone(),
                self.hot_pixel_mask.clone(),
            )?;
            let frame = py
                .detach(|| spec.generate(&stream))
                .map_err(map_representation_error)?;
            frames.push(frame);
        }
        // An empty batch still needs the frame's `[C, H, W]` + dtype: derive them from an
        // empty slice rendered with the same representation.
        let template;
        let template = match frames.first() {
            Some(frame) => frame,
            None => {
                let (width, height) = self.inner.lock().unwrap().sensor_size();
                let empty = EventStreamBuilder::new(width, height, 0.001).build();
                template = spec.generate(&empty).map_err(map_representation_error)?;
                &template
            }
        };
        stack_frames(py, &frames, template)
    }

    /// Converts the whole recording to `path` without loading it — slices are read in order and
    /// appended as they go, with any deferred ops (`crop`, `mask`, `hot_pixel_filter`, …) applied.
    ///
    /// Targets E2VID's interchange (`.zip`, or `format="e2vid"` for a `.txt`), which is what a
    /// multi-gigabyte recording usually needs converting *to*. Saving to a round-tripping format
    /// works from a slice: `eventcv.save(reader.slice(0), "frame.npz")`.
    #[pyo3(signature = (path, *, format=None))]
    fn save(&self, py: Python<'_>, path: &str, format: Option<String>) -> PyResult<()> {
        self.save_streaming(py, path, format.as_deref())
    }

    /// Events whose index lies in `[i0, i1)` (clamped to the file). Rendered to the reader's
    /// `EventFrame` when a representation is set (`open(repr=…)` / `with_repr`), else a raw
    /// `EventStream`.
    #[pyo3(signature = (i0, i1))]
    fn slice_count(&self, py: Python<'_>, i0: i64, i1: i64) -> PyResult<Py<PyAny>> {
        let i0 =
            usize::try_from(i0).map_err(|_| PyValueError::new_err("i0 must be non-negative"))?;
        let i1 =
            usize::try_from(i1).map_err(|_| PyValueError::new_err("i1 must be non-negative"))?;
        self.fetch_slice(py, SliceWindow::Index(i0, i1), 0)
    }

    /// Lazy iterator of consecutive windows: each is `[start, start + span_ms)` and
    /// `start` advances by `step_ms`. `step_ms` defaults to the `dt_ms` set at `open`
    /// (so `windows()` walks every `slice(n)`), and `span_ms` defaults to `step_ms`
    /// (non-overlapping). For a `max_events` reader, `windows()` without arguments walks
    /// the fixed-count slices instead. Streams a multi-GB file window-by-window without
    /// loading it. Each item is a rendered `EventFrame` when the reader carries a
    /// representation (`open(repr=…)` / `with_repr`), else a raw `EventStream`.
    #[pyo3(signature = (*, step_ms=None, step_s=None, step_us=None, step_ns=None, span_ms=None, span_s=None, span_us=None, span_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn windows(
        &self,
        step_ms: Option<f64>,
        step_s: Option<f64>,
        step_us: Option<f64>,
        step_ns: Option<f64>,
        span_ms: Option<f64>,
        span_s: Option<f64>,
        span_us: Option<f64>,
        span_ns: Option<f64>,
    ) -> PyResult<PyWindowIterator> {
        let step = resolve_duration_us("step", step_s, step_ms, step_us, step_ns)?;
        let span = resolve_duration_us("span", span_s, span_ms, span_us, span_ns)?;
        let guard = self.inner.lock().unwrap();
        let (lo, hi) = guard.time_span();
        let n_events = guard.n_events();
        drop(guard);
        if let (None, None, Some(SliceMode::Count(len))) = (step, span, self.mode) {
            return Ok(self.window_iterator(WindowCursor::Index {
                next: self.base_index,
                len,
                total: n_events,
            }));
        }
        let step_us = match step {
            Some(step) => step,
            None => match self.mode {
                Some(SliceMode::Duration(dt)) => dt,
                _ => {
                    return Err(PyValueError::new_err(
                        "windows() needs a step, or open the file with dt_ms",
                    ))
                }
            },
        };
        let span_us = span.unwrap_or(step_us);
        Ok(self.window_iterator(WindowCursor::Time {
            next: self.origin(lo),
            step_us,
            span_us,
            end: if n_events == 0 { i64::MIN } else { hi },
        }))
    }
}

impl PyEventReader {
    /// Converts the whole recording to `path` without materialising it: events are read in
    /// index chunks and appended as they arrive, with the reader's deferred ops and hot-pixel
    /// mask applied to each chunk exactly as `slice` would.
    ///
    /// Only the E2VID interchange streams today; the round-tripping formats are written from an
    /// `EventStream`, so those say so rather than quietly loading a file that may not fit.
    fn save_streaming(&self, py: Python<'_>, path: &str, format: Option<&str>) -> PyResult<()> {
        if !eventcv_core::io::is_e2vid_target(path, format) {
            return Err(PyValueError::new_err(format!(
                "saving a whole EventReader streams only to E2VID (.zip, or format=\"e2vid\"); \
                 for {path} save a slice — eventcv.save(reader.slice(0), {path:?})"
            )));
        }
        let total = self.inner.lock().unwrap().n_events();
        let mut writer = eventcv_core::io::E2vidWriter::create(path).map_err(map_io_error)?;
        let mut start = self.base_index;
        while start < total {
            let end = (start + EXPORT_CHUNK).min(total);
            let stream = fetch_window(
                py,
                &self.inner,
                SliceWindow::Index(start, end),
                // Export walks the recording in fixed chunks rather than slices, so the chunk
                // ordinal is the stable identifier here — the same export always augments the same.
                (start - self.base_index) / EXPORT_CHUNK,
                self.slice_ops.clone(),
                self.hot_pixel_mask.clone(),
            )?;
            py.detach(|| writer.append(&stream)).map_err(map_io_error)?;
            start = end;
            py.check_signals()?; // a multi-GB conversion has to stay interruptible
        }
        py.detach(|| writer.finish()).map_err(map_io_error)
    }

    /// Reads `window` off-GIL into an `EventStream` carrying the reader's stored repr,
    /// applying the per-slice op (e.g. `efast`) if one is set.
    fn fetch(&self, py: Python<'_>, window: SliceWindow, index: usize) -> PyResult<PyEventStream> {
        fetch_window(
            py,
            &self.inner,
            window,
            index,
            self.slice_ops.clone(),
            self.hot_pixel_mask.clone(),
        )
        .map(|inner| PyEventStream {
            inner,
            repr: self.repr,
        })
    }

    /// Reads `window` and wraps it for a public slice API: rendered to the reader's
    /// `EventFrame` when a representation is set, else the raw `EventStream`.
    fn fetch_slice(&self, py: Python<'_>, window: SliceWindow, index: usize) -> PyResult<Py<PyAny>> {
        let stream = fetch_window(
            py,
            &self.inner,
            window,
            index,
            self.slice_ops.clone(),
            self.hot_pixel_mask.clone(),
        )?;
        render_slice(py, self.repr, stream)
    }

    /// A new reader over the same file (and mode/`repr`) that applies `op` to every slice after
    /// this reader's existing deferred ops (so stream ops chain, e.g. `flip_x().hot_pixel_filter()`).
    fn with_slice_op(&self, op: SliceOp) -> PyEventReader {
        let mut ops = self.slice_ops.to_vec();
        ops.push(op);
        PyEventReader {
            inner: Arc::clone(&self.inner),
            mode: self.mode,
            repr: self.repr,
            slice_ops: Arc::from(ops),
            offset_us: self.offset_us,
            base_index: self.base_index,
            hot_pixel_mask: self.hot_pixel_mask.clone(),
        }
    }

    /// An iterator over this reader's file starting at `cursor` (shares ops/repr).
    fn window_iterator(&self, cursor: WindowCursor) -> PyWindowIterator {
        PyWindowIterator {
            reader: Arc::clone(&self.inner),
            cursor,
            emitted: 0,
            slice_ops: self.slice_ops.clone(),
            repr: self.repr,
            hot_pixel_mask: self.hot_pixel_mask.clone(),
        }
    }

    fn require_mode(&self) -> PyResult<SliceMode> {
        self.mode.ok_or_else(|| {
            PyValueError::new_err(
                "reader was opened without dt_ms or max_events; pass open(path, dt_ms=…) or open(path, max_events=…) to use integer slice indices",
            )
        })
    }

    /// The slice ordinal deferred ops should seed from, resolving a negative index the same way
    /// [`PyEventReader::window_for_index`] does so `reader[-1]` and `reader[n - 1]` — the same
    /// slice — augment identically. Out-of-range indices fall back to `0`; the fetch that follows
    /// raises the real `IndexError`, and this must not be the thing that reports it.
    fn slice_seed(&self, index: i64) -> usize {
        let total = match self.require_mode() {
            Ok(mode) => self.frame_count(mode) as i64,
            Err(_) => return 0,
        };
        let resolved = if index < 0 { index + total } else { index };
        usize::try_from(resolved).unwrap_or(0)
    }

    /// The window for the `index`-th slice under the reader's mode, validating the range
    /// (negative indices count back from the end). Framing starts at the offset origin
    /// (`t_min + offset` for duration mode, `base_index` for count mode).
    fn window_for_index(&self, index: i64) -> PyResult<SliceWindow> {
        let mode = self.require_mode()?;
        let guard = self.inner.lock().unwrap();
        let (n_events, (lo, _)) = (guard.n_events(), guard.time_span());
        drop(guard);
        let total = self.frame_count(mode) as i64;
        let resolved = if index < 0 { index + total } else { index };
        if resolved < 0 || resolved >= total {
            return Err(PyIndexError::new_err(format!(
                "slice index {index} out of range for {total} slices"
            )));
        }
        Ok(match mode {
            SliceMode::Duration(dt) => {
                let t0 = self.origin(lo) + resolved * dt;
                SliceWindow::Time(t0, t0 + dt)
            }
            SliceMode::Count(len) => {
                let i0 = self.base_index + resolved as usize * len;
                SliceWindow::Index(i0, (i0 + len).min(n_events))
            }
        })
    }

    /// Number of fixed frames under `mode`, measured from the offset origin: duration frames
    /// spanning `[t_min + offset, t_max]`, or count frames over the events at/after `base_index`.
    fn frame_count(&self, mode: SliceMode) -> usize {
        let guard = self.inner.lock().unwrap();
        let (lo, hi) = guard.time_span();
        let n_events = guard.n_events();
        drop(guard);
        match mode {
            SliceMode::Duration(dt) => {
                let origin = self.origin(lo);
                if n_events == 0 || origin > hi {
                    0
                } else {
                    ((hi - origin) / dt + 1) as usize
                }
            }
            SliceMode::Count(len) => n_events.saturating_sub(self.base_index).div_ceil(len),
        }
    }

    /// The absolute framing origin (µs) given the recording's `t_min` (`lo`): the offset
    /// timestamp clamped up to `t_min`, or `t_min` itself when no offset was set.
    fn origin(&self, lo: i64) -> i64 {
        self.offset_us.map_or(lo, |offset| offset.max(lo))
    }
}

/// Iterator returned by `EventReader.windows`; yields an `EventStream` per window.
#[pyclass]
struct PyWindowIterator {
    reader: Arc<Mutex<Reader>>,
    cursor: WindowCursor,
    /// How many windows have been yielded — the ordinal deferred ops seed from, so iterating a
    /// recording augments the same way indexing it does.
    emitted: usize,
    slice_ops: SliceOps,
    /// The reader's stored representation. When set (`open(repr=…)` / `with_repr`) each yielded
    /// window is rendered to that `EventFrame`; otherwise a raw `EventStream` is yielded.
    repr: Option<ReprSpec>,
    /// The reader's whole-recording hot-pixel mask, applied to each yielded window.
    hot_pixel_mask: Option<Arc<[bool]>>,
}

/// Walks a recording as consecutive time windows (`windows(step_ms=…)` / `dt_ms`
/// readers) or fixed-count event chunks (`max_events` readers).
enum WindowCursor {
    Time {
        next: i64,
        step_us: i64,
        span_us: i64,
        end: i64,
    },
    Index {
        next: usize,
        len: usize,
        total: usize,
    },
}

impl WindowCursor {
    /// The next window to fetch, advancing the cursor; `None` when exhausted.
    fn advance(&mut self) -> Option<SliceWindow> {
        match self {
            Self::Time {
                next,
                step_us,
                span_us,
                end,
            } => {
                if *next > *end {
                    return None;
                }
                let t0 = *next;
                *next = next.saturating_add(*step_us);
                Some(SliceWindow::Time(t0, t0.saturating_add(*span_us)))
            }
            Self::Index { next, len, total } => {
                if *next >= *total {
                    return None;
                }
                let i0 = *next;
                *next = i0 + *len;
                Some(SliceWindow::Index(i0, (i0 + *len).min(*total)))
            }
        }
    }
}

#[pymethods]
impl PyWindowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(window) = self.cursor.advance() else {
            return Ok(None);
        };
        let index = self.emitted;
        self.emitted += 1;
        let stream = fetch_window(
            py,
            &self.reader,
            window,
            index,
            self.slice_ops.clone(),
            self.hot_pixel_mask.clone(),
        )?;
        render_slice(py, self.repr, stream).map(Some)
    }
}

/// Saves an `EventStream`, `EventFrame`, or `EventReader` to `path` (format by extension) — the
/// OpenCV-style free-function form of `obj.save(path)`. `topic` names the rosbag connection;
/// `colormap` / `normalize` apply only to `.png` frame export; `format` overrides the extension
/// (`"e2vid"` writes E2VID's `t x y p` seconds layout to a `.txt`, which `.zip` already implies).
///
/// An `EventReader` is converted **window by window**, so a recording far larger than memory can
/// be re-exported without loading it.
#[pyfunction]
#[pyo3(signature = (obj, path, *, topic=None, colormap="viridis", normalize=true, format=None))]
fn save(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    path: &str,
    topic: Option<String>,
    colormap: &str,
    normalize: bool,
    format: Option<String>,
) -> PyResult<()> {
    if let Ok(stream) = obj.extract::<PyRef<PyEventStream>>() {
        let options = SaveOptions {
            topic,
            format,
            ..SaveOptions::default()
        };
        let inner = &stream.inner; // capture a plain &EventStream, not the GIL-bound PyRef
        return py
            .detach(|| eventcv_core::io::save_stream(path, inner, &options))
            .map_err(map_io_error);
    }
    if let Ok(frame) = obj.extract::<PyRef<PyEventFrame>>() {
        let options = SaveOptions {
            colormap: parse_colormap(colormap)?,
            normalize: Some(normalize),
            format,
            ..SaveOptions::default()
        };
        let inner = &frame.inner;
        return py
            .detach(|| eventcv_core::io::save_frame(path, inner, &options))
            .map_err(map_io_error);
    }
    if let Ok(reader) = obj.extract::<PyRef<PyEventReader>>() {
        return reader.save_streaming(py, path, format.as_deref());
    }
    Err(PyTypeError::new_err(
        "save expects an EventStream, EventFrame, or EventReader",
    ))
}

/// Reads an `EventFrame` previously written by `frame.save(...)` / `eventcv.save(...)`
/// (`.npz` or `.h5`), restoring its dtype, `kind`, and `channel_names`.
/// Simulates a DVS camera watching `source`, returning the events it would have produced.
///
/// `source` is a video file path, or an array of intensity frames — `[N, H, W]` greyscale or
/// `[N, H, W, 3]` RGB, any dtype, with values read as `[0, 255]` for integers and `[0, 1]` for
/// floats. Frames need an explicit `fps`; a video file carries its own.
#[pyfunction]
#[pyo3(signature = (
    source, *, pos_thres=0.2, neg_thres=0.2, sigma_thres=0.03, refractory_us=100,
    cutoff_hz=200.0, leak_rate_hz=1.0, shot_noise_rate_hz=10.0, seed=0,
    upsample=None, max_events_per_pixel=1.0, fps=None, scale=None, max_frames=None
))]
#[allow(clippy::too_many_arguments)]
fn simulate(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    pos_thres: f32,
    neg_thres: f32,
    sigma_thres: f32,
    refractory_us: i64,
    cutoff_hz: f32,
    leak_rate_hz: f32,
    shot_noise_rate_hz: f32,
    seed: u64,
    upsample: Option<&str>,
    max_events_per_pixel: f32,
    fps: Option<f64>,
    scale: Option<(usize, usize)>,
    max_frames: Option<usize>,
) -> PyResult<PyEventStream> {
    use eventcv_core::simulate::{simulate_video, Simulator, SimulatorConfig, Upsample};

    let upsample = match upsample {
        None | Some("adaptive") => Upsample::Adaptive {
            max_events_per_pixel,
        },
        Some("off") | Some("none") => Upsample::Off,
        Some(other) => match other.parse::<usize>() {
            Ok(n) => Upsample::Fixed(n),
            Err(_) => {
                return Err(PyValueError::new_err(format!(
                    "upsample must be \"adaptive\", \"off\", or an integer factor, not {other:?}"
                )))
            }
        },
    };
    let config = SimulatorConfig {
        pos_thres,
        neg_thres,
        sigma_thres,
        refractory_us,
        cutoff_hz,
        leak_rate_hz,
        shot_noise_rate_hz,
        seed,
        upsample,
    };

    // Collecting the per-interval streams and concatenating once at the end keeps the simulator
    // itself streaming; only this convenience wrapper materialises the whole recording, which is
    // what a caller asking for one EventStream has asked for.
    let mut chunks: Vec<EventStream> = Vec::new();

    if let Ok(path) = source.extract::<String>() {
        py.detach(|| {
            simulate_video(
                std::path::Path::new(&path),
                config,
                scale,
                max_frames,
                |stream| {
                    chunks.push(stream);
                    Ok(())
                },
            )
        })
        .map_err(map_std_io_error)?;
    } else {
        let frames = source
            .call_method1("astype", ("float32",))
            .unwrap_or_else(|_| source.clone())
            .extract::<numpy::PyReadonlyArrayDyn<f32>>()
            .map_err(|_| {
                PyValueError::new_err(
                    "simulate takes a video path or an array of frames ([N,H,W] or [N,H,W,3])",
                )
            })?;
        let array = frames.as_array();
        let shape = array.shape().to_vec();
        let (count, height, width, channels) = match shape.as_slice() {
            [n, h, w] => (*n, *h, *w, 1),
            [n, h, w, c] if *c == 1 || *c == 3 => (*n, *h, *w, *c),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "frames must be [N,H,W] or [N,H,W,3], got {shape:?}"
                )))
            }
        };
        let fps = fps.ok_or_else(|| {
            PyValueError::new_err("simulating from frames needs fps= (a video file carries its own)")
        })?;
        if !(fps.is_finite() && fps > 0.0) {
            return Err(PyValueError::new_err("fps must be finite and positive"));
        }
        // Integer-valued input is 0..255; float input is already 0..1. Deciding per array rather
        // than per frame keeps the scale consistent across a recording that happens to start dark.
        let peak = array.iter().copied().fold(0.0_f32, f32::max);
        let divisor = if peak > 1.0 { 255.0 } else { 1.0 };
        let us_per_frame = (1_000_000.0 / fps).round() as i64;
        let values: Vec<f32> = array.iter().copied().collect();
        drop(frames);

        py.detach(|| {
            let mut simulator = Simulator::new(width, height, config);
            let plane = width * height;
            let mut luma = vec![0.0_f32; plane];
            let limit = max_frames.map_or(count, |limit| limit.min(count));
            for index in 0..limit {
                let base = index * plane * channels;
                for pixel in 0..plane {
                    luma[pixel] = if channels == 3 {
                        let offset = base + pixel * 3;
                        eventcv_core::simulate::linear_luma(
                            (values[offset] / divisor * 255.0).clamp(0.0, 255.0) as u8,
                            (values[offset + 1] / divisor * 255.0).clamp(0.0, 255.0) as u8,
                            (values[offset + 2] / divisor * 255.0).clamp(0.0, 255.0) as u8,
                        )
                    } else {
                        // Greyscale goes through the same sRGB transfer as colour — the weights
                        // sum to one, so this is exactly "linearise" and nothing else. Skipping it
                        // here would make a grey scene simulate differently depending on whether it
                        // arrived as one channel or three.
                        let level = (values[base + pixel] / divisor * 255.0).clamp(0.0, 255.0) as u8;
                        eventcv_core::simulate::linear_luma(level, level, level)
                    };
                }
                let stream = simulator.push_frame(&luma, index as i64 * us_per_frame);
                if !stream.is_empty() {
                    chunks.push(stream);
                }
            }
        });
    }

    // `concat` takes references and preserves order, so the globally-sorted invariant the simulator
    // guarantees per interval survives the join.
    let inner = py.detach(|| match chunks.split_first() {
        Some((first, rest)) => first.concat(&rest.iter().collect::<Vec<_>>()),
        None => EventStreamBuilder::new(
            scale.map_or(0, |(w, _)| w),
            scale.map_or(0, |(_, h)| h),
            0.001,
        )
        .build(),
    });
    Ok(PyEventStream { inner, repr: None })
}

/// Reconstructs an intensity video from events by running a trained model over each slice.
///
/// `reader` must carry the representation the model expects (`open(repr=…)` / `with_repr`), and
/// `model` is any ONNX graph taking that representation and returning a single-channel image —
/// E2VID and its relatives. EventCV supplies the runner and the tensors, not the weights.
///
/// Writes to `path` in whatever format the extension names (`.gif`, `.apng`, `.mp4`) and returns
/// the number of frames written.
#[cfg(feature = "onnx")]
#[pyfunction]
#[pyo3(signature = (reader, model, path, *, fps=30.0, colormap="grayscale", max_frames=None))]
fn reconstruct(
    py: Python<'_>,
    reader: PyRef<'_, PyEventReader>,
    model: PyRef<'_, PyModel>,
    path: &str,
    fps: f64,
    colormap: &str,
    max_frames: Option<usize>,
) -> PyResult<usize> {
    use eventcv_core::representation::{EventFrame, EventFrameData};
    use eventcv_core::video::{encoder_for, Fps};
    use eventcv_core::viz::{render_frame_scaled, Scale};

    let spec = reader.repr.ok_or_else(|| {
        PyValueError::new_err(
            "reconstruct needs the representation the model expects; open(repr=…) or with_repr(…)",
        )
    })?;
    let colormap = parse_colormap(colormap)?;
    let mode = reader.require_mode()?;
    let total = reader.frame_count(mode);
    let frames = max_frames.map_or(total, |limit| limit.min(total));
    if frames == 0 {
        return Err(PyValueError::new_err(
            "nothing to reconstruct: the reader has no slices",
        ));
    }

    // Capture plain references, not the GIL-bound PyRefs: `py.detach` needs a Send closure.
    let session = &model.inner;
    let mut encoder: Option<Box<dyn eventcv_core::video::AnimationEncoder + Send>> = None;
    for index in 0..frames {
        let stream = fetch_window(
            py,
            &reader.inner,
            reader.window_for_index(index as i64)?,
            index,
            reader.slice_ops.clone(),
            reader.hot_pixel_mask.clone(),
        )?;
        let input = py
            .detach(|| spec.generate(&stream))
            .map_err(map_representation_error)?;
        let (channels, height, width) = input.shape();
        let tensor = frame_to_f32_array(&input, channels, height, width);
        let mut outputs = py
            .detach(|| {
                session
                    .lock()
                    .expect("model mutex was poisoned by a previous panic")
                    .run(tensor)
            })
            .map_err(map_model_error)?;
        let output = outputs
            .drain(..)
            .next()
            .ok_or_else(|| PyValueError::new_err("the model produced no output"))?;

        // Reconstruction models emit [N,1,H,W] or [1,H,W]; either way the trailing H*W is the
        // image, so the leading axes are collapsed rather than demanding one exact rank.
        let shape = output.shape().to_vec();
        let (out_h, out_w) = match shape.as_slice() {
            [.., h, w] => (*h, *w),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "expected an image output with at least two dimensions, got {shape:?}"
                )))
            }
        };
        let samples: Vec<f32> = output.iter().copied().take(out_h * out_w).collect();
        let image_frame = EventFrame::intensity(EventFrameData::F32(samples), out_w, out_h)
            .map_err(map_representation_error)?;
        // Natural scale: a reconstruction already lives in [0, 1], and auto-contrast would make
        // the brightness pump across the sequence exactly as it does for rendered representations.
        let image = py.detach(|| render_frame_scaled(&image_frame, colormap, Scale::Natural));

        let encoder = match encoder.as_mut() {
            Some(encoder) => encoder,
            None => encoder.insert(
                encoder_for(
                    std::path::Path::new(path),
                    frames as u32,
                    image.width,
                    image.height,
                    Fps::new(fps),
                )
                .map_err(map_std_io_error)?,
            ),
        };
        encoder.write_frame(&image).map_err(map_std_io_error)?;
        py.check_signals()?;
    }
    if let Some(encoder) = encoder.take() {
        py.detach(|| encoder.finish()).map_err(map_std_io_error)?;
    }
    Ok(frames)
}

/// Copies a frame into the `[1, C, H, W]` float tensor a model expects, adding the batch axis the
/// graph almost certainly declares.
#[cfg(feature = "onnx")]
fn frame_to_f32_array(
    frame: &eventcv_core::representation::EventFrame,
    channels: usize,
    height: usize,
    width: usize,
) -> numpy::ndarray::ArrayD<f32> {
    use eventcv_core::representation::EventFrameData;
    let values: Vec<f32> = match frame.data() {
        EventFrameData::U8(data) => data.iter().map(|&v| f32::from(v)).collect(),
        EventFrameData::U16(data) => data.iter().map(|&v| f32::from(v)).collect(),
        EventFrameData::U64(data) => data.iter().map(|&v| v as f32).collect(),
        EventFrameData::F32(data) => data.clone(),
    };
    numpy::ndarray::ArrayD::from_shape_vec(
        numpy::ndarray::IxDyn(&[1, channels, height, width]),
        values,
    )
    .expect("frame shape matches its data by construction")
}

#[pyfunction]
fn load_frame(py: Python<'_>, path: &str) -> PyResult<PyEventFrame> {
    py.detach(|| eventcv_core::io::load_frame(path))
        .map(|inner| PyEventFrame { inner })
        .map_err(map_io_error)
}

/// The `(width, height)` sensor an ROI mask is built for — the same `sensor_size` order the rest
/// of eventcv uses, even though the mask itself comes back as `(H, W)` to match NumPy.
fn parse_mask_size(sensor_size: (i64, i64)) -> PyResult<(usize, usize)> {
    match parse_sensor_size(Some(sensor_size))? {
        Some((width, height)) if width > 0 && height > 0 => Ok((width, height)),
        _ => Err(PyValueError::new_err(
            "sensor_size must be (width, height), both positive",
        )),
    }
}

/// Builds the ROI mask keeping the `width`×`height` rectangle at `(x0, y0)`.
#[pyfunction]
fn rect_mask(
    py: Python<'_>,
    sensor_size: (i64, i64),
    x0: f64,
    y0: f64,
    width: f64,
    height: f64,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    let (w, h) = parse_mask_size(sensor_size)?;
    mask_to_py(
        py,
        eventcv_core::mask::rect(w, h, x0, y0, width, height),
        w,
        h,
    )
}

/// Builds the ROI mask keeping the ellipse centred on `(cx, cy)` with semi-axes `rx`, `ry`.
#[pyfunction]
fn ellipse_mask(
    py: Python<'_>,
    sensor_size: (i64, i64),
    cx: f64,
    cy: f64,
    rx: f64,
    ry: f64,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    let (w, h) = parse_mask_size(sensor_size)?;
    mask_to_py(py, eventcv_core::mask::ellipse(w, h, cx, cy, rx, ry), w, h)
}

/// Builds the ROI mask keeping the circle of radius `r` centred on `(cx, cy)`.
#[pyfunction]
fn circle_mask(
    py: Python<'_>,
    sensor_size: (i64, i64),
    cx: f64,
    cy: f64,
    r: f64,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    let (w, h) = parse_mask_size(sensor_size)?;
    mask_to_py(py, eventcv_core::mask::ellipse(w, h, cx, cy, r, r), w, h)
}

/// Builds the ROI mask keeping the interior of the closed polygon through `points`
/// (`[(x, y), …]`, by the even-odd rule).
#[pyfunction]
fn polygon_mask(
    py: Python<'_>,
    sensor_size: (i64, i64),
    points: Vec<(f64, f64)>,
) -> PyResult<Bound<'_, PyArray2<bool>>> {
    let (w, h) = parse_mask_size(sensor_size)?;
    mask_to_py(py, eventcv_core::mask::polygon(w, h, &points), w, h)
}

/// Writes an ROI mask to an 8-bit greyscale `.png` (white keeps, black drops).
#[pyfunction]
fn save_mask(py: Python<'_>, mask: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    let (flat, width, height) = mask_array(mask)?;
    py.detach(|| eventcv_core::io::write_mask(path, &flat, width, height))
        .map_err(map_io_error)
}

/// Reads an ROI mask from a `.png` as an `(H, W)` boolean array — a pixel is kept where the image
/// is non-black and not fully transparent, so a mask binarised in any tool loads as-is.
#[pyfunction]
fn load_mask<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (mask, width, height) = py
        .detach(|| eventcv_core::io::read_mask(path))
        .map_err(map_io_error)?;
    mask_to_py(py, mask, width, height)
}

/// Streams `EventFrame`s into one extendable `[N, C, H, W]` HDF5 dataset — for writing a
/// representation window-by-window over a whole recording. The dtype and `[C, H, W]` shape
/// are fixed by the first appended frame.
#[cfg(feature = "hdf5")]
#[pyclass(name = "FrameSink")]
struct PyFrameSink {
    inner: Option<eventcv_core::io::Hdf5FrameSink>,
}

#[cfg(feature = "hdf5")]
#[pymethods]
impl PyFrameSink {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = eventcv_core::io::Hdf5FrameSink::open(path).map_err(map_io_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Appends one frame as the next slice; every frame must match the first's dtype + shape.
    fn append(&mut self, frame: PyRef<PyEventFrame>) -> PyResult<()> {
        let sink = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("FrameSink is already closed"))?;
        sink.append(&frame.inner).map_err(map_io_error)
    }

    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.as_ref().map_or(0, |sink| sink.n_frames())
    }

    /// Flushes and closes the file; further appends raise. Idempotent.
    fn finish(&mut self) -> PyResult<()> {
        match self.inner.take() {
            Some(sink) => sink.finish().map_err(map_io_error),
            None => Ok(()),
        }
    }
}

/// Streams `EventStream` windows into extendable `events/{x,y,t,p}` HDF5 datasets — the event-level
/// twin of `FrameSink`. Where `FrameSink` appends computed representations, this appends the raw
/// events, so a live camera (or any window source) can be recorded to disk continuously without
/// holding the whole session in memory. The sensor size and time base are taken from the first
/// appended stream, and the file reads straight back with `eventcv.open` / `eventcv.load`.
#[cfg(feature = "hdf5")]
#[pyclass(name = "EventSink")]
struct PyEventSink {
    inner: Option<eventcv_core::io::Hdf5EventSink>,
}

#[cfg(feature = "hdf5")]
#[pymethods]
impl PyEventSink {
    /// Opens `path` for writing. `compression` is an optional gzip level (`0..=9`); omit it (the
    /// default) for uncompressed columns and the fastest writes.
    #[new]
    #[pyo3(signature = (path, *, compression=None))]
    fn new(path: &str, compression: Option<u8>) -> PyResult<Self> {
        if let Some(level) = compression {
            if level > 9 {
                return Err(PyValueError::new_err(
                    "compression must be a gzip level in 0..=9",
                ));
            }
        }
        let inner =
            eventcv_core::io::Hdf5EventSink::open(path, compression).map_err(map_io_error)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Appends one `EventStream` window to the end of the file. Empty windows are a no-op.
    fn append(&mut self, stream: PyRef<PyEventStream>) -> PyResult<()> {
        let sink = self
            .inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("EventSink is already closed"))?;
        sink.append(&stream.inner).map_err(map_io_error)
    }

    #[getter]
    fn n_events(&self) -> usize {
        self.inner.as_ref().map_or(0, |sink| sink.n_events())
    }

    /// Forces buffered events out to disk without closing, so a crash mid-recording keeps
    /// everything appended so far. Call periodically during a long capture.
    fn flush(&self) -> PyResult<()> {
        match self.inner.as_ref() {
            Some(sink) => sink.flush().map_err(map_io_error),
            None => Ok(()),
        }
    }

    /// Flushes and closes the file; further appends raise. Idempotent.
    fn finish(&mut self) -> PyResult<()> {
        match self.inner.take() {
            Some(sink) => sink.finish().map_err(map_io_error),
            None => Ok(()),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.finish()?;
        Ok(false) // never suppress an exception raised in the `with` body
    }
}

fn map_io_error(error: IoError) -> PyErr {
    match error {
        IoError::Io(error) if error.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(error.to_string())
        }
        IoError::Io(error) => PyOSError::new_err(error.to_string()),
        IoError::Parse { .. }
        | IoError::Format(_)
        | IoError::InvalidSensorSize
        | IoError::Unsupported(_) => PyValueError::new_err(error.to_string()),
    }
}

/// The animation encoders return plain `std::io::Error` rather than the reader/writer `IoError`,
/// so they get their own mapper. Missing-file becomes `FileNotFoundError`, which is also what a
/// missing `ffmpeg` surfaces as — its message already names the fix.
fn map_std_io_error(error: std::io::Error) -> PyErr {
    if error.kind() == std::io::ErrorKind::NotFound {
        PyFileNotFoundError::new_err(error.to_string())
    } else if error.kind() == std::io::ErrorKind::InvalidInput {
        PyValueError::new_err(error.to_string())
    } else {
        PyOSError::new_err(error.to_string())
    }
}

/// Maps the Python model name onto a warp, checking that rotation was given the intrinsics it needs.
fn parse_warp_model(
    name: &str,
    camera: Option<&PyCamera>,
) -> PyResult<eventcv_core::cmax::WarpModel> {
    use eventcv_core::cmax::WarpModel;
    match name {
        "translation" => Ok(WarpModel::Translation),
        "rotation" => {
            let camera = camera.ok_or_else(|| {
                PyValueError::new_err(
                    "the rotation model needs camera= intrinsics to map pixels onto rays",
                )
            })?;
            Ok(WarpModel::Rotation {
                camera: camera.inner,
            })
        }
        other => Err(PyValueError::new_err(format!(
            "model must be \"translation\" or \"rotation\", not {other:?}"
        ))),
    }
}

fn parse_objective(name: &str) -> PyResult<eventcv_core::cmax::Objective> {
    use eventcv_core::cmax::Objective;
    match name {
        "variance" => Ok(Objective::Variance),
        "sos" | "sum_of_squares" => Ok(Objective::SumOfSquares),
        "soe" | "sum_of_exponentials" => Ok(Objective::SumOfExponentials),
        other => Err(PyValueError::new_err(format!(
            "objective must be \"variance\", \"sos\" or \"soe\", not {other:?}"
        ))),
    }
}

fn map_cmax_error(error: eventcv_core::cmax::CmaxError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Follows connected components across frames, keeping their identities.
///
/// Connected-component labels are assigned in scan order and renumber whenever anything moves, so
/// they mean nothing across time. A tracker adds the memory.
#[pyclass(name = "Tracker")]
struct PyTracker {
    inner: eventcv_core::track::Tracker,
}

#[pymethods]
impl PyTracker {
    /// `min_area` filters noise (a hot pixel makes a one-pixel component), `max_distance` is the
    /// gate beyond which a blob is a different object, and `max_missed` is how many frames a track
    /// survives unmatched — what carries it through a brief occlusion.
    #[new]
    #[pyo3(signature = (*, connectivity=8, min_area=4, max_distance=20.0, max_missed=3))]
    fn new(connectivity: u8, min_area: usize, max_distance: f64, max_missed: usize) -> PyResult<Self> {
        if connectivity != 4 && connectivity != 8 {
            return Err(PyValueError::new_err("connectivity must be 4 or 8"));
        }
        Ok(Self {
            inner: eventcv_core::track::Tracker::new(eventcv_core::track::TrackerConfig {
                connectivity,
                min_area,
                max_distance,
                max_missed,
            }),
        })
    }

    /// Segments `frame`, matches its blobs to the existing tracks, and returns the live tracks as
    /// a list of dicts: `id`, `centroid`, `velocity`, `area`, `age`, `missed`.
    fn update<'py>(
        &mut self,
        py: Python<'py>,
        frame: PyRef<'_, PyEventFrame>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let inner = &frame.inner;
        let tracks = py
            .detach(|| self.inner.update(inner).map(|tracks| tracks.to_vec()))
            .map_err(map_cluster_error)?;
        tracks.iter().map(|track| track_to_dict(py, track)).collect()
    }

    /// The live tracks without advancing the tracker.
    #[getter]
    fn tracks<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .tracks()
            .iter()
            .map(|track| track_to_dict(py, track))
            .collect()
    }

    /// Forgets every track. Ids continue from where they left off, so an old track and a new one
    /// can never be confused in a log.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!("Tracker({} live)", self.inner.tracks().len())
    }
}

fn track_to_dict<'py>(
    py: Python<'py>,
    track: &eventcv_core::track::Track,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", track.id)?;
    dict.set_item("centroid", track.centroid)?;
    dict.set_item("velocity", track.velocity)?;
    dict.set_item("area", track.area)?;
    dict.set_item("age", track.age)?;
    dict.set_item("missed", track.missed)?;
    Ok(dict)
}

/// Sends events over UDP, wire-compatible with aestream and SPIF.
#[pyclass(name = "UdpSender")]
struct PyUdpSender {
    inner: eventcv_core::net::UdpSender,
}

#[pymethods]
impl PyUdpSender {
    /// `host_endian` defaults to True to match aestream, whose packing is host-endian despite its
    /// own comments saying it should not be. Set it False for correct network byte order.
    #[new]
    #[pyo3(signature = (target, *, timestamps=false, host_endian=true))]
    fn new(target: &str, timestamps: bool, host_endian: bool) -> PyResult<Self> {
        let format = eventcv_core::net::WireFormat {
            timestamps,
            host_endian,
        };
        eventcv_core::net::UdpSender::connect(target, format)
            .map(|inner| Self { inner })
            .map_err(map_std_io_error)
    }

    /// Sends every event, split across datagrams. Returns how many were sent.
    fn send(&self, py: Python<'_>, stream: PyRef<'_, PyEventStream>) -> PyResult<usize> {
        let inner = &stream.inner;
        py.detach(|| self.inner.send(inner)).map_err(map_std_io_error)
    }

    #[getter]
    fn address(&self) -> PyResult<String> {
        self.inner
            .local_addr()
            .map(|addr| addr.to_string())
            .map_err(map_std_io_error)
    }
}

/// Receives events over UDP.
#[pyclass(name = "UdpReceiver")]
struct PyUdpReceiver {
    inner: eventcv_core::net::UdpReceiver,
}

#[pymethods]
impl PyUdpReceiver {
    #[new]
    #[pyo3(signature = (address, sensor_size, *, timestamps=false, host_endian=true))]
    fn new(
        address: &str,
        sensor_size: (usize, usize),
        timestamps: bool,
        host_endian: bool,
    ) -> PyResult<Self> {
        let format = eventcv_core::net::WireFormat {
            timestamps,
            host_endian,
        };
        eventcv_core::net::UdpReceiver::bind(address, sensor_size.0, sensor_size.1, format)
            .map(|inner| Self { inner })
            .map_err(map_std_io_error)
    }

    /// Collects whatever arrives within `dt_ms`. An empty stream means silence, not failure.
    #[pyo3(signature = (dt_ms=30.0))]
    fn recv(&self, py: Python<'_>, dt_ms: f64) -> PyResult<PyEventStream> {
        if !(dt_ms.is_finite() && dt_ms > 0.0) {
            return Err(PyValueError::new_err("dt_ms must be finite and positive"));
        }
        let window = std::time::Duration::from_micros((dt_ms * 1000.0) as u64);
        py.detach(|| self.inner.recv_window(window))
            .map(|inner| PyEventStream { inner, repr: None })
            .map_err(map_std_io_error)
    }

    #[getter]
    fn address(&self) -> PyResult<String> {
        self.inner
            .local_addr()
            .map(|addr| addr.to_string())
            .map_err(map_std_io_error)
    }
}

fn map_representation_error(error: RepresentationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn map_flow_error(error: FlowError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn map_cluster_error(error: ClusterError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Copies an [`EventFrame`] into an owned `[C, H, W]` numpy array of its native dtype —
/// shared by `EventFrame.numpy()` and the reader's dense dataset path.
fn frame_numpy<'py>(py: Python<'py>, frame: &EventFrame) -> Bound<'py, PyAny> {
    let shape = frame.shape();
    match frame.data() {
        EventFrameData::U8(data) => numpy::ndarray::Array3::from_shape_vec(shape, data.clone())
            .expect("EventFrame shape must match its data")
            .into_pyarray(py)
            .into_any(),
        EventFrameData::U16(data) => numpy::ndarray::Array3::from_shape_vec(shape, data.clone())
            .expect("EventFrame shape must match its data")
            .into_pyarray(py)
            .into_any(),
        EventFrameData::U64(data) => numpy::ndarray::Array3::from_shape_vec(shape, data.clone())
            .expect("EventFrame shape must match its data")
            .into_pyarray(py)
            .into_any(),
        EventFrameData::F32(data) => numpy::ndarray::Array3::from_shape_vec(shape, data.clone())
            .expect("EventFrame shape must match its data")
            .into_pyarray(py)
            .into_any(),
    }
}

/// Stacks equally-shaped frames into one owned `[B, C, H, W]` numpy array. `template`
/// supplies the shape + dtype when `frames` is empty (yielding a `[0, C, H, W]` array).
fn stack_frames<'py>(
    py: Python<'py>,
    frames: &[EventFrame],
    template: &EventFrame,
) -> PyResult<Bound<'py, PyAny>> {
    let (channels, height, width) = template.shape();
    let batch = frames.len();
    let shape = (batch, channels, height, width);
    for frame in frames {
        if frame.shape() != (channels, height, width) {
            return Err(PyValueError::new_err(
                "cannot batch frames of differing shapes",
            ));
        }
    }

    // Concatenate each frame's contiguous plane data in order; `Array4` reads it as `[B,C,H,W]`.
    macro_rules! stack {
        ($variant:path, $len:expr) => {{
            let mut data = Vec::with_capacity(batch * $len);
            for frame in frames {
                let $variant(values) = frame.data() else {
                    return Err(PyValueError::new_err(
                        "cannot batch frames of differing dtypes",
                    ));
                };
                data.extend_from_slice(values);
            }
            numpy::ndarray::Array4::from_shape_vec(shape, data)
                .expect("batch shape must match its data")
                .into_pyarray(py)
                .into_any()
        }};
    }

    let plane = channels * height * width;
    Ok(match template.data() {
        EventFrameData::U8(_) => stack!(EventFrameData::U8, plane),
        EventFrameData::U16(_) => stack!(EventFrameData::U16, plane),
        EventFrameData::U64(_) => stack!(EventFrameData::U64, plane),
        EventFrameData::F32(_) => stack!(EventFrameData::F32, plane),
    })
}

/// A representation choice + its parameters, attached to an [`PyEventReader`] so each slice
/// renders to a dense frame (the `DataLoader`-friendly mode). One place that maps a repr name
/// to the core generator, shared by `open(repr=…)`, `with_repr`, `__getitem__`, and `batch`.
#[derive(Clone, Copy, Debug)]
enum ReprSpec {
    Binary,
    Count { normalize: bool },
    Polarity { normalize: bool },
    Voxel { bins: usize, window_ms: f64 },
    TimeSurface { tau_ms: f64 },
    AveragedTimeSurface { tau_ms: f64 },
    Tencode { window_ms: f64 },
    CountMask { pct: f64, white_frame: bool },
    Mcts { max_window_ms: f64 },
    Flow { window: usize },
}

impl ReprSpec {
    /// Builds a spec from a repr name and the optional overrides (unset ones take the
    /// same defaults as the `EventStream` methods: count raw `u64`, polarity normalized).
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        bins: Option<i64>,
        window_ms: Option<f64>,
        tau_ms: Option<f64>,
        max_window_ms: Option<f64>,
        window: Option<i64>,
        normalize: Option<bool>,
        pct: Option<f64>,
        white_frame: Option<bool>,
    ) -> PyResult<Self> {
        Ok(match name {
            "binary" => Self::Binary,
            "count" => Self::Count {
                normalize: normalize.unwrap_or(false),
            },
            "polarity" => Self::Polarity {
                normalize: normalize.unwrap_or(true),
            },
            "voxel" => Self::Voxel {
                bins: bins
                    .map(|value| {
                        usize::try_from(value)
                            .map_err(|_| PyValueError::new_err("bins must be at least 1"))
                    })
                    .transpose()?
                    .unwrap_or(9),
                window_ms: window_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "tsurf" => Self::TimeSurface {
                tau_ms: tau_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "atsurf" => Self::AveragedTimeSurface {
                tau_ms: tau_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "tencode" => Self::Tencode {
                window_ms: window_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "countmask" => Self::CountMask {
                pct: pct.unwrap_or(99.0),
                white_frame: white_frame.unwrap_or(false),
            },
            "mcts" => Self::Mcts {
                max_window_ms: max_window_ms.unwrap_or(DEFAULT_SPAN_MS),
            },
            "flow" => Self::Flow {
                window: window
                    .map(|value| {
                        usize::try_from(value)
                            .ok()
                            .filter(|&w| w >= 1)
                            .ok_or_else(|| PyValueError::new_err("window must be at least 1"))
                    })
                    .transpose()?
                    .unwrap_or(3),
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown representation: {other}"
                )))
            }
        })
    }

    /// A spec from a name alone, for the callers that don't accept per-representation options —
    /// `open(repr=…)`, `flatten`/`view`, `camera.show` — which take at most `normalize`.
    fn named(name: &str, normalize: Option<bool>) -> PyResult<Self> {
        Self::new(name, None, None, None, None, None, normalize, None, None)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Count { .. } => "count",
            Self::Polarity { .. } => "polarity",
            Self::Voxel { .. } => "voxel",
            Self::TimeSurface { .. } => "tsurf",
            Self::AveragedTimeSurface { .. } => "atsurf",
            Self::Tencode { .. } => "tencode",
            Self::CountMask { .. } => "countmask",
            Self::Mcts { .. } => "mcts",
            Self::Flow { .. } => "flow",
        }
    }

    fn generate(self, stream: &EventStream) -> Result<EventFrame, RepresentationError> {
        match self {
            Self::Binary => Binary.generate(stream),
            Self::Count { normalize } => EventCount::new(normalize).generate(stream),
            Self::Polarity { normalize } => Polarity::new(normalize).generate(stream),
            Self::Voxel { bins, window_ms } => VoxelGrid::new(bins, window_ms).generate(stream),
            Self::TimeSurface { tau_ms } => TimeSurface::new(tau_ms).generate(stream),
            Self::AveragedTimeSurface { tau_ms } => {
                AveragedTimeSurface::new(tau_ms).generate(stream)
            }
            Self::Tencode { window_ms } => Tencode::new(window_ms).generate(stream),
            Self::CountMask { pct, white_frame } => {
                CountMask::new(pct, white_frame).generate(stream)
            }
            Self::Mcts { max_window_ms } => Mcts::new(max_window_ms).generate(stream),
            Self::Flow { window } => stream.optical_flow(window).map_err(|error| match error {
                FlowError::SizeOverflow => RepresentationError::SizeOverflow,
                FlowError::InvalidParameter(name) => RepresentationError::InvalidParameter(name),
            }),
        }
    }
}

fn map_resize_error(error: ResizeError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn map_feast_error(error: FeastError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Runs a trained ONNX network over a representation.
///
/// Deliberately a runner, not a model zoo: it takes any `.onnx` file and knows nothing about the
/// architecture inside it. EventCV's job is to produce the tensor — a voxel grid or time surface is
/// already a dense `f32` array in the layout ONNX wants — and to get out of the way.
///
/// Requires a build with the `onnx` feature; `eventcv --version` reports whether this one has it.
#[cfg(feature = "onnx")]
#[pyclass(name = "Model")]
struct PyModel {
    // ONNX Runtime's session internals are not thread-safe and `run` needs `&mut`, but a `#[pyclass]`
    // hands out shared references. The mutex makes concurrent calls from Python threads safe rather
    // than a data race; inference is long enough that the lock is not the bottleneck.
    inner: Mutex<eventcv_core::model::Model>,
    inputs: Vec<eventcv_core::model::Port>,
    outputs: Vec<eventcv_core::model::Port>,
}

#[cfg(feature = "onnx")]
#[pymethods]
impl PyModel {
    /// Loads an ONNX model from `path`.
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        let inner = py
            .detach(|| eventcv_core::model::Model::load(path))
            .map_err(map_model_error)?;
        Ok(Self {
            inputs: inner.inputs().to_vec(),
            outputs: inner.outputs().to_vec(),
            inner: Mutex::new(inner),
        })
    }

    /// What the graph expects: a list of `{"name", "shape", "dtype"}`, with `-1` for any dimension
    /// the graph leaves free (usually the batch axis).
    #[getter]
    fn inputs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        ports_to_dicts(py, &self.inputs)
    }

    /// What the graph produces, in the same form as `inputs`.
    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        ports_to_dicts(py, &self.outputs)
    }

    /// Runs the model over `data` — an `EventFrame` or any float32-convertible array.
    ///
    /// Returns one array for a single-output graph, or a list in the graph's declared output order.
    /// A missing batch axis is added when the graph asks for one, so a `[C, H, W]` representation
    /// feeds a `[N, C, H, W]` network without the caller reshaping it.
    fn __call__<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let array = to_f32_array(py, data)?;
        let array = self.match_input_rank(array);
        let mut outputs = py
            .detach(|| {
                self.inner
                    .lock()
                    .expect("model mutex was poisoned by a previous panic")
                    .run(array)
            })
            .map_err(map_model_error)?;
        if outputs.len() == 1 {
            // The overwhelmingly common case; returning a one-element list would make every call
            // site index it for no reason.
            return Ok(outputs.remove(0).into_pyarray(py).into_any());
        }
        let list = PyList::empty(py);
        for output in outputs {
            list.append(output.into_pyarray(py))?;
        }
        Ok(list.into_any())
    }

    /// Runs the model with every input bound by name, returning `{name: array}`.
    ///
    /// The form `__call__` cannot express. A recurrent graph takes its hidden state as extra
    /// inputs and returns the updated state as extra outputs; feeding those back is what
    /// `StatefulModel` does on top of this.
    fn run_named<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut bound = Vec::with_capacity(inputs.len());
        for (key, value) in inputs.iter() {
            let name: String = key.extract().map_err(|_| {
                PyValueError::new_err("input names must be strings")
            })?;
            bound.push((name, to_f32_array(py, &value)?));
        }
        let session = &self.inner;
        let outputs = py
            .detach(|| {
                session
                    .lock()
                    .expect("model mutex was poisoned by a previous panic")
                    .run_named(bound)
            })
            .map_err(map_model_error)?;
        let result = PyDict::new(py);
        for (name, array) in outputs {
            result.set_item(name, array.into_pyarray(py))?;
        }
        Ok(result)
    }

    fn __repr__(&self) -> String {
        let shape = self
            .inputs
            .first()
            .map(|port| format!("{:?}", port.shape))
            .unwrap_or_else(|| "?".into());
        format!(
            "Model(inputs={}, outputs={}, input_shape={})",
            self.inputs.len(),
            self.outputs.len(),
            shape
        )
    }
}

#[cfg(feature = "onnx")]
impl PyModel {
    /// Adds a leading batch axis when the graph wants one and the caller did not supply it.
    ///
    /// A representation is `[C, H, W]` while nearly every trained network is `[N, C, H, W]`, so
    /// without this every call site would carry the same `arr[None]`. Only ever *adds* an axis, and
    /// only when the ranks differ by exactly one — anything else is a real mismatch the graph
    /// should reject with its own message.
    fn match_input_rank(&self, array: numpy::ndarray::ArrayD<f32>) -> numpy::ndarray::ArrayD<f32> {
        let Some(expected) = self.inputs.first().map(|port| port.shape.len()) else {
            return array;
        };
        if expected == array.ndim() + 1 {
            let mut shape = vec![1usize];
            shape.extend_from_slice(array.shape());
            if let Ok(reshaped) = array.clone().into_shape_with_order(shape) {
                return reshaped;
            }
        }
        array
    }
}

/// Renders model port metadata as plain dicts, so the shape/dtype of a graph can be inspected
/// without importing anything or learning a new type.
#[cfg(feature = "onnx")]
fn ports_to_dicts<'py>(
    py: Python<'py>,
    ports: &[eventcv_core::model::Port],
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    ports
        .iter()
        .map(|port| {
            let dict = PyDict::new(py);
            dict.set_item("name", port.name.as_str())?;
            dict.set_item("shape", port.shape.clone())?;
            dict.set_item("dtype", port.dtype.as_str())?;
            Ok(dict)
        })
        .collect()
}

/// Coerces a model input to an owned `f32` array, accepting an `EventFrame` directly.
#[cfg(feature = "onnx")]
fn to_f32_array(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<numpy::ndarray::ArrayD<f32>> {
    if let Ok(frame) = data.extract::<PyRef<PyEventFrame>>() {
        // Go through the frame's own numpy view so every EventFrameData dtype is handled in one
        // place rather than matched again here.
        let array = frame_numpy(py, &frame.inner);
        return to_f32_array(py, &array);
    }
    let array = data
        .call_method1("astype", ("float32",))
        .unwrap_or_else(|_| data.clone());
    let readonly = array.extract::<numpy::PyReadonlyArrayDyn<f32>>().map_err(|_| {
        PyValueError::new_err(
            "model input must be an EventFrame or a numpy array convertible to float32",
        )
    })?;
    Ok(readonly.as_array().to_owned())
}

#[cfg(feature = "onnx")]
fn map_model_error(error: eventcv_core::model::ModelError) -> PyErr {
    use eventcv_core::model::ModelError;
    match error {
        // A path that does not exist is the common mistake and deserves the specific exception.
        // ort words this as "File at `…` does not exist"; match loosely so a reworded message in
        // a later release degrades to ValueError rather than being mapped to something wrong.
        ModelError::Load(message)
            if message.contains("does not exist") || message.contains("No such file") =>
        {
            PyFileNotFoundError::new_err(message)
        }
        ModelError::Load(message)
        | ModelError::Run(message)
        | ModelError::UnsupportedOutput(message) => PyValueError::new_err(message),
    }
}

/// FEAST feature extractor (Afshar et al., 2020) — an unsupervised, online, sklearn-style model.
/// `fit` adapts feature weights and selection thresholds event-by-event (repeatable across
/// recordings/epochs); `transform` then maps each event to its nearest feature id.
#[pyclass(name = "FEAST")]
struct PyFeast {
    inner: Feast,
}

#[pymethods]
impl PyFeast {
    #[new]
    #[pyo3(signature = (n_features=100, patch=11, tau_ms=None, eta=0.001, delta_i=0.001,
        delta_e=0.003, per_polarity=true, seed=0, *, tau_s=None, tau_us=None, tau_ns=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_features: usize,
        patch: usize,
        tau_ms: Option<f64>,
        eta: f32,
        delta_i: f32,
        delta_e: f32,
        per_polarity: bool,
        seed: u64,
        tau_s: Option<f64>,
        tau_us: Option<f64>,
        tau_ns: Option<f64>,
    ) -> PyResult<Self> {
        let config = FeastConfig {
            n_features,
            patch,
            tau_ms: resolve_ms("tau", tau_s, tau_ms, tau_us, tau_ns)?.unwrap_or(DEFAULT_SPAN_MS),
            eta,
            delta_i,
            delta_e,
            per_polarity,
            seed,
        };
        Feast::new(config)
            .map(|inner| Self { inner })
            .map_err(map_feast_error)
    }

    /// Trains on `stream` for `epochs` online passes; returns the final-epoch **miss rate** (the
    /// fraction of in-bounds events matching no feature), which settles low as the network
    /// converges. Call repeatedly to train across multiple recordings.
    #[pyo3(signature = (stream, epochs=1))]
    fn fit(&mut self, py: Python<'_>, stream: PyRef<'_, PyEventStream>, epochs: usize) -> f64 {
        let events = &stream.inner;
        py.detach(|| self.inner.fit(events, epochs))
    }

    /// Feature id for every event (nearest feature by cosine distance; `-1` for border events).
    /// Shape `(len(stream),)`, dtype int32; ids are `population * n_features + feature`.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        stream: PyRef<'_, PyEventStream>,
    ) -> Bound<'py, PyArray1<i32>> {
        let events = &stream.inner;
        py.detach(|| self.inner.transform(events)).into_pyarray(py)
    }

    /// Pooled feature-event counts over `stream` (the paper's classifier input). Shape
    /// `(n_features_total,)`, dtype uint32.
    fn histogram<'py>(
        &self,
        py: Python<'py>,
        stream: PyRef<'_, PyEventStream>,
    ) -> Bound<'py, PyArray1<u32>> {
        let events = &stream.inner;
        py.detach(|| self.inner.histogram(events)).into_pyarray(py)
    }

    /// The learned features as an `(n_features_total, patch, patch)` float32 array — reproduces
    /// the paper's feature grids for visualisation.
    fn feature_images<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let patch = self.inner.config().patch;
        let shape = (self.inner.n_features_total(), patch, patch);
        numpy::ndarray::Array3::from_shape_vec(shape, self.inner.weights().to_vec())
            .map_err(|_| PyRuntimeError::new_err("feature weights shape mismatch"))
            .map(|array| array.into_pyarray(py))
    }

    /// The current selection thresholds, shape `(n_features_total,)`, dtype float32.
    #[getter]
    fn thresholds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.thresholds().to_vec().into_pyarray(py)
    }

    /// Miss rate recorded by the most recent `fit` (0.0 before any training).
    #[getter]
    fn missed_rate(&self) -> f64 {
        self.inner.missed_rate()
    }

    /// The constructor parameters as a dict (sklearn-style); consumed by `eventcv.save`.
    fn get_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let config = self.inner.config();
        let params = PyDict::new(py);
        params.set_item("n_features", config.n_features)?;
        params.set_item("patch", config.patch)?;
        params.set_item("tau_ms", config.tau_ms)?;
        params.set_item("eta", config.eta)?;
        params.set_item("delta_i", config.delta_i)?;
        params.set_item("delta_e", config.delta_e)?;
        params.set_item("per_polarity", config.per_polarity)?;
        params.set_item("seed", config.seed)?;
        Ok(params)
    }

    /// Replaces the weights/thresholds from saved arrays — the `eventcv.load_feast` rehydration
    /// path. Both arrays must match the model's configured sizes.
    fn _load_state(
        &mut self,
        features: PyReadonlyArray3<f32>,
        thresholds: PyReadonlyArray1<f32>,
    ) -> PyResult<()> {
        let weights = features.as_array().iter().copied().collect();
        let thresholds = thresholds.as_array().iter().copied().collect();
        self.inner = Feast::from_state(*self.inner.config(), weights, thresholds)
            .map_err(map_feast_error)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let config = self.inner.config();
        format!(
            "FEAST(n_features={}, patch={}, tau_ms={}, per_polarity={})",
            config.n_features, config.patch, config.tau_ms, config.per_polarity
        )
    }
}

// ---- Live USB event-camera streaming (`camera` feature) --------------------------------
//
// `list_cameras()` enumerates connected devices; `stream(...)` opens one as an `EventCamera`.
// An `EventCamera` is the live twin of `EventReader`: iterate it for fixed windows of events (or
// representations), `show()` a live view, or `record()` to a file. The heavy lifting lives in
// `eventcv_core::device` (capture) and `eventcv_core::viz` (raw rendering); this is the PyO3 glue.

/// How long the live representation view accumulates events between rendered frames (~30 FPS).
/// Decoupled from `stream(dt_ms=…)` (that windows the *data* API); this paces the *display*.
#[cfg(feature = "camera")]
const LIVE_REPR_WINDOW_MS: u64 = 33;

/// Wall-clock budget the live viewer spends **decoding** events per display frame. At very high
/// event rates (a 1280×720 Prophesee sensor can emit hundreds of millions of events per second)
/// decoding a whole frame's worth of events takes far longer than a display frame, so the viewer
/// decodes only this much of the freshest data each frame and drops the surplus — keeping the
/// display at ~60 FPS instead of stalling to a couple of frames per second. The dropped events are
/// invisible at that rate anyway. Iteration/`record` are unaffected (they process every event).
#[cfg(feature = "camera")]
const LIVE_DRAIN_BUDGET_MS: u64 = 10;

/// How often the capture thread renders a fresh frame for the display (~60 FPS). This paces only
/// the *rendering*; the thread keeps draining the driver ring continuously between renders, so the
/// USB pipeline stays fed even while the GPU thread is blocked on vsync.
#[cfg(feature = "camera")]
const LIVE_RENDER_INTERVAL_MS: u64 = 16;

/// How often a continuous recording is forced out to disk, so a crash mid-session keeps everything
/// captured up to about a second ago. Shared by `record()` and `stream(record=…)`.
#[cfg(all(feature = "camera", feature = "hdf5"))]
const RECORD_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// What the live viewer is opened for: watching the stream (`show`) or drawing an ROI over it
/// (`draw_mask`). Both run the same capture thread and renderer.
#[cfg(feature = "camera")]
#[derive(Clone, Copy)]
enum LiveMode {
    Show,
    Draw,
}

/// How the live event stream is turned into frames for the interactive viewer.
#[cfg(feature = "camera")]
enum LiveRenderer {
    /// Raw polarity view with exponential decay — the default. Events are stamped straight onto a
    /// persistent surface, which is re-rendered each display frame.
    Raw(eventcv_core::viz::RawSurface),
    /// A representation, colour-mapped to a 2-D image (voxel/tsurf collapse to 2-D). Events
    /// accumulate into `builder`, rendered every `window` of wall-clock time.
    Repr {
        spec: ReprSpec,
        colormap: Colormap,
        normalize: bool,
        width: usize,
        height: usize,
        window: std::time::Duration,
        builder: EventStreamBuilder,
        last_render: std::time::Instant,
    },
}

#[cfg(feature = "camera")]
impl LiveRenderer {
    /// Drains the driver ring (budgeted) and folds the events into the running view — stamping the
    /// raw surface, or accumulating into the representation builder. Produces no image; that is
    /// [`render`](Self::render)'s job, paced separately to the display.
    ///
    /// Meant to be called back-to-back on the capture thread so the ring is emptied continuously and
    /// the driver never stalls (a stop-start consumer makes the driver deliver in large bursts, which
    /// is what made the single-threaded viewer sluggish). Returns whether any events arrived this
    /// call, so an idle loop can back off instead of busy-spinning.
    fn pump(&mut self, capture: &mut eventcv_core::device::Capture) -> Result<bool, String> {
        let budget = std::time::Duration::from_millis(LIVE_DRAIN_BUDGET_MS);
        let mut lit = false;
        let overflow = match self {
            LiveRenderer::Raw(surface) => {
                capture.drain_events_budgeted(budget, |x, y, t_us, positive| {
                    lit = true;
                    surface.stamp(x as usize, y as usize, t_us as f64 * 0.001, positive);
                })?
            }
            LiveRenderer::Repr { builder, .. } => {
                capture.drain_events_budgeted(budget, |x, y, t_us, positive| {
                    lit = true;
                    builder.push(x, y, t_us, positive);
                })?
            }
        };
        if overflow {
            note_overflow();
        }
        Ok(lit)
    }

    /// Produces the next frame to display, or `None` when nothing new is due. The raw surface renders
    /// every call (it always reflects the freshest events, and the decay animates even between
    /// events); a representation renders only once its accumulation `window` of wall-clock time has
    /// elapsed, then resets its builder. Paced by the caller to the display refresh, independent of
    /// [`pump`](Self::pump)'s continuous drain cadence.
    fn render(&mut self) -> Result<Option<eventcv_core::viz::Rgb8Image>, String> {
        match self {
            LiveRenderer::Raw(surface) => Ok(Some(surface.render())),
            LiveRenderer::Repr {
                spec,
                colormap,
                normalize,
                width,
                height,
                window,
                builder,
                last_render,
            } => {
                if last_render.elapsed() < *window || builder.is_empty() {
                    return Ok(None);
                }
                let fresh = EventStreamBuilder::new(*width, *height, 0.001);
                let stream = std::mem::replace(builder, fresh).build();
                *last_render = std::time::Instant::now();
                let frame = spec.generate(&stream).map_err(|error| error.to_string())?;
                Ok(Some(eventcv_core::viz::render_frame(
                    &frame, *colormap, *normalize,
                )))
            }
        }
    }
}

/// Shared state between the capture thread (drains + renders) and the main/GPU thread (presents).
/// The capture thread publishes the freshest frame into `image` and any fatal error into `error`;
/// the main thread sets `stop` when the window closes.
#[cfg(feature = "camera")]
struct LiveShared {
    image: Mutex<Option<eventcv_core::viz::Rgb8Image>>,
    error: Mutex<Option<String>>,
    stop: std::sync::atomic::AtomicBool,
}

/// Runs the live viewer with the draining decoupled onto a dedicated thread.
///
/// A background thread takes ownership of `capture` + `renderer` and continuously drains the driver
/// ring — so the USB pipeline keeps flowing even while the GPU thread is blocked in `present()`
/// waiting for vsync (the coupling that throttled the old single-threaded loop to ~22 FPS). It
/// renders a frame at [`LIVE_RENDER_INTERVAL_MS`] and publishes it; the main thread's producer just
/// hands the viewer whatever frame is freshest. Returns the reclaimed `capture` (so the camera stays
/// usable after `show()` returns) and the viewer's result. `capture` is `None` only if the drain
/// thread panicked.
#[cfg(feature = "camera")]
fn run_live_threaded(
    capture: eventcv_core::device::Capture,
    renderer: LiveRenderer,
    width: u32,
    height: u32,
    title: String,
    mode: LiveMode,
) -> (
    Option<eventcv_core::device::Capture>,
    Result<Option<Vec<bool>>, String>,
) {
    use std::sync::atomic::Ordering;

    let shared = Arc::new(LiveShared {
        image: Mutex::new(None),
        error: Mutex::new(None),
        stop: std::sync::atomic::AtomicBool::new(false),
    });

    let worker = Arc::clone(&shared);
    let handle = std::thread::spawn(move || {
        let mut capture = capture;
        let mut renderer = renderer;
        let render_interval = std::time::Duration::from_millis(LIVE_RENDER_INTERVAL_MS);
        let mut last_render = std::time::Instant::now();
        while !worker.stop.load(Ordering::Acquire) {
            let lit = match renderer.pump(&mut capture) {
                Ok(lit) => lit,
                Err(message) => {
                    *worker.error.lock().unwrap() = Some(message);
                    break;
                }
            };
            if last_render.elapsed() >= render_interval {
                match renderer.render() {
                    Ok(Some(image)) => *worker.image.lock().unwrap() = Some(image),
                    Ok(None) => {}
                    Err(message) => {
                        *worker.error.lock().unwrap() = Some(message);
                        break;
                    }
                }
                last_render = std::time::Instant::now();
            }
            // Nothing to decode: sleep briefly so an idle scene doesn't peg a core. When events are
            // flowing this never runs, so draining stays continuous.
            if !lit {
                std::thread::park_timeout(std::time::Duration::from_millis(1));
            }
        }
        capture
    });

    // Main-thread producer: hand the viewer the freshest published frame (taken, so an unchanged
    // view isn't re-uploaded), surfacing any fatal drain error.
    let producer_shared = Arc::clone(&shared);
    let producer = move || -> Result<Option<eventcv_core::viz::Rgb8Image>, String> {
        if let Some(message) = producer_shared.error.lock().unwrap().take() {
            return Err(message);
        }
        Ok(producer_shared.image.lock().unwrap().take())
    };

    let result = match mode {
        LiveMode::Show => viewer::run_live(producer, width, height, title).map(|()| None),
        LiveMode::Draw => viewer::draw_mask(producer, width as usize, height as usize, title),
    };

    // Stop the drain thread and reclaim the capture.
    shared.stop.store(true, Ordering::Release);
    handle.thread().unpark();
    let capture = handle.join().ok();
    // A drain-thread error is the root cause, so prefer it over the viewer's shutdown result.
    let result = match shared.error.lock().unwrap().take() {
        Some(message) => Err(message),
        None => result,
    };
    (capture, result)
}

/// How often the overflow warning may repeat. A camera that outruns the host overflows on window
/// after window, so warning per occurrence buries everything else the program prints; the count on
/// `EventCamera.n_overflows` is the complete record.
#[cfg(feature = "camera")]
const OVERFLOW_WARNING_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// Warns that the driver's ring buffer overflowed and dropped events, at most once every
/// [`OVERFLOW_WARNING_INTERVAL`].
#[cfg(feature = "camera")]
fn note_overflow() {
    static LAST_WARNING: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let now = std::time::Instant::now();
    let mut last = LAST_WARNING.lock().unwrap();
    if last.is_some_and(|last| now.duration_since(last) < OVERFLOW_WARNING_INTERVAL) {
        return;
    }
    *last = Some(now);
    eprintln!(
        "eventcv: camera ring buffer overflowed — some events were dropped \
         (lower the event rate, increase dt_ms, or process each window faster). \
         Further overflows are counted in EventCamera.n_overflows rather than printed."
    );
}

/// The open file behind `stream(record=…)` plus its flush clock: every window the camera hands out
/// is appended here before it is rendered, so one loop both archives the raw events and processes
/// representations.
#[cfg(all(feature = "camera", feature = "hdf5"))]
pub(crate) struct Recorder {
    sink: eventcv_core::io::Hdf5EventSink,
    last_flush: std::time::Instant,
}

/// Without HDF5 there is nothing to record into — `stream(record=…)` is rejected up front, so this
/// placeholder only exists to keep the camera's plumbing free of feature gates.
#[cfg(all(feature = "camera", not(feature = "hdf5")))]
pub(crate) struct Recorder;

#[cfg(all(feature = "camera", not(feature = "hdf5")))]
impl Recorder {
    pub(crate) fn append(&mut self, _stream: &EventStream) -> Result<(), IoError> {
        Ok(())
    }

    pub(crate) fn n_events(&self) -> usize {
        0
    }

    pub(crate) fn finish(self) -> Result<(), IoError> {
        Ok(())
    }
}

#[cfg(all(feature = "camera", feature = "hdf5"))]
impl Recorder {
    /// Checks a `record=` request without touching the filesystem, so a typo is rejected before the
    /// camera is opened — and with the same error whether or not a device is attached.
    fn validate(path: &str, compression: Option<u8>) -> PyResult<()> {
        if compression.is_some_and(|level| level > 9) {
            return Err(PyValueError::new_err(
                "compression must be a gzip level in 0..=9",
            ));
        }
        if !eventcv_core::io::supports_event_append(path) {
            return Err(PyValueError::new_err(format!(
                "record={path} needs a format that can be appended window-by-window (.h5/.hdf5); \
                 for npz/txt/bag record the whole session at once with camera.record({path:?})"
            )));
        }
        Ok(())
    }

    fn open(path: &str, compression: Option<u8>) -> PyResult<Self> {
        Ok(Self {
            sink: eventcv_core::io::Hdf5EventSink::open(path, compression).map_err(map_io_error)?,
            last_flush: std::time::Instant::now(),
        })
    }

    /// Appends one window, flushing about once a second.
    pub(crate) fn append(&mut self, stream: &EventStream) -> Result<(), IoError> {
        self.sink.append(stream)?;
        if self.last_flush.elapsed() >= RECORD_FLUSH_INTERVAL {
            self.sink.flush()?;
            self.last_flush = std::time::Instant::now();
        }
        Ok(())
    }

    pub(crate) fn n_events(&self) -> usize {
        self.sink.n_events()
    }

    /// Flushes and closes the recording.
    pub(crate) fn finish(self) -> Result<(), IoError> {
        self.sink.finish()
    }
}

/// How long a read waits on the pump before surfacing to Python to check its deadline and Ctrl+C.
#[cfg(feature = "camera")]
const READ_SLICE: std::time::Duration = std::time::Duration::from_millis(20);

/// Where the open camera currently lives. Reading runs it on the pump thread; `show`, `record`, and
/// `close` need it back on this thread, so they park the pump first.
#[cfg(feature = "camera")]
enum CameraState {
    /// Held here, decoding nothing until a window is read.
    Idle {
        capture: eventcv_core::device::Capture,
        recorder: Option<Recorder>,
    },
    /// The pump thread owns the camera and decodes continuously.
    Pumping(capture::Pump),
    /// Closed via `close()` / `__exit__`.
    Closed,
}

/// A live USB event camera — the streaming twin of [`PyEventReader`]. Open one with
/// `eventcv.stream(...)`.
#[cfg(feature = "camera")]
#[pyclass(name = "EventCamera", unsendable)]
struct PyEventCamera {
    state: CameraState,
    // Representation iterated windows render to (`open(repr=…)` twin); `None` yields raw streams.
    repr: Option<ReprSpec>,
    // Default decay time constant (ms) for the raw `show()` view.
    decay_ms: f64,
    // What the pump does with windows the loop hasn't collected (`stream(latest=True)`).
    mode: capture::Backpressure,
    // Cached at open so the getters keep working while the pump thread owns the capture.
    name: String,
    serial: String,
    sensor_size: (usize, usize),
    // The ROI mask the capture is filtering through, mirrored here for the same reason.
    mask: Option<Vec<bool>>,
    // The `stream(roi=…)` rectangle and where it is enforced, cached for the same reason.
    roi: Option<(
        (usize, usize, usize, usize),
        eventcv_core::device::RoiPlacement,
    )>,
    // The `stream(record=…)` file, kept so `record()` can refuse to split the recording in two.
    record_path: Option<String>,
    // Skips and overflows from pumps already stopped, so the counters survive a `show()` round trip.
    skipped: usize,
    overflows: usize,
}

#[cfg(feature = "camera")]
#[pymethods]
impl PyEventCamera {
    #[getter]
    fn sensor_size(&self) -> PyResult<(usize, usize)> {
        self.open_check()?;
        Ok(self.sensor_size)
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        self.open_check()?;
        Ok(self.name.clone())
    }

    #[getter]
    fn serial(&self) -> PyResult<String> {
        self.open_check()?;
        Ok(self.serial.clone())
    }

    /// Buffers waiting in the driver's ring. The pump thread drains it continuously, so this stays
    /// near zero unless the camera is producing faster than events can be decoded and written.
    #[getter]
    fn backlog(&self) -> PyResult<usize> {
        match &self.state {
            CameraState::Pumping(pump) => Ok(pump.backlog()),
            CameraState::Idle { capture, .. } => Ok(capture.backlog()),
            CameraState::Closed => Err(PyRuntimeError::new_err("camera is closed")),
        }
    }

    /// Windows a `stream(latest=True)` loop skipped to stay live — the count of frames the loop was
    /// too slow to process. Always `0` without `latest=True` (that mode never skips). A steadily
    /// rising value means the per-window work is slower than the camera; recordings stay complete
    /// either way.
    #[getter]
    fn n_skipped(&self) -> usize {
        self.skipped + self.pump().map_or(0, capture::Pump::n_skipped)
    }

    /// Times the driver's ring overflowed and dropped events before a window — the loss `n_skipped`
    /// can't prevent, because it happens upstream of eventcv. Non-zero means the camera outran the
    /// decoder or the recording; lower the event rate, or record uncompressed.
    #[getter]
    fn n_overflows(&self) -> usize {
        self.overflows + self.pump().map_or(0, capture::Pump::n_overflows)
    }

    /// The region-of-interest mask events are filtered through — an `(H, W)` boolean array, or
    /// `None` when the whole sensor is in use.
    ///
    /// Assign one to change it mid-session (`camera.mask = eventcv.circle_mask(...)`, or `None` to
    /// clear it); `stream(mask=…)` sets it at open. Masked events are dropped **as they are
    /// decoded**, so they never reach a `record=` file, the windows a loop reads, or `show()` —
    /// and they cost nothing downstream. Where `stream(roi=…)` is a rectangle, blocked on-chip
    /// wherever the sensor can do it, this is any shape and always enforced on the host — so it can
    /// be changed while the camera runs.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        let (width, height) = self.sensor_size;
        self.mask
            .clone()
            .map(|mask| mask_to_py(py, mask, width, height))
            .transpose()
    }

    #[setter]
    fn set_mask(&mut self, mask: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let mask = mask
            .map(|mask| parse_mask(mask, self.sensor_size))
            .transpose()?;
        self.apply_mask(mask)
    }

    /// The `stream(roi=…)` rectangle, as `{"rect": (x0, y0, width, height), "applied": …}` — or
    /// `None` when the whole sensor is in use.
    ///
    /// `applied` is `"hardware"` when the sensor is blocking those pixels on-chip (they cost no USB
    /// bandwidth, no decoding, and nothing downstream) or `"host"` when eventcv is filtering them as
    /// they are decoded, because this sensor has no region masks. Both drop the same events; only
    /// the saving differs. Unlike `mask`, this is fixed when the camera is opened — the sensor
    /// writes its region masks during the opening handshake.
    #[getter]
    fn roi(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some((rect, placement)) = self.roi else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("rect", rect)?;
        dict.set_item(
            "applied",
            match placement {
                eventcv_core::device::RoiPlacement::Hardware => "hardware",
                eventcv_core::device::RoiPlacement::Host => "host",
            },
        )?;
        Ok(Some(dict.unbind()))
    }

    /// Events written so far by `stream(record=…)` (`0` when the camera isn't recording).
    #[getter]
    fn n_recorded(&self) -> usize {
        match &self.state {
            CameraState::Pumping(pump) => pump.n_recorded(),
            CameraState::Idle { recorder, .. } => recorder.as_ref().map_or(0, Recorder::n_events),
            CameraState::Closed => 0,
        }
    }

    /// What the adaptive-bias controller is doing, or `None` when the camera was opened without
    /// `stream(adaptive_bias=…)`.
    ///
    /// A dict of the last measured `event_rate` (events/second), the `target_rate` band being
    /// aimed at, whether it is still `calibrating`, the five bias values the controller drives, its
    /// `authority`, and `n_slow_steps` — how many times the slow loop has had to shift the
    /// operating point.
    ///
    /// For the first second or so `calibrating` is true: the controller is measuring the scene at
    /// the camera's stock biases and has changed nothing. `target_rate` afterwards is the band it
    /// settled on, which is the number to look at if the results surprise you.
    ///
    /// `authority` says how the fast loop is coping, and is the field to watch:
    ///
    /// - `"tracking"` — fine: the rate is in your band, or the loop is on its way there.
    /// - `"starved"` / `"flooded"` — the refractory bias is at one end of its travel and the rate
    ///   is still out of band, so the slow loop is shifting the operating point to help.
    /// - `"hunting"` — the loop keeps reversing without ever landing in the band, which means no
    ///   bias setting reaches it. Almost always a sign that `target_rate` does not match what the
    ///   scene can produce; measure the scene unbiased and re-bracket it.
    ///
    /// Note `event_rate` was measured over the period *before* the biases now reported were
    /// applied — the two come from consecutive periods, not the same one.
    #[getter]
    fn bias_state(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let state = match &self.state {
            CameraState::Pumping(pump) => pump.bias_state(),
            CameraState::Idle { capture, .. } => capture.bias_state(),
            CameraState::Closed => return Ok(None),
        };
        let Some(state) = state else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("event_rate", state.event_rate)?;
        dict.set_item("target_rate", state.target_rate)?;
        dict.set_item("calibrating", state.calibrating)?;
        dict.set_item("refractory", state.values.refractory)?;
        dict.set_item("photoreceptor", state.values.photoreceptor)?;
        dict.set_item("follower", state.values.follower)?;
        dict.set_item("on_threshold", state.values.on_threshold)?;
        dict.set_item("off_threshold", state.values.off_threshold)?;
        dict.set_item(
            "authority",
            match state.authority {
                eventcv_core::bias::Authority::Tracking => "tracking",
                eventcv_core::bias::Authority::Starved => "starved",
                eventcv_core::bias::Authority::Flooded => "flooded",
                eventcv_core::bias::Authority::Hunting => "hunting",
            },
        )?;
        dict.set_item("n_slow_steps", state.slow_steps)?;
        Ok(Some(dict.unbind()))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Blocks until the next window is ready and returns it — an `EventFrame` when the camera was
    /// opened with a representation (`stream(repr=…)`), else a raw `EventStream`. Ends iteration
    /// (raises `StopIteration`) only once the camera is closed. `Ctrl+C` breaks the loop.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if matches!(self.state, CameraState::Closed) {
            return Ok(None); // closed -> StopIteration
        }
        self.poll_next(py, None)
    }

    /// Returns the next window for a `while` loop — an `EventFrame` when the camera was opened with
    /// a representation (`stream(repr=…)`), else a raw `EventStream`. The window spans exactly the
    /// `dt_ms` / `max_events` chosen at `stream(...)` (so `stream(dt_ms=50, repr="mcts").read()`
    /// hands back one mcts frame per ~50 ms of events).
    ///
    /// Blocks until that window is ready. Pass `timeout_ms` to cap the wait: if no window completes
    /// within that many milliseconds `read` returns `None`, so the loop can do other work or exit
    /// instead of blocking forever on an idle scene. (`timeout_ms` is a wait cap, *not* the window
    /// length.) `Ctrl+C` breaks out. Raises if the camera is closed.
    ///
    /// By default windows are returned in order, so a loop slower than the camera falls further and
    /// further behind live. `stream(latest=True)` returns the newest completed window instead,
    /// counting what it skipped in `n_skipped` — `record=` still archives every window either way.
    #[pyo3(signature = (*, timeout_ms=None))]
    fn read(&mut self, py: Python<'_>, timeout_ms: Option<f64>) -> PyResult<Option<Py<PyAny>>> {
        self.open_check()?;
        let timeout = timeout_ms.map(|ms| std::time::Duration::from_secs_f64(ms.max(0.0) / 1000.0));
        self.poll_next(py, timeout)
    }

    /// Opens the interactive live viewer. With no `representation`, shows the **raw** event stream
    /// (polarity dots with exponential decay — the default view). Pass a representation name
    /// (`"count"`, `"tencode"`, `"voxel"`, …) to render that live instead, or `"raw"` to force the
    /// raw view. Blocks on the main thread until the window is closed. `decay_ms` tunes the raw
    /// fade; `colormap` / `normalize` apply to representations.
    #[pyo3(signature = (representation=None, *, decay_ms=None, colormap="viridis", normalize=true))]
    fn show(
        &mut self,
        py: Python<'_>,
        representation: Option<String>,
        decay_ms: Option<f64>,
        colormap: &str,
        normalize: bool,
    ) -> PyResult<()> {
        self.live(
            py,
            representation,
            decay_ms,
            colormap,
            normalize,
            LiveMode::Show,
        )?;
        Ok(())
    }

    /// Draws a region of interest over the **live** view and returns it as an `(H, W)` boolean
    /// mask, applying it to this camera as well — so the next `read()`, loop, or `record=` window
    /// is already filtered. Everything `show()` takes applies here too (the view you draw over).
    ///
    /// Drag to keep an area, shift+drag to drop one, `e`/`r`/`f` to switch between ellipse,
    /// rectangle, and freehand, `a`/`c` to select all or clear, `z` to undo. Whatever stays bright
    /// is what the mask keeps. `Enter` accepts; closing the window or `Esc` returns `None` and
    /// leaves the camera's current mask alone.
    #[pyo3(signature = (representation=None, *, decay_ms=None, colormap="viridis", normalize=true))]
    fn draw_mask<'py>(
        &mut self,
        py: Python<'py>,
        representation: Option<String>,
        decay_ms: Option<f64>,
        colormap: &str,
        normalize: bool,
    ) -> PyResult<Option<Bound<'py, PyArray2<bool>>>> {
        // Draw over the whole sensor rather than through whatever ROI is already set — otherwise
        // the view hides the very events you are deciding about. Restored if the drawing is
        // cancelled, replaced if it isn't.
        let previous = self.mask.clone();
        self.apply_mask(None)?;
        let drawn = self.live(
            py,
            representation,
            decay_ms,
            colormap,
            normalize,
            LiveMode::Draw,
        )?;
        let Some(mask) = drawn else {
            self.apply_mask(previous)?;
            return Ok(None);
        };
        self.apply_mask(Some(mask.clone()))?;
        let (width, height) = self.sensor_size;
        mask_to_py(py, mask, width, height).map(Some)
    }

    /// Records the live stream to `path` (format by extension, like `eventcv.save`). Blocks until
    /// `seconds` elapses (if given) or `Ctrl+C`, then returns the number of events saved.
    ///
    /// HDF5 targets (`.h5` / `.hdf5`) are written **continuously**, window-by-window straight to
    /// disk via an `EventSink`, and flushed roughly once a second — so a long session never piles up
    /// in memory and a crash keeps everything captured so far. `compression` is an optional gzip
    /// level (`0..=9`) for that path; omit it for the fastest, uncompressed writes. Other formats
    /// (npz/txt/bag can't be appended incrementally) buffer the whole recording in memory and write
    /// once at the end, ignoring `compression`.
    ///
    /// Only events are saved — a DAVIS346's APS frames and IMU samples are dropped.
    #[pyo3(signature = (path, *, seconds=None, compression=None))]
    fn record(
        &mut self,
        py: Python<'_>,
        path: &str,
        seconds: Option<f64>,
        compression: Option<u8>,
    ) -> PyResult<usize> {
        // Both write the same windows, so running them together would tear the recording in two:
        // everything captured here would be missing from the `stream(record=…)` file.
        if let Some(open) = &self.record_path {
            return Err(PyValueError::new_err(format!(
                "this camera is already recording to {open:?} (from stream(record=…)); close it \
                 first, or drop record= from stream() and use camera.record({path:?}) alone"
            )));
        }
        #[cfg(feature = "hdf5")]
        if eventcv_core::io::supports_event_append(path) {
            return self.record_streaming(py, path, seconds, compression);
        }
        let _ = compression; // only the HDF5 streaming path compresses; buffered formats ignore it.
        self.record_buffered(py, path, seconds)
    }

    /// Headless diagnostic: splits the cost of a captured event into **parsing** it out of the
    /// device's wire format and **accumulating** it into a window, which decides whether decode
    /// throughput is worth parallelising.
    ///
    /// Both phases first let the driver's ring fill for `fill_ms`, then decode that backlog
    /// back-to-back, so the measurement is CPU-bound rather than limited by how fast the scene
    /// happens to produce events. Phase one runs `drain_events` with a counting callback (wire
    /// format only); phase two decodes the same data into the window builder (the path a live
    /// `read()` uses). Returns each phase's events, milliseconds, and nanoseconds per event.
    #[pyo3(signature = (*, fill_ms=1000.0))]
    fn _decode_benchmark(&mut self, py: Python<'_>, fill_ms: f64) -> PyResult<Py<PyDict>> {
        let capture = self.parked_capture()?;
        let fill = std::time::Duration::from_secs_f64(fill_ms.max(0.0) / 1000.0);
        let measure = |capture: &mut eventcv_core::device::Capture, accumulate: bool| {
            std::thread::sleep(fill); // let the ring back up so there is a batch to decode
            let backlog = capture.backlog();
            let mut events = 0usize;
            let start = std::time::Instant::now();
            if accumulate {
                while capture.decode_next(std::time::Duration::ZERO)? {
                    while let Some(window) = capture.take_pending() {
                        events += window.stream.len();
                    }
                }
            } else {
                capture.drain_events(|_x, _y, _t, _polarity| events += 1)?;
            }
            Ok::<_, String>((events, start.elapsed().as_secs_f64() * 1000.0, backlog))
        };
        let ((parse_events, parse_ms, parse_backlog), (window_events, window_ms, window_backlog)) =
            py.detach(|| Ok::<_, String>((measure(capture, false)?, measure(capture, true)?)))
                .map_err(PyRuntimeError::new_err)?;

        let per_event = |events: usize, ms: f64| {
            if events == 0 {
                0.0
            } else {
                ms * 1e6 / events as f64
            }
        };
        let dict = PyDict::new(py);
        dict.set_item("parse_events", parse_events)?;
        dict.set_item("parse_ms", parse_ms)?;
        dict.set_item("parse_ns_per_event", per_event(parse_events, parse_ms))?;
        dict.set_item("parse_backlog", parse_backlog)?;
        dict.set_item("window_events", window_events)?;
        dict.set_item("window_ms", window_ms)?;
        dict.set_item("window_ns_per_event", per_event(window_events, window_ms))?;
        dict.set_item("window_backlog", window_backlog)?;
        Ok(dict.unbind())
    }

    /// Headless diagnostic: runs the live drain/render loop for `seconds` at ~60 FPS **without**
    /// opening a window, returning `{frames, events, events_per_s, max_backlog, seconds}`. A
    /// healthy viewer keeps `max_backlog` near zero and renders ~60 frames/second.
    ///
    /// By default it mirrors the live viewer: a `budget_ms` wall-clock decode budget per frame
    /// (the freshest events are decoded, the surplus dropped) — this is what keeps the display at
    /// ~60 FPS on high-rate sensors. Pass `budget_ms=None` to decode every queued event per frame
    /// (`drain`-based); on a fast camera that collapses to a couple of frames/second. `drain=False`
    /// uses the old one-poll-per-frame path for comparison (which lets the ring back up).
    #[pyo3(signature = (seconds=3.0, *, budget_ms=Some(LIVE_DRAIN_BUDGET_MS as f64), drain=true))]
    fn _render_benchmark(
        &mut self,
        py: Python<'_>,
        seconds: f64,
        budget_ms: Option<f64>,
        drain: bool,
    ) -> PyResult<Py<PyDict>> {
        let decay = self.decay_ms;
        let capture = self.parked_capture()?;
        let (width, height) = (capture.width(), capture.height());
        let mut frames = 0u64;
        let mut events = 0u64;
        let mut max_backlog = 0usize;
        let result: Result<(), String> = py.detach(|| {
            let mut surface = eventcv_core::viz::RawSurface::new(width, height, decay);
            let frame_interval = std::time::Duration::from_millis(16);
            let start = std::time::Instant::now();
            while start.elapsed().as_secs_f64() < seconds {
                let frame_start = std::time::Instant::now();
                if let Some(budget_ms) = budget_ms {
                    let budget = std::time::Duration::from_secs_f64(budget_ms / 1000.0);
                    capture.drain_events_budgeted(budget, |x, y, t_us, positive| {
                        events += 1;
                        surface.stamp(x as usize, y as usize, t_us as f64 * 0.001, positive);
                    })?;
                } else if drain {
                    capture.drain_events(|x, y, t_us, positive| {
                        events += 1;
                        surface.stamp(x as usize, y as usize, t_us as f64 * 0.001, positive);
                    })?;
                } else {
                    // Old behaviour: one poll per frame stops at the first unsealed window.
                    while let Some(window) = capture.poll()? {
                        for event in window.stream.iter() {
                            events += 1;
                            surface.stamp(
                                event.x,
                                event.y,
                                event.timestamp as f64 * 0.001,
                                event.polarity,
                            );
                        }
                    }
                }
                let _image = surface.render();
                frames += 1;
                max_backlog = max_backlog.max(capture.backlog());
                let elapsed = frame_start.elapsed();
                if elapsed < frame_interval {
                    std::thread::sleep(frame_interval - elapsed);
                }
            }
            Ok(())
        });
        result.map_err(PyRuntimeError::new_err)?;

        let dict = PyDict::new(py);
        dict.set_item("frames", frames)?;
        dict.set_item("events", events)?;
        dict.set_item("events_per_s", (events as f64 / seconds).round() as u64)?;
        dict.set_item("max_backlog", max_backlog)?;
        dict.set_item("seconds", seconds)?;
        Ok(dict.unbind())
    }

    /// Stops the pump, closes the device, and finishes a `stream(record=…)` file if one is open.
    /// Idempotent; further use raises.
    ///
    /// Dropping the camera does the same thing, so a throwaway `eventcv.stream(...)` still releases
    /// the device and closes its recording — but only at whatever point Python collects it. Call
    /// this (or use `with eventcv.stream(...) as camera:`) to control when that happens.
    fn close(&mut self) -> PyResult<()> {
        self.shutdown().map_err(map_io_error)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false) // never suppress an exception raised in the `with` body
    }
}

/// Releases the camera when Python drops it without `close()` — a throwaway
/// `eventcv.stream(...).record(...)` is the common case. Without this the `Recorder` was dropped
/// with its last events unflushed and its file left to `hdf5`'s own teardown, and the USB device
/// was released at whatever point the pyclass happened to be collected.
#[cfg(feature = "camera")]
impl Drop for PyEventCamera {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("eventcv: closing the camera's recording failed: {error}");
        }
    }
}

/// Seals the partial window a stopped capture was still filling and takes anything else it had
/// queued — the tail a recording keeps when the session ends. A decode pass can seal several
/// windows while the consumer takes one per call, so this drains rather than popping once.
#[cfg(feature = "camera")]
fn drain_tail(
    capture: &mut eventcv_core::device::Capture,
) -> Vec<eventcv_core::device::CaptureWindow> {
    let mut tail: Vec<_> = capture.finish().into_iter().collect();
    while let Some(window) = capture.take_pending() {
        tail.push(window);
    }
    tail
}

#[cfg(feature = "camera")]
impl PyEventCamera {
    /// Shared body of [`close`](Self::close) and [`Drop`]: stop the pump, seal the trailing window
    /// into the recording, close the file, and release the device — in that order, so the last
    /// events reach disk and the USB claim is dropped before anything can fail. Idempotent.
    fn shutdown(&mut self) -> Result<(), IoError> {
        self.park(); // joins the pump thread, handing the camera and recording back
        let (capture, recorder) = match std::mem::replace(&mut self.state, CameraState::Closed) {
            CameraState::Idle { capture, recorder } => (Some(capture), recorder),
            _ => (None, None),
        };
        let mut capture = capture;
        let tail = capture.as_mut().map(drain_tail).unwrap_or_default();
        drop(capture); // release the device before touching the file
        let Some(mut recorder) = recorder else {
            return Ok(());
        };
        let appended = tail
            .iter()
            .try_for_each(|window| recorder.append(&window.stream));
        // Close the file even if that last append failed: a short flushed recording beats a
        // half-written one, and the append error is the one reported.
        appended.and(recorder.finish())
    }

    fn open_check(&self) -> PyResult<()> {
        match self.state {
            CameraState::Closed => Err(PyRuntimeError::new_err("camera is closed")),
            _ => Ok(()),
        }
    }

    fn pump(&self) -> Option<&capture::Pump> {
        match &self.state {
            CameraState::Pumping(pump) => Some(pump),
            _ => None,
        }
    }

    /// Starts the pump if it isn't already running, so reads come from the decode thread.
    fn pumping(&mut self) -> PyResult<&capture::Pump> {
        if matches!(self.state, CameraState::Idle { .. }) {
            let CameraState::Idle { capture, recorder } =
                std::mem::replace(&mut self.state, CameraState::Closed)
            else {
                unreachable!("checked immediately above")
            };
            self.state = CameraState::Pumping(capture::Pump::start(capture, recorder, self.mode));
        }
        self.pump()
            .ok_or_else(|| PyRuntimeError::new_err("camera is closed"))
    }

    /// Stops the pump and takes the camera back onto this thread — `show`, `record`, and `close`
    /// drive the capture directly. Counters are folded in so they survive the restart.
    fn park(&mut self) {
        if let CameraState::Pumping(pump) = &mut self.state {
            self.skipped += pump.n_skipped();
            self.overflows += pump.n_overflows();
            let (capture, recorder) = pump.stop();
            self.state = match capture {
                Some(capture) => CameraState::Idle { capture, recorder },
                None => CameraState::Closed,
            };
        }
    }

    /// Shared body of `show` and `draw_mask`: builds the live renderer and runs the viewer,
    /// returning the drawn mask in [`LiveMode::Draw`].
    fn live(
        &mut self,
        py: Python<'_>,
        representation: Option<String>,
        decay_ms: Option<f64>,
        colormap: &str,
        normalize: bool,
        mode: LiveMode,
    ) -> PyResult<Option<Vec<bool>>> {
        let default_decay = self.decay_ms;
        let stored_repr = self.repr;
        let colormap_value = parse_colormap(colormap)?;
        // The viewer decodes on a wall-clock budget and drops the surplus to hold ~60 FPS, so
        // feeding the recording from it would write a file full of holes. The recording pauses
        // instead — say so, rather than let a session come back short.
        if let Some(path) = &self.record_path {
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
                std::ffi::CString::new(format!(
                    "the recording to {path} is paused while the live viewer is open — the viewer \
                     drops events to keep up, so archiving through it would be lossy. Events \
                     during this period are not saved"
                ))?
                .as_c_str(),
                1,
            )?;
        }
        // Take ownership so the capture can move onto the viewer's own drain thread; it is restored
        // into `self` when the viewer closes, so the camera stays usable afterwards.
        self.park();
        let (capture, recorder) = match std::mem::replace(&mut self.state, CameraState::Closed) {
            CameraState::Idle { capture, recorder } => (capture, recorder),
            _ => return Err(PyRuntimeError::new_err("camera is closed")),
        };
        let (width, height) = (capture.width() as u32, capture.height() as u32);
        let title = capture.name().to_owned();

        let raw = |decay: Option<f64>| {
            LiveRenderer::Raw(eventcv_core::viz::RawSurface::new(
                width as usize,
                height as usize,
                decay.unwrap_or(default_decay),
            ))
        };
        let repr = |spec: ReprSpec| LiveRenderer::Repr {
            spec,
            colormap: colormap_value,
            normalize,
            width: width as usize,
            height: height as usize,
            window: std::time::Duration::from_millis(LIVE_REPR_WINDOW_MS),
            builder: EventStreamBuilder::new(width as usize, height as usize, 0.001),
            last_render: std::time::Instant::now(),
        };
        let renderer = match representation.as_deref() {
            Some("raw") => raw(decay_ms),
            Some(name) => repr(ReprSpec::named(name, Some(normalize))?),
            None => match stored_repr {
                Some(spec) => repr(spec),
                None => raw(decay_ms),
            },
        };

        // Draining runs on a dedicated thread that owns the capture, decoupled from the GPU thread's
        // vsync-bound present loop (see `run_live_threaded`). The camera is reclaimed and restored
        // into `self` on close so it stays usable.
        let (capture, result) =
            py.detach(move || run_live_threaded(capture, renderer, width, height, title, mode));
        self.state = match capture {
            Some(capture) => CameraState::Idle { capture, recorder },
            None => CameraState::Closed,
        };
        result.map_err(PyRuntimeError::new_err)
    }

    /// Puts `mask` on the capture — parking the pump, which owns it, first — and mirrors it here so
    /// the getter keeps working once decoding restarts.
    fn apply_mask(&mut self, mask: Option<Vec<bool>>) -> PyResult<()> {
        self.parked_capture()?
            .set_mask(mask.clone())
            .map_err(PyRuntimeError::new_err)?;
        self.mask = mask;
        Ok(())
    }

    /// Parks the pump and borrows the capture for the direct-drive paths.
    fn parked_capture(&mut self) -> PyResult<&mut eventcv_core::device::Capture> {
        self.park();
        match &mut self.state {
            CameraState::Idle { capture, .. } => Ok(capture),
            _ => Err(PyRuntimeError::new_err("camera is closed")),
        }
    }

    /// Shared poll loop behind `__next__` and `read`: waits for the next completed window and
    /// renders it (or returns `None` once `timeout` elapses, if given). Assumes the camera is open
    /// — callers handle the closed case. `Ctrl+C` surfaces as an error via `check_signals`.
    fn poll_next(
        &mut self,
        py: Python<'_>,
        timeout: Option<std::time::Duration>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let repr = self.repr;
        // Decoding, archiving, and the `latest=` skip policy all run on the pump thread; this loop
        // only collects finished windows and renders them.
        let pump = self.pumping()?;
        let deadline = timeout.map(|t| std::time::Instant::now() + t);
        loop {
            match py.detach(|| pump.next_window(READ_SLICE)) {
                Ok(Some(window)) => {
                    if window.first_after_overflow {
                        note_overflow();
                    }
                    return render_slice(py, repr, window.stream).map(Some);
                }
                Ok(None) => {} // nothing decoded yet — keep waiting
                Err(message) => return Err(PyRuntimeError::new_err(message)),
            }
            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    return Ok(None); // waited long enough without a window
                }
            }
            py.check_signals()?; // let Ctrl+C break out of the live loop
        }
    }

    /// Streams the capture straight to an [`Hdf5EventSink`], appending each polled window to disk
    /// and flushing about once a second, so nothing accumulates in memory. Stops on the deadline or
    /// `Ctrl+C`, keeping whatever was already written.
    #[cfg(feature = "hdf5")]
    fn record_streaming(
        &mut self,
        py: Python<'_>,
        path: &str,
        seconds: Option<f64>,
        compression: Option<u8>,
    ) -> PyResult<usize> {
        let capture = self.parked_capture()?;
        let mut sink =
            eventcv_core::io::Hdf5EventSink::open(path, compression).map_err(map_io_error)?;
        // The loop is kept separate so the sink is flushed and closed on *both* exits: a write
        // error used to return through `?` and drop it unfinished, losing the last second.
        let result = Self::fill_sink(py, capture, &mut sink, seconds);
        let count = sink.n_events();
        let closed = py.detach(|| sink.finish()).map_err(map_io_error);
        result?; // the error that ended the recording is the one worth raising
        closed?;
        Ok(count)
    }

    /// Polls the camera into `sink` until the deadline or `Ctrl+C`, flushing about once a second.
    #[cfg(feature = "hdf5")]
    fn fill_sink(
        py: Python<'_>,
        capture: &mut eventcv_core::device::Capture,
        sink: &mut eventcv_core::io::Hdf5EventSink,
        seconds: Option<f64>,
    ) -> PyResult<()> {
        let deadline =
            seconds.map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s));
        let mut last_flush = std::time::Instant::now();
        loop {
            match py.detach(|| capture.poll()) {
                Ok(Some(window)) => {
                    if window.first_after_overflow {
                        note_overflow();
                    }
                    py.detach(|| sink.append(&window.stream))
                        .map_err(map_io_error)?;
                }
                Ok(None) => {}
                Err(message) => return Err(PyRuntimeError::new_err(message)),
            }
            if last_flush.elapsed() >= RECORD_FLUSH_INTERVAL {
                py.detach(|| sink.flush()).map_err(map_io_error)?;
                last_flush = std::time::Instant::now();
            }
            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }
            // Ctrl+C stops the recording and keeps what was captured, rather than discarding it.
            if py.check_signals().is_err() {
                break;
            }
        }
        // The recording ends here, so the window still filling belongs in it — without this the
        // file stopped at the last whole `dt_ms` boundary and dropped up to one window of events.
        for window in drain_tail(capture) {
            py.detach(|| sink.append(&window.stream))
                .map_err(map_io_error)?;
        }
        Ok(())
    }

    /// Buffers the whole recording in memory and writes it once at the end via `save_stream`. Used
    /// for formats that can't be appended incrementally (npz/txt/bag).
    fn record_buffered(
        &mut self,
        py: Python<'_>,
        path: &str,
        seconds: Option<f64>,
    ) -> PyResult<usize> {
        let capture = self.parked_capture()?;
        let mut builder = EventStreamBuilder::new(capture.width(), capture.height(), 0.001);
        let deadline =
            seconds.map(|s| std::time::Instant::now() + std::time::Duration::from_secs_f64(s));
        loop {
            match py.detach(|| capture.poll()) {
                Ok(Some(window)) => {
                    if window.first_after_overflow {
                        note_overflow();
                    }
                    for event in window.stream.iter() {
                        builder.push(
                            event.x as u16,
                            event.y as u16,
                            event.timestamp as i64,
                            event.polarity,
                        );
                    }
                }
                Ok(None) => {}
                Err(message) => return Err(PyRuntimeError::new_err(message)),
            }
            if let Some(deadline) = deadline {
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }
            // Ctrl+C stops the recording and saves what was captured, rather than discarding it.
            if py.check_signals().is_err() {
                break;
            }
        }
        // The recording ends here, so the window still filling belongs in it.
        for window in drain_tail(capture) {
            for event in window.stream.iter() {
                builder.push(
                    event.x as u16,
                    event.y as u16,
                    event.timestamp as i64,
                    event.polarity,
                );
            }
        }
        let count = builder.len();
        let stream = builder.build();
        let options = SaveOptions::default();
        py.detach(|| eventcv_core::io::save_stream(path, &stream, &options))
            .map_err(map_io_error)?;
        Ok(count)
    }
}

/// Validates the source-side caps before the camera is opened, so a bad rate or rectangle raises
/// the same way with or without hardware attached.
#[cfg(feature = "camera")]
fn parse_limits(
    max_event_rate: Option<f64>,
    roi: Option<(i64, i64, i64, i64)>,
) -> PyResult<eventcv_core::device::Limits> {
    let max_event_rate = max_event_rate
        .map(|rate| {
            if !rate.is_finite() || rate < 1.0 {
                return Err(PyValueError::new_err(
                    "max_event_rate must be at least 1 event per second",
                ));
            }
            Ok(rate as u64)
        })
        .transpose()?;
    let roi = roi
        .map(|(x0, y0, width, height)| {
            if x0 < 0 || y0 < 0 {
                return Err(PyValueError::new_err(
                    "roi must be (x0, y0, width, height) with a non-negative origin",
                ));
            }
            if width < 1 || height < 1 {
                return Err(PyValueError::new_err(
                    "roi width and height must be at least 1",
                ));
            }
            Ok((x0 as usize, y0 as usize, width as usize, height as usize))
        })
        .transpose()?;
    Ok(eventcv_core::device::Limits {
        max_event_rate,
        roi,
    })
}

/// Parses `adaptive_bias=` into a controller configuration: `True` for the reference tuning,
/// `False`/`None` for off, or a dict overriding individual fields of that tuning. Validated here so
/// bad options raise before the device is touched, with or without hardware attached.
#[cfg(feature = "camera")]
fn parse_adaptive_bias(
    adaptive_bias: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<eventcv_core::bias::BiasOverrides>> {
    use eventcv_core::bias::BiasOverrides;

    let Some(value) = adaptive_bias else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    // `bool` first: in Python `True` is also an int, so a looser extract would swallow it.
    if let Ok(enabled) = value.extract::<bool>() {
        return Ok(enabled.then(BiasOverrides::default));
    }
    let Ok(options) = value.cast::<PyDict>() else {
        return Err(PyTypeError::new_err(
            "adaptive_bias must be True, False, or a dict of controller options",
        ));
    };
    // Only what the caller names is overridden; the rest comes from the camera's own defaults once
    // it is open, since the sensible rates and register ranges differ per sensor.
    let mut overrides = BiasOverrides::default();
    for (key, option) in options.iter() {
        match key.extract::<String>()?.as_str() {
            "period_ms" => {
                let ms = option.extract::<f64>()?;
                if !ms.is_finite() || ms <= 0.0 {
                    return Err(PyValueError::new_err("period_ms must be greater than zero"));
                }
                overrides.period = Some(std::time::Duration::from_secs_f64(ms / 1000.0));
            }
            "target_rate" => overrides.target_rate = Some(option.extract()?),
            "throttle_range" => overrides.throttle_range = Some(option.extract()?),
            "max_slew" => overrides.max_slew = Some(option.extract()?),
            "calibrate" => overrides.calibrate = Some(option.extract()?),
            "patience" => overrides.patience = Some(option.extract()?),
            "step" => overrides.step = Some(option.extract()?),
            "limits" => overrides.limits = Some(parse_bias_limits(&option)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown adaptive_bias option {other:?} — expected any of period_ms, \
                     target_rate, calibrate, throttle_range, max_slew, patience, step, limits"
                )))
            }
        }
    }
    overrides.validate().map_err(PyValueError::new_err)?;
    Ok(Some(overrides))
}

/// Parses `limits`: one `(low, high)` for every bias, or a dict naming them individually. The
/// per-bias form exists because a sensor whose thresholds are defined against a shared reference
/// (the IMX636's `diff`) needs the ON and OFF bounds on opposite sides of it.
#[cfg(feature = "camera")]
fn parse_bias_limits(value: &Bound<'_, PyAny>) -> PyResult<eventcv_core::bias::BiasLimits> {
    use eventcv_core::bias::BiasLimits;

    if let Ok((low, high)) = value.extract::<(u16, u16)>() {
        return Ok(BiasLimits::uniform(low, high));
    }
    let Ok(per_bias) = value.cast::<PyDict>() else {
        return Err(PyTypeError::new_err(
            "limits must be (low, high) or a dict of per-bias (low, high) ranges",
        ));
    };
    // Unnamed biases keep the sensor's default, which is only known once the camera is open — so
    // start from the widest range and let the resolved config narrow nothing it wasn't asked to.
    let mut limits = BiasLimits::uniform(0, u16::MAX);
    for (key, range) in per_bias.iter() {
        let range = range.extract::<(u16, u16)>()?;
        match key.extract::<String>()?.as_str() {
            "photoreceptor" => limits.photoreceptor = range,
            "follower" => limits.follower = range,
            "on_threshold" => limits.on_threshold = range,
            "off_threshold" => limits.off_threshold = range,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown limits entry {other:?} — expected any of photoreceptor, follower, \
                     on_threshold, off_threshold"
                )))
            }
        }
    }
    Ok(limits)
}

/// Lists every connected, supported event camera as a list of dicts (`kind`, `name`, `serial`,
/// `bus`, `address`, `speed`). A `serial` can be passed to `eventcv.stream(serial=…)`.
#[cfg(feature = "camera")]
#[pyfunction]
fn list_cameras(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    let listing = py.detach(eventcv_core::device::list_cameras);
    // A permissions failure means a camera is probably attached and unreadable — worth saying, but
    // not worth making "what is plugged in?" a fatal question. Everything else (no USB subsystem,
    // nothing attached) is an ordinary empty answer and passes silently.
    if let Some(message) = listing.warning {
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            std::ffi::CString::new(message)?.as_c_str(),
            1,
        )?;
    }
    listing
        .cameras
        .into_iter()
        .map(|info| {
            let dict = PyDict::new(py);
            dict.set_item("kind", info.kind)?;
            dict.set_item("name", info.name)?;
            dict.set_item("serial", info.serial)?;
            dict.set_item("bus", info.bus_number)?;
            dict.set_item("address", info.address)?;
            dict.set_item("speed", info.speed)?;
            Ok(dict.unbind())
        })
        .collect()
}

/// Opens a live event camera. `serial` selects a specific device (from `list_cameras`); `None`
/// opens the first found. Windowing mirrors `eventcv.open`: `dt_ms` for fixed-duration windows or
/// `max_events` for fixed-count (mutually exclusive; default `dt_ms=30`). `repr` sets what iterating
/// the camera yields (a representation name → `EventFrame`s; omit for raw `EventStream`s), with the
/// same per-representation options `EventReader.with_repr` takes (`bins`, `window_ms`, `tau_ms`,
/// `max_window_ms`, `window`, `normalize`, `pct`, `white_frame`). `record` archives every window's
/// raw events to an HDF5
/// file as it is read (`compression` is its optional gzip level). `latest` keeps a slow loop on live
/// data by handing back the newest window instead of the oldest. `max_event_rate` and `roi` cap what
/// the sensor emits, in hardware. `adaptive_bias` holds the event rate steady across changing light
/// by retuning the sensor's biases as it runs. `decay_ms` is the default fade for the raw `show()`
/// view.
#[cfg(feature = "camera")]
#[pyfunction]
#[pyo3(signature = (
    serial=None, *, dt_ms=None, dt_s=None, dt_us=None, dt_ns=None, max_events=None, repr=None,
    bins=None, window_ms=None, window_s=None, window_us=None, window_ns=None,
    tau_ms=None, tau_s=None, tau_us=None, tau_ns=None,
    max_window_ms=None, max_window_s=None, max_window_us=None, max_window_ns=None,
    window=None, normalize=None, pct=None, white_frame=None, record=None,
    compression=None, latest=false, max_event_rate=None, roi=None, mask=None, adaptive_bias=None,
    decay_ms=None, decay_s=None, decay_us=None, decay_ns=None
))]
#[allow(clippy::too_many_arguments)]
fn stream(
    py: Python<'_>,
    serial: Option<String>,
    dt_ms: Option<f64>,
    dt_s: Option<f64>,
    dt_us: Option<f64>,
    dt_ns: Option<f64>,
    max_events: Option<usize>,
    repr: Option<String>,
    bins: Option<i64>,
    window_ms: Option<f64>,
    window_s: Option<f64>,
    window_us: Option<f64>,
    window_ns: Option<f64>,
    tau_ms: Option<f64>,
    tau_s: Option<f64>,
    tau_us: Option<f64>,
    tau_ns: Option<f64>,
    max_window_ms: Option<f64>,
    max_window_s: Option<f64>,
    max_window_us: Option<f64>,
    max_window_ns: Option<f64>,
    window: Option<i64>,
    normalize: Option<bool>,
    pct: Option<f64>,
    white_frame: Option<bool>,
    record: Option<String>,
    compression: Option<u8>,
    latest: bool,
    max_event_rate: Option<f64>,
    roi: Option<(i64, i64, i64, i64)>,
    mask: Option<&Bound<'_, PyAny>>,
    adaptive_bias: Option<&Bound<'_, PyAny>>,
    decay_ms: Option<f64>,
    decay_s: Option<f64>,
    decay_us: Option<f64>,
    decay_ns: Option<f64>,
) -> PyResult<PyEventCamera> {
    let dt = resolve_duration_us("dt", dt_s, dt_ms, dt_us, dt_ns)?.map(|us| us as f64 / 1000.0);
    let decay_ms =
        resolve_ms("decay", decay_s, decay_ms, decay_us, decay_ns)?.unwrap_or(DEFAULT_SPAN_MS);
    let windowing = match (dt, max_events) {
        (Some(_), Some(_)) => return Err(PyValueError::new_err("pass dt or max_events, not both")),
        (Some(dt), None) => eventcv_core::device::Window::Duration(dt),
        (None, Some(count)) => eventcv_core::device::Window::Count(count),
        (None, None) => eventcv_core::device::Window::Duration(DEFAULT_SPAN_MS),
    };
    let repr = match repr.as_deref() {
        None | Some("raw") => None,
        Some(name) => {
            // Unset time spans follow the capture window, so a live representation covers exactly
            // the events it is handed: `stream(dt_ms=50, repr="tencode")` encodes all 50 ms rather
            // than the 30 ms default, which would silently drop the oldest 20 ms of every window.
            // An explicit `window_ms=`/`tau_ms=`/`max_window_ms=` always wins.
            let span = match windowing {
                eventcv_core::device::Window::Duration(dt) => Some(dt),
                eventcv_core::device::Window::Count(_) => None,
            };
            Some(ReprSpec::new(
                name,
                bins,
                resolve_ms("window", window_s, window_ms, window_us, window_ns)?.or(span),
                resolve_ms("tau", tau_s, tau_ms, tau_us, tau_ns)?.or(span),
                resolve_ms(
                    "max_window",
                    max_window_s,
                    max_window_ms,
                    max_window_us,
                    max_window_ns,
                )?
                .or(span),
                window,
                normalize,
                pct,
                white_frame,
            )?)
        }
    };
    #[cfg(feature = "hdf5")]
    if let Some(path) = record.as_deref() {
        Recorder::validate(path, compression)?;
    }
    #[cfg(not(feature = "hdf5"))]
    if record.is_some() {
        let _ = compression;
        return Err(PyRuntimeError::new_err(
            "record= needs HDF5 support, which this build of eventcv was compiled without",
        ));
    }
    let limits = parse_limits(max_event_rate, roi)?;
    let bias = parse_adaptive_bias(adaptive_bias)?;
    // Read the mask before the device is touched, so a bad array raises the same way with or
    // without hardware; it can only be *sized* against the sensor once the camera is open.
    let mask = mask.map(mask_array).transpose()?;
    let mut capture = py
        .detach(|| eventcv_core::device::Capture::open(serial.as_deref(), windowing, limits, bias))
        .map_err(PyRuntimeError::new_err)?;
    // A `roi=` this sensor can't block on-chip is honoured as a host-side mask instead — the same
    // events are dropped, but they still cross the cable, so say so rather than let the caller
    // believe the source was capped.
    if let Some((rect, eventcv_core::device::RoiPlacement::Host)) = capture.roi() {
        let (x0, y0, width, height) = rect;
        PyErr::warn(
            py,
            &py.get_type::<pyo3::exceptions::PyRuntimeWarning>(),
            std::ffi::CString::new(format!(
                "roi=({x0}, {y0}, {width}, {height}) is filtered on the host: the {} has no \
                 on-chip region masks, so those events are still sent over USB and decoded before \
                 being dropped",
                capture.name(),
            ))?
            .as_c_str(),
            1,
        )?;
    }
    let mask = match mask {
        Some((flat, width, height)) => {
            check_mask_sensor((width, height), (capture.width(), capture.height()))?;
            // The host-side `roi=` fallback also arrives as a capture mask; combine rather than
            // replace, so `roi=` and `mask=` together keep only what both allow.
            let flat = match capture.mask() {
                Some(existing) => existing
                    .iter()
                    .zip(&flat)
                    .map(|(&a, &b)| a && b)
                    .collect::<Vec<bool>>(),
                None => flat,
            };
            capture
                .set_mask(Some(flat.clone()))
                .map_err(PyRuntimeError::new_err)?;
            Some(flat)
        }
        None => capture.mask().map(<[bool]>::to_vec),
    };
    // Created only now, so a failed camera open never leaves an empty recording behind.
    #[cfg(feature = "hdf5")]
    let recorder = record
        .as_deref()
        .map(|path| Recorder::open(path, compression))
        .transpose()?;
    #[cfg(not(feature = "hdf5"))]
    let recorder = None;
    Ok(PyEventCamera {
        name: capture.name().to_owned(),
        serial: capture.serial().to_owned(),
        sensor_size: (capture.width(), capture.height()),
        mask,
        roi: capture.roi(),
        record_path: recorder.as_ref().and(record),
        state: CameraState::Idle { capture, recorder },
        repr,
        decay_ms,
        mode: if latest {
            capture::Backpressure::Latest
        } else {
            capture::Backpressure::Buffer
        },
        skipped: 0,
        overflows: 0,
    })
}

/// Records a camera to `path` in one call: opens the first (or `serial`-selected) device, captures
/// for `seconds`, closes it, and returns the number of events saved.
///
/// The one-shot form of `stream(...).record(...)`, and the one to prefer when a script only wants a
/// file: the camera is closed before this returns, so the recording is complete and the device is
/// free the moment the next line runs. Everything `stream` takes about *what the sensor sends*
/// (`dt_ms`, `roi`, `mask`, `max_event_rate`, `adaptive_bias`) applies; the viewer and
/// representation options do not, since nothing is displayed.
#[cfg(feature = "camera")]
#[pyfunction]
#[pyo3(signature = (
    path, *, seconds=None, serial=None, dt_ms=None, dt_s=None, dt_us=None, dt_ns=None,
    max_events=None, compression=None, max_event_rate=None, roi=None, mask=None, adaptive_bias=None
))]
#[allow(clippy::too_many_arguments)]
fn record(
    py: Python<'_>,
    path: &str,
    seconds: Option<f64>,
    serial: Option<String>,
    dt_ms: Option<f64>,
    dt_s: Option<f64>,
    dt_us: Option<f64>,
    dt_ns: Option<f64>,
    max_events: Option<usize>,
    compression: Option<u8>,
    max_event_rate: Option<f64>,
    roi: Option<(i64, i64, i64, i64)>,
    mask: Option<&Bound<'_, PyAny>>,
    adaptive_bias: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    // Everything passed as `None` here is one of `stream`'s representation, viewer, or own-recorder
    // options — all off, because this call archives raw events and displays nothing.
    let mut camera = stream(
        py,
        serial,
        dt_ms,
        dt_s,
        dt_us,
        dt_ns,
        max_events,
        /* repr */ None,
        /* bins */ None,
        /* window_* */ None,
        None,
        None,
        None,
        /* tau_* */ None,
        None,
        None,
        None,
        /* max_window_* */ None,
        None,
        None,
        None,
        /* window */ None,
        /* normalize */ None,
        /* pct */ None,
        /* white_frame */ None,
        /* record */ None,
        /* compression */ None,
        /* latest */ false,
        max_event_rate,
        roi,
        mask,
        adaptive_bias,
        /* decay_* */ None,
        None,
        None,
        None,
    )?;
    let recorded = camera.record(py, path, seconds, compression);
    // Close before raising, so a failed recording still releases the device.
    let closed = camera.close();
    let count = recorded?;
    closed?;
    Ok(count)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Baked in from this crate's Cargo.toml, so it describes the extension actually loaded rather
    // than whatever distribution metadata happens to sit beside it — which is the version worth
    // having in a bug report. `tests/test_version.py` keeps it in step with pyproject.toml.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Which optional features this build has, so `eventcv --version` can say whether HDF5 reading,
    // USB camera streaming and ONNX inference are actually compiled in (the published wheels have
    // all three).
    m.add("__features__", FEATURES)?;
    m.add_class::<PyEventStream>()?;
    m.add_class::<PyEventFrame>()?;
    m.add_class::<PyEventPointSet>()?;
    m.add_class::<PyEventReader>()?;
    m.add_class::<PyWindowIterator>()?;
    m.add_class::<PyPolarity>()?;
    m.add_class::<PyCamera>()?;
    m.add_class::<PyFeast>()?;
    m.add_class::<PyTracker>()?;
    m.add_class::<PyUdpSender>()?;
    m.add_class::<PyUdpReceiver>()?;
    #[cfg(feature = "onnx")]
    m.add_class::<PyModel>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load_frame, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    #[cfg(feature = "onnx")]
    m.add_function(wrap_pyfunction!(reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(rect_mask, m)?)?;
    m.add_function(wrap_pyfunction!(ellipse_mask, m)?)?;
    m.add_function(wrap_pyfunction!(circle_mask, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_mask, m)?)?;
    m.add_function(wrap_pyfunction!(save_mask, m)?)?;
    m.add_function(wrap_pyfunction!(load_mask, m)?)?;
    #[cfg(feature = "hdf5")]
    {
        m.add_class::<PyFrameSink>()?;
        m.add_class::<PyEventSink>()?;
    }
    #[cfg(feature = "camera")]
    {
        m.add_class::<PyEventCamera>()?;
        m.add_function(wrap_pyfunction!(list_cameras, m)?)?;
        m.add_function(wrap_pyfunction!(stream, m)?)?;
        m.add_function(wrap_pyfunction!(record, m)?)?;
    }
    Ok(())
}
