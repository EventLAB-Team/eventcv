mod viewer;

use std::sync::{Arc, Mutex};

use eventcv_core::{
    camera::Camera,
    cluster::ClusterError,
    flow::FlowError,
    image::{PoolingMethod, ResizeError},
    io::{open as open_reader, ColumnOrder, IoError, LoadOptions, Reader, SaveOptions, TimeUnit},
    representation::{
        AveragedTimeSurface, Binary, EventCount, EventFrame, EventFrameData, EventPointSet, Mcts,
        PointSet, Polarity, Representation, RepresentationError, Tencode, TimeSurface, VoxelGrid,
    },
    viz::Colormap,
    EventStream, EventStreamBuilder,
};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

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

    /// Saves the stream to `path`, format chosen by extension (`.npz`/`.txt`/`.h5`/`.bag`) —
    /// the counterpart of `eventcv.load`. npz/HDF5/rosbag round-trip exactly; `topic` names
    /// the rosbag connection.
    #[pyo3(signature = (path, *, topic=None))]
    fn save(&self, py: Python<'_>, path: &str, topic: Option<String>) -> PyResult<()> {
        let options = SaveOptions {
            topic,
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

    /// Keeps events where the `(H, W)` boolean mask is `True`.
    fn mask(&self, py: Python<'_>, mask: PyReadonlyArray2<bool>) -> PyEventStream {
        let view = mask.as_array();
        let (h, w) = (view.shape()[0], view.shape()[1]);
        let flat: Vec<bool> = view.iter().copied().collect();
        self.wrap(py.detach(|| self.inner.mask(&flat, w, h)))
    }

    /// Keeps events whose timestamp lies in the half-open window `[t0, t1)` (microseconds).
    fn time_window(&self, py: Python<'_>, t0: i64, t1: i64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.time_window(t0, t1)))
    }

    /// Shifts every timestamp by `dt` microseconds.
    fn time_shift(&self, py: Python<'_>, dt: i64) -> PyEventStream {
        self.wrap(py.detach(|| self.inner.time_shift(dt)))
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

    /// Dense Lucas-Kanade optical flow on the time surface. Returns a two-channel `(flow_x,
    /// flow_y)` frame in pixels/ms; `window` is the half-width of the least-squares neighbourhood.
    #[pyo3(signature = (*, window=3))]
    fn optical_flow(&self, py: Python<'_>, window: usize) -> PyResult<PyEventFrame> {
        py.detach(|| self.inner.optical_flow(window))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_flow_error)
    }

    #[pyo3(signature = (representation=None, *, normalize=true, binary=false))]
    fn flatten(
        &self,
        py: Python<'_>,
        representation: Option<&Bound<'_, PyAny>>,
        normalize: bool,
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

    #[pyo3(signature = (*, bins=9, window_ms=30.0))]
    fn voxel(&self, py: Python<'_>, bins: i64, window_ms: f64) -> PyResult<PyEventFrame> {
        let bins =
            usize::try_from(bins).map_err(|_| PyValueError::new_err("bins must be at least 1"))?;
        py.detach(|| VoxelGrid::new(bins, window_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    #[pyo3(signature = (*, tau_ms=30.0))]
    fn tsurf(&self, py: Python<'_>, tau_ms: f64) -> PyResult<PyEventFrame> {
        py.detach(|| TimeSurface::new(tau_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Averaged time surface — the per-pixel mean of `exp(-age/tau_ms)` over all events
    /// (two polarity channels, float32). Brighter where activity recurs; see `tsurf`.
    #[pyo3(signature = (*, tau_ms=30.0))]
    fn atsurf(&self, py: Python<'_>, tau_ms: f64) -> PyResult<PyEventFrame> {
        py.detach(|| AveragedTimeSurface::new(tau_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Event-count image — one channel of total events per pixel (both polarities). `uint64`
    /// counts, or `uint8` rescaled to the busiest pixel when `normalize=True`.
    #[pyo3(signature = (*, normalize=true))]
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

    #[pyo3(signature = (*, window_ms=30.0))]
    fn tencode(&self, py: Python<'_>, window_ms: f64) -> PyResult<PyEventFrame> {
        py.detach(|| Tencode::new(window_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    #[pyo3(signature = (*, max_window_ms=30.0))]
    fn mcts(&self, py: Python<'_>, max_window_ms: f64) -> PyResult<PyEventFrame> {
        py.detach(|| Mcts::new(max_window_ms).generate(&self.inner))
            .map(|inner| PyEventFrame { inner })
            .map_err(map_representation_error)
    }

    /// Opens the interactive viewer on this stream. Pass a representation **name** to choose what
    /// to show — `stream.view("flow")`, `stream.view("count")`, `stream.view("voxel")`, … — or
    /// omit it to use the stream's stored representation (from `open(repr=…)`), falling back to
    /// the polarity image. (Equivalent to `stream.<repr>().view()`.)
    #[pyo3(signature = (representation=None, *, colormap="viridis", normalize=true))]
    fn view(
        &self,
        py: Python<'_>,
        representation: Option<&Bound<'_, PyAny>>,
        colormap: &str,
        normalize: bool,
    ) -> PyResult<()> {
        let frame = match representation {
            Some(representation) => self.generate(py, representation, normalize)?,
            None => self.default_frame(py, normalize)?,
        };
        frame.view(py, colormap, normalize)
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
    fn default_frame(&self, py: Python<'_>, normalize: bool) -> PyResult<PyEventFrame> {
        match self.repr {
            Some(spec) => py
                .detach(|| spec.generate(&self.inner))
                .map(|inner| PyEventFrame { inner })
                .map_err(map_representation_error),
            None => self.generate_polarity(py, Polarity::new(normalize)),
        }
    }

    fn generate(
        &self,
        py: Python<'_>,
        representation: &Bound<'_, PyAny>,
        normalize: bool,
    ) -> PyResult<PyEventFrame> {
        if representation.extract::<PyRef<'_, PyPolarity>>().is_ok() {
            return self.generate_polarity(py, Polarity::new(normalize));
        }
        // A representation *name* ("count", "flow", "voxel", …) renders that repr with its
        // defaults — so `stream.view("flow")` / `stream.flatten("count")` read naturally.
        if let Ok(name) = representation.extract::<String>() {
            let spec = ReprSpec::new(&name, None, None, None, None, None, normalize)?;
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

/// `None` or `"auto"` means infer the unit from the data; anything else is explicit.
fn parse_time_unit(time_unit: Option<&str>) -> PyResult<Option<TimeUnit>> {
    Ok(Some(match time_unit {
        None | Some("auto") => return Ok(None),
        Some("seconds") | Some("s") => TimeUnit::Seconds,
        Some("milliseconds") | Some("ms") => TimeUnit::Milliseconds,
        Some("microseconds") | Some("us") => TimeUnit::Microseconds,
        Some("nanoseconds") | Some("ns") => TimeUnit::Nanoseconds,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unsupported time_unit: {other}"
            )))
        }
    }))
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

fn ms_to_us(ms: f64) -> i64 {
    (ms * 1000.0).round() as i64
}

fn us_to_ms(us: i64) -> f64 {
    us as f64 / 1000.0
}

#[pyfunction]
#[pyo3(signature = (path, *, sensor_size=None, time_unit=None, order="txyp", topic=None, max_events=None))]
fn load(
    py: Python<'_>,
    path: &str,
    sensor_size: Option<(i64, i64)>,
    time_unit: Option<&str>,
    order: &str,
    topic: Option<String>,
    max_events: Option<i64>,
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
    };
    py.detach(|| eventcv_core::io::load(path, options))
        .map(|inner| PyEventStream { inner, repr: None })
        .map_err(map_io_error)
}

fn parse_dt_ms(dt_ms: Option<f64>) -> PyResult<Option<i64>> {
    match dt_ms {
        None => Ok(None),
        Some(ms) if ms.is_nan() || ms <= 0.0 => {
            Err(PyValueError::new_err("dt_ms must be positive"))
        }
        Some(ms) => Ok(Some(ms_to_us(ms).max(1))),
    }
}

/// Number of `dt`-length (µs) windows covering `[t_min, t_max]`; 0 when empty.
fn slice_total(n_events: usize, span: (i64, i64), dt: i64) -> usize {
    if n_events == 0 {
        return 0;
    }
    let (lo, hi) = span;
    ((hi - lo) / dt + 1) as usize
}

#[pyfunction]
#[pyo3(signature = (path, *, dt_ms=None, repr=None, sensor_size=None, time_unit=None, order="txyp", topic=None))]
#[allow(clippy::too_many_arguments)]
fn open(
    py: Python<'_>,
    path: &str,
    dt_ms: Option<f64>,
    repr: Option<&str>,
    sensor_size: Option<(i64, i64)>,
    time_unit: Option<&str>,
    order: &str,
    topic: Option<String>,
) -> PyResult<PyEventReader> {
    let dt_us = parse_dt_ms(dt_ms)?;
    let repr = repr
        .map(|name| ReprSpec::new(name, None, None, None, None, None, true))
        .transpose()?;
    let options = LoadOptions {
        sensor_size: parse_sensor_size(sensor_size)?,
        time_unit: parse_time_unit(time_unit)?,
        order: parse_order(order)?,
        topic,
        max_events: None,
    };
    let reader = py
        .detach(|| open_reader(path, options))
        .map_err(map_io_error)?;
    Ok(PyEventReader {
        inner: Arc::new(Mutex::new(reader)),
        dt_us,
        repr,
        slice_op: None,
    })
}

/// Lazy, seekable handle over a file's events — the `VideoCapture` to `load`'s
/// `imread`. Slices are fetched on demand (HDF5 by binary-searching its timestamps on
/// disk), so multi-GB files never need to be fully resident. Opening with `dt_ms` fixes
/// a frame duration so `slice(n)` returns the `n`-th frame (like seeking a video).
#[pyclass(name = "EventReader", frozen)]
struct PyEventReader {
    inner: Arc<Mutex<Reader>>,
    /// Fixed slice duration in µs from `open(dt_ms=…)`; enables integer `slice(n)`.
    dt_us: Option<i64>,
    /// Per-slice dense rendering set by `open(repr=…)` / `with_repr` — makes the reader a
    /// map-style dataset (`reader[i]` → `[C, H, W]`, `batch` → `[B, C, H, W]`).
    repr: Option<ReprSpec>,
    /// Per-slice stream op set by `efast`/`harris_corners` — applied to every slice before any
    /// `repr`, so corner detection composes with `slice`/`windows`/`with_repr`.
    slice_op: Option<SliceOp>,
}

/// A per-slice `EventStream` → `EventStream` operation a reader carries (Phase 5 corner
/// detectors). Applied off-GIL to each fetched slice.
#[derive(Clone, Copy)]
enum SliceOp {
    Efast,
    Harris { threshold: f64 },
}

impl SliceOp {
    fn apply(self, stream: &EventStream) -> EventStream {
        match self {
            Self::Efast => stream.efast(),
            Self::Harris { threshold } => stream.harris_corners(threshold),
        }
    }
}

/// Applies a reader's per-slice op (if any) to a freshly fetched slice.
fn apply_slice_op(op: Option<SliceOp>, stream: EventStream) -> EventStream {
    match op {
        Some(op) => op.apply(&stream),
        None => stream,
    }
}

#[pymethods]
impl PyEventReader {
    /// `n_slices` in the `dt_ms` frame-dataset mode (so `len(reader) == len(reader[:])`),
    /// else the raw event count. Lets `DataLoader` iterate the reader directly.
    fn __len__(&self) -> PyResult<usize> {
        match self.dt_us {
            Some(dt) => {
                let guard = self.inner.lock().unwrap();
                Ok(slice_total(guard.n_events(), guard.time_span(), dt))
            }
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

    #[getter]
    fn duration_ms(&self) -> f64 {
        let (lo, hi) = self.inner.lock().unwrap().time_span();
        us_to_ms(hi - lo)
    }

    /// Number of fixed `dt_ms` slices spanning the recording (requires `open(dt_ms=…)`).
    #[getter]
    fn n_slices(&self) -> PyResult<usize> {
        let dt = self.require_dt()?;
        let guard = self.inner.lock().unwrap();
        Ok(slice_total(guard.n_events(), guard.time_span(), dt))
    }

    /// The fixed slice duration set at `open`, or `None` if it was not given.
    #[getter]
    fn dt_ms(&self) -> Option<f64> {
        self.dt_us.map(us_to_ms)
    }

    /// One slice as an `EventStream`. With a positional index `n` (requires
    /// `open(dt_ms=…)`), returns the `n`-th fixed `dt_ms` frame measured from the
    /// recording start — `[t_min + n·dt, t_min + (n+1)·dt)`; negative `n` counts from
    /// the end. Otherwise returns the half-open time window `[t0_ms, t1_ms)`, with
    /// omitted bounds extending to the recording's start / end.
    #[pyo3(signature = (index=None, *, t0_ms=None, t1_ms=None))]
    fn slice(
        &self,
        py: Python<'_>,
        index: Option<i64>,
        t0_ms: Option<f64>,
        t1_ms: Option<f64>,
    ) -> PyResult<PyEventStream> {
        let (t0, t1) = if let Some(index) = index {
            if t0_ms.is_some() || t1_ms.is_some() {
                return Err(PyValueError::new_err(
                    "pass either a slice index or t0_ms/t1_ms, not both",
                ));
            }
            self.window_for_index(index)?
        } else {
            let (lo, hi) = self.inner.lock().unwrap().time_span();
            (
                t0_ms.map_or(lo, ms_to_us),
                t1_ms.map_or(hi.saturating_add(1), ms_to_us), // +1 keeps the last event
            )
        };
        self.fetch_time(py, t0, t1)
    }

    /// `reader[n]` — the `n`-th `dt_ms` frame (requires `open(dt_ms=…)`). Without a
    /// representation it returns the raw `EventStream` (like `slice(n)`); with one set
    /// (`open(repr=…)` / `with_repr`) it returns the dense `[C, H, W]` array, so a
    /// `torch.utils.data.DataLoader` can collate the reader directly.
    fn __getitem__(&self, py: Python<'_>, index: i64) -> PyResult<Py<PyAny>> {
        let (t0, t1) = self.window_for_index(index)?;
        let stream = self.fetch_time(py, t0, t1)?;
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
    #[pyo3(signature = (repr, *, bins=None, window_ms=None, tau_ms=None, max_window_ms=None, window=None, normalize=true))]
    #[allow(clippy::too_many_arguments)]
    fn with_repr(
        &self,
        repr: &str,
        bins: Option<i64>,
        window_ms: Option<f64>,
        tau_ms: Option<f64>,
        max_window_ms: Option<f64>,
        window: Option<i64>,
        normalize: bool,
    ) -> PyResult<PyEventReader> {
        let spec = ReprSpec::new(
            repr,
            bins,
            window_ms,
            tau_ms,
            max_window_ms,
            window,
            normalize,
        )?;
        Ok(PyEventReader {
            inner: Arc::clone(&self.inner),
            dt_us: self.dt_us,
            repr: Some(spec),
            slice_op: self.slice_op,
        })
    }

    // ---- Per-slice feature detection (Phase 5). Each returns a new reader that applies the
    // corner detector to every slice, so it composes with slice/windows/with_repr/batch:
    //   corners = reader.efast()          # reader whose slices are corner sub-streams
    //   corners.slice(500).count()        # corner count image for frame 500
    //   corners.with_repr("count")[500]   # same, as a dense [C, H, W] array

    /// Returns a new reader whose every slice is passed through [`EventStream::efast`], so the
    /// reader yields corner sub-streams (chain `.count()` / `with_repr` to visualise them).
    fn efast(&self) -> PyEventReader {
        self.with_slice_op(SliceOp::Efast)
    }

    /// Returns a new reader whose every slice is passed through [`EventStream::harris_corners`].
    #[pyo3(signature = (threshold=0.0))]
    fn harris_corners(&self, threshold: f64) -> PyEventReader {
        self.with_slice_op(SliceOp::Harris { threshold })
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
            let (t0, t1) = self.window_for_index(index)?;
            let reader = Arc::clone(&self.inner);
            let op = self.slice_op;
            let stream = py
                .detach(move || {
                    reader
                        .lock()
                        .unwrap()
                        .slice_time(t0, t1)
                        .map(|stream| apply_slice_op(op, stream))
                })
                .map_err(map_io_error)?;
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

    /// Events whose index lies in `[i0, i1)` (clamped to the file).
    #[pyo3(signature = (i0, i1))]
    fn slice_count(&self, py: Python<'_>, i0: i64, i1: i64) -> PyResult<PyEventStream> {
        let i0 =
            usize::try_from(i0).map_err(|_| PyValueError::new_err("i0 must be non-negative"))?;
        let i1 =
            usize::try_from(i1).map_err(|_| PyValueError::new_err("i1 must be non-negative"))?;
        let reader = Arc::clone(&self.inner);
        let op = self.slice_op;
        py.detach(move || {
            reader
                .lock()
                .unwrap()
                .slice_index(i0, i1)
                .map(|stream| apply_slice_op(op, stream))
        })
        .map(|inner| PyEventStream {
            inner,
            repr: self.repr,
        })
        .map_err(map_io_error)
    }

    /// Lazy iterator of consecutive windows: each is `[start, start + span_ms)` and
    /// `start` advances by `step_ms`. `step_ms` defaults to the `dt_ms` set at `open`
    /// (so `windows()` walks every `slice(n)`), and `span_ms` defaults to `step_ms`
    /// (non-overlapping). Streams a multi-GB file window-by-window without loading it.
    #[pyo3(signature = (*, step_ms=None, span_ms=None))]
    fn windows(&self, step_ms: Option<f64>, span_ms: Option<f64>) -> PyResult<PyWindowIterator> {
        let step_us = match step_ms {
            Some(ms) if ms.is_nan() || ms <= 0.0 => {
                return Err(PyValueError::new_err("step_ms must be positive"))
            }
            Some(ms) => ms_to_us(ms).max(1),
            None => self.dt_us.ok_or_else(|| {
                PyValueError::new_err("windows() needs step_ms, or open the file with dt_ms")
            })?,
        };
        let span_us = match span_ms {
            Some(ms) if ms.is_nan() || ms <= 0.0 => {
                return Err(PyValueError::new_err("span_ms must be positive"))
            }
            Some(ms) => ms_to_us(ms).max(1),
            None => step_us,
        };
        let guard = self.inner.lock().unwrap();
        let (lo, hi) = guard.time_span();
        let empty = guard.n_events() == 0;
        drop(guard);
        Ok(PyWindowIterator {
            reader: Arc::clone(&self.inner),
            cursor: lo,
            step_us,
            span_us,
            end: if empty { i64::MIN } else { hi },
            slice_op: self.slice_op,
            repr: self.repr,
        })
    }
}

impl PyEventReader {
    /// Reads the half-open time window `[t0, t1)` (µs) off-GIL into an `EventStream`, applying the
    /// reader's per-slice op (e.g. `efast`) if one is set.
    fn fetch_time(&self, py: Python<'_>, t0: i64, t1: i64) -> PyResult<PyEventStream> {
        let reader = Arc::clone(&self.inner);
        let op = self.slice_op;
        py.detach(move || {
            reader
                .lock()
                .unwrap()
                .slice_time(t0, t1)
                .map(|stream| apply_slice_op(op, stream))
        })
        .map(|inner| PyEventStream {
            inner,
            repr: self.repr,
        })
        .map_err(map_io_error)
    }

    /// A new reader over the same file (and `dt`/`repr`) that applies `op` to every slice.
    fn with_slice_op(&self, op: SliceOp) -> PyEventReader {
        PyEventReader {
            inner: Arc::clone(&self.inner),
            dt_us: self.dt_us,
            repr: self.repr,
            slice_op: Some(op),
        }
    }

    fn require_dt(&self) -> PyResult<i64> {
        self.dt_us.ok_or_else(|| {
            PyValueError::new_err(
                "reader was opened without dt_ms; pass open(path, dt_ms=…) to use integer slice indices",
            )
        })
    }

    /// The `[t0, t1)` window (µs) for the `index`-th `dt` slice, validating the range
    /// (negative indices count back from the end).
    fn window_for_index(&self, index: i64) -> PyResult<(i64, i64)> {
        let dt = self.require_dt()?;
        let guard = self.inner.lock().unwrap();
        let total = slice_total(guard.n_events(), guard.time_span(), dt) as i64;
        let (lo, _hi) = guard.time_span();
        drop(guard);
        let resolved = if index < 0 { index + total } else { index };
        if resolved < 0 || resolved >= total {
            return Err(PyIndexError::new_err(format!(
                "slice index {index} out of range for {total} slices"
            )));
        }
        let t0 = lo + resolved * dt;
        Ok((t0, t0 + dt))
    }
}

/// Iterator returned by `EventReader.windows`; yields an `EventStream` per window.
#[pyclass]
struct PyWindowIterator {
    reader: Arc<Mutex<Reader>>,
    cursor: i64,
    step_us: i64,
    span_us: i64,
    end: i64,
    slice_op: Option<SliceOp>,
    /// The reader's stored representation, carried onto each yielded window (so `open(repr=…)`
    /// survives `for w in reader.windows(): w.view()`).
    repr: Option<ReprSpec>,
}

#[pymethods]
impl PyWindowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyEventStream>> {
        if self.cursor > self.end {
            return Ok(None);
        }
        let t0 = self.cursor;
        let t1 = t0.saturating_add(self.span_us);
        self.cursor = self.cursor.saturating_add(self.step_us);
        let reader = Arc::clone(&self.reader);
        let op = self.slice_op;
        let stream = py
            .detach(move || {
                reader
                    .lock()
                    .unwrap()
                    .slice_time(t0, t1)
                    .map(|stream| apply_slice_op(op, stream))
            })
            .map_err(map_io_error)?;
        Ok(Some(PyEventStream {
            inner: stream,
            repr: self.repr,
        }))
    }
}

/// Saves an `EventStream` or `EventFrame` to `path` (format by extension) — the OpenCV-style
/// free-function form of `obj.save(path)`. `topic` names the rosbag connection; `colormap` /
/// `normalize` apply only to `.png` frame export.
#[pyfunction]
#[pyo3(signature = (obj, path, *, topic=None, colormap="viridis", normalize=true))]
fn save(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    path: &str,
    topic: Option<String>,
    colormap: &str,
    normalize: bool,
) -> PyResult<()> {
    if let Ok(stream) = obj.extract::<PyRef<PyEventStream>>() {
        let options = SaveOptions {
            topic,
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
            ..SaveOptions::default()
        };
        let inner = &frame.inner;
        return py
            .detach(|| eventcv_core::io::save_frame(path, inner, &options))
            .map_err(map_io_error);
    }
    Err(PyTypeError::new_err(
        "save expects an EventStream or EventFrame",
    ))
}

/// Reads an `EventFrame` previously written by `frame.save(...)` / `eventcv.save(...)`
/// (`.npz` or `.h5`), restoring its dtype, `kind`, and `channel_names`.
#[pyfunction]
fn load_frame(py: Python<'_>, path: &str) -> PyResult<PyEventFrame> {
    py.detach(|| eventcv_core::io::load_frame(path))
        .map(|inner| PyEventFrame { inner })
        .map_err(map_io_error)
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
    Mcts { max_window_ms: f64 },
    Flow { window: usize },
}

impl ReprSpec {
    /// Builds a spec from a repr name and the optional overrides (unset ones take the
    /// same defaults as the `EventStream` methods).
    fn new(
        name: &str,
        bins: Option<i64>,
        window_ms: Option<f64>,
        tau_ms: Option<f64>,
        max_window_ms: Option<f64>,
        window: Option<i64>,
        normalize: bool,
    ) -> PyResult<Self> {
        Ok(match name {
            "binary" => Self::Binary,
            "count" => Self::Count { normalize },
            "polarity" => Self::Polarity { normalize },
            "voxel" => Self::Voxel {
                bins: bins
                    .map(|value| {
                        usize::try_from(value)
                            .map_err(|_| PyValueError::new_err("bins must be at least 1"))
                    })
                    .transpose()?
                    .unwrap_or(9),
                window_ms: window_ms.unwrap_or(30.0),
            },
            "tsurf" => Self::TimeSurface {
                tau_ms: tau_ms.unwrap_or(30.0),
            },
            "atsurf" => Self::AveragedTimeSurface {
                tau_ms: tau_ms.unwrap_or(30.0),
            },
            "tencode" => Self::Tencode {
                window_ms: window_ms.unwrap_or(30.0),
            },
            "mcts" => Self::Mcts {
                max_window_ms: max_window_ms.unwrap_or(30.0),
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

    fn name(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Count { .. } => "count",
            Self::Polarity { .. } => "polarity",
            Self::Voxel { .. } => "voxel",
            Self::TimeSurface { .. } => "tsurf",
            Self::AveragedTimeSurface { .. } => "atsurf",
            Self::Tencode { .. } => "tencode",
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

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventStream>()?;
    m.add_class::<PyEventFrame>()?;
    m.add_class::<PyEventPointSet>()?;
    m.add_class::<PyEventReader>()?;
    m.add_class::<PyWindowIterator>()?;
    m.add_class::<PyPolarity>()?;
    m.add_class::<PyCamera>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load_frame, m)?)?;
    #[cfg(feature = "hdf5")]
    m.add_class::<PyFrameSink>()?;
    Ok(())
}
