mod viewer;

use std::sync::{Arc, Mutex};

use eventcv_core::{
    image::{PoolingMethod, ResizeError},
    io::{open as open_reader, ColumnOrder, IoError, LoadOptions, Reader, TimeUnit},
    representation::{
        Binary, EventFrame, EventFrameData, EventPointSet, Mcts, PointSet, Polarity,
        Representation, RepresentationError, Tencode, TimeSurface, VoxelGrid,
    },
};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{
    PyFileNotFoundError, PyIndexError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

#[pyclass(name = "EventStream", frozen)]
struct PyEventStream {
    inner: eventcv_core::EventStream,
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

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        self.inner.to_array2().into_pyarray(py)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn resize(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let _ = (args, kwargs);
        Err(PyTypeError::new_err(
            "Resize cannot be performed directly on a stream; it must be performed on a representation first. The correct syntax is data.flatten().resize(width, height).",
        ))
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
            None => self.generate_polarity(py, Polarity::new(normalize)),
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

    #[pyo3(signature = (representation=None, *, normalize=true))]
    fn view(
        &self,
        py: Python<'_>,
        representation: Option<&Bound<'_, PyAny>>,
        normalize: bool,
    ) -> PyResult<()> {
        let frame = match representation {
            Some(representation) => self.generate(py, representation, normalize)?,
            None => self.generate_polarity(py, Polarity::new(normalize))?,
        };
        frame.view(py, normalize)
    }
}

impl PyEventStream {
    fn generate(
        &self,
        py: Python<'_>,
        representation: &Bound<'_, PyAny>,
        normalize: bool,
    ) -> PyResult<PyEventFrame> {
        if representation.extract::<PyRef<'_, PyPolarity>>().is_ok() {
            self.generate_polarity(py, Polarity::new(normalize))
        } else {
            let name = representation.get_type().name()?.extract::<String>()?;
            Err(PyTypeError::new_err(format!(
                "unsupported representation: {name}"
            )))
        }
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
        match self.inner.data() {
            EventFrameData::U8(data) => {
                numpy::ndarray::Array3::from_shape_vec(self.inner.shape(), data.clone())
                    .expect("EventFrame shape must match its data")
                    .into_pyarray(py)
                    .into_any()
            }
            EventFrameData::U16(data) => {
                numpy::ndarray::Array3::from_shape_vec(self.inner.shape(), data.clone())
                    .expect("EventFrame shape must match its data")
                    .into_pyarray(py)
                    .into_any()
            }
            EventFrameData::U64(data) => {
                numpy::ndarray::Array3::from_shape_vec(self.inner.shape(), data.clone())
                    .expect("EventFrame shape must match its data")
                    .into_pyarray(py)
                    .into_any()
            }
            EventFrameData::F32(data) => {
                numpy::ndarray::Array3::from_shape_vec(self.inner.shape(), data.clone())
                    .expect("EventFrame shape must match its data")
                    .into_pyarray(py)
                    .into_any()
            }
        }
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

    #[pyo3(signature = (*, normalize=true))]
    fn view(&self, py: Python<'_>, normalize: bool) -> PyResult<()> {
        py.detach(|| viewer::view(&self.inner, normalize))
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

fn parse_time_unit(time_unit: &str) -> PyResult<TimeUnit> {
    Ok(match time_unit {
        "seconds" | "s" => TimeUnit::Seconds,
        "milliseconds" | "ms" => TimeUnit::Milliseconds,
        "microseconds" | "us" => TimeUnit::Microseconds,
        "nanoseconds" | "ns" => TimeUnit::Nanoseconds,
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported time_unit: {other}"
            )))
        }
    })
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
#[pyo3(signature = (path, *, sensor_size=None, time_unit="seconds", order="txyp", topic=None, max_events=None))]
fn load(
    py: Python<'_>,
    path: &str,
    sensor_size: Option<(i64, i64)>,
    time_unit: &str,
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
        .map(|inner| PyEventStream { inner })
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
#[pyo3(signature = (path, *, dt_ms=None, sensor_size=None, time_unit="seconds", order="txyp", topic=None))]
fn open(
    py: Python<'_>,
    path: &str,
    dt_ms: Option<f64>,
    sensor_size: Option<(i64, i64)>,
    time_unit: &str,
    order: &str,
    topic: Option<String>,
) -> PyResult<PyEventReader> {
    let dt_us = parse_dt_ms(dt_ms)?;
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
}

#[pymethods]
impl PyEventReader {
    fn __len__(&self) -> usize {
        self.inner.lock().unwrap().n_events()
    }

    #[getter]
    fn n_events(&self) -> usize {
        self.inner.lock().unwrap().n_events()
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

    /// `reader[n]` — alias for `slice(n)` (requires `open(dt_ms=…)`).
    fn __getitem__(&self, py: Python<'_>, index: i64) -> PyResult<PyEventStream> {
        let (t0, t1) = self.window_for_index(index)?;
        self.fetch_time(py, t0, t1)
    }

    /// Events whose index lies in `[i0, i1)` (clamped to the file).
    #[pyo3(signature = (i0, i1))]
    fn slice_count(&self, py: Python<'_>, i0: i64, i1: i64) -> PyResult<PyEventStream> {
        let i0 =
            usize::try_from(i0).map_err(|_| PyValueError::new_err("i0 must be non-negative"))?;
        let i1 =
            usize::try_from(i1).map_err(|_| PyValueError::new_err("i1 must be non-negative"))?;
        let reader = Arc::clone(&self.inner);
        py.detach(move || reader.lock().unwrap().slice_index(i0, i1))
            .map(|inner| PyEventStream { inner })
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
        })
    }
}

impl PyEventReader {
    /// Reads the half-open time window `[t0, t1)` (µs) off-GIL into an `EventStream`.
    fn fetch_time(&self, py: Python<'_>, t0: i64, t1: i64) -> PyResult<PyEventStream> {
        let reader = Arc::clone(&self.inner);
        py.detach(move || reader.lock().unwrap().slice_time(t0, t1))
            .map(|inner| PyEventStream { inner })
            .map_err(map_io_error)
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
        let stream = py
            .detach(move || reader.lock().unwrap().slice_time(t0, t1))
            .map_err(map_io_error)?;
        Ok(Some(PyEventStream { inner: stream }))
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
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    Ok(())
}
