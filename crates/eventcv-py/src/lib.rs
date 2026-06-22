mod viewer;

use eventcv_core::representation::{
    EventFrame, EventFrameData, Polarity, Representation, RepresentationError,
};
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{
    PyFileNotFoundError, PyOSError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

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
        self.inner.as_array().dim()
    }

    #[getter]
    fn columns(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        ("x", "y", "t", "p")
    }

    #[getter]
    fn sensor_size(&self) -> (usize, usize) {
        self.inner.sensor_size()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        self.inner.as_array().to_owned().into_pyarray(py)
    }

    #[pyo3(signature = (representation=None, *, normalize=true))]
    fn flatten(
        &self,
        py: Python<'_>,
        representation: Option<&Bound<'_, PyAny>>,
        normalize: bool,
    ) -> PyResult<PyEventFrame> {
        match representation {
            Some(representation) => self.generate(py, representation, normalize),
            None => self.generate_polarity(py, Polarity::new(normalize)),
        }
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
            let name = representation.get_type().name()?.to_str()?.to_owned();
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
        PyTuple::new(py, self.inner.channel_names().iter().copied())
    }

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self.inner.data() {
            EventFrameData::U8(data) => numpy::ndarray::Array3::from_shape_vec(
                self.inner.shape(),
                data.clone(),
            )
            .expect("EventFrame shape must match its data")
            .into_pyarray(py)
            .into_any(),
            EventFrameData::U16(data) => numpy::ndarray::Array3::from_shape_vec(
                self.inner.shape(),
                data.clone(),
            )
            .expect("EventFrame shape must match its data")
            .into_pyarray(py)
            .into_any(),
        }
    }

    #[pyo3(signature = (*, normalize=true))]
    fn view(&self, py: Python<'_>, normalize: bool) -> PyResult<()> {
        py.detach(|| viewer::view(&self.inner, normalize))
            .map_err(PyRuntimeError::new_err)
    }
}

#[pyfunction]
fn load(path: &str) -> PyResult<PyEventStream> {
    eventcv_core::load(path)
        .map(|inner| PyEventStream { inner })
        .map_err(map_load_error)
}

fn map_load_error(error: eventcv_core::LoadError) -> PyErr {
    match error {
        eventcv_core::LoadError::Io(error) if error.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(error.to_string())
        }
        eventcv_core::LoadError::Io(error) => PyOSError::new_err(error.to_string()),
        eventcv_core::LoadError::InvalidFormat(message) => PyValueError::new_err(message),
    }
}

fn map_representation_error(error: RepresentationError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventStream>()?;
    m.add_class::<PyEventFrame>()?;
    m.add_class::<PyPolarity>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
