use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

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

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u64>> {
        self.inner.as_array().to_owned().into_pyarray(py)
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

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEventStream>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
