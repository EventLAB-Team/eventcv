//! ONNX inference — running a trained network over a representation.
//!
//! # Scope
//!
//! This is deliberately a *runner*, not a model zoo. EventCV builds the tensors; what consumes them
//! is the user's business, and bundling architectures or weights would mean tracking every upstream
//! model's preprocessing forever. So there is one type, [`Model`], it takes any `.onnx` file, and it
//! knows nothing about RVT, E2VID or YOLOX beyond their input shapes — which the graph already
//! declares and [`Model::inputs`] reports.
//!
//! The pairing that makes this worth having is with [`crate::representation`]: a voxel grid or time
//! surface is already a dense `f32` tensor in the layout ONNX wants, so feeding one to a network is
//! a shape check and a pointer, not a conversion.

use std::ffi::CStr;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use ndarray::{ArrayD, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::sys;
use ort::value::{Tensor, Value};

/// What the platform's dynamic loader calls the ONNX Runtime when `ORT_DYLIB_PATH` names nothing.
#[cfg(target_os = "windows")]
const DEFAULT_DYLIB: &str = "onnxruntime.dll";
#[cfg(target_vendor = "apple")]
const DEFAULT_DYLIB: &str = "libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_vendor = "apple")))]
const DEFAULT_DYLIB: &str = "libonnxruntime.so";

/// What went wrong loading or running a model.
#[derive(Debug)]
pub enum ModelError {
    /// The file could not be read, or is not a valid ONNX graph.
    Load(String),
    /// Inference failed — usually a shape or dtype the graph would not accept.
    Run(String),
    /// The graph produced an output this crate cannot represent as an `f32` array.
    UnsupportedOutput(String),
    /// No ONNX Runtime library could be opened. Separate from `Load` because nothing is wrong
    /// with the model or the call — the environment is missing a piece, and the fix is an install.
    RuntimeMissing(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(message) => write!(f, "could not load the ONNX model: {message}"),
            Self::Run(message) => write!(f, "inference failed: {message}"),
            Self::UnsupportedOutput(message) => {
                write!(f, "unsupported model output: {message}")
            }
            Self::RuntimeMissing(message) => write!(f, "ONNX Runtime is unavailable: {message}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// One input or output of a graph, as the graph itself declares it.
///
/// `shape` uses `-1` for a dimension the graph leaves free — the batch axis, usually — which is why
/// it is signed rather than a `usize`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}

/// A loaded ONNX graph, ready to run.
pub struct Model {
    session: Session,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
}

impl Model {
    /// Loads `path` and prepares it for inference.
    ///
    /// Optimisation is set to `Level3` at load time rather than left at the default: a model is
    /// loaded once and then run per slice, so paying the graph-rewrite cost up front is always the
    /// right trade here.
    pub fn load(path: &str) -> Result<Self, ModelError> {
        // Before the first ort call, because that is where a missing runtime would otherwise
        // abort the process rather than return.
        ensure_runtime()?;
        // Stepwise rather than a combinator chain: ort's error type is generic over the stage, so
        // the intermediate results don't share one `Result` type to chain through.
        let builder = Session::builder().map_err(|error| ModelError::Load(error.to_string()))?;
        let mut builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|error| ModelError::Load(error.to_string()))?;
        let session = builder
            .commit_from_file(path)
            .map_err(|error| ModelError::Load(error.to_string()))?;
        let inputs = session
            .inputs()
            .iter()
            .map(|input| Port {
                name: input.name().to_string(),
                shape: tensor_shape(input.dtype()),
                dtype: type_name(input.dtype()),
            })
            .collect();
        let outputs = session
            .outputs()
            .iter()
            .map(|output| Port {
                name: output.name().to_string(),
                shape: tensor_shape(output.dtype()),
                dtype: type_name(output.dtype()),
            })
            .collect();
        Ok(Self {
            session,
            inputs,
            outputs,
        })
    }

    /// What the graph expects to be given.
    pub fn inputs(&self) -> &[Port] {
        &self.inputs
    }

    /// What the graph produces.
    pub fn outputs(&self) -> &[Port] {
        &self.outputs
    }

    /// Runs the graph over a single `f32` input, returning every output as an `f32` array.
    ///
    /// Takes `&mut self` because ONNX Runtime's session internals are not thread-safe; the caller
    /// holds the lock. Outputs come back in the graph's declared order, so
    /// `run(...)[i]` corresponds to [`Model::outputs`]`()[i]`.
    pub fn run(&mut self, input: ArrayD<f32>) -> Result<Vec<ArrayD<f32>>, ModelError> {
        let name = self
            .inputs
            .first()
            .map(|port| port.name.clone())
            .ok_or_else(|| ModelError::Run("the graph declares no inputs".into()))?;
        Ok(self
            .run_named(vec![(name, input)])?
            .into_iter()
            .map(|(_, array)| array)
            .collect())
    }

    /// Runs the graph with every input bound by name, returning every output paired with its name.
    ///
    /// The multi-input form [`Model::run`] cannot express. Recurrent networks — E2VID and its
    /// relatives — take `(data, state_0, …)` and return `(image, new_state_0, …)`; feeding the new
    /// states back on the next call is what makes them recurrent, and that needs both ends bound by
    /// name rather than by position.
    ///
    /// Outputs come back in the graph's declared order, so they line up with [`Model::outputs`].
    pub fn run_named(
        &mut self,
        inputs: Vec<(String, ArrayD<f32>)>,
    ) -> Result<Vec<(String, ArrayD<f32>)>, ModelError> {
        if inputs.is_empty() {
            return Err(ModelError::Run("no inputs were supplied".into()));
        }
        let declared: Vec<&str> = self.inputs.iter().map(|port| port.name.as_str()).collect();
        for (name, _) in &inputs {
            if !declared.contains(&name.as_str()) {
                return Err(ModelError::Run(format!(
                    "the graph has no input named {name:?}; it declares {declared:?}"
                )));
            }
        }
        let mut values = Vec::with_capacity(inputs.len());
        for (name, array) in inputs {
            let shape: Vec<i64> = array.shape().iter().map(|&dim| dim as i64).collect();
            let (data, _) = array.into_raw_vec_and_offset();
            let tensor = Tensor::from_array((shape, data))
                .map_err(|error| ModelError::Run(error.to_string()))?;
            values.push((name, tensor));
        }
        let outputs = self
            .session
            .run(values)
            .map_err(|error| ModelError::Run(error.to_string()))?;
        self.outputs
            .iter()
            .map(|port| {
                let value = outputs.get(port.name.as_str()).ok_or_else(|| {
                    ModelError::UnsupportedOutput(format!(
                        "{} is missing from the result",
                        port.name
                    ))
                })?;
                Ok((port.name.clone(), extract_f32(value, &port.name)?))
            })
            .collect()
    }
}

/// Copies an output tensor out as an owned `f32` array.
///
/// Only `f32` is handled: it is what every event-vision network in practice emits, and silently
/// casting an `i64` class index or a `bool` mask to float would hide a real mismatch rather than
/// report it.
fn extract_f32(value: &Value, name: &str) -> Result<ArrayD<f32>, ModelError> {
    let (shape, data) = value.try_extract_tensor::<f32>().map_err(|error| {
        ModelError::UnsupportedOutput(format!(
            "{name} is not a float32 tensor ({error}); export the model with float outputs"
        ))
    })?;
    let dims: Vec<usize> = shape.iter().map(|&dim| dim as usize).collect();
    ArrayD::from_shape_vec(IxDyn(&dims), data.to_vec()).map_err(|error| {
        ModelError::UnsupportedOutput(format!("{name} has an inconsistent shape: {error}"))
    })
}

/// Opens the ONNX Runtime library and checks it is usable, once per process.
///
/// The runtime is loaded at run time rather than linked in (see the `ort` entry in this crate's
/// `Cargo.toml` for why), which moves three failures out of build time and into the user's
/// machine: no runtime at all, a file that is not ONNX Runtime, and an ONNX Runtime older than
/// the API level compiled in. `ort` meets all three the same way — a panic inside its lazy
/// initialiser, reaching Python as a `PanicException` with no advice in it — so they are
/// diagnosed here instead, before the first ort call. The library stays resident afterwards, so
/// ort's own load finds it already open.
///
/// Returns the runtime's version string, which `eventcv --version` reports.
fn probe_runtime() -> &'static Result<String, String> {
    static PROBE: OnceLock<Result<String, String>> = OnceLock::new();
    PROBE.get_or_init(|| {
        // The same resolution order ort itself uses, so the probe cannot succeed where the real
        // load would fail: an explicit `ORT_DYLIB_PATH` (set by `python/eventcv/_ort.py` from
        // whatever the installation provides), else the platform's default name, which the
        // loader looks for on the usual search path.
        let target = std::env::var_os("ORT_DYLIB_PATH")
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_DYLIB.into());
        let target = Path::new(&target);

        // SAFETY: loading a library runs its initialisers, which is exactly what ort would do a
        // moment later; there is no way to check one without opening it.
        let library = unsafe { libloading::Library::new(target) }
            .map_err(|error| format!("could not open {} ({error})", target.display()))?;
        let version = {
            // SAFETY: `OrtGetApiBase` is ONNX Runtime's documented entry point and has had this
            // signature since 1.0; a library without it is rejected below rather than called.
            let entry: libloading::Symbol<unsafe extern "system" fn() -> *const sys::OrtApiBase> =
                unsafe { library.get("OrtGetApiBase") }.map_err(|_| {
                    format!(
                        "{} is a library, but not ONNX Runtime — it exports no OrtGetApiBase",
                        target.display()
                    )
                })?;
            // SAFETY: the pointer comes from ONNX Runtime's own entry point and points at static
            // storage inside it; the library outlives this scope (it is leaked below).
            let base = unsafe { entry() };
            if base.is_null() {
                return Err(format!("{}: OrtGetApiBase returned null", target.display()));
            }
            let version = unsafe { CStr::from_ptr(((*base).GetVersionString)()) }
                .to_string_lossy()
                .into_owned();
            if unsafe { ((*base).GetApi)(sys::ORT_API_VERSION) }.is_null() {
                return Err(format!(
                    "{} is ONNX Runtime {version}, which is too old: EventCV needs API level {} \
                     (ONNX Runtime 1.{} or newer)",
                    target.display(),
                    sys::ORT_API_VERSION,
                    sys::ORT_API_VERSION
                ));
            }
            version
        };
        // Deliberately never unloaded: ort is about to open the same file, and dropping it here
        // would only unmap it in between.
        std::mem::forget(library);
        Ok(version)
    })
}

/// The ONNX Runtime version in use, or `None` if there is none to load. Never fails — it is for
/// `eventcv --version`, which has to print something useful either way.
pub fn runtime_version() -> Option<String> {
    probe_runtime().as_ref().ok().cloned()
}

/// [`probe_runtime`] as a precondition, with the message a user can act on.
fn ensure_runtime() -> Result<(), ModelError> {
    match probe_runtime() {
        Ok(_) => Ok(()),
        Err(reason) => Err(ModelError::RuntimeMissing(format!(
            "{reason}. EventCV loads ONNX Runtime at run time instead of compiling it in, and the \
             wheels ship a copy — reinstall from PyPI (`pip install --force-reinstall eventcv`), \
             install one alongside (`pip install eventcv[onnx]`, or conda's `onnxruntime-cpp`), or \
             point ORT_DYLIB_PATH at a libonnxruntime you already have"
        ))),
    }
}

/// The declared shape of a tensor port, with `-1` for free dimensions. Non-tensor ports (sequences,
/// maps) report an empty shape rather than failing the whole load — the graph may still be usable.
fn tensor_shape(value_type: &ort::value::ValueType) -> Vec<i64> {
    match value_type {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => Vec::new(),
    }
}

fn type_name(value_type: &ort::value::ValueType) -> String {
    match value_type {
        ort::value::ValueType::Tensor { ty, .. } => format!("{ty:?}").to_lowercase(),
        other => format!("{other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether this machine has an ONNX Runtime to open at all.
    ///
    /// The runtime is installed rather than linked in, so these tests would otherwise fail on a
    /// machine that simply hasn't got one — a false alarm about the library. They return early
    /// instead; the Python suite (`tests/test_model.py`) runs with the `onnxruntime` wheel
    /// installed and covers the same paths, plus the missing-runtime message itself.
    fn runtime_available() -> bool {
        !matches!(
            Model::load("/nonexistent/model.onnx"),
            Err(ModelError::RuntimeMissing(_))
        )
    }

    #[test]
    fn a_missing_file_is_a_load_error() {
        if !runtime_available() {
            return;
        }
        let error = Model::load("/nonexistent/model.onnx")
            .err()
            .expect("loading a missing file must fail");
        assert!(matches!(error, ModelError::Load(_)));
        assert!(error.to_string().contains("ONNX"));
    }

    #[test]
    fn a_non_onnx_file_is_a_load_error() {
        if !runtime_available() {
            return;
        }
        let mut path = std::env::temp_dir();
        path.push(format!("eventcv-not-a-model-{}.onnx", std::process::id()));
        std::fs::write(&path, b"this is not a protobuf").unwrap();
        let error = Model::load(path.to_str().unwrap())
            .err()
            .expect("loading a non-model must fail");
        assert!(matches!(error, ModelError::Load(_)));
        std::fs::remove_file(&path).ok();
    }
}
