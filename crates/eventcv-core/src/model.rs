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

use std::fmt;

use ndarray::{ArrayD, IxDyn};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{Tensor, Value};

/// What went wrong loading or running a model.
#[derive(Debug)]
pub enum ModelError {
    /// The file could not be read, or is not a valid ONNX graph.
    Load(String),
    /// Inference failed — usually a shape or dtype the graph would not accept.
    Run(String),
    /// The graph produced an output this crate cannot represent as an `f32` array.
    UnsupportedOutput(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(message) => write!(f, "could not load the ONNX model: {message}"),
            Self::Run(message) => write!(f, "inference failed: {message}"),
            Self::UnsupportedOutput(message) => {
                write!(f, "unsupported model output: {message}")
            }
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
        let shape: Vec<i64> = input.shape().iter().map(|&dim| dim as i64).collect();
        let (data, _) = input.into_raw_vec_and_offset();
        let tensor = Tensor::from_array((shape, data))
            .map_err(|error| ModelError::Run(error.to_string()))?;
        let name = self
            .inputs
            .first()
            .map(|port| port.name.clone())
            .ok_or_else(|| ModelError::Run("the graph declares no inputs".into()))?;
        let outputs = self
            .session
            .run(ort::inputs![name => tensor])
            .map_err(|error| ModelError::Run(error.to_string()))?;

        // Collect by the graph's declared output order rather than by iterating the map, so the
        // returned index always lines up with `outputs()`.
        self.outputs
            .iter()
            .map(|port| {
                let value = outputs.get(port.name.as_str()).ok_or_else(|| {
                    ModelError::UnsupportedOutput(format!(
                        "{} is missing from the result",
                        port.name
                    ))
                })?;
                extract_f32(value, &port.name)
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

    #[test]
    fn a_missing_file_is_a_load_error() {
        let error = Model::load("/nonexistent/model.onnx")
            .err()
            .expect("loading a missing file must fail");
        assert!(matches!(error, ModelError::Load(_)));
        assert!(error.to_string().contains("ONNX"));
    }

    #[test]
    fn a_non_onnx_file_is_a_load_error() {
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
