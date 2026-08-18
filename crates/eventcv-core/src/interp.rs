//! Learned frame interpolation, run *before* simulation.
//!
//! [`crate::simulate::Upsample`] already subdivides the interval between two frames, but the levels
//! it subdivides are a **linear** blend of them. That is exactly right when the intensity at a pixel
//! moves linearly over a frame gap, and wrong when it does not — an edge crossing a pixel makes it
//! step, not ramp, and a linear blend puts every event it should have produced at the wrong moment.
//! v2e reaches for Super-SloMo here; this reaches for whatever ONNX graph the caller has.
//!
//! It sits *before* the simulator rather than inside it, which is also where v2e puts it: the
//! interpolated frames are simply more source frames, with timestamps to match, and the simulator's
//! own [`Upsample`](crate::simulate::Upsample) then refines what is left. That means the pixel model
//! and its hot loop are untouched, and a run without an interpolator is byte-for-byte the run that
//! came before this module existed.
//!
//! No weights are bundled. eventcv runs graphs; it does not ship a zoo. Export RIFE (or anything
//! with the same shape) yourself and hand over the path.

use crate::representation::RepresentationError;

/// Produces frames between two others.
///
/// Frames are single-plane luma in `[0, 1]`, row-major — what [`crate::simulate::Simulator`]
/// consumes and what the video decoder already reduces its RGB to. A model trained on colour still
/// works: a grey image is a valid one, and the simulator would have thrown the colour away anyway.
pub trait FrameInterpolator: Send {
    /// The frames at each of `fractions` (each strictly between 0 and 1) along the path from `a`
    /// to `b`, in the order given.
    fn between(
        &mut self,
        a: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
        fractions: &[f32],
    ) -> Result<Vec<Vec<f32>>, InterpError>;
}

#[derive(Debug)]
pub enum InterpError {
    /// The graph's inputs are not a shape this can drive. Carries what it saw and what it wanted.
    Unsupported(String),
    /// The model itself failed.
    Model(String),
}

impl std::fmt::Display for InterpError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message) | Self::Model(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for InterpError {}

impl From<InterpError> for RepresentationError {
    fn from(error: InterpError) -> Self {
        Self::Device(error.to_string())
    }
}

/// The linear blend the simulator already does, as a [`FrameInterpolator`].
///
/// Not useful in production — the simulator interpolates linearly on its own, for free — but it is
/// what makes the *plumbing* testable without an ONNX export to hand, and it is the baseline any
/// learned interpolator has to beat.
pub struct LinearInterpolator;

impl FrameInterpolator for LinearInterpolator {
    fn between(
        &mut self,
        a: &[f32],
        b: &[f32],
        _width: usize,
        _height: usize,
        fractions: &[f32],
    ) -> Result<Vec<Vec<f32>>, InterpError> {
        Ok(fractions
            .iter()
            .map(|fraction| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| a + (b - a) * fraction)
                    .collect()
            })
            .collect())
    }
}

/// How an ONNX interpolator wants its frame pair.
///
/// Exports disagree, and the disagreement is entirely about packaging rather than about what the
/// network does, so it is detected from the declared inputs rather than configured.
#[cfg(feature = "onnx")]
#[derive(Clone, Debug, PartialEq, Eq)]
enum Layout {
    /// One input of six channels: the two frames stacked.
    Stacked { image: String, timestep: Option<String> },
    /// Two image inputs.
    Separate {
        first: String,
        second: String,
        timestep: Option<String>,
    },
}

/// Drives an ONNX frame interpolator — RIFE and anything shaped like it.
///
/// # Which exports work
///
/// The graph must take a frame pair, either stacked into one six-channel input or as two
/// three-channel ones, and return one image. A scalar `timestep` input (RIFE v4 and later) is used
/// when present, which is what allows an arbitrary fraction in one call. Without it the network can
/// only produce the midpoint, so a fraction of `k/2^n` is reached by bisecting `n` times and
/// anything else is refused rather than approximated.
#[cfg(feature = "onnx")]
pub struct OnnxInterpolator {
    model: crate::model::Model,
    layout: Layout,
}

#[cfg(feature = "onnx")]
impl OnnxInterpolator {
    /// Loads `path` and works out how to feed it.
    pub fn load(path: &str) -> Result<Self, InterpError> {
        let model = crate::model::Model::load(path).map_err(|error| InterpError::Model(error.to_string()))?;
        let layout = Self::detect(&model)?;
        Ok(Self { model, layout })
    }

    /// Reads the graph's declared inputs and decides how the pair is passed.
    fn detect(model: &crate::model::Model) -> Result<Layout, InterpError> {
        // A timestep is the input that is not an image: rank below three, or a single element.
        let images: Vec<&crate::model::Port> = model
            .inputs()
            .iter()
            .filter(|port| port.shape.len() >= 3)
            .collect();
        let timestep = model
            .inputs()
            .iter()
            .find(|port| port.shape.len() < 3 || port.shape.iter().all(|dim| *dim == 1))
            .map(|port| port.name.clone());
        let channels = |port: &crate::model::Port| port.shape.get(port.shape.len() - 3).copied();

        match images.as_slice() {
            [image] if matches!(channels(image), Some(6) | Some(-1)) => Ok(Layout::Stacked {
                image: image.name.clone(),
                timestep,
            }),
            [first, second] => Ok(Layout::Separate {
                first: first.name.clone(),
                second: second.name.clone(),
                timestep,
            }),
            other => Err(InterpError::Unsupported(format!(
                "a frame interpolator should take a frame pair — one six-channel input or two \
                 three-channel ones, optionally with a timestep — but this graph declares {} \
                 image-shaped inputs: {:?}",
                other.len(),
                model
                    .inputs()
                    .iter()
                    .map(|port| (port.name.as_str(), &port.shape))
                    .collect::<Vec<_>>()
            ))),
        }
    }

    fn timestep(&self) -> Option<&str> {
        match &self.layout {
            Layout::Stacked { timestep, .. } | Layout::Separate { timestep, .. } => {
                timestep.as_deref()
            }
        }
    }

    /// One forward pass: the frame `fraction` of the way from `a` to `b`.
    fn once(
        &mut self,
        a: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
        fraction: f32,
    ) -> Result<Vec<f32>, InterpError> {
        use ndarray::{Array, IxDyn};

        // Luma replicated across RGB: these networks are trained on colour, and a grey image is
        // simply one where the channels agree.
        let rgb = |plane: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(plane.len() * 3);
            for _ in 0..3 {
                out.extend_from_slice(plane);
            }
            out
        };
        let shaped = |data: Vec<f32>, channels: usize| {
            Array::from_shape_vec(IxDyn(&[1, channels, height, width]), data)
                .map_err(|error| InterpError::Model(error.to_string()))
        };

        let mut inputs = match self.layout.clone() {
            Layout::Stacked { image, .. } => {
                let mut both = rgb(a);
                both.extend(rgb(b));
                vec![(image, shaped(both, 6)?)]
            }
            Layout::Separate { first, second, .. } => vec![
                (first, shaped(rgb(a), 3)?),
                (second, shaped(rgb(b), 3)?),
            ],
        };
        if let Some(name) = self.timestep().map(str::to_owned) {
            inputs.push((
                name,
                Array::from_shape_vec(IxDyn(&[1]), vec![fraction])
                    .map_err(|error| InterpError::Model(error.to_string()))?,
            ));
        }

        let outputs = self
            .model
            .run_named(inputs)
            .map_err(|error| InterpError::Model(error.to_string()))?;
        let (_, image) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InterpError::Model("the graph returned nothing".into()))?;
        Ok(to_luma(image.as_slice().unwrap_or(&[]), width * height))
    }

    /// The frame at `fraction` when the graph has no timestep input: bisect towards it, which only
    /// reaches fractions of the form `k / 2^n`.
    fn bisect(
        &mut self,
        a: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
        fraction: f32,
        depth: usize,
    ) -> Result<Vec<f32>, InterpError> {
        const MAX_DEPTH: usize = 4;
        if (fraction - 0.5).abs() < 1e-6 {
            return self.once(a, b, width, height, 0.5);
        }
        if depth >= MAX_DEPTH {
            return Err(InterpError::Unsupported(format!(
                "this export has no timestep input, so it can only produce midpoints; a fraction \
                 of {fraction} would need more than {MAX_DEPTH} bisections. Use an interpolation \
                 factor that is a power of two, or export a model that takes a timestep."
            )));
        }
        let middle = self.once(a, b, width, height, 0.5)?;
        if fraction < 0.5 {
            self.bisect(a, &middle, width, height, fraction * 2.0, depth + 1)
        } else {
            self.bisect(&middle, b, width, height, (fraction - 0.5) * 2.0, depth + 1)
        }
    }
}

/// Collapses a model's `[1, C, H, W]` output back to one luma plane.
#[cfg(feature = "onnx")]
fn to_luma(data: &[f32], plane: usize) -> Vec<f32> {
    if plane == 0 {
        return Vec::new();
    }
    let channels = (data.len() / plane).max(1);
    (0..plane)
        .map(|index| {
            let sum: f32 = (0..channels.min(3))
                .map(|channel| data.get(channel * plane + index).copied().unwrap_or(0.0))
                .sum();
            (sum / channels.min(3) as f32).clamp(0.0, 1.0)
        })
        .collect()
}

#[cfg(feature = "onnx")]
impl FrameInterpolator for OnnxInterpolator {
    fn between(
        &mut self,
        a: &[f32],
        b: &[f32],
        width: usize,
        height: usize,
        fractions: &[f32],
    ) -> Result<Vec<Vec<f32>>, InterpError> {
        let timed = self.timestep().is_some();
        fractions
            .iter()
            .map(|fraction| {
                if timed {
                    self.once(a, b, width, height, *fraction)
                } else {
                    self.bisect(a, b, width, height, *fraction, 0)
                }
            })
            .collect()
    }
}

/// An interpolator and how many sub-frames it should produce.
///
/// `factor` is the number of intervals each source pair becomes, so `4` inserts three frames.
pub struct Interpolation<'a> {
    pub interpolator: &'a mut dyn FrameInterpolator,
    pub factor: usize,
}

impl Interpolation<'_> {
    /// The fractions between two source frames, `factor - 1` of them.
    pub fn fractions(&self) -> Vec<f32> {
        (1..self.factor.max(1))
            .map(|step| step as f32 / self.factor as f32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameInterpolator, Interpolation, LinearInterpolator};

    #[test]
    fn fractions_split_the_interval_evenly_and_exclude_the_endpoints() {
        let mut linear = LinearInterpolator;
        let plan = Interpolation {
            interpolator: &mut linear,
            factor: 4,
        };
        assert_eq!(plan.fractions(), vec![0.25, 0.5, 0.75]);
    }

    #[test]
    fn a_factor_of_one_inserts_nothing() {
        let mut linear = LinearInterpolator;
        let plan = Interpolation {
            interpolator: &mut linear,
            factor: 1,
        };
        assert!(plan.fractions().is_empty());
    }

    #[test]
    fn the_linear_baseline_blends_the_way_the_simulator_would() {
        let mut linear = LinearInterpolator;
        let frames = linear
            .between(&[0.0, 1.0], &[1.0, 0.0], 2, 1, &[0.25, 0.75])
            .unwrap();
        assert_eq!(frames, vec![vec![0.25, 0.75], vec![0.75, 0.25]]);
    }
}
