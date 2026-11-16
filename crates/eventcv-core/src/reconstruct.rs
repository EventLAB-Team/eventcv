//! Neural reconstruction — recovering an intensity video from events with a trained network.
//!
//! [`E2vid`] is E2VID (Rebecq et al., *High Speed and High Dynamic Range Video with an Event
//! Camera*, T-PAMI 2019): a recurrent UNet whose ConvLSTM states carry the scene from one window
//! to the next. It runs on burn rather than through the ONNX runtime (`crate::model`), so it runs on
//! the shared wgpu device as well as the CPU. No weights are compiled in: [`E2vid::load`] reads
//! the `.safetensors` file `scripts/convert_e2vid.py` writes, which the Python package fetches on
//! first use.
//!
//! Everything around the network follows the reference implementation's defaults: a voxel grid
//! spanning exactly the window, normalised over its nonzero cells, reflection-padded to a multiple
//! of `2^encoders`, and an unsharp mask on the output. On the CPU a frame matches rpg_e2vid's for
//! the same events to ~1e-6. The GPU convolves in half precision on the tensor cores, ~1e-3
//! from it, which a long sequence can carry further in a few sensitive pixels.

mod winograd;

use std::fmt;
use std::path::Path;

use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::module::{conv2d, conv_transpose2d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, IndexingUpdateOp, PadMode};
use burn::tensor::{DType, Device, Int, Slice, Tensor, TensorData};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use crate::accel;
use crate::representation::RepresentationError;
use crate::EventStream;

/// An output buffer written by many tasks at once, each to elements no other task touches — the
/// Winograd convolution's output tiles.
#[derive(Clone, Copy)]
struct Output(*mut f32);

impl Output {
    fn write(&self, index: usize, value: f32) {
        // SAFETY: callers pass an index inside the buffer and partition the elements among their
        // tasks, so no two tasks write the same one; the buffer outlives the parallel loop.
        unsafe { *self.0.add(index) = value };
    }
}

// SAFETY: see `write`.
unsafe impl Send for Output {}
unsafe impl Sync for Output {}

/// State cells per thread in [`convlstm_host`].
const LSTM_CHUNK: usize = 1 << 14;

/// Events per thread when binning a window: enough to amortise the hand-off, few enough to spread
/// a window over the cores.
const BIN_CHUNK: usize = 1 << 15;

/// What the weights file's metadata must say for [`E2vid::load`] to accept it.
const FORMAT: &str = "eventcv-e2vid";
const VERSION: &str = "1";

#[derive(Debug)]
pub enum ReconstructError {
    /// The weights file is missing, malformed, or describes a network this build cannot run.
    Load(String),
    /// The window could not be turned into a voxel grid.
    Representation(RepresentationError),
    /// `device="gpu"` was asked for and cannot be had here.
    Unavailable(String),
    /// A window from a different sensor size arrived while recurrent state was held.
    SizeChanged {
        expected: (usize, usize),
        got: (usize, usize),
    },
}

impl fmt::Display for ReconstructError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(message) => write!(f, "could not load the reconstruction model: {message}"),
            Self::Representation(error) => write!(f, "{error}"),
            Self::Unavailable(message) => write!(f, "{message}"),
            Self::SizeChanged { expected, got } => write!(
                f,
                "the sensor size changed from {}x{} to {}x{} mid-sequence; call reset() between \
                 recordings",
                expected.0, expected.1, got.0, got.1
            ),
        }
    }
}

impl std::error::Error for ReconstructError {}

/// Arithmetic precision on the GPU. The CPU always computes in single precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Convolutions in half precision on the tensor cores: ~3x faster, ~1e-3 from the reference
    /// per frame, which a long sequence can carry further in a few sensitive pixels.
    #[default]
    Half,
    /// Single precision throughout, matching the reference (and the CPU) to ~1e-6.
    Single,
}

impl Precision {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "half" => Some(Self::Half),
            "single" => Some(Self::Single),
            _ => None,
        }
    }
}

/// How a convolution is run, which follows from the device and the precision.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kernel {
    /// burn's own convolution in f32: the CPU.
    Native,
    /// burn's convolution in f16, on the GPU's tensor cores.
    Half,
    /// im2col and an f32 matmul: the GPU in single precision, where burn's f32 convolution has
    /// only its direct kernel (610 against 33 ms a 346x260 frame on an RTX 2080).
    Im2col,
}

impl Kernel {
    fn new(device: accel::Device, precision: Precision) -> Self {
        match (device, precision) {
            (accel::Device::Cpu, _) => Self::Native,
            (accel::Device::Gpu, Precision::Half) => Self::Half,
            (accel::Device::Gpu, Precision::Single) => Self::Im2col,
        }
    }

    fn dtype(self) -> DType {
        if self == Self::Half {
            DType::F16
        } else {
            DType::F32
        }
    }
}

/// A convolution as trained, stride 1 or 2 with `kernel / 2` padding.
///
/// The weight is held in the dtype the kernel computes in and the bias in f32, added after rather
/// than passed to the convolution: given a bias, cubecl's tensor-core implicit-GEMM kernels decline
/// and its direct kernel runs instead — 42 against 1.1 ms for one ConvLSTM gate convolution at
/// 346x260 on an RTX 2080.
struct Conv {
    weight: Tensor<4>,
    bias: Tensor<4>,
    stride: usize,
    kernel: Kernel,
    /// On the CPU, a 3x3 stride-1 convolution — the ConvLSTM gates and residual blocks — goes
    /// through Winograd on the host instead of flex's im2col: 4x fewer multiplies, 2-2.5x faster.
    winograd: Option<winograd::Winograd>,
}

impl Conv {
    fn new(weight: Tensor<4>, bias: Tensor<1>, stride: usize, kernel: Kernel) -> Self {
        let channels = bias.dims()[0];
        let [out_channels, in_channels, k, _] = weight.dims();
        let winograd = (kernel == Kernel::Native && k == 3 && stride == 1).then(|| {
            let values = |t: Tensor<1>| t.into_data().try_to_vec::<f32>().expect("loaded as f32");
            let weight = values(weight.clone().reshape([out_channels * in_channels * 9]));
            winograd::Winograd::new(&weight, &values(bias.clone()), out_channels, in_channels)
        });
        Self {
            weight: weight.cast(kernel.dtype()),
            bias: bias.reshape([1, channels, 1, 1]),
            stride,
            kernel,
            winograd,
        }
    }

    /// `x` in the compute dtype, so callers can cast before concatenating and move half the bytes.
    fn input(&self, x: Tensor<4>) -> Tensor<4> {
        x.cast(self.weight.dtype())
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [out_channels, in_channels, k, _] = self.weight.dims();
        if let Some(winograd) = &self.winograd {
            // E2VID runs one window at a time, so the batch is always 1. Biased by the transform.
            let [_, _, height, width] = x.dims();
            let device = x.device();
            let x = x
                .into_data()
                .try_into_vec::<f32>()
                .expect("activations are f32");
            let y = winograd.forward(&x, height, width);
            return Tensor::from_data(
                TensorData::new(y, [1, out_channels, height, width]),
                &device,
            );
        }
        let y = if k == 1 && out_channels == 1 {
            // The prediction layer, 32 channels -> 1: a weighted sum over channels, which as a
            // convolution or a matmul spent 2-6 ms per 1280x720 frame on tiles a single output
            // channel cannot fill, and as one multiply-and-reduce is a memory pass (~1 ms).
            let weight = self.weight.clone().cast(DType::F32);
            (x * weight.reshape([1, in_channels, 1, 1])).sum_dim(1)
        } else if self.kernel == Kernel::Im2col {
            self.im2col(x)
        } else {
            let options = ConvOptions::new([self.stride; 2], [k / 2; 2], [1, 1], 1);
            conv2d(self.input(x), self.weight.clone(), None, options).cast(DType::F32)
        };
        y + self.bias.clone()
    }

    /// One matmul over an im2col matrix built from shifted (for stride 2, strided) views of the
    /// padded input.
    fn im2col(&self, x: Tensor<4>) -> Tensor<4> {
        let [out_channels, in_channels, k, _] = self.weight.dims();
        let [batch, _, height, width] = x.dims();
        let (pad, stride) = (k / 2, self.stride);
        let (out_h, out_w) = ((height - 1) / stride + 1, (width - 1) / stride + 1);
        let x = x.pad((pad, pad, pad, pad), PadMode::Constant(0.0));
        let view = |offset: usize, len: usize| {
            let end = (offset + stride * (len - 1) + 1) as isize;
            Slice::new(offset as isize, Some(end), stride as isize)
        };
        let taps = (0..k * k)
            .map(|tap| {
                let (dy, dx) = (tap / k, tap % k);
                x.clone().slice([
                    Slice::full(),
                    Slice::full(),
                    view(dy, out_h),
                    view(dx, out_w),
                ])
            })
            .collect();
        // [batch, in, k*k, out_h, out_w] flattens to the same (in, ky, kx) order as the weight.
        let columns =
            Tensor::stack::<5>(taps, 2).reshape([batch, in_channels * k * k, out_h * out_w]);
        let weight = self
            .weight
            .clone()
            .reshape([1, out_channels, in_channels * k * k]);
        weight
            .matmul(columns)
            .reshape([batch, out_channels, out_h, out_w])
    }
}

/// A decoder's stride-2 transposed convolution (5x5, padding 2, `output_padding` 1), which
/// doubles the size exactly, run as four stride-1 convolutions — one per output pixel parity —
/// interleaved.
///
/// Each output pixel of a stride-2 transposed convolution only meets the kernel taps of its own
/// parity: even rows take taps 4, 2, 0 over input rows `m - 1 ..= m + 1`, odd rows taps 3, 1 over
/// `m ..= m + 1`, and the same for columns. So the four phases are ordinary convolutions with
/// 3x3, 3x2, 2x3 and 2x2 slices of the flipped kernel: the same arithmetic, none of it spent on
/// the zeros a transposed convolution inserts, and all of it on the fast convolution kernels.
/// That is the GPU's path; on the CPU, flex's own transposed convolution is faster than the four
/// phases and the interleave between them.
// A burn tensor handle is ~260 bytes, so the variants differ in size; there are three of these.
#[allow(clippy::large_enum_variant)]
enum Upsample {
    Phases {
        /// `[row parity][column parity]` kernels, in the compute dtype.
        phases: [[Tensor<4>; 2]; 2],
        bias: Tensor<4>,
    },
    Transposed {
        weight: Tensor<4>,
        bias: Tensor<1>,
    },
}

/// Taps of the flipped 5x5 kernel each output parity uses, and the `(before, after)` padding that
/// lines them up with input offsets `-1..=1` (even) or `0..=1` (odd).
const PHASE_TAPS: [(&[i32], (usize, usize)); 2] = [(&[4, 2, 0], (1, 1)), (&[3, 1], (0, 1))];

impl Upsample {
    /// From a transposed-convolution weight `[in, out, 5, 5]`.
    fn new(weight: Tensor<4>, bias: Tensor<1>, kernel: Kernel, device: &Device) -> Self {
        // In f32, burn's transposed convolution is the faster of the two on either device.
        match kernel {
            Kernel::Half => Self::phased(weight, bias, DType::F16, device),
            _ => Self::Transposed { weight, bias },
        }
    }

    /// The four phase kernels, in `dtype`.
    fn phased(weight: Tensor<4>, bias: Tensor<1>, dtype: DType, device: &Device) -> Self {
        let weight = weight.swap_dims(0, 1);
        let taps = |taps: &[i32]| {
            Tensor::<1, Int>::from_data(TensorData::new(taps.to_vec(), [taps.len()]), device)
        };
        let phase = |row: usize, col: usize| {
            weight
                .clone()
                .select(2, taps(PHASE_TAPS[row].0))
                .select(3, taps(PHASE_TAPS[col].0))
                .cast(dtype)
        };
        let channels = bias.dims()[0];
        Self::Phases {
            phases: [[phase(0, 0), phase(0, 1)], [phase(1, 0), phase(1, 1)]],
            bias: bias.reshape([1, channels, 1, 1]),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let (phases, bias) = match self {
            Self::Phases { phases, bias } => (phases, bias),
            Self::Transposed { weight, bias } => {
                let options = ConvTransposeOptions::new([2, 2], [2, 2], [1, 1], [1, 1], 1);
                return conv_transpose2d(x, weight.clone(), Some(bias.clone()), options);
            }
        };
        let [batch, _, height, width] = x.dims();
        let channels = bias.dims()[1];
        let x = x.cast(phases[0][0].dtype());
        let rows = (0..2)
            .map(|row| {
                let phases = (0..2)
                    .map(|col| {
                        let ((top, bottom), (left, right)) = (PHASE_TAPS[row].1, PHASE_TAPS[col].1);
                        let options = ConvOptions::new_with_padding(
                            [1, 1],
                            [(top, bottom), (left, right)],
                            [1, 1],
                            1,
                        );
                        conv2d(x.clone(), phases[row][col].clone(), None, options)
                    })
                    .collect();
                // [batch, channels, height, width, column parity] -> columns interleaved.
                Tensor::stack::<5>(phases, 4).reshape([batch, channels, height, 2 * width])
            })
            .collect();
        // Interleaved in the compute dtype, then cast and biased once rather than per phase.
        Tensor::stack::<5>(rows, 3)
            .reshape([batch, channels, 2 * height, 2 * width])
            .cast(DType::F32)
            + bias.clone()
    }
}

/// One ConvLSTM step on the CPU, on the host: concatenation, the gate convolution and the gate
/// arithmetic in one pass, returning the new hidden and cell states.
///
/// As burn ops on flex this was a concatenation, two copies across the host boundary and five
/// activation passes — flex's `sigmoid` and `tanh` are scalar libm calls (~15 ns each), and the
/// update cost more than the gate convolution itself. Here the concatenation is two appends, the
/// convolution is the Winograd one, and the activations go through [`exp_fast`].
fn convlstm_host(
    gates: &Conv,
    x: Tensor<4>,
    hidden: Tensor<4>,
    cell: Tensor<4>,
) -> (Tensor<4>, Tensor<4>) {
    let winograd = gates
        .winograd
        .as_ref()
        .expect("the CPU's gate convolution is 3x3, stride 1");
    let (device, dims) = (cell.device(), cell.dims());
    let host = |t: Tensor<4>| t.into_data().try_into_vec::<f32>().expect("states are f32");
    let mut input = host(x);
    input.extend(host(hidden));
    let gates = winograd.forward(&input, dims[2], dims[3]);
    let previous = host(cell);
    let n = previous.len();
    let sigmoid = |x: f32| 1.0 / (1.0 + exp_fast(-x));
    let tanh = |x: f32| 1.0 - 2.0 / (exp_fast(2.0 * x) + 1.0);
    let (mut hidden, mut cell) = (vec![0.0_f32; n], vec![0.0_f32; n]);
    hidden
        .par_chunks_mut(LSTM_CHUNK)
        .zip(cell.par_chunks_mut(LSTM_CHUNK))
        .enumerate()
        .for_each(|(chunk, (hidden, cell))| {
            let (at, len) = (chunk * LSTM_CHUNK, hidden.len());
            // Input, remember, output and candidate gates, as the convolution laid them out.
            let (input, remember) = (&gates[at..][..len], &gates[n + at..][..len]);
            let (output, candidate) = (&gates[2 * n + at..][..len], &gates[3 * n + at..][..len]);
            let previous = &previous[at..][..len];
            for j in 0..len {
                let value =
                    sigmoid(remember[j]) * previous[j] + sigmoid(input[j]) * tanh(candidate[j]);
                cell[j] = value;
                hidden[j] = sigmoid(output[j]) * tanh(value);
            }
        });
    let tensor = |values| Tensor::<4>::from_data(TensorData::new(values, dims), &device);
    (tensor(hidden), tensor(cell))
}

/// `e^x` to ~1 ulp — Cephes' `expf` reduction and polynomial, written branch-free so the gate loop
/// vectorises (libm's `expf`/`tanhf` are scalar calls). Rounding to the nearest integer adds and
/// subtracts 1.5 * 2^23, which needs nothing beyond SSE2, the wheels' baseline.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    const ROUND: f32 = 12_582_912.0;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E + ROUND) - ROUND;
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = ((((1.987_569_2e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 1.666_666_5e-1)
        * r
        + 0.5;
    f32::from_bits(((n as i32 + 127) << 23) as u32) * (p * r * r + r + 1.0)
}

/// Each encoder's ConvLSTM `(hidden, cell)`, and the sensor size they were built for.
type State = ((usize, usize), Vec<(Tensor<4>, Tensor<4>)>);

/// The recurrent E2VID network with its weights loaded, and the state it carries between windows.
pub struct E2vid {
    bins: usize,
    events_per_pixel: f64,
    head: Conv,
    /// Per encoder: the strided convolution, then the ConvLSTM's gate convolution.
    encoders: Vec<(Conv, Conv)>,
    resblocks: Vec<(Conv, Conv)>,
    decoders: Vec<Upsample>,
    pred: Conv,
    unsharp: Tensor<4>,
    unsharp_amount: f32,
    device: Device,
    /// On the CPU the ConvLSTM step runs on the host (see [`convlstm_host`]).
    cpu: bool,
    state: Option<State>,
}

impl E2vid {
    /// Loads the weights `scripts/convert_e2vid.py` wrote, onto `device`; `precision` applies to
    /// the GPU.
    pub fn load(
        path: impl AsRef<Path>,
        device: accel::Device,
        precision: Precision,
    ) -> Result<Self, ReconstructError> {
        let path = path.as_ref();
        let load =
            |message: String| ReconstructError::Load(format!("{}: {message}", path.display()));
        let bytes = std::fs::read(path).map_err(|error| load(error.to_string()))?;
        let file = SafeTensors::deserialize(&bytes).map_err(|error| load(error.to_string()))?;
        let (_, metadata) =
            SafeTensors::read_metadata(&bytes).map_err(|error| load(error.to_string()))?;
        let meta = metadata.metadata().clone().unwrap_or_default();
        let field = |key: &str| {
            meta.get(key)
                .ok_or_else(|| load(format!("its metadata has no {key:?}")))
        };
        if field("format")? != FORMAT || field("version")? != VERSION {
            return Err(load(format!(
                "not an {FORMAT} v{VERSION} file; convert the checkpoint with scripts/convert_e2vid.py"
            )));
        }
        let number = |key: &str| -> Result<f64, ReconstructError> {
            field(key)?
                .parse()
                .map_err(|_| load(format!("{key:?} is not a number")))
        };
        let count = |key: &str| number(key).map(|value| value as usize);

        let kernel = Kernel::new(device, precision);
        let device = burn_device(device)?;
        let tensor = |name: &str| -> Result<(Vec<usize>, Vec<f32>), ReconstructError> {
            let view = file
                .tensor(name)
                .map_err(|_| load(format!("it has no tensor {name:?}")))?;
            if view.dtype() != Dtype::F32 {
                return Err(load(format!("{name:?} is {:?}, not F32", view.dtype())));
            }
            let values = view
                .data()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();
            Ok((view.shape().to_vec(), values))
        };
        let weight = |name: &str| -> Result<Tensor<4>, ReconstructError> {
            let (shape, values) = tensor(name)?;
            if shape.len() != 4 {
                return Err(load(format!(
                    "{name:?} has shape {shape:?}; expected 4 dimensions"
                )));
            }
            Ok(Tensor::from_data(TensorData::new(values, shape), &device))
        };
        let bias = |name: &str| -> Result<Tensor<1>, ReconstructError> {
            let (shape, values) = tensor(&format!("{name}.bias"))?;
            Ok(Tensor::from_data(TensorData::new(values, shape), &device))
        };
        let conv = |name: &str, stride: usize| -> Result<Conv, ReconstructError> {
            Ok(Conv::new(
                weight(&format!("{name}.weight"))?,
                bias(name)?,
                stride,
                kernel,
            ))
        };

        let bins = count("num_bins")?;
        let head = conv("head", 1)?;
        if head.weight.dims()[1] != bins {
            return Err(load(format!(
                "the head takes {} channels but num_bins is {bins}",
                head.weight.dims()[1]
            )));
        }
        let encoders = (0..count("num_encoders")?)
            .map(|i| {
                Ok((
                    conv(&format!("encoders.{i}.conv"), 2)?,
                    conv(&format!("encoders.{i}.gates"), 1)?,
                ))
            })
            .collect::<Result<Vec<_>, ReconstructError>>()?;
        let resblocks = (0..count("num_residual_blocks")?)
            .map(|i| {
                Ok((
                    conv(&format!("resblocks.{i}.conv1"), 1)?,
                    conv(&format!("resblocks.{i}.conv2"), 1)?,
                ))
            })
            .collect::<Result<Vec<_>, ReconstructError>>()?;
        let decoders = (0..encoders.len())
            .map(|i| {
                let name = format!("decoders.{i}");
                let weight_t = weight(&format!("{name}.weight"))?;
                if weight_t.dims()[2..] != [5, 5] {
                    return Err(load(format!("{name} is not a 5x5 transposed convolution")));
                }
                Ok(Upsample::new(weight_t, bias(&name)?, kernel, &device))
            })
            .collect::<Result<Vec<_>, ReconstructError>>()?;
        Ok(Self {
            bins,
            events_per_pixel: number("events_per_pixel")?,
            head,
            encoders,
            resblocks,
            decoders,
            pred: conv("pred", 1)?,
            unsharp: weight("unsharp.kernel")?,
            unsharp_amount: number("unsharp_amount")? as f32,
            device,
            cpu: kernel == Kernel::Native,
            state: None,
        })
    }

    /// Voxel-grid bins the network was trained on.
    pub fn bins(&self) -> usize {
        self.bins
    }

    /// Events per window, per sensor pixel, the network was trained on: a window should hold
    /// `events_per_pixel * width * height` events.
    pub fn events_per_pixel(&self) -> f64 {
        self.events_per_pixel
    }

    /// Forgets the recurrent state, so the next window starts a new sequence.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Reconstructs the frame at the end of `stream`, one window of a sequence, returning the
    /// `height * width` row-major intensities in `[0, 1]` and the frame's `(width, height)`.
    pub fn step(
        &mut self,
        stream: &EventStream,
    ) -> Result<(Vec<f32>, usize, usize), ReconstructError> {
        let (width, height) = stream.sensor_size();
        if let Some((expected, _)) = &self.state {
            if *expected != (width, height) {
                return Err(ReconstructError::SizeChanged {
                    expected: *expected,
                    got: (width, height),
                });
            }
        }
        let input = voxel_grid(stream, self.bins, &self.device)?;

        // Reflection padding up to what the encoders can halve evenly, more of it top and left.
        let multiple = 1 << self.encoders.len();
        let pad_h = height.next_multiple_of(multiple) - height;
        let pad_w = width.next_multiple_of(multiple) - width;
        let (top, left) = (pad_h.div_ceil(2), pad_w.div_ceil(2));
        let input = input.pad((left, pad_w - left, top, pad_h - top), PadMode::Reflect);

        let image = self.forward(input, (width, height));
        // Unsharp mask on the padded frame with zero borders, then the crop: the reference's order.
        let pad = self.unsharp.dims()[2] / 2;
        let blurred = conv2d(
            image.clone(),
            self.unsharp.clone(),
            None,
            ConvOptions::new([1, 1], [pad; 2], [1, 1], 1),
        );
        let image = image * (1.0 + self.unsharp_amount) - blurred * self.unsharp_amount;
        // The whole padded frame comes back and is cropped on the host, where it is a row copy.
        let padded = image
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|error| ReconstructError::Load(format!("{error:?}")))?;
        let values = padded
            .chunks_exact(width + pad_w)
            .skip(top)
            .take(height)
            .flat_map(|row| &row[left..left + width])
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        Ok((values, width, height))
    }

    /// The UNet: head, ConvLSTM encoders, residual blocks, then decoders with summed skips.
    fn forward(&mut self, x: Tensor<4>, size: (usize, usize)) -> Tensor<4> {
        let head = relu(self.head.forward(x));
        let previous = self.state.take().map(|(_, state)| state);
        let mut x = head.clone();
        let mut state = Vec::with_capacity(self.encoders.len());
        for (i, (conv, gates)) in self.encoders.iter().enumerate() {
            x = relu(conv.forward(x));
            let (hidden, cell) = match &previous {
                Some(previous) => previous[i].clone(),
                None => (x.zeros_like(), x.zeros_like()),
            };
            let cell = if self.cpu {
                let (hidden, cell) = convlstm_host(gates, x, hidden, cell);
                x = hidden;
                cell
            } else {
                let gates =
                    gates.forward(Tensor::cat(vec![gates.input(x), gates.input(hidden)], 1));
                let [input, remember, output, candidate]: [Tensor<4>; 4] =
                    gates.chunk(4, 1).try_into().unwrap_or_else(|_| {
                        unreachable!("the gate convolution has 4x the hidden channels")
                    });
                let cell = sigmoid(remember) * cell + sigmoid(input) * tanh(candidate);
                x = sigmoid(output) * tanh(cell.clone());
                cell
            };
            state.push((x.clone(), cell));
        }
        for (conv1, conv2) in &self.resblocks {
            x = relu(conv2.forward(relu(conv1.forward(x.clone()))) + x);
        }
        // Deepest skip first: each decoder takes the hidden state of the encoder that mirrors it.
        for (decoder, (skip, _)) in self.decoders.iter().zip(state.iter().rev()) {
            x = relu(decoder.forward(x + skip.clone()));
        }
        self.state = Some((size, state));
        sigmoid(self.pred.forward(x + head))
    }
}

/// Each event's two votes into a `bins x height x width` voxel grid spanning exactly the window, as
/// flat cell indices and signed weights: rpg_e2vid's `(B - 1)(t - t0) / (tN - t0)`, split
/// bilinearly between the two bins it falls between, polarity as +-1. One pass over the events, so
/// what goes to the device is two numbers per vote rather than a dense grid.
fn bin_events(stream: &EventStream, bins: usize) -> Result<(Vec<i32>, Vec<f32>), ReconstructError> {
    let (width, height) = stream.sensor_size();
    let (xs, ys, ts, ps) = (stream.xs(), stream.ys(), stream.ts(), stream.ps());
    let (first, last) = ts
        .iter()
        .fold((i64::MAX, i64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)));
    // A window with one timestamp puts everything in the first bin, as the reference does.
    let scale = (bins - 1) as f64 / last.saturating_sub(first).max(1) as f64;
    // Split across threads: a 1280x720 window is ~320k events, 3 ms on one core.
    let chunks = ts
        .par_chunks(BIN_CHUNK)
        .enumerate()
        .map(|(chunk, times)| {
            let mut index = Vec::with_capacity(2 * times.len());
            let mut weight = Vec::with_capacity(2 * times.len());
            for i in chunk * BIN_CHUNK..chunk * BIN_CHUNK + times.len() {
                let (x, y) = (usize::from(xs[i]), usize::from(ys[i]));
                if x >= width || y >= height {
                    return Err(ReconstructError::Representation(
                        RepresentationError::EventOutOfBounds {
                            x,
                            y,
                            width,
                            height,
                        },
                    ));
                }
                let position = (ts[i] - first) as f64 * scale;
                let lower = position as usize;
                let upper_weight = (position - lower as f64) as f32;
                let polarity = if ps[i] { 1.0 } else { -1.0 };
                let cell = (lower * height + y) * width + x;
                index.push(cell as i32);
                weight.push(polarity * (1.0 - upper_weight));
                if lower + 1 < bins {
                    index.push((cell + width * height) as i32);
                    weight.push(polarity * upper_weight);
                }
            }
            Ok((index, weight))
        })
        .collect::<Result<Vec<_>, ReconstructError>>()?;
    let votes = chunks.iter().map(|(index, _)| index.len()).sum();
    let (mut index, mut weight) = (Vec::with_capacity(votes), Vec::with_capacity(votes));
    for (chunk_index, chunk_weight) in chunks {
        index.extend(chunk_index);
        weight.extend(chunk_weight);
    }
    Ok((index, weight))
}

/// The window's voxel grid on `device`, normalised so its nonzero cells have mean 0 and standard
/// deviation 1 while empty cells stay 0 — the network sees the same contrast whatever the event
/// rate. Built and normalised on the device: on the host this was a quarter of a 1280x720 frame.
fn voxel_grid(
    stream: &EventStream,
    bins: usize,
    device: &Device,
) -> Result<Tensor<4>, ReconstructError> {
    let (width, height) = stream.sensor_size();
    let (index, weight) = bin_events(stream, bins)?;
    let votes = index.len();
    let mut grid = Tensor::<1>::zeros([bins * height * width], device);
    if votes > 0 {
        grid = grid.scatter(
            0,
            Tensor::<1, Int>::from_data(TensorData::new(index, [votes]), device),
            Tensor::<1>::from_data(TensorData::new(weight, [votes]), device),
            IndexingUpdateOp::Add,
        );
    }
    let empty = grid.clone().equal_elem(0.0);
    let count = empty.clone().bool_not().float().sum().clamp_min(1.0);
    let mean = grid.clone().sum() / count.clone();
    let variance = grid.clone().powi_scalar(2).sum() / count - mean.clone().powi_scalar(2);
    // A constant window has no spread to normalise by; it comes out all zeros rather than NaN.
    let std = variance.sqrt().clamp_min(f32::MIN_POSITIVE);
    Ok(((grid - mean) / std)
        .mask_fill(empty, 0.0)
        .reshape([1, bins, height, width]))
}

/// burn's handle on the device eventcv was asked to use.
fn burn_device(device: accel::Device) -> Result<Device, ReconstructError> {
    match device {
        accel::Device::Cpu => Ok(Device::from(burn::backend::FlexDevice)),
        accel::Device::Gpu => {
            #[cfg(feature = "gpu")]
            if let Some(device) = accel::gpu::with_context(|context| context.burn.clone()) {
                return Ok(device);
            }
            Err(ReconstructError::Unavailable(accel::unavailable_reason()))
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Device;
    use ndarray::array;

    use super::{bin_events, voxel_grid};
    use crate::EventStream;

    #[test]
    fn events_vote_across_the_whole_window_whatever_its_offset() {
        // rpg_e2vid: t* = (B - 1)(t - t0)/(tN - t0) = 4(t - 1000)/100 for t in 1000..=1100,
        // voting (1 - frac) into floor(t*) and frac into floor(t*) + 1, with polarity as +-1.
        let stream = EventStream::from_array2(
            array![[0, 0, 1000, 1], [0, 0, 1010, 0], [0, 0, 1100, 1]],
            1,
            1,
            0.001,
        );
        let (index, weight) = bin_events(&stream, 5).unwrap();
        let mut grid = [0.0_f32; 5];
        for (cell, vote) in index.iter().zip(weight) {
            grid[*cell as usize] += vote;
        }
        let expected = [1.0 - 0.6, -0.4, 0.0, 0.0, 1.0];
        for (got, want) in grid.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{grid:?} != {expected:?}");
        }
    }

    #[test]
    fn the_phase_split_is_the_transposed_convolution() {
        use burn::tensor::module::conv_transpose2d;
        use burn::tensor::ops::ConvTransposeOptions;
        use burn::tensor::{DType, Distribution, Tensor};

        let device = Device::from(burn::backend::FlexDevice);
        let x = Tensor::<4>::random([1, 3, 5, 7], Distribution::Normal(0.0, 1.0), &device);
        let weight = Tensor::<4>::random([3, 2, 5, 5], Distribution::Normal(0.0, 1.0), &device);
        let bias = Tensor::<1>::random([2], Distribution::Normal(0.0, 1.0), &device);
        let options = ConvTransposeOptions::new([2, 2], [2, 2], [1, 1], [1, 1], 1);
        let expected = conv_transpose2d(x.clone(), weight.clone(), Some(bias.clone()), options);
        let got = super::Upsample::phased(weight, bias, DType::F32, &device).forward(x);
        assert_eq!(got.dims(), [1, 2, 10, 14]);
        let (got, expected) = (
            got.into_data().try_to_vec::<f32>().unwrap(),
            expected.into_data().try_to_vec::<f32>().unwrap(),
        );
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-4, "{g} != {e}");
        }
    }

    #[test]
    fn the_fast_exp_is_as_good_as_libm() {
        for i in -8700..=8800 {
            let x = i as f32 * 0.01;
            let (got, want) = (super::exp_fast(x), x.exp());
            assert!(
                ((got - want) / want).abs() < 3e-7,
                "exp({x}) = {got}, not {want}"
            );
        }
    }

    #[test]
    fn normalisation_ignores_and_keeps_empty_cells() {
        // Two pixels, one bin: votes of +1 and +3 normalise to -1 and +1; the rest stay empty.
        let stream = EventStream::from_array2(
            array![[0, 0, 0, 1], [2, 0, 0, 1], [2, 0, 0, 1], [2, 0, 0, 1]],
            4,
            1,
            0.001,
        );
        let device = Device::from(burn::backend::FlexDevice);
        let grid = voxel_grid(&stream, 1, &device).unwrap();
        let values = grid.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(values, [-1.0, 0.0, 1.0, 0.0]);
    }
}
