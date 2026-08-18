//! Where a computation runs — the CPU (always) or a GPU (when the `gpu` feature is built in and an
//! adapter exists).
//!
//! Every representation has a CPU implementation, and that implementation is the reference: it is
//! what the tests assert against, what the benchmarks compare to, and what runs unless something
//! asks otherwise. A GPU kernel is an *alternative* to it, never a replacement, so a build without
//! the feature, a machine without a GPU, and a machine with one all produce the same answers to
//! within the tolerance each kernel documents.
//!
//! The backend is [`wgpu`], which compiles to Metal on macOS, Vulkan on Linux and Android, and
//! DX12 on Windows — one set of shaders rather than a CUDA path plus a Metal path, and no vendor
//! toolkit at build time. It is already in the tree for the viewer.
//!
//! # Choosing
//!
//! The default is [`Device::Cpu`]. It is read once from `EVENTCV_DEVICE` (`cpu` / `gpu`) and can be
//! changed for the session with [`set_default_device`]; the bindings expose both, plus a per-call
//! `device=`. Asking for a GPU that is not there is an **error**, never a quiet fall back to the
//! CPU — "my GPU is not being used" should not be something a user has to time a benchmark to find
//! out.

use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "gpu")]
pub(crate) mod gpu;
#[cfg(feature = "gpu")]
pub(crate) mod sim;

/// What a representation asks a kernel to do. Named here rather than in [`gpu`] so the call sites
/// describing their kernel compile whether or not the feature is on — only the dispatch itself is
/// behind the flag.
#[cfg(feature = "gpu")]
pub(crate) type GpuDispatch = gpu::Dispatch;

/// The same shape when the feature is off, so `representation` needs no `cfg` of its own. Nothing
/// reads the fields in that build — the dispatch is constructed and then refused — which is exactly
/// what keeps every kernel's description in one place instead of behind a `cfg` at each call site.
#[cfg(not(feature = "gpu"))]
#[allow(dead_code)]
pub(crate) struct GpuDispatch {
    pub(crate) entry: &'static str,
    pub(crate) cells: usize,
    pub(crate) initial: i32,
    pub(crate) bins: u32,
    pub(crate) span_ms: f32,
    pub(crate) fixed_one: f32,
    pub(crate) window_ms: Option<f64>,
    pub(crate) needs_ages: bool,
}

/// Fixed-point scale the accumulating kernels use; see [`gpu::FIXED_ONE`].
#[cfg(feature = "gpu")]
pub(crate) const FIXED_ONE: f32 = gpu::FIXED_ONE;
#[cfg(not(feature = "gpu"))]
pub(crate) const FIXED_ONE: f32 = 65536.0;

/// Which backend a representation runs on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// The reference implementation. Parallel where it pays, and fast enough that this remains the
    /// default for everything but large-batch work.
    #[default]
    Cpu,
    /// A wgpu compute kernel. Worth it once a call is large enough to cover the upload and
    /// readback — see the `representations` benchmark group for where that crossover sits.
    Gpu,
}

impl Device {
    /// Parses `"cpu"` / `"gpu"` (case-insensitively), the spelling used by `EVENTCV_DEVICE`, the
    /// Python `device=` argument, and `set_device`.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "gpu" | "cuda" | "metal" => Some(Self::Gpu),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

/// The session default, as a `u8` so it can be swapped without a lock. `u8::MAX` means "not yet
/// read from the environment".
static DEFAULT_DEVICE: AtomicU8 = AtomicU8::new(u8::MAX);

/// The device used when a caller does not name one.
///
/// Seeded once from `EVENTCV_DEVICE`, so a CI job or a shell can select the GPU without touching
/// any call site; an unset or unrecognised value leaves it on the CPU.
pub fn default_device() -> Device {
    match DEFAULT_DEVICE.load(Ordering::Relaxed) {
        0 => Device::Cpu,
        1 => Device::Gpu,
        _ => {
            let device = std::env::var("EVENTCV_DEVICE")
                .ok()
                .and_then(|name| Device::parse(&name))
                .unwrap_or_default();
            set_default_device(device);
            device
        }
    }
}

/// Sets the device used when a caller does not name one, for the rest of the session.
pub fn set_default_device(device: Device) {
    DEFAULT_DEVICE.store(device as u8, Ordering::Relaxed);
}

/// Whether a GPU kernel can actually run here: the `gpu` feature is built in *and* an adapter was
/// found. Probing opens the adapter once and caches it, so the first call is the expensive one.
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu::with_context(|_| ()).is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Closes the GPU if one was opened, waiting for anything still queued.
///
/// Call at the end of a process. A GPU device that is still open while the process tears itself
/// down faults intermittently on some drivers — after the last line of the program has run, which
/// makes it look like the library corrupted something when it did not. The Python bindings register
/// this with `atexit`, so nothing has to remember. Idempotent, and a later call simply reopens.
pub fn shutdown() {
    #[cfg(feature = "gpu")]
    {
        gpu::shutdown();
    }
}

/// [`unavailable_reason`] for callers outside the crate — the bindings, which raise it when
/// `set_device("gpu")` is asked for on a machine or a build that cannot do it.
pub fn unavailable_reason_public() -> String {
    unavailable_reason()
}

/// Why a GPU run could not happen, as a sentence that says what to do about it. Returned rather
/// than falling back, so a caller that asked for the GPU learns it did not get one.
pub(crate) fn unavailable_reason() -> String {
    #[cfg(feature = "gpu")]
    {
        "device=\"gpu\" was requested but no compatible adapter was found (wgpu could not open a \
         Vulkan, Metal or DX12 device here); use device=\"cpu\""
            .to_owned()
    }
    #[cfg(not(feature = "gpu"))]
    {
        "device=\"gpu\" was requested but this build has no GPU support; rebuild with \
         --features gpu, or use device=\"cpu\""
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{default_device, set_default_device, Device};

    #[test]
    fn device_names_round_trip() {
        for device in [Device::Cpu, Device::Gpu] {
            assert_eq!(Device::parse(device.as_str()), Some(device));
        }
        // The vendor names people reach for map onto the one portable backend.
        assert_eq!(Device::parse("CUDA"), Some(Device::Gpu));
        assert_eq!(Device::parse(" Metal "), Some(Device::Gpu));
        assert_eq!(Device::parse("tpu"), None);
    }

    #[test]
    fn the_default_is_the_cpu_and_can_be_moved() {
        set_default_device(Device::Cpu);
        assert_eq!(default_device(), Device::Cpu);
        set_default_device(Device::Gpu);
        assert_eq!(default_device(), Device::Gpu);
        set_default_device(Device::Cpu);
    }
}

/// Every GPU kernel, checked against the CPU implementation it mirrors.
///
/// These skip when no adapter is available rather than failing, so the suite is honest on a machine
/// without a GPU and on CI. `EVENTCV_REQUIRE_GPU=1` turns the skip into a failure, which is how a
/// machine that *should* have one keeps these from quietly going silent.
#[cfg(test)]
mod gpu_tests {
    use super::{gpu_available, Device};
    use crate::representation::{
        AveragedTimeSurface, CountMask, EventCount, EventFrameData, Polarity, Representation,
        TimeSurface, VoxelGrid,
    };
    use crate::{EventStream, EventStreamBuilder};

    fn skip_without_gpu() -> bool {
        if gpu_available() {
            return false;
        }
        assert!(
            std::env::var("EVENTCV_REQUIRE_GPU").is_err(),
            "EVENTCV_REQUIRE_GPU is set but no adapter was found"
        );
        true
    }

    /// A recording busy enough that pixels collide — which is the whole point, since collisions are
    /// where an order-dependent accumulator would show up.
    fn stream() -> EventStream {
        let mut builder = EventStreamBuilder::new(64, 48, 0.001);
        for index in 0..20_000i64 {
            let x = ((index * 37) % 64) as u16;
            let y = ((index * 11) % 48) as u16;
            builder.push(x, y, index * 3, index % 3 != 0);
        }
        builder.build()
    }

    fn floats(frame: &crate::representation::EventFrame) -> Vec<f32> {
        match frame.data() {
            EventFrameData::F32(values) => values.clone(),
            other => panic!("expected float data, got {other:?}"),
        }
    }

    #[test]
    fn integer_kernels_match_the_cpu_exactly() {
        if skip_without_gpu() {
            return;
        }
        let stream = stream();
        for normalize in [false, true] {
            let counter = EventCount::new(normalize);
            assert_eq!(
                counter.generate_on(&stream, Device::Gpu).unwrap().data(),
                counter.generate(&stream).unwrap().data(),
                "event counts are integer sums and must be identical, not merely close"
            );
            let polarity = Polarity::new(normalize);
            assert_eq!(
                polarity.generate_on(&stream, Device::Gpu).unwrap().data(),
                polarity.generate(&stream).unwrap().data()
            );
        }
        let mask = CountMask::new(99.0, false);
        assert_eq!(
            mask.generate_on(&stream, Device::Gpu).unwrap().data(),
            mask.generate(&stream).unwrap().data()
        );
    }

    #[test]
    fn float_kernels_match_the_cpu_within_the_fixed_point_quantum() {
        if skip_without_gpu() {
            return;
        }
        let stream = stream();
        // Q16.16 rounding plus `f32` versus the CPU's `f64` age arithmetic. A voxel cell here holds
        // hundreds of events, so this bounds the *accumulated* difference, not a single one.
        let tolerance = 1e-3;
        for (name, cpu, gpu) in [
            (
                "voxel",
                floats(&VoxelGrid::new(5, 30.0).generate(&stream).unwrap()),
                floats(
                    &VoxelGrid::new(5, 30.0)
                        .generate_on(&stream, Device::Gpu)
                        .unwrap(),
                ),
            ),
            (
                "tsurf",
                floats(&TimeSurface::new(30.0).generate(&stream).unwrap()),
                floats(
                    &TimeSurface::new(30.0)
                        .generate_on(&stream, Device::Gpu)
                        .unwrap(),
                ),
            ),
            (
                "atsurf",
                floats(&AveragedTimeSurface::new(30.0).generate(&stream).unwrap()),
                floats(
                    &AveragedTimeSurface::new(30.0)
                        .generate_on(&stream, Device::Gpu)
                        .unwrap(),
                ),
            ),
        ] {
            assert_eq!(cpu.len(), gpu.len(), "{name}: shape");
            let worst = cpu
                .iter()
                .zip(&gpu)
                .map(|(cpu, gpu)| (cpu - gpu).abs())
                .fold(0.0_f32, f32::max);
            assert!(worst <= tolerance, "{name}: worst cell differs by {worst}");
        }
    }

    #[test]
    fn a_kernel_gives_the_same_answer_every_run() {
        if skip_without_gpu() {
            return;
        }
        let stream = stream();
        let first = floats(
            &VoxelGrid::new(5, 30.0)
                .generate_on(&stream, Device::Gpu)
                .unwrap(),
        );
        for _ in 0..3 {
            assert_eq!(
                first,
                floats(
                    &VoxelGrid::new(5, 30.0)
                        .generate_on(&stream, Device::Gpu)
                        .unwrap()
                ),
                "integer accumulation must make repeated runs bit-identical"
            );
        }
    }
}
