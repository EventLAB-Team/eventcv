# EventCV

**"OpenCV for event-based vision."** EventCV pairs a fast Rust core (`eventcv-core`) with
a NumPy-friendly Python API for loading, transforming, and representing event-camera data.

- **Read** any common format — `.npz`, `.txt`/`.csv`, ROS `.bag`, `.hdf5`, `.aedat`
  (AEDAT 2.0), `.aedat4` (iniVation DV), and Prophesee `.dat` / EVT2 / EVT3 `.raw` — through
  one {func}`eventcv.load`.
  A DAVIS recording also yields its APS frames and IMU, not just its events, whether it
  arrives as a rosbag or as AEDAT: {func}`eventcv.read_frames`, {func}`eventcv.read_imu`
  and {func}`eventcv.read_camera_info` take a path or an open reader either way.
- **Write every format it reads** — {func}`eventcv.save` is symmetric with {func}`eventcv.load`,
  and every event container can also be written *a window at a time* with
  {class}`~eventcv.EventSink`, so a capture longer than memory streams to disk in whichever format
  the rest of your pipeline wants.
- **Stream** multi-gigabyte recordings without loading them whole via
  {func}`eventcv.open`, with on-disk indexed slicing.
- **Capture live** from a USB event camera with {func}`eventcv.stream` — representations
  per window, recording straight to disk, and hardware caps on the event rate.
- **Transform** streams with chainable, functional geometry / temporal / polarity ops.
- **Represent** events as dense tensors (voxel grids, time surfaces, count images, …)
  ready for NumPy and PyTorch.
- **Detect features** — corner detectors and the unsupervised, trainable
  {class}`~eventcv.FEAST` feature extractor (the event analogue of `features2d`).
- **Estimate motion** — contrast maximisation to recover camera motion, and a tracker that follows
  objects across frames.
- **Augment** for training with seeded, reproducible random ops that compose straight into a
  PyTorch `DataLoader`.
- **Visualise** — colormapped frames, animated `.gif` / `.apng` / `.mp4` export, and
  event-rate analytics.
- **Run a model** — feed a representation to any ONNX network with {class}`~eventcv.Model`.
- **Simulate** events from video with a v2e-grade sensor model — optionally with learned frame
  interpolation in front of it — and **reconstruct** intensity video back from events.
- **Move the heavy work to a GPU** when it pays, with `device="gpu"` or
  {func}`eventcv.set_device`. Off by default: the CPU implementations are the reference, and on a
  modern machine they win for everything but the simulator. See {doc}`representations`.
- **Call it your way** — every operation is available both as a method
  (`stream.voxel()`) and as an OpenCV-style free function (`eventcv.voxel(stream)`).

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
representations
augmentation
feature-detection
motion
streaming
simulation
reconstruction
video
models
cli
tutorials/index
api
```

## Installation

```console
pip install eventcv
```

The wheel bundles its own libhdf5 and its own ONNX Runtime, so `.h5`/`.hdf5` reading and
{class}`~eventcv.Model` inference work with no extra installs. Writing `.mp4` is the one thing
that expects something on your system — `ffmpeg` on your `PATH`; `.gif` and `.apng` need nothing.
Run `eventcv --version` to see which optional features a build has, and which ONNX Runtime it
found ([where that comes from](models.md)).

### Standalone command line

On a machine without Python — a capture box, a CI runner, a colleague who only needs to look at
recordings — the `eventcv` [command](cli.md) also comes as a single binary:

```console
# Linux and macOS
curl -LsSf https://github.com/EventLAB-Team/eventcv/releases/latest/download/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://github.com/EventLAB-Team/eventcv/releases/latest/download/install.ps1 | iex"
```

The script checks the download's SHA-256, then installs to `~/.local/bin`, or on Windows to
`%LOCALAPPDATA%\Programs\eventcv\bin`, which it adds to your user `PATH`. Set
`EVENTCV_INSTALL_DIR` to install somewhere else, and `EVENTCV_VERSION=X.Y.Z` to pin a release. To
upgrade, run it again; to uninstall, delete the binary. Builds cover Linux (x86_64 and aarch64,
glibc 2.28 or newer), macOS (Apple silicon and Intel) and Windows (x64).

The binary reads every format the package does, HDF5 included, and talks to USB cameras. It has
only the commands that need no Python; [Command line](cli.md) lists them. As with the package,
`.mp4` and `simulate` need `ffmpeg`, and on Linux a camera needs udev rules for non-root access.

To build it from a checkout instead, you need a Rust toolchain, a C compiler and cmake:

```console
cargo install --path crates/eventcv-cli --features hdf5,camera
```

### Python 3.8 / ROS 1 Noetic

EventCV supports Python 3.8, so it installs into the system Python of Ubuntu 20.04 and ROS
Noetic. Upgrade pip first: the pip 20.0.2 that Ubuntu 20.04 ships predates the `manylinux_2_28`
wheel tag, so it skips the prebuilt wheel and tries to compile EventCV from source, which needs a
Rust toolchain and cmake.

```console
python3 -m pip install --upgrade pip    # add --user outside a virtualenv
python3 -m pip install eventcv
```

Use `python3 -m pip` rather than `pip` for the second command, so the upgraded pip runs and not
the old one still on your `PATH`. EventCV reads `.bag` files itself, so no ROS Python packages
are needed.

The Rust core's API is documented separately on [docs.rs](https://docs.rs/eventcv-core).

## What EventCV is not

**A dataset library.** N-MNIST, DVS-Gesture, N-Caltech101 and friends are
[tonic](https://tonic.readthedocs.io)'s job, and it does it well — auto-downloading, caching and
versioning roughly twenty standard datasets. EventCV reads files and builds tensors; point tonic at
a downloaded dataset and load its files with {func}`eventcv.load`, or use tonic end-to-end. There is
no value in a second, worse copy of that.

**A model zoo.** {class}`~eventcv.Model` runs any ONNX graph you give it — including the
reconstruction networks in [Reconstruction](reconstruction.md) — but EventCV ships no
architectures and no weights. Bundling them would mean tracking every upstream model's
preprocessing forever; building the tensors correctly is the part that belongs here.
