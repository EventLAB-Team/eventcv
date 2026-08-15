# EventCV

**"OpenCV for event-based vision."** EventCV pairs a fast Rust core (`eventcv-core`) with
a NumPy-friendly Python API for loading, transforming, and representing event-camera data.

- **Read** any common format — `.npz`, `.txt`/`.csv`, ROS `.bag`, `.hdf5`, `.aedat`
  (AEDAT 2.0), and Prophesee `.dat` / EVT3 `.raw` — through one {func}`eventcv.load`.
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
- **Simulate** events from video with a v2e-grade sensor model, and **reconstruct** intensity video
  back from events.
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

The wheel bundles its own libhdf5, so `.h5`/`.hdf5` support works with no extra installs, and its
own ONNX Runtime for {class}`~eventcv.Model`. Writing `.mp4` is the one thing that expects
something on your system — `ffmpeg` on `PATH`; `.gif` and `.apng` need nothing. Run
`eventcv --version` to see which optional features a build has.

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
