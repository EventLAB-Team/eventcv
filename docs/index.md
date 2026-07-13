# EventCV

**"OpenCV for event-based vision."** EventCV pairs a fast Rust core (`eventcv-core`) with
a NumPy-friendly Python API for loading, transforming, and representing event-camera data.

- **Read** any common format — `.npz`, `.txt`/`.csv`, ROS `.bag`, `.hdf5`, `.aedat`
  (AEDAT 2.0), and Prophesee `.dat` — through one {func}`eventcv.load`.
- **Stream** multi-gigabyte recordings without loading them whole via
  {func}`eventcv.open`, with on-disk indexed slicing.
- **Transform** streams with chainable, functional geometry / temporal / polarity ops.
- **Represent** events as dense tensors (voxel grids, time surfaces, count images, …)
  ready for NumPy and PyTorch.
- **Detect features** — corner detectors and the unsupervised, trainable
  {class}`~eventcv.FEAST` feature extractor (the event analogue of `features2d`).
- **Call it your way** — every operation is available both as a method
  (`stream.voxel()`) and as an OpenCV-style free function (`eventcv.voxel(stream)`).

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
representations
feature-detection
tutorials/index
api
```

## Installation

```console
pip install eventcv
```

The wheel bundles its own libhdf5, so `.h5`/`.hdf5` support works with no extra installs.
The Rust core's API is documented separately on [docs.rs](https://docs.rs/eventcv-core).
