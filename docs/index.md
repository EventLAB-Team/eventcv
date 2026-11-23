# EventCV

<div class="hero">
<img src="_static/logo.svg" class="hero-logo" alt="EventCV">
<p class="hero-tagline">The open-source event-based computer vision library</p>
<p class="hero-sub">Load, transform and represent event-camera data in Python, on a fast Rust core.</p>
<p class="hero-badges">
<a href="https://pypi.org/project/eventcv/"><img src="https://img.shields.io/pypi/v/eventcv?style=flat-square&label=PyPI&logo=pypi&logoColor=ffffff" alt="PyPI"></a>
<a href="https://anaconda.org/conda-forge/eventcv"><img src="https://img.shields.io/conda/vn/conda-forge/eventcv.svg?style=flat-square&logo=anaconda&logoColor=ffffff" alt="conda-forge"></a>
<a href="https://github.com/EventLAB-Team/eventcv"><img src="https://img.shields.io/github/stars/EventLAB-Team/eventcv?style=flat-square&logo=github" alt="GitHub"></a>
<a href="https://eventcv.net/discord"><img src="https://img.shields.io/discord/1572116415547318323?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=%235865F2&color=%235b209a" alt="Discord"></a>
</p>
</div>

## Installation

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card}
:class-card: install-card
<img src="https://cdn.simpleicons.org/pypi" class="install-logo" alt=""> **pip**

```console
pip install eventcv
```
:::

:::{grid-item-card}
:class-card: install-card
<img src="https://cdn.simpleicons.org/anaconda" class="install-logo" alt=""> **conda**

```console
conda install -c conda-forge eventcv
```
:::

:::{grid-item-card}
:class-card: install-card
<img src="https://raw.githubusercontent.com/prefix-dev/pixi/main/docs/assets/pixi.png" class="install-logo" alt=""> **pixi**

```console
pixi add eventcv
```
:::

:::{grid-item-card}
:class-card: install-card
<img src="https://cdn.simpleicons.org/curl" class="install-logo" alt=""> **cURL** (standalone CLI)

```console
curl -fsSL https://eventcv.net/install | sh
```
:::
::::

See {doc}`installation` for supported platforms, Windows and troubleshooting.


## Example

```python
import eventcv as ecv

stream = ecv.load("recording.h5")  # any supported format
stream = stream.crop(0, 0, 320, 240).hot_pixel_filter()  # transforms chain
voxel = stream.voxel(bins=5).numpy()  # (5, H, W) float32 array

reader = ecv.open("huge.h5", dt_ms=30)  # a large file, sliced lazily
frame = reader[100].tsurf()  # window 100 (30 ms each) as a time surface
```

## Navigation

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
Load, transform and save a recording in five minutes.
:::

:::{grid-item-card} Loading
:link: loading
:link-type: doc
Read any format, or build streams from NumPy.
:::

:::{grid-item-card} Large files
:link: large-files
:link-type: doc
Slice multi-gigabyte recordings without loading them.
:::

:::{grid-item-card} Transforms
:link: transforms
:link-type: doc
Crop, flip, filter, mask and undistort.
:::

:::{grid-item-card} Representations
:link: representations
:link-type: doc
Turn events into tensors: voxel grids, time surfaces, counts.
:::

:::{grid-item-card} Live cameras
:link: streaming
:link-type: doc
Stream, record and process from a USB event camera.
:::

:::{grid-item-card} Augmentation
:link: augmentation
:link-type: doc
Seeded random transforms for training.
:::

:::{grid-item-card} Feature detection
:link: feature-detection
:link-type: doc
Corner detectors and FEAST feature learning.
:::

:::{grid-item-card} Motion estimation
:link: motion
:link-type: doc
Contrast maximisation and object tracking.
:::

:::{grid-item-card} Simulation
:link: simulation
:link-type: doc
Generate events from video.
:::

:::{grid-item-card} Reconstruction
:link: reconstruction
:link-type: doc
Reconstruct intensity video from events with E2VID.
:::

:::{grid-item-card} Video and playback
:link: video
:link-type: doc
Export animations and play recordings.
:::

:::{grid-item-card} Models
:link: models
:link-type: doc
Run ONNX networks on representations.
:::

:::{grid-item-card} Command line
:link: cli
:link-type: doc
Inspect, convert and render from the shell.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Every function and class.
:::
::::

The Rust core is documented on [docs.rs](https://docs.rs/eventcv-core).

## Supported formats

| Format | Extension | Read | Write |
|---|---|:-:|:-:|
| NumPy | `.npz` | ✓ | ✓ |
| Text / CSV | `.txt`, `.csv` | ✓ | ✓ |
| HDF5 | `.h5`, `.hdf5` | ✓ | ✓ |
| ROS bag | `.bag` | ✓ | ✓ |
| AEDAT 2.0 | `.aedat` | ✓ | ✓ (DAVIS sizes only) |
| AEDAT 4 (iniVation DV) | `.aedat4` | ✓ | ✓ |
| Prophesee DAT | `.dat` | ✓ | ✓ |
| Prophesee RAW (EVT2/EVT3) | `.raw` | ✓ | ✓ |
| E2VID text | `.zip` | | ✓ |

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
```

```{toctree}
:hidden:
:caption: User guide

concepts
loading
large-files
saving
transforms
representations
augmentation
feature-detection
motion
streaming
network
video
simulation
reconstruction
models
gpu
```

```{toctree}
:hidden:
:caption: Tools

cli
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: Reference

api
```
