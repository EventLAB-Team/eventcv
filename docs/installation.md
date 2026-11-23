# Installation

::::{grid} 1
:gutter: 2

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
<img src="https://cdn.simpleicons.org/curl" class="install-logo" alt=""> **cURL**: the {doc}`command line <cli>` as a single binary, no Python needed

```console
curl -fsSL https://eventcv.net/install | sh
```

On Windows (PowerShell):

```console
powershell -ExecutionPolicy ByPass -c "irm https://github.com/EventLAB-Team/eventcv/releases/latest/download/install.ps1 | iex"
```
:::
::::

## Platforms

| | Python package | CLI binary |
|---|---|---|
| Linux | x86_64, aarch64 | x86_64, aarch64 (glibc 2.28+) |
| macOS | Apple silicon | Apple silicon, Intel |
| Windows | x64 | x64 |
| Python | CPython 3.8+ | Not needed |

Both read every format, including HDF5, and stream from USB cameras. The Python package also bundles
ONNX Runtime for {class}`~eventcv.Model`.

## Optional dependencies

| For | You need |
|---|---|
| Writing `.mp4` | `ffmpeg` on your `PATH`. `.gif` and `.apng` need nothing. |
| USB cameras on Linux | [udev rules](https://github.com/neuromorphicsystems/neuromorphic-drivers#udev-rules) for non-root access. |

## More

:::{dropdown} Check which features your build has
```console
eventcv --version
```
Lists the optional features compiled in, and which ONNX Runtime was found.
:::

:::{dropdown} CLI installer options
The installer verifies the download's SHA-256, installs to `~/.local/bin` (Windows:
`%LOCALAPPDATA%\Programs\eventcv\bin`) and adds it to your `PATH`.

| Variable | Effect |
|---|---|
| `EVENTCV_INSTALL_DIR` | Install somewhere else. |
| `EVENTCV_VERSION=X.Y.Z` | Install a specific release. |

Re-run the installer to upgrade. Delete the binary to uninstall.
:::

:::{dropdown} pip tries to compile EventCV (Ubuntu 20.04, ROS Noetic)
Ubuntu 20.04 ships pip 20.0.2, which is too old to recognise EventCV's wheels, so it falls back to
building from source. Upgrade pip first:

```console
python3 -m pip install --upgrade pip
python3 -m pip install eventcv
```

Use `python3 -m pip` so the upgraded pip runs. No ROS Python packages are needed: EventCV reads
`.bag` files itself.
:::

:::{dropdown} Build from source
Requires a Rust toolchain, a C compiler and cmake.

```console
pip install maturin cmake
maturin develop                                                  # Python package
cargo install --path crates/eventcv-cli --features hdf5,camera   # CLI binary
```
:::
