<div align="center">

<img src="https://eventcv.net/wp-content/uploads/2026/10/logo.svg" width="200">

#  _The open-source event-based computer vision library_
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat-square&logo=rust&logoColor=white&logoSize=auto)
![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54&logoSize=auto)
[![PyPI - Version](https://img.shields.io/pypi/v/eventcv?style=flat-square&label=PyPI&logo=pypi&logoColor=ffffff&logoSize=auto)](https://pypi.org/project/eventcv/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/eventcv.svg?style=flat-square&logo=Anaconda&logoColor=ffffff&logoSize=auto)](https://anaconda.org/conda-forge/eventcv)

[![PyPI Downloads](https://img.shields.io/pypi/dm/eventcv?style=flat-square&label=PyPI%20downloads&labelColor=grey&color=green)](https://pepy.tech/projects/eventcv)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/eventcv.svg?style=flat-square&label=Conda%20downloads)](
https://anaconda.org/conda-forge/eventcv)
[![Discord](https://img.shields.io/discord/1572116415547318323?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=%235865F2&color=%235b209a)](https://eventcv.net/discord)

**EventCV** is "OpenCV" for computer vision applications using **event cameras** and **event-based** data.

_Supported for:_ 

<img src="https://cdn.simpleicons.org/linux" width="60" alt="Linux">
<picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.simpleicons.org/apple/FFFFFF"><img src="https://cdn.simpleicons.org/apple/000000" width="60" alt="Apple"></picture> &nbsp;
<img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/windows11/windows11-original.svg" width="60" alt="Windows">

</div>

## Installation
<div align="center">
<table>
<tbody>
<tr>
<td align="center"><img src="https://cdn.simpleicons.org/pypi" width="50" alt="PyPI"><br><b>pip</b></td>
<td>

```console
pip install eventcv
```

</td>
<td align="center"><img src="https://cdn.simpleicons.org/anaconda" width="50" alt="Anaconda"><br><b>conda</b></td>
<td>

```console
conda install -c conda-forge eventcv
```

</td>
</tr>
</tbody>
<tbody>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/prefix-dev/pixi/main/docs/assets/pixi.png" width="50" alt="Pixi"><br><b>pixi</b></td>
<td>

```console
pixi add eventcv
```

</td>
<td align="center"><img src="https://cdn.simpleicons.org/curl" width="50" alt="cURL"><br><b>cURL</b></td>
<td>

```console
curl -fsSL https://eventcv.net/install | sh
```

</td>
</tr>
</tbody>
</table>

</div>

## Getting started
### Basic usage
```python
import eventcv as ecv

# open an event stream file
events = ecv.open("events.hdf5")

# slice events and generate an event represenetation
slice = ecv.open("events.hdf5", repr="voxel").slice(100)

# stream from an event camera
ecv.stream().show()
```
> [!IMPORTANT]
> Linux users will need to set UDEV rules to stream from cameras, [click for more details.](https://github.com/neuromorphicsystems/neuromorphic-drivers#udev-rules)

### Text-based user interface (TUI)
Simply run `eventcv` for an intuitive, smooth TUI in your terminal that **streams from cameras** and **plays back recordings**.

<div align="center">

![EventCV Demo](https://eventcv.net/videos/tuihome.webp)

</div>

> [!TIP]
> For best results, we recommend using a high-resolution terminal, [such as Kitty.](https://sw.kovidgoyal.net/kitty/)


## Resources
- <img src="https://cdn.simpleicons.org/discord" width="20" align="absmiddle" alt="Discord"> **Discord Server** - https://eventcv.net/discord
- <img src="https://cdn.simpleicons.org/readthedocs/FFFFFF" width="20" align="absmiddle" alt="Read the docs"> **Documentation** - https://docs.eventcv.net
- <img src="https://cdn.simpleicons.org/rust/D34516" width="20" align="absmiddle" alt="Rust"> **Rust core API** - https://docs.rs/eventcv-core
- 🔧 **Supported event cameras** - [https://github.com/neuromorphicsystems](https://github.com/neuromorphicsystems/neuromorphic-drivers#supported-devices)
- 📚 **Tutorials** - https://eventcv.net/tutorials
- 🌐 **Website** - https://eventcv.net

## Contributing
EventCV is an open-sourced project for the event-based computer vision community. We welcome and appreciate contributions from users who would like to propose improvements or feature additions.

Please read the [contribution guidelines](https://github.com/EventLAB-Team/eventcv/blob/main/.github/CONTRIBUTING.md) for more details.