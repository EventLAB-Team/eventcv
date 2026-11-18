<h1 align="center">
<img src="https://eventcv.net/wp-content/uploads/2026/10/logo.svg" width="200">
</h1><br>


<div align="center">

# EventCV: The open-source event-based computer vision library
![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat-square&logo=rust&logoColor=white&logoSize=auto)
![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54&logoSize=auto)
[![PyPI - Version](https://img.shields.io/pypi/v/eventcv?style=flat-square&label=PyPI&logoColor=ffffff&logoSize=auto)](https://pypi.org/project/eventcv/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/eventcv.svg?style=flat-square&logo=Anaconda&logoColor=ffffff&logoSize=auto)](https://anaconda.org/conda-forge/eventcv)

[![PyPI Downloads](https://img.shields.io/pypi/dm/eventcv?style=flat-square&label=PyPI%20downloads&labelColor=grey&color=green)](https://pepy.tech/projects/eventcv)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/eventcv.svg?style=flat-square&label=Conda%20downloads)](
https://anaconda.org/conda-forge/eventcv)
[![Discord](https://img.shields.io/discord/1572116415547318323?style=flat-square&logo=discord&logoColor=white&label=Discord&labelColor=%235865F2&color=%235b209a)](https://eventcv.net/discord)

_EventCV is "OpenCV" for event-based computer vision applications._

</div>

## Resources
- **Website** - https://eventcv.net
- **Documentation** - https://docs.eventcv.net
- **Rust core API** - https://docs.rs/eventcv-core
- **Discord Server** - https://eventcv.net/discord
- **Supported event cameras** - https://github.com/neuromorphicsystems/neuromorphic-drivers#supported-devices

## Installation
```console
# pip
pip install eventcv

# conda
conda install eventcv

# pixi
pixi add eventcv

# cURL
curl -fsSL https://eventcv.net/install | sh
```

## Getting started
### Basic usage
```python
import eventcv as ecv

# open an event stream file
events = ecv.open("events.hdf5")

# stream from an event camera
ecv.stream()
```
> [!IMPORTANT]
> Linux users will need to set UDEV rules for using cameras, [click for more details.](https://github.com/neuromorphicsystems/neuromorphic-drivers#udev-rules)

## Contributing
EventCV is an open-sourced project for the event-based computer vision community. We welcome and appreciate contributions from users who would like to propose improvements or feature additions.

Please read the [contribution guidelines](https://github.com/EventLAB-Team/eventcv/blob/main/.github/CONTRIBUTING.md) for more details.