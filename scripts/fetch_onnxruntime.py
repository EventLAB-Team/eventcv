#!/usr/bin/env python3
"""Fetch the ONNX Runtime shared library that the wheels bundle.

`Model` opens ONNX Runtime at run time rather than linking it in, and the wheels carry a copy so
that `pip install eventcv` needs no companion package. This script puts that copy in place: it
downloads the pinned `onnxruntime` wheel for the host platform and lifts the library out of it.

Why Microsoft's build and not the one `ort` can fetch for itself: the prebuilt *static* runtime is
compiled against glibc 2.38, so linking it makes the whole extension unloadable on Ubuntu 22.04,
Debian 12 and RHEL 9. The library inside the `onnxruntime` wheel is built in a manylinux container
and needs glibc 2.27, which is below the floor the eventcv wheels themselves declare.

Run it before building a wheel — CI does, in every wheel job:

    python scripts/fetch_onnxruntime.py

A source build that skips it is fine: eventcv then falls back to an ONNX Runtime found elsewhere
(`pip install eventcv[onnx]`, a conda environment, `ORT_DYLIB_PATH`), and says so if there is none.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# The version the wheels ship — the newest on PyPI, which trails the GitHub release by a little.
# `api-17` in eventcv-core's `ort` dependency means anything from ONNX Runtime 1.17 up satisfies
# the binding, so this pin can move freely; bump it here and nowhere else.
ONNXRUNTIME_VERSION = "1.28.0"

# Where the library lands, next to the Python package so it travels in the wheel.
DESTINATION = Path(__file__).resolve().parent.parent / "python" / "eventcv" / "_libs"

# What to lift out of the wheel. The library only — `onnxruntime_pybind11_state*.so` is the Python
# binding (another 26 MB) and the providers shim is for execution providers eventcv does not
# register. The licence files are an obligation, not an option: the binary is redistributed.
if sys.platform == "win32":
    _LIBRARY_PATTERNS = ("onnxruntime/capi/onnxruntime.dll",)
elif sys.platform == "darwin":
    _LIBRARY_PATTERNS = (
        "onnxruntime/capi/libonnxruntime.dylib",
        "onnxruntime/capi/libonnxruntime.*.dylib",
    )
else:
    _LIBRARY_PATTERNS = ("onnxruntime/capi/libonnxruntime.so*",)

# Renamed on the way out so they cannot be mistaken for eventcv's own licence.
_LICENCE_FILES = {
    "onnxruntime/LICENSE": "LICENSE.onnxruntime",
    "onnxruntime/ThirdPartyNotices.txt": "ThirdPartyNotices.onnxruntime.txt",
}


def _download(version: str, into: Path) -> Path:
    """Downloads the `onnxruntime` wheel for this platform and returns its path.

    `pip download` rather than a hand-written URL table: it resolves the right wheel for the host
    platform, and checks the hash the index publishes.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            f"onnxruntime=={version}",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(into),
        ],
        check=True,
    )
    wheels = sorted(into.glob("onnxruntime-*.whl"))
    if not wheels:
        raise SystemExit(f"pip downloaded no onnxruntime wheel into {into}")
    return wheels[0]


def _copy(archive: zipfile.ZipFile, name: str, target: Path, mode: int) -> Path:
    with archive.open(name) as source, open(target, "wb") as sink:
        shutil.copyfileobj(source, sink)
    target.chmod(mode)
    return target


def _extract(wheel: Path, destination: Path) -> list[Path]:
    """Copies the library and its licences out of `wheel`, flattening the paths."""
    written = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        libraries = [
            name
            for name in names
            if any(fnmatch.fnmatch(name, pattern) for pattern in _LIBRARY_PATTERNS)
        ]
        if not libraries:
            raise SystemExit(
                f"{wheel.name} contains no ONNX Runtime library matching {_LIBRARY_PATTERNS}"
            )
        for name in libraries:
            written.append(_copy(archive, name, destination / Path(name).name, 0o755))
        for name, renamed in _LICENCE_FILES.items():
            if name in names:
                written.append(_copy(archive, name, destination / renamed, 0o644))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--version",
        default=ONNXRUNTIME_VERSION,
        help=f"ONNX Runtime version to bundle (default: {ONNXRUNTIME_VERSION})",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DESTINATION,
        help=f"where to put it (default: {DESTINATION})",
    )
    args = parser.parse_args(argv)

    # Replaced wholesale rather than added to, so switching versions cannot leave two libraries
    # behind for the loader to choose between.
    if args.destination.exists():
        shutil.rmtree(args.destination)
    args.destination.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as scratch:
        wheel = _download(args.version, Path(scratch))
        written = _extract(wheel, args.destination)

    for path in written:
        print(f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
