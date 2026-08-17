"""Finding the ONNX Runtime shared library that :class:`~eventcv.Model` runs on.

EventCV links ONNX Runtime dynamically: the extension opens `libonnxruntime` on first use rather
than compiling a copy into itself. That is not a packaging preference — the prebuilt *static*
runtime is built against glibc 2.38, so linking it made the entire extension fail to import on
Ubuntu 22.04, Debian 12 and RHEL 9, event reading and all.

So something has to say *which* library to open, and this module answers that from whatever the
installation happens to provide, in a fixed order:

1. `ORT_DYLIB_PATH` — an explicit choice always wins.
2. An already-imported `onnxruntime`, so a process using both does not hold two runtimes.
3. The copy bundled in the wheel (`eventcv/_libs/`) — the ordinary `pip install eventcv` case.
4. The environment's own `lib` directory — conda-forge's `onnxruntime-cpp` lives there.
5. An installed but unimported `onnxruntime` package — source builds with `eventcv[onnx]`.

Nothing found leaves `ORT_DYLIB_PATH` unset, and the extension falls back to asking the loader for
the platform's default name, which finds a system-wide install. Only if *that* fails does
`Model(...)` raise, and the message says how to fix it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# What the library is called on each platform. Linux and macOS builds carry the version in the
# filename (`libonnxruntime.so.1.28.0`), so the patterns have to match both forms.
if sys.platform == "win32":
    _PATTERNS = ("onnxruntime.dll",)
elif sys.platform == "darwin":
    _PATTERNS = ("libonnxruntime.dylib", "libonnxruntime.*.dylib")
else:
    _PATTERNS = ("libonnxruntime.so", "libonnxruntime.so.*")

# Where an environment keeps its shared libraries, relative to `sys.prefix`. Conda puts DLLs under
# `Library/bin` on Windows and everything else in `lib`.
_PREFIX_DIRECTORIES = ("Library/bin", "Library/lib") if sys.platform == "win32" else ("lib",)


def _first_match(directory: Path) -> str | None:
    """The library in `directory`, or `None`. Deterministic when several versions sit together."""
    for pattern in _PATTERNS:
        # Sorted for a stable answer; the last name wins, which is the highest version for the
        # usual `.so.1.28.0` suffixes.
        matches = sorted(path for path in directory.glob(pattern) if path.is_file())
        if matches:
            return str(matches[-1])
    return None


def _from_environment() -> str | None:
    path = os.environ.get("ORT_DYLIB_PATH")
    return path or None


def _from_imported_onnxruntime() -> str | None:
    """The library of an `onnxruntime` this process has already imported.

    Preferred over our own bundled copy: the operating system would keep both resident, and the
    one already open is by definition working.
    """
    module = sys.modules.get("onnxruntime")
    origin = getattr(module, "__file__", None)
    return _first_match(Path(origin).parent / "capi") if origin else None


def _bundled() -> str | None:
    """The copy the wheel ships, put there by `scripts/fetch_onnxruntime.py` at build time."""
    return _first_match(Path(__file__).resolve().parent / "_libs")


def _from_prefix() -> str | None:
    """The environment's own copy — conda-forge installs `onnxruntime-cpp` into `$PREFIX/lib`."""
    for relative in _PREFIX_DIRECTORIES:
        found = _first_match(Path(sys.prefix) / relative)
        if found:
            return found
    return None


def _from_installed_onnxruntime() -> str | None:
    """The `onnxruntime` package's library, without importing it.

    `find_spec` rather than an import: importing costs roughly a second and loads a second copy of
    the runtime, neither of which is wanted just to learn where a file is.
    """
    try:
        spec = importlib.util.find_spec("onnxruntime")
    except (ImportError, ValueError):  # pragma: no cover - a broken install, not our problem
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        found = _first_match(Path(location) / "capi")
        if found:
            return found
    return None


# In priority order; the labels are what `eventcv --version` reports, so they read as answers to
# "where did this runtime come from?".
_SOURCES = (
    ("ORT_DYLIB_PATH", _from_environment),
    ("imported onnxruntime", _from_imported_onnxruntime),
    ("bundled", _bundled),
    ("environment", _from_prefix),
    ("onnxruntime package", _from_installed_onnxruntime),
)


def find_runtime() -> tuple[str, str] | None:
    """The library to load and where it came from, or `None` if this machine has none."""
    for origin, locate in _SOURCES:
        found = locate()
        if found is not None:
            return found, origin
    return None


# What `configure()` settled on. Remembered because configuring *is* setting `ORT_DYLIB_PATH`, so
# asking again afterwards would only ever answer "ORT_DYLIB_PATH" and lose the real origin.
_resolved: tuple[str, str] | None = None


def configure() -> None:
    """Points `ort` at the runtime this installation provides, unless the user chose one.

    Called at import. ort reads `ORT_DYLIB_PATH` lazily, on the first model load, so setting it
    this early is enough and costs a few `glob`s. A build without the `onnx` feature is a
    different case entirely, handled by `_MissingModel` in `load.py`.
    """
    global _resolved
    chosen = os.environ.get("ORT_DYLIB_PATH")
    if chosen:
        _resolved = (chosen, "ORT_DYLIB_PATH")
        return
    _resolved = find_runtime()
    if _resolved is not None:
        os.environ["ORT_DYLIB_PATH"] = _resolved[0]


def describe() -> str:
    """One line for `eventcv --version`: which library, and which of the five sources it came from."""
    found = _resolved if _resolved is not None else find_runtime()
    # A path set after import wins over what was resolved at import — it is what ort will open.
    current = os.environ.get("ORT_DYLIB_PATH")
    if current and (found is None or current != found[0]):
        found = (current, "ORT_DYLIB_PATH")
    if found is None:
        return "not found"
    path, origin = found
    return f"{path} ({origin})"
