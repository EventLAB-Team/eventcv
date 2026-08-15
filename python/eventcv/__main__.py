"""Command-line entry point: ``eventcv --version`` (or ``python -m eventcv --version``).

Deliberately tiny — EventCV is a library, and this exists so the version and build can be read off
an installed copy without opening a REPL. That matters for bug reports: which version is installed,
and whether it has HDF5 reading and USB camera streaming compiled in, are the first two questions
an issue raises.
"""

from __future__ import annotations

import argparse
import platform

from . import __version__, _rust


def version_text() -> str:
    """``eventcv 1.0.5 (hdf5, camera)`` plus the interpreter it is installed under."""
    features = ", ".join(getattr(_rust, "__features__", ())) or "no optional features"
    return (
        f"eventcv {__version__} ({features})\n"
        f"Python {platform.python_version()} on {platform.platform()}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eventcv",
        description="EventCV — OpenCV for event-based vision. Import `eventcv` to use it.",
    )
    # Printed by hand rather than with argparse's `version` action, which re-wraps its text to the
    # terminal width and would run the two lines together.
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="print the installed version, the features it was built with, and the interpreter",
    )
    if parser.parse_args(argv).version:
        print(version_text())
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
