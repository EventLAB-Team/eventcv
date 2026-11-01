
#!/usr/bin/env python3
"""Validate a proposed EventCV release version.

Usage:
    python scripts/validate_version.py 1.0.11

Checks:
    - Valid semantic version format (MAJOR.MINOR.PATCH)
    - Proposed version is newer than current version
    - Git tag does not already exist

Exits with code 0 on success, 1 on failure.
"""

import argparse
import pathlib
import re
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent

VERSION_PATTERN = re.compile(
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
)


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse and validate an X.Y.Z version."""
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            f"Invalid version '{version}'. "
            "Expected MAJOR.MINOR.PATCH (e.g. 1.0.11)."
        )

    return tuple(map(int, version.split(".")))


def get_current_version() -> str:
    """Read version from root Cargo.toml."""
    with (ROOT / "Cargo.toml").open("rb") as f:
        cargo = tomllib.load(f)

    return cargo["workspace"]["package"]["version"]


def tag_exists(version: str) -> bool:
    """Check whether the release tag already exists."""
    tag = f"refs/tags/v{version}"

    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", tag],
        cwd=ROOT,
        check=False,
    )

    if result.returncode not in (0, 1):
        raise RuntimeError("Failed to check Git tags")

    return result.returncode == 0


def validate_version(proposed: str) -> None:
    """Run all release version checks."""
    current = get_current_version()

    print(f"Current version:  {current}")
    print(f"Proposed version: {proposed}")

    new = parse_version(proposed)
    old = parse_version(current)

    # Check version ordering
    if new <= old:
        raise ValueError(
            f"Proposed version {proposed} must be "
            f"newer than current version {current}."
        )

    # Check existing tags
    if tag_exists(proposed):
        raise ValueError(
            f"Git tag v{proposed} already exists."
        )

    print(f"Validation passed: {current} -> {proposed}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a proposed EventCV release version."
    )

    parser.add_argument(
        "version",
        help="Proposed release version (e.g. 1.0.11)",
    )

    args = parser.parse_args()

    try:
        validate_version(args.version)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
