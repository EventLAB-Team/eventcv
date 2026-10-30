#!/usr/bin/env python3
"""Bump the eventcv release version in the working tree.

The version lives once, in the root `Cargo.toml` (`[workspace.package]`): both crates inherit it,
maturin reads it for the wheel, and `eventcv.__version__` / the Sphinx docs follow from that. This
script rewrites that line, refreshes `Cargo.lock` for the two eventcv crates, updates the sample
`--version` output in `docs/cli.md`, and prints the diff stat. It never commits.

    python scripts/bump_version.py 1.0.11
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKSPACE_VERSION = re.compile(r'(\[workspace\.package\]\s*version = ")([^"]+)(")')


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        print(__doc__.strip().splitlines()[-1].strip(), file=sys.stderr)
        return 2
    new = sys.argv[1]

    cargo = ROOT / "Cargo.toml"
    text = cargo.read_text()
    match = WORKSPACE_VERSION.search(text)
    if match is None:
        print("no [workspace.package] version in Cargo.toml", file=sys.stderr)
        return 1
    old = match.group(2)
    if old == new:
        print(f"already at {new}")
        return 0
    cargo.write_text(WORKSPACE_VERSION.sub(rf"\g<1>{new}\g<3>", text, count=1))

    cli_doc = ROOT / "docs" / "cli.md"
    cli_doc.write_text(cli_doc.read_text().replace(f"eventcv {old} (", f"eventcv {new} ("))

    subprocess.run(
        ["cargo", "update", "--offline", "-p", "eventcv-core", "-p", "eventcv-py"],
        cwd=ROOT,
        check=True,
    )
    print(f"\n{old} -> {new}")
    subprocess.run(["git", "diff", "--stat"], cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
