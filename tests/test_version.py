"""Version reporting, and the guard that keeps the declarations from drifting apart.

``eventcv.__version__`` comes from the **compiled extension** (baked in from the crate version),
because that names the binary actually loaded. The version is declared once, in the root
``Cargo.toml`` ``[workspace.package]``: both crates inherit it and ``pyproject.toml`` is
``dynamic`` so maturin publishes the wheel under the same number. These tests keep it that way —
a hardcoded version creeping back into any of them would let a release ship two numbers.
"""

import re
import subprocess
import sys
import unittest
from pathlib import Path

import eventcv

ROOT = Path(__file__).parents[1]


def _declared(path, pattern=r'^version\s*=\s*"([^"]+)"'):
    """The first `version = "..."` in a TOML file, without needing a TOML parser on 3.9/3.10."""
    for line in Path(path).read_text().splitlines():
        found = re.match(pattern, line)
        if found:
            return found.group(1)
    raise AssertionError(f"no version declared in {path}")


class VersionTests(unittest.TestCase):
    def test_package_reports_a_version(self):
        self.assertRegex(eventcv.__version__, r"^\d+\.\d+\.\d+")

    def test_every_declaration_agrees(self):
        # One declaration (bumped by `scripts/bump_version.py`); everything else must inherit it.
        expected = _declared(ROOT / "Cargo.toml")
        self.assertEqual(eventcv.__version__, expected, "compiled extension vs root Cargo.toml")
        for crate in ("eventcv-py", "eventcv-core"):
            manifest = (ROOT / "crates" / crate / "Cargo.toml").read_text()
            self.assertRegex(manifest, r"(?m)^version\.workspace\s*=\s*true", crate)
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertRegex(pyproject, r'(?m)^dynamic\s*=\s*\[[^\]]*"version"')
        self.assertNotRegex(pyproject, r"(?m)^version\s*=")
        self.assertIn(f"eventcv {expected} (", (ROOT / "docs" / "cli.md").read_text())

    def test_features_list_what_was_built_in(self):
        # Published wheels carry all four; a plain `cargo build` of the bindings carries none.
        features = eventcv._rust.__features__
        self.assertIsInstance(features, list)
        self.assertLessEqual(set(features), {"hdf5", "camera", "onnx", "gpu"})
        # Whatever they say must match what the module actually exposes. `FrameSink` is the
        # HDF5-only class — `EventSink` writes every event format now, so it is always bound and
        # proves nothing about HDF5.
        self.assertEqual("hdf5" in features, eventcv.FrameSink is not None)
        self.assertEqual("camera" in features, eventcv.EventCamera is not None)
        # `Model` is always bound — to the real class, or to a stub that explains the rebuild —
        # so presence of the name proves nothing and the check has to be for the working one.
        self.assertEqual("onnx" in features, eventcv.Model is getattr(eventcv._rust, "Model", None))
        # `gpu` says the shaders were compiled in, not that a device exists, so the only thing that
        # must hold is the one direction: no feature, no adapter.
        if eventcv.gpu_available():
            self.assertIn("gpu", features)


class CommandLineTests(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "eventcv", *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_version_flag_prints_the_version(self):
        for flag in ("--version", "-V"):
            result = self._run(flag)
            self.assertEqual(result.returncode, 0, result.stderr)
            first = result.stdout.splitlines()[0]
            self.assertTrue(
                first.startswith(f"eventcv {eventcv.__version__}"),
                f"{flag} printed {first!r}",
            )

    def test_version_reports_the_build_and_interpreter(self):
        lines = self._run("--version").stdout.splitlines()
        for feature in eventcv._rust.__features__:
            self.assertIn(feature, lines[0], "the build's features belong on the version line")
        self.assertTrue(lines[-1].startswith("Python "), lines[-1])

    @unittest.skipUnless(
        "onnx" in eventcv._rust.__features__, "needs a build with the onnx feature"
    )
    def test_version_reports_which_onnx_runtime_was_loaded(self):
        # Which library `Model` opened, and where it came from, is decided at run time — the one
        # thing about a model bug report that a build-time feature list cannot answer.
        lines = self._run("--version").stdout.splitlines()
        self.assertTrue(
            any(line.startswith("ONNX Runtime") for line in lines[1:-1]),
            f"no ONNX Runtime line in {lines}",
        )

    def test_bare_invocation_prints_help_and_succeeds(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--version", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
