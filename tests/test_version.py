"""Version reporting, and the guard that keeps the declarations from drifting apart.

``eventcv.__version__`` comes from the **compiled extension** (baked in from
``crates/eventcv-py/Cargo.toml``), because that names the binary actually loaded. Nothing in the
build makes that agree with ``pyproject.toml``'s version, which is what a wheel is published under
— so these tests are what keeps a release from shipping two different numbers.
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
        # The four places a release has to be bumped. They are separate files with no build-time
        # link, so drift is silent until a wheel is published under the wrong number.
        expected = _declared(ROOT / "pyproject.toml")
        self.assertEqual(eventcv.__version__, expected, "compiled extension vs pyproject.toml")
        self.assertEqual(_declared(ROOT / "crates" / "eventcv-py" / "Cargo.toml"), expected)
        self.assertEqual(_declared(ROOT / "crates" / "eventcv-core" / "Cargo.toml"), expected)

    def test_features_list_what_was_built_in(self):
        # Published wheels carry all three; a plain `cargo build` of the bindings carries none.
        features = eventcv._rust.__features__
        self.assertIsInstance(features, list)
        self.assertLessEqual(set(features), {"hdf5", "camera", "onnx"})
        # Whatever they say must match what the module actually exposes.
        self.assertEqual("hdf5" in features, eventcv.EventSink is not None)
        self.assertEqual("camera" in features, eventcv.EventCamera is not None)
        # `Model` is always bound — to the real class, or to a stub that explains the rebuild —
        # so presence of the name proves nothing and the check has to be for the working one.
        self.assertEqual("onnx" in features, eventcv.Model is getattr(eventcv._rust, "Model", None))


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
        self.assertTrue(lines[1].startswith("Python "), lines[1])

    def test_bare_invocation_prints_help_and_succeeds(self):
        result = self._run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--version", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
