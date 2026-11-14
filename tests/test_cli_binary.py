"""The standalone `eventcv` binary against the Python command line it ports.

The two must stay interchangeable for the commands they share — same output, same files — so
scripts and docs work against either. Needs a built binary, so it runs only when `EVENTCV_BIN`
points at one: `cargo build -p eventcv-cli --features hdf5,camera`, then
`EVENTCV_BIN=target/debug/eventcv pytest tests/test_cli_binary.py`.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"
BINARY = os.environ.get("EVENTCV_BIN")


@unittest.skipUnless(BINARY, "set EVENTCV_BIN to a built eventcv binary")
class ParityTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _both(self, *argv):
        """Runs `argv` through each CLI, with `{out}` naming a separate output file per CLI."""
        results = []
        for name, command in (("rust", [BINARY]), ("python", [sys.executable, "-m", "eventcv"])):
            out = str(self.dir / name)
            args = [arg.replace("{out}", out) for arg in argv]
            done = subprocess.run(command + args, capture_output=True, text=True)
            written = [Path(new) for new, old in zip(args, argv) if new != old]
            results.append(
                (done.returncode, done.stdout.replace(out, "{out}"), done.stderr, written)
            )
        return results

    def assert_same(self, *argv):
        rust, python = self._both(*argv)
        self.assertEqual(rust[:3], python[:3], argv)
        for ours, theirs in zip(rust[3], python[3]):
            self.assertEqual(ours.read_bytes(), theirs.read_bytes(), argv)

    def test_info(self):
        self.assert_same("info", str(EXAMPLE_NPZ))
        self.assert_same("info", str(EXAMPLE_NPZ), "--rate-bin-ms", "5")

    def test_missing_file(self):
        self.assert_same("info", "/nonexistent/nope.npz")

    def test_convert(self):
        for extension in ("npz", "txt"):
            self.assert_same("convert", str(EXAMPLE_NPZ), f"{{out}}.{extension}")

    def test_render(self):
        self.assert_same("render", str(EXAMPLE_NPZ), "{out}.gif", "--dt-ms", "10")
        self.assert_same(
            "render", str(EXAMPLE_NPZ), "{out}.gif", "--repr", "tencode", "--dt-ms", "20"
        )
        self.assert_same(
            "render",
            str(EXAMPLE_NPZ),
            "{out}.gif",
            "--repr",
            "count",
            "--clim",
            "0",
            "--fps",
            "12.5",
        )


if __name__ == "__main__":
    unittest.main()
