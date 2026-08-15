"""The `eventcv` command line.

Every command is a thin wrapper over the library, so these tests check the wrapping rather than
the underlying behaviour: that arguments reach the right call, that output lands where it was
asked to, and that a failure prints an actionable line instead of a traceback.
"""

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv
from eventcv.__main__ import main

EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"


def _run(*argv):
    """Runs the CLI, returning `(exit_code, stdout, stderr)` instead of exiting the process."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class VersionTests(unittest.TestCase):
    def test_version_reports_version_and_features(self):
        code, out, _ = _run("--version")
        self.assertEqual(code, 0)
        self.assertIn(eventcv.__version__, out)
        # The build's optional features are the second thing a bug report needs.
        self.assertIn("(", out)

    def test_no_arguments_prints_help_rather_than_failing(self):
        code, out, _ = _run()
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)


class InfoTests(unittest.TestCase):
    def test_info_matches_the_library(self):
        code, out, _ = _run("info", str(EXAMPLE_NPZ))
        self.assertEqual(code, 0)
        stream = eventcv.load(str(EXAMPLE_NPZ))
        width, height = stream.sensor_size
        self.assertIn(f"{width} x {height}", out)
        self.assertIn(f"{len(stream):,}", out)

    def test_rate_bin_adds_the_peak(self):
        code, out, _ = _run("info", str(EXAMPLE_NPZ), "--rate-bin-ms", "5")
        self.assertEqual(code, 0)
        self.assertIn("peak rate", out)

    def test_missing_file_names_the_path_and_exits_nonzero(self):
        code, _, err = _run("info", "/nonexistent/nope.npz")
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertIn("/nonexistent/nope.npz", err)
        self.assertNotIn("Traceback", err)


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_npz_round_trips_exactly(self):
        target = self.dir / "out.npz"
        code, out, _ = _run("convert", str(EXAMPLE_NPZ), str(target))
        self.assertEqual(code, 0)
        self.assertIn("wrote", out)
        np.testing.assert_array_equal(
            eventcv.load(str(target)).numpy(), eventcv.load(str(EXAMPLE_NPZ)).numpy()
        )

    def test_text_round_trips_when_the_unit_is_given(self):
        # A text file carries no unit, so the writer's µs have to be declared on the way back in.
        target = self.dir / "out.txt"
        self.assertEqual(_run("convert", str(EXAMPLE_NPZ), str(target), "-q")[0], 0)
        np.testing.assert_array_equal(
            eventcv.load(str(target), time_unit="us").numpy(),
            eventcv.load(str(EXAMPLE_NPZ)).numpy(),
        )

    def test_quiet_suppresses_the_summary(self):
        code, out, _ = _run("convert", str(EXAMPLE_NPZ), str(self.dir / "q.npz"), "-q")
        self.assertEqual(code, 0)
        self.assertEqual(out, "")

    def test_an_unwritable_format_reports_an_error(self):
        code, _, err = _run("convert", str(EXAMPLE_NPZ), str(self.dir / "out.bogus"))
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertNotIn("Traceback", err)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_render_writes_a_gif(self):
        target = self.dir / "out.gif"
        code, out, _ = _run(
            "render", str(EXAMPLE_NPZ), str(target), "--dt-ms", "5", "--fps", "10"
        )
        self.assertEqual(code, 0)
        self.assertIn("frames", out)
        self.assertEqual(target.read_bytes()[:6], b"GIF89a")

    def test_max_frames_is_honoured(self):
        target = self.dir / "short.gif"
        code, out, _ = _run(
            "render", str(EXAMPLE_NPZ), str(target), "--dt-ms", "5", "--max-frames", "2"
        )
        self.assertEqual(code, 0)
        self.assertIn("(2 frames", out)

    def test_representation_flag_is_forwarded(self):
        target = self.dir / "count.gif"
        code, _, _ = _run(
            "render", str(EXAMPLE_NPZ), str(target), "--dt-ms", "5", "--repr", "count"
        )
        self.assertEqual(code, 0)
        self.assertTrue(target.stat().st_size > 0)

    def test_an_unknown_extension_reports_an_error(self):
        code, _, err = _run("render", str(EXAMPLE_NPZ), str(self.dir / "out.avi"))
        self.assertEqual(code, 1)
        self.assertIn(".gif", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
