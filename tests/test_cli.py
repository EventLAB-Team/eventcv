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
from unittest import mock
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


class PlayTests(unittest.TestCase):
    def test_play_without_a_file_opens_the_empty_player(self):
        with mock.patch("eventcv.play") as play:
            code, _, _ = _run("play", "--speed", "0.5", "--refresh-hz", "120")
        self.assertEqual(code, 0)
        play.assert_called_once()
        self.assertEqual(play.call_args.kwargs["speed"], 0.5)
        self.assertEqual(play.call_args.kwargs["refresh_hz"], 120.0)
        self.assertIsNone(play.call_args.kwargs["fps"])

    def test_play_file_preserves_source_and_view_options(self):
        reader = mock.Mock()
        with mock.patch("eventcv.open", return_value=reader) as open_reader:
            code, _, _ = _run(
                "play",
                str(EXAMPLE_NPZ),
                "--time-unit",
                "us",
                "--repr",
                "count",
                "--clim",
                "4",
                "--fps",
                "20",
            )
        self.assertEqual(code, 0)
        open_reader.assert_called_once_with(str(EXAMPLE_NPZ), time_unit="us")
        reader.play.assert_called_once()
        self.assertEqual(reader.play.call_args.kwargs["repr"], "count")
        self.assertEqual(reader.play.call_args.kwargs["clim"], 4.0)
        self.assertEqual(reader.play.call_args.kwargs["fps"], 20.0)

    def test_public_play_accepts_no_source(self):
        with mock.patch.object(eventcv._rust, "play_gui") as player:
            eventcv.play(speed=2.0)
        player.assert_called_once_with(speed=2.0)

    def test_refresh_rate_is_bounded_before_launch(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 240"):
            eventcv.play(refresh_hz=0)

    def test_legacy_fps_warns_before_launching(self):
        with self.assertWarns(DeprecationWarning), self.assertRaises(ValueError):
            eventcv.play(fps=30.0, max_frames=0)


class SimulateCommandTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
    def test_simulate_writes_events_from_a_video(self):
        import subprocess

        source = self.dir / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=32x24:rate=30:duration=1", "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )
        target = self.dir / "events.npz"
        code, out, _ = _run("simulate", str(source), str(target))
        self.assertEqual(code, 0)
        self.assertIn("events", out)
        events = eventcv.load(str(target))
        self.assertGreater(len(events), 0)
        self.assertEqual(events.sensor_size, (32, 24))

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
    def test_simulate_flags_reach_the_simulator(self):
        import subprocess

        source = self.dir / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=32x24:rate=30:duration=1", "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )
        # A high threshold must produce fewer events than a low one; if the flag were ignored the
        # two runs would be identical.
        counts = []
        for threshold in ("0.5", "0.1"):
            target = self.dir / f"t{threshold}.npz"
            self.assertEqual(
                _run("simulate", str(source), str(target), "--threshold", threshold, "-q")[0], 0
            )
            counts.append(len(eventcv.load(str(target))))
        self.assertLess(counts[0], counts[1])

    def test_a_missing_video_reports_an_error(self):
        code, _, err = _run("simulate", "/nonexistent/clip.mp4", str(self.dir / "e.npz"))
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
