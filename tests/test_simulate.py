"""Frames → events simulation.

The assertions here are analytic and statistical rather than shape checks: an ideal pixel's event
count is arithmetic given the contrast, each noise source has an expected rate, and the timestamps
have properties (ascending, inside the interval, spread across it) that a simulator collapsing
everything to the frame time would fail.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

HAVE_FFMPEG = shutil.which("ffmpeg") is not None

# Every noise source off, so an assertion about counts is about the pixel model alone.
IDEAL = dict(
    sigma_thres=0.0,
    leak_rate_hz=0.0,
    shot_noise_rate_hz=0.0,
    cutoff_hz=0.0,
    refractory_us=0,
    upsample="off",
)


def _moving_edge(frames=20, size=32, start=6):
    """An edge sweeping right one pixel per frame — a known amount of new contrast each step."""
    video = np.zeros((frames, size, size), dtype=np.uint8)
    for i in range(frames):
        video[i, :, : start + i] = 200
    return video


def _make_video(path, size="64x48", rate=30, duration=1):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc=size={size}:rate={rate}:duration={duration}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


class FromFramesTests(unittest.TestCase):
    def test_a_static_scene_is_silent_when_ideal(self):
        still = np.full((10, 16, 16), 128, dtype=np.uint8)
        self.assertEqual(len(eventcv.simulate(still, fps=1000, **IDEAL)), 0)

    def test_brightening_emits_on_and_darkening_emits_off(self):
        up = np.stack([np.full((8, 8), 40, np.uint8), np.full((8, 8), 200, np.uint8)])
        down = up[::-1].copy()
        self.assertTrue(np.all(eventcv.simulate(up, fps=100, **IDEAL).numpy()[:, 3] == 1))
        self.assertTrue(np.all(eventcv.simulate(down, fps=100, **IDEAL).numpy()[:, 3] == 0))

    def test_halving_the_threshold_doubles_the_events(self):
        # The defining property of a contrast-threshold sensor. Only exact with the refractory
        # period disabled — otherwise the burst is capped and the relationship saturates.
        frames = _moving_edge()
        counts = {
            threshold: len(
                eventcv.simulate(
                    frames, fps=1000, pos_thres=threshold, neg_thres=threshold, **IDEAL
                )
            )
            for threshold in (0.2, 0.1)
        }
        self.assertAlmostEqual(counts[0.1] / counts[0.2], 2.0, delta=0.05)

    def test_the_refractory_period_caps_a_burst(self):
        frames = _moving_edge()
        free = len(eventcv.simulate(frames, fps=1000, pos_thres=0.05, **IDEAL))
        limited_args = {**IDEAL, "refractory_us": 200}
        limited = len(eventcv.simulate(frames, fps=1000, pos_thres=0.05, **limited_args))
        self.assertLess(limited, free)
        self.assertGreater(limited, 0)

    def test_timestamps_are_ascending_and_within_the_recording(self):
        frames = _moving_edge()
        events = eventcv.simulate(frames, fps=1000, **IDEAL).numpy()
        t = events[:, 2]
        self.assertTrue(np.all(np.diff(t) >= 0), "events must be globally time-ordered")
        self.assertGreaterEqual(t.min(), 0, "negative timestamps corrupt time surfaces")
        self.assertLessEqual(t.max(), 20 * 1000)

    def test_timestamps_are_spread_not_collapsed_to_frame_times(self):
        # The flaw in the only comparable implementation: several crossings at one pixel all
        # stamped with the frame time. Here they must land between frames.
        frames = np.stack([np.full((4, 4), 20, np.uint8), np.full((4, 4), 220, np.uint8)])
        t = eventcv.simulate(frames, fps=100, pos_thres=0.1, **IDEAL).numpy()[:, 2]
        self.assertGreater(len(np.unique(t)), 1, "all timestamps identical — no interpolation")
        # 100 fps means a 10,000 us interval; the crossings must be distributed inside it.
        self.assertTrue(np.all((t >= 0) & (t <= 10_000)))

    def test_rgb_and_greyscale_agree_on_a_grey_scene(self):
        grey = _moving_edge(frames=6, size=16)
        rgb = np.repeat(grey[..., None], 3, axis=3)
        self.assertEqual(
            len(eventcv.simulate(grey, fps=500, **IDEAL)),
            len(eventcv.simulate(rgb, fps=500, **IDEAL)),
        )

    def test_float_frames_are_accepted(self):
        frames = _moving_edge(frames=6, size=16)
        as_float = frames.astype("float32") / 255.0
        self.assertEqual(
            len(eventcv.simulate(frames, fps=500, **IDEAL)),
            len(eventcv.simulate(as_float, fps=500, **IDEAL)),
        )

    def test_max_frames_truncates(self):
        frames = _moving_edge(frames=20)
        short = eventcv.simulate(frames, fps=1000, max_frames=5, **IDEAL)
        full = eventcv.simulate(frames, fps=1000, **IDEAL)
        self.assertLess(len(short), len(full))


class NoiseModelTests(unittest.TestCase):
    """Each source in isolation, with the scene static so motion cannot contribute."""

    def _static(self, frames=11, size=16, value=128):
        return np.full((frames, size, size), value, dtype=np.uint8)

    def test_leak_fires_at_about_its_rate(self):
        # 10 frames of 100 ms = 1 s; at 10 Hz over 256 pixels, expect ~2560 ON events.
        events = eventcv.simulate(
            self._static(), fps=10, leak_rate_hz=10.0,
            shot_noise_rate_hz=0.0, sigma_thres=0.0, cutoff_hz=0.0,
            refractory_us=0, upsample="off",
        )
        self.assertTrue(np.all(events.numpy()[:, 3] == 1), "leak emits ON events")
        expected = 10.0 * 16 * 16
        self.assertGreater(len(events), expected * 0.5)
        self.assertLess(len(events), expected * 1.5)

    def test_shot_noise_scales_with_its_rate(self):
        def count(rate):
            return len(
                eventcv.simulate(
                    self._static(), fps=10, shot_noise_rate_hz=rate,
                    leak_rate_hz=0.0, cutoff_hz=0.0, upsample="off",
                )
            )

        self.assertEqual(count(0.0), 0, "no noise configured means no events at all")
        low, high = count(5.0), count(50.0)
        self.assertGreater(low, 0)
        self.assertGreater(high, low * 3)

    def test_shot_noise_is_quieter_in_bright_pixels(self):
        # v2e's intensity dependence: the brightest pixels are ~4x quieter than the darkest.
        # The rate has to stay well below one event per interval per polarity, or every pixel
        # saturates at its one-event ceiling and the dependence is invisible.
        def count(value):
            return len(
                eventcv.simulate(
                    self._static(value=value), fps=10, shot_noise_rate_hz=2.0,
                    leak_rate_hz=0.0, cutoff_hz=0.0, upsample="off",
                )
            )

        dark, bright = count(5), count(250)
        self.assertGreater(dark, bright)
        self.assertGreater(dark / max(bright, 1), 2.0, "the dependence should be substantial")

    def test_threshold_mismatch_desynchronises_pixels(self):
        def distinct_times(sigma):
            frames = np.stack(
                [np.full((16, 16), 80, np.uint8), np.full((16, 16), 120, np.uint8)]
            )
            events = eventcv.simulate(
                frames, fps=100, sigma_thres=sigma, leak_rate_hz=0.0,
                shot_noise_rate_hz=0.0, cutoff_hz=0.0, refractory_us=0, upsample="off",
            )
            return len(np.unique(events.numpy()[:, 2]))

        self.assertLess(distinct_times(0.0), distinct_times(0.05))


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_events(self):
        frames = _moving_edge(frames=8, size=16)
        first = eventcv.simulate(frames, fps=500, seed=7)
        second = eventcv.simulate(frames, fps=500, seed=7)
        np.testing.assert_array_equal(first.numpy(), second.numpy())

    def test_different_seeds_differ(self):
        frames = _moving_edge(frames=8, size=16)
        self.assertFalse(
            np.array_equal(
                eventcv.simulate(frames, fps=500, seed=1).numpy(),
                eventcv.simulate(frames, fps=500, seed=2).numpy(),
            )
        )


class UpsamplingTests(unittest.TestCase):
    def test_modes_are_accepted(self):
        frames = _moving_edge(frames=6, size=16)
        for mode in (None, "adaptive", "off", "4"):
            with self.subTest(upsample=mode):
                self.assertGreater(len(eventcv.simulate(frames, fps=500, upsample=mode)), 0)

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            eventcv.simulate(_moving_edge(frames=3, size=8), fps=100, upsample="sometimes")

    def test_subdividing_does_not_coarsen_timing(self):
        frames = np.stack([np.full((8, 8), 20, np.uint8), np.full((8, 8), 220, np.uint8)])
        common = dict(fps=100, pos_thres=0.1, sigma_thres=0.0, leak_rate_hz=0.0,
                      shot_noise_rate_hz=0.0, cutoff_hz=0.0, refractory_us=0)
        off = len(np.unique(eventcv.simulate(frames, upsample="off", **common).numpy()[:, 2]))
        adaptive = len(
            np.unique(eventcv.simulate(frames, upsample="adaptive", **common).numpy()[:, 2])
        )
        self.assertGreaterEqual(adaptive, off)


class ErrorTests(unittest.TestCase):
    def test_frames_need_an_explicit_fps(self):
        with self.assertRaises(ValueError) as caught:
            eventcv.simulate(_moving_edge(frames=3, size=8))
        self.assertIn("fps", str(caught.exception))

    def test_a_bad_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            eventcv.simulate(np.zeros((8, 8), dtype=np.uint8), fps=100)

    def test_a_missing_video_is_reported(self):
        with self.assertRaises((OSError, ValueError)):
            eventcv.simulate("/nonexistent/clip.mp4")


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
class FromVideoTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.video = _make_video(self.dir / "clip.mp4")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_moving_test_pattern_generates_ordered_events(self):
        events = eventcv.simulate(str(self.video))
        self.assertGreater(len(events), 0)
        self.assertEqual(events.sensor_size, (64, 48))
        t = events.numpy()[:, 2]
        self.assertTrue(np.all(np.diff(t) >= 0))
        # One second of video, so timestamps should span roughly that.
        self.assertGreater(t.max(), 900_000)

    def test_scale_changes_the_sensor_size(self):
        events = eventcv.simulate(str(self.video), scale=(32, 24))
        self.assertEqual(events.sensor_size, (32, 24))

    def test_max_frames_shortens_the_recording(self):
        short = eventcv.simulate(str(self.video), max_frames=5)
        self.assertLess(short.numpy()[:, 2].max(), 300_000)

    def test_simulated_events_round_trip_through_a_reader(self):
        # The output has to be a first-class stream: saveable, re-openable, and sliceable into
        # representations like any recorded one.
        events = eventcv.simulate(str(self.video))
        path = self.dir / "events.npz"
        eventcv.save(events, str(path))
        reader = eventcv.open(str(path), dt_ms=33).with_repr("voxel", bins=5)
        self.assertGreater(len(reader), 0)
        self.assertEqual(np.asarray(reader[0]).shape, (5, 48, 64))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class SimulateToFileTests(unittest.TestCase):
    """`out=` — writing the events as they are produced instead of accumulating them.

    A realistic sensor emits far more than fits in memory at any real resolution (a second of
    1080p is ~167 M events, several GB as an in-memory stream), so streaming to disk is what makes
    the simulator usable at all above toy sizes.
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="eventcv-simout-"))
        cls.video = cls.dir / "clip.mp4"
        _make_video(cls.video)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_streaming_to_hdf5_matches_the_in_memory_run(self):
        if not _hdf5_supported():
            self.skipTest("built without the hdf5 feature")
        path = self.dir / "events.h5"
        result = eventcv.simulate(str(self.video), out=str(path))
        self.assertEqual(result.path, str(path))
        self.assertGreater(result.events, 0)
        self.assertEqual(len(result), result.events)

        # Same seed, same events — the file is not a lossy shortcut, it is the same simulation.
        memory = eventcv.simulate(str(self.video))
        self.assertEqual(result.events, len(memory))
        reloaded = eventcv.load(str(path))
        np.testing.assert_array_equal(reloaded.numpy(), memory.numpy())

    def test_streaming_to_a_buffered_format_still_writes(self):
        # `.npz` cannot be appended to, so it is buffered and written once — the caller should not
        # have to know which formats stream.
        path = self.dir / "events.npz"
        result = eventcv.simulate(str(self.video), out=str(path), max_frames=5)
        self.assertTrue(path.exists())
        self.assertEqual(len(eventcv.load(str(path))), result.events)

    def test_compression_shrinks_the_file_and_round_trips(self):
        if not _hdf5_supported():
            self.skipTest("built without the hdf5 feature")
        packed, plain = self.dir / "packed.h5", self.dir / "plain.h5"
        eventcv.simulate(str(self.video), out=str(packed))
        eventcv.simulate(str(self.video), out=str(plain), compression=False)
        self.assertLess(packed.stat().st_size, plain.stat().st_size)
        np.testing.assert_array_equal(
            eventcv.load(str(packed)).numpy(), eventcv.load(str(plain)).numpy()
        )

    def test_max_upsample_caps_the_work(self):
        # Capping the subdivision must not change what the simulator *is*, only how finely it
        # resolves time — so the events stay well-formed and ordered.
        events = eventcv.simulate(str(self.video), max_upsample=1, max_frames=5)
        self.assertGreater(len(events), 0)
        self.assertTrue(np.all(np.diff(events.numpy()[:, 2]) >= 0))


def _hdf5_supported() -> bool:
    """True if the extension was built with the `hdf5` feature."""
    try:
        eventcv.load("___eventcv_no_such_file___.h5", sensor_size=(1, 1))
    except FileNotFoundError:
        return True
    except Exception:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
