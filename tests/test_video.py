"""Animated export and temporal analytics.

The files written here are checked by their container signatures rather than by decoding them —
the point is that a real, complete GIF/APNG/MP4 arrived on disk, which is exactly what a
half-finished encoder or an unflushed writer gets wrong.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"


def _reader(**kwargs):
    kwargs.setdefault("dt_ms", 5)
    kwargs.setdefault("repr", "count")
    return eventcv.open(str(EXAMPLE_NPZ), **kwargs)


class EventRateTests(unittest.TestCase):
    def setUp(self):
        self.stream = eventcv.load(str(EXAMPLE_NPZ))

    def test_counts_account_for_every_event(self):
        rate = self.stream.event_rate(bin_ms=5)
        self.assertEqual(int(rate["count"].sum()), len(self.stream))

    def test_polarities_sum_to_the_total(self):
        rate = self.stream.event_rate(bin_ms=5)
        np.testing.assert_array_equal(
            rate["positive"] + rate["negative"], rate["count"]
        )

    def test_rate_is_counts_over_the_bin_width(self):
        rate = self.stream.event_rate(bin_ms=5)
        np.testing.assert_allclose(
            rate["rate"], rate["count"] / (rate["bin_us"] / 1e6), rtol=1e-9
        )

    def test_bins_are_evenly_spaced_and_ascending(self):
        rate = self.stream.event_rate(bin_ms=5)
        gaps = np.diff(rate["t"])
        self.assertTrue(np.all(gaps == rate["bin_us"]))

    def test_finer_bins_give_more_of_them(self):
        coarse = self.stream.event_rate(bin_ms=10)
        fine = self.stream.event_rate(bin_ms=1)
        self.assertGreater(len(fine["t"]), len(coarse["t"]))
        # Total events are conserved however the axis is chopped up.
        self.assertEqual(int(fine["count"].sum()), int(coarse["count"].sum()))

    def test_every_time_unit_agrees(self):
        reference = self.stream.event_rate(bin_us=5000.0)
        for kwargs in ({"bin_ms": 5.0}, {"bin_s": 0.005}, {"bin_ns": 5_000_000.0}):
            with self.subTest(**kwargs):
                np.testing.assert_array_equal(
                    self.stream.event_rate(**kwargs)["count"], reference["count"]
                )

    def test_a_non_positive_bin_is_rejected(self):
        with self.assertRaises(ValueError):
            self.stream.event_rate(bin_ms=0)


class SaveVideoTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_gif_is_written_and_complete(self):
        path = self.dir / "out.gif"
        frames = _reader().save_video(str(path), fps=10)
        data = path.read_bytes()
        self.assertEqual(frames, len(_reader()))
        self.assertEqual(data[:6], b"GIF89a")
        self.assertEqual(data[-1], 0x3B)  # trailer — the file was finished, not just flushed

    def test_apng_is_animated_not_a_still(self):
        path = self.dir / "out.apng"
        _reader().save_video(str(path), fps=10)
        data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        # acTL is what separates an animated PNG from a still one that silently kept one frame.
        self.assertIn(b"acTL", data)
        self.assertIn(b"fcTL", data)

    def test_png_extension_also_means_animated(self):
        path = self.dir / "out.png"
        _reader().save_video(str(path), fps=10)
        self.assertIn(b"acTL", path.read_bytes())

    def test_max_frames_limits_the_render(self):
        path = self.dir / "short.gif"
        self.assertEqual(_reader().save_video(str(path), max_frames=3), 3)

    def test_a_representation_is_required(self):
        with self.assertRaises(ValueError) as caught:
            eventcv.open(str(EXAMPLE_NPZ), dt_ms=5).save_video(str(self.dir / "x.gif"))
        self.assertIn("representation", str(caught.exception))

    def test_an_unknown_extension_names_the_supported_ones(self):
        with self.assertRaises(ValueError) as caught:
            _reader().save_video(str(self.dir / "out.avi"))
        message = str(caught.exception)
        self.assertIn(".gif", message)

    def test_explicit_clim_is_accepted_and_deterministic(self):
        first, second = self.dir / "a.gif", self.dir / "b.gif"
        _reader().save_video(str(first), clim=50.0)
        _reader().save_video(str(second), clim=50.0)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_clim_changes_the_output(self):
        dim, bright = self.dir / "dim.gif", self.dir / "bright.gif"
        _reader().save_video(str(dim), clim=1000.0)
        _reader().save_video(str(bright), clim=2.0)
        self.assertNotEqual(dim.read_bytes(), bright.read_bytes())

    def test_augmentations_apply_to_the_export(self):
        plain, augmented = self.dir / "plain.gif", self.dir / "aug.gif"
        _reader().save_video(str(plain))
        eventcv.open(str(EXAMPLE_NPZ), dt_ms=5).event_drop(0.8, seed=1).with_repr(
            "count"
        ).save_video(str(augmented))
        self.assertNotEqual(plain.read_bytes(), augmented.read_bytes())

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
    def test_mp4_is_written_when_ffmpeg_is_available(self):
        path = self.dir / "out.mp4"
        frames = _reader().save_video(str(path), fps=10)
        self.assertEqual(frames, len(_reader()))
        self.assertGreater(path.stat().st_size, 0)
        # `ftyp` is the first box of any ISO base-media file; its absence means ffmpeg never
        # finished writing the header, which is what a missing stdin close looks like.
        self.assertIn(b"ftyp", path.read_bytes()[:64])

    @unittest.skipIf(shutil.which("ffmpeg") is not None, "ffmpeg is installed")
    def test_missing_ffmpeg_explains_how_to_fix_it(self):
        with self.assertRaises(FileNotFoundError) as caught:
            _reader().save_video(str(self.dir / "out.mp4"))
        message = str(caught.exception)
        self.assertIn("ffmpeg", message)
        self.assertIn(".gif", message)  # points at the dependency-free alternative


if __name__ == "__main__":
    unittest.main()
