"""Event-domain transforms (Workstream B), exercised across several file formats.

The same invariant suite runs on streams loaded from the committed N-ImageNet ``.npz``
fixture, a synthetic ``.txt`` recording, and (when present) the real ``.aedat`` file — so
the transforms are validated against more than one reader/coordinate convention.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"
AEDAT2 = "data/development/+0+2+0_l_qry.aedat"


def _synthetic_txt() -> str:
    """A small EV-IMO-style `t x y p` recording on a 64×48 sensor."""
    path = Path(tempfile.mkdtemp()) / "events.txt"
    lines = [f"{i * 100} {i % 64} {i % 48} {i % 2}" for i in range(500)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _load_cases():
    """(_name, stream) pairs across formats; the .aedat case is included only if present."""
    cases = [
        ("npz", eventcv.load(str(EXAMPLE_NPZ))),
        ("txt", eventcv.load(_synthetic_txt(), time_unit="us")),
    ]
    if os.path.exists(AEDAT2):
        cases.append(("aedat", eventcv.load(AEDAT2, max_events=200_000)))
    return cases


class TransformInvariantTests(unittest.TestCase):
    """Format-agnostic invariants — run once per loaded format."""

    def test_invariants_across_formats(self):
        for name, stream in _load_cases():
            with self.subTest(format=name):
                self._check(stream)

    def _check(self, stream):
        width, height = stream.sensor_size
        n = len(stream)
        self.assertGreater(n, 0)

        # Flip is its own inverse (coords restored), and sensor size is preserved.
        flipped = stream.flip_x()
        self.assertEqual(flipped.sensor_size, (width, height))
        np.testing.assert_array_equal(flipped.flip_x().numpy(), stream.numpy())
        np.testing.assert_array_equal(stream.flip_y().flip_y().numpy(), stream.numpy())

        # rotate90 four times is the identity; one turn swaps the sensor dims.
        self.assertEqual(stream.rotate90(1).sensor_size, (height, width))
        np.testing.assert_array_equal(
            stream.rotate90(1).rotate90(1).rotate90(1).rotate90(1).numpy(), stream.numpy()
        )

        # Polarity split is exhaustive and disjoint (count conserved).
        on, off = stream.filter_polarity(True), stream.filter_polarity(False)
        self.assertEqual(len(on) + len(off), n)

        # Downscale rebins losslessly into the new grid.
        half = stream.resize(max(1, width // 2), max(1, height // 2))
        self.assertEqual(len(half), n)
        coords = half.numpy()
        self.assertTrue((coords[:, 0] < half.sensor_size[0]).all())
        self.assertTrue((coords[:, 1] < half.sensor_size[1]).all())

        # normalize_time anchors the earliest event at zero.
        self.assertEqual(int(stream.normalize_time().numpy()[:, 2].min()), 0)

        # decimate(2) keeps ~half the events.
        self.assertEqual(len(stream.decimate(2)), (n + 1) // 2)


class TransformChainingTests(unittest.TestCase):
    """Chaining + sensor bookkeeping on the committed npz fixture."""

    def setUp(self):
        self.stream = eventcv.load(str(EXAMPLE_NPZ))  # 640×480 N-ImageNet

    def test_pipeline_chains_and_tracks_sensor_size(self):
        out = (
            self.stream.crop(0, 0, 320, 240)
            .flip_x()
            .resize(160, 120)
            .filter_polarity(1)
        )
        self.assertIsInstance(out, eventcv.EventStream)
        self.assertEqual(out.sensor_size, (160, 120))
        coords = out.numpy()
        self.assertTrue((coords[:, 0] < 160).all() and (coords[:, 1] < 120).all())
        self.assertTrue(set(np.unique(coords[:, 3])).issubset({1}))
        # Each narrowing step only ever drops events.
        self.assertLessEqual(len(out), len(self.stream))

    def test_transform_feeds_into_a_representation(self):
        frame = self.stream.flip_x().voxel(bins=4)
        self.assertIsInstance(frame, eventcv.EventFrame)
        # voxel shape is (2*bins, H, W); flip_x keeps the sensor dims.
        self.assertEqual(frame.shape[1:], (480, 640))

    def test_mask_selects_a_region(self):
        width, height = self.stream.sensor_size
        keep = np.zeros((height, width), dtype=bool)
        keep[: height // 2, : width // 2] = True  # top-left quadrant
        masked = self.stream.mask(keep).numpy()
        if len(masked):
            self.assertTrue((masked[:, 0] < width // 2).all())
            self.assertTrue((masked[:, 1] < height // 2).all())

    def test_concat_appends_and_takes_max_sensor(self):
        a = self.stream.crop(0, 0, 100, 100)
        b = self.stream.crop(0, 0, 200, 150)
        combined = a.concat([b])
        self.assertEqual(len(combined), len(a) + len(b))
        self.assertEqual(combined.sensor_size, (200, 150))


class UndistortTests(unittest.TestCase):
    """Camera intrinsics + lens undistortion."""

    def setUp(self):
        self.stream = eventcv.load(str(EXAMPLE_NPZ))  # 640×480

    def test_no_distortion_is_identity(self):
        cam = eventcv.Camera(fx=300.0, fy=300.0, cx=320.0, cy=240.0)  # no distortion coeffs
        np.testing.assert_array_equal(self.stream.undistort(cam).numpy(), self.stream.numpy())

    def test_undistort_returns_stream_on_same_grid(self):
        cam = eventcv.Camera(fx=300.0, fy=300.0, cx=320.0, cy=240.0, k1=-0.3, k2=0.1)
        out = self.stream.undistort(cam)
        self.assertIsInstance(out, eventcv.EventStream)
        self.assertEqual(out.sensor_size, (640, 480))
        self.assertLessEqual(len(out), len(self.stream))  # some events may rectify off-grid
        coords = out.numpy()
        if len(coords):
            self.assertTrue((coords[:, 0] < 640).all() and (coords[:, 1] < 480).all())

    def test_camera_exposes_parameters(self):
        cam = eventcv.Camera(1.0, 2.0, 3.0, 4.0, k1=0.5)
        self.assertEqual((cam.fx, cam.fy, cam.cx, cam.cy), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(cam.distortion, (0.5, 0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
