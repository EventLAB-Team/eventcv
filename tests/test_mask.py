"""ROI masking: the shape builders, the PNG round trip, and applying a mask to events.

Covers the paths that don't need a camera attached — the mask builders, `EventStream.mask` /
`EventReader.mask`, and `stream(mask=…)` argument validation. Drawing a mask interactively and
filtering a live camera need a window and hardware, so they are exercised by hand.
"""

import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"
SENSOR = (640, 480)  # (width, height), as everywhere else in eventcv


class MaskBuilderTests(unittest.TestCase):
    def test_builders_return_boolean_arrays_shaped_like_the_sensor(self):
        # sensor_size is (width, height); the array is (H, W), like any other numpy image.
        for mask in (
            eventcv.circle_mask(SENSOR, 320, 240, 100),
            eventcv.ellipse_mask(SENSOR, 320, 240, 100, 50),
            eventcv.rect_mask(SENSOR, 0, 0, 64, 32),
            eventcv.polygon_mask(SENSOR, [(0, 0), (100, 0), (0, 100)]),
        ):
            self.assertEqual(mask.shape, (480, 640))
            self.assertEqual(mask.dtype, np.dtype(bool))

    def test_circle_covers_its_area_and_nothing_outside_it(self):
        mask = eventcv.circle_mask(SENSOR, 320, 240, 100)
        # Pixel-centre sampling puts the count within a percent of the true area.
        self.assertAlmostEqual(mask.sum() / (math.pi * 100**2), 1.0, delta=0.01)
        y, x = np.nonzero(mask)
        self.assertTrue((((x + 0.5 - 320) ** 2 + (y + 0.5 - 240) ** 2) <= 100**2).all())

    def test_rect_keeps_exactly_its_box_and_clamps_off_sensor(self):
        mask = eventcv.rect_mask(SENSOR, 10, 20, 30, 40)
        self.assertTrue(mask[20:60, 10:40].all())
        self.assertEqual(mask.sum(), 30 * 40)
        # A box hanging off the edge keeps the overlap rather than raising.
        self.assertEqual(eventcv.rect_mask(SENSOR, -10, -10, 20, 20).sum(), 10 * 10)

    def test_masks_compose_with_numpy_operators(self):
        circle = eventcv.circle_mask(SENSOR, 320, 240, 100)
        corner = eventcv.rect_mask(SENSOR, 0, 0, 64, 64)
        self.assertEqual((circle | corner).sum(), circle.sum() + corner.sum())
        self.assertEqual((~circle).sum(), 640 * 480 - circle.sum())

    def test_sensor_size_must_be_positive(self):
        for bad in ((0, 480), (640, -1)):
            with self.assertRaises(ValueError):
                eventcv.circle_mask(bad, 1, 1, 1)


class MaskFileTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "roi.png")

    def test_png_round_trip_is_exact(self):
        mask = eventcv.circle_mask(SENSOR, 300, 200, 120)
        eventcv.save_mask(mask, self.path)
        loaded = eventcv.load_mask(self.path)
        self.assertEqual(loaded.dtype, np.dtype(bool))
        np.testing.assert_array_equal(loaded, mask)

    def test_only_png_is_accepted(self):
        with self.assertRaises(Exception):
            eventcv.save_mask(eventcv.rect_mask(SENSOR, 0, 0, 8, 8), self.path[:-4] + ".npz")


class MaskApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = eventcv.load(str(EXAMPLE_NPZ))

    def test_mask_keeps_only_events_inside_it(self):
        mask = eventcv.circle_mask(self.stream.sensor_size, 320, 240, 100)
        masked = self.stream.mask(mask)
        self.assertLess(len(masked), len(self.stream))
        x, y = masked.numpy()[:, 0], masked.numpy()[:, 1]
        self.assertTrue(mask[y, x].all())
        # Coordinates and the sensor grid are untouched — a mask drops events, it doesn't crop.
        self.assertEqual(masked.sensor_size, self.stream.sensor_size)

    def test_an_8_bit_map_masks_the_same_way_as_a_boolean_one(self):
        mask = eventcv.circle_mask(self.stream.sensor_size, 320, 240, 100)
        as_8bit = (mask * 255).astype(np.uint8)
        self.assertEqual(len(self.stream.mask(as_8bit)), len(self.stream.mask(mask)))
        # Any non-zero value keeps the pixel, not just 255.
        self.assertEqual(len(self.stream.mask(mask.astype(np.uint8))), len(self.stream.mask(mask)))

    def test_a_mask_of_the_wrong_size_raises_rather_than_dropping_everything(self):
        with self.assertRaises(ValueError) as caught:
            self.stream.mask(np.ones((10, 10), dtype=bool))
        self.assertIn("640x480", str(caught.exception))

    def test_an_unsupported_dtype_points_at_the_fix(self):
        with self.assertRaises(TypeError) as caught:
            self.stream.mask(np.ones((480, 640), dtype=np.float32))
        self.assertIn("mask > 0", str(caught.exception))

    def test_reader_defers_the_mask_onto_every_slice(self):
        reader = eventcv.open(str(EXAMPLE_NPZ), dt_ms=30)
        mask = eventcv.circle_mask(reader.sensor_size, 320, 240, 100)
        masked = reader.mask(mask)
        for index in range(min(3, masked.n_slices)):
            events = masked.slice(index).numpy()
            if len(events):
                self.assertTrue(mask[events[:, 1], events[:, 0]].all())
        self.assertLess(len(masked.slice(0)), len(reader.slice(0)))

    def test_the_functional_form_forwards(self):
        mask = eventcv.rect_mask(self.stream.sensor_size, 0, 0, 320, 240)
        self.assertEqual(len(eventcv.mask(self.stream, mask)), len(self.stream.mask(mask)))
        self.assertIn("draw_mask", dir(eventcv))


@unittest.skipUnless(eventcv.EventCamera is not None, "built without camera feature")
class CameraMaskArgumentTests(unittest.TestCase):
    """`stream(mask=…)` validates its argument before the device is touched."""

    def test_a_bad_mask_raises_without_hardware(self):
        for bad in (np.ones((4, 4), dtype=np.float32), np.ones(16, dtype=bool), "aperture.png"):
            with self.assertRaises(TypeError):
                eventcv.stream(mask=bad)

    def test_the_camera_exposes_the_mask_surface(self):
        for name in ("mask", "draw_mask"):
            self.assertTrue(hasattr(eventcv.EventCamera, name), name)


if __name__ == "__main__":
    unittest.main()
