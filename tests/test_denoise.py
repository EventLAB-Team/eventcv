"""Denoising filters (Phase 3): background-activity, refractory, and hot-pixel removal.

Each filter is exercised on a synthetic recording built so that "signal" (spatially and
temporally correlated events) is separable from planted noise.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

SENSOR = (128, 96)


def _load(events):
    """Write `(t, x, y, p)` rows to a temp `t x y p` txt file and load them (µs timestamps)."""
    path = Path(tempfile.mkdtemp()) / "events.txt"
    lines = [f"{t} {x} {y} {p}" for (t, x, y, p) in events]
    path.write_text("\n".join(lines) + "\n")
    return eventcv.load(str(path), sensor_size=SENSOR, time_unit="us")


def _signal_and_noise():
    """A correlated 4×4 block sweeping the left half + isolated noise on a sparse right-half grid.

    Returns `(events, noise_pixels)` with `events` time-sorted. Signal pixels keep `x < 60`;
    every noise pixel sits at `x >= 80` on an 8-pixel grid, so none are 3×3-adjacent to each
    other or to the signal — background-activity must drop them all.
    """
    events = []
    noise_pixels = set()
    t = 0
    for step in range(150):
        bx = 10 + (step % 40)
        for dy in range(4):
            for dx in range(4):
                events.append((t, bx + dx, 10 + dy, step % 2))
                t += 1
        if step % 3 == 0:  # sprinkle one isolated noise event between blocks
            nx, ny = 80 + (step % 5) * 8, 10 + (step % 9) * 8
            noise_pixels.add((nx, ny))
            events.append((t, nx, ny, 1))
            t += 1
        t += 100  # gap between blocks
    events.sort(key=lambda e: e[0])
    return events, noise_pixels


class DenoiseTests(unittest.TestCase):
    def setUp(self):
        events, self.noise_pixels = _signal_and_noise()
        self.stream = _load(events)

    def test_background_activity_removes_isolated_noise_keeps_signal(self):
        out = self.stream.background_activity_filter(1_000)
        self.assertIsInstance(out, eventcv.EventStream)
        self.assertEqual(out.sensor_size, SENSOR)
        self.assertLessEqual(len(out), len(self.stream))

        coords = out.numpy()
        surviving = set(zip(coords[:, 0].tolist(), coords[:, 1].tolist()))
        self.assertTrue(surviving.isdisjoint(self.noise_pixels))  # every noise pixel gone
        self.assertTrue((coords[:, 0] < 60).all())  # only the signal region remains
        self.assertGreater(len(out), len(self.stream) // 2)  # bulk of the signal retained

    def test_refractory_filter_only_drops_events(self):
        out = self.stream.refractory_filter(50)
        self.assertEqual(out.sensor_size, SENSOR)
        self.assertLessEqual(len(out), len(self.stream))
        # A large dead time collapses each pixel to at most one event.
        sparse = self.stream.refractory_filter(10_000_000).numpy()
        pixels = set(zip(sparse[:, 0].tolist(), sparse[:, 1].tolist()))
        self.assertEqual(len(sparse), len(pixels))

    def test_hot_pixel_filter_removes_a_stuck_pixel(self):
        events, _ = _signal_and_noise()
        events += [(900_000 + i, 0, 0, 1) for i in range(2_000)]  # stuck pixel at (0,0)
        events.sort(key=lambda e: e[0])
        stream = _load(events)

        out = stream.hot_pixel_filter(3.0)
        self.assertEqual(out.sensor_size, SENSOR)
        coords = out.numpy()
        surviving = set(zip(coords[:, 0].tolist(), coords[:, 1].tolist()))
        self.assertNotIn((0, 0), surviving)
        self.assertLess(len(out), len(stream))

    def test_filters_chain_and_handle_empty(self):
        out = self.stream.hot_pixel_filter().refractory_filter(10).background_activity_filter(500)
        self.assertIsInstance(out, eventcv.EventStream)
        empty = self.stream.time_window(10**12, 10**12 + 1)  # selects nothing
        self.assertEqual(len(empty), 0)
        self.assertEqual(len(empty.background_activity_filter(100)), 0)
        self.assertEqual(len(empty.refractory_filter(100)), 0)
        self.assertEqual(len(empty.hot_pixel_filter()), 0)


if __name__ == "__main__":
    unittest.main()
