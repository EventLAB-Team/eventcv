"""Phase 5 algorithms: corner detection (eFAST / Harris), Lucas-Kanade optical flow, and
connected-component labelling. Corner detectors return a chainable corner sub-stream; flow and
labels return EventFrames. Streams are built from synthetic ``.txt`` recordings (real readers)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv


def _write_txt(lines) -> str:
    path = Path(tempfile.mkdtemp()) / "events.txt"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _moving_corner(width=32, height=32) -> "eventcv.EventStream":
    """An L-shaped edge (a horizontal then a vertical arm) sweeping over time — has a corner."""
    lines = []
    t = 0
    for x in range(width):
        lines.append(f"{t} {x} {height // 2} 1")
        t += 10
    for y in range(height):
        lines.append(f"{t} {width // 2} {y} 1")
        t += 10
    return eventcv.load(_write_txt(lines), time_unit="us", sensor_size=(width, height))


class CornerTests(unittest.TestCase):
    def test_efast_returns_chainable_corner_substream(self):
        stream = _moving_corner()
        corners = stream.efast()
        self.assertIsInstance(corners, eventcv.EventStream)
        self.assertLessEqual(len(corners), len(stream))
        self.assertEqual(corners.sensor_size, stream.sensor_size)
        # Corners feed representations like any stream.
        frame = corners.count()
        self.assertEqual(frame.shape[1:], (32, 32))

    def test_harris_returns_subset_and_threshold_is_monotone(self):
        stream = _moving_corner()
        loose = stream.harris_corners(-1.0)
        tight = stream.harris_corners(1.0, tau_ms=50.0)
        self.assertLessEqual(len(tight), len(loose))
        self.assertLessEqual(len(loose), len(stream))

    def test_corner_detectors_handle_empty_stream(self):
        empty = eventcv.load(_write_txt(["0 0 0 1"]), time_unit="us", sensor_size=(32, 32))
        # A single event has no corner support; both return empty.
        self.assertEqual(len(empty.efast()), 0)
        self.assertEqual(len(empty.harris_corners(0.0)), 0)


class FlowTests(unittest.TestCase):
    def test_optical_flow_shape_and_dtype(self):
        stream = _moving_corner()
        flow = stream.optical_flow(window=3)
        self.assertEqual(flow.shape, (2, 32, 32))
        self.assertEqual(flow.channel_names, ("flow_x", "flow_y"))
        self.assertEqual(flow.numpy().dtype, np.float32)

    def test_optical_flow_direction_on_a_moving_bar(self):
        # A vertical bar sweeping left→right: column x fires at 10·x, so flow points along +x.
        lines = [f"{10 * x} {x} {y} 1" for x in range(16) for y in range(16)]
        stream = eventcv.load(_write_txt(lines), time_unit="us", sensor_size=(16, 16))
        flow = stream.optical_flow(window=2).numpy()
        fx, fy = flow[0, 8, 8], flow[1, 8, 8]
        self.assertGreater(fx, 0.0)
        self.assertLess(abs(fy), abs(fx) * 0.1)

    def test_optical_flow_rejects_zero_window(self):
        with self.assertRaises(ValueError):
            _moving_corner().optical_flow(window=0)


class ClusterTests(unittest.TestCase):
    def test_connected_components_separates_distant_blobs(self):
        # Two single-pixel events far apart → two components; background stays 0.
        lines = ["0 1 1 1", "10 6 6 1"]
        stream = eventcv.load(_write_txt(lines), time_unit="us", sensor_size=(8, 8))
        labels = stream.count().connected_components(connectivity=4).numpy()
        self.assertEqual(labels.shape, (1, 8, 8))
        self.assertEqual(labels.dtype, np.uint64)
        self.assertEqual(labels[0, 1, 1], 1)
        self.assertEqual(labels[0, 6, 6], 2)
        self.assertEqual(int(labels.max()), 2)
        self.assertEqual(int(labels[0, 0, 0]), 0)

    def test_connected_components_rejects_bad_connectivity(self):
        stream = eventcv.load(_write_txt(["0 1 1 1"]), time_unit="us", sensor_size=(8, 8))
        with self.assertRaises(ValueError):
            stream.count().connected_components(connectivity=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
