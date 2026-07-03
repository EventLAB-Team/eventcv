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


def _moving_corner_lines(width=32, height=32, dt_us=1000):
    """L-shaped edge sweeping over time: a horizontal arm then a vertical arm, one column/row
    per `dt_us`. Spread over time so it can be sliced into frames by a reader."""
    lines = []
    t = 0
    for x in range(width):
        lines.append(f"{t} {x} {height // 2} 1")
        t += dt_us
    for y in range(height):
        lines.append(f"{t} {width // 2} {y} 1")
        t += dt_us
    return lines


def _moving_corner(width=32, height=32) -> "eventcv.EventStream":
    lines = _moving_corner_lines(width, height, dt_us=10)
    return eventcv.load(_write_txt(lines), time_unit="us", sensor_size=(width, height))


def _moving_corner_reader(dt_ms=2.0, repr=None):
    """An EventReader over the swept L-corner recording (32×32, ~64 ms span)."""
    path = _write_txt(_moving_corner_lines(dt_us=1000))
    return eventcv.open(
        path, dt_ms=dt_ms, repr=repr, time_unit="us", sensor_size=(32, 32)
    )


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
        loose = stream.harris_corners(-1e9)   # keep essentially everything scored
        tight = stream.harris_corners(1e6)    # strict
        self.assertLessEqual(len(tight), len(loose))
        self.assertLessEqual(len(loose), len(stream))

    def test_harris_rejects_a_straight_moving_edge(self):
        # A lone vertical edge sweeping +x has no corner; threshold=0 (default) should keep ~none.
        lines = []
        t = 0
        for step in range(30):
            for y in range(2, 28):
                lines.append(f"{t} {5 + step} {y} 1")
            t += 100
        edge = eventcv.load(_write_txt(lines), time_unit="us", sensor_size=(40, 30))
        self.assertLess(len(edge.harris_corners()), len(edge) * 0.05)

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

    def test_view_by_representation_name_builds_the_frame(self):
        # `stream.view("flow")` renders the flow frame; verify the underlying generator (shared
        # with flatten) resolves the name to the right representation without opening a window.
        s = _moving_corner()
        self.assertEqual(s.flatten("flow").channel_names, ("flow_x", "flow_y"))
        self.assertEqual(s.flatten("count").channel_names, ("count",))
        with self.assertRaises(ValueError):
            s.flatten("not_a_repr")


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


class ReaderPipelineTests(unittest.TestCase):
    """Phase 5 ops applied across an EventReader — the `data.efast()...` and video workflows."""

    def test_reader_efast_composes_with_slice_and_count(self):
        reader = _moving_corner_reader()
        corners = reader.efast()
        # Each slice is now a corner sub-stream; index it and reduce to a frame.
        first = corners.slice(0)
        self.assertIsInstance(first, eventcv.EventStream)
        raw = reader.slice(0)
        self.assertLessEqual(len(first), len(raw))
        frame = corners.slice(0).count()
        self.assertEqual(frame.shape[1:], (32, 32))

    def test_reader_efast_windows_yield_corner_streams(self):
        reader = _moving_corner_reader()
        windows = list(reader.efast().windows())
        self.assertEqual(len(windows), reader.n_slices)
        self.assertTrue(all(isinstance(w, eventcv.EventStream) for w in windows))
        # Corners are a subset of the raw slices.
        for corner_w, raw_w in zip(reader.efast().windows(), reader.windows()):
            self.assertLessEqual(len(corner_w), len(raw_w))

    def test_reader_harris_and_efast_return_readers(self):
        reader = _moving_corner_reader()
        self.assertEqual(reader.efast().n_slices, reader.n_slices)
        self.assertEqual(reader.harris_corners(0.0).n_slices, reader.n_slices)

    def test_flow_representation_makes_reader_a_dense_dataset(self):
        reader = _moving_corner_reader(repr="flow")
        self.assertEqual(reader.repr, "flow")
        frame = reader[0]  # dense [C, H, W] numpy array
        self.assertEqual(frame.shape, (2, 32, 32))
        self.assertEqual(frame.dtype, np.float32)
        batch = reader.batch([0, 1, 2])
        self.assertEqual(batch.shape, (3, 2, 32, 32))

    def test_with_repr_flow_window_override(self):
        reader = _moving_corner_reader().with_repr("flow", window=5)
        self.assertEqual(reader.repr, "flow")
        self.assertEqual(reader[0].shape, (2, 32, 32))

    def test_export_png_video_of_corner_and_flow_frames(self):
        reader = _moving_corner_reader()
        with tempfile.TemporaryDirectory() as out:
            # Corner-detection video: one PNG per slice.
            corner_paths = eventcv.export_png(
                (w.count() for w in reader.efast().windows()), out, prefix="corner_"
            )
            # Optical-flow video over the same recording.
            flow_paths = eventcv.export_png(
                (w.optical_flow() for w in reader.windows()), out, prefix="flow_"
            )
        self.assertEqual(len(corner_paths), reader.n_slices)
        self.assertEqual(len(flow_paths), reader.n_slices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
