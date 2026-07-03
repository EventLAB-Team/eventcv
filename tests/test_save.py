"""Saving / serialization (Workstream C): stream and frame writers round-trip the readers.

Streams are saved to every supported format and reloaded; npz/HDF5/rosbag round-trip exactly,
txt at the event level (it carries no metadata header, so size/unit are passed on load). Frames
(computed representations) round-trip through npz and HDF5 preserving shape/dtype/kind/names.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

SENSOR = (64, 48)


def _make_stream():
    """A small, time-sorted synthetic recording loaded via the txt reader (µs timestamps)."""
    events = []
    for step in range(200):
        t = step * 1000 + 5  # strictly increasing microseconds
        events.append((t, step % SENSOR[0], (step * 7) % SENSOR[1], step % 2))
    path = Path(tempfile.mkdtemp()) / "in.txt"
    path.write_text("\n".join(f"{t} {x} {y} {p}" for (t, x, y, p) in events) + "\n")
    return eventcv.load(str(path), sensor_size=SENSOR, time_unit="us")


def _tmp(name):
    return str(Path(tempfile.mkdtemp()) / name)


class SaveStreamTests(unittest.TestCase):
    def setUp(self):
        self.stream = _make_stream()

    def _assert_exact(self, loaded):
        np.testing.assert_array_equal(loaded.numpy(), self.stream.numpy())
        self.assertEqual(loaded.sensor_size, self.stream.sensor_size)

    def test_npz_round_trips_exactly(self):
        path = _tmp("out.npz")
        self.stream.save(path)
        self._assert_exact(eventcv.load(path))  # metadata stored: no options needed

    def test_hdf5_round_trips_exactly(self):
        path = _tmp("out.h5")
        self.stream.save(path)
        self._assert_exact(eventcv.load(path))

    def test_rosbag_round_trips_exactly(self):
        path = _tmp("out.bag")
        self.stream.save(path)
        self._assert_exact(eventcv.load(path))

    def test_text_round_trips_at_event_level(self):
        path = _tmp("out.txt")
        self.stream.save(path)
        # txt has no header; supply the grid + unit the way a user would.
        loaded = eventcv.load(path, sensor_size=SENSOR, time_unit="us")
        np.testing.assert_array_equal(loaded.numpy(), self.stream.numpy())

    def test_module_level_save_dispatches_streams(self):
        path = _tmp("out.npz")
        eventcv.save(self.stream, path)
        self._assert_exact(eventcv.load(path))

    def test_empty_stream_round_trips(self):
        empty = self.stream.time_window(10**12, 10**12 + 1)  # selects nothing
        self.assertEqual(len(empty), 0)
        for ext in (".npz", ".h5", ".bag"):
            path = _tmp(f"empty{ext}")
            empty.save(path)
            loaded = eventcv.load(path)
            self.assertEqual(len(loaded), 0)
            self.assertEqual(loaded.sensor_size, SENSOR)

    def test_unsupported_save_format_raises(self):
        with self.assertRaises(ValueError):
            self.stream.save(_tmp("out.aedat"))


class SaveFrameTests(unittest.TestCase):
    def setUp(self):
        self.stream = _make_stream()
        self.frame = self.stream.voxel(bins=3)

    def _assert_frame_eq(self, loaded):
        self.assertEqual(loaded.shape, self.frame.shape)
        self.assertEqual(loaded.kind, self.frame.kind)
        self.assertEqual(tuple(loaded.channel_names), tuple(self.frame.channel_names))
        np.testing.assert_array_equal(loaded.numpy(), self.frame.numpy())

    def test_frame_round_trips_npz(self):
        path = _tmp("frame.npz")
        self.frame.save(path)
        self._assert_frame_eq(eventcv.load_frame(path))

    def test_frame_round_trips_hdf5(self):
        path = _tmp("frame.h5")
        self.frame.save(path)
        self._assert_frame_eq(eventcv.load_frame(path))

    def test_module_level_save_dispatches_frames(self):
        path = _tmp("frame.npz")
        eventcv.save(self.frame, path)
        self._assert_frame_eq(eventcv.load_frame(path))


class PngExportTests(unittest.TestCase):
    def setUp(self):
        self.stream = _make_stream()

    def _read_png_header(self, path):
        """Returns (width, height) from a PNG's IHDR without a decoder dependency."""
        data = Path(path).read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        # IHDR is the first chunk: 8-byte sig, 4-byte len, 4-byte "IHDR", then w,h (big-endian u32).
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height

    def test_frame_saves_a_png_view(self):
        path = _tmp("frame.png")
        self.stream.count().save(path, colormap="turbo")
        self.assertEqual(self._read_png_header(path), SENSOR)

    def test_save_rejects_unknown_colormap(self):
        with self.assertRaises(ValueError):
            self.stream.count().save(_tmp("frame.png"), colormap="rainbow")

    def test_export_png_writes_a_numbered_sequence(self):
        out = Path(tempfile.mkdtemp()) / "seq"
        frames = (self.stream.count() for _ in range(3))
        paths = eventcv.export_png(frames, str(out), prefix="f_", colormap="grayscale")

        self.assertEqual([Path(p).name for p in paths], ["f_00000.png", "f_00001.png", "f_00002.png"])
        for path in paths:
            self.assertEqual(self._read_png_header(path), SENSOR)

    def test_export_png_accepts_a_single_frame(self):
        out = Path(tempfile.mkdtemp()) / "one"
        paths = eventcv.export_png(self.stream.atsurf(), str(out))
        self.assertEqual(len(paths), 1)


class FrameSinkTests(unittest.TestCase):
    def test_sink_streams_a_window_stack(self):
        if eventcv.FrameSink is None:
            self.skipTest("extension built without HDF5 support")
        stream = _make_stream()
        frame = stream.voxel(bins=2)

        path = _tmp("stack.h5")
        sink = eventcv.FrameSink(path)
        for _ in range(4):
            sink.append(frame)
        self.assertEqual(sink.n_frames, 4)
        sink.finish()
        self.assertTrue(Path(path).exists())

    def test_save_rejects_arbitrary_objects(self):
        with self.assertRaises(TypeError):
            eventcv.save([1, 2, 3], _tmp("bad.npz"))


if __name__ == "__main__":
    unittest.main()
