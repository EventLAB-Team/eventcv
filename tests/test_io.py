import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def _hdf5_supported() -> bool:
    """True if the extension was built with the `hdf5` feature."""
    try:
        eventcv.load("___eventcv_no_such_file___.h5", sensor_size=(1, 1))
    except FileNotFoundError:
        return True  # dispatched into the reader
    except Exception:
        return False  # recognised but not built in
    return True


class TextLoadTests(unittest.TestCase):
    def _write(self, text: str, name: str = "events.txt") -> str:
        path = Path(tempfile.mkdtemp()) / name
        path.write_text(text)
        return str(path)

    def test_loads_txyp_seconds_and_drops_out_of_bounds(self):
        path = self._write(
            "0.0 1 2 1\n0.000002 3 0 0\n\n# comment\n0.00001 0 4 1\n0.00002 4 0 1\n"
        )
        stream = eventcv.load(path, sensor_size=(4, 5))

        self.assertIsInstance(stream, eventcv.EventStream)
        self.assertEqual(len(stream), 3)  # (4, 0) dropped: x == width
        self.assertEqual(stream.sensor_size, (4, 5))
        self.assertEqual(stream.timestamp_scale_ms, 0.001)

        events = stream.numpy()
        self.assertEqual(events.dtype, np.uint64)
        np.testing.assert_array_equal(events[:, 0], [1, 3, 0])
        np.testing.assert_array_equal(events[:, 1], [2, 0, 4])
        np.testing.assert_array_equal(events[:, 2], [0, 2, 10])
        np.testing.assert_array_equal(events[:, 3], [1, 0, 1])

    def test_xytp_order_and_negative_polarity(self):
        path = self._write("1 2 0.5 -1\n")
        events = eventcv.load(path, sensor_size=(8, 8), order="xytp").numpy()
        np.testing.assert_array_equal(events[0], [1, 2, 500000, 0])

    def test_microsecond_unit(self):
        path = self._write("7 0 0 1\n")
        stream = eventcv.load(path, sensor_size=(4, 4), time_unit="us")
        self.assertEqual(stream.numpy()[0, 2], 7)

    def test_max_events_caps_reading(self):
        path = self._write("0 0 0 1\n1 1 0 1\n2 2 0 1\n3 3 0 1\n")
        stream = eventcv.load(path, sensor_size=(8, 8), time_unit="us", max_events=2)
        self.assertEqual(len(stream), 2)

    def test_feeds_representations(self):
        path = self._write("0.0 0 0 1\n0.001 1 1 0\n")
        frame = eventcv.load(path, sensor_size=(4, 4)).voxel()
        self.assertEqual(frame.shape, (9, 4, 4))

    def test_parse_error_reports_line_number(self):
        path = self._write("0.0 0 0 1\n0.0 nope 0 1\n")
        with self.assertRaisesRegex(ValueError, "line 2"):
            eventcv.load(path, sensor_size=(4, 4))

    def test_text_requires_sensor_size(self):
        with self.assertRaisesRegex(ValueError, "sensor_size"):
            eventcv.load("recording.txt")

    def test_invalid_time_unit(self):
        path = self._write("0.0 0 0 1\n")
        with self.assertRaisesRegex(ValueError, "time_unit"):
            eventcv.load(path, sensor_size=(4, 4), time_unit="furlongs")

    def test_invalid_order(self):
        path = self._write("0.0 0 0 1\n")
        with self.assertRaisesRegex(ValueError, "order"):
            eventcv.load(path, sensor_size=(4, 4), order="pyxt")

    def test_zero_sensor_size(self):
        path = self._write("0.0 0 0 1\n")
        with self.assertRaises(ValueError):
            eventcv.load(path, sensor_size=(0, 4))

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            eventcv.load("does-not-exist.txt", sensor_size=(4, 4))


class DispatchTests(unittest.TestCase):
    def test_unknown_extension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "extension"):
            eventcv.load("recording.mp4")

    def test_hdf5_extension_recognised(self):
        # Either dispatched (needs sensor_size) or reported as not built in — both
        # are ValueErrors that mention HDF5.
        with self.assertRaisesRegex(ValueError, "HDF5"):
            eventcv.load("recording.h5")


@unittest.skipUnless(_hdf5_supported() and h5py is not None, "built without hdf5 feature")
class Hdf5Tests(unittest.TestCase):
    def _write_lzf(self, path: str) -> None:
        with h5py.File(path, "w") as handle:
            group = handle.create_group("events")
            group.create_dataset(
                "x", data=np.array([1, 3, 0, 2, 5], dtype=np.uint16), compression="lzf", chunks=(5,)
            )
            group.create_dataset(
                "y", data=np.array([2, 0, 1, 3, 0], dtype=np.uint16), compression="lzf", chunks=(5,)
            )
            group.create_dataset(
                "t",
                data=np.array([1, 2, 3, 4, 5], dtype=np.uint64) * 1_000_000,  # nanoseconds
                compression="lzf",
                chunks=(5,),
            )
            group.create_dataset(
                "p", data=np.array([1, 0, 1, 0, 1], dtype=bool), compression="lzf", chunks=(5,)
            )

    def test_reads_lzf_bool_nanoseconds(self):
        path = str(Path(tempfile.mkdtemp()) / "events.h5")
        self._write_lzf(path)

        stream = eventcv.load(path, sensor_size=(4, 4), time_unit="ns")
        events = stream.numpy()

        self.assertEqual(len(stream), 4)  # x == 5 dropped
        self.assertEqual(stream.sensor_size, (4, 4))
        np.testing.assert_array_equal(events[:, 0], [1, 3, 0, 2])
        np.testing.assert_array_equal(events[:, 1], [2, 0, 1, 3])
        np.testing.assert_array_equal(events[:, 2], [1000, 2000, 3000, 4000])  # ns -> us
        np.testing.assert_array_equal(events[:, 3], [1, 0, 1, 0])

    def test_max_events_caps_hdf5(self):
        path = str(Path(tempfile.mkdtemp()) / "events.h5")
        self._write_lzf(path)
        stream = eventcv.load(path, sensor_size=(8, 8), time_unit="ns", max_events=2)
        self.assertEqual(len(stream), 2)

    def test_requires_sensor_size(self):
        path = str(Path(tempfile.mkdtemp()) / "events.h5")
        self._write_lzf(path)
        with self.assertRaisesRegex(ValueError, "sensor_size"):
            eventcv.load(path, time_unit="ns")


if __name__ == "__main__":
    unittest.main()
