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

    def test_infers_sensor_size_and_time_unit(self):
        # Integer µs (span 5 s -> microseconds); coords up to (3, 2) -> 4x3.
        path = self._write("1000000 0 0 1\n3000000 3 1 0\n6000000 1 2 1\n")
        stream = eventcv.load(path)  # no sensor_size, no time_unit -> both inferred
        self.assertEqual(stream.sensor_size, (4, 3))
        np.testing.assert_array_equal(stream.numpy()[:, 2], [1000000, 3000000, 6000000])

    def test_fractional_text_infers_seconds(self):
        path = self._write("0.0 0 0 1\n0.5 1 1 0\n")
        stream = eventcv.load(path, sensor_size=(8, 8))  # unit inferred from the decimal
        np.testing.assert_array_equal(stream.numpy()[:, 2], [0, 500000])  # 0.5 s -> µs

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
        # `.h5` is recognised: with the feature a missing file reaches the reader
        # (FileNotFoundError); without it, a ValueError that mentions HDF5.
        if _hdf5_supported():
            with self.assertRaises(FileNotFoundError):
                eventcv.load("recording.h5")
        else:
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

    def test_infers_sensor_size_and_time_unit(self):
        path = str(Path(tempfile.mkdtemp()) / "events.h5")
        with h5py.File(path, "w") as handle:
            group = handle.create_group("events")
            group["x"] = np.array([0, 10, 345], dtype=np.uint16)
            group["y"] = np.array([0, 5, 259], dtype=np.uint16)
            # nanoseconds spanning 2 s -> inferred as ns (a sub-second span is ambiguous).
            group["t"] = np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.uint64)
            group["p"] = np.array([1, 0, 1], dtype=bool)

        stream = eventcv.load(path)  # no sensor_size, no time_unit -> both inferred
        self.assertEqual(stream.sensor_size, (346, 260))  # max (345, 259) -> +1
        np.testing.assert_array_equal(stream.numpy()[:, 2], [0, 1_000_000, 2_000_000])  # ns -> µs

    def test_open_slices_hdf5_in_place(self):
        path = str(Path(tempfile.mkdtemp()) / "events.h5")
        self._write_lzf(path)  # t = [1, 2, 3, 4, 5] ms; sensor (8, 8) keeps all 5

        reader = eventcv.open(path, sensor_size=(8, 8), time_unit="ns")
        self.assertIsInstance(reader, eventcv.EventReader)
        self.assertEqual(reader.n_events, 5)
        self.assertEqual(len(reader), 5)
        self.assertEqual(reader.sensor_size, (8, 8))
        self.assertAlmostEqual(reader.duration_ms, 4.0)  # (5000 - 1000) us

        # Half-open [2 ms, 4 ms) -> timestamps 2000, 3000 us.
        window = reader.slice(t0_ms=2.0, t1_ms=4.0).numpy()
        np.testing.assert_array_equal(window[:, 2], [2000, 3000])

        # slice_count matches the matching rows of a full eager load.
        full = eventcv.load(path, sensor_size=(8, 8), time_unit="ns").numpy()
        np.testing.assert_array_equal(reader.slice_count(1, 4).numpy(), full[1:4])

        # No-arg slice returns the whole stream; windows tile it without overlap.
        np.testing.assert_array_equal(reader.slice().numpy(), full)
        self.assertEqual(sum(len(w) for w in reader.windows(step_ms=1.0)), reader.n_events)

    def test_indexed_frames_hdf5(self):
        path = str(Path(tempfile.mkdtemp()) / "events.h5")
        self._write_lzf(path)  # t = [1, 2, 3, 4, 5] ms; sensor (8, 8) keeps all 5

        e = eventcv.open(path, dt_ms=1.0, sensor_size=(8, 8), time_unit="ns")
        self.assertEqual(e.n_slices, 5)
        np.testing.assert_array_equal(e.slice(0).numpy()[:, 2], [1000])  # us
        np.testing.assert_array_equal(e[4].numpy()[:, 2], [5000])
        self.assertEqual(sum(len(e.slice(n)) for n in range(e.n_slices)), e.n_events)


class OpenReaderTests(unittest.TestCase):
    """`open()` works on every format via the in-memory fallback (here: text)."""

    def _reader(self):
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n20 2 2 1\n30 3 3 0\n")
        return eventcv.open(str(path), sensor_size=(8, 8), time_unit="us")

    def _frames(self):
        # dt_ms = 0.01 ms = 10 us -> events at t = 0, 10, 20, 30 us land in frames 0..3.
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n20 2 2 1\n30 3 3 0\n")
        return eventcv.open(str(path), dt_ms=0.01, sensor_size=(8, 8), time_unit="us")

    def test_reports_metadata(self):
        reader = self._reader()
        self.assertEqual(reader.n_events, 4)
        self.assertEqual(reader.sensor_size, (8, 8))
        self.assertEqual(reader.time_span_ms, (0.0, 0.03))
        self.assertAlmostEqual(reader.duration_ms, 0.03)

    def test_slice_by_time_and_count(self):
        reader = self._reader()
        np.testing.assert_array_equal(reader.slice(t0_ms=0.01, t1_ms=0.03).numpy()[:, 2], [10, 20])
        np.testing.assert_array_equal(reader.slice_count(1, 3).numpy()[:, 2], [10, 20])
        self.assertEqual(len(reader.slice()), 4)  # no bounds -> whole stream

    def test_windows_are_lazy_and_tile(self):
        reader = self._reader()
        windows = list(reader.windows(step_ms=0.01))
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(isinstance(w, eventcv.EventStream) for w in windows))
        self.assertEqual(sum(len(w) for w in windows), reader.n_events)

    def test_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            self._reader().windows(step_ms=0.0)

    def test_indexed_frames(self):
        e = self._frames()
        self.assertEqual(e.dt_ms, 0.01)
        self.assertEqual(e.n_slices, 4)
        for n in range(e.n_slices):  # frame n holds the event at t = n*10 us
            np.testing.assert_array_equal(e.slice(n).numpy()[:, 2], [n * 10])
        np.testing.assert_array_equal(e[2].numpy()[:, 2], [20])  # __getitem__ alias
        np.testing.assert_array_equal(e.slice(-1).numpy()[:, 2], [30])  # negative index
        self.assertEqual(sum(len(e.slice(n)) for n in range(e.n_slices)), e.n_events)

    def test_windows_default_to_dt(self):
        e = self._frames()
        self.assertEqual(sum(len(w) for w in e.windows()), e.n_events)

    def test_index_out_of_range_raises(self):
        e = self._frames()
        with self.assertRaises(IndexError):
            e.slice(e.n_slices)
        with self.assertRaises(IndexError):
            e.slice(-e.n_slices - 1)

    def test_integer_index_requires_dt(self):
        r = self._reader()  # opened without dt_ms
        with self.assertRaisesRegex(ValueError, "dt_ms"):
            r.slice(0)
        with self.assertRaisesRegex(ValueError, "dt_ms"):
            _ = r.n_slices
        with self.assertRaisesRegex(ValueError, "dt_ms"):
            r.windows()  # no step_ms and no dt_ms

    def test_index_and_time_bounds_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            self._frames().slice(0, t1_ms=5.0)


if __name__ == "__main__":
    unittest.main()
