import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

_MVSEC_BAG = "data/development/outdoor_day1_data.bag"
_AEDAT2 = "data/development/+0+2+0_l_qry.aedat"

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


class DatasetReaderTests(unittest.TestCase):
    """`EventReader` as a PyTorch-style map dataset (Workstream D2)."""

    def _dataset(self, repr="count"):
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n20 2 2 1\n30 3 3 0\n")
        return eventcv.open(str(path), dt_ms=0.01, repr=repr, sensor_size=(8, 8), time_unit="us")

    def test_len_is_n_slices_in_dataset_mode(self):
        reader = self._dataset()
        self.assertEqual(reader.repr, "count")
        self.assertEqual(len(reader), reader.n_slices)  # 4
        self.assertEqual(len(reader), 4)

    def test_len_stays_event_count_without_dt(self):
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n")
        reader = eventcv.open(str(path), sensor_size=(8, 8), time_unit="us")
        self.assertIsNone(reader.repr)
        self.assertEqual(len(reader), 2)

    def test_getitem_returns_dense_frame_when_repr_set(self):
        reader = self._dataset()
        frame = reader[0]
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (1, 8, 8))  # count = 1 channel
        self.assertEqual(frame.dtype, np.uint8)   # normalized by default
        # Frame 0 holds exactly the event at (0,0).
        self.assertEqual(frame.sum(), frame[0, 0, 0])

    def test_getitem_stays_stream_without_repr(self):
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n20 2 2 1\n30 3 3 0\n")
        reader = eventcv.open(str(path), dt_ms=0.01, sensor_size=(8, 8), time_unit="us")
        self.assertIsInstance(reader[0], eventcv.EventStream)

    def test_with_repr_sets_parameters(self):
        reader = self._dataset(repr=None).with_repr("voxel", bins=5)
        self.assertEqual(reader.repr, "voxel")
        self.assertEqual(reader[0].shape, (5, 8, 8))
        self.assertEqual(reader[0].dtype, np.float32)

    def test_slice_carries_open_repr(self):
        # A slice from a reader opened with a representation remembers it, so `view()`/
        # `flatten()` render that repr instead of the default polarity image.
        reader = self._dataset(repr="voxel")
        stream = reader.slice(0)
        self.assertIsInstance(stream, eventcv.EventStream)
        self.assertEqual(stream.repr, "voxel")
        self.assertEqual(stream.flatten().kind, "voxel")  # no explicit repr -> the stored one
        # Transforms and windows keep it; an explicit argument still overrides it.
        self.assertEqual(stream.flip_x().repr, "voxel")
        self.assertEqual(next(reader.windows()).repr, "voxel")
        self.assertEqual(stream.flatten("count").kind, "count")

    def test_slice_has_no_repr_without_open_repr(self):
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n20 2 2 1\n30 3 3 0\n")
        reader = eventcv.open(str(path), dt_ms=0.01, sensor_size=(8, 8), time_unit="us")
        stream = reader.slice(0)
        self.assertIsNone(stream.repr)
        self.assertEqual(stream.flatten().kind, "polarity")  # falls back to polarity

    def test_batch_stacks_indices(self):
        reader = self._dataset()
        batch = reader.batch([0, 2, 3])
        self.assertEqual(batch.shape, (3, 1, 8, 8))
        self.assertEqual(batch.dtype, np.uint8)
        # Each row matches the corresponding single-index render.
        for row, index in enumerate([0, 2, 3]):
            np.testing.assert_array_equal(batch[row], reader[index])

    def test_batch_accepts_range_and_is_empty_safe(self):
        reader = self._dataset()
        self.assertEqual(reader.batch(range(reader.n_slices)).shape, (4, 1, 8, 8))
        self.assertEqual(reader.batch([]).shape, (0, 1, 8, 8))

    def test_batch_without_repr_raises(self):
        path = Path(tempfile.mkdtemp()) / "events.txt"
        path.write_text("0 0 0 1\n10 1 1 0\n")
        reader = eventcv.open(str(path), dt_ms=0.01, sensor_size=(8, 8), time_unit="us")
        with self.assertRaisesRegex(ValueError, "representation"):
            reader.batch([0])

    def test_dataloader_style_iteration_tiles_every_slice(self):
        # The "automatic" path: __len__ + __getitem__ let a sampler walk all frames.
        reader = self._dataset()
        seen = [reader[i] for i in range(len(reader))]
        self.assertEqual(len(seen), 4)
        total_events = sum(int(frame.sum()) for frame in seen)  # normalized counts
        self.assertGreater(total_events, 0)

    def test_collate_batches_raw_streams_as_a_list(self):
        # A repr-less reader yields EventStreams; ecv.collate returns them as a plain list
        # (they can't stack into a tensor) so a DataLoader still batches them.
        reader = self._dataset(repr=None)
        batch = eventcv.collate([reader[0], reader[1]])
        self.assertIsInstance(batch, list)
        self.assertEqual([type(s).__name__ for s in batch], ["EventStream", "EventStream"])

    def test_collate_defers_dense_batches_to_torch(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not installed")
        import torch

        reader = self._dataset()  # repr="count" -> dense [C, H, W] arrays
        loader = torch.utils.data.DataLoader(reader, batch_size=4, collate_fn=eventcv.collate)
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(tuple(batch.shape), (4, 1, 8, 8))


@unittest.skipUnless(os.path.exists(_MVSEC_BAG), "MVSEC bag fixture not present")
class BagSliceTests(unittest.TestCase):
    """In-place rosbag slicing via the chunk index (gated on the real 8.6 GB MVSEC bag)."""

    def test_open_and_time_slice_in_place(self):
        reader = eventcv.open(_MVSEC_BAG, dt_ms=30)
        self.assertEqual(reader.sensor_size, (346, 260))
        self.assertGreater(reader.n_slices, 0)

        frame = reader.slice(reader.n_slices // 2)  # a mid-file frame, no full read
        self.assertIsInstance(frame, eventcv.EventStream)
        timestamps = frame.numpy()[:, 2]
        if len(timestamps):
            start = reader.time_span_ms[0]
            lo = round((start + (reader.n_slices // 2) * 30) * 1000)
            hi = round((start + (reader.n_slices // 2 + 1) * 30) * 1000)
            self.assertTrue((timestamps >= lo).all() and (timestamps < hi).all())


class PropheseeDatTests(unittest.TestCase):
    """Synthetic Prophesee .dat round-trip (byte layout exercised without a real fixture)."""

    def _write(self, events) -> str:
        # Header (%-comment lines), then [event_type, event_size], then 8-byte LE records:
        # uint32 timestamp, uint32 (x | y<<14 | p<<28).
        import struct

        path = Path(tempfile.mkdtemp()) / "events.dat"
        with open(path, "wb") as handle:
            handle.write(b"% Date 2024-01-01 00:00:00\n% Width 640\n% Height 480\n% Version 2\n")
            handle.write(bytes([0x00, 0x08]))
            for t, x, y, p in events:
                handle.write(struct.pack("<II", t, (x & 0x3FFF) | ((y & 0x3FFF) << 14) | (p << 28)))
        return str(path)

    def test_round_trip(self):
        events = [(100, 10, 20, 1), (105, 600, 400, 0), (110, 0, 0, 1)]
        path = self._write(events)
        stream = eventcv.load(path).numpy()

        self.assertEqual(len(stream), 3)
        np.testing.assert_array_equal(stream[:, 0], [10, 600, 0])
        np.testing.assert_array_equal(stream[:, 1], [20, 400, 0])
        np.testing.assert_array_equal(stream[:, 2], [100, 105, 110])
        np.testing.assert_array_equal(stream[:, 3], [1, 0, 1])

    def test_sensor_size_inferred_from_header(self):
        reader = eventcv.open(self._write([(1, 1, 1, 1)]))
        self.assertEqual(reader.sensor_size, (640, 480))


@unittest.skipUnless(os.path.exists(_AEDAT2), "AEDAT 2.0 fixture not present")
class Aedat2Tests(unittest.TestCase):
    """AEDAT 2.0 reader on the real DAVIS346 recording (gated on the local fixture)."""

    def test_decodes_dvs_events(self):
        stream = eventcv.load(_AEDAT2, max_events=1_000_000)
        events = stream.numpy()
        self.assertEqual(stream.sensor_size, (346, 260))  # from the Davis346 chip header
        self.assertEqual(len(events), 1_000_000)
        self.assertTrue((events[:, 0] < 346).all())
        self.assertTrue((events[:, 1] < 260).all())
        self.assertTrue(set(np.unique(events[:, 3])).issubset({0, 1}))
        self.assertTrue((np.diff(events[:, 2].astype(np.int64)) >= 0).all())  # µs, monotonic


if __name__ == "__main__":
    unittest.main()
