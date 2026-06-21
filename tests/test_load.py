import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv


EXAMPLE_PATH = Path(__file__).parents[1] / "data" / "test" / "example.npz"
EVENT_DTYPE = [("x", "<u2"), ("y", "<u2"), ("t", "<u2"), ("p", "?")]


class LoadTests(unittest.TestCase):
    def test_loads_rust_backed_stream(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        self.assertIsInstance(stream, eventcv.EventStream)
        self.assertEqual(len(stream), 44_761)
        self.assertEqual(stream.shape, (44_761, 4))
        self.assertEqual(stream.columns, ("x", "y", "t", "p"))

        events = stream.to_numpy()
        self.assertEqual(events.dtype, np.uint64)
        np.testing.assert_array_equal(events[0], [9, 312, 0, 1])
        np.testing.assert_array_equal(events[-1], [2, 311, 50_107, 0])

        events[0] = 0
        np.testing.assert_array_equal(stream.to_numpy()[0], [9, 312, 0, 1])

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            eventcv.load("missing.npz")

    def test_missing_event_data_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-event-data.npz"
            np.savez(path, other=np.array([], dtype=EVENT_DTYPE))

            with self.assertRaisesRegex(ValueError, "missing event_data"):
                eventcv.load(str(path))

    def test_incompatible_dtype_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-dtype.npz"
            dtype = [("x", "<u4"), ("y", "<u2"), ("t", "<u2"), ("p", "?")]
            np.savez(path, event_data=np.array([(1, 2, 3, True)], dtype=dtype))

            with self.assertRaises(ValueError):
                eventcv.load(str(path))


if __name__ == "__main__":
    unittest.main()
