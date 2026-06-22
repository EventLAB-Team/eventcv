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
        expected = np.load(EXAMPLE_PATH)["event_data"]

        self.assertIsInstance(stream, eventcv.EventStream)
        self.assertEqual(len(stream), len(expected))
        self.assertEqual(stream.shape, (len(expected), 4))
        self.assertEqual(stream.columns, ("x", "y", "t", "p"))
        self.assertEqual(stream.sensor_size, (640, 480))

        events = stream.numpy()
        self.assertEqual(events.dtype, np.uint64)
        np.testing.assert_array_equal(events[:, 0], expected["x"])
        np.testing.assert_array_equal(events[:, 1], expected["y"])
        np.testing.assert_array_equal(events[:, 2], expected["t"])
        np.testing.assert_array_equal(events[:, 3], expected["p"])

        first_event = events[0].copy()
        events[0] = 0
        np.testing.assert_array_equal(stream.numpy()[0], first_event)

    def test_generates_polarity_representation(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        frame = stream.flatten(normalize=False)
        events = frame.numpy()

        self.assertIsInstance(frame, eventcv.EventFrame)
        self.assertEqual(frame.shape, (2, 480, 640))
        self.assertEqual(frame.channel_names, ("positive", "negative"))
        self.assertEqual(events.dtype, np.uint16)
        self.assertEqual(events.shape, frame.shape)
        self.assertEqual(events.sum(), len(stream))

        events.fill(0)
        self.assertEqual(frame.numpy().sum(), len(stream))

    def test_generates_normalized_polarity_representation(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        events = stream.flatten().numpy()
        explicit = stream.flatten(normalize=True).numpy()

        self.assertEqual(events.dtype, np.uint8)
        self.assertEqual(events.shape, (2, 480, 640))
        self.assertEqual(events.max(), np.iinfo(np.uint8).max)
        np.testing.assert_array_equal(events, explicit)

    def test_rejects_unknown_representation(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        with self.assertRaisesRegex(TypeError, "unsupported representation"):
            stream.flatten(object())

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

    def test_out_of_bounds_coordinate_raises_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out-of-bounds.npz"
            np.savez(path, event_data=np.array([(640, 0, 0, True)], dtype=EVENT_DTYPE))

            with self.assertRaisesRegex(ValueError, "exceeds sensor size 640x480"):
                eventcv.load(str(path))


if __name__ == "__main__":
    unittest.main()
