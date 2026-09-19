"""Red/blue frames must match the float32 NumPy training representation exactly."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv


def reference(events, height, width, pct):
    x, y, _, p = events.T
    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros_like(pos)
    positive = p > 0
    np.add.at(pos, (y[positive], x[positive]), 1)
    np.add.at(neg, (y[~positive], x[~positive]), 1)

    def norm(a):
        if a.max() == 0:
            return a
        thr = np.percentile(a[a > 0], pct)
        if thr <= 0:
            thr = float(a.max())
        return np.clip(a, 0, thr) / thr

    pos_n, neg_n = norm(pos), norm(neg)
    dominate_pos = pos_n >= neg_n
    inten_pos = pos_n * dominate_pos
    inten_neg = neg_n * (~dominate_pos)
    red = np.ones_like(pos)
    green = np.ones_like(pos)
    blue = np.ones_like(pos)
    green -= inten_pos
    blue -= inten_pos
    red -= inten_neg
    green -= inten_neg
    img = np.stack([np.clip(red, 0, 1), np.clip(green, 0, 1), np.clip(blue, 0, 1)])
    return (img * 255).astype(np.uint8)


def stream(events, size=(8, 6)):
    return eventcv.from_numpy(events, sensor_size=size, time_unit="us")


class RedBlueTests(unittest.TestCase):
    def test_matches_numpy(self):
        rng = np.random.default_rng(12345)
        for n in (0, 1, 40, 500, 5000):
            events = np.column_stack((rng.integers(8, size=n), rng.integers(6, size=n),
                                      np.arange(n), rng.integers(2, size=n)))
            for pct in (0, 25.3, 50, 73.5, 99, 100):
                with self.subTest(n=n, pct=pct):
                    np.testing.assert_array_equal(
                        stream(events).redblue(pct=pct).numpy(), reference(events, 6, 8, pct)
                    )

    def test_dominance_and_independent_scales(self):
        # Positive counts [4, 1, 1, 0], negative [1, 1, 0, 1].
        events = np.array([[0, 0, 0, 1]] * 4 + [[1, 0, 0, 1], [2, 0, 0, 1]]
                          + [[0, 0, 0, 0], [1, 0, 0, 0], [3, 0, 0, 0]])
        s = stream(events, (5, 1))
        expected = np.array([[[255, 0, 255, 0, 255]],
                             [[0, 0, 191, 0, 255]],
                             [[0, 255, 191, 255, 255]]], dtype=np.uint8)
        frame = s.redblue(pct=100)
        self.assertEqual(frame.kind, "redblue")
        self.assertEqual(frame.shape, (3, 1, 5))
        self.assertEqual(frame.channel_names, ("red", "green", "blue"))
        np.testing.assert_array_equal(frame.numpy(), expected)
        np.testing.assert_array_equal(eventcv.redblue(s, pct=100, white_frame=False).numpy(), expected)
        events[:, 2] = np.arange(len(events)) * 1000
        np.testing.assert_array_equal(stream(events, (5, 1)).redblue(pct=100).numpy(), expected)

    def test_empty_and_single_polarity(self):
        empty = stream(np.empty((0, 4), dtype=np.int64)).redblue().numpy()
        self.assertTrue((empty == 255).all())
        for p, rgb in ((0, [0, 0, 255]), (1, [255, 0, 0])):
            frame = stream(np.array([[0, 0, 0, p]])).redblue().numpy()
            np.testing.assert_array_equal(frame[:, 0, 0], rgb)
            self.assertTrue((frame[:, 1:, :] == 255).all())

    def test_invalid_percentiles(self):
        s = stream(np.empty((0, 4), dtype=np.int64))
        for pct in (-1, 100.1, np.nan, np.inf, -np.inf):
            with self.subTest(pct=pct), self.assertRaises(ValueError):
                s.redblue(pct=pct)

    def test_named_reader_and_roundtrips(self):
        events = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 2, 0]])
        s = stream(events)
        expected = s.redblue().numpy()
        np.testing.assert_array_equal(s.flatten("redblue").numpy(), expected)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.npz"
            s.save(str(source))
            reader = eventcv.open(str(source), dt_ms=1, repr="redblue")
            self.assertEqual(reader.repr, "redblue")
            np.testing.assert_array_equal(reader[0], expected)
            np.testing.assert_array_equal(reader.batch([0]), expected[None])
            configured = eventcv.open(str(source), dt_ms=1).with_repr("redblue", pct=75, white_frame=False)
            np.testing.assert_array_equal(configured[0], s.redblue(pct=75).numpy())
            for suffix in ("npz", "h5"):
                path = Path(directory) / f"frame.{suffix}"
                s.redblue().save(str(path))
                loaded = eventcv.load_frame(str(path))
                self.assertEqual(loaded.kind, "redblue")
                self.assertEqual(loaded.channel_names, ("red", "green", "blue"))
                np.testing.assert_array_equal(loaded.numpy(), expected)
