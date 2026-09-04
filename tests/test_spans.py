"""Unset representation spans follow the slice window on the file path.

The live-camera rule ("time spans follow the capture window", ``stream()``) applied to
``open()``: a representation built from a ``dt_ms`` slice, a ``slice(t0, t1)`` window, or a
``windows()`` yield defaults its ``window_ms``/``tau_ms``/``max_window_ms`` to that window's
duration instead of the fixed 30 ms, so it covers exactly the events it was handed. An explicit
span always wins, and count-based slicing (``max_events``) keeps the 30 ms default.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv


class SliceSpanTests(unittest.TestCase):
    """Events at 0, 10 and 49 ms: a 30 ms default drops the oldest, a 50 ms window keeps it."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = str(Path(cls.directory.name) / "spans.npz")
        events = np.array(
            [[0, 0, 0, 1], [1, 0, 10_000, 1], [2, 0, 49_000, 0]], dtype=np.int64
        )
        stream = eventcv.from_numpy(events, sensor_size=(4, 2), time_unit="us")
        eventcv.save(stream, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def open(self, **kwargs):
        return eventcv.open(self.path, sensor_size=(4, 2), time_unit="us", **kwargs)

    def test_direct_representations_follow_dt(self):
        slice0 = self.open(dt_ms=50).slice(0)
        for name, span_kwarg in (
            ("mcts", "max_window_ms"),
            ("voxel", "window_ms"),
            ("tsurf", "tau_ms"),
            ("atsurf", "tau_ms"),
            ("tencode", "window_ms"),
        ):
            with self.subTest(name=name):
                followed = getattr(slice0, name)().numpy()
                np.testing.assert_array_equal(
                    followed, getattr(slice0, name)(**{span_kwarg: 50}).numpy()
                )
                self.assertFalse(
                    np.array_equal(followed, getattr(slice0, name)(**{span_kwarg: 30}).numpy())
                )

    def test_named_representation_follows_dt(self):
        expected = self.open(dt_ms=50).slice(0).mcts(max_window_ms=50).numpy()
        np.testing.assert_array_equal(
            self.open(dt_ms=50, repr="mcts").slice(0).numpy(), expected
        )
        np.testing.assert_array_equal(
            self.open(dt_ms=50).slice(0).flatten("mcts").numpy(), expected
        )
        np.testing.assert_array_equal(
            self.open(dt_ms=50).with_repr("mcts")[0], expected
        )

    def test_explicit_span_wins_over_dt(self):
        explicit = self.open(dt_ms=50).slice(0).mcts(max_window_ms=30).numpy()
        np.testing.assert_array_equal(
            self.open(dt_ms=50).with_repr("mcts", max_window_ms=30)[0], explicit
        )

    def test_time_bounds_slice_follows_its_duration(self):
        window = self.open().slice(t0_ms=0, t1_ms=40)
        followed = window.tsurf().numpy()
        np.testing.assert_array_equal(followed, window.tsurf(tau_ms=40).numpy())
        self.assertFalse(np.array_equal(followed, window.tsurf(tau_ms=30).numpy()))

    def test_windows_iterator_follows_the_span(self):
        first = next(iter(self.open(dt_ms=50).windows()))
        np.testing.assert_array_equal(
            first.mcts().numpy(), first.mcts(max_window_ms=50).numpy()
        )

    def test_count_slices_keep_the_default_span(self):
        slice0 = self.open(max_events=3).slice(0)
        followed = slice0.mcts().numpy()
        np.testing.assert_array_equal(followed, slice0.mcts(max_window_ms=30).numpy())
        self.assertFalse(np.array_equal(followed, slice0.mcts(max_window_ms=50).numpy()))

    def test_span_survives_chained_transforms(self):
        flipped = self.open(dt_ms=50).slice(0).flip_x()
        followed = flipped.mcts().numpy()
        np.testing.assert_array_equal(followed, flipped.mcts(max_window_ms=50).numpy())
        self.assertFalse(np.array_equal(followed, flipped.mcts(max_window_ms=30).numpy()))

    def test_whole_loads_keep_the_default_span(self):
        stream = eventcv.load(self.path, sensor_size=(4, 2), time_unit="us")
        np.testing.assert_array_equal(
            stream.mcts().numpy(), stream.mcts(max_window_ms=30).numpy()
        )

    def test_explicit_mcts_windows_beat_the_dt_follow_default(self):
        reader = self.open(dt_ms=50).with_repr("mcts", windows_ms=[1, 5, 20])
        expected = self.open(dt_ms=50).slice(0).mcts(windows_ms=[1, 5, 20]).numpy()
        self.assertEqual(reader[0].shape, (6, 2, 4))
        np.testing.assert_array_equal(reader[0], expected)


if __name__ == "__main__":
    unittest.main()
