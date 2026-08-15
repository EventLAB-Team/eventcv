"""Flexible timescales: every time argument accepts ``_s`` / ``_ms`` / ``_us`` / ``_ns`` siblings.

The units are alternative spellings of one quantity, so the same duration expressed four ways must
produce identical results, and passing two of them for the same quantity must raise rather than
silently picking one.
"""

import unittest
from pathlib import Path

import numpy as np

import eventcv

EXAMPLE = Path(__file__).parents[1] / "data" / "test" / "example.npz"


class OpenUnitTests(unittest.TestCase):
    def test_dt_is_the_same_reader_in_any_unit(self):
        readers = [
            eventcv.open(str(EXAMPLE), dt_ms=30),
            eventcv.open(str(EXAMPLE), dt_s=0.03),
            eventcv.open(str(EXAMPLE), dt_us=30_000),
            eventcv.open(str(EXAMPLE), dt_ns=30_000_000),
        ]
        counts = {reader.n_slices for reader in readers}
        self.assertEqual(len(counts), 1, f"n_slices differed across units: {counts}")
        first = readers[0].slice(0).numpy()
        for reader in readers[1:]:
            np.testing.assert_array_equal(reader.slice(0).numpy(), first)

    def test_dt_getter_reports_any_unit(self):
        reader = eventcv.open(str(EXAMPLE), dt_ms=30)
        self.assertAlmostEqual(reader.dt(), 30.0)
        self.assertAlmostEqual(reader.dt("s"), 0.03)
        self.assertAlmostEqual(reader.dt("us"), 30_000.0)
        self.assertAlmostEqual(reader.dt("ns"), 30_000_000.0)
        # Reported spans follow the same unit vocabulary.
        self.assertAlmostEqual(reader.duration("ms"), reader.duration("s") * 1000)
        lo_ms, hi_ms = reader.time_span("ms")
        lo_us, hi_us = reader.time_span("us")
        self.assertAlmostEqual(lo_ms * 1000, lo_us)
        self.assertAlmostEqual(hi_ms * 1000, hi_us)

    def test_two_units_for_one_quantity_raise(self):
        for kwargs in (
            {"dt_ms": 30, "dt_us": 30_000},
            {"dt_s": 0.03, "dt_ns": 3e7},
            {"dt_ms": 30, "offset": 1, "offset_ms": 1},
        ):
            with self.assertRaises(ValueError, msg=f"{kwargs} must be rejected"):
                eventcv.open(str(EXAMPLE), **kwargs)

    def test_a_sub_microsecond_duration_raises_rather_than_rounding_up(self):
        # Timestamps are microseconds, so 100 ns cannot be honoured — and silently making it 1 us
        # would be ten times what was asked for.
        with self.assertRaises(ValueError):
            eventcv.open(str(EXAMPLE), dt_ns=100)
        with self.assertRaises(ValueError):
            eventcv.open(str(EXAMPLE), dt_ms=0)
        with self.assertRaises(ValueError):
            eventcv.open(str(EXAMPLE), dt_ms=-5)

    def test_an_unknown_unit_suffix_is_a_type_error(self):
        # `dt_minutes` is not a kwarg at all, so Python rejects it before eventcv sees it.
        with self.assertRaises(TypeError):
            eventcv.open(str(EXAMPLE), dt_minutes=1)


class SliceUnitTests(unittest.TestCase):
    def setUp(self):
        self.reader = eventcv.open(str(EXAMPLE), dt_ms=30)
        self.lo, _ = self.reader.time_span_ms

    def test_slice_bounds_take_any_unit(self):
        t0, t1 = self.lo, self.lo + 10
        expected = self.reader.slice(t0_ms=t0, t1_ms=t1).numpy()
        for kwargs in (
            {"t0_us": t0 * 1e3, "t1_us": t1 * 1e3},
            {"t0_s": t0 / 1e3, "t1_s": t1 / 1e3},
            {"t0_ns": t0 * 1e6, "t1_ns": t1 * 1e6},
        ):
            np.testing.assert_array_equal(self.reader.slice(**kwargs).numpy(), expected)

    def test_slice_rejects_an_index_together_with_bounds(self):
        with self.assertRaises(ValueError):
            self.reader.slice(0, t0_us=0)

    def test_windows_step_and_span_take_any_unit(self):
        by_ms = [len(w) for w in self.reader.windows(step_ms=20, span_ms=10)]
        by_us = [len(w) for w in self.reader.windows(step_us=20_000, span_us=10_000)]
        by_s = [len(w) for w in self.reader.windows(step_s=0.02, span_s=0.01)]
        self.assertEqual(by_ms, by_us)
        self.assertEqual(by_ms, by_s)

    def test_windows_rejects_two_units_for_a_step(self):
        with self.assertRaises(ValueError):
            self.reader.windows(step_ms=20, step_us=20_000)


class StreamOpUnitTests(unittest.TestCase):
    def setUp(self):
        self.stream = eventcv.load(str(EXAMPLE))

    def test_time_window_accepts_positional_microseconds_or_a_unit(self):
        t = self.stream.numpy()[:, 2]
        t0, t1 = int(t.min()), int(t.min()) + 5_000
        positional = self.stream.time_window(t0, t1).numpy()
        np.testing.assert_array_equal(
            self.stream.time_window(t0_us=t0, t1_us=t1).numpy(), positional
        )
        np.testing.assert_array_equal(
            self.stream.time_window(t0_ms=t0 / 1e3, t1_ms=t1 / 1e3).numpy(), positional
        )

    def test_time_window_rejects_positional_and_suffixed_together(self):
        with self.assertRaises(ValueError):
            self.stream.time_window(0, t0_us=0)

    def test_time_shift_accepts_any_unit_and_may_be_negative(self):
        shifted = self.stream.time_shift(dt_ms=1.5)
        np.testing.assert_array_equal(
            shifted.numpy()[:, 2], self.stream.numpy()[:, 2] + 1500
        )
        back = shifted.time_shift(dt_us=-1500)
        np.testing.assert_array_equal(back.numpy(), self.stream.numpy())

    def test_representation_spans_take_any_unit(self):
        by_ms = self.stream.voxel(bins=4, window_ms=20).numpy()
        np.testing.assert_array_equal(self.stream.voxel(bins=4, window_us=20_000).numpy(), by_ms)
        np.testing.assert_array_equal(self.stream.voxel(bins=4, window_s=0.02).numpy(), by_ms)
        tau = self.stream.tsurf(tau_ms=10).numpy()
        np.testing.assert_array_equal(self.stream.tsurf(tau_us=10_000).numpy(), tau)
        with self.assertRaises(ValueError):
            self.stream.tsurf(tau_ms=10, tau_us=10_000)


class TimeUnitNameTests(unittest.TestCase):
    def test_load_accepts_the_unit_synonyms(self):
        expected = eventcv.load(str(EXAMPLE), time_unit="us").numpy()
        for name in ("us", "US", "microseconds", "microsecond", "usec", " us "):
            np.testing.assert_array_equal(
                eventcv.load(str(EXAMPLE), time_unit=name).numpy(),
                expected,
                err_msg=f"time_unit={name!r}",
            )

    def test_an_unknown_unit_name_names_the_parameter(self):
        with self.assertRaisesRegex(ValueError, "time_unit"):
            eventcv.load(str(EXAMPLE), time_unit="furlongs")

    def test_reporting_getters_reject_an_unknown_unit(self):
        reader = eventcv.open(str(EXAMPLE), dt_ms=30)
        with self.assertRaises(ValueError):
            reader.duration("furlongs")


if __name__ == "__main__":
    unittest.main(verbosity=2)
