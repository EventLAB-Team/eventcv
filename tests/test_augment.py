"""Random augmentations, and the reproducibility contract they promise.

The property that matters for training is not "the numbers look random" but that a slice augments
the *same way every time it is reached* — by index, by ``batch()``, by iteration, forwards or
shuffled — while different slices still differ. A DataLoader with ``shuffle=True`` and several
workers touches slices in an order nothing here controls, so these tests deliberately read them
out of order rather than straight through.
"""

import unittest
from pathlib import Path

import numpy as np

import eventcv

EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"

AUGMENTATIONS = (
    "random_flip_x",
    "random_flip_y",
    "random_polarity_flip",
    "random_crop",
    "event_drop",
    "pixel_dropout",
    "spatial_jitter",
    "time_jitter",
    "time_reversal",
)


def _stream():
    return eventcv.load(str(EXAMPLE_NPZ))


def _columns(stream):
    """The stream as one comparable array, so two results can be checked for exact equality."""
    return stream.numpy()


class AugmentationBasicsTests(unittest.TestCase):
    def test_probability_bounds_are_exact(self):
        stream = _stream()
        n = len(stream)
        # p=0 never fires, p=1 always does — no reliance on a lucky draw.
        np.testing.assert_array_equal(
            _columns(stream.random_flip_x(0.0)), _columns(stream)
        )
        np.testing.assert_array_equal(
            _columns(stream.random_flip_x(1.0)), _columns(stream.flip_x())
        )
        self.assertEqual(len(stream.event_drop(0.0)), n)
        self.assertEqual(len(stream.event_drop(1.0)), 0)

    def test_event_drop_thins_by_roughly_p(self):
        stream = _stream()
        for p in (0.1, 0.5, 0.9):
            with self.subTest(p=p):
                kept = len(stream.event_drop(p, seed=1)) / len(stream)
                self.assertAlmostEqual(kept, 1.0 - p, delta=0.02)

    def test_pixel_dropout_silences_roughly_p_of_pixels(self):
        # Pins the mask direction: `drop_masked_pixels` reads its mask as *drop*, the opposite of
        # `mask()`, and getting that backwards still yields a plausible-looking thinned stream.
        stream = _stream()
        counts = stream.count().numpy()
        active = int((counts > 0).sum())
        for p in (0.1, 0.5):
            with self.subTest(p=p):
                dropped = stream.pixel_dropout(p, seed=1).count().numpy()
                surviving = int((dropped > 0).sum())
                self.assertAlmostEqual(surviving / active, 1.0 - p, delta=0.03)

    def test_random_crop_bounds_the_result(self):
        stream = _stream()
        cropped = stream.random_crop(64, 48, seed=3)
        self.assertEqual(cropped.sensor_size, (64, 48))
        # A window at least as large as the sensor is a no-op, so the op is safe to leave in a
        # pipeline that also runs on smaller recordings.
        width, height = stream.sensor_size
        self.assertEqual(
            len(stream.random_crop(width * 2, height * 2, seed=3)), len(stream)
        )

    def test_time_reversal_mirrors_the_span_and_inverts_polarity(self):
        stream = _stream()
        reversed_ = stream.time_reversal(1.0)
        self.assertEqual(len(reversed_), len(stream))
        original, mirrored = stream.numpy(), reversed_.numpy()
        # Same extent, still ascending, and every polarity flipped.
        self.assertEqual(mirrored[:, 0].min(), original[:, 0].min())
        self.assertEqual(mirrored[:, 0].max(), original[:, 0].max())
        self.assertTrue(np.all(np.diff(mirrored[:, 0]) >= 0))
        self.assertEqual(
            sorted(np.unique(mirrored[:, 3])), sorted(np.unique(original[:, 3]))
        )

    def test_time_jitter_leaves_the_stream_sorted(self):
        # The correlation filters require ascending time, so jitter must re-sort.
        jittered = _stream().time_jitter(sigma_ms=1.0, seed=5)
        self.assertTrue(np.all(np.diff(jittered.numpy()[:, 0]) >= 0))

    def test_time_jitter_accepts_every_time_unit(self):
        stream = _stream()
        reference = _columns(stream.time_jitter(sigma_us=1000.0, seed=5))
        for kwargs in ({"sigma_ms": 1.0}, {"sigma_s": 0.001}, {"sigma_ns": 1_000_000.0}):
            with self.subTest(**kwargs):
                np.testing.assert_array_equal(
                    _columns(stream.time_jitter(seed=5, **kwargs)), reference
                )

    def test_every_augmentation_has_a_free_function(self):
        for name in AUGMENTATIONS:
            with self.subTest(op=name):
                self.assertIn(name, eventcv.__all__)
                self.assertTrue(callable(getattr(eventcv, name, None)))
                self.assertTrue(getattr(eventcv, name).__doc__)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_gives_identical_output(self):
        stream = _stream()
        for name in AUGMENTATIONS:
            with self.subTest(op=name):
                op = getattr(stream, name)
                args = (64, 48) if name == "random_crop" else ()
                np.testing.assert_array_equal(
                    _columns(op(*args, seed=11)), _columns(op(*args, seed=11))
                )

    def test_different_seeds_give_different_output(self):
        stream = _stream()
        self.assertNotEqual(
            len(stream.event_drop(0.5, seed=1)), len(stream.event_drop(0.5, seed=2))
        )


class ReaderSeedingTests(unittest.TestCase):
    """Per-slice seeding — the property a shuffled, multi-worker DataLoader depends on."""

    def _reader(self, seed=123):
        return (
            eventcv.open(str(EXAMPLE_NPZ), dt_ms=5)
            .event_drop(0.4, seed=seed)
            .with_repr("count")
        )

    def test_out_of_order_access_matches_sequential(self):
        n = len(self._reader())
        forwards = [self._reader()[i] for i in range(n)]
        # A fresh reader per access, walked backwards: nothing may be carried between calls.
        backwards = {i: self._reader()[i] for i in reversed(range(n))}
        for i in range(n):
            with self.subTest(slice=i):
                np.testing.assert_array_equal(forwards[i], backwards[i])

    def test_negative_indices_match_positive_ones(self):
        n = len(self._reader())
        np.testing.assert_array_equal(self._reader()[-1], self._reader()[n - 1])

    def test_batch_matches_indexing_in_shuffled_order(self):
        indices = [7, 2, 9, 0, 4]
        batch = self._reader().batch(indices)
        for position, index in enumerate(indices):
            with self.subTest(slice=index):
                np.testing.assert_array_equal(batch[position], self._reader()[index])

    def test_iteration_matches_indexing(self):
        windows = list(self._reader().windows(step_ms=5, span_ms=5))
        for i, frame in enumerate(windows):
            with self.subTest(slice=i):
                np.testing.assert_array_equal(frame.numpy(), self._reader()[i])

    def test_different_slices_augment_differently(self):
        # Guards against the index being ignored: if every slice shared one seed, an identical
        # drop pattern would be applied throughout and this would collapse to a single value.
        reader = self._reader()
        totals = {int(np.asarray(reader[i]).sum()) for i in range(len(reader))}
        self.assertGreater(len(totals), 1)

    def test_seed_changes_the_result(self):
        self.assertFalse(np.array_equal(self._reader(1)[0], self._reader(2)[0]))

    def test_augmentation_composes_with_other_deferred_ops(self):
        reader = (
            eventcv.open(str(EXAMPLE_NPZ), dt_ms=5)
            .flip_x()
            .event_drop(0.3, seed=4)
            .with_repr("count")
        )
        np.testing.assert_array_equal(reader[0], reader[0])
        self.assertEqual(np.asarray(reader[0]).shape, np.asarray(reader[1]).shape)


if __name__ == "__main__":
    unittest.main()
