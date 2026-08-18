"""The optional GPU backend: same answers as the CPU, and the CPU stays the default.

`device=` selects where a representation runs; `eventcv.set_device` moves the session default. The
CPU implementations are the reference, so every test here asserts the GPU against them rather than
against a stored expectation — the point of the backend is that it changes the speed and nothing
else.

These skip when no adapter is available (a build without `--features gpu`, or a machine without a
GPU). Set ``EVENTCV_REQUIRE_GPU=1`` to make that a failure instead, so a machine that should have
one cannot quietly stop testing it.
"""

import os
import unittest

import numpy as np

import eventcv

#: Kernels whose arithmetic is integer, so the GPU's answer is not close to the CPU's — it is the
#: same numbers. Order cannot matter to an integer sum or a minimum.
EXACT = ["count", "polarity", "countmask", "tsurf"]

#: Kernels that accumulate in Q16.16 fixed point. The quantisation is per event and bounded, but it
#: accumulates over the hundreds of events that share a cell, so this is what the total may differ
#: by — several orders below the values themselves.
TOLERANCE = 1e-3


def _gpu_ready():
    if eventcv.gpu_available():
        return True
    if os.environ.get("EVENTCV_REQUIRE_GPU"):
        raise AssertionError("EVENTCV_REQUIRE_GPU is set but no GPU adapter was found")
    return False


@unittest.skipUnless(_gpu_ready(), "no GPU adapter available")
class GpuKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream = eventcv.load("data/test/example.npz")

    def _both(self, name, **kwargs):
        method = getattr(self.stream, name)
        return (
            method(device="cpu", **kwargs).numpy().astype("float64"),
            method(device="gpu", **kwargs).numpy().astype("float64"),
        )

    def test_integer_kernels_are_bit_identical(self):
        for name, kwargs in [
            ("count", {}),
            ("count", {"normalize": True}),
            ("countmask", {}),
            ("tsurf", {"tau_ms": 30}),
        ]:
            with self.subTest(kernel=name, **kwargs):
                cpu, gpu = self._both(name, **kwargs)
                np.testing.assert_array_equal(gpu, cpu)

    def test_accumulating_kernels_match_within_the_fixed_point_quantum(self):
        for name, kwargs in [("voxel", {"bins": 5, "window_ms": 30}), ("atsurf", {"tau_ms": 30})]:
            with self.subTest(kernel=name, **kwargs):
                cpu, gpu = self._both(name, **kwargs)
                self.assertEqual(cpu.shape, gpu.shape)
                np.testing.assert_allclose(gpu, cpu, atol=TOLERANCE)

    def test_a_kernel_gives_the_same_answer_every_run(self):
        """Integer accumulation is what buys this: a float sum would drift with the scheduling."""
        first = self.stream.voxel(bins=5, window_ms=30, device="gpu").numpy()
        for _ in range(3):
            np.testing.assert_array_equal(
                self.stream.voxel(bins=5, window_ms=30, device="gpu").numpy(), first
            )

    def test_an_empty_stream_is_not_a_special_case(self):
        """A window with nothing in it is ordinary during a live capture, and a kernel dispatched
        over zero events must still produce the frame full of zeros the CPU produces."""
        empty = eventcv.from_numpy(
            np.zeros((0, 4), dtype=np.int64), sensor_size=(64, 48), time_unit="us"
        )
        np.testing.assert_array_equal(
            empty.count(device="gpu").numpy(), empty.count(device="cpu").numpy()
        )
        np.testing.assert_array_equal(
            empty.voxel(bins=3, device="gpu").numpy(), empty.voxel(bins=3, device="cpu").numpy()
        )


class DeviceSelectionTests(unittest.TestCase):
    """The toggle itself, which works whether or not a GPU is present."""

    def setUp(self):
        self.stream = eventcv.load("data/test/example.npz")
        self.addCleanup(eventcv.set_device, "cpu")

    def test_the_default_is_the_cpu(self):
        self.assertEqual(eventcv.get_device(), "cpu")

    def test_an_unknown_device_name_is_refused(self):
        with self.assertRaises(ValueError) as raised:
            self.stream.count(device="gpi")
        self.assertIn("cpu", str(raised.exception))
        with self.assertRaises(ValueError):
            eventcv.set_device("tpu")

    def test_asking_for_an_absent_gpu_says_so_rather_than_falling_back(self):
        """A silent fall back to the CPU is the one behaviour that must not happen: it turns
        'my GPU is not being used' into something you can only discover by timing a benchmark."""
        if eventcv.gpu_available():
            self.skipTest("this machine has a GPU, so the absent-GPU path cannot be exercised")
        with self.assertRaises(Exception) as raised:
            self.stream.count(device="gpu")
        self.assertIn("gpu", str(raised.exception).lower())

    @unittest.skipUnless(_gpu_ready(), "no GPU adapter available")
    def test_the_session_default_reaches_calls_that_name_no_device(self):
        expected = self.stream.count(device="gpu").numpy()
        eventcv.set_device("gpu")
        self.assertEqual(eventcv.get_device(), "gpu")
        np.testing.assert_array_equal(self.stream.count().numpy(), expected)

    @unittest.skipUnless(_gpu_ready(), "no GPU adapter available")
    def test_a_representation_without_a_kernel_still_works_on_the_gpu(self):
        """`device="gpu"` is a request, not an assertion that every step has a kernel — a pipeline
        that sets it once should not have to know which representations were ported."""
        eventcv.set_device("gpu")
        np.testing.assert_array_equal(
            self.stream.tencode().numpy(),
            eventcv.load("data/test/example.npz").tencode().numpy(),
        )


if __name__ == "__main__":
    unittest.main()
