"""ONNX inference.

The model used here is built in-process from a two-node graph (multiply, then add) rather than
committed as a fixture, so the expected output is arithmetic anyone can check by eye and there is
no binary blob in the repository. `onnx` is only needed to *write* it; eventcv reads it with its
own runtime.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

import eventcv

try:
    import onnx
    from onnx import TensorProto, helper

    HAVE_ONNX_BUILDER = True
except ImportError:  # pragma: no cover - depends on the dev environment
    HAVE_ONNX_BUILDER = False

HAVE_ONNX_RUNTIME = "onnx" in getattr(eventcv._rust, "__features__", ())
EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"


def _write_scale_and_shift(path, shape=(1, 1, 4, 4)):
    """An ONNX graph computing `x * 2 + 1`, so correctness is checkable without a framework."""
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input", "scale"], ["scaled"]),
            helper.make_node("Add", ["scaled", "bias"], ["output"]),
        ],
        "scale_and_shift",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [2.0]),
            helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@unittest.skipUnless(HAVE_ONNX_RUNTIME, "this build has no onnx feature")
@unittest.skipUnless(HAVE_ONNX_BUILDER, "the onnx package is needed to write the test model")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._dir = TemporaryDirectory()
        cls.path = _write_scale_and_shift(Path(cls._dir.name) / "scale_and_shift.onnx")

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def test_inference_is_numerically_correct(self):
        model = eventcv.Model(str(self.path))
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        np.testing.assert_allclose(model(x), x * 2 + 1)

    def test_ports_report_the_graphs_own_shapes(self):
        model = eventcv.Model(str(self.path))
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(model.inputs[0]["name"], "input")
        self.assertEqual(model.inputs[0]["shape"], [1, 1, 4, 4])
        self.assertEqual(model.inputs[0]["dtype"], "float32")
        self.assertEqual(model.outputs[0]["name"], "output")

    def test_a_missing_batch_axis_is_added(self):
        # A representation is [C, H, W]; a trained network wants [N, C, H, W]. Callers should not
        # have to write `arr[None]` at every site.
        model = eventcv.Model(str(self.path))
        without_batch = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        result = model(without_batch)
        self.assertEqual(result.shape, (1, 1, 4, 4))
        np.testing.assert_allclose(result, without_batch[None] * 2 + 1)

    def test_non_float_input_is_coerced(self):
        model = eventcv.Model(str(self.path))
        x = np.arange(16, dtype=np.int64).reshape(1, 1, 4, 4)
        np.testing.assert_allclose(model(x), x.astype(np.float32) * 2 + 1)

    def test_an_event_frame_can_be_fed_directly(self):
        # The pairing that makes this worth having: a representation is already the tensor.
        frame = eventcv.load(str(EXAMPLE_NPZ)).count()
        channels, height, width = frame.numpy().shape
        path = _write_scale_and_shift(
            Path(self._dir.name) / "frame.onnx", (1, channels, height, width)
        )
        model = eventcv.Model(str(path))
        np.testing.assert_allclose(model(frame), frame.numpy()[None] * 2 + 1)

    def test_repeated_calls_are_stable(self):
        model = eventcv.Model(str(self.path))
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        np.testing.assert_array_equal(model(x), model(x))

    def test_a_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            eventcv.Model("/nonexistent/model.onnx")

    def test_a_non_model_file_raises_a_value_error(self):
        bogus = Path(self._dir.name) / "not-a-model.onnx"
        bogus.write_bytes(b"definitely not a protobuf")
        with self.assertRaises(ValueError):
            eventcv.Model(str(bogus))

    def test_a_wrong_shape_is_reported(self):
        model = eventcv.Model(str(self.path))
        with self.assertRaises(ValueError):
            model(np.zeros((1, 1, 8, 8), dtype=np.float32))

    def test_a_non_array_input_is_reported(self):
        model = eventcv.Model(str(self.path))
        with self.assertRaises(ValueError):
            model("not an array")


@unittest.skipUnless(HAVE_ONNX_RUNTIME, "needs a build with the onnx feature")
class RuntimeDiscoveryTests(unittest.TestCase):
    """ONNX Runtime is opened at run time, not linked in, so finding it is part of the job.

    The wheels carry a copy, but a source build, a conda environment and an explicit
    `ORT_DYLIB_PATH` are all supported ways to supply one — which makes the search order a
    contract, not an implementation detail.
    """

    def setUp(self):
        # `configure()` caches what it chose, and some tests here call it; put the module back
        # the way it was so a later test's `describe()` still reports the truth.
        resolved = eventcv._ort._resolved
        self.addCleanup(setattr, eventcv._ort, "_resolved", resolved)

    def test_a_runtime_is_found_and_its_origin_named(self):
        found = eventcv._ort.find_runtime()
        self.assertIsNotNone(found, "no ONNX Runtime anywhere — see scripts/fetch_onnxruntime.py")
        path, origin = found
        self.assertTrue(Path(path).is_file())
        self.assertIn(origin, [name for name, _ in eventcv._ort._SOURCES])

    def test_the_search_order_is_the_documented_one(self):
        # Ordering is what makes the behaviour predictable: an explicit choice beats a runtime the
        # process already holds, which beats the bundled copy, which beats the environment's.
        self.assertEqual(
            [name for name, _ in eventcv._ort._SOURCES],
            [
                "ORT_DYLIB_PATH",
                "imported onnxruntime",
                "bundled",
                "environment",
                "onnxruntime package",
            ],
        )

    def test_the_first_source_with_an_answer_wins(self):
        sources = (
            ("nothing here", lambda: None),
            ("second", lambda: "/second/libonnxruntime.so"),
            ("third", lambda: "/third/libonnxruntime.so"),
        )
        with mock.patch.object(eventcv._ort, "_SOURCES", sources):
            self.assertEqual(
                eventcv._ort.find_runtime(), ("/second/libonnxruntime.so", "second")
            )

    def test_the_bundled_copy_is_what_a_wheel_uses(self):
        bundled = eventcv._ort._bundled()
        if bundled is None:
            self.skipTest("source build without scripts/fetch_onnxruntime.py")
        # Neither an explicit path nor an already-imported onnxruntime, which both outrank it.
        with mock.patch.dict(os.environ), mock.patch.dict(sys.modules):
            os.environ.pop("ORT_DYLIB_PATH", None)
            sys.modules.pop("onnxruntime", None)
            self.assertEqual(eventcv._ort.find_runtime(), (bundled, "bundled"))

    def test_an_explicit_path_is_never_overridden(self):
        chosen = "/somewhere/of/my/own/libonnxruntime.so"
        with mock.patch.dict(os.environ, {"ORT_DYLIB_PATH": chosen}):
            eventcv._ort.configure()
            self.assertEqual(os.environ["ORT_DYLIB_PATH"], chosen)
            self.assertEqual(eventcv._ort.describe(), f"{chosen} (ORT_DYLIB_PATH)")

    def test_a_missing_runtime_is_reported_with_the_fix(self):
        # In a subprocess because the probe is cached for the life of the process: by the time
        # this runs, the tests above have already loaded a runtime successfully. The subprocess
        # also proves the failure is an exception rather than the panic ort raises on its own.
        result = self._load_with_runtime("/nonexistent/libonnxruntime.so")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RuntimeError", result.stderr)
        self.assertIn("eventcv[onnx]", result.stderr)

    def test_a_library_that_is_not_onnx_runtime_says_so(self):
        # The extension itself: a perfectly good shared library with none of the right symbols,
        # which is what a wrong `ORT_DYLIB_PATH` usually points at.
        result = self._load_with_runtime(eventcv._rust.__file__)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not ONNX Runtime", result.stderr)

    def _load_with_runtime(self, path):
        return subprocess.run(
            [sys.executable, "-c", "import eventcv; eventcv.Model('anything.onnx')"],
            env={**os.environ, "ORT_DYLIB_PATH": str(path)},
            capture_output=True,
            text=True,
        )


@unittest.skipIf(HAVE_ONNX_RUNTIME, "this build has the onnx feature")
class MissingFeatureTests(unittest.TestCase):
    def test_the_error_names_the_fix(self):
        # A source build without `--features onnx` must say so, not raise AttributeError.
        with self.assertRaises(RuntimeError) as caught:
            eventcv.Model("anything.onnx")
        self.assertIn("onnx", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
