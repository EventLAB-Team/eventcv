"""Events → intensity video, and the recurrent-model plumbing it needs.

The models here are built in-process from tiny ONNX graphs rather than committed as fixtures: an
accumulator makes recurrence *observable* (call it n times and the answer must be n), which a real
E2VID export could not do as legibly.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import eventcv

try:
    import onnx
    from onnx import TensorProto, helper

    HAVE_ONNX_BUILDER = True
except ImportError:  # pragma: no cover - depends on the dev environment
    HAVE_ONNX_BUILDER = False

HAVE_ONNX_RUNTIME = "onnx" in getattr(eventcv._rust, "__features__", ())
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
EXAMPLE_NPZ = Path(__file__).resolve().parent.parent / "data" / "test" / "example.npz"


def _write_reducer(path, bins, height, width):
    """A stand-in reconstruction model: sum the voxel bins, squash to [0, 1].

    Not E2VID, but the same tensor contract — `[1, C, H, W]` in, `[1, 1, H, W]` out — which is what
    the plumbing has to get right.
    """
    graph = helper.make_graph(
        [
            helper.make_node("ReduceSum", ["voxel", "axes"], ["summed"], keepdims=1),
            helper.make_node("Sigmoid", ["summed"], ["image"]),
        ],
        "reducer",
        [helper.make_tensor_value_info("voxel", TensorProto.FLOAT, [1, bins, height, width])],
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, height, width])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _write_accumulator(path, size=2):
    """A recurrent graph: `new_state = state + data`, `image = new_state`."""
    shape = [1, 1, size, size]
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["data", "state"], ["new_state"]),
            helper.make_node("Identity", ["new_state"], ["image"]),
        ],
        "accumulator",
        [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, shape),
        ],
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("new_state", TensorProto.FLOAT, shape),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


@unittest.skipUnless(HAVE_ONNX_RUNTIME, "this build has no onnx feature")
@unittest.skipUnless(HAVE_ONNX_BUILDER, "the onnx package is needed to write the test models")
class ReconstructTests(unittest.TestCase):
    BINS = 5

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        stream = eventcv.load(str(EXAMPLE_NPZ))
        self.width, self.height = stream.sensor_size
        self.model_path = _write_reducer(
            self.dir / "reducer.onnx", self.BINS, self.height, self.width
        )

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _reader(self):
        return eventcv.open(str(EXAMPLE_NPZ), dt_ms=10).with_repr("voxel", bins=self.BINS)

    def test_reconstruction_writes_every_frame(self):
        target = self.dir / "out.gif"
        expected = len(self._reader())
        written = eventcv.reconstruct(
            self._reader(), eventcv.Model(str(self.model_path)), str(target)
        )
        self.assertEqual(written, expected)
        self.assertEqual(target.read_bytes()[:6], b"GIF89a")

    def test_max_frames_stops_early(self):
        target = self.dir / "short.gif"
        self.assertEqual(
            eventcv.reconstruct(
                self._reader(), eventcv.Model(str(self.model_path)), str(target), max_frames=3
            ),
            3,
        )

    def test_a_representation_is_required(self):
        with self.assertRaises(ValueError) as caught:
            eventcv.reconstruct(
                eventcv.open(str(EXAMPLE_NPZ), dt_ms=10),
                eventcv.Model(str(self.model_path)),
                str(self.dir / "x.gif"),
            )
        self.assertIn("representation", str(caught.exception))

    def test_an_unknown_extension_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            eventcv.reconstruct(
                self._reader(), eventcv.Model(str(self.model_path)), str(self.dir / "x.avi")
            )
        self.assertIn(".gif", str(caught.exception))

    def test_a_mismatched_model_is_reported(self):
        # A model expecting a different number of bins must fail loudly, not silently reshape.
        wrong = _write_reducer(self.dir / "wrong.onnx", self.BINS + 2, self.height, self.width)
        with self.assertRaises(ValueError):
            eventcv.reconstruct(
                self._reader(), eventcv.Model(str(wrong)), str(self.dir / "y.gif")
            )

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
    def test_the_full_round_trip(self):
        # video -> events -> reconstruction. The end-to-end path both directions have to support.
        source = self.dir / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=64x48:rate=30:duration=1", "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )
        events = eventcv.simulate(str(source))
        self.assertGreater(len(events), 0)

        path = self.dir / "events.npz"
        eventcv.save(events, str(path))
        reader = eventcv.open(str(path), dt_ms=33).with_repr("voxel", bins=self.BINS)
        model = _write_reducer(self.dir / "rt.onnx", self.BINS, 48, 64)

        target = self.dir / "reconstructed.mp4"
        frames = eventcv.reconstruct(reader, eventcv.Model(str(model)), str(target))
        self.assertGreater(frames, 0)
        self.assertIn(b"ftyp", target.read_bytes()[:64])


@unittest.skipUnless(HAVE_ONNX_RUNTIME, "this build has no onnx feature")
@unittest.skipUnless(HAVE_ONNX_BUILDER, "the onnx package is needed to write the test models")
class RecurrentModelTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = _write_accumulator(self.dir / "accumulator.onnx")
        self.model = eventcv.Model(str(self.path))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_run_named_binds_several_inputs(self):
        outputs = self.model.run_named(
            {
                "data": np.ones((1, 1, 2, 2), "float32"),
                "state": np.full((1, 1, 2, 2), 5.0, "float32"),
            }
        )
        self.assertEqual(sorted(outputs), ["image", "new_state"])
        np.testing.assert_allclose(outputs["image"], 6.0)

    def test_run_named_rejects_an_unknown_input(self):
        with self.assertRaises(ValueError) as caught:
            self.model.run_named({"nope": np.ones((1, 1, 2, 2), "float32")})
        self.assertIn("nope", str(caught.exception))

    def test_state_carries_between_calls(self):
        # The property that makes a model recurrent: feed ones repeatedly and the accumulator
        # must count. Without state feedback every call would return 1.
        stateful = eventcv.StatefulModel(self.model, state_map={"new_state": "state"})
        ones = np.ones((1, 2, 2), "float32")
        values = [float(stateful(ones)[0, 0, 0, 0]) for _ in range(4)]
        self.assertEqual(values, [1.0, 2.0, 3.0, 4.0])

    def test_reset_starts_a_fresh_sequence(self):
        stateful = eventcv.StatefulModel(self.model, state_map={"new_state": "state"})
        ones = np.ones((1, 2, 2), "float32")
        for _ in range(3):
            stateful(ones)
        stateful.reset()
        self.assertEqual(float(stateful(ones)[0, 0, 0, 0]), 1.0)

    def test_a_missing_batch_axis_is_added(self):
        stateful = eventcv.StatefulModel(self.model, state_map={"new_state": "state"})
        self.assertEqual(stateful(np.ones((1, 2, 2), "float32")).shape, (1, 1, 2, 2))

    def test_state_map_is_validated(self):
        for bad in ({"nope": "state"}, {"new_state": "nope"}):
            with self.subTest(state_map=bad):
                with self.assertRaises(ValueError):
                    eventcv.StatefulModel(self.model, state_map=bad)


if __name__ == "__main__":
    unittest.main()
