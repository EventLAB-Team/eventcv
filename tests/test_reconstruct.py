"""Events → intensity video, and the recurrent-model plumbing it needs.

The models here are built in-process from tiny ONNX graphs rather than committed as fixtures: an
accumulator makes recurrence *observable* (call it n times and the answer must be n), which a real
E2VID export could not do as legibly.
"""

import hashlib
import http.server
import json
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import eventcv
import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper

    HAVE_ONNX_BUILDER = True
except ImportError:  # pragma: no cover - depends on the dev environment
    HAVE_ONNX_BUILDER = False

HAVE_ONNX_RUNTIME = "onnx" in getattr(eventcv._rust, "__features__", ())
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_NN = "nn" in getattr(eventcv._rust, "__features__", ())
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
            eventcv.reconstruct(self._reader(), eventcv.Model(str(wrong)), str(self.dir / "y.gif"))

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg is not installed")
    def test_the_full_round_trip(self):
        # video -> events -> reconstruction. The end-to-end path both directions have to support.
        source = self.dir / "clip.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30:duration=1",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
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


def _write_tiny_e2vid(path, bins=5, base=2, encoders=2, resblocks=1, seed=0):
    """A randomly initialised E2VID small enough to run in a test, in the converter's file format.

    Random weights cannot say whether a reconstruction is *good* — the parity test against the
    reference does that — but they exercise every shape, the padding and the recurrent state.
    """
    rng = np.random.default_rng(seed)
    tensors = {}

    def conv(name, out_ch, in_ch, k, transposed=False):
        shape = (in_ch, out_ch, k, k) if transposed else (out_ch, in_ch, k, k)
        tensors[f"{name}.weight"] = rng.normal(0, 0.3, shape).astype("<f4")
        tensors[f"{name}.bias"] = rng.normal(0, 0.1, out_ch).astype("<f4")

    conv("head", base, bins, 5)
    channels = base
    for i in range(encoders):
        conv(f"encoders.{i}.conv", channels * 2, channels, 5)
        channels *= 2
        conv(f"encoders.{i}.gates", 4 * channels, 2 * channels, 3)
    for i in range(resblocks):
        conv(f"resblocks.{i}.conv1", channels, channels, 3)
        conv(f"resblocks.{i}.conv2", channels, channels, 3)
    for i in range(encoders):
        conv(f"decoders.{i}", channels // 2, channels, 5, transposed=True)
        channels //= 2
    conv("pred", 1, base, 1)
    kernel = np.outer(*[np.array([1, 4, 6, 4, 1], "<f4")] * 2)
    tensors["unsharp.kernel"] = (kernel / kernel.sum()).reshape(1, 1, 5, 5).astype("<f4")
    metadata = {
        "format": "eventcv-e2vid",
        "version": "1",
        "num_bins": str(bins),
        "num_encoders": str(encoders),
        "num_residual_blocks": str(resblocks),
        "events_per_pixel": "0.35",
        "unsharp_amount": "0.3",
    }
    header, offset = {"__metadata__": metadata}, 0
    for name, array in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded)
        for array in tensors.values():
            f.write(array.tobytes())
    return path


def _random_stream(width, height, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    events = np.stack(
        [
            rng.integers(0, width, n),
            rng.integers(0, height, n),
            np.sort(rng.integers(0, 100_000, n)),
            rng.integers(0, 2, n),
        ],
        axis=1,
    )
    return eventcv.from_numpy(events, sensor_size=(width, height), time_unit="us")


@unittest.skipUnless(HAVE_NN, "this build has no nn feature")
class E2VIDTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.weights = _write_tiny_e2vid(self.dir / "tiny.safetensors")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _model(self, device="cpu", precision="half"):
        return eventcv.E2VID(self.weights, device=device, precision=precision)

    def test_frames_match_the_sensor_at_sizes_that_need_padding(self):
        # 37x29 is not a multiple of 4 either way, so this goes through reflection padding and back.
        image = self._model()(_random_stream(37, 29))
        self.assertEqual(image.shape, (29, 37))
        self.assertEqual(image.dtype, np.float32)
        self.assertTrue(((image >= 0) & (image <= 1)).all())

    def test_state_carries_and_reset_forgets_it(self):
        model, window = self._model(), _random_stream(32, 24)
        first = model(window)
        second = model(window)
        self.assertFalse(np.array_equal(first, second), "the second call ignored the state")
        model.reset()
        np.testing.assert_array_equal(model(window), first)

    def test_an_empty_window_still_makes_a_frame(self):
        empty = eventcv.from_numpy(np.zeros((0, 4), "int64"), sensor_size=(32, 24), time_unit="us")
        image = self._model()(empty)
        self.assertEqual(image.shape, (24, 32))
        self.assertTrue(np.isfinite(image).all())

    def test_a_new_sensor_size_mid_sequence_is_refused(self):
        model = self._model()
        model(_random_stream(32, 24))
        with self.assertRaises(ValueError) as caught:
            model(_random_stream(40, 24))
        self.assertIn("reset()", str(caught.exception))

    def test_reconstruct_returns_the_frames_it_would_write(self):
        reader = eventcv.open(_random_stream(32, 24), max_events=1000)
        frames = eventcv.reconstruct(reader, self._model())
        self.assertEqual(frames.shape, (len(reader), 24, 32))
        model = self._model()
        expected = np.stack([model(reader[i]) for i in range(len(reader))])
        np.testing.assert_array_equal(frames, expected)

        target = self.dir / "out.gif"
        written = eventcv.reconstruct(reader, self._model(), str(target), max_frames=2)
        self.assertEqual(written, 2)
        self.assertEqual(target.read_bytes()[:6], b"GIF89a")

    def test_unsliced_sources_get_the_trained_window_size(self):
        # 0.35 events per pixel on 32x24 is 269 events a window.
        stream = _random_stream(32, 24)
        frames = eventcv.reconstruct(stream, self._model())
        self.assertEqual(len(frames), -(-len(stream) // round(0.35 * 32 * 24)))

    def test_an_onnx_model_still_needs_a_reader(self):
        with self.assertRaises(TypeError):
            eventcv.reconstruct(_random_stream(32, 24), object())

    def test_a_file_that_is_not_e2vid_weights_is_rejected(self):
        bogus = self.dir / "bogus.safetensors"
        bogus.write_bytes(b"not a safetensors file")
        with self.assertRaises(ValueError):
            eventcv.E2VID(bogus, device="cpu")

    @unittest.skipUnless(eventcv.gpu_available(), "no GPU here")
    def test_the_gpu_agrees_with_the_cpu(self):
        # The GPU convolves in half precision (the tensor cores); ~1e-3 apart is that, not a bug.
        window = _random_stream(37, 29)
        cpu, gpu = self._model("cpu"), self._model("gpu")
        for _ in range(3):  # through the recurrent state, not just one frame
            np.testing.assert_allclose(gpu(window), cpu(window), atol=5e-3)

    @unittest.skipUnless(eventcv.gpu_available(), "no GPU here")
    def test_single_precision_on_the_gpu_is_the_cpu(self):
        window = _random_stream(37, 29)
        cpu, gpu = self._model("cpu"), self._model("gpu", precision="single")
        self.assertEqual(gpu.precision, "single")
        for _ in range(3):
            np.testing.assert_allclose(gpu(window), cpu(window), atol=1e-4)

    def test_an_unknown_precision_is_refused(self):
        with self.assertRaises(ValueError):
            self._model(precision="double")


class HubTests(unittest.TestCase):
    PAYLOAD = b"pretend weights"

    def setUp(self):
        from eventcv import hub

        self.hub = hub
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "served" / "org" / "m" / "resolve" / "rev").mkdir(parents=True)
        (self.dir / "served" / "org" / "m" / "resolve" / "rev" / "w.bin").write_bytes(self.PAYLOAD)
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(  # noqa: E731
            *a, directory=str(self.dir / "served"), **k
        )
        handler.log_message = lambda *a: None
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.entry = {
            "repo": "org/m",
            "revision": "rev",
            "file": "w.bin",
            "sha256": hashlib.sha256(self.PAYLOAD).hexdigest(),
            "license": "test",
        }
        self.env = mock.patch.dict(
            os.environ,
            {
                "EVENTCV_CACHE": str(self.dir / "cache"),
                "EVENTCV_HUB_URL": f"http://127.0.0.1:{self.server.server_address[1]}",
            },
        )
        self.env.start()
        os.environ.pop("EVENTCV_OFFLINE", None)

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fetch(self, entry):
        with mock.patch.dict(self.hub.MODELS, {"m": entry}), mock.patch("sys.stderr"):
            return self.hub.fetch("m")

    def test_download_is_verified_cached_and_reused_offline(self):
        path = self._fetch(self.entry)
        self.assertEqual(path.read_bytes(), self.PAYLOAD)
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(self._fetch(self.entry), path)

    def test_a_checksum_mismatch_caches_nothing(self):
        with self.assertRaises(RuntimeError):
            self._fetch(dict(self.entry, sha256="0" * 64))
        cached = self.dir / "cache" / "hub" / "org" / "m" / "rev"
        self.assertEqual(list(cached.iterdir()), [])

    def test_offline_mode_refuses_to_download(self):
        with mock.patch.dict(os.environ, {"EVENTCV_OFFLINE": "1"}):
            with self.assertRaises(RuntimeError):
                self._fetch(self.entry)

    def test_unknown_models_are_named(self):
        with self.assertRaises(ValueError):
            self.hub.fetch("nope")


REFERENCE = os.environ.get("EVENTCV_E2VID_REFERENCE")


@unittest.skipUnless(HAVE_NN, "this build has no nn feature")
@unittest.skipUnless(REFERENCE, "set EVENTCV_E2VID_REFERENCE to scripts/convert_e2vid.py's --out")
class E2VIDParityTests(unittest.TestCase):
    """The real weights against rpg_e2vid's own reconstruction of the same windows."""

    def _check(self, device, atol, precision="half"):
        reference = np.load(Path(REFERENCE) / "reference.npz")
        expected = reference["frames"]
        reader = eventcv.open(str(EXAMPLE_NPZ), max_events=int(reference["max_events"]))
        model = eventcv.E2VID(
            Path(REFERENCE) / "e2vid.safetensors", device=device, precision=precision
        )
        frames = eventcv.reconstruct(reader, model, max_frames=len(expected))
        np.testing.assert_allclose(frames, expected, atol=atol)

    def test_cpu_matches_the_reference(self):
        self._check("cpu", 1e-4)

    @unittest.skipUnless(eventcv.gpu_available(), "no GPU here")
    def test_gpu_matches_the_reference(self):
        # Half-precision convolutions: 2.1e-3 at most over these 20 windows on an RTX 2080.
        self._check("gpu", 5e-3)

    @unittest.skipUnless(eventcv.gpu_available(), "no GPU here")
    def test_gpu_in_single_precision_matches_the_reference(self):
        self._check("gpu", 1e-4, precision="single")


if __name__ == "__main__":
    unittest.main()
