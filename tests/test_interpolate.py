"""Learned frame interpolation before simulation.

``simulate(..., interpolate=...)`` runs an ONNX frame interpolator over the source frames and hands
the results to the simulator as ordinary frames at proportional timestamps — the v2e/Super-SloMo
arrangement, so the pixel model itself is untouched. These tests drive that path with graphs built
here rather than with real RIFE weights: what has to be checked is that eventcv feeds the model the
pair the way the model declares it wants it, uses the ``timestep`` input when there is one, and puts
the frames back in the right place in time. A trained network is not needed to check any of that.
"""

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


def _blend_with_timestep(path, height, width):
    """`a + (b - a) * t` over two 3-channel inputs and a scalar timestep — a RIFE v4-shaped graph.

    Linear, so the events it produces can be compared against the run with no interpolation at all:
    a straight-line interpolator is exactly what the simulator already assumes between frames.
    """
    image = lambda name: helper.make_tensor_value_info(  # noqa: E731
        name, TensorProto.FLOAT, [1, 3, height, width]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["second", "first"], ["delta"]),
            helper.make_node("Mul", ["delta", "timestep"], ["scaled"]),
            helper.make_node("Add", ["first", "scaled"], ["output"]),
        ],
        "blend",
        [image("first"), image("second"), helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])],
        [image("output")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _midpoint_stacked(path, height, width):
    """The mean of a stacked six-channel input, with no timestep — the older export shape.

    Slices the two halves back apart and averages them, so it can only ever produce a midpoint. That
    is the constraint eventcv has to detect and work within by bisecting.
    """
    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["pair", "lo0", "hi0", "axes"], ["first"]),
            helper.make_node("Slice", ["pair", "lo1", "hi1", "axes"], ["second"]),
            helper.make_node("Add", ["first", "second"], ["sum"]),
            helper.make_node("Mul", ["sum", "half"], ["output"]),
        ],
        "midpoint",
        [helper.make_tensor_value_info("pair", TensorProto.FLOAT, [1, 6, height, width])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, height, width])],
        initializer=[
            helper.make_tensor("lo0", TensorProto.INT64, [1], [0]),
            helper.make_tensor("hi0", TensorProto.INT64, [1], [3]),
            helper.make_tensor("lo1", TensorProto.INT64, [1], [3]),
            helper.make_tensor("hi1", TensorProto.INT64, [1], [6]),
            helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
            helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _step_at_midpoint(path, height, width):
    """An interpolator for a scene that *jumps* halfway through: `first` before, `second` after.

    Not a plausible network, but it is a frame interpolator with a timestep whose output is not a
    linear blend — which is the only property needed to show that the interpolated frames reach the
    simulator and reach it at the right times.
    """
    image = lambda name: helper.make_tensor_value_info(  # noqa: E731
        name, TensorProto.FLOAT, [1, 3, height, width]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Greater", ["timestep", "half"], ["late"]),
            helper.make_node("Where", ["late", "second", "first"], ["output"]),
        ],
        "step",
        [image("first"), image("second"), helper.make_tensor_value_info("timestep", TensorProto.FLOAT, [1])],
        [image("output")],
        initializer=[helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _moving_edge(width, height, count):
    frames = np.zeros((count, height, width), dtype=np.float32)
    for index in range(count):
        edge = int(width * index / count)
        frames[index, :, :edge] = 0.85
        frames[index, :, edge:] = 0.15
    return frames


@unittest.skipUnless(HAVE_ONNX_RUNTIME, "this build has no onnx feature")
@unittest.skipUnless(HAVE_ONNX_BUILDER, "the onnx package is needed to write the test models")
class InterpolationTests(unittest.TestCase):
    WIDTH, HEIGHT, FRAMES = 32, 24, 8

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        root = Path(cls._dir.name)
        cls.timed = _blend_with_timestep(root / "blend.onnx", cls.HEIGHT, cls.WIDTH)
        cls.stacked = _midpoint_stacked(root / "midpoint.onnx", cls.HEIGHT, cls.WIDTH)
        cls.frames = _moving_edge(cls.WIDTH, cls.HEIGHT, cls.FRAMES)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def _simulate(self, **kwargs):
        return eventcv.simulate(
            self.frames,
            fps=100,
            sigma_thres=0.0,
            cutoff_hz=0,
            leak_rate_hz=0,
            shot_noise_rate_hz=0,
            refractory_us=0,
            upsample="off",
            **kwargs,
        )

    def test_a_linear_interpolator_does_not_change_what_the_sensor_sees(self):
        """Inserting straight-line frames is what the simulator already assumes between two of
        them, so the events must not move. This is the check that interpolation is a preprocessing
        stage and not a change to the pixel model."""
        plain = len(self._simulate().numpy())
        self.assertGreater(plain, 0)
        for factor in (2, 4):
            with self.subTest(factor=factor):
                interpolated = len(
                    self._simulate(interpolate=self.timed, interpolate_factor=factor).numpy()
                )
                self.assertLess(
                    abs(interpolated - plain) / plain,
                    0.02,
                    f"{plain} events became {interpolated}",
                )

    def test_a_non_linear_interpolator_moves_the_events_it_should(self):
        """The reason the feature exists, demonstrated on the case it exists for.

        The simulator assumes a pixel's intensity ramps between two frames. When it *steps* instead
        — an edge crossing it — a linear assumption spreads the events evenly across the interval
        when they all belong at the moment of the step. ``step.onnx`` is an interpolator that knows
        the scene jumps at the halfway point, so feeding it should push those events later. If
        interpolation were being ignored, or the interpolated frames were landing at the wrong
        times, this distribution would not move."""
        step = _step_at_midpoint(Path(self._dir.name) / "step.onnx", self.HEIGHT, self.WIDTH)
        us_per_frame = 10_000

        def phase(events):
            """Where in its frame interval each event falls, in [0, 1)."""
            return (events.numpy()[:, 2] % us_per_frame) / us_per_frame

        linear = phase(self._simulate(interpolate=self.timed, interpolate_factor=4))
        stepped = phase(self._simulate(interpolate=step, interpolate_factor=4))
        self.assertGreater(
            stepped.mean(),
            linear.mean() + 0.05,
            f"a step interpolator should move events later in the interval, but the mean phase "
            f"went from {linear.mean():.3f} to {stepped.mean():.3f}",
        )

    def test_a_stacked_export_without_a_timestep_is_driven_by_bisection(self):
        """An older export can only produce midpoints; eventcv should detect that and bisect rather
        than feed it a timestep it does not declare."""
        events = self._simulate(interpolate=self.stacked, interpolate_factor=2)
        self.assertGreater(len(events.numpy()), 0)

    def test_a_factor_a_midpoint_model_cannot_reach_is_refused(self):
        """Thirds are not reachable by bisection. Saying so beats silently returning a midpoint."""
        with self.assertRaises(Exception) as raised:
            self._simulate(interpolate=self.stacked, interpolate_factor=3)
        self.assertIn("timestep", str(raised.exception))

    def test_a_graph_that_is_not_an_interpolator_is_refused_at_load(self):
        path = Path(self._dir.name) / "wrong.onnx"
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "identity",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        onnx.save(model, str(path))
        with self.assertRaises(ValueError) as raised:
            self._simulate(interpolate=str(path))
        self.assertIn("frame pair", str(raised.exception))

    def test_a_factor_below_two_is_refused(self):
        with self.assertRaises(ValueError):
            self._simulate(interpolate=self.timed, interpolate_factor=1)


if __name__ == "__main__":
    unittest.main()
