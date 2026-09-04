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

    def test_offset_skips_events_before_absolute_time(self):
        stream = eventcv.load(str(EXAMPLE_PATH))
        t = stream.numpy()[:, 2]
        cutoff_us = int(t.min()) + 10_000  # an absolute timestamp 10 ms after t_min

        skipped = eventcv.load(str(EXAMPLE_PATH), offset=cutoff_us / 1000)  # offset is ms

        expected = int((t >= cutoff_us).sum())
        self.assertEqual(len(skipped), expected)
        self.assertGreaterEqual(int(skipped.numpy()[:, 2].min()), cutoff_us)

    def test_offset_composes_with_max_events(self):
        stream = eventcv.load(str(EXAMPLE_PATH))
        t = stream.numpy()[:, 2]
        cutoff_us = int(t.min()) + 10_000

        window = eventcv.load(str(EXAMPLE_PATH), offset=cutoff_us / 1000, max_events=100)

        self.assertEqual(len(window), 100)  # capped after the offset
        # The first kept event is the first at/after the cutoff, not the file's first.
        self.assertEqual(int(window.numpy()[0, 2]), int(t[t >= cutoff_us].min()))

    def test_offset_rejects_negative(self):
        with self.assertRaisesRegex(ValueError, "offset"):
            eventcv.load(str(EXAMPLE_PATH), offset=-1)

    def test_from_numpy_round_trips_a_stream(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        rebuilt = eventcv.from_numpy(
            stream.numpy(), sensor_size=(640, 480), time_unit="us"
        )

        self.assertIsInstance(rebuilt, eventcv.EventStream)
        self.assertEqual(rebuilt.sensor_size, (640, 480))
        np.testing.assert_array_equal(rebuilt.numpy(), stream.numpy())

    def test_from_numpy_infers_sensor_size_and_accepts_txyp(self):
        events = np.array([[100, 3, 1, 1], [250, 0, 2, -1]])  # t x y p

        stream = eventcv.from_numpy(events, order="txyp", time_unit="us")

        self.assertEqual(stream.sensor_size, (4, 3))
        np.testing.assert_array_equal(
            stream.numpy(), [[3, 1, 100, 1], [0, 2, 250, 0]]
        )

    def test_from_numpy_converts_float_seconds(self):
        events = np.array([[0.0, 0.0, 0.5, 1.0], [1.0, 1.0, 1.5, 0.0]])  # x y t p

        stream = eventcv.from_numpy(events)  # fractional t -> seconds

        np.testing.assert_array_equal(stream.numpy()[:, 2], [500_000, 1_500_000])

    def test_from_numpy_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "4"):
            eventcv.from_numpy(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "coordinate"):
            eventcv.from_numpy(np.array([[-1, 0, 100, 1]]), time_unit="us")
        with self.assertRaisesRegex(TypeError, "numpy array"):
            eventcv.from_numpy([[0, 0, 100, 1]])

    def test_generates_polarity_representation(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        frame = stream.flatten(normalize=False)
        events = frame.numpy()

        self.assertIsInstance(frame, eventcv.EventFrame)
        self.assertEqual(frame.shape, (2, 480, 640))
        self.assertEqual(frame.channel_names, ("positive", "negative"))
        self.assertEqual(events.dtype, np.uint64)
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

    def test_generates_binary_occupancy(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        binary = stream.flatten(binary=True, normalize=False)
        expected = np.any(stream.flatten(normalize=False).numpy() > 0, axis=0)

        self.assertEqual(binary.kind, "binary")
        self.assertEqual(binary.shape, (1, 480, 640))
        self.assertEqual(binary.channel_names, ("event",))
        self.assertEqual(binary.numpy().dtype, np.uint8)
        np.testing.assert_array_equal(binary.numpy()[0], expected)
        np.testing.assert_array_equal(
            binary.numpy(), stream.flatten(binary=True, normalize=True).numpy()
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            stream.flatten(eventcv.Polarity(), binary=True)

    def test_generates_common_owned_representations(self):
        stream = eventcv.load(str(EXAMPLE_PATH))
        original = stream.numpy()

        voxel = stream.voxel()
        surface = stream.tsurf()
        points = stream.pset()
        tencode = stream.tencode()
        mcts = stream.mcts()

        self.assertEqual(stream.timestamp_scale_ms, 0.001)
        self.assertEqual((voxel.kind, voxel.shape), ("voxel", (9, 480, 640)))
        self.assertEqual(voxel.numpy().dtype, np.float32)
        self.assertEqual(
            (surface.kind, surface.shape), ("tsurf", (2, 480, 640))
        )
        self.assertEqual(surface.channel_names, ("positive", "negative"))
        self.assertEqual(surface.numpy().dtype, np.float32)
        self.assertIsInstance(points, eventcv.EventPointSet)
        self.assertEqual(points.shape, (len(stream), 4))
        self.assertEqual(points.columns, ("x", "y", "t", "p"))
        self.assertEqual(points.numpy().dtype, np.float32)
        self.assertEqual((tencode.kind, tencode.shape), ("tencode", (3, 480, 640)))
        self.assertEqual(tencode.channel_names, ("positive", "age", "negative"))
        self.assertEqual(tencode.numpy().dtype, np.uint8)
        self.assertEqual((mcts.kind, mcts.shape), ("mcts", (10, 480, 640)))
        self.assertTrue(all(name.startswith("negative_") for name in mcts.channel_names[:5]))
        self.assertTrue(all(name.startswith("positive_") for name in mcts.channel_names[5:]))
        self.assertEqual(mcts.numpy().dtype, np.float32)
        self.assertTrue(callable(voxel.view))
        self.assertTrue(callable(surface.view))
        self.assertTrue(callable(points.view))
        self.assertTrue(callable(tencode.view))
        self.assertTrue(callable(mcts.view))

        voxel_values = voxel.numpy()
        voxel_values.fill(0)
        self.assertNotEqual(stream.voxel().numpy().sum(), 0)
        np.testing.assert_array_equal(stream.numpy(), original)

    def test_generates_count_and_averaged_time_surface(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        count = stream.count()
        count_norm = stream.count(normalize=True)
        atsurf = stream.atsurf()

        self.assertEqual((count.kind, count.shape), ("count", (1, 480, 640)))
        self.assertEqual(count.channel_names, ("count",))
        self.assertEqual(count.numpy().dtype, np.uint64)
        # Total events land somewhere on the single count plane (polarities summed).
        self.assertEqual(count.numpy().sum(), len(stream))
        self.assertEqual(count_norm.numpy().dtype, np.uint8)
        self.assertEqual(count_norm.numpy().max(), 255)

        self.assertEqual((atsurf.kind, atsurf.shape), ("atsurf", (2, 480, 640)))
        self.assertEqual(atsurf.channel_names, ("positive", "negative"))
        self.assertEqual(atsurf.numpy().dtype, np.float32)
        # Averaged responses are bounded by 1 (a mean of exp(-age/tau) values).
        self.assertLessEqual(atsurf.numpy().max(), 1.0 + 1e-6)
        self.assertTrue(callable(count.view))
        self.assertTrue(callable(atsurf.view))

    def test_generates_countmask(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        countmask = stream.countmask()

        self.assertEqual(
            (countmask.kind, countmask.shape), ("countmask", (3, 480, 640))
        )
        self.assertEqual(
            countmask.channel_names, ("positive", "activity", "negative")
        )
        self.assertEqual(countmask.numpy().dtype, np.uint8)
        self.assertTrue(callable(countmask.view))

        # The activity mask is full-scale wherever either count plane fired, so green covers at
        # least as many pixels as red or blue and is far brighter. A dark green plane means the
        # background got inverted.
        red, green, blue = countmask.numpy()
        np.testing.assert_array_equal(green > 0, (red > 0) | (blue > 0))
        self.assertEqual(set(np.unique(green).tolist()), {0, 255})
        self.assertGreater(green.mean(), 2 * max(red.mean(), blue.mean()))

        # white_frame flips the encoding before the 8-bit cast, so it is not a plain `255 - x` on
        # the counts (both 0.5 and 1 - 0.5 truncate to 127). It is exact on the binary mask, and
        # pixels that saw no event go from black to white.
        inverted = stream.countmask(white_frame=True).numpy()
        np.testing.assert_array_equal(inverted[1], 255 - green)
        idle = green == 0
        self.assertTrue((inverted[:, idle] == 255).all())

    def test_countmask_matches_reference_renderer(self):
        # 40 pseudo-random events on a 6x8 sensor, from the reference renderer's
        # np.random.default_rng(12345) fixture. Here the pooled 99th percentile is 2, so a count
        # of 1 must truncate to 127 rather than round to 128.
        x = [5, 1, 6, 2, 1, 6, 5, 5, 7, 3, 6, 2, 4, 4, 1, 1, 1, 5, 4, 7,
             5, 1, 7, 7, 5, 5, 1, 0, 2, 3, 0, 7, 3, 5, 1, 2, 0, 5, 6, 1]
        y = [4, 0, 2, 0, 4, 2, 2, 2, 2, 1, 3, 4, 2, 1, 0, 0, 0, 0, 0, 3,
             4, 5, 3, 3, 1, 5, 3, 4, 4, 5, 4, 5, 3, 3, 1, 5, 3, 2, 1, 1]
        p = [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
             0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        events = np.stack([x, y, np.arange(40), p], axis=1)
        stream = eventcv.from_numpy(events, sensor_size=(8, 6), time_unit="us")

        expected = np.array(
            [
                [[0, 255, 0, 0, 0, 127, 0, 0],
                 [0, 127, 0, 0, 0, 0, 127, 0],
                 [0, 0, 0, 0, 127, 255, 0, 127],
                 [0, 0, 0, 0, 0, 0, 127, 127],
                 [255, 127, 127, 0, 0, 127, 0, 0],
                 [0, 127, 0, 0, 0, 127, 0, 127]],
                [[0, 255, 255, 0, 255, 255, 0, 0],
                 [0, 255, 0, 255, 255, 255, 255, 0],
                 [0, 0, 0, 0, 255, 255, 255, 255],
                 [255, 255, 0, 255, 0, 255, 255, 255],
                 [255, 255, 255, 0, 0, 255, 0, 0],
                 [0, 255, 255, 255, 0, 255, 0, 255]],
                [[0, 255, 127, 0, 127, 0, 0, 0],
                 [0, 127, 0, 127, 127, 127, 0, 0],
                 [0, 0, 0, 0, 0, 127, 255, 0],
                 [127, 127, 0, 127, 0, 127, 0, 255],
                 [0, 0, 127, 0, 0, 127, 0, 0],
                 [0, 0, 127, 127, 0, 0, 0, 0]],
            ],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(stream.countmask().numpy(), expected)

    def test_mcts_accepts_explicit_windows(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        frame = stream.mcts(windows_ms=[1, 5, 20])

        self.assertEqual((frame.kind, frame.shape), ("mcts", (6, 480, 640)))
        self.assertEqual(
            frame.channel_names,
            (
                "negative_1.000ms",
                "negative_5.000ms",
                "negative_20.000ms",
                "positive_1.000ms",
                "positive_5.000ms",
                "positive_20.000ms",
            ),
        )
        self.assertEqual(frame.numpy().dtype, np.float32)
        # An int list and a float list are the same request.
        np.testing.assert_array_equal(
            stream.mcts(windows_ms=[1.0, 5.0, 20.0]).numpy(), frame.numpy()
        )

    def test_validates_representation_parameters(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        for generate in (
            lambda: stream.voxel(bins=0),
            lambda: stream.voxel(window_ms=float("nan")),
            lambda: stream.tsurf(tau_ms=0),
            lambda: stream.atsurf(tau_ms=0),
            lambda: stream.tencode(window_ms=float("inf")),
            lambda: stream.mcts(max_window_ms=0.5),
            lambda: stream.mcts(windows_ms=[]),
            lambda: stream.mcts(windows_ms=[0]),
            lambda: stream.mcts(windows_ms=[-1.0]),
            lambda: stream.mcts(windows_ms=[5], max_window_ms=30),
            lambda: stream.countmask(pct=100.5),
            lambda: stream.countmask(pct=-1),
        ):
            with self.subTest(generate=generate):
                with self.assertRaises(ValueError):
                    generate()

    def test_empty_stream_representations_have_stable_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.npz"
            np.savez(path, event_data=np.array([], dtype=EVENT_DTYPE))
            stream = eventcv.load(str(path))

            self.assertEqual(stream.flatten(binary=True).shape, (1, 480, 640))
            self.assertEqual(stream.count().shape, (1, 480, 640))
            self.assertEqual(stream.voxel().shape, (9, 480, 640))
            self.assertEqual(stream.tsurf().shape, (2, 480, 640))
            self.assertEqual(stream.atsurf().shape, (2, 480, 640))
            self.assertEqual(stream.tencode().shape, (3, 480, 640))
            self.assertEqual(stream.countmask().shape, (3, 480, 640))
            self.assertEqual(stream.mcts().shape, (10, 480, 640))
            self.assertEqual(stream.pset().shape, (0, 4))

    def test_resizes_polarity_representation(self):
        stream = eventcv.load(str(EXAMPLE_PATH))
        frame = stream.flatten()

        resized = frame.resize(256, 256)
        events = resized.numpy()

        self.assertIsInstance(resized, eventcv.EventFrame)
        self.assertEqual(resized.shape, (2, 256, 256))
        self.assertEqual(resized.channel_names, frame.channel_names)
        self.assertEqual(events.dtype, np.uint8)
        self.assertEqual(frame.shape, (2, 480, 640))

    def test_resizes_float_representations(self):
        voxel = eventcv.load(str(EXAMPLE_PATH)).voxel()

        average = voxel.resize(256, 256)
        summed = voxel.resize(256, 256, pooling="sum")

        self.assertEqual(average.shape, (9, 256, 256))
        self.assertEqual(average.numpy().dtype, np.float32)
        self.assertEqual(summed.numpy().dtype, np.float32)
        self.assertEqual(average.kind, "voxel")
        self.assertEqual(average.channel_names, voxel.channel_names)

    def test_event_domain_resize_returns_a_stream(self):
        # stream.resize now resizes in the event domain (Workstream B), distinct from the
        # frame-domain EventFrame.resize tested above. It returns a chainable EventStream.
        stream = eventcv.load(str(EXAMPLE_PATH))
        resized = stream.resize(320, 240)
        self.assertIsInstance(resized, eventcv.EventStream)
        self.assertEqual(resized.sensor_size, (320, 240))
        self.assertEqual(len(resized), len(stream))  # rebinning conserves count

    def test_sum_resize_preserves_raw_event_count(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        events = stream.flatten(normalize=False).resize(
            256, 256, pooling="sum"
        ).numpy()

        self.assertEqual(events.dtype, np.uint64)
        self.assertEqual(events.shape, (2, 256, 256))
        self.assertEqual(events.sum(), len(stream))

    def test_resizes_with_bilinear_enlargement(self):
        frame = eventcv.load(str(EXAMPLE_PATH)).flatten(normalize=False)

        events = frame.resize(800, 600).numpy()

        self.assertEqual(events.dtype, np.uint64)
        self.assertEqual(events.shape, (2, 600, 800))

    def test_rejects_invalid_resize_arguments(self):
        frame = eventcv.load(str(EXAMPLE_PATH)).flatten()

        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            frame.resize(0, 256)
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            frame.resize(-1, 256)
        with self.assertRaisesRegex(ValueError, "unsupported pooling method"):
            frame.resize(256, 256, pooling="maximum")

    def test_rejects_unknown_representation(self):
        stream = eventcv.load(str(EXAMPLE_PATH))

        with self.assertRaisesRegex(TypeError, "unsupported representation"):
            stream.flatten(object())

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            eventcv.load("missing.npz")

    def test_unrecognised_npz_layout_raises_value_error(self):
        # An npz with neither the N-ImageNet `event_data` array nor the native `x/y/t/p`
        # columns is not a recognisable event archive (the reader falls through to the
        # native layout and reports the first missing column).
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-event-data.npz"
            np.savez(path, other=np.array([], dtype=EVENT_DTYPE))

            with self.assertRaisesRegex(ValueError, "missing array"):
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
