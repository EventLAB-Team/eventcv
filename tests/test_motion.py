"""Motion estimation and network streaming, through the Python API.

The Rust core has its own unit tests; these check the binding layer — that arguments reach the right
call, that results come back in a usable shape, and that the ground-truth recovery survives the trip
through Python.
"""

import threading
import time
import unittest

import numpy as np

import eventcv

IDEAL = dict(sigma_thres=0.0, leak_rate_hz=0.0, shot_noise_rate_hz=0.0, cutoff_hz=0.0)


def _moving_bar(pixels_per_second, fps=500, frames=24, width=96, height=64):
    """A bar sweeping right at a known rate, as simulated events."""
    video = np.zeros((frames, height, width), dtype=np.uint8)
    for i in range(frames):
        bar = 12 + int(pixels_per_second * i / fps)
        video[i, :, bar : bar + 4] = 230
    return eventcv.simulate(video, fps=fps, **IDEAL)


class ContrastMaximisationTests(unittest.TestCase):
    def test_recovers_the_simulated_velocity(self):
        # The ground-truth claim, end to end through Python: nothing in the chain is told the answer.
        truth = 200.0
        result = _moving_bar(truth).contrast_maximise()
        self.assertAlmostEqual(result["params"][0], truth, delta=truth * 0.35)
        self.assertGreater(result["improvement"], 1.0)

    def test_the_result_carries_everything_needed_to_judge_it(self):
        result = _moving_bar(200.0).contrast_maximise()
        for key in ("params", "score", "score_at_rest", "improvement", "iterations"):
            self.assertIn(key, result)
        self.assertEqual(len(result["params"]), 2)
        self.assertGreater(result["iterations"], 0)

    def test_every_objective_is_accepted(self):
        events = _moving_bar(200.0)
        for objective in ("variance", "sos", "soe"):
            with self.subTest(objective=objective):
                result = events.contrast_maximise(objective=objective)
                self.assertTrue(np.isfinite(result["score"]))

    def test_an_unknown_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            _moving_bar(200.0).contrast_maximise(objective="sharpness")

    def test_rotation_needs_intrinsics(self):
        events = _moving_bar(200.0)
        with self.assertRaises(ValueError) as caught:
            events.contrast_maximise(model="rotation")
        self.assertIn("camera", str(caught.exception))

        camera = eventcv.Camera(fx=100.0, fy=100.0, cx=48.0, cy=32.0)
        result = events.contrast_maximise(model="rotation", camera=camera, initial_step=1.0)
        self.assertEqual(len(result["params"]), 3)

    def test_the_iwe_is_sharper_at_the_recovered_motion(self):
        events = _moving_bar(300.0)
        result = events.contrast_maximise()
        warped = eventcv.iwe(events, result["params"]).numpy()
        still = eventcv.iwe(events, [0.0, 0.0]).numpy()
        self.assertEqual(warped.shape, (1, 64, 96))
        # Sharper means the same events occupy fewer pixels.
        self.assertLess(int((warped > 0.01).sum()), int((still > 0.01).sum()))

    def test_an_empty_stream_is_rejected(self):
        empty = eventcv.load("data/test/example.npz").time_window(0, 0)
        with self.assertRaises(ValueError):
            empty.contrast_maximise()

    def test_the_free_function_form_exists(self):
        for name in ("contrast_maximise", "iwe"):
            with self.subTest(op=name):
                self.assertIn(name, eventcv.__all__)
                self.assertTrue(callable(getattr(eventcv, name)))


class TrackerTests(unittest.TestCase):
    def _frame_with(self, centres, size=6, width=96, height=64):
        """A count frame containing square blobs, built from events so it is a real frame."""
        rows = []
        for index, (cx, cy) in enumerate(centres):
            for dy in range(size):
                for dx in range(size):
                    rows.append([cx + dx, cy + dy, index * 10 + dx, 1])
        events = eventcv.from_numpy(
            np.array(rows, dtype=np.int64), sensor_size=(width, height), time_unit="us"
        )
        return events.count()

    def test_a_track_keeps_its_id_while_the_object_moves(self):
        tracker = eventcv.Tracker(min_area=4)
        ids = []
        for step in range(8):
            tracks = tracker.update(self._frame_with([(5 + step * 6, 20)]))
            self.assertEqual(len(tracks), 1, f"step {step}")
            ids.append(tracks[0]["id"])
        self.assertEqual(len(set(ids)), 1, f"id changed: {ids}")

    def test_a_track_reports_its_velocity(self):
        tracker = eventcv.Tracker(min_area=4)
        for step in range(5):
            tracker.update(self._frame_with([(5 + step * 6, 20)]))
        self.assertAlmostEqual(tracker.tracks[0]["velocity"][0], 6.0, places=6)

    def test_two_objects_get_two_ids(self):
        tracker = eventcv.Tracker(min_area=4)
        for step in range(5):
            tracks = tracker.update(self._frame_with([(5 + step * 4, 8), (5 + step * 4, 45)]))
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(tracks[0]["id"], tracks[1]["id"])

    def test_reset_clears_tracks_without_reusing_ids(self):
        tracker = eventcv.Tracker(min_area=4)
        first = tracker.update(self._frame_with([(10, 10)]))[0]["id"]
        tracker.reset()
        self.assertEqual(len(tracker.tracks), 0)
        second = tracker.update(self._frame_with([(10, 10)]))[0]["id"]
        self.assertGreater(second, first, "ids must continue past a reset")

    def test_a_track_carries_the_expected_fields(self):
        tracker = eventcv.Tracker(min_area=4)
        track = tracker.update(self._frame_with([(10, 10)]))[0]
        for key in ("id", "centroid", "velocity", "area", "age", "missed"):
            self.assertIn(key, track)

    def test_connectivity_is_validated(self):
        with self.assertRaises(ValueError):
            eventcv.Tracker(connectivity=6)


class UdpTests(unittest.TestCase):
    def test_a_stream_round_trips_with_a_concurrent_receiver(self):
        # UDP drops when a burst outruns the receiver, so the realistic pattern is a receiver
        # already draining — which is what this asserts rather than send-then-receive.
        receiver = eventcv.UdpReceiver("127.0.0.1:0", (640, 480))
        received = []
        thread = threading.Thread(target=lambda: received.append(receiver.recv(dt_ms=500)))
        thread.start()
        time.sleep(0.05)

        source = eventcv.load("data/test/example.npz").time_window(0, 3000)
        sent = eventcv.UdpSender(receiver.address).send(source)
        thread.join()

        self.assertEqual(sent, len(source))
        self.assertGreater(len(received[0]), 0)
        # Whatever arrived must be faithful, even if not everything arrives.
        arrived = received[0].numpy()
        original = source.numpy()
        np.testing.assert_array_equal(arrived[:, 0], original[: len(arrived), 0])
        np.testing.assert_array_equal(arrived[:, 1], original[: len(arrived), 1])

    def test_timestamps_survive_when_requested(self):
        receiver = eventcv.UdpReceiver("127.0.0.1:0", (640, 480), timestamps=True)
        received = []
        thread = threading.Thread(target=lambda: received.append(receiver.recv(dt_ms=500)))
        thread.start()
        time.sleep(0.05)

        source = eventcv.load("data/test/example.npz").time_window(0, 2000)
        eventcv.UdpSender(receiver.address, timestamps=True).send(source)
        thread.join()

        arrived = received[0].numpy()
        self.assertGreater(len(arrived), 0)
        np.testing.assert_array_equal(arrived[:, 2], source.numpy()[: len(arrived), 2])

    def test_a_quiet_link_returns_an_empty_stream(self):
        receiver = eventcv.UdpReceiver("127.0.0.1:0", (64, 64))
        self.assertEqual(len(receiver.recv(dt_ms=20)), 0, "silence is not an error")

    def test_a_bad_window_is_rejected(self):
        receiver = eventcv.UdpReceiver("127.0.0.1:0", (64, 64))
        for dt_ms in (0.0, -5.0, float("nan")):
            with self.subTest(dt_ms=dt_ms):
                with self.assertRaises(ValueError):
                    receiver.recv(dt_ms=dt_ms)

    def test_addresses_are_reported(self):
        receiver = eventcv.UdpReceiver("127.0.0.1:0", (64, 64))
        self.assertTrue(receiver.address.startswith("127.0.0.1:"))
        self.assertTrue(eventcv.UdpSender(receiver.address).address)


if __name__ == "__main__":
    unittest.main()
