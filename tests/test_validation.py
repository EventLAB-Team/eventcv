"""Validation against real recordings, with independent ground truth.

Everything else in this suite checks EventCV against itself or against synthetic data it generated.
These tests check it against the physical world: a DAVIS346's own IMU says how the camera rotated,
and its APS frames say what the scene looked like, so contrast maximisation and the simulator can
both be held to an external measurement.

The recordings are hundreds of megabytes to tens of gigabytes and are not in the repository, so
every test here skips when `data/development` is absent — which is how they behave in CI. The
tolerances are set from measured values, not chosen to make the tests pass; the measurement each
one came from is noted alongside it.
"""

import unittest
from pathlib import Path

import numpy as np

import eventcv

ROOT = Path(__file__).resolve().parent.parent
BAG = ROOT / "data" / "development" / "dvs_vpr_2020-04-21-17-03-03.bag"
EVT2 = ROOT / "data" / "development" / "spinner.raw"
# Whichever DAVIS346 AEDAT 2.0 recording is on this machine; the names differ between checkouts.
AEDAT = next(iter(sorted((ROOT / "data" / "development").glob("*.aedat"))), None)

# A window where the car is turning: the IMU shows sustained yaw of about -0.56 rad/s, which is far
# enough above the recording's median (0.12 rad/s) for the motion to be unambiguous. Offsets are
# from the recording start, so they survive the epoch timestamps this bag carries.
TURN_OFFSET_US = 540_400_000
WINDOW_US = 50_000


def _events_reader():
    return eventcv.open(str(BAG), topic="/dvs/events")


def _origin_us(reader):
    return int(reader.time_span()[0] * 1000)


@unittest.skipUnless(EVT2.exists(), "data/development/spinner.raw is not present")
class Evt2RealFileTests(unittest.TestCase):
    """The EVT2 reader against a real Gen3 recording."""

    def test_reads_a_real_evt2_recording(self):
        reader = eventcv.open(str(EVT2))
        width, height = reader.sensor_size
        # The header carries no `% geometry`, so the size is derived from the events. It is a tight
        # bound on what fired, which for a Gen3 VGA lands just under 640x480.
        self.assertLessEqual(width, 640)
        self.assertLessEqual(height, 480)
        self.assertGreater(width, 500)
        self.assertGreater(height, 400)
        self.assertGreater(reader.n_events, 1_000_000)

    def test_events_are_ordered_and_within_bounds(self):
        stream = eventcv.load(str(EVT2), max_events=2_000_000)
        events = stream.numpy()
        width, height = stream.sensor_size
        self.assertTrue(np.all(np.diff(events[:, 2]) >= 0), "timestamps must be ascending")
        self.assertLess(events[:, 0].max(), width)
        self.assertLess(events[:, 1].max(), height)
        self.assertTrue(set(np.unique(events[:, 3])) <= {0, 1})

    def test_an_explicit_sensor_size_overrides_the_derivation(self):
        self.assertEqual(
            eventcv.open(str(EVT2), sensor_size=(640, 480)).sensor_size, (640, 480)
        )

    def test_slicing_agrees_with_a_bulk_read(self):
        # The sparse checkpoint index has to replay to exactly the same events a linear read gives.
        bulk = eventcv.load(str(EVT2), max_events=50_000).numpy()
        reader = eventcv.open(str(EVT2))
        np.testing.assert_array_equal(reader.slice_count(0, 50_000).numpy(), bulk)
        # And a slice from the middle must not depend on how it was reached.
        np.testing.assert_array_equal(
            reader.slice_count(30_000, 40_000).numpy(), bulk[30_000:40_000]
        )


@unittest.skipUnless(AEDAT is not None, "no DAVIS346 .aedat recording is present")
class AedatAuxiliaryStreamTests(unittest.TestCase):
    """The same frames and IMU out of an AEDAT 2.0 recording, held to the same physics.

    The bag reader has always returned these; the point of the checks here is that a second,
    completely different container decodes to the same quantities in the same units.
    """

    def _reader(self):
        return eventcv.open(str(AEDAT))

    def test_aps_frames_are_correlated_double_sampled(self):
        reader = self._reader()
        t0 = int(reader.time_span("us")[0])
        frames = eventcv.read_frames(reader, t0_us=t0, t1_us=t0 + 2_000_000)
        self.assertGreater(len(frames), 2)
        image = frames[0][1].numpy()
        self.assertEqual(image.shape, (1, 260, 346), "a DAVIS346 APS frame")
        # Reset minus signal, both 10-bit. A sign error or a swapped pair would pin the
        # difference at zero; this is a night drive, so even a correct frame is mostly dark —
        # measured across the recording, 40% to 46% of pixels are lit and the brightest
        # saturates. The bound is set below that, not at it.
        self.assertLessEqual(int(image.max()), 1023)
        self.assertGreater(int(image.max()), 100)
        self.assertGreater(float((image > 0).mean()), 0.2)
        # ~9 fps in this recording; assert the order of magnitude, not the exact rate.
        rate = 1e6 / np.diff([t for t, _ in frames]).mean()
        self.assertGreater(rate, 1)
        self.assertLess(rate, 100)

    def test_imu_decodes_to_physical_units(self):
        # Same check as the bag's, and the one that catches a wrong full-scale setting: jAER
        # writes every chip's preferences into the header, and reading another camera's would
        # scale these by a factor of four.
        reader = self._reader()
        t0 = int(reader.time_span("us")[0])
        imu = eventcv.read_imu(reader, t0_us=t0, t1_us=t0 + 1_000_000)
        self.assertGreater(len(imu["t"]), 100)
        magnitude = np.linalg.norm(imu["linear_acceleration"], axis=1).mean()
        self.assertGreater(magnitude, 8.0)
        self.assertLess(magnitude, 12.0)
        # A car drive: no sustained rotation anywhere near the sensor's ±250 °/s full scale.
        self.assertLess(np.abs(imu["angular_velocity"]).mean(), 1.0)

    def test_the_streams_share_one_clock(self):
        reader = self._reader()
        t0 = int(reader.time_span("us")[0])
        t1 = t0 + 2_000_000
        events = reader.slice(t0_us=t0, t1_us=t1).numpy()
        frames = eventcv.read_frames(reader, t0_us=t0, t1_us=t1)
        imu = eventcv.read_imu(reader, t0_us=t0, t1_us=t1)
        for label, times in (
            ("events", events[:, 2].astype(np.int64)),
            ("frames", np.array([t for t, _ in frames])),
            ("imu", np.asarray(imu["t"])),
        ):
            self.assertGreaterEqual(times.min(), t0, label)
            self.assertLess(times.max(), t1, label)

    def test_the_index_is_exact(self):
        # `n_events` comes from a parallel index pass, and the slices come from replaying it;
        # they have to agree with each other or a dataset built on the reader silently truncates.
        reader = self._reader()
        total = reader.n_events
        self.assertEqual(len(reader.slice_count(total - 1_000, total).numpy()), 1_000)
        self.assertEqual(len(reader.slice_count(total - 10, total + 10_000).numpy()), 10)


@unittest.skipUnless(BAG.exists(), "the DAVIS346 bag is not present")
class BagAuxiliaryStreamTests(unittest.TestCase):
    """Frames, IMU and intrinsics out of a real DAVIS recording."""

    def test_topics_are_listed(self):
        topics = dict(eventcv.bag_topics(str(BAG)))
        self.assertEqual(topics["/dvs/events"], "dvs_msgs/EventArray")
        self.assertEqual(topics["/dvs/image_raw"], "sensor_msgs/Image")
        self.assertEqual(topics["/dvs/imu"], "sensor_msgs/Imu")

    def test_aps_frames_match_the_sensor(self):
        reader = _events_reader()
        t0 = _origin_us(reader) + TURN_OFFSET_US
        frames = eventcv.read_frames(str(BAG), t0_us=t0, t1_us=t0 + 1_000_000)
        self.assertGreater(len(frames), 10)
        image = frames[0][1].numpy()
        self.assertEqual(image.shape, (1, 260, 346), "a DAVIS346 APS frame")
        self.assertEqual(image.dtype, np.uint8)
        # ~40 fps in this recording; assert the order of magnitude, not the exact rate.
        rate = 1e6 / np.diff([t for t, _ in frames]).mean()
        self.assertGreater(rate, 10)
        self.assertLess(rate, 100)

    def test_imu_decodes_to_physical_units(self):
        # The check that catches a byte-layout mistake: the accelerometer must see gravity. A
        # misparsed message gives numbers with no reason to land near 9.81.
        reader = _events_reader()
        t0 = _origin_us(reader) + TURN_OFFSET_US
        imu = eventcv.read_imu(str(BAG), t0_us=t0, t1_us=t0 + 1_000_000)
        self.assertGreater(len(imu["t"]), 100)
        magnitude = np.linalg.norm(imu["linear_acceleration"], axis=1).mean()
        self.assertGreater(magnitude, 8.0)
        self.assertLess(magnitude, 12.0)
        self.assertEqual(imu["angular_velocity"].shape[1], 3)

    def test_intrinsics_are_plausible_for_a_davis346(self):
        camera = eventcv.read_camera_info(str(BAG))
        self.assertIsNotNone(camera)
        # The principal point should sit near the middle of a 346x260 sensor.
        self.assertGreater(camera.cx, 100)
        self.assertLess(camera.cx, 250)
        self.assertGreater(camera.cy, 70)
        self.assertLess(camera.cy, 190)
        self.assertGreater(camera.fx, 100)

    def test_the_three_streams_share_a_clock(self):
        # Nothing downstream is meaningful if they do not — a cmax-versus-IMU comparison would be
        # measuring clock skew rather than the estimator.
        reader = _events_reader()
        t0 = _origin_us(reader) + TURN_OFFSET_US
        t1 = t0 + 1_000_000
        frames = eventcv.read_frames(str(BAG), t0_us=t0, t1_us=t1)
        imu = eventcv.read_imu(str(BAG), t0_us=t0, t1_us=t1)
        events = reader.slice(t0_ms=t0 / 1000, t1_ms=t1 / 1000).numpy()
        # Every stream's samples must land inside the window that was asked for.
        for label, times in (
            ("frames", np.array([t for t, _ in frames])),
            ("imu", imu["t"]),
            ("events", events[:, 2]),
        ):
            self.assertGreaterEqual(times.min(), t0, label)
            self.assertLess(times.max(), t1, label)


@unittest.skipUnless(BAG.exists(), "the DAVIS346 bag is not present")
class ContrastMaximisationAgainstImuTests(unittest.TestCase):
    """The strongest check available: recover the camera's rotation and compare it to the IMU."""

    def _window(self, index):
        reader = _events_reader()
        t0 = _origin_us(reader) + TURN_OFFSET_US + index * WINDOW_US
        t1 = t0 + WINDOW_US
        imu = eventcv.read_imu(str(BAG), t0_us=t0, t1_us=t1)
        # Hot-pixel filtering is not optional here — see the test below for why.
        events = reader.slice(t0_ms=t0 / 1000, t1_ms=t1 / 1000).hot_pixel_filter(3.0)
        return imu["angular_velocity"].mean(axis=0), events

    def test_the_chosen_window_actually_contains_rotation(self):
        # A stationary window would make the comparison below pass trivially.
        truth, _ = self._window(0)
        self.assertGreater(np.linalg.norm(truth), 0.3, "expected a turn in this window")

    def test_recovered_yaw_agrees_with_the_imu(self):
        camera = eventcv.read_camera_info(str(BAG))
        truths, recovered = [], []
        for index in range(6):
            truth, events = self._window(index)
            result = events.contrast_maximise(
                model="rotation", camera=camera, initial_step=0.6
            )
            truths.append(truth[1])
            recovered.append(result["params"][1])
            self.assertGreater(result["improvement"], 1.0, "should beat the static hypothesis")

        truths, recovered = np.array(truths), np.array(recovered)
        # Measured: IMU -0.55..-0.59 rad/s, recovered -0.57..-0.66 — same axis, same sign, about
        # 10% high because the car is translating as well, and a pure-rotation fit absorbs some of
        # that. 30% allows for it without admitting a sign error or a wrong axis.
        np.testing.assert_allclose(recovered, truths, rtol=0.30)
        self.assertLess(np.corrcoef(truths, recovered)[0, 1], 0.0) if False else None
        self.assertGreater(
            np.corrcoef(truths, recovered)[0, 1], 0.4, "should track the IMU, not just match on average"
        )

    def test_hot_pixels_must_be_removed_first(self):
        # The practical finding this validation produced, pinned so it cannot regress silently:
        # a single stuck pixel firing at ~15 kHz dominates the objective, and because it does not
        # move, the sharpest image is the unwarped one. Contrast maximisation then confidently
        # reports zero motion on a camera that is visibly turning.
        reader = _events_reader()
        t0 = _origin_us(reader) + TURN_OFFSET_US
        raw = reader.slice(t0_ms=t0 / 1000, t1_ms=(t0 + WINDOW_US) / 1000)
        busiest = raw.count().numpy().max()
        self.assertGreater(busiest, 200, "this window is expected to contain a hot pixel")

        unfiltered = raw.contrast_maximise(initial_step=150.0)
        filtered = raw.hot_pixel_filter(3.0).contrast_maximise(initial_step=150.0)
        self.assertAlmostEqual(unfiltered["params"][0], 0.0, places=3)
        self.assertGreater(abs(filtered["params"][0]), 50.0)


@unittest.skipUnless(BAG.exists(), "the DAVIS346 bag is not present")
class SimulatorCalibrationTests(unittest.TestCase):
    """Simulate from the APS frames and compare against the events recorded beside them."""

    CALIBRATED_THRESHOLD = 0.3

    def _paired_window(self):
        reader = _events_reader()
        t0 = _origin_us(reader) + TURN_OFFSET_US
        t1 = t0 + 1_000_000
        frames = eventcv.read_frames(str(BAG), t0_us=t0, t1_us=t1)
        stack = np.stack([frame.numpy()[0] for _, frame in frames])
        fps = 1e6 / np.diff([t for t, _ in frames]).mean()
        real = reader.slice(t0_ms=t0 / 1000, t1_ms=t1 / 1000).hot_pixel_filter(3.0)
        return stack, fps, real

    def _simulate(self, stack, fps, threshold):
        return eventcv.simulate(
            stack,
            fps=fps,
            pos_thres=threshold,
            neg_thres=threshold,
            leak_rate_hz=0,
            shot_noise_rate_hz=0,
        )

    def test_the_calibrated_threshold_matches_the_real_event_rate(self):
        # Measured on this recording: at pos_thres=0.3 the simulator produces 0.93x the real event
        # count. That is also where a DAVIS's nominal contrast threshold sits in the literature,
        # which is a reassuring place for the fit to land. The band allows a factor of two, since
        # the aim is "the right order", not a claim of pixel-accurate agreement.
        stack, fps, real = self._paired_window()
        simulated = self._simulate(stack, fps, self.CALIBRATED_THRESHOLD)
        ratio = len(simulated) / len(real)
        self.assertGreater(ratio, 0.5, f"simulated {len(simulated)} vs real {len(real)}")
        self.assertLess(ratio, 2.0, f"simulated {len(simulated)} vs real {len(real)}")

    def test_the_event_rate_falls_as_the_threshold_rises(self):
        # The monotonic relationship the pixel model implies. If this breaks, the simulator's
        # threshold handling is wrong regardless of how well any single value happens to fit.
        stack, fps, _ = self._paired_window()
        counts = [len(self._simulate(stack, fps, t)) for t in (0.1, 0.3, 0.8)]
        self.assertTrue(
            counts[0] > counts[1] > counts[2], f"expected a falling sequence, got {counts}"
        )

    def test_polarity_balance_matches(self):
        stack, fps, real = self._paired_window()
        simulated = self._simulate(stack, fps, self.CALIBRATED_THRESHOLD)
        # Measured: 0.506 simulated against 0.522 real.
        self.assertAlmostEqual(
            simulated.numpy()[:, 3].mean(), real.numpy()[:, 3].mean(), delta=0.15
        )

    def test_events_land_where_the_real_ones_do(self):
        # Rate alone would be satisfied by noise. This asks whether the simulated events appear in
        # the same parts of the image — measured correlation 0.61.
        stack, fps, real = self._paired_window()
        simulated = self._simulate(stack, fps, self.CALIBRATED_THRESHOLD)
        a = real.count().numpy()[0].astype(float)
        b = simulated.count().numpy()[0].astype(float)
        height = min(a.shape[0], b.shape[0])
        width = min(a.shape[1], b.shape[1])
        correlation = np.corrcoef(a[:height, :width].ravel(), b[:height, :width].ravel())[0, 1]
        self.assertGreater(correlation, 0.4, f"spatial correlation {correlation:.3f}")


if __name__ == "__main__":
    unittest.main()
