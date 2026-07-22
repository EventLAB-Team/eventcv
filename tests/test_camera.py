"""Live USB event-camera API (``eventcv.stream`` / ``eventcv.list_cameras`` / ``EventCamera``).

These cover the paths that don't need a camera attached: device enumeration, argument validation,
and the surface of the ``EventCamera`` class. The actual capture/viewer/record paths need hardware
and are exercised separately. Skips entirely when the extension was built without the ``camera``
feature."""

import unittest

import eventcv


@unittest.skipUnless(eventcv.EventCamera is not None, "built without camera feature")
class CameraApiTests(unittest.TestCase):
    def test_list_cameras_returns_a_list(self):
        cameras = eventcv.list_cameras()
        self.assertIsInstance(cameras, list)
        # If a device happens to be attached, every entry is a dict with the documented keys.
        for camera in cameras:
            self.assertIsInstance(camera, dict)
            self.assertLessEqual(
                {"kind", "name", "serial", "bus", "address", "speed"},
                set(camera),
            )

    def test_stream_without_camera_raises(self):
        if eventcv.list_cameras():
            self.skipTest("a camera is attached")
        # No device present -> a clear runtime error, not a crash.
        with self.assertRaises(RuntimeError):
            eventcv.stream()

    def test_dt_ms_and_max_events_are_mutually_exclusive(self):
        # Validated before the device is touched, so it raises regardless of attached hardware.
        with self.assertRaises(ValueError):
            eventcv.stream(dt_ms=30, max_events=1000)

    def test_unknown_representation_rejected_before_open(self):
        # The repr name is parsed before opening the device, so a bad name is a ValueError
        # (not a "no device found" RuntimeError) even with no camera.
        with self.assertRaises(ValueError):
            eventcv.stream(repr="not-a-representation")

    def test_event_camera_surface(self):
        for method in (
            "show",
            "record",
            "read",
            "close",
            "__enter__",
            "__exit__",
            "__iter__",
            "__next__",
        ):
            self.assertTrue(
                hasattr(eventcv.EventCamera, method),
                f"EventCamera is missing {method}",
            )
        for prop in ("sensor_size", "name", "serial", "backlog"):
            self.assertTrue(
                hasattr(eventcv.EventCamera, prop),
                f"EventCamera is missing property {prop}",
            )

    def test_public_api_is_exported(self):
        for name in ("stream", "list_cameras", "EventCamera"):
            self.assertIn(name, eventcv.__all__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
