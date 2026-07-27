"""Live USB event-camera API (``eventcv.stream`` / ``eventcv.list_cameras`` / ``EventCamera``).

These cover the paths that don't need a camera attached: device enumeration, argument validation,
and the surface of the ``EventCamera`` class. The actual capture/viewer/record paths need hardware
and are exercised separately. Skips entirely when the extension was built without the ``camera``
feature."""

import os
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

    def test_record_rejects_formats_that_cannot_be_appended(self):
        # `record=` archives window-by-window, so it needs HDF5; npz/txt/bag are pointed at
        # `camera.record(...)` instead. Checked before the device is touched.
        for path in ("session.npz", "session.txt", "session.bag"):
            with self.assertRaises(ValueError) as caught:
                eventcv.stream(record=path)
            self.assertIn("record", str(caught.exception))
        self.assertFalse(os.path.exists("session.npz"), "rejected path must not be created")

    def test_record_rejects_a_bad_compression_level(self):
        with self.assertRaises(ValueError):
            eventcv.stream(record="session.h5", compression=42)
        self.assertFalse(os.path.exists("session.h5"), "rejected path must not be created")

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
        for prop in (
            "sensor_size",
            "name",
            "serial",
            "backlog",
            "n_recorded",
            "n_skipped",
            "n_overflows",
        ):
            self.assertTrue(
                hasattr(eventcv.EventCamera, prop),
                f"EventCamera is missing property {prop}",
            )

    def test_stream_accepts_the_representation_options(self):
        # The same per-representation options `EventReader.with_repr` takes, so a live tencode /
        # voxel / tsurf can be tuned instead of being stuck on the 30 ms defaults.
        import inspect

        parameters = inspect.signature(eventcv.stream).parameters
        for name in ("bins", "window_ms", "tau_ms", "max_window_ms", "window", "normalize"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default, f"{name} must default to 'follow dt_ms'")
        # Parsed before the device is touched, so a bad option raises regardless of hardware.
        with self.assertRaises(ValueError):
            eventcv.stream(repr="voxel", bins=-1)
        with self.assertRaises(ValueError):
            eventcv.stream(repr="flow", window=0)

    def test_stream_accepts_the_source_limit_options(self):
        import inspect

        parameters = inspect.signature(eventcv.stream).parameters
        for name in ("max_event_rate", "roi"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameters[name].default)

    def test_source_limits_are_validated_before_the_device_is_opened(self):
        for kwargs in (
            {"max_event_rate": 0},
            {"max_event_rate": -1},
            {"roi": (-1, 0, 10, 10)},
            {"roi": (0, 0, 0, 10)},
        ):
            with self.assertRaises(ValueError, msg=f"{kwargs} must be rejected"):
                eventcv.stream(**kwargs)

    def test_stream_accepts_the_sink_options(self):
        # `record`/`compression`/`latest` are keyword-only and reach the Rust binding — a signature
        # mismatch would raise TypeError here rather than at capture time.
        import inspect

        parameters = inspect.signature(eventcv.stream).parameters
        for name in ("record", "compression", "latest"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["latest"].default, False)

    def test_public_api_is_exported(self):
        for name in ("stream", "list_cameras", "EventCamera"):
            self.assertIn(name, eventcv.__all__)


if __name__ == "__main__":
    unittest.main(verbosity=2)
