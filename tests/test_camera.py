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
    def test_enumeration_never_raises(self):
        # The regression this file exists to prevent. `list_cameras()` used to raise on any machine
        # whose USB subsystem could not be initialised — no camera attached, a container, a CI
        # runner — which made a question about what is plugged in into a fatal error, and made the
        # behaviour differ between Linux and macOS. Asking is always allowed; the answer may be
        # nothing.
        #
        # `test_list_cameras_returns_a_list` below asserts the *shape* of the answer, but it only
        # reaches its assertion on a machine where the call already worked. This asserts the call
        # itself.
        import warnings

        with warnings.catch_warnings():
            # A RuntimeWarning is legitimate here (a camera attached without udev rules), so it must
            # not be promoted to an error — but nothing may be raised.
            warnings.simplefilter("always")
            cameras = eventcv.list_cameras()
        self.assertIsInstance(cameras, list)

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

    def test_record_rejects_a_path_that_is_not_an_event_file(self):
        # Every event format can be appended window-by-window, so the only thing `record=` turns
        # away is a path that is not an event container at all. Checked before the device is
        # touched, so it raises the same way with or without hardware.
        with self.assertRaises(ValueError) as caught:
            eventcv.stream(record="session.png")
        self.assertIn("record", str(caught.exception))
        self.assertFalse(os.path.exists("session.png"), "rejected path must not be created")

    def test_record_accepts_every_event_format(self):
        # The regression this replaces: `record=` used to write HDF5 whatever the extension said,
        # so a `.npz` path produced an HDF5 file wearing the wrong name. These paths must now get
        # *past* validation — with no camera attached that shows up as the device error rather than
        # a ValueError, which is exactly the distinction worth pinning.
        if eventcv.list_cameras():
            self.skipTest("a camera is attached, so opening one would succeed and record for real")
        for path in ("session.npz", "session.txt", "session.bag", "session.aedat4", "session.raw"):
            with self.subTest(path=path):
                with self.assertRaises(RuntimeError):
                    eventcv.stream(record=path)
                self.assertFalse(
                    os.path.exists(path),
                    "a recording must not be created before the camera opens",
                )

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
            "bias_state",
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

    def test_camera_reports_where_its_roi_is_enforced(self):
        # `roi=` is on-chip where the sensor has region masks and a host-side mask where it does
        # not, so the placement has to be inspectable rather than assumed.
        self.assertTrue(hasattr(eventcv.EventCamera, "roi"))

    def test_stream_accepts_the_adaptive_bias_option(self):
        import inspect

        parameters = inspect.signature(eventcv.stream).parameters
        self.assertIn("adaptive_bias", parameters)
        self.assertIs(parameters["adaptive_bias"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(parameters["adaptive_bias"].default, "biasing must be opt-in")

    def test_adaptive_bias_is_validated_before_the_device_is_opened(self):
        # Every one of these is rejected while parsing the argument, so the error is the same with
        # or without a camera attached.
        for kwargs, expected in (
            ({"adaptive_bias": "yes"}, TypeError),
            ({"adaptive_bias": 3}, TypeError),
            ({"adaptive_bias": {"not_an_option": 1}}, ValueError),
            ({"adaptive_bias": {"period_ms": 0}}, ValueError),
            ({"adaptive_bias": {"period_ms": -5}}, ValueError),
            # A band the controller cannot map: high must exceed low.
            ({"adaptive_bias": {"target_rate": (2.5e6, 5e5)}}, ValueError),
            # Zero slew would freeze the fast loop where it started.
            ({"adaptive_bias": {"max_slew": 0}}, ValueError),
            ({"adaptive_bias": {"throttle_range": (1084, 801)}}, ValueError),
            ({"adaptive_bias": {"limits": (2040, 0)}}, ValueError),
            ({"adaptive_bias": {"limits": {"nope": (0, 10)}}}, ValueError),
            ({"adaptive_bias": {"limits": "wide"}}, TypeError),
        ):
            with self.assertRaises(expected, msg=f"{kwargs} must be rejected"):
                eventcv.stream(**kwargs)

    def test_adaptive_bias_accepts_both_limit_forms(self):
        # One range for every bias, or a dict naming them — the per-bias form is what an IMX636
        # needs, since its ON and OFF thresholds sit on opposite sides of a shared reference.
        # With no camera these reach *open* and fail there, not while parsing.
        if eventcv.list_cameras():
            self.skipTest("a camera is attached")
        for limits in ((30, 190), {"on_threshold": (83, 140), "off_threshold": (35, 76)}):
            with self.assertRaises(RuntimeError):
                eventcv.stream(adaptive_bias={"limits": limits})

    def test_adaptive_bias_disabled_forms_do_not_raise_on_parsing(self):
        # False and None mean "off" and must be indistinguishable from omitting it: with no camera
        # attached they fail at *open*, not at parsing.
        if eventcv.list_cameras():
            self.skipTest("a camera is attached")
        for value in (False, None):
            with self.assertRaises(RuntimeError):
                eventcv.stream(adaptive_bias=value)

    def test_stream_accepts_the_sink_options(self):
        # `record`/`compression`/`latest` are keyword-only and reach the Rust binding — a signature
        # mismatch would raise TypeError here rather than at capture time.
        import inspect

        parameters = inspect.signature(eventcv.stream).parameters
        for name in ("record", "compression", "latest"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameters["latest"].default, False)

    def test_stream_forwards_every_representation_option(self):
        # The Python wrapper hand-lists `stream`'s kwargs, so an option the Rust binding accepts but
        # the wrapper forgets is a TypeError raised before the call ever reaches Rust — which is how
        # `pct`/`white_frame` made `repr="countmask"` unreachable from Python.
        import inspect

        parameters = inspect.signature(eventcv.stream).parameters
        for name in ("pct", "white_frame"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        if eventcv.list_cameras():
            self.skipTest("a camera is attached")
        # Reaches *open* and fails there — the point being that it is not a TypeError.
        with self.assertRaises(RuntimeError):
            eventcv.stream(repr="countmask", pct=95, white_frame=True)

    def test_record_is_a_one_shot_function(self):
        # `record(path, ...)` opens, captures, and closes in one call — the safe form of
        # `stream(...).record(...)`, which leaves the camera open until Python collects it.
        import inspect

        parameters = inspect.signature(eventcv.record).parameters
        self.assertIs(
            parameters["path"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        for name in ("seconds", "serial", "dt_ms", "roi", "mask", "max_event_rate"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_record_validates_before_the_device_is_opened(self):
        # Windowing and source limits are parsed up front, so these raise with or without hardware.
        for kwargs in (
            {"dt_ms": 30, "max_events": 1000},
            {"roi": (0, 0, 0, 10)},
            {"max_event_rate": -1},
        ):
            with self.assertRaises(ValueError, msg=f"{kwargs} must be rejected"):
                eventcv.record("session.h5", seconds=1, **kwargs)
        self.assertFalse(os.path.exists("session.h5"), "rejected path must not be created")

    def test_public_api_is_exported(self):
        for name in ("stream", "record", "list_cameras", "EventCamera"):
            self.assertIn(name, eventcv.__all__)

    def test_auxiliary_streams_are_on_the_camera_surface(self):
        # A DAVIS's frames and IMU reach Python under the same names, and in the same shapes, as
        # they do from a file — so code written against a recording runs on a live camera. The
        # decode paths need hardware; the surface does not.
        import inspect

        for name in ("read_frames", "read_imu", "n_frames", "n_imu"):
            self.assertTrue(hasattr(eventcv.EventCamera, name), name)
        parameters = inspect.signature(eventcv.stream).parameters
        for name in ("frames", "imu"):
            self.assertIn(name, parameters)
            self.assertIs(parameters[name].default, False, f"{name} must be opt-in")


if __name__ == "__main__":
    unittest.main(verbosity=2)
