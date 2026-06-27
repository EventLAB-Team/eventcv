from __future__ import annotations

from . import _rust

EventStream = _rust.EventStream
EventFrame = _rust.EventFrame
EventPointSet = _rust.EventPointSet
EventReader = _rust.EventReader
Polarity = _rust.Polarity
Camera = _rust.Camera


def load(
    path: str,
    *,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "txyp",
    topic: str | None = None,
    max_events: int | None = None,
) -> EventStream:
    """Load events from any supported file, detected by its extension.

    Supported today: ``.npz`` (N-ImageNet), ``.txt``/``.csv`` (e.g. EV-IMO
    ``t x y p``), ``.bag`` (ROS ``dvs_msgs/EventArray``), ``.hdf5``/``.h5``,
    ``.aedat`` (AEDAT 2.0, jAER/DAVIS), and ``.dat`` (Prophesee CD events).

    ``sensor_size`` and ``time_unit`` are **auto-detected** when omitted and only act
    as overrides: rosbags carry both in the message; HDF5/text infer the time unit
    from the timestamps (a fractional text value means seconds) and the resolution
    from the coordinate range. Passing ``sensor_size`` for HDF5 also skips that scan.
    ``time_unit`` is ``seconds``/``milliseconds``/``microseconds``/``nanoseconds`` (or
    ``auto``); ``order`` (``txyp``/``xytp``) applies to text. ``topic`` selects the
    rosbag topic (default ``/davis/left/events``). ``max_events`` caps how many events
    are read, handy for previewing very large files.
    """
    return _rust.load(
        path,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
        max_events=max_events,
    )


def open(
    path: str,
    *,
    dt_ms: float | None = None,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "txyp",
    topic: str | None = None,
) -> EventReader:
    """Open a file for lazy slicing without loading it whole.

    Where :func:`load` is OpenCV's ``imread`` (read the entire stream eagerly),
    ``open`` is its ``VideoCapture``: it returns an :class:`EventReader` that points
    at the original file and fetches a slice on demand. For HDF5 this binary-searches
    the on-disk timestamps, so a slice of a multi-gigabyte recording costs a handful
    of reads — the file is never fully materialised. Other formats are loaded once and
    sliced in memory.

    Pass ``dt_ms`` to treat the recording as a sequence of fixed-duration frames: the
    reader reports ``n_slices`` and ``reader.slice(n)`` returns the ``n``-th frame
    (``reader[n]`` works too). Frame ``n`` is measured from the recording start, so you
    never deal with absolute timestamps (which may be epoch-based). Without ``dt_ms``,
    slice by explicit time/count window instead.

    ``sensor_size`` and ``time_unit`` are **auto-detected** when omitted (see
    :func:`load`); ``order``/``topic`` match :func:`load`. For a multi-GB HDF5, pass
    ``sensor_size`` to skip the one-time coordinate scan resolution inference needs.

    Example::

        r = eventcv.open("rec.hdf5", dt_ms=30)   # resolution + time unit auto-detected
        r.n_slices                               # how many 30 ms frames
        r.slice(50).mcts().view()                # the 50th 30 ms frame
        for frame in r.windows():                # walk every frame (step defaults to dt_ms)
            voxel = frame.voxel()

        # Pass sensor_size to skip the coordinate scan on a huge HDF5:
        r2 = eventcv.open("rec.hdf5", dt_ms=30, sensor_size=(346, 260))
    """
    return _rust.open(
        path,
        dt_ms=dt_ms,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
    )


# `FrameSink` (streaming HDF5 representation writer) is only built when the extension
# includes HDF5 support; published wheels do, but keep the import resilient otherwise.
FrameSink = getattr(_rust, "FrameSink", None)


def save(obj, path: str, *, topic: str | None = None) -> None:
    """Save an :class:`EventStream` or :class:`EventFrame` to ``path``.

    The mirror of :func:`load`: the format is chosen by the file extension. Streams go to
    ``.npz``/``.txt``/``.h5``/``.bag`` (npz, HDF5, and rosbag round-trip exactly; txt stores
    ``t x y p`` and recovers the sensor size/unit on load via inference or options). Frames
    (computed representations) go to ``.npz`` or ``.h5``, preserving shape, dtype, ``kind``,
    and ``channel_names``. ``topic`` names the rosbag connection. Equivalent to ``obj.save(path)``.
    """
    return _rust.save(obj, path, topic=topic)


def load_frame(path: str) -> EventFrame:
    """Load an :class:`EventFrame` written by :func:`save` (``.npz`` or ``.h5``).

    Restores the representation's shape, dtype, ``kind``, and ``channel_names``.
    """
    return _rust.load_frame(path)


__all__ = [
    "Camera",
    "EventFrame",
    "EventPointSet",
    "EventReader",
    "EventStream",
    "FrameSink",
    "Polarity",
    "load",
    "load_frame",
    "open",
    "save",
]
