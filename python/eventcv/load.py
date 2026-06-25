from __future__ import annotations

from . import _rust

EventStream = _rust.EventStream
EventFrame = _rust.EventFrame
EventPointSet = _rust.EventPointSet
EventReader = _rust.EventReader
Polarity = _rust.Polarity


def load(
    path: str,
    *,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str = "seconds",
    order: str = "txyp",
    topic: str | None = None,
    max_events: int | None = None,
) -> EventStream:
    """Load events from any supported file, detected by its extension.

    Supported today: ``.npz`` (N-ImageNet), ``.txt``/``.csv`` (e.g. EV-IMO
    ``t x y p``), and ``.bag`` (ROS ``dvs_msgs/EventArray``).

    ``sensor_size`` is ``(width, height)`` and is required for text files.
    ``time_unit`` (``seconds``/``milliseconds``/``microseconds``/``nanoseconds``)
    and ``order`` (``txyp``/``xytp``) apply to text files. ``topic`` selects the
    rosbag topic (default ``/davis/left/events``). ``max_events`` caps how many
    events are read, which is handy for previewing very large files.
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
    time_unit: str = "seconds",
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

    ``sensor_size``/``time_unit``/``order``/``topic`` mean the same as in :func:`load`.

    Example::

        r = eventcv.open("rec.hdf5", dt_ms=30, sensor_size=(346, 260), time_unit="ns")
        r.n_slices                       # how many 30 ms frames
        r.slice(50).mcts().view()        # the 50th 30 ms frame
        for frame in r.windows():        # walk every frame (step defaults to dt_ms)
            voxel = frame.voxel()

        # Or slice by an explicit window when no dt_ms is set:
        r2 = eventcv.open("rec.hdf5", sensor_size=(346, 260), time_unit="ns")
        r2.slice(t0_ms=r2.time_span_ms[0] + 1000, t1_ms=r2.time_span_ms[0] + 1030)
    """
    return _rust.open(
        path,
        dt_ms=dt_ms,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
    )


__all__ = [
    "EventFrame",
    "EventPointSet",
    "EventReader",
    "EventStream",
    "Polarity",
    "load",
    "open",
]
