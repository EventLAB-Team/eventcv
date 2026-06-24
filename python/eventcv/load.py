from __future__ import annotations

from . import _rust

EventStream = _rust.EventStream
EventFrame = _rust.EventFrame
EventPointSet = _rust.EventPointSet
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


__all__ = ["EventFrame", "EventPointSet", "EventStream", "Polarity", "load"]
