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
    repr: str | None = None,
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

    Pass ``repr`` (a representation name — ``"count"``, ``"voxel"``, ``"tsurf"``, ``"flow"``
    for optical flow, …) to make the reader a PyTorch-style **map dataset**:
    ``len(reader) == n_slices``, ``reader[i]`` returns the dense ``[C, H, W]`` array for frame
    ``i``, and ``reader.batch(indices)`` stacks a ``[B, C, H, W]`` batch — so a ``DataLoader``
    can collate the reader directly. Use ``reader.with_repr(name, **opts)`` to set
    per-representation options (e.g. ``bins=5``, or ``window=5`` for ``"flow"``). Without
    ``repr``, ``reader[i]`` stays a raw :class:`EventStream`; to still batch those through a
    ``DataLoader`` pass ``collate_fn=eventcv.collate`` (each batch is a ``list[EventStream]``,
    since sparse streams can't stack into a tensor).

    Phase 5 algorithms apply **per slice**: ``reader.efast()`` / ``reader.harris_corners(thr)``
    return a new reader whose every slice is the corner sub-stream, composing with
    ``slice``/``windows``/``with_repr``. To render an algorithm as a video, map it over
    ``windows()`` and hand the frames to :func:`export_png` (then assemble with ffmpeg).

    Note ``repr`` governs the **array/dataset** path (``reader[i]`` → NumPy). To *view* a slice
    interactively, ``slice(i)`` returns the raw :class:`EventStream`, so name the representation
    on the stream: ``data.slice(1000).view("flow")`` (or ``.slice(1000).optical_flow().view()``).

    Example::

        r = eventcv.open("rec.hdf5", dt_ms=30)   # resolution + time unit auto-detected
        r.n_slices                               # how many 30 ms frames
        r.slice(50).mcts().view()                # the 50th 30 ms frame
        for frame in r.windows():                # walk every frame (step defaults to dt_ms)
            voxel = frame.voxel()

        # As a training dataset:
        ds = eventcv.open("rec.hdf5", dt_ms=30, repr="count")
        loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

        # Corner-detection video (one PNG per frame):
        corners = eventcv.open("rec.hdf5", dt_ms=30).efast()
        eventcv.export_png((w.count() for w in corners.windows()), "corners/", colormap="turbo")
        # Optical-flow video:
        eventcv.export_png((w.optical_flow() for w in r.windows()), "flow/")
    """
    return _rust.open(
        path,
        dt_ms=dt_ms,
        repr=repr,
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


def export_png(
    frames,
    out_dir: str,
    *,
    colormap: str = "viridis",
    normalize: bool = True,
    prefix: str = "frame_",
    start: int = 0,
    digits: int = 5,
):
    """Write one or many :class:`EventFrame` s to numbered ``.png`` files — the
    "frame sequence → video frames" export.

    ``frames`` is a single :class:`EventFrame` or any iterable of them (e.g. a
    generator over a reader's windows), so a whole recording renders lazily without
    materialising every frame at once::

        r = eventcv.open("rec.hdf5", dt_ms=30)
        eventcv.export_png((w.count() for w in r.windows()), "out/", colormap="turbo")

    Each frame is colormapped through the same path as ``frame.save("x.png")``
    (``colormap``: ``viridis``/``turbo``/``grayscale``/``redblue``; ``normalize``
    auto-contrasts). Files are named ``{prefix}{index:0{digits}d}.png`` counting from
    ``start``. Returns the list of written paths (assemble a video with, e.g.,
    ``ffmpeg -i out/frame_%05d.png out.mp4``).
    """
    import os

    if isinstance(frames, EventFrame):
        frames = [frames]

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for offset, frame in enumerate(frames):
        path = os.path.join(out_dir, f"{prefix}{start + offset:0{digits}d}.png")
        frame.save(path, colormap=colormap, normalize=normalize)
        paths.append(path)
    return paths


def collate(batch):
    """``collate_fn`` for :class:`torch.utils.data.DataLoader` over an :class:`EventReader`.

    A reader opened **with** a representation (``open(repr=…)``) yields dense ``[C, H, W]``
    arrays that torch's default collate stacks into a ``[B, C, H, W]`` tensor with no help —
    so you only need this for a reader opened **without** ``repr``, whose ``reader[i]`` is a
    raw :class:`EventStream`. Those are variable-length and sparse, so they can't stack into a
    tensor; this returns the batch as a plain ``list`` of streams instead (dense/array batches
    still defer to torch's default collate). Pass it explicitly::

        loader = torch.utils.data.DataLoader(reader, batch_size=32, collate_fn=eventcv.collate)
        for batch in loader:        # batch is a list[EventStream]
            batch[0].view()
    """
    if batch and isinstance(batch[0], EventStream):
        return list(batch)
    from torch.utils.data import default_collate

    return default_collate(batch)


__all__ = [
    "Camera",
    "EventFrame",
    "EventPointSet",
    "EventReader",
    "EventStream",
    "FrameSink",
    "Polarity",
    "collate",
    "export_png",
    "load",
    "load_frame",
    "open",
    "save",
]


# ---------------------------------------------------------------------------
# Functional (OpenCV-style) call API — §D1.
#
# Every op that exists as a *method* on a stream or frame is also exposed here as a
# free function taking the object as its first argument, so ``ecv.flip_x(stream)`` reads
# like ``cv2.resize(img, …)`` and mirrors ``stream.flip_x()`` exactly. The methods stay
# the single source of truth; these forwarders are generated by introspection so they can
# never drift from the Rust definitions. Names already curated above (``save``, ``load``,
# the classes, …) and read-only properties (``sensor_size``, ``shape``, …) are skipped.
# ---------------------------------------------------------------------------


def _make_op(name: str):
    def _op(obj, *args, **kwargs):
        return getattr(obj, name)(*args, **kwargs)

    doc = getattr(getattr(EventStream, name, None), "__doc__", None) or getattr(
        getattr(EventFrame, name, None), "__doc__", None
    )
    summary = doc.strip().splitlines()[0] if doc else f"Calls ``obj.{name}(...)``."
    _op.__name__ = name
    _op.__qualname__ = name
    _op.__doc__ = f"{summary}\n\nFree-function form of ``obj.{name}(*args, **kwargs)``."
    return _op


_op_names = sorted(
    {
        name
        for cls in (EventStream, EventFrame)
        for name in dir(cls)
        if not name.startswith("_")
        and callable(getattr(cls, name))
        and name not in __all__
    }
)

for _name in _op_names:
    globals()[_name] = _make_op(_name)

__all__ += _op_names
