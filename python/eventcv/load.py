from __future__ import annotations

from . import _rust

EventStream = _rust.EventStream
EventFrame = _rust.EventFrame
EventPointSet = _rust.EventPointSet
EventReader = _rust.EventReader
Polarity = _rust.Polarity
Camera = _rust.Camera
FEAST = _rust.FEAST


def load(
    path: str,
    *,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "txyp",
    topic: str | None = None,
    max_events: int | None = None,
    offset: float | None = None,
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
    rosbag topic (default ``/davis/left/events``). ``offset`` is an **absolute timestamp
    in milliseconds** (the file's own time base — the same base as ``stream.numpy()[:, 2]``
    scaled to ms): events before it are skipped, and ``max_events`` then caps how many are
    kept *after* it — together they read a window, handy for previewing a slice of a very
    large file. For a recording whose timestamps are epoch-based, pass the epoch time in ms
    (e.g. ``offset=1_587_540_271_650``); ``<= 0`` reads from the start.
    """
    return _rust.load(
        path,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
        max_events=max_events,
        offset=offset,
    )


def from_numpy(
    events,
    *,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "xytp",
) -> EventStream:
    """Build an :class:`EventStream` from an in-memory ``(N, 4)`` NumPy array.

    The constructor mirror of :meth:`EventStream.numpy`: ``order`` defaults to ``xytp``
    (the column layout ``numpy()`` emits, with timestamps in microseconds), so
    ``ecv.from_numpy(stream.numpy(), time_unit="us")`` round-trips a stream. Pass
    ``order="txyp"`` for arrays in the common ``t x y p`` dataset layout. Any integer or
    float dtype is accepted; polarity is positive when its value is greater than zero
    (both ``0/1`` and ``-1/1`` conventions work).

    ``sensor_size`` and ``time_unit`` are **auto-detected** when omitted, exactly like
    :func:`load`: the sensor is the smallest grid holding every event, and the time unit
    is inferred from the timestamp span (fractional values mean seconds; the inference
    assumes a recording of at least ~1 s, so pass ``time_unit`` explicitly for short
    arrays). Events outside an explicit ``sensor_size`` are dropped.

    Example::

        events = np.array([[0, 0, 100, 1], [1, 2, 250, 0]])   # x y t p
        stream = ecv.from_numpy(events, time_unit="us")
        stream.count().numpy()
    """
    return _rust.from_numpy(
        events,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
    )


def open(
    path: str,
    *,
    dt_ms: float | None = None,
    max_events: int | None = None,
    offset: float | None = None,
    repr: str | None = None,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "txyp",
    topic: str | None = None,
    hot_pixel_filter: bool = False,
    hot_pixel_std: float = 3.0,
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
    never deal with absolute timestamps (which may be epoch-based). ``max_events`` is
    the event-count twin: ``open(path, max_events=10_000)`` makes each slice exactly
    10 000 consecutive events (the last one may be shorter), which keeps the event rate
    per frame constant instead of the duration. The two are mutually exclusive — pass
    one or the other, not both. Without either, slice by explicit time/count window
    instead.

    ``offset`` is an **absolute timestamp in milliseconds** (the file's own time base —
    exactly what ``slice(t0_ms=…)`` takes) that moves the framing origin: ``slice(0)``,
    ``windows()``, and ``n_slices`` all begin at that time, and events before it fall
    outside every indexed frame. It is clamped up to ``t_min``, so an offset before the
    recording is a no-op, and one past the end yields zero frames. For an epoch-based
    recording pass the epoch time in ms (e.g. ``offset=1_587_540_271_650``). It composes
    with either ``dt_ms`` or ``max_events``.

    ``sensor_size`` and ``time_unit`` are **auto-detected** when omitted (see
    :func:`load`); ``order``/``topic`` match :func:`load`. For a multi-GB HDF5, pass
    ``sensor_size`` to skip the one-time coordinate scan resolution inference needs.

    Pass ``hot_pixel_filter=True`` to strip *stuck* pixels consistently across the whole
    recording. ``open`` scans the file once up front, flags every pixel whose event count exceeds
    ``mean + hot_pixel_std·std`` (over the active pixels), and drops those pixels from every slice
    it returns — the filter runs before any per-slice op (``efast`` / ``repr`` / …). This is
    deliberately **global**: calling :meth:`EventStream.hot_pixel_filter` per slice re-thresholds
    each window, so genuinely hot pixels survive at long ``dt_ms``. The pre-scan is kept
    lightweight — it reads only the event coordinates (skipping the timestamp/polarity columns)
    and, for a large recording, samples windows spread evenly across it rather than every event, so
    the mask is a robust estimate rather than an exact tally. ``hot_pixel_std`` (default ``3.0``)
    tunes how aggressive it is, matching :meth:`EventStream.hot_pixel_filter`.

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
        max_events=max_events,
        offset=offset,
        repr=repr,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
        hot_pixel_filter=hot_pixel_filter,
        hot_pixel_std=hot_pixel_std,
    )


# `FrameSink` (streaming HDF5 representation writer) is only built when the extension
# includes HDF5 support; published wheels do, but keep the import resilient otherwise.
FrameSink = getattr(_rust, "FrameSink", None)


def save(obj, path: str, *, topic: str | None = None) -> None:
    """Save an :class:`EventStream`, :class:`EventFrame`, or :class:`FEAST` model to ``path``.

    The mirror of :func:`load`: the format is chosen by the file extension. Streams go to
    ``.npz``/``.txt``/``.h5``/``.bag`` (npz, HDF5, and rosbag round-trip exactly; txt stores
    ``t x y p`` and recovers the sensor size/unit on load via inference or options). Frames
    (computed representations) go to ``.npz`` or ``.h5``, preserving shape, dtype, ``kind``,
    and ``channel_names``. A trained :class:`FEAST` model is written to ``.npz`` (its learned
    features, thresholds, and parameters) and reloaded with :func:`load_feast`. ``topic`` names
    the rosbag connection. Equivalent to ``obj.save(path)`` for streams/frames.
    """
    if isinstance(obj, FEAST):
        import numpy as np

        np.savez(
            path,
            features=obj.feature_images(),
            thresholds=obj.thresholds,
            **obj.get_params(),
        )
        return
    return _rust.save(obj, path, topic=topic)


def load_feast(path: str) -> FEAST:
    """Load a :class:`FEAST` model written by :func:`save` (``.npz``).

    Restores the learned features, selection thresholds, and every constructor parameter, so the
    reloaded model reproduces :meth:`FEAST.transform` exactly and can resume training with
    :meth:`FEAST.fit`.
    """
    import numpy as np

    state = np.load(path)
    model = FEAST(
        n_features=int(state["n_features"]),
        patch=int(state["patch"]),
        tau_ms=float(state["tau_ms"]),
        eta=float(state["eta"]),
        delta_i=float(state["delta_i"]),
        delta_e=float(state["delta_e"]),
        per_polarity=bool(state["per_polarity"]),
        seed=int(state["seed"]),
    )
    model._load_state(state["features"], state["thresholds"])
    return model


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
    "FEAST",
    "FrameSink",
    "Polarity",
    "collate",
    "export_png",
    "from_numpy",
    "load",
    "load_feast",
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
