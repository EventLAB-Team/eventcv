from __future__ import annotations

import atexit as _atexit

from . import _ort, _rust

# `Model` opens ONNX Runtime at run time, so settle which library that is — bundled, conda's,
# pip's — before anything can ask for a model. Cheap, and a no-op when the user chose one.
_ort.configure()

EventStream = _rust.EventStream
EventFrame = _rust.EventFrame
EventPointSet = _rust.EventPointSet
EventReader = _rust.EventReader
Polarity = _rust.Polarity
Camera = _rust.Camera
FEAST = _rust.FEAST


class _MissingModel:
    """Stand-in for `Model` in a build without the `onnx` feature.

    The published wheels have it, so this is only reached by a source build that left the feature
    off. Raising at construction with the actual fix beats `AttributeError: no attribute 'Model'`,
    which reads like the feature does not exist at all.
    """

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "eventcv was built without ONNX support, so Model is unavailable. Reinstall from "
            "PyPI (`pip install --force-reinstall eventcv`), or rebuild with "
            "`maturin develop --features onnx`. `eventcv --version` lists the features this "
            "build has."
        )


Model = getattr(_rust, "Model", _MissingModel)
Tracker = _rust.Tracker
UdpSender = _rust.UdpSender
UdpReceiver = _rust.UdpReceiver

# Where representations run. The session starts on the CPU — those implementations are the
# reference — and `set_device("gpu")` moves everything that has a kernel; `device=` on an individual
# call overrides it for that call. `EVENTCV_DEVICE` sets the session default for a whole process.
set_device = _rust.set_device
get_device = _rust.get_device
gpu_available = _rust.gpu_available

# Close the GPU before the interpreter tears itself down. A device left open across CPython's
# finalisation faults intermittently on some drivers — after the script's last line, so it looks
# like a crash in whatever ran last rather than in the teardown. A no-op when no GPU was used.
_atexit.register(_rust.gpu_shutdown)


def load(
    path: str,
    *,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "txyp",
    topic: str | None = None,
    max_events: int | None = None,
    offset: float | None = None,
    offset_s: float | None = None,
    offset_ms: float | None = None,
    offset_us: float | None = None,
    offset_ns: float | None = None,
    keys: dict[str, str] | None = None,
) -> EventStream:
    """Load events from any supported file, detected by its extension.

    Supported today: ``.npz`` (N-ImageNet), ``.txt``/``.csv`` (e.g. EV-IMO
    ``t x y p``), ``.bag`` (ROS ``dvs_msgs/EventArray``), ``.hdf5``/``.h5``,
    ``.aedat`` (AEDAT 2.0, jAER/DAVIS), ``.aedat4`` (AEDAT 4.0, iniVation DV),
    ``.dat`` (Prophesee CD events), and Prophesee EVT2 and EVT3 ``.raw`` recordings.

    A `.raw` file's encoding comes from its header, and its sensor size from ``% geometry`` when
    present — some recorders omit it, and the size is then derived from the events themselves,
    which is a tight bound on what fired rather than the physical sensor. Pass ``sensor_size``
    to override.

    An ``.aedat`` file's geometry comes from the chip named in its header (the DAVIS family;
    DVS128 and older chips use a different address layout and are refused rather than misread);
    an ``.aedat4`` file's comes from the stream description DV writes into its header. In both,
    the APS frames and IMU samples are not events and so are not returned here — read them with
    :func:`read_frames` and :func:`read_imu`.

    ``sensor_size`` and ``time_unit`` are **auto-detected** when omitted and only act
    as overrides: rosbags carry both in the message; HDF5/text infer the time unit
    from the timestamps (a fractional text value means seconds) and the resolution
    from the coordinate range. Passing ``sensor_size`` for HDF5 also skips that scan.
    ``time_unit`` is ``seconds``/``milliseconds``/``microseconds``/``nanoseconds`` (or
    ``auto``); ``order`` (``txyp``/``xytp``) applies to headerless text. ``topic`` selects the
    rosbag topic (default ``/davis/left/events``). ``offset`` is an **absolute timestamp**
    in the file's own time base — the same base as ``stream.numpy()[:, 2]``: events before it
    are skipped, and ``max_events`` then caps how many are kept *after* it — together they read
    a window, handy for previewing a slice of a very large file. For a recording whose
    timestamps are epoch-based, pass the epoch time (e.g. ``offset=1_587_540_271_650``); ``<= 0``
    reads from the start. The bare ``offset`` is milliseconds; ``offset_s``, ``offset_ms``,
    ``offset_us``, and ``offset_ns`` say which unit you mean, and passing two of them raises.

    **The x/y/t/p columns are found automatically**, whatever they're named or nested:
    an HDF5 file's datasets are searched recursively and matched by synonym (``x``,
    ``x_coordinates``, ``u``, …; ``t``/``timestamp``/``timestamps``; ``polarity``/``pol``;
    etc.), and a single compound/structured dataset with those fields is read too — so the
    common ROS ``dvs_msgs`` layout (``events/{x_coordinates, y_coordinates, timestamps,
    polarities}``) just works. A ``.csv``/``.txt`` file with a header row is mapped by its
    column names (comma **or** whitespace separated). When detection can't identify the
    columns it raises, listing what the file contains. Pass ``keys`` to name them explicitly:
    ``keys={"x": …, "y": …, "t": …, "p": …}`` — for HDF5 each value is a dataset path (or
    ``dataset/field`` to pick a compound field); for text a header name or 0-based column
    index. ``keys`` overrides auto-detection (and ``order``).
    """
    return _rust.load(
        path,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
        max_events=max_events,
        offset=offset,
        offset_s=offset_s,
        offset_ms=offset_ms,
        offset_us=offset_us,
        offset_ns=offset_ns,
        keys=keys,
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
    dt_s: float | None = None,
    dt_us: float | None = None,
    dt_ns: float | None = None,
    max_events: int | None = None,
    offset: float | None = None,
    offset_s: float | None = None,
    offset_ms: float | None = None,
    offset_us: float | None = None,
    offset_ns: float | None = None,
    repr: str | None = None,
    sensor_size: tuple[int, int] | None = None,
    time_unit: str | None = None,
    order: str = "txyp",
    topic: str | None = None,
    hot_pixel_filter: bool = False,
    hot_pixel_std: float = 3.0,
    keys: dict[str, str] | None = None,
) -> EventReader:
    """Open a file for lazy slicing without loading it whole.

    Where :func:`load` is OpenCV's ``imread`` (read the entire stream eagerly),
    ``open`` is its ``VideoCapture``: it returns an :class:`EventReader` that points
    at the original file and fetches a slice on demand. HDF5 binary-searches the on-disk
    timestamps; Prophesee EVT3 ``.raw`` uses a sparse byte/time index. In both cases a
    slice of a multi-gigabyte recording is decoded on demand and the file is never fully
    materialised. Other formats are loaded once and sliced in memory.

    Pass ``dt_ms`` to treat the recording as a sequence of fixed-duration frames: the
    reader reports ``n_slices`` and ``reader.slice(n)`` returns the ``n``-th frame
    (``reader[n]`` works too). Frame ``n`` is measured from the recording start, so you
    never deal with absolute timestamps (which may be epoch-based). ``max_events`` is
    the event-count twin: ``open(path, max_events=10_000)`` makes each slice exactly
    10 000 consecutive events (the last one may be shorter), which keeps the event rate
    per frame constant instead of the duration. The two are mutually exclusive — pass
    one or the other, not both. Without either, slice by explicit time/count window
    instead.

    **Any timescale works.** Every time argument comes in four units — ``dt_s``, ``dt_ms``,
    ``dt_us``, ``dt_ns`` — and they mean the same thing, so ``dt_us=500`` and ``dt_ms=0.5`` open
    the same reader. Passing two units for the same quantity raises rather than picking one. The
    same applies to ``offset_*`` here, to ``t0_*``/``t1_*`` on :meth:`EventReader.slice`, and to
    ``step_*``/``span_*`` on :meth:`EventReader.windows`. Timestamps are stored in microseconds,
    so a ``_ns`` value is rounded to the nearest microsecond, and a duration below half a
    microsecond raises rather than being silently rounded up::

        ecv.open("rec.h5", dt_us=500)                  # 500 us frames
        ecv.open("rec.h5", dt_s=0.03).slice(t0_us=1e6, t1_us=2e6)

    ``offset`` is an **absolute timestamp** in the file's own time base — exactly what
    ``slice(t0_ms=…)`` takes — that moves the framing origin: ``slice(0)``, ``windows()``, and
    ``n_slices`` all begin at that time, and events before it fall outside every indexed frame.
    It is clamped up to ``t_min``, so an offset before the recording is a no-op, and one past the
    end yields zero frames. For an epoch-based recording pass the epoch time (e.g.
    ``offset=1_587_540_271_650``). It composes with either ``dt_ms`` or ``max_events``. The bare
    ``offset`` is milliseconds; ``offset_s``/``offset_ms``/``offset_us``/``offset_ns`` are explicit.

    ``sensor_size`` and ``time_unit`` are **auto-detected** when omitted (see
    :func:`load`); ``order``/``topic`` match :func:`load`. For a multi-GB HDF5, pass
    ``sensor_size`` to skip the one-time coordinate scan resolution inference needs.

    The x/y/t/p columns are **found automatically** whatever their names or nesting (HDF5
    synonym/recursive/compound detection; text header rows), exactly as in :func:`load`.
    Pass ``keys={"x": …, "y": …, "t": …, "p": …}`` to name them explicitly when a file's
    layout can't be guessed (HDF5 dataset paths or ``dataset/field``; text header names or
    0-based indices).

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

    When ``repr`` is set, ``slice``/``slice_count``/``windows`` also apply it: each returns the
    rendered :class:`EventFrame` (the rich object — ``.numpy()``, ``.view()``, ``.save()``),
    while ``reader[i]``/``batch`` stay dense NumPy arrays for the ``DataLoader`` path. So
    ``open(path, repr="mcts").slice(0)`` equals ``open(path).slice(0).mcts()``. Without ``repr``,
    ``slice(i)`` returns the raw :class:`EventStream`, so name the representation on the stream:
    ``data.slice(1000).view("flow")`` (or ``.slice(1000).optical_flow().view()``).

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
        dt_s=dt_s,
        dt_us=dt_us,
        dt_ns=dt_ns,
        max_events=max_events,
        offset=offset,
        offset_s=offset_s,
        offset_ms=offset_ms,
        offset_us=offset_us,
        offset_ns=offset_ns,
        repr=repr,
        sensor_size=sensor_size,
        time_unit=time_unit,
        order=order,
        topic=topic,
        hot_pixel_filter=hot_pixel_filter,
        hot_pixel_std=hot_pixel_std,
        keys=keys,
    )


# `FrameSink` (streaming HDF5 representation writer) and `EventSink` (streaming HDF5 event
# writer) are only built when the extension includes HDF5 support; published wheels do, but
# keep the import resilient otherwise.
FrameSink = getattr(_rust, "FrameSink", None)
EventSink = getattr(_rust, "EventSink", None)

# What `simulate(..., out=…)` hands back in place of an EventStream. Exported so it can be type-
# checked and referenced, not because it is meant to be constructed.
SimulationResult = _rust.SimulationResult

# `EventCamera` and the live-streaming functions are only built when the extension includes
# USB camera support (the published wheels do); keep imports resilient otherwise.
EventCamera = getattr(_rust, "EventCamera", None)

# The ROS 2 nodes are built when the extension includes the `ros2` feature. They need no ROS
# installation — the wire is Zenoh, over hiroz — but the ROS 2 side must be running
# `rmw_zenoh_cpp`, or have a `zenoh-bridge-ros2dds` in front of a DDS one.
Ros2Context = getattr(_rust, "Ros2Context", None)
Ros2Publisher = getattr(_rust, "Ros2Publisher", None)
Ros2Subscriber = getattr(_rust, "Ros2Subscriber", None)
Ros2ImagePublisher = getattr(_rust, "Ros2ImagePublisher", None)

# Sentinel for "to the end of the recording" — i64::MAX on the Rust side.
_MAX_US = (1 << 63) - 1

_NO_CAMERA = (
    "eventcv was built without USB camera support. Install a build with the `camera` feature "
    "(the published wheels include it)."
)


def list_cameras() -> list[dict]:
    """List every connected, supported USB event camera.

    Returns one dict per device with keys ``kind`` (machine name, e.g. ``"prophesee_evk4"``),
    ``name`` (model), ``serial``, ``bus``, ``address``, and ``speed``. A device's ``serial`` can
    be passed to :func:`stream` to select it when several are attached; an empty list means no
    supported camera was found.

    **Asking never fails.** An empty list is the answer on a machine with no camera, in a container
    without USB passthrough, or on a CI runner with no USB subsystem at all — none of which are
    errors. The only exception raised is when the build has no camera support compiled in.

    On Linux, accessing the device needs udev rules. If a camera is plugged in but not listed, this
    emits a :class:`RuntimeWarning` saying so — install the rules (see the neuromorphic-drivers
    README) and re-plug. It is a warning rather than an error because the list is genuinely empty
    either way; the warning only tells you *why*.
    """
    if not hasattr(_rust, "list_cameras"):
        raise RuntimeError(_NO_CAMERA)
    return _rust.list_cameras()


def stream(
    serial: str | None = None,
    *,
    dt_ms: float | None = None,
    dt_s: float | None = None,
    dt_us: float | None = None,
    dt_ns: float | None = None,
    max_events: int | None = None,
    repr: str | None = None,
    bins: int | None = None,
    window_ms: float | None = None,
    window_s: float | None = None,
    window_us: float | None = None,
    window_ns: float | None = None,
    tau_ms: float | None = None,
    tau_s: float | None = None,
    tau_us: float | None = None,
    tau_ns: float | None = None,
    max_window_ms: float | None = None,
    max_window_s: float | None = None,
    max_window_us: float | None = None,
    max_window_ns: float | None = None,
    window: int | None = None,
    normalize: bool | None = None,
    pct: float | None = None,
    white_frame: bool | None = None,
    record: str | None = None,
    compression: int | None = None,
    latest: bool = False,
    max_event_rate: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    mask=None,
    adaptive_bias: bool | dict | None = None,
    decay_ms: float | None = None,
    decay_s: float | None = None,
    decay_us: float | None = None,
    decay_ns: float | None = None,
    frames: bool = False,
    imu: bool = False,
) -> EventCamera:
    """Open a live USB event camera — the streaming twin of :func:`open`.

    Where :func:`open` turns a *file* into sliceable :class:`EventReader` windows, ``stream`` turns
    a *camera* into a live :class:`EventCamera` that yields the very same :class:`EventStream`
    windows — so every representation, transform, feature detector, and viewer in eventcv composes
    on a live feed exactly as it does on a recording.

    ``serial`` selects a specific device (from :func:`list_cameras`); ``None`` opens the first found.
    Windowing mirrors :func:`open`: pass ``dt_ms`` for fixed-duration windows **or** ``max_events``
    for a fixed event count (mutually exclusive; default ``dt_ms=30``). Only non-empty windows are
    yielded, so a loop never spins on idle time.

    Pass ``repr`` (a representation name — ``"count"``, ``"voxel"``, ``"tencode"``, …) to make
    iteration (and :meth:`EventCamera.read`) yield rendered :class:`EventFrame` s instead of raw
    streams, mirroring ``open(repr=…)``. The per-representation options are the same ones
    :meth:`EventReader.with_repr` takes — ``bins``, ``window_ms``, ``tau_ms``, ``max_window_ms``,
    ``window`` (for ``"flow"``), ``normalize``, ``pct``, and ``white_frame``. ``decay_ms`` sets the
    fade time constant of the raw :meth:`EventCamera.show` view.

    **Close the camera when you are done with it.** The device and any ``record=`` file are released
    when the camera is closed — by :meth:`EventCamera.close`, by leaving a ``with`` block, or, if you
    do neither, whenever Python happens to collect the object. Prefer the first two, and prefer
    :func:`record` over ``stream(...).record(...)`` when a script only wants a file::

        with ecv.stream(dt_ms=50) as cam:    # closed on the way out
            ...
        ecv.record("session.h5", seconds=10)  # opened, recorded, and closed in one call

    **Time spans follow the capture window.** An unset ``window_ms`` / ``tau_ms`` /
    ``max_window_ms`` defaults to ``dt_ms``, so a live representation covers exactly the events it
    was handed: ``stream(dt_ms=50, repr="tencode")`` encodes the full 50 ms instead of the 30 ms
    default (which would silently discard the oldest 20 ms of every window). Set one explicitly to
    override — ``stream(dt_ms=50, repr="tencode", window_ms=20)`` keeps only the newest 20 ms. In
    ``max_events`` mode there is no fixed duration, so the 30 ms defaults stand.

    Pass ``record`` (a path in any event format :func:`save` writes — ``.h5``, ``.npz``, ``.txt``,
    ``.csv``, ``.bag``, ``.aedat``, ``.aedat4``, ``.dat``, ``.raw``) to archive the session while you
    work: every window the loop reads has its **raw** events appended to that file first, so a
    ``repr=`` loop processes representations live and still keeps the full-resolution recording for
    later. Writing happens in Rust as each window is decoded — no per-window Python round trip — and
    is flushed about once a second, so a crash keeps everything up to a second ago.

    Formats differ in *when* the file becomes readable, which matters for a capture that might be
    interrupted: HDF5, text, AEDAT and the Prophesee formats append straight to disk and are a valid
    recording after every window, while npz and rosbag have a header or an index to write at the end
    and are only complete once the camera is closed. ``compression`` is an optional gzip level
    (``0..=9``) for HDF5; omit it for the fastest writes. The file is closed by :meth:`EventCamera.close` or
    the ``with`` block, and :attr:`EventCamera.n_recorded` counts what has been written. Iteration and
    the event GUI both keep this background recorder running; :meth:`EventCamera.record` is the
    blocking alternative when no interactive view or loop is needed.

    **Capping the source.** Every event costs time to decode, window, and render, so the cheapest
    event is one the camera never sends. ``max_event_rate`` (events per second) enables the sensor's
    on-chip event-rate controller, and ``roi=(x0, y0, width, height)`` masks every pixel outside that
    rectangle, so neither costs the host anything. A saturating scene on a 1280×720 sensor can emit
    far more than one core can decode, and capping the source is the only fix that keeps the events
    you *do* get contiguous rather than punched full of dropout holes::

        ecv.stream(dt_ms=50, max_event_rate=40_000_000)     # 40 Mev/s ceiling, enforced on-chip
        ecv.stream(dt_ms=50, roi=(320, 180, 640, 360))      # centre quarter only

    Both are on-chip on the sensors built around Prophesee's pipeline — the EVK4, the EVK3 HD, and
    the CenturyArks VGA. The iniVation cameras (DVXplorer, DAVIS346) have neither, so there ``roi=``
    falls back to a host-side mask (the same events are dropped, but only after crossing the cable
    and being decoded) and warns that it did; ``max_event_rate`` still raises, since capping the rate
    on the host would save none of the work it exists to avoid. :attr:`EventCamera.roi` reports the
    rectangle and whether it was ``"hardware"`` or ``"host"``.

    **Region of interest.** ``mask`` takes an arbitrarily shaped ROI — an ``(H, W)`` boolean array
    (or 8-bit map, where non-zero keeps the pixel) covering the sensor. Events outside it are
    dropped **as they are decoded**, so they never reach a ``record=`` file, the windows your loop
    reads, or :meth:`EventCamera.show`, and they cost nothing downstream. Where ``roi=`` is a
    rectangle fixed at open, ``mask`` is any shape and can be changed while the camera runs — so it
    suits sensors whose useful data is a circle::

        aperture = ecv.circle_mask((640, 480), cx=320, cy=240, r=230)
        with ecv.stream(dt_ms=50, mask=aperture, record="session.h5") as cam:
            for events in cam:              # masked already, as is the recording
                track(events)

    Set :attr:`EventCamera.mask` to change it mid-session (``None`` clears it), or draw one over the
    live view with :meth:`EventCamera.draw_mask`. Build masks with :func:`circle_mask`,
    :func:`ellipse_mask`, :func:`rect_mask`, :func:`polygon_mask`, or load one with
    :func:`load_mask`.

    **Adaptive biasing.** A camera left on fixed biases produces wildly different event rates as the
    light changes — the same scene can starve at dusk and saturate in sun, and anything downstream
    that was tuned on one no longer works on the other. ``adaptive_bias=True`` closes the loop:
    eventcv measures the event rate and retunes the sensor's bias currents as it runs, so the stream
    stays comparable across conditions. Two loops do it, after Nair et al., *Enhancing Visual Place
    Recognition via Fast and Slow Adaptive Biasing in Event Cameras* (IROS 2024) — a fast one that
    maps the rate onto the refractory period several times a second, and a slow one that shifts the
    photoreceptor and threshold biases whenever the fast one runs out of room::

        ecv.stream(adaptive_bias=True)                              # the paper's tuning
        ecv.stream(adaptive_bias={"target_rate": (2e5, 1e6)})       # aim for a quieter stream

    It starts from whatever biases the camera is already running (its stock configuration) and
    adjusts from there, so turning it on never jumps the picture. Pass a dict to override any of
    ``period_ms``, ``target_rate`` (a ``(low, high)`` band in events/second), ``throttle_range``,
    ``max_slew``, ``patience``, ``step``, or ``limits``; :attr:`EventCamera.bias_state` reports what
    the controller is doing, including an ``authority`` of ``"hunting"`` when the band you asked for
    is not reachable at all. Supported on the iniVation DAVIS346 and the Prophesee EVK4; other
    cameras raise rather than silently doing nothing.

    Defaults are per sensor, so ``adaptive_bias=True`` means something sensible on either — the EVK4
    has 10x the pixels and plain byte registers rather than the DAVIS346's 2041-step current ladder,
    so its rates, step sizes and bounds all differ. Anything you pass overrides just that field and
    leaves the rest to the camera. ``limits`` takes either one ``(low, high)`` for every bias or a
    dict naming them (``photoreceptor``, ``follower``, ``on_threshold``, ``off_threshold``); the
    per-bias form matters on an IMX636, whose ON and OFF thresholds must stay on opposite sides of
    its ``diff`` reference.

    **It measures your scene first.** For about the first second the controller changes nothing and
    just watches, then centres its target band on the rate the camera was actually producing. So
    ``adaptive_bias=True`` means "hold the event rate wherever this scene started" — the reference
    condition is whatever it saw at startup, and the loops then work to keep later conditions
    looking like it. That is what you want for consistency across lighting, and it avoids the trap
    of asking for a rate the scene cannot supply at any bias setting, which would drive the sensor
    into the regime where it is amplifying its own noise.

    Pass ``target_rate`` only when you need a *specific* absolute rate; doing so skips the
    measurement. ``bias_state`` reports the band that was chosen, and ``calibrating`` while it is
    still measuring::

        ecv.stream(adaptive_bias=True)                              # hold this scene's rate
        ecv.stream(adaptive_bias={"target_rate": (3e4, 1.2e5)})     # hold this exact band
        ecv.stream(adaptive_bias={"calibrate": 20})                 # measure for longer first

    This complements ``max_event_rate`` rather than replacing it: the rate limiter is a hard on-chip
    ceiling that *discards* events once you exceed it, while adaptive biasing changes what the pixels
    produce in the first place, so the events you keep stay evenly spread rather than clipped.

    Decoding, and any ``record=`` writing, run on a **background thread** that owns the camera, so
    the driver's ring is drained continuously no matter what your loop is doing — the loop only
    collects windows that are already decoded. On a Prophesee EVK4 at ``dt_ms=50`` this leaves
    ~80% of each window's wall-clock budget free for your own work: 40 ms of per-frame processing
    still holds 19.9 of an ideal 20 fps with the ring empty. The event GUI uses this same pump;
    :meth:`EventCamera.record`, :meth:`EventCamera.draw_mask`, APS viewing, and
    :meth:`EventCamera.close` take the camera back directly.

    Windows are delivered in order, so a loop whose per-window work is slower than the camera still
    falls behind — the thread buffers a few windows, then applies backpressure, and under *sustained*
    overload the driver's ring is what finally overflows (watch :attr:`EventCamera.n_overflows`).
    Pass ``latest=True`` to trade completeness for freshness instead: reads hand back the **newest**
    decoded window and drop what it overtook. Buffers queued behind the freshest one are *skimmed* —
    decoded far enough to keep the sensor's stream state exact, then discarded without being built
    into windows — which is both cheaper than building results nobody reads and, more importantly,
    keeps the driver's ring drained. That matters because a full ring backs pressure up into the
    sensor, where events are lost with nothing able to count them. Two counters say what went:
    :attr:`EventCamera.n_skipped` counts whole windows passed over and
    :attr:`EventCamera.n_skimmed_events` counts events discarded before a window was built. Both are
    exact; :attr:`EventCamera.n_overflows` remains the part the driver dropped, which only it sees.

    Skimming is off while ``record=`` is open, because the file is promised every window and a
    skimmed buffer's events are gone before the recorder could archive them. With a recording
    attached the loop still gets the newest window and the file still gets all of them, at the old
    cost — asking for a complete recording is asking for complete decoding to go with it.

    The returned camera is a context manager and an iterator::

        # Live raw event view (the default) — polarity dots with exponential decay:
        eventcv.stream().show()                      # 30 ms window, up to 60 refreshes/s

        # Live representation view:
        eventcv.stream(dt_ms=30).show("count", refresh_hz=60)

        # Iterate windows and run any eventcv op on the live feed:
        with eventcv.stream(dt_ms=30) as cam:
            for events in cam:              # each `events` is a 30 ms EventStream
                corners = events.efast()
                if done:
                    break

        # A `while` loop that returns one representation per time window — `read()` blocks for
        # the next window (here one MCTS frame per ~50 ms of events):
        cam = eventcv.stream(dt_ms=50, repr="mcts")
        while running:
            frame = cam.read()              # EventFrame (mcts); use max_events=N for count windows
            infer(frame.numpy())

        # Process representations live *and* archive the raw events in the same loop:
        with eventcv.stream(dt_ms=50, repr="mcts", record="session.h5") as cam:
            while running:
                infer(cam.read().numpy())   # raw events for this window are already on disk

        # `latest=True` keeps slow work on live data — the file still gets every window:
        with eventcv.stream(dt_ms=30, repr="count", record="session.h5", latest=True) as cam:
            while running:
                slow_inference(cam.read().numpy())
            print(cam.n_skipped, "windows skipped,", cam.n_recorded, "events recorded")

        # Record continuously, straight to disk (window-by-window, never buffering the whole
        # session), stopping after 10 s or on Ctrl+C. `eventcv.record` is the one-shot form of
        # this and closes the camera for you:
        eventcv.record("session.h5", seconds=10)

        # Or drive the recorder yourself with an EventSink, mixing capture and processing:
        with eventcv.stream(dt_ms=50) as cam, eventcv.EventSink("session.h5") as sink:
            for events in cam:
                sink.append(events)         # to disk
                track(events)               # ...and process the same window live

    **APS frames and IMU.** A DAVIS produces a greyscale video and an IMU stream alongside its
    events. Pass ``frames=True`` / ``imu=True`` to collect them, then take them with
    :meth:`EventCamera.read_frames` and :meth:`EventCamera.read_imu` — the same shapes
    :func:`read_frames` and :func:`read_imu` return for a recording, so code written against a
    file works unchanged on a live camera::

        with ecv.stream(dt_ms=50, frames=True, imu=True) as cam:
            for events in cam:
                for t_us, frame in cam.read_frames():   # EventFrame, uint16 greyscale
                    show(frame.numpy())
                gyro = cam.read_imu()["angular_velocity"]

        ecv.stream(frames=True).show("aps")             # watch the APS video live

    Both are off by default: copying a frame out of every decode buffer is not free, and other
    sensors have neither stream. Collect them as you read — only the most recent few dozen frames
    and a few thousand samples are held, and the oldest are dropped rather than buffered forever.

    Requires a build with camera support and, on Linux, udev rules for USB access.
    """
    if not hasattr(_rust, "stream"):
        raise RuntimeError(_NO_CAMERA)
    return _rust.stream(
        serial,
        dt_ms=dt_ms,
        dt_s=dt_s,
        dt_us=dt_us,
        dt_ns=dt_ns,
        max_events=max_events,
        repr=repr,
        bins=bins,
        window_ms=window_ms,
        window_s=window_s,
        window_us=window_us,
        window_ns=window_ns,
        tau_ms=tau_ms,
        tau_s=tau_s,
        tau_us=tau_us,
        tau_ns=tau_ns,
        max_window_ms=max_window_ms,
        max_window_s=max_window_s,
        max_window_us=max_window_us,
        max_window_ns=max_window_ns,
        window=window,
        normalize=normalize,
        pct=pct,
        white_frame=white_frame,
        record=record,
        compression=compression,
        latest=latest,
        max_event_rate=max_event_rate,
        roi=roi,
        mask=mask,
        adaptive_bias=adaptive_bias,
        decay_ms=decay_ms,
        decay_s=decay_s,
        decay_us=decay_us,
        decay_ns=decay_ns,
        frames=frames,
        imu=imu,
    )


def record(
    path: str,
    *,
    seconds: float | None = None,
    serial: str | None = None,
    dt_ms: float | None = None,
    dt_s: float | None = None,
    dt_us: float | None = None,
    dt_ns: float | None = None,
    max_events: int | None = None,
    compression: int | None = None,
    max_event_rate: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    mask=None,
    adaptive_bias: bool | dict | None = None,
) -> int:
    """Record a camera to ``path`` in one call, and return the number of events saved.

    The one-shot form of ``stream(...).record(...)``, and the one to reach for when a script only
    wants a file: the camera is opened, captured from for ``seconds`` (or until ``Ctrl+C``), and
    **closed before this returns** — so the recording is complete and the device is free by the time
    the next line runs::

        ecv.record("session.h5", seconds=10)
        reader = ecv.open("session.h5", dt_ms=50)      # safe: nothing is still writing

    The format follows the extension, as in :func:`save`, and every one of them is written
    continuously, window-by-window — a recording is never held in memory, however long it runs.
    ``compression`` is an optional gzip level (``0..=9``) for HDF5. npz and bag are the two that
    only become readable once the recording closes, since both have something to write at the end.

    Everything :func:`stream` takes about *what the sensor sends* applies — ``serial``, ``dt_ms`` /
    ``max_events``, ``roi``, ``mask``, ``max_event_rate``, ``adaptive_bias``. The representation and
    viewer options do not, since nothing is displayed. Open a camera with :func:`stream` instead when
    you want to process windows as they arrive.
    """
    if not hasattr(_rust, "record"):
        raise RuntimeError(_NO_CAMERA)
    return _rust.record(
        path,
        seconds=seconds,
        serial=serial,
        dt_ms=dt_ms,
        dt_s=dt_s,
        dt_us=dt_us,
        dt_ns=dt_ns,
        max_events=max_events,
        compression=compression,
        max_event_rate=max_event_rate,
        roi=roi,
        mask=mask,
        adaptive_bias=adaptive_bias,
    )


def save(
    obj,
    path: str,
    *,
    topic: str | None = None,
    format: str | None = None,
    compression: int | bool | None = None,
) -> None:
    """Save an :class:`EventStream`, :class:`EventFrame`, :class:`EventReader`, or :class:`FEAST`
    model to ``path``.

    The mirror of :func:`load`: the format is chosen by the file extension. Streams go to
    ``.npz``/``.txt``/``.h5``/``.bag`` (npz, HDF5, and rosbag round-trip exactly; txt stores
    ``t x y p`` and recovers the sensor size/unit on load via inference or options). Frames
    (computed representations) go to ``.npz`` or ``.h5``, preserving shape, dtype, ``kind``,
    and ``channel_names``. A trained :class:`FEAST` model is written to ``.npz`` (its learned
    features, thresholds, and parameters) and reloaded with :func:`load_feast`. ``topic`` names
    the rosbag connection. Equivalent to ``obj.save(path)`` for streams/frames.

    **E2VID export.** A ``.zip`` target (or ``format="e2vid"`` on a ``.txt``) writes the layout
    `E2VID <https://github.com/uzh-rpg/rpg_e2vid>`_ reads — a ``width height`` header, then
    ``t x y p`` per event with ``t`` in float seconds — so a recording goes straight into
    event-to-video reconstruction without a conversion script::

        ecv.save(ecv.open("rec.h5"), "events.zip")
        # python run_reconstruction.py --input_file events.zip --fixed_duration -T 33

    Passing an :class:`EventReader` converts it **window by window**, so a recording far larger
    than memory re-exports without being loaded, and any deferred ops on the reader (``crop``,
    ``mask``, ``hot_pixel_filter``, …) apply to what is written. eventcv does not read the E2VID
    layout back — keep an ``.npz``/``.h5`` if you need the recording itself.

    **HDF5 compression.** Events are stored as four columns — ``x``/``y`` (``uint16``), ``t``
    (``int64``) and ``p`` (``uint8``) — which is 13 raw bytes an event, so a large recording is
    enormous stored plainly. ``.h5`` is therefore chunked, byte-shuffled and gzipped by default,
    which typically cuts it by more than half at some cost in write speed. Pass
    ``compression=False`` for the fastest possible write, or ``1``–``9`` to pick a level::

        ecv.save(events, "rec.h5")                     # gzip 1, the default
        ecv.save(events, "rec.h5", compression=False)  # fastest, largest
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
    return _rust.save(obj, path, topic=topic, format=format, compression=compression)


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


def simulate(
    source,
    *,
    pos_thres: float = 0.2,
    neg_thres: float = 0.2,
    sigma_thres: float = 0.03,
    refractory_us: int = 100,
    cutoff_hz: float = 200.0,
    leak_rate_hz: float = 1.0,
    shot_noise_rate_hz: float = 10.0,
    seed: int = 0,
    upsample: str | None = None,
    max_events_per_pixel: float = 1.0,
    max_upsample: int | None = None,
    fps: float | None = None,
    scale: tuple[int, int] | None = None,
    max_frames: int | None = None,
    out: str | None = None,
    compression: int | bool | None = None,
    progress: bool = False,
    device: str | None = None,
    interpolate: str | None = None,
    interpolate_factor: int = 2,
):
    """Simulate a DVS camera watching ``source``, returning the events it would have produced.

    ``source`` is a **video file path** (which carries its own frame rate) or an **array of
    frames** — ``[N, H, W]`` greyscale or ``[N, H, W, 3]`` RGB, integer ``0..255`` or float
    ``0..1`` — which needs an explicit ``fps``. Writing ``.mp4`` and reading video both go
    through ``ffmpeg``, so a video source needs it on ``PATH``.

    The pixel model follows v2e (Hu et al., CVPRW 2021). Frames are linearised out of sRGB and
    mapped lin-log before differencing, so contrast is measured in light rather than in
    display-encoded values::

        events = ecv.simulate("clip.mp4")                  # realistic defaults
        events = ecv.simulate(frames, fps=1000)            # from an array
        events = ecv.simulate("clip.mp4", sigma_thres=0.0, leak_rate_hz=0,
                              shot_noise_rate_hz=0)        # an ideal, noiseless sensor

    **Writing straight to a file.** A realistic sensor emits a *lot* — a second of 1080p is on the
    order of a hundred million events, which is many gigabytes held in RAM before you can save any
    of it. ``out=`` writes each frame interval as it is produced and returns a
    :class:`SimulationResult` (``.path``, ``.frames``, ``.events``; ``len()`` is the event count)
    instead of an :class:`EventStream`, so memory stays flat at one interval::

        result = ecv.simulate("clip.mp4", out="events.h5", progress=True)
        reader = ecv.open(result.path, dt_ms=20)

    ``.h5`` streams; other formats are buffered and written once at the end. ``compression`` picks
    the HDF5 filter (gzip 1 by default, ``False`` for none), and ``progress=True`` prints frames
    and event counts to stderr as it runs.

    Parameters mirror the sensor: ``pos_thres``/``neg_thres`` are the log-contrast thresholds,
    ``sigma_thres`` their per-pixel spread, ``cutoff_hz`` the photoreceptor bandwidth for a white
    pixel, ``leak_rate_hz`` spontaneous ON events, ``shot_noise_rate_hz`` the noise floor in dark
    pixels, and ``refractory_us`` the dead time after each event. Set the noise terms to ``0`` for
    a clean stream; the defaults are what a real camera does.

    ``upsample`` subdivides each frame interval before simulating, which is what keeps timestamps
    accurate when a lot happens between two frames: ``"adaptive"`` (the default) subdivides until
    no pixel would emit more than ``max_events_per_pixel`` per sub-interval, ``"off"`` disables it,
    and an integer string is a fixed factor. Each sub-step is a full pass over every pixel and the
    *busiest* pixel sets the factor, so one hard edge can cost the maximum on an otherwise quiet
    frame — ``max_upsample`` (default 64) caps that, trading timestamp accuracy for time.

    ``scale`` decodes video at a different resolution, which is both much cheaper than resizing
    afterwards and the most effective way to cut the event count — halving each side quarters the
    pixels and roughly quarters the output. ``max_frames`` stops early, and ``seed`` makes a run
    reproducible from its configuration (including across machines: the simulation runs on every
    core, but its noise is seeded so that the result does not depend on how many there are).

    ``device="gpu"`` runs the pixel model as a compute shader instead, which is worth it on a large
    sensor: this is the heaviest loop in the library, and the one place the GPU wins clearly. The
    two backends are not interchangeable to the last event, and the difference is worth knowing
    before you rely on it:

    * With ``shot_noise_rate_hz=0`` and ``leak_rate_hz=0`` they produce **identical** events.
    * With ``sigma_thres`` on they produce the same events with the same polarities, and a couple of
      timestamps in ~25 000 land a microsecond apart — the crossing fraction is double precision on
      the CPU and shaders have no double precision.
    * With noise on they draw **different random numbers** by construction (a per-pixel kernel
      cannot replay a sequential generator), so the streams agree in rate, not event for event. A
      GPU run is still bit-reproducible from its seed, run to run and across devices.

    So compare a GPU run against a stored CPU one only with the noise terms off.

    ``interpolate`` inserts *learned* frames before simulating, which is the other half of the
    timestamp-accuracy problem. ``upsample`` subdivides the interval, but the levels it subdivides
    are a straight-line blend of the two frames — right when a pixel's intensity ramps over the gap,
    wrong when an edge crosses it and the pixel steps. Point ``interpolate`` at a RIFE ONNX export
    (or anything shaped like one: a frame pair in, one image out, ideally with a ``timestep`` input)
    and ``interpolate_factor`` says how many intervals each source pair becomes::

        events = ecv.simulate("clip.mp4", interpolate="rife_v4.onnx", interpolate_factor=4)

    The interpolated frames are fed to the simulator as ordinary source frames at proportional
    timestamps, exactly as v2e uses Super-SloMo — the pixel model is untouched, and ``upsample``
    still refines whatever is left. Frames are interpolated in luma, since that is all the simulator
    consumes. No weights ship with eventcv; export the model yourself.
    """
    return _rust.simulate(
        source,
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=sigma_thres,
        refractory_us=refractory_us,
        cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        seed=seed,
        upsample=upsample,
        max_events_per_pixel=max_events_per_pixel,
        max_upsample=max_upsample,
        fps=fps,
        scale=scale,
        max_frames=max_frames,
        out=out,
        compression=compression,
        progress=progress,
        device=device,
        interpolate=interpolate,
        interpolate_factor=interpolate_factor,
    )


def save_video(
    source,
    path: str,
    *,
    fps: float = 30.0,
    repr: str | None = None,
    dt_ms: float | None = None,
    decay_ms: float | None = None,
    colormap: str = "viridis",
    clim: float | None = None,
    max_frames: int | None = None,
) -> int:
    """Render ``source`` as an animation, returning the number of frames written.

    ``source`` is an :class:`EventReader` or an :class:`EventStream`; the container comes from
    ``path``'s extension (``.gif``, ``.apng``, ``.mp4`` — the last needs ``ffmpeg`` on ``PATH``).

    With no ``repr`` this writes the **raw stream**: every event lights its pixel in its polarity
    colour and fades over ``decay_ms``, which is the picture a camera's own viewer shows and needs
    no choice of representation to be meaningful::

        ecv.save_video(ecv.open("rec.h5"), "raw.mp4", dt_ms=5)
        ecv.save_video(events, "raw.gif", dt_ms=5, decay_ms=20)

    ``dt_ms`` is how much of the recording each rendered frame covers — the shutter, not the
    playback rate, which is ``fps``. Together they set the speed: ``dt_ms=5, fps=30`` plays six
    times slower than real time. Passing it here rather than at :func:`open` is deliberate; it is
    a property of the view, and a reader is often opened for something else entirely.

    Pass ``repr="count"`` (or any representation name) to render that instead, with ``colormap``
    and ``clim`` as in :meth:`EventReader.save_video`. A reader opened with ``repr=`` uses it
    unless ``repr="raw"`` overrides.
    """
    if isinstance(source, str):
        source = open(source)
    kwargs = dict(
        fps=fps,
        colormap=colormap,
        clim=clim,
        max_frames=max_frames,
        repr=repr,
        decay_ms=decay_ms,
    )
    # A stream always needs a window (it has no opened-with default to fall back on), so it takes
    # the argument's own default rather than `None`.
    if dt_ms is not None:
        kwargs["dt_ms"] = dt_ms
    return source.save_video(path, **kwargs)


def play(source=None, **kwargs) -> None:
    """Open the interactive player, optionally with an :class:`EventReader`,
    :class:`EventStream`, or path already loaded.

    The offline twin of :meth:`EventCamera.show`: with no ``repr`` it shows the raw event stream,
    polarity dots fading by age, so watching a recording takes no more setup than watching a
    camera::

        ecv.play()                              # choose a recording in the GUI
        ecv.play("rec.h5", dt_ms=5)
        ecv.play("rec.h5", repr="count", dt_ms=20)
        ecv.play(events, speed=0.25)

    At ``speed=1`` recording time matches wall time. ``dt_ms`` is the accumulation width while
    ``refresh_hz`` (default 60) controls how often that window advances, so overlapping windows are
    cheap. ``fps`` remains accepted for compatibility but is deprecated. The call blocks on the
    main thread until the window is closed.
    """
    if source is None:
        return _rust.play_gui(**kwargs)
    if isinstance(source, str):
        source = open(source)
    return source.play(**kwargs)


def reconstruct(
    reader,
    model,
    path: str,
    *,
    fps: float = 30.0,
    colormap: str = "grayscale",
    max_frames: int | None = None,
) -> int:
    """Reconstruct an intensity video from events, returning the number of frames written.

    ``reader`` must carry the representation the model expects, and ``model`` is an
    :class:`Model` wrapping any ONNX graph that maps that representation to a single-channel
    image — E2VID and its relatives::

        reader = ecv.open("recording.h5", dt_ms=33, repr="voxel", bins=5)
        ecv.reconstruct(reader, ecv.Model("e2vid.onnx"), "out.mp4")

    EventCV supplies the runner and the tensors, not the weights: there is no official E2VID ONNX
    export, so export one yourself with ``torch.onnx.export``. A recurrent model must have its
    hidden state exposed as explicit inputs and outputs; a stateless export (``--no-recurrent``)
    works as-is.

    The output format comes from ``path``'s extension (``.gif``, ``.apng``, ``.mp4``), and the
    reconstruction is rendered at its natural scale rather than auto-contrasted, so brightness
    stays consistent across the sequence.
    """
    if Model is _MissingModel:
        raise RuntimeError(
            "eventcv was built without ONNX support, so reconstruct is unavailable. "
            "Reinstall from PyPI, or rebuild with `maturin develop --features onnx`."
        )
    return _rust.reconstruct(
        reader,
        model,
        path,
        fps=fps,
        colormap=colormap,
        max_frames=max_frames,
    )


class StatefulModel:
    """Drives a recurrent ONNX model, carrying its hidden state between calls.

    E2VID and its relatives take ``(data, state_0, …)`` and return ``(image, new_state_0, …)``;
    what makes them recurrent is feeding those states back, which a plain :class:`Model` call
    deliberately does not do. ``state_map`` says which output feeds which input::

        model = ecv.StatefulModel(
            ecv.Model("e2vid_recurrent.onnx"),
            state_map={"new_state": "state"},
        )
        for i in range(len(reader)):
            image = model(reader[i])
        model.reset()      # before an unrelated recording

    State lives here rather than inside :class:`Model` so that ``Model`` stays a pure function and
    resetting is something you ask for rather than something that happens invisibly. On the first
    call — and after :meth:`reset` — each state input is seeded with zeros of its declared shape,
    with any free dimension taken as 1.

    There is no official recurrent E2VID export; produce one with ``torch.onnx.export``, exposing
    the ConvLSTM states as explicit inputs and outputs. A stateless export needs none of this and
    works with :class:`Model` directly.
    """

    def __init__(self, model, state_map: dict[str, str]):
        self.model = model
        self.state_map = dict(state_map)
        self._state: dict = {}
        self._data_input = None
        for port in model.inputs:
            if port["name"] not in self.state_map.values():
                self._data_input = port["name"]
                break
        if self._data_input is None:
            raise ValueError(
                "every input is claimed by state_map, leaving nothing to feed the data into"
            )
        unknown = set(self.state_map.values()) - {p["name"] for p in model.inputs}
        if unknown:
            raise ValueError(f"state_map names inputs the graph does not have: {sorted(unknown)}")
        outputs = {p["name"] for p in model.outputs}
        missing = set(self.state_map) - outputs
        if missing:
            raise ValueError(f"state_map names outputs the graph does not have: {sorted(missing)}")

    def reset(self) -> None:
        """Forgets the hidden state, so the next call starts a fresh sequence."""
        self._state = {}

    def _zeros_for(self, name):
        import numpy as np

        shape = next(p["shape"] for p in self.model.inputs if p["name"] == name)
        # A free dimension (-1) is almost always the batch axis; one is the only sane choice.
        return np.zeros([1 if dim < 0 else dim for dim in shape], dtype="float32")

    def __call__(self, data):
        import numpy as np

        array = np.asarray(data.numpy() if hasattr(data, "numpy") else data, dtype="float32")
        expected = next(p["shape"] for p in self.model.inputs if p["name"] == self._data_input)
        if array.ndim + 1 == len(expected):
            array = array[None]
        inputs = {self._data_input: array}
        for output_name, input_name in self.state_map.items():
            inputs[input_name] = self._state.get(input_name, self._zeros_for(input_name))
        outputs = self.model.run_named(inputs)
        for output_name, input_name in self.state_map.items():
            self._state[input_name] = outputs[output_name]
        # The image is whatever is not state — the graph's first non-state output.
        for port in self.model.outputs:
            if port["name"] not in self.state_map:
                return outputs[port["name"]]
        raise ValueError("every output is claimed by state_map, leaving no image to return")


def bag_topics(path: str) -> list[tuple[str, str]]:
    """List every topic in a ROS bag with its message type.

    Bags disagree about names — a DAVIS recorded through ``dvs_ros_driver`` publishes
    ``/dvs/events`` where :func:`load` defaults to ``/davis/left/events`` — so this is usually the
    first thing to call on an unfamiliar file::

        >>> ecv.bag_topics("recording.bag")
        [('/dvs/camera_info', 'sensor_msgs/CameraInfo'),
         ('/dvs/events', 'dvs_msgs/EventArray'),
         ('/dvs/image_raw', 'sensor_msgs/Image'),
         ('/dvs/imu', 'sensor_msgs/Imu')]
    """
    return _rust.bag_topics(path)


def read_frames(
    source,
    *,
    topic: str | None = None,
    t0_us: int = 0,
    t1_us: int | None = None,
) -> list[tuple[int, EventFrame]]:
    """Read DAVIS APS frames as ``[(t_us, EventFrame), …]``.

    The greyscale images a DAVIS captures alongside its events — the reference the simulator is
    calibrated against, since the frames and the events see the same scene through the same lens.

    ``source`` is a path or an open :class:`EventReader`; ROS bags and AEDAT 2.0 recordings carry
    frames, and any other format returns an empty list. Frames come back at the sensor's own
    precision, so ``frame.numpy()`` is ``uint8`` from a bag (ROS ``mono8``) and ``uint16`` from
    AEDAT (a 10-bit ADC).

    ``topic`` selects the bag topic and defaults to its only ``sensor_msgs/Image`` topic when
    there is exactly one; it does not apply to other formats. The time window matters: these
    recordings run to tens of gigabytes, and whole chunks outside it are skipped without being
    decoded. From a bag only ``mono8``/``rgb8``/``bgr8`` are supported; any other encoding raises
    rather than being misread.
    """
    t1_us = _MAX_US if t1_us is None else t1_us
    if isinstance(source, EventReader):
        return source.read_frames(t0_us=t0_us, t1_us=t1_us)
    return _rust.read_frames(source, topic=topic, t0_us=t0_us, t1_us=t1_us)


def read_imu(
    source,
    *,
    topic: str | None = None,
    t0_us: int = 0,
    t1_us: int | None = None,
) -> dict:
    """Read IMU samples as a dict of arrays.

    Keys are ``t`` (µs), ``angular_velocity`` and ``linear_acceleration``, the latter two ``[N, 3]``
    in rad/s and m/s². On a DAVIS this is the only independent measurement of how the camera moved,
    which makes it the ground truth for :meth:`~eventcv.EventStream.contrast_maximise`::

        imu = ecv.read_imu("recording.bag", t0_us=t0, t1_us=t1)
        truth = imu["angular_velocity"].mean(axis=0)

    ``source`` is a path or an open :class:`EventReader`; ROS bags and AEDAT 2.0 recordings carry
    an IMU, and any other format returns empty arrays.

    Orientation is not returned: on a DAVIS it is dead-reckoned from this same gyro, so it is not
    an independent quantity. Temperature is not returned either.
    """
    t1_us = _MAX_US if t1_us is None else t1_us
    if isinstance(source, EventReader):
        return source.read_imu(t0_us=t0_us, t1_us=t1_us)
    return _rust.read_imu(source, topic=topic, t0_us=t0_us, t1_us=t1_us)


def read_camera_info(source, *, topic: str | None = None):
    """Read camera intrinsics from a recording, or ``None`` if it carries none.

    Returns a :class:`Camera` ready to pass to
    ``contrast_maximise(model="rotation", camera=…)``, which needs intrinsics to map pixels onto
    rays. ``source`` is a path or an open :class:`EventReader`; only ROS bags
    (``sensor_msgs/CameraInfo``) record intrinsics today. Distortion coefficients are not applied
    — build a :class:`Camera` yourself from the bag's ``D`` if you need undistortion.
    """
    if isinstance(source, EventReader):
        return source.read_camera_info()
    return _rust.read_camera_info(source, topic=topic)


def circle_mask(sensor_size: tuple[int, int], cx: float, cy: float, r: float):
    """Build a circular ROI mask of radius ``r`` centred on ``(cx, cy)``.

    Returns an ``(H, W)`` boolean NumPy array — ``True`` where events are **kept** — ready for
    :meth:`EventStream.mask`, :meth:`EventReader.mask`, or ``ecv.stream(mask=…)``. Note the two
    orders: ``sensor_size`` is ``(width, height)`` like everywhere else in eventcv, while the array
    that comes back is ``(H, W)`` like every other NumPy image.

    Masks are plain arrays, so they compose with NumPy's own operators — ``|`` unions, ``&``
    intersects, ``~`` inverts::

        aperture = ecv.circle_mask((640, 480), cx=320, cy=240, r=230)
        aperture &= ~ecv.rect_mask((640, 480), 0, 0, 64, 64)   # minus a corner
        ecv.load("rec.npz").mask(aperture)

    Coordinates are continuous, not pixel indices: a pixel is kept when its centre falls inside the
    shape, and geometry reaching off the sensor is clamped. See also :func:`ellipse_mask`,
    :func:`rect_mask`, :func:`polygon_mask`, :func:`load_mask`, and — to draw one by hand —
    :meth:`EventStream.draw_mask` / :meth:`EventCamera.draw_mask`.
    """
    return _rust.circle_mask(sensor_size, cx, cy, r)


def ellipse_mask(
    sensor_size: tuple[int, int], cx: float, cy: float, rx: float, ry: float
):
    """Build an elliptical ROI mask centred on ``(cx, cy)`` with semi-axes ``rx``, ``ry``.

    The axis-independent form of :func:`circle_mask`; see it for the conventions.
    """
    return _rust.ellipse_mask(sensor_size, cx, cy, rx, ry)


def rect_mask(
    sensor_size: tuple[int, int], x0: float, y0: float, width: float, height: float
):
    """Build a rectangular ROI mask keeping the ``width``×``height`` box at ``(x0, y0)``.

    Unlike :meth:`EventStream.crop`, which shrinks the sensor grid, a mask keeps the coordinates
    (and the sensor size) unchanged and only drops events. See :func:`circle_mask` for the
    conventions.
    """
    return _rust.rect_mask(sensor_size, x0, y0, width, height)


def polygon_mask(sensor_size: tuple[int, int], points):
    """Build an ROI mask keeping the interior of the closed polygon through ``points``.

    ``points`` is a sequence of ``(x, y)`` vertices; the last joins back to the first. Filled by
    the even-odd rule, so a self-intersecting outline leaves holes. See :func:`circle_mask` for the
    conventions.
    """
    return _rust.polygon_mask(sensor_size, [tuple(point) for point in points])


def save_mask(mask, path: str) -> None:
    """Write an ROI mask to an 8-bit greyscale ``.png`` — white keeps, black drops.

    Lets an ROI drawn or computed once be reused across sessions (and checked in any image viewer).
    Read it back with :func:`load_mask`.
    """
    return _rust.save_mask(mask, path)


def load_mask(path: str):
    """Read an ROI mask from a ``.png`` as an ``(H, W)`` boolean array.

    A pixel is kept where the image is **non-black and not fully transparent**, so a mask binarised
    in any other tool loads as-is — greyscale, palette, or colour, with or without alpha::

        ecv.stream(dt_ms=50, mask=ecv.load_mask("aperture.png"))

    The counterpart of :func:`save_mask`.
    """
    return _rust.load_mask(path)


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
    "EventCamera",
    "EventFrame",
    "EventPointSet",
    "EventReader",
    "EventSink",
    "EventStream",
    "FEAST",
    "FrameSink",
    "Model",
    "Polarity",
    "Ros2Context",
    "Ros2ImagePublisher",
    "Ros2Publisher",
    "Ros2Subscriber",
    "SimulationResult",
    "StatefulModel",
    "Tracker",
    "UdpReceiver",
    "UdpSender",
    "circle_mask",
    "get_device",
    "gpu_available",
    "set_device",
    "bag_topics",
    "collate",
    "ellipse_mask",
    "export_png",
    "from_numpy",
    "list_cameras",
    "load",
    "load_feast",
    "load_frame",
    "load_mask",
    "open",
    "play",
    "polygon_mask",
    "read_camera_info",
    "read_frames",
    "read_imu",
    "record",
    "reconstruct",
    "rect_mask",
    "save",
    "save_mask",
    "save_video",
    "simulate",
    "stream",
]


# ---------------------------------------------------------------------------
# Functional (OpenCV-style) call API — §D1.
#
# Every op that exists as a *method* on a stream, frame, reader, point set, or camera is also
# exposed here as a free function taking the object as its first argument, so ``ecv.flip_x(x)``
# reads like ``cv2.resize(img, …)`` and mirrors ``x.flip_x()`` exactly. A stream op applied to an
# ``EventReader`` (e.g. ``ecv.hot_pixel_filter(reader)``) forwards to the reader's own method,
# which defers it lazily onto every slice. The methods stay the single source of truth; these
# forwarders are generated by introspection so they can never drift from the Rust definitions.
# Names already curated above (``save``, ``load``, the classes, …) and read-only properties
# (``sensor_size``, ``shape``, …) are skipped.
# ---------------------------------------------------------------------------

# Every compiled type whose public methods get a free-function form.
_OP_CLASSES = (EventStream, EventFrame, EventReader, EventPointSet, Camera)


def _make_op(name: str):
    def _op(obj, *args, **kwargs):
        return getattr(obj, name)(*args, **kwargs)

    doc = next(
        (
            getattr(getattr(cls, name, None), "__doc__", None)
            for cls in _OP_CLASSES
            if getattr(getattr(cls, name, None), "__doc__", None)
        ),
        None,
    )
    summary = doc.strip().splitlines()[0] if doc else f"Calls ``obj.{name}(...)``."
    _op.__name__ = name
    _op.__qualname__ = name
    _op.__doc__ = f"{summary}\n\nFree-function form of ``obj.{name}(*args, **kwargs)``."
    return _op


_op_names = sorted(
    {
        name
        for cls in _OP_CLASSES
        for name in dir(cls)
        if not name.startswith("_")
        and callable(getattr(cls, name))
        and name not in __all__
    }
)

for _name in _op_names:
    globals()[_name] = _make_op(_name)

__all__ += _op_names
