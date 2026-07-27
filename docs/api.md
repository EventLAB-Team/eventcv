# API Reference

The Python API lives in the top-level `eventcv` package. Every operation exists both as
a **method** on a core type ({class}`~eventcv.EventStream`, {class}`~eventcv.EventFrame`,
{class}`~eventcv.EventReader`, {class}`~eventcv.EventPointSet`, {class}`~eventcv.Camera`)
and as an OpenCV-style **free function** (listed under
[Functional API](#functional-opencv-style-api) below); the free functions are generated
from the methods, so the two forms stay in sync.

## Loading & saving

```{eval-rst}
.. currentmodule:: eventcv

.. autofunction:: load
.. autofunction:: from_numpy
.. autofunction:: open
.. autofunction:: save
.. autofunction:: load_frame
.. autofunction:: load_feast
.. autofunction:: export_png
.. autofunction:: collate
```

## Core types

```{eval-rst}
.. autoclass:: eventcv.EventStream
   :members:

.. autoclass:: eventcv.EventFrame
   :members:

.. autoclass:: eventcv.EventReader
   :members:

.. autoclass:: eventcv.EventPointSet
   :members:

.. autoclass:: eventcv.Camera
   :members:
```

## FEAST feature learning

{class}`~eventcv.FEAST` is an unsupervised, online event-feature extractor (Afshar et al.,
*Event-based Feature Extraction Using Adaptive Selection Thresholds*,
[Sensors 2020](https://www.mdpi.com/1424-8220/20/6/1600) /
[arXiv:1907.07853](https://arxiv.org/abs/1907.07853)). Unlike the stateless stream/frame ops,
it is a **stateful, trainable model** (scikit-learn-style `fit`/`transform`): `fit` adapts a set
of feature prototypes and their selection thresholds event-by-event, and `transform` then maps
each event to its nearest learned feature.

```python
import eventcv as ecv

stream = ecv.load("recording.npz")
feast = ecv.FEAST(n_features=100, patch=11, tau_ms=30.0, per_polarity=True, seed=0)
feast.fit(stream, epochs=2)             # online, unsupervised; returns the miss rate

ids  = feast.transform(stream)          # (N,) nearest-feature id per event (-1 at borders)
hist = feast.histogram(stream)          # pooled feature-event counts (classifier input)
imgs = feast.feature_images()           # (n_features_total, patch, patch) learned patches

ecv.save(feast, "model.npz")            # persist the trained model
feast = ecv.load_feast("model.npz")     # ...and reload it
```

```{eval-rst}
.. autoclass:: eventcv.FEAST
   :members:
```

## Live camera streaming

`eventcv.stream(...)` opens a USB event camera (Prophesee EVK3-HD/EVK4, iniVation
DVXplorer/DAVIS346, CenturyArks) as a live {class}`~eventcv.EventCamera` — the streaming twin of
{func}`~eventcv.open`. It yields the same {class}`~eventcv.EventStream` windows the file readers do,
so every representation, transform, feature detector, and viewer composes on a live feed. Windowing
mirrors `open`: `dt_ms` for fixed-duration windows or `max_events` for a fixed event count. These
functions are built into wheels that include camera support; on Linux the camera needs udev rules
for non-root USB access.

**A `while` loop that returns a representation per window.** `read()` blocks until the next window
(spanning exactly the `dt_ms` / `max_events` set at `stream(...)`) completes, then returns it —
an {class}`~eventcv.EventFrame` when opened with `repr=`, else a raw {class}`~eventcv.EventStream`:

```python
import eventcv as ecv

cam = ecv.stream(dt_ms=50, repr="mcts")     # one MCTS frame per ~50 ms of events
while running:
    frame = cam.read()                      # blocks for the next window
    infer(frame.numpy())

# Pass a wait cap so an idle scene doesn't block the loop (the cap is *not* the window length):
frame = cam.read(timeout_ms=100)            # None if no window completed within 100 ms
```

Iterating the camera (`for frame in cam:`) is the same thing in `for` form; `Ctrl+C` breaks either.

`stream` takes the same per-representation options as {meth}`~eventcv.EventReader.with_repr`
(`bins`, `window_ms`, `tau_ms`, `max_window_ms`, `window`, `normalize`), and **unset time spans
follow the capture window** so a live representation covers exactly the events it was handed:

```python
ecv.stream(dt_ms=50, repr="tencode")                  # window_ms = 50 (not the 30 ms default)
ecv.stream(dt_ms=50, repr="tencode", window_ms=20)    # explicit: keep only the newest 20 ms
ecv.stream(dt_ms=50, repr="voxel", bins=5)            # bins=5 over the 50 ms window
ecv.stream(max_events=50_000, repr="tsurf")           # no fixed duration -> tau_ms stays 30 ms
```

Without this, `dt_ms=50` with tencode's 30 ms default silently discards the oldest 20 ms of every
window — on a real 50 ms window that was 54% of the lit pixels.

**Processing and archiving in one loop.** Pass `record=` an `.h5`/`.hdf5` path and every window the
loop reads has its **raw** events appended to that file before it is rendered — so the loop works on
representations while the file keeps the full-resolution recording for later. The append happens in
Rust as each window is polled, so there is no per-window Python round trip, and the file is flushed
about once a second:

```python
with ecv.stream(dt_ms=50, repr="mcts", record="session.h5") as cam:
    while running:
        infer(cam.read().numpy())       # this window's raw events are already on disk

# `cam.n_recorded` counts events written; the `with` block (or `close()`) closes the file, and
# `ecv.open("session.h5")` reads it back like any other recording.
```

**Capture runs on its own thread.** Decoding — and any `record=` writing — happens on a background
thread that owns the camera, so the driver's ring drains continuously whatever your loop is doing;
the loop only collects windows that are already decoded. This is what keeps the per-window budget
yours: on an EVK4 at `dt_ms=50`, 40 ms of processing per frame still holds 19.9 of an ideal 20 fps
with the ring empty. `show()`, `record()`, and `close()` pause the thread to take the camera back,
and it restarts on the next read. `backlog` (ring buffers waiting) should sit near zero; `n_overflows`
counts the times the driver dropped events anyway — the loss eventcv can't prevent, because it
happens upstream.

**Keeping a slow loop on live data.** Windows are handed back in order, so a loop whose per-window
work is slower than the camera still falls behind: the thread buffers a few windows, then applies
backpressure, and under sustained overload the ring is what finally overflows. `latest=True` trades
completeness for freshness instead — each read returns the **newest** decoded window and drops what
it overtook, holding latency at about one window however slow the loop is. On a Prophesee EVK4 with
a 200 ms per-window workload, a 5 s session ends 4.3 s behind in order versus 0.2 s behind with
`latest=True`. Skipped windows are counted in `n_skipped` — and are still written by `record=` from
the capture thread, so the archive stays complete even though the loop never sees them:

```python
with ecv.stream(dt_ms=30, repr="count", record="session.h5", latest=True) as cam:
    while running:
        slow_inference(cam.read().numpy())
print(cam.n_skipped, "windows skipped,", cam.n_recorded, "events recorded")
```

**Recording from camera to file, continuously.** `record()` writes straight to disk. For HDF5
targets each window is appended as it arrives and flushed about once a second, so a long or busy
session never accumulates in memory and a crash keeps everything captured so far (npz/txt/bag can't
be appended incrementally, so they buffer and write once at the end). Only events are saved — a
DAVIS346's APS frames and IMU samples are dropped.

```python
ecv.stream().record("session.h5", seconds=10)       # 10 s, or Ctrl+C; returns the event count
ecv.stream().record("session.h5", compression=4)    # gzip the columns (HDF5 only)
```

For full control, drive the continuous writer yourself with an {class}`~eventcv.EventSink` — the
event-level twin of {class}`~eventcv.FrameSink` — which appends {class}`~eventcv.EventStream`
windows to an extendable HDF5 file (sensor size and time base taken from the first window). The
result reads straight back with {func}`~eventcv.load` / {func}`~eventcv.open`:

```python
with ecv.stream(dt_ms=50) as cam, ecv.EventSink("session.h5") as sink:
    for events in cam:
        sink.append(events)     # to disk, window-by-window
        track(events)           # ...and process the same window live
```

```{eval-rst}
.. currentmodule:: eventcv

.. autofunction:: stream
.. autofunction:: list_cameras
```

(functional-opencv-style-api)=
## Functional (OpenCV-style) API

Each function below forwards to the identically named method on the object passed as its
first argument — e.g. `eventcv.voxel(stream, bins=5)` is `stream.voxel(bins=5)`. They are
generated by introspecting the compiled types, so this list always matches the methods
above. Applying a stream op to an {class}`~eventcv.EventReader` — e.g.
`eventcv.hot_pixel_filter(reader)` — returns a new lazy reader that applies the op to every
slice it yields (identical to `reader.hot_pixel_filter()`), so filtering/geometry compose
with `slice`/`windows`/`with_repr` without loading the file.

```{eval-rst}
.. automodule:: eventcv
   :members:
   :exclude-members: EventStream, EventFrame, EventReader, EventPointSet, Camera,
      Polarity, FrameSink, EventSink, EventCamera, FEAST, load, from_numpy, open, save,
      load_frame, load_feast, export_png, collate, stream, list_cameras
```
