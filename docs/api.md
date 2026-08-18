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

## Devices

Where representations and the simulator run. The CPU is the default and the reference; see
{doc}`representations` for what the GPU changes numerically and when it is worth using.

```{eval-rst}
.. autofunction:: set_device
.. autofunction:: get_device
.. autofunction:: gpu_available
```

## Viewing

```{eval-rst}
.. autofunction:: play
.. autofunction:: save_video
```

See :doc:`video` for the raw view, and how ``dt_ms`` and ``fps`` combine into playback speed.

## Simulation and reconstruction

```{eval-rst}
.. autofunction:: simulate
.. autofunction:: reconstruct
.. autoclass:: eventcv.StatefulModel
   :members:
```

See :doc:`simulation` for the pixel model and :doc:`reconstruction` for the ONNX export
requirements.

## Models

```{eval-rst}
.. autoclass:: eventcv.Model
   :members:
```

Requires a build with the ``onnx`` feature (the published wheels have it). See
:doc:`models` for scope and usage.

### E2VID export

A `.zip` target — or `format="e2vid"` on a `.txt` — writes the layout
[E2VID](https://github.com/uzh-rpg/rpg_e2vid) reads: a `width height` header line, then one
whitespace-separated `t x y p` row per event with `t` in float seconds and `p` as `0`/`1`,
ascending in time. That removes the conversion script between an EventCV recording and
event-to-video reconstruction.

```python
ecv.save(stream, "events.zip")                       # from a stream
ecv.save(ecv.open("rec.h5"), "events.zip")           # or straight from a reader
# python run_reconstruction.py --input_file events.zip --fixed_duration -T 33
```

Passing an {class}`~eventcv.EventReader` converts it **window by window**, so a recording far
larger than memory re-exports without being loaded, and any deferred ops on the reader (`crop`,
`mask`, `hot_pixel_filter`, …) apply to what gets written. The export is one-way — EventCV does
not read the E2VID layout back, so keep an `.npz`/`.h5` if you need the recording itself.

(roi-masking)=
## Region-of-interest masking

A mask is a plain `(H, W)` boolean NumPy array — `True` where events are **kept** — so it composes
with NumPy's own operators (`|`, `&`, `~`) and can be built, drawn, or loaded. Pass one to
{meth}`~eventcv.EventStream.mask`, to {meth}`~eventcv.EventReader.mask` (deferred onto every
slice), or to `ecv.stream(mask=…)`, where events outside it are dropped as they are decoded — so
they never reach a `record=` file or the windows a loop reads.

```python
import eventcv as ecv

aperture = ecv.circle_mask((640, 480), cx=320, cy=240, r=230)   # a circular sensor aperture
aperture &= ~ecv.rect_mask((640, 480), 0, 0, 64, 64)            # minus a hot corner
ecv.save_mask(aperture, "aperture.png")                         # reuse it next session

ecv.open("rec.h5", dt_ms=30).mask(aperture)                     # every slice, lazily
ecv.stream(dt_ms=50, mask=aperture, record="session.h5")        # live, before the recorder
```

`sensor_size` is `(width, height)` — the same order as everywhere else in eventcv — while the array
you get back is `(H, W)`, like any other NumPy image. Coordinates are continuous rather than pixel
indices: a pixel is kept when its centre falls inside the shape, and geometry off the sensor is
clamped. An 8-bit map works anywhere a boolean one does (any non-zero value keeps the pixel), so a
mask binarised elsewhere is passed straight in. A mask that isn't the size of the sensor raises,
rather than silently dropping every event.

To draw one instead, {meth}`~eventcv.EventStream.draw_mask` (or
{meth}`~eventcv.EventFrame.draw_mask`) opens the viewer on a still frame, and
{meth}`~eventcv.EventCamera.draw_mask` draws over the live camera and applies the result. Drag to
keep an area, shift+drag to drop one, `e`/`r`/`f` to switch between ellipse, rectangle, and
freehand, `a`/`c` to select all or clear, `z` to undo; whatever stays bright is what the mask
keeps. `Enter` accepts, `Esc` returns `None`.

```{eval-rst}
.. autofunction:: circle_mask
.. autofunction:: ellipse_mask
.. autofunction:: rect_mask
.. autofunction:: polygon_mask
.. autofunction:: save_mask
.. autofunction:: load_mask
```

## Motion estimation

```{eval-rst}
.. autoclass:: eventcv.Tracker
   :members:
```

See :doc:`motion` for contrast maximisation and the tracker's association rule.

## Networking

```{eval-rst}
.. autoclass:: eventcv.UdpSender
   :members:
.. autoclass:: eventcv.UdpReceiver
   :members:
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

.. autoclass:: eventcv.EventCamera
   :members:

.. autoclass:: eventcv.EventSink
   :members:

.. autoclass:: eventcv.FrameSink
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

(live-camera-streaming)=
## Live camera streaming

`eventcv.stream(...)` opens a USB event camera (Prophesee EVK4/EVK3-HD, iniVation
DVXplorer/DAVIS346, CenturyArks) as a live {class}`~eventcv.EventCamera` — the streaming twin of
{func}`~eventcv.open`. It yields the same {class}`~eventcv.EventStream` windows the file readers do,
so every representation, transform, feature detector, and viewer composes on a live feed:

```python
import eventcv as ecv

with ecv.stream(dt_ms=50, repr="mcts", record="session.h5") as cam:
    while running:
        infer(cam.read().numpy())   # a representation per window; raw events archived as you go
```

{func}`~eventcv.record` is the one-shot form for scripts that only want a file — it opens the
camera, captures for `seconds`, and closes it before returning, so the recording is complete and the
device free by the next line:

```python
ecv.record("session.h5", seconds=10)
reader = ecv.open("session.h5", dt_ms=50)
```

Windowing mirrors `open` (`dt_ms` or `max_events`), `repr=` and its options render each window,
`record=` archives the raw events, `latest=True` keeps a slow loop on live data,
`max_event_rate` / `roi` cap what the sensor emits in hardware, and `mask=` restricts it to an
arbitrarily shaped {ref}`region of interest <roi-masking>` on the host. See the
[streaming guide](streaming.md) for the full picture — recording, staying live under load, source
caps, and troubleshooting. These functions are built into wheels that include camera support; on
Linux the camera needs udev rules for non-root USB access.

### Adaptive biasing

Fixed biases make the same scene produce wildly different event rates as the light changes.
`adaptive_bias=True` measures the rate and retunes the sensor's bias currents as it runs, after
Nair et al., *Enhancing Visual Place Recognition via Fast and Slow Adaptive Biasing in Event
Cameras* (IROS 2024) — a fast loop mapping the rate onto the refractory period several times a
second, and a slow loop shifting the photoreceptor and threshold biases when the fast one runs out
of travel. It starts from the camera's stock biases, so enabling it never jumps the picture, and
{attr}`~eventcv.EventCamera.bias_state` reports what it is doing.

```python
with ecv.stream(dt_ms=50, adaptive_bias={"target_rate": (3e4, 1.2e5)}) as cam:
    while running:
        infer(cam.read().numpy())
        print(cam.bias_state)   # event_rate, the five bias values, authority, n_slow_steps
```

**It measures your scene first.** For about the first second the controller changes nothing and
just watches, then centres its band on the rate the camera was actually producing — so
`adaptive_bias=True` means "hold the event rate wherever this scene started", with no numbers to
pick. Asking for a rate a scene cannot supply at any bias setting is the one way to get bad results
(it drives the sensor into amplifying its own noise), and measuring first avoids it. Pass
`target_rate` only for a specific absolute rate; that skips the measurement. `bias_state` reports
the band chosen, whether it is still `calibrating`, and an `authority` of `"hunting"` if the band
turns out to be unreachable anyway.

Supported on the iniVation DAVIS346 and the Prophesee EVK4, each with its own defaults. Measured on
a DAVIS346 over a static indoor scene, the median rate held within 1.2x across runs where the
unbiased camera wandered 5.7x; on an EVK4, 1.8x against 4.2x. Other cameras raise rather than
silently doing nothing.

```{eval-rst}
.. currentmodule:: eventcv

.. autofunction:: stream
.. autofunction:: record
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
      load_frame, load_feast, export_png, collate, stream, list_cameras, Model,
      simulate, reconstruct, StatefulModel, Tracker, UdpSender, UdpReceiver,
      play, save_video, set_device, get_device, gpu_available
```
