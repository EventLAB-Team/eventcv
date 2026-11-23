# Representations

A representation turns events into a dense image or tensor that NumPy, PyTorch and the viewer can
use. Each call returns an {class}`~eventcv.EventFrame`; `.numpy()` gives the `(C, H, W)` array.

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
frame = stream.voxel(bins=5)
array = frame.numpy()  # (5, H, W) float32
```

## Choose a representation

Click a representation to jump to its parameters. All images are from the same 50 ms recording.

::::{grid} 2 2 3 3
:gutter: 2

:::{grid-item-card} polarity
:img-top: images/representations/polarity.png
:link: repr-polarity
:link-type: ref
Counts per pixel, per polarity.
:::

:::{grid-item-card} count
:img-top: images/representations/count.png
:link: repr-count
:link-type: ref
Counts per pixel, polarities summed.
:::

:::{grid-item-card} binary
:img-top: images/representations/binary.png
:link: repr-binary
:link-type: ref
Whether a pixel fired.
:::

:::{grid-item-card} voxel
:img-top: images/representations/voxel.png
:link: repr-voxel
:link-type: ref
Polarity spread over time bins.
:::

:::{grid-item-card} tsurf
:img-top: images/representations/tsurf.png
:link: repr-tsurf
:link-type: ref
How recently each pixel fired.
:::

:::{grid-item-card} atsurf
:img-top: images/representations/atsurf.png
:link: repr-atsurf
:link-type: ref
Time surface averaged over all events.
:::

:::{grid-item-card} mcts
:img-top: images/representations/mcts.png
:link: repr-mcts
:link-type: ref
Time surfaces over several windows.
:::

:::{grid-item-card} tencode
:img-top: images/representations/tencode.png
:link: repr-tencode
:link-type: ref
Latest polarity and event age as RGB.
:::

:::{grid-item-card} countmask
:img-top: images/representations/countmask.png
:link: repr-countmask
:link-type: ref
Counts in red/blue, activity in green.
:::

:::{grid-item-card} redblue
:img-top: images/representations/redblue.png
:link: repr-redblue
:link-type: ref
Red/blue counts on white.
:::

:::{grid-item-card} flow
:img-top: images/representations/flow.png
:link: repr-flow
:link-type: ref
Dense optical flow.
:::
::::

## Where to use one

| Where | Example | Parameters |
|---|---|---|
| A stream | `stream.voxel(bins=5)` or `ecv.voxel(stream, bins=5)` | All |
| Every slice of a file | `ecv.open(path, dt_ms=30, repr="voxel")` | Defaults only |
| Every slice, with parameters | `ecv.open(path, dt_ms=30).with_repr("voxel", bins=5)` | All |
| A live camera | `ecv.stream(dt_ms=50, repr="voxel")` | All, as keywords |
| The viewer | `stream.view("voxel")` | Defaults only |
| By name on a stream | `stream.flatten("voxel")` | Defaults only |

With `repr=` set, a reader returns frames from `slice()` and arrays from `reader[i]`, so it works
directly as a PyTorch dataset. See {doc}`large-files`.

## Common parameters

| Parameter | Applies to | Default | Meaning |
|---|---|---|---|
| `window_ms`, `tau_ms`, `max_window_ms` | voxel, tencode, tsurf, atsurf, mcts | The slice length, else 30 ms | Time span the representation covers. |
| `normalize` | polarity, count | `True` for polarity, `False` for count | Scale to `uint8` with the busiest pixel at 255, or keep raw `uint64` counts. |
| `device` | count, voxel, tsurf, atsurf, countmask, redblue | `"cpu"` | Run on `"gpu"`. The CPU is usually faster; see {doc}`gpu`. |

Time spans also accept `_s`, `_us` and `_ns` forms, e.g. `window_us=500`. An explicit span always
overrides the slice length.

## Counts

(repr-polarity)=
### polarity

```python
frame = stream.flatten()  # or stream.flatten("polarity")
```

| Parameter | Default | Meaning |
|---|---|---|
| `normalize` | `True` | See [common parameters](#common-parameters). |
| `downsample` | `None` | Bin `k`×`k` pixels into one cell while counting. |

**Output:** 2 channels `[positive, negative]` · `uint8`, or `uint64` with `normalize=False`.

The default frame when no representation is named. There is no `.polarity()` method.

(repr-count)=
### count

```python
frame = stream.count()
```

| Parameter | Default | Meaning |
|---|---|---|
| `normalize` | `False` | See [common parameters](#common-parameters). |

**Output:** 1 channel · `uint64`, or `uint8` with `normalize=True`.

The raw default keeps exact totals: `stream.count().numpy().sum() == len(stream)`.

(repr-binary)=
### binary

```python
frame = stream.flatten(binary=True)  # or stream.flatten("binary")
```

**Output:** 1 channel · `uint8`. `1` where a pixel fired, `0` elsewhere. No parameters.

## Time-based

(repr-voxel)=
### voxel

```python
frame = stream.voxel(bins=9, window_ms=30)
```

| Parameter | Default | Meaning |
|---|---|---|
| `bins` | `9` | Number of time bins. |
| `window_ms` | slice length | Span ending at the last event. Earlier events are dropped. |

**Output:** `bins` channels · `float32`.

Each event adds `+1` or `-1` by polarity, split between its two nearest bins.

(repr-tsurf)=
### tsurf

```python
frame = stream.tsurf(tau_ms=30)
```

| Parameter | Default | Meaning |
|---|---|---|
| `tau_ms` | slice length | Decay constant. |

**Output:** 2 channels `[positive, negative]` · `float32`.

Each pixel holds `exp(-age / tau)` of its most recent event, or `0` if it never fired.

(repr-atsurf)=
### atsurf

```python
frame = stream.atsurf(tau_ms=30)
```

| Parameter | Default | Meaning |
|---|---|---|
| `tau_ms` | slice length | Decay constant. |

**Output:** 2 channels `[positive, negative]` · `float32`.

Like `tsurf`, but averaged over every event at the pixel, so repeated activity reads brighter.

(repr-mcts)=
### mcts

```python
frame = stream.mcts(max_window_ms=30)  # 5 log-spaced windows, 10 channels
frame = stream.mcts(windows_ms=[1, 5, 20])  # your own windows, 6 channels
```

| Parameter | Default | Meaning |
|---|---|---|
| `max_window_ms` | slice length | Largest of 5 log-spaced windows starting at 1 ms. |
| `windows_ms` | `None` | Explicit windows instead. Cannot be combined with `max_window_ms`. |

**Output:** 2 channels per window, negative then positive (e.g. `negative_1.000ms`) · `float32`.

Each channel holds `max(1 - age / window)` over the events in that window.

(repr-tencode)=
### tencode

```python
frame = stream.tencode(window_ms=30)
```

| Parameter | Default | Meaning |
|---|---|---|
| `window_ms` | slice length | Span used to scale event age. |

**Output:** 3 channels `[positive, age, negative]` · `uint8`.

The polarity that fired last is set to 255. The age channel scales that event's age to 0–255.

## Colour images

(repr-countmask)=
### countmask

```python
frame = stream.countmask(pct=99)
```

| Parameter | Default | Meaning |
|---|---|---|
| `pct` | `99.0` | Percentile of non-zero counts that maps to full brightness. |
| `white_frame` | `False` | Invert to a white background. |

**Output:** 3 channels `[positive, activity, negative]` · `uint8`.

Red and blue hold polarity counts on one shared scale. Green is 255 wherever an event landed.
Timestamps are ignored. This is the event image from the GEPT tracker.

(repr-redblue)=
### redblue

```python
frame = stream.redblue(pct=99)
```

| Parameter | Default | Meaning |
|---|---|---|
| `pct` | `99.0` | Percentile of non-zero counts that maps to full colour. |

**Output:** 3 channels RGB · `uint8`.

Positive events shade the white background toward red, negative toward blue. Each polarity is
scaled separately, and the stronger one wins at each pixel. Timestamps are ignored.

## Other outputs

(repr-flow)=
### flow

```python
frame = stream.optical_flow(window=3)  # or stream.view("flow")
```

| Parameter | Default | Meaning |
|---|---|---|
| `window` | `3` | Half-width of the fitting patch. |

**Output:** 2 channels `[flow_x, flow_y]` in pixels/ms · `float32`.

Lucas–Kanade on the time surface, set only at pixels that saw an event. The stream must be sorted by
time.

### pset

```python
points = stream.pset().numpy()  # (N, 4) float32: x, y, t, p
```

A sparse point set rather than an image, returned as an {class}`~eventcv.EventPointSet`. `x`, `y` and
`t` are scaled to `[0, 1]`; `p` is `+1` or `-1`. It can't be used with `repr=`.

### Connected components

```python
labels = stream.flatten(binary=True).connected_components(connectivity=8)
```

Labels each connected blob of non-zero pixels in a frame `1, 2, …`, with background `0`.
`connectivity` is `4` or `8`. Output is 1 channel, `uint64`.
