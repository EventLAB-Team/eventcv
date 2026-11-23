# Fundamentals

The types and conventions used throughout EventCV.

## Core types

| Type | What it holds | Get one from |
|---|---|---|
| {class}`~eventcv.EventStream` | Events in memory. | {func}`~eventcv.load`, {func}`~eventcv.from_numpy`, a reader slice |
| {class}`~eventcv.EventReader` | A lazy handle on a file. Reads only the window you ask for. | {func}`~eventcv.open` |
| {class}`~eventcv.EventFrame` | A dense `(C, H, W)` image, such as a voxel grid. | A {doc}`representation <representations>` |
| {class}`~eventcv.EventPointSet` | A sparse `(N, 4)` point array. | {meth}`~eventcv.EventStream.pset` |
| {class}`~eventcv.EventCamera` | A live USB camera. | {func}`~eventcv.stream` |

## Events

Each event has four fields:

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | `uint16` | Pixel coordinates. |
| `t` | `int64` | Timestamp in microseconds. |
| `p` | `bool` | Polarity: `True` for ON (brighter), `False` for OFF. |

```python
stream = ecv.load("recording.h5")
len(stream)  # number of events
stream.numpy()  # (N, 4) uint64 array, columns x, y, t, p
```

`numpy()` returns a copy, so changing the array doesn't change the stream.

## Sensor size and array shape

`sensor_size` is always `(width, height)`. Arrays follow NumPy and are `(H, W)` or `(C, H, W)`.

```python
stream.sensor_size  # (640, 480)
stream.count().numpy().shape  # (1, 480, 640)
```

## Operations return new values

Transforms and representations never change their input. Each returns a new stream or frame, so
calls chain:

```python
clean = stream.crop(0, 0, 320, 240).hot_pixel_filter()  # stream is unchanged
```

## Methods and functions

Every method is also a free function that takes the object as its first argument:

```python
stream.voxel(bins=5)
ecv.voxel(stream, bins=5)  # identical
```

Applied to a reader, a transform returns a new lazy reader that applies it to every slice:

```python
reader = ecv.open("huge.h5", dt_ms=30).hot_pixel_filter()  # nothing read yet
reader[10]  # window 10, hot pixels removed
```

## Time units

Every time argument comes in four forms. Use whichever unit suits you:

| Suffix | Unit | Example |
|---|---|---|
| `_s` | seconds | `dt_s=0.03` |
| `_ms` | milliseconds | `dt_ms=30` |
| `_us` | microseconds | `dt_us=30_000` |
| `_ns` | nanoseconds | `dt_ns=30_000_000` |

This covers `dt`, `offset`, `t0`/`t1`, `step`/`span`, `window`, `tau`, `max_window` and `decay`.
Passing two units for one argument raises an error.

:::{note}
Timestamps are stored in microseconds. A `_ns` value is rounded to the nearest microsecond, and a
duration below half a microsecond raises an error.
:::

Wall-clock arguments such as `timeout_ms` are plain milliseconds and have no other forms.
