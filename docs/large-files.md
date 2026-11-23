# Large files

{func}`~eventcv.open` returns an {class}`~eventcv.EventReader`: a lazy handle that reads only the
window you ask for.

```python
import eventcv as ecv

reader = ecv.open("huge.h5", dt_ms=30)  # 30 ms windows
reader.n_slices  # number of windows
window = reader[50]  # window 50, as an EventStream
```

`open` takes the same options as {func}`~eventcv.load` (`sensor_size`, `time_unit`, `keys`, …).

## Which formats read in place

| Format | How `open` reads it |
|---|---|
| HDF5, ROS bag, text, AEDAT 2.0, AEDAT 4, Prophesee RAW | In place. Seeks to each window without loading the file. |
| NumPy `.npz`, Prophesee `.dat` | Loaded once, then sliced in memory. |

`open` also accepts an {class}`~eventcv.EventStream` or a NumPy array in place of a path.

## Fixed-duration windows

```python
reader = ecv.open("huge.h5", dt_ms=30)
reader[0]  # first 30 ms
reader[-1]  # last window
```

Window `n` covers `[start + n·dt, start + (n+1)·dt)`, measured from the first event. You never need
the file's absolute timestamps.

## Fixed-count windows

```python
reader = ecv.open("huge.h5", max_events=10_000)
reader[0]  # first 10,000 events
```

Each window holds exactly `max_events` events; the last may hold fewer. `max_events` and `dt_ms`
can't be combined.

## Other ways to slice

| Call | Returns |
|---|---|
| `reader.slice(n)` | Window `n`. Same as `reader[n]`. |
| `reader.slice(t0_ms=5000, t1_ms=5100)` | Events in `[t0, t1)`, in the file's own time base. |
| `reader.slice_count(i0, i1)` | Events with index in `[i0, i1)`. |
| `reader.windows(step_ms=10, span_ms=30)` | A lazy iterator of windows, `span` long, every `step`. |

`windows()` with no arguments walks every window in turn:

```python
for window in reader.windows():
    frame = window.voxel()
```

## Start later

```python
reader = ecv.open("huge.h5", dt_ms=30, offset=1_000)  # window 0 starts at t = 1 s
```

`offset` is an absolute timestamp in the file's time base. Events before it are outside every
window.

## Transform every window

Transforms on a reader return a new reader that applies them to each window as it is read:

```python
clean = reader.crop(0, 0, 320, 240).hot_pixel_filter()
clean[50]  # window 50, cropped and filtered
```

To remove stuck pixels with one mask for the whole recording, filter at `open`:

```python
reader = ecv.open("huge.h5", dt_ms=30, hot_pixel_filter=True)
```

This scans the file once. It is more reliable than `hot_pixel_filter()` per window, whose threshold
changes from window to window. `hot_pixel_std` (default `3.0`) sets how aggressive it is.

## Representations and PyTorch

Pass `repr=` and the reader returns frames instead of streams:

```python
reader = ecv.open("huge.h5", dt_ms=30, repr="voxel")
reader.slice(50)  # EventFrame
reader[50]  # (C, H, W) ndarray
reader.batch([0, 1, 2])  # (3, C, H, W) ndarray
```

Use {meth}`~eventcv.EventReader.with_repr` to set the representation's parameters:

```python
reader = ecv.open("huge.h5", dt_ms=30).with_repr("voxel", bins=5)
```

A reader with `repr=` is a PyTorch map-style dataset:

```python
import torch

loader = torch.utils.data.DataLoader(reader, batch_size=32, shuffle=True)
```

Without `repr=`, items are streams of different lengths. Pass {func}`~eventcv.collate` to batch them
as lists:

```python
reader = ecv.open("huge.h5", dt_ms=30)
loader = torch.utils.data.DataLoader(reader, batch_size=32, collate_fn=ecv.collate)
```

## Recording information

| Attribute | Meaning |
|---|---|
| `reader.n_slices` | Number of windows. |
| `reader.n_events` | Number of events in the file. |
| `reader.sensor_size` | `(width, height)`. |
| `reader.time_span("ms")` | First and last timestamp. |
| `reader.duration("s")` | Length of the recording. |
| `reader.dt("us")` | Window length set at `open`. |
