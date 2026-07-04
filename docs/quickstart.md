# Quickstart

## Load a recording

{func}`eventcv.load` reads an entire stream, detecting the format from the file
extension. Sensor size and time unit are auto-detected; pass them only to override.

```python
import eventcv as ecv

stream = ecv.load("recording.hdf5")
print(stream.sensor_size)      # (width, height)
print(len(stream))             # number of events
events = stream.numpy()        # N×4 array of [t, x, y, p]
```

## Transform (chainable, functional)

Transforms return a **new** stream, so they chain — and every one is also a free
function taking the stream as its first argument.

```python
# method form
clipped = stream.crop(0, 0, 320, 240).flip_x().hot_pixel_filter()

# equivalent free-function (OpenCV-style) form
clipped = ecv.hot_pixel_filter(ecv.flip_x(ecv.crop(stream, 0, 0, 320, 240)))
```

## Represent as a dense tensor

```python
voxel = stream.voxel(bins=5)   # or ecv.voxel(stream, bins=5)
array = voxel.numpy()          # [C, H, W] ndarray, ready for NumPy / PyTorch
```

## Stream a large file

{func}`eventcv.open` returns a lazy {class}`~eventcv.EventReader` that slices the file
on demand instead of materialising it.

```python
reader = ecv.open("huge.hdf5", dt_ms=30)   # treat as 30 ms frames
print(reader.n_slices)
frame = reader.slice(50).voxel()           # the 50th 30 ms frame → voxel grid
```

Pass `repr=` to give the reader a **default representation**. It is remembered by every
slice, so `slice(n).view()` renders it (no need to name it again), and it makes the reader
a PyTorch map-style dataset where `reader[i]` is the dense `[C, H, W]` array:

```python
import torch

reader = ecv.open("huge.hdf5", dt_ms=30, repr="mcts")
reader.slice(500).view()               # shows MCTS — the stored repr (not raw polarity)
reader.slice(500).view("count")        # an explicit name still overrides it

# as a PyTorch map-style dataset (dense arrays collate straight into a tensor)
ds = ecv.open("huge.hdf5", dt_ms=30, repr="count")
# len(ds) == n_slices; ds[i] -> [C, H, W]; ds.batch(idxs) -> [B, C, H, W]
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
```

Without `repr`, `reader[i]` stays a raw {class}`~eventcv.EventStream`. Those are sparse and
variable-length, so a `DataLoader` can't stack them — pass {func}`eventcv.collate` to batch
them as a `list` instead:

```python
import torch

reader = ecv.open("huge.hdf5", dt_ms=30)
loader = torch.utils.data.DataLoader(reader, batch_size=32, collate_fn=ecv.collate)
for batch in loader:                   # batch is a list[EventStream]
    batch[0].view()
```

## Save

{func}`eventcv.save` mirrors {func}`~eventcv.load`, choosing the format by extension.

```python
ecv.save(clipped, "out.npz")        # streams: .npz / .txt / .h5 / .bag
ecv.save(voxel, "voxel.h5")         # frames: .npz / .h5
```

See the [API reference](api.md) for the full list of readers, transforms, and
representations.
