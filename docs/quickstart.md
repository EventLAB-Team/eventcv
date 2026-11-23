# Quickstart

A five-minute tour: load a recording, transform it, turn it into a tensor, and save it.

## Load

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
stream.sensor_size  # (width, height)
len(stream)  # number of events
stream.numpy()  # (N, 4) array: x, y, t, p
```

{func}`~eventcv.load` picks the reader from the file extension and infers the sensor size and time
unit. Pass `sensor_size=` or `time_unit=` to override. See {doc}`loading`.

Events already in memory? Build a stream from an `(N, 4)` or structured array:

```python
stream = ecv.from_numpy(events, sensor_size=(640, 480), time_unit="us")
```

## Transform

Transforms return a new stream and never change the original:

```python
cropped = stream.crop(0, 0, 320, 240)
clean = cropped.hot_pixel_filter()
```

Every method is also a free function that takes the stream first:

```python
clean = ecv.hot_pixel_filter(cropped)
```

See {doc}`transforms` for every transform.

## Represent

Turn events into a dense `[C, H, W]` tensor:

```python
frame = stream.voxel(bins=5)  # an EventFrame
array = frame.numpy()  # (5, H, W) float32 ndarray
```

See {doc}`representations` for the full list: voxel grids, time surfaces, counts and more.

## Slice a large file

{func}`~eventcv.open` returns a lazy {class}`~eventcv.EventReader`. It reads only the window you ask
for.

```python
reader = ecv.open("huge.h5", dt_ms=30)  # 30 ms windows
reader.n_slices  # number of windows
window = reader[50]  # window 50, as an EventStream
```

Pass `repr=` to get tensors directly. The reader then works as a PyTorch dataset:

```python
import torch

dataset = ecv.open("huge.h5", dt_ms=30, repr="voxel")
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

See {doc}`large-files` for sliding windows, time ranges and more.

## Time units

Every time argument accepts four units. These are equivalent:

```python
ecv.open("huge.h5", dt_ms=0.5)
ecv.open("huge.h5", dt_us=500)
```

The suffixes are `_s`, `_ms`, `_us` and `_ns`. See {doc}`concepts`.

## Save

{func}`~eventcv.save` picks the format from the extension:

```python
ecv.save(clean, "clean.h5")  # streams: any supported format
ecv.save(frame, "voxel.npz")  # frames: .npz, .h5 or .png
```

See {doc}`saving` to write large files window by window.

## View

```python
stream.view()  # open the viewer on one frame
ecv.play("recording.h5")  # play a recording
ecv.save_video(stream, "clip.mp4", fps=30)  # export an animation
```

See {doc}`video` for playback controls and export options.

## Next steps

- {doc}`representations`: every tensor representation and its parameters.
- {doc}`streaming`: capture from a live camera.
- {doc}`augmentation`: seeded random transforms for training.
- {doc}`api`: the full API.
