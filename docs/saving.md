# Saving

{func}`~eventcv.save` writes a stream or frame. It picks the format from the extension.

```python
import eventcv as ecv

ecv.save(stream, "recording.h5")
ecv.save(stream.voxel(bins=5), "voxel.npz")
```

## Streams

Every format EventCV reads can be written. See the [format table](index.md#supported-formats).

| Option | Applies to | Meaning |
|---|---|---|
| `compression` | `.h5` | `1`–`9` for the gzip level, or `False` for the fastest write. Default `1`. |
| `topic` | `.bag` | Topic to write. |
| `format` | any | Override the extension, e.g. `format="e2vid"` on a `.txt`. |

NumPy, HDF5 and ROS bag files round-trip exactly.

## Frames

```python
frame = stream.voxel(bins=5)
ecv.save(frame, "voxel.h5")  # raw array, with kind and channel names
ecv.save(frame, "voxel.png")  # a colormapped image
frame = ecv.load_frame("voxel.h5")
```

`.npz` and `.h5` keep the array exactly. `.png` saves a picture, not the data. To write one image
per window, use {func}`~eventcv.export_png`.

## Write window by window

{class}`~eventcv.EventSink` appends streams to a file as you go, so a recording never has to fit in
memory:

```python
reader = ecv.open("huge.bag", dt_ms=100)

with ecv.EventSink("huge.h5") as sink:
    for window in reader.windows():
        sink.append(window.hot_pixel_filter())
```

The sink takes the sensor size and time base from the first window, and writes any format `save`
does. HDF5, text, AEDAT and Prophesee files are readable as they grow; `.npz` and `.bag` are complete
only once the sink closes. Call `sink.flush()` to force data to disk.

{class}`~eventcv.FrameSink` does the same for frames, into one `(N, C, H, W)` HDF5 dataset.

## E2VID export

A `.zip` writes the text layout that [E2VID](https://github.com/uzh-rpg/rpg_e2vid) reads, ready for
event-to-video reconstruction:

```python
ecv.save(stream, "events.zip")
ecv.save(ecv.open("huge.h5"), "events.zip")  # a whole file, window by window
```

The export is one-way: EventCV can't load it back. EventCV can also reconstruct video itself; see
{doc}`reconstruction`.
