# Video and playback

Export recordings as animations, play them interactively, and plot the event rate.

## Export an animation

```python
import eventcv as ecv

ecv.save_video(ecv.open("recording.h5"), "out.mp4", dt_ms=5, fps=30)
```

{func}`~eventcv.save_video` takes a reader or a stream and returns the number of frames written.

| Option | Default | Meaning |
|---|---|---|
| `dt_ms` | the reader's `dt_ms`, else `30` | Recording time per frame. |
| `fps` | `30` | Playback rate. With `dt_ms=5, fps=30`, playback is 6× slower than real time. |
| `repr` | raw events | A {doc}`representation <representations>` to render instead. |
| `decay_ms` | one frame | Raw view only: how long each event takes to fade. |
| `colormap` | `"viridis"` | `"viridis"`, `"turbo"`, `"grayscale"` or `"redblue"`. |
| `clim` | calibrated | Value mapped to full brightness. See [Brightness](#brightness). |
| `max_frames` | all | Stop early, e.g. to preview a long recording. |

The format comes from the extension:

| Extension | Best for | Needs |
|---|---|---|
| `.gif` | Issues, READMEs, slides | Nothing |
| `.apng`, `.png` | Lossless, full colour, larger files | Nothing |
| `.mp4`, `.m4v`, `.mov` | Long recordings, talks | `ffmpeg` on your `PATH` |

A `.png` path writes an animated PNG. For a single image, use {meth}`~eventcv.EventFrame.save`; for
one PNG per frame, use {func}`~eventcv.export_png`.

By default the video shows the raw events: each one lights its pixel in its polarity colour and fades.

```python
ecv.save_video(reader, "raw.mp4", dt_ms=5, decay_ms=20)  # longer trails
ecv.save_video(reader, "count.mp4", dt_ms=5, repr="count")  # a representation
```

Transforms and augmentations on the reader are applied, so an export shows exactly what a training
pipeline produces.

### Brightness

`save_video` sets one brightness scale from the first few frames and keeps it for the whole video,
so brightness doesn't flicker.

```python
ecv.save_video(reader, "out.mp4", clim=500)  # fixed: 500 events = full brightness
ecv.save_video(reader, "out.mp4", clim=0)  # rescale every frame
```

Pass the same `clim` to videos you want to compare side by side.

## Play a recording

```python
ecv.play("recording.h5", dt_ms=5)
ecv.play("recording.h5", repr="count")
ecv.play(stream, speed=0.25)  # an in-memory stream, quarter speed
ecv.play()  # open empty, then choose or drop a file
```

{func}`~eventcv.play` opens a window with play/pause, stepping, seeking, speed and loop controls,
live statistics, and a panel of denoising filters. Large files are read as they play, never loaded
whole.

| Option | Default | Meaning |
|---|---|---|
| `dt_ms` | `30` | Time shown in each frame. |
| `speed` | `1.0` | Playback rate. `1.0` is real time. |
| `refresh_hz` | `60` | Display refresh rate, independent of `dt_ms`. |
| `repr`, `decay_ms`, `colormap`, `clim` | | As for `save_video`. |

To get the raw view as an image without opening a window:

```python
image = stream.raw_image(decay_ms=20)  # (H, W, 3) uint8
```

## Event rate

{meth}`~eventcv.EventStream.event_rate` counts events in fixed time bins:

```python
import matplotlib.pyplot as plt

rate = stream.event_rate(bin_ms=10)
plt.plot(rate["t"] / 1e6, rate["rate"])
plt.xlabel("time (s)")
plt.ylabel("events / s")
```

| Key | Contents |
|---|---|
| `t` | Start of each bin, in µs. |
| `count` | Events per bin. Sums to `len(stream)`. |
| `positive`, `negative` | Events per bin by polarity. |
| `rate` | Events per second. |
| `bin_us` | Bin width used. |

A peak usually marks the motion you're after; pass its time to `open(offset_us=…)`. A flat plateau
at the top suggests the sensor saturated.
