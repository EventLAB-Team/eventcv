# Video and analytics

## Exporting an animation

{func}`eventcv.save_video` renders a recording and writes it as one animation. The container comes
from the file extension:

```python
import eventcv as ecv

ecv.save_video(ecv.open("recording.h5"), "out.mp4", dt_ms=5, fps=30)
```

| Extension | Encoder | Needs |
| --- | --- | --- |
| `.gif` | built in | nothing |
| `.apng`, `.png` | built in | nothing |
| `.mp4`, `.m4v`, `.mov` | system `ffmpeg` | `ffmpeg` on `PATH` |

A `.png` path means an *animated* PNG — a caller asking `save_video` for a `.png` wants a moving
one. For a single still frame, use {meth}`~eventcv.EventFrame.save`.

It takes an {class}`~eventcv.EventReader` or an {class}`~eventcv.EventStream`, returns the number of
frames written, and `max_frames=` stops early — useful for checking a long recording renders
sensibly before committing to all of it. The same thing is available as a method on both
({meth}`~eventcv.EventReader.save_video`).

### The raw stream

With no `repr`, this writes the events **themselves**: each one lights its pixel in its polarity
colour and fades over `decay_ms`, which is the picture a camera's own viewer shows. It needs no
choice of representation to be meaningful, which is what makes it the right default for *looking at*
a recording rather than training on one.

```python
ecv.save_video(reader, "raw.mp4", dt_ms=5)                 # raw, the default
ecv.save_video(reader, "raw.mp4", dt_ms=5, decay_ms=20)    # longer trails
ecv.save_video(reader, "count.mp4", dt_ms=5, repr="count") # a representation instead
```

A reader opened with `repr=` uses that representation unless `repr="raw"` overrides it.

### `dt_ms` and `fps` are different things

`dt_ms` is how much of the recording each rendered frame covers — the shutter. `fps` is how fast
those frames play back. Together they set the apparent speed: `dt_ms=5, fps=30` plays six times
slower than real time, `dt_ms=33, fps=30` is roughly real time.

`dt_ms` is passed to `save_video` rather than to {func}`eventcv.open` because it is a property of
the view, not of the data — and a reader is usually opened for something else. Passing it at
`open` still works and is used when `save_video` is not given one.

### Choosing a format

**`.gif`** is the one to paste into an issue, a README or a slide. It is limited to 256 colours per
frame, which for event visualisations — mostly a colormap ramp over a dark ground — is hard to
notice.

**`.apng`** is lossless and full-colour, and costs no extra dependency. Files are considerably
larger than the equivalent GIF; use it when colour fidelity matters more than size.

**`.mp4`** is by far the smallest for a long recording, and the only sensible choice for a talk. It
is produced by piping raw frames to a system `ffmpeg` rather than by an encoder built into EventCV.
That is deliberate: linking an H.264 encoder is not free the way GIF and APNG are — x264 is GPL,
which EventCV's Apache-2.0 licence cannot absorb, and openh264 carries patent obligations that are
only cleanly discharged by shipping Cisco's prebuilt binary. Piping to a tool you already have keeps
the licence position clean and adds nothing to the wheel. If `ffmpeg` is missing, `save_video`
raises a `FileNotFoundError` naming the install command.

### Brightness: the `clim` argument

By default, rendering a *single* frame auto-contrasts it against its own data range. Applied frame
by frame across a sequence, that makes the brightness pump visibly — a quiet moment is stretched to
look as busy as a loud one.

`save_video` therefore calibrates one scale from the first few slices and holds it for the whole
video. Override it when you need to:

```python
reader.save_video("out.mp4", clim=500.0)   # 500 maps to the top of the colormap
reader.save_video("out.mp4", clim=0)       # per-frame auto-contrast (the flickering behaviour)
```

Pass an explicit `clim` whenever two videos need to be **comparable** — two conditions, two
cameras, before and after a filter. Without it each is scaled to itself and the two cannot be read
against each other.

`colormap=` takes the same names as elsewhere: `viridis` (default), `turbo`, `grayscale`,
`redblue`. Signed representations always use the diverging `redblue` map, and `tencode` and
`countmask` carry their own colour, so `colormap` does not apply to them.

### Augmentations apply

`save_video` runs the reader's deferred ops, so an export is a direct way to *see* what a training
pipeline is feeding a model:

```python
(
    ecv.open("recording.h5", dt_ms=30)
    .event_drop(0.5, seed=0)
    .with_repr("tencode")
    .save_video("what_the_model_sees.gif")
)
```

### Frame sequences

For a directory of numbered PNGs instead — to assemble externally, or to drop individual frames
into a paper — use {func}`eventcv.export_png`.

## Playing a recording

{func}`eventcv.play` opens an interactive window on a recording — the offline twin of
{meth}`~eventcv.EventCamera.show`, and the same raw polarity-and-decay view by default:

```python
ecv.play("recording.h5", dt_ms=5)                    # raw
ecv.play("recording.h5", dt_ms=20, repr="count")     # a representation
ecv.play(events, speed=0.25)                         # an in-memory stream, quarter speed
```

It blocks on the main thread until the window is closed (`Esc` or the close button). `speed`
multiplies the playback rate and `loop_=True` restarts at the end instead of closing. Frames are
decoded and rendered as they are shown, so a file larger than memory plays without being loaded —
and unlike the live camera viewer, nothing is dropped to keep up: a slow decode plays slower rather
than skipping events, because a file has no ring buffer to overflow.

A single still image of the raw view, without a window:

```python
image = events.raw_image(decay_ms=20)   # (H, W, 3) uint8
events.view("raw")                      # or show it
```

From the command line:

```console
$ eventcv play recording.h5 --dt-ms 5
$ eventcv play recording.h5 --dt-ms 5 --speed 0.25 --loop
```

## Event-rate analytics

{meth}`~eventcv.EventStream.event_rate` bins a stream in time and counts events per bin. It returns
plain NumPy arrays, so it plots with whatever you already use:

```python
import matplotlib.pyplot as plt

rate = ecv.load("recording.h5").event_rate(bin_ms=10)
plt.plot(rate["t"] / 1e6, rate["rate"])
plt.xlabel("time (s)")
plt.ylabel("events / second")
```

| Key | Contents |
| --- | --- |
| `t` | Left edge of each bin, in µs (the same time base as `stream.numpy()[:, 2]`) |
| `count` | Events of either polarity per bin |
| `positive`, `negative` | Per-polarity counts; they sum to `count` |
| `rate` | `count` divided by the bin width, in events/second |
| `bin_us` | The bin width actually used |

Bins span the recording's own extent, evenly spaced, with a final short bin if the duration does not
divide evenly — it is still counted, so `count.sum()` always equals `len(stream)`. Input does not
need to be sorted.

This is the temporal view; the spatial one is
{meth}`~eventcv.EventReader.pixel_counts`, which totals events per pixel over a whole file and shows
*where* activity concentrated rather than *when*.

Two things it is good for. **Finding the interesting part of a long recording** — the peak in the
rate curve is usually the motion you care about, and its `t` feeds straight back into
`open(offset_us=…)`. And **spotting saturation**: a rate curve that flattens into a plateau is a
sensor hitting its bandwidth limit, not a scene that happened to be uniformly busy. If you are
capturing live, that is what the event-rate cap and adaptive biasing in
[Streaming](streaming.md) are for.

The command line has a shortcut for a first look:

```console
$ eventcv info recording.h5 --rate-bin-ms 10
```
