# Command line

Installing EventCV puts an `eventcv` command on your `PATH`. It is a thin wrapper over the library
for the jobs where opening a REPL is the annoying part — checking what is in a recording,
converting a folder of them, producing a video to attach to an issue.

Everything below works equally as `python -m eventcv …`, which is useful when several environments
are in play and you want to be sure which interpreter is answering.

## `eventcv --version`

```console
$ eventcv --version
eventcv 1.0.6 (hdf5, camera, onnx)
Python 3.12.10 on macOS-14.8.3-arm64-arm-64bit
```

The first line is the two things a bug report needs: the version, and which optional features were
compiled in. If `hdf5` is absent, `.h5` files will not open; if `onnx` is absent,
{class}`~eventcv.Model` is unavailable.

## `eventcv info`

```console
$ eventcv info recording.h5
     file  recording.h5
   sensor  640 x 480
   events  106,295
 duration  0.050 s
mean rate  2,137,958 ev/s
```

Opens the index only, so it stays fast on a multi-gigabyte file. Add `--rate-bin-ms` to also report
the peak rate, which is what tells you whether a recording saturated the sensor:

```console
$ eventcv info recording.h5 --rate-bin-ms 10
...
peak rate  2,751,400 ev/s
     bins  5 x 10.0 ms
```

## `eventcv convert`

```console
$ eventcv convert recording.raw recording.h5
wrote recording.h5 (14,203,881 events, streamed)
```

The output format comes from the extension. The trailing word says how it was done: `streamed`
means the recording never had to fit in memory (HDF5 and E2VID support appending); `loaded` means
it was read whole first, which is the case for the other formats.

Converting to E2VID's input format for
[rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid) is a `.zip` away:

```console
$ eventcv convert recording.h5 events.zip
```

## `eventcv render`

```console
$ eventcv render recording.h5 preview.gif --dt-ms 20 --fps 25
wrote preview.gif (48 frames at 25 fps)
```

Writes `.gif`, `.apng` or `.mp4` — see [Video and analytics](video.md) for the trade-offs, and note
that `.mp4` needs `ffmpeg` on your `PATH`.

Useful flags:

- `--dt-ms` — time per frame; controls how much motion each frame accumulates.
- `--fps` — playback rate. Independent of `--dt-ms`: setting `--dt-ms 10 --fps 100` plays back in
  real time, while `--fps 10` makes the same recording a slow-motion clip.
- `--repr` — what to draw. The default `raw` is the event stream itself (polarity dots fading by
  age), which needs no choice of representation to be meaningful; pass a representation name
  (`tencode`, `count`, `tsurf`, …) to render that instead.
- `--decay-ms` — raw view only: how long an event's trail takes to fade.
- `--clim` — fix the brightness scale, so two renders are comparable.
- `--max-frames` — stop early, to check a long recording before committing to all of it.

## `eventcv play`

```console
$ eventcv play recording.h5 --dt-ms 5
```

Opens an interactive window and plays the recording — the offline twin of a camera's live view, and
raw by default for the same reason `render` is. It blocks until the window is closed (`Esc` or the
close button).

Takes `--dt-ms`, `--fps`, `--repr`, `--decay-ms` and `--colormap` exactly as `render` does, plus:

- `--speed` — playback rate multiplier; `--speed 0.25` for a slow look at a fast scene.
- `--loop` — restart at the end instead of closing.

## `eventcv simulate`

```console
$ eventcv simulate clip.mp4 events.h5 --scale 960 540
simulating: 39/39 frames, 18.7M events
wrote events.h5 (18,742,110 events from 39 frames)
```

Turns a video into synthetic events (needs `ffmpeg` on `PATH`). The events are written as they are
produced, so memory stays flat however long the clip is, and progress is shown when stderr is a
terminal. See [Simulation](simulation.md) for the pixel model.

Beyond the model parameters (`--threshold`, `--sigma-thres`, `--cutoff-hz`, `--leak-rate-hz`,
`--shot-noise-rate-hz`, `--refractory-us`, `--seed`):

- `--scale W H` — decode at a smaller size. The most effective lever on both time and file size:
  halving each side quarters the pixels and roughly quarters the output.
- `--upsample`, `--max-upsample` — how finely each frame interval is subdivided, and a ceiling on
  it. Each sub-step is a full pass over every pixel.
- `--compression` — HDF5 gzip level `0..9`, `0` for uncompressed (default `1`).

## Reading text and rosbag sources

Text formats carry no timestamp unit or column order, and a rosbag has no single obvious
connection, so `info`, `convert`, `render` and `play` all accept:

- `--time-unit {s,ms,us,ns}` — without this, a `.txt` recording can report a duration off by a
  factor of a thousand.
- `--order` — column order of a headerless text file, e.g. `txyp` or `xytp`.
- `--topic` — rosbag connection to read.

```console
$ eventcv info events.txt --time-unit us --order txyp
```

## Exit codes and errors

`0` on success, `1` on a reported error, `130` on interrupt. Failures print a single actionable
line to stderr rather than a traceback:

```console
$ eventcv render recording.h5 out.mp4
error: writing .mp4 needs ffmpeg on PATH (macOS: `brew install ffmpeg`, ...)
```

Long conversions and renders are interruptible with Ctrl-C.
