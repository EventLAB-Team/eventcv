# Command line

Installing EventCV puts an `eventcv` command on your `PATH`. It is a thin wrapper over the library
for the jobs where opening a REPL is the annoying part — checking what is in a recording,
converting a folder of them, producing a video to attach to an issue.

The command comes two ways:

- **With the Python package** (`pip install eventcv`), it has every command below except `cameras`
  and `record`, whose library forms are {func}`eventcv.list_cameras` and {func}`eventcv.record`.
  It also works as `python -m eventcv …`, which is useful when several environments are in play
  and you want to be sure which interpreter is answering.
- **As a standalone binary** that needs no Python ([installing it](index.md)), it has `info`,
  `convert`, `render`, `simulate`, `tui`, `cameras` and `record`. `play`, `reconstruct` and the
  `ros2-*` commands need the GPU viewer, an ONNX Runtime or ROS, so they stay with the package.

The commands both have take the same flags and print the same output. With both installed,
whichever comes first on your `PATH` answers — `eventcv --version` says which one that is.

## `eventcv --version`

```console
$ eventcv --version
eventcv 1.0.12 (hdf5, camera, onnx)
Python 3.12.10 on macOS-14.8.3-arm64-arm-64bit
```

The first line is the two things a bug report needs: the version, and which optional features were
compiled in. If `hdf5` is absent, `.h5` files will not open; if `onnx` is absent,
{class}`~eventcv.Model` is unavailable.

The standalone binary names itself on the second line instead of an interpreter:

```console
$ eventcv --version
eventcv 1.0.12 (hdf5, camera)
standalone binary on linux-x86_64
```

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
wrote recording.h5 (14,203,881 events, loaded)
```

The output format comes from the extension. The trailing word says how it was done: `streamed`
means the recording never had to fit in memory, which is the case for E2VID's `.zip`; `loaded`
means it was read whole first, which is the case for the other formats.

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
$ eventcv play
$ eventcv play recording.h5 --dt-ms 5
```

Opens the interactive player, empty when no recording is supplied. A file can also be opened from
the window or dropped onto it. The view is raw by default for the same reason `render` is.

At `--speed 1`, recording time matches wall time. `--fps` is accepted only for compatibility and
is deprecated. The player also takes `--dt-ms`, `--refresh-hz`, `--repr`, `--decay-ms`,
`--colormap`, and `--clim`.

- `--speed` — playback rate multiplier; `--speed 0.25` for a slow look at a fast scene.
- `--refresh-hz` — sliding-window refresh cap; defaults to 60 Hz independently of `--dt-ms`.
- `--loop` — restart at the end instead of pausing on the final frame.

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

## `eventcv tui`

```console
$ eventcv                              # the home screen (also `eventcv tui`)
$ eventcv tui recording.h5
$ eventcv tui --camera                 # the first camera found; --camera SERIAL picks one
```

Run in a terminal with nothing else, `eventcv` opens a home screen. It lists the cameras plugged
in and the recordings in the current folder. Pick one with the arrow keys and `enter`. `tab`
switches between the two lists, Backspace goes up a folder, and `r` looks for cameras again. A viewer
opened from here returns to it on `q`, which also releases the camera. When `eventcv` runs in a
script or a pipe, it prints its help instead.

The events in the terminal, next to the numbers that matter when something looks wrong — for a
quick look over SSH, on a headless capture box, or anywhere a GPU window is more than the job needs.
Each character cell draws two pixels (a `▀` half block in two colours). The sensor is scaled to fit,
keeping the brightest pixel of each block it shrinks, so a lone event does not vanish into the
average.

| Key | |
|---|---|
| `space` | pause / resume — at the end of a recording, play it again |
| `←` `→` | seek a second back / forward (recordings) |
| `+` `-` | double / halve the playback speed (recordings) |
| `[` `]` | halve / double the window on screen, between 0.1 ms and 1 s |
| `r` | next view: raw, then each representation in turn |
| `q`, `Esc`, `Ctrl-C` | quit |

The panel shows the sensor and either the playhead (position, duration, speed) or, for a camera,
its health:

- **backlog** — buffers waiting in the driver's ring. A number that keeps rising means the machine
  is not keeping up.
- **overflows** — times the ring filled and the driver dropped data.
- **dropped** — buffers skipped undecoded so the screen stays live at rates no terminal could show.
  Once that happens the rate is labelled `decoded`, because it no longer counts everything.
- with `--adaptive-bias`, the controller's measured rate, target band and state.

Below that, for the window on screen: the event count, the ON/OFF split, the rate, and a sparkline
of the rate over time.

Flags: `--dt-ms` (window, default 30), `--repr`, `--decay-ms`, `--colormap` and `--speed` as for
`play`, and `--refresh-hz` (default 30 — a full-colour screen is a lot of bytes, so lower it over a
slow SSH link). A camera also takes `--max-event-rate`, `--roi` and `--adaptive-bias`, as `record`
does.

Colour is 24-bit when the terminal says it supports it (`COLORTERM=truecolor`, and always on
Windows) and the nearest xterm-256 colour otherwise. On Windows, use Windows Terminal; the legacy
console draws slowly.

## `eventcv cameras`

Standalone binary only.

```console
$ eventcv cameras
name            kind            serial    speed  usb
Prophesee EVK4  prophesee_evk4  00051234  super  2:5
```

Lists the connected USB event cameras. An empty list with a camera plugged in usually means another
program has it open or, on Linux, that udev rules are missing — when enumeration was blocked by
permissions the command says so.

## `eventcv record`

Standalone binary only.

```console
$ eventcv record session.h5 --seconds 10
recorded 31,402,118 events in 10.0 s -> session.h5
```

Records a camera straight to disk. Every event format works, and the extension picks it; `.h5` is
the one for long sessions. Without `--seconds` it records until `Ctrl-C`, which ends the recording
cleanly and keeps everything captured (a second `Ctrl-C` quits at once).

- `--serial` — which camera, from `eventcv cameras` (default: the first found).
- `--compression` — HDF5 gzip level `0..9`. The default is uncompressed, so the disk keeps up with
  the sensor.
- `--max-event-rate`, `--roi X0 Y0 W H` — limit what the sensor emits, as in
  [Live camera streaming](streaming.md).
- `--adaptive-bias` — retune the biases as it runs to hold the event rate steady across changing
  light.

## Reading text and rosbag sources

Text formats carry no timestamp unit or column order, and a rosbag has no single obvious
connection, so `info`, `convert`, `render`, `play` and `tui` all accept:

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
