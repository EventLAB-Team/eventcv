# Command line

The `eventcv` command inspects, converts and renders recordings without writing any Python.

```console
eventcv info recording.h5
eventcv convert recording.raw recording.h5
eventcv render recording.h5 preview.gif
```

It comes with the Python package (also as `python -m eventcv`) and as a standalone binary that needs
no Python (see {doc}`installation`). Run `eventcv --version` to see which one is on your `PATH` and
which features it has.

| Command | Does | Package | Binary |
|---|---|:-:|:-:|
| [`info`](#info) | Summarise a recording. | ✓ | ✓ |
| [`convert`](#convert) | Convert to another format. | ✓ | ✓ |
| [`render`](#render) | Render to `.gif`, `.apng` or `.mp4`. | ✓ | ✓ |
| [`play`](#play) | Open the interactive player. | ✓ | |
| [`tui`](#tui) | Watch a recording or camera in the terminal. | ✓ | ✓ |
| [`simulate`](#simulate) | Turn a video into events. | ✓ | ✓ |
| [`reconstruct`](#reconstruct) | Turn events into a video. | ✓ | |
| [`cameras`](#cameras) | List connected cameras. | | ✓ |
| [`record`](#record) | Record a camera to a file. | | ✓ |
| [`ros2-*`](#ros-2) | Publish, record and process ROS 2 topics. | `ros2` builds | |

In Python, `cameras` and `record` are {func}`~eventcv.list_cameras` and {func}`~eventcv.record`.

## Reading text and ROS bags

Text files don't record their time unit or column order, and a bag can hold several topics. Commands
that read a recording accept:

| Flag | Meaning |
|---|---|
| `--time-unit {s,ms,us,ns}` | Timestamp unit of a text file. |
| `--order COLS` | Column order of a headerless text file, e.g. `txyp`. |
| `--topic TOPIC` | Bag topic to read. |

## info

```console
$ eventcv info recording.h5
     file  recording.h5
   sensor  640 x 480
   events  106,295
 duration  0.050 s
mean rate  2,137,958 ev/s
```

Reads only the index, so it's fast on large files. `--rate-bin-ms 10` also reports the peak event
rate.

## convert

```console
$ eventcv convert recording.raw recording.h5
wrote recording.h5 (14,203,881 events, loaded)
```

The output format comes from the extension. `.zip` writes E2VID's input format and streams the file;
other formats load it into memory first.

## render

```console
$ eventcv render recording.h5 preview.gif --dt-ms 20 --fps 25
wrote preview.gif (48 frames at 25 fps)
```

| Flag | Default | Meaning |
|---|---|---|
| `--dt-ms` | `30` | Recording time per frame. |
| `--fps` | `30` | Playback rate. |
| `--repr` | `raw` | `raw` events, or a representation name. |
| `--decay-ms` | one frame | Raw view only: trail fade time. |
| `--colormap` | `viridis` | Colormap for single-channel representations. |
| `--clim` | calibrated | Fixed brightness scale, to make renders comparable. |
| `--max-frames` | all | Stop early. |

`.mp4` needs `ffmpeg`. See {doc}`video`.

## play

```console
eventcv play recording.h5 --dt-ms 5 --speed 0.25 --loop
```

Opens the interactive player; with no file, opens empty. Takes the `render` flags plus `--speed`
(default `1`), `--refresh-hz` (default `60`) and `--loop`.

## tui

```console
eventcv                       # home screen: pick a camera or recording
eventcv tui recording.h5
eventcv tui --camera          # first camera; --camera SERIAL for a specific one
```

Shows events in the terminal, alongside rate, ON/OFF split and, for cameras, buffer health. Useful
over SSH or on a machine without a display. Terminals with image support (kitty, Ghostty, WezTerm,
iTerm2, foot, Konsole) show full resolution; others use half-block characters.

| Key | Action |
|---|---|
| `space` | Pause / resume |
| `←` `→` | Seek 1 s (recordings) |
| `+` `-` | Double / halve speed (recordings) |
| `[` `]` | Halve / double the time window |
| `w` | Type a time window in ms |
| `f` | Filters: background activity, refractory, hot pixels |
| `r` | Cycle raw view and representations |
| `q`, `Esc` | Quit |

Flags: `--dt-ms`, `--repr`, `--decay-ms`, `--colormap`, `--speed`, `--refresh-hz` (default `30`;
lower it over slow SSH). For cameras, also `--max-event-rate`, `--roi X0 Y0 W H` and
`--adaptive-bias`.

## simulate

```console
$ eventcv simulate clip.mp4 events.h5 --scale 960 540
wrote events.h5 (18,742,110 events from 39 frames)
```

Writes events as they're generated, so memory stays flat. Needs `ffmpeg`.

| Flag | Meaning |
|---|---|
| `--scale W H` | Decode at a smaller size. The biggest saving in time and file size. |
| `--threshold`, `--sigma-thres`, `--cutoff-hz`, `--leak-rate-hz`, `--shot-noise-rate-hz`, `--refractory-us`, `--seed` | Sensor model. See {doc}`simulation`. |
| `--upsample`, `--max-upsample` | Frame subdivision. |
| `--max-frames` | Stop early. |
| `--compression` | HDF5 gzip level `0`–`9` (default `1`). |

## reconstruct

```console
eventcv reconstruct recording.h5 out.mp4 --device gpu
eventcv reconstruct recording.h5 out.mp4 my_model.onnx --dt-ms 33 --bins 5
```

Uses E2VID unless an ONNX model is given. Flags: `--dt-ms`, `--device {cpu,gpu}`, `--fps`,
`--max-frames`, and for ONNX models `--repr` and `--bins`. See {doc}`reconstruction`.

## cameras

```console
$ eventcv cameras
name            kind            serial    speed  usb
Prophesee EVK4  prophesee_evk4  00051234  super  2:5
```

An empty list with a camera attached usually means another program has it open, or udev rules are
missing on Linux.

## record

```console
$ eventcv record session.h5 --seconds 10
recorded 31,402,118 events in 10.0 s -> session.h5
```

Without `--seconds`, records until `Ctrl-C`. Flags: `--serial`, `--compression` (default
uncompressed), `--max-event-rate`, `--roi X0 Y0 W H`, `--adaptive-bias`. See {doc}`streaming`.

## ROS 2

| Command | Does |
|---|---|
| `ros2-publish FILE` | Replay a recording onto a topic. |
| `ros2-record FILE` | Record a topic to a file. |
| `ros2-represent` | Subscribe to events and republish a representation as `sensor_msgs/Image`. |

These need a build with the `ros2` feature, which the published wheels don't include. No ROS install
is needed: they talk Zenoh, so the ROS 2 side must run `rmw_zenoh_cpp` (or a `zenoh-bridge-ros2dds`).
Run `eventcv ros2-publish --help` for flags.

## Exit codes

`0` on success, `1` on error, `130` on `Ctrl-C`. Errors print one line to stderr:

```console
$ eventcv render recording.h5 out.mp4
error: writing .mp4 needs ffmpeg on PATH (macOS: `brew install ffmpeg`, ...)
```
