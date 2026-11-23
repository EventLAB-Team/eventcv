# Loading

{func}`~eventcv.load` reads a whole recording into an {class}`~eventcv.EventStream`. It picks the
reader from the file extension.

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
```

For files too large to load at once, use {func}`~eventcv.open` instead. See {doc}`large-files`.

## Formats

| Format | Extension | Notes |
|---|---|---|
| NumPy | `.npz` | |
| Text / CSV | `.txt`, `.csv` | Comma- or space-separated, with or without a header row. |
| HDF5 | `.h5`, `.hdf5` | Columns found by name, at any depth. |
| ROS bag | `.bag` | `dvs_msgs/EventArray`. Pick the topic with `topic=`. |
| AEDAT 2.0 | `.aedat` | DAVIS cameras. DVS128 and older chips are refused. |
| AEDAT 4 | `.aedat4` | iniVation DV. |
| Prophesee DAT | `.dat` | |
| Prophesee RAW | `.raw` | EVT2 and EVT3. |

## Options

| Option | Default | Meaning |
|---|---|---|
| `sensor_size` | inferred | `(width, height)`. Events outside it are dropped. |
| `time_unit` | inferred | Timestamp unit in the file: `"s"`, `"ms"`, `"us"` or `"ns"`. |
| `keys` | inferred | Column names or indices, e.g. `{"x": "u", "y": "v", "t": "ts", "p": "pol"}`. |
| `order` | `"txyp"` | Column order for headerless text. |
| `topic` | `/davis/left/events` | Topic to read from a ROS bag. |
| `offset` | `None` | Skip events before this absolute timestamp (ms, in the file's time base). |
| `max_events` | `None` | Keep at most this many events after `offset`. |

## What is inferred

EventCV fills in what you leave out:

- **Sensor size:** read from the file header where there is one (ROS bags, AEDAT, RAW), else the
  smallest grid holding every event.
- **Time unit:** from the timestamp span. Fractional values in a text file mean seconds.
- **Columns:** matched by name in HDF5 datasets and text headers. `x`, `x_coordinates` and `u` all
  map to `x`; `t`, `ts` and `timestamp` to `t`; and so on.

Pass the option explicitly to override any of these. If the columns can't be found, the error
lists what the file contains.

:::{tip}
For a large HDF5 file, passing `sensor_size` skips a scan of the coordinates.
:::

## Name the columns

```python
# HDF5: dataset paths, or dataset/field for a compound dataset
stream = ecv.load("events.h5", keys={"x": "ev/x", "y": "ev/y", "t": "ev/t", "p": "ev/p"})

# Text: header names or 0-based column indices
stream = ecv.load("events.csv", keys={"x": 1, "y": 2, "t": 0, "p": 3})
```

## Read part of a file

```python
stream = ecv.load("recording.h5", offset=1_000, max_events=500_000)  # from t = 1 s
```

`offset` is an absolute timestamp in milliseconds. For epoch-based recordings, pass the epoch
time, e.g. `offset=1_587_540_271_650`. `offset_s`, `offset_us` and `offset_ns` also work.

## From NumPy

{func}`~eventcv.from_numpy` builds a stream from an `(N, 4)` array. It is the inverse of
`stream.numpy()`.

```python
import numpy as np

events = np.array([[0, 0, 100, 1], [1, 2, 250, 0]])  # x, y, t, p
stream = ecv.from_numpy(events, sensor_size=(640, 480), time_unit="us")
```

Columns default to `xytp`; pass `order="txyp"` or `keys=` for other layouts. Structured arrays are
matched by field name, so [Tonic](https://tonic.readthedocs.io) samples work directly:

```python
import tonic

events, label = tonic.datasets.NMNIST(save_to="data")[0]
stream = ecv.from_numpy(events, sensor_size=(34, 34), time_unit="us")
```

:::{note}
Time-unit inference assumes a recording of at least about a second. Pass `time_unit` for short
arrays.
:::

## Frames, IMU and intrinsics

DAVIS recordings carry more than events. These read the extra data from a path or an open reader:

| Function | Returns | Formats |
|---|---|---|
| {func}`~eventcv.read_frames` | Greyscale frames as `[(t_us, EventFrame), …]` | ROS bag, AEDAT 2.0 |
| {func}`~eventcv.read_imu` | `{"t", "angular_velocity", "linear_acceleration"}` arrays | ROS bag, AEDAT 2.0 |
| {func}`~eventcv.read_camera_info` | A {class}`~eventcv.Camera`, or `None` | ROS bag |

```python
frames = ecv.read_frames("recording.bag", t0_us=0, t1_us=1_000_000)
imu = ecv.read_imu("recording.bag")
```

Pass a ROS bag or AEDAT 2.0 path. For other formats, call the method on an open reader
(`ecv.open(path).read_frames()`), which returns an empty result.
