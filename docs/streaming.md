# Live cameras

{func}`~eventcv.stream` opens a USB event camera. It produces the same windows as a file reader, so
every transform, representation and viewer works on a live feed.

```python
import eventcv as ecv

ecv.list_cameras()  # [{'kind': 'prophesee_evk4', 'name': 'Prophesee EVK4', 'serial': ...}]
ecv.stream().show()  # live view; close the window to stop
```

| Supported cameras |
|---|
| Prophesee EVK4, EVK3 HD |
| iniVation DVXplorer, DAVIS346 |
| CenturyArks VGA |

:::{note}
On Linux, cameras need
[udev rules](https://github.com/neuromorphicsystems/neuromorphic-drivers#udev-rules) for non-root
access.
:::

## Read windows

```python
with ecv.stream(dt_ms=50) as cam:
    for events in cam:  # one EventStream per 50 ms
        process(events)
```

Pass `repr=` to receive frames instead. Representation options (`bins`, `window_ms`, `tau_ms`, …) go
straight to `stream`:

```python
with ecv.stream(dt_ms=50, repr="voxel", bins=5) as cam:
    while True:
        frame = cam.read()  # blocks until the next window
        infer(frame.numpy())
```

| Option | Meaning |
|---|---|
| `dt_ms` | Window length. |
| `max_events` | Fixed event count per window, instead of `dt_ms`. |
| `repr` | Render each window as this {doc}`representation <representations>`. |
| `serial` | Open a specific camera when several are connected. |

`cam.read(timeout_ms=100)` returns `None` if no window arrives in time. Empty windows are never
returned.

:::{important}
A camera can only be open once. Use `with`, or call `cam.close()`, so the device is released
promptly.
:::

## Record

Pass `record=` to save every window's raw events while you process them:

```python
with ecv.stream(dt_ms=50, repr="voxel", record="session.h5") as cam:
    for frame in cam:
        infer(frame.numpy())
```

To record without processing, use {func}`~eventcv.record`. It opens the camera, records and closes:

```python
ecv.record("session.h5", seconds=10)  # returns the number of events
```

Any event format works, chosen by extension. If a recording might be interrupted, choose one that
is readable while it grows:

| Readable during recording | Complete only once the camera closes |
|---|---|
| `.h5`, `.txt`, `.csv`, `.aedat`, `.aedat4`, `.dat`, `.raw` | `.npz`, `.bag` |

`compression=` sets the HDF5 gzip level (0–9). `cam.record(path, seconds=…)` records from a camera
you already have open. For full control, append windows yourself with an
{class}`~eventcv.EventSink` (see {doc}`saving`).

## DAVIS frames and IMU

Pass `frames=True` and `imu=True` to collect a DAVIS346's greyscale frames and IMU samples:

```python
with ecv.stream(dt_ms=50, frames=True, imu=True) as cam:
    for events in cam:
        frames = cam.read_frames()  # [(t_us, EventFrame), ...]
        imu = cam.read_imu()  # {"t", "angular_velocity", "linear_acceleration"}
```

They are returned in the same shape as {func}`~eventcv.read_frames` and {func}`~eventcv.read_imu`
return for a file. Read them every window: only the most recent few dozen are kept. These are not
written to `record=` files.

## Keep up with the camera

Capture runs on a background thread, so the camera is drained while your code works. If your loop is
slower than the camera, windows queue up and you fall behind. `latest=True` always returns the newest
window and skips the rest:

```python
with ecv.stream(dt_ms=30, latest=True, record="session.h5") as cam:
    for events in cam:
        slow_inference(events)  # skipped windows are still recorded
```

| Property | Meaning |
|---|---|
| `cam.backlog` | Buffers waiting in the driver. Should stay near zero. |
| `cam.n_skipped` | Windows skipped by `latest=True`. |
| `cam.n_overflows` | Times the driver dropped events before EventCV saw them. |
| `cam.n_recorded` | Events written by `record=`. |

## Limit what the sensor sends

A busy scene can produce more events than the host can decode. Limit them at the sensor:

```python
ecv.stream(dt_ms=50, max_event_rate=40_000_000)  # events per second
ecv.stream(dt_ms=50, roi=(320, 180, 640, 360))  # x0, y0, width, height
```

| Camera | `max_event_rate` | `roi` |
|---|---|---|
| Prophesee EVK4, EVK3 HD, CenturyArks VGA | On the sensor | On the sensor |
| iniVation DVXplorer, DAVIS346 | Not supported (raises) | Applied on the host, with a warning |

`cam.roi` reports where the ROI was applied. A rate cap thins events evenly across the scene, so it
is usually the better first choice; on an EVK4, around 40 million events per second keeps up on
one core.

## Mask a region

For a region that isn't a rectangle, or one you want to change while streaming, pass a `mask=`.
Events outside it are dropped before they reach your loop or the recording:

```python
aperture = ecv.circle_mask((640, 480), cx=320, cy=240, r=230)

with ecv.stream(dt_ms=50, mask=aperture) as cam:
    ...
```

Or draw one over the live view:

```python
cam = ecv.stream(dt_ms=50)
cam.draw_mask()  # draw, then press Enter; applied to the camera
ecv.save_mask(cam.mask, "aperture.png")
```

Set `cam.mask` to change it mid-stream, or to `None` to clear it. A mask is applied on the host, so it
doesn't reduce USB traffic; combine it with `roi=` on Prophesee cameras. See
{ref}`roi-masking` for building masks and the drawing controls.

## Adaptive biasing

Fixed sensor biases give very different event rates as lighting changes. `adaptive_bias=True` adjusts
them as the camera runs to hold the event rate steady (Nair et al., IROS 2024):

```python
with ecv.stream(dt_ms=50, adaptive_bias=True) as cam:
    for events in cam:
        print(cam.bias_state)  # event rate, bias values, controller state
```

For about the first second it measures the scene, then holds the rate it found. To aim for a
specific rate instead, pass `adaptive_bias={"target_rate": (3e4, 1.2e5)}` (events per second).
Supported on the DAVIS346 and EVK4; other cameras raise an error.

## Live viewer

```python
ecv.stream().show()  # raw events
ecv.stream(dt_ms=30).show("count")  # a representation
```

`show()` blocks until the window closes. Recording continues while the view is paused.
`show("aps")` shows a DAVIS's greyscale frames. See {doc}`video` for the viewer's controls.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Frame rate below `1000 / dt_ms`, `backlog` rising | Cap with `max_event_rate`, or narrow with `roi`. |
| `n_overflows` rising | The driver is dropping events. Cap the source as above. |
| Frames lag behind real time | Use `latest=True`. |
| "No event camera found" with one attached | Another handle has it open, or udev rules are missing. |
| Process aborts with a `usb.rs` panic on Windows after closing a camera | A known race in the underlying driver. Open the camera once per session rather than repeatedly. |
