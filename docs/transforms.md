# Transforms

Transforms return a new {class}`~eventcv.EventStream`, so they chain. The input is never changed.

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
cropped = stream.crop(0, 0, 320, 240)
clean = cropped.hot_pixel_filter()
```

Each one is also a free function (`ecv.flip_x(stream)`) and works on a reader, where it applies to
every window. For random transforms used in training, see {doc}`augmentation`.

## Geometry

| Method | Effect |
|---|---|
| `crop(x0, y0, w, h)` | Keep a `w`×`h` window at `(x0, y0)` and move it to the origin. |
| `flip_x()`, `flip_y()` | Mirror horizontally or vertically. |
| `rotate90(k)` | Rotate by `k` quarter turns clockwise. |
| `transpose()` | Swap `x` and `y`. |
| `translate(dx, dy)` | Shift. Events that leave the sensor are dropped. |
| `resize(width, height)` | Rebin onto a new grid. Keeps every event. |
| `scale(sx, sy)` | Resize by a factor. |
| `warp_affine(matrix)` | Apply a 2×3 affine matrix. |
| `warp_perspective(matrix)` | Apply a 3×3 homography. |
| `undistort(camera)` | Remove lens distortion. See [below](#lens-undistortion). |

Coordinates are rounded, not interpolated: events move to a pixel, they aren't spread over several.

## Time

| Method | Effect |
|---|---|
| `time_window(t0_ms=…, t1_ms=…)` | Keep events in `[t0, t1)`. |
| `time_shift(dt_ms=…)` | Add a (possibly negative) offset to every timestamp. |
| `time_scale(factor)` | Multiply every timestamp. |
| `normalize_time()` | Shift so the first event is at `t = 0`. |
| `sort_by_time()` | Sort events by timestamp. |

`time_window` and `time_shift` also take a bare positional value in microseconds.

## Polarity

| Method | Effect |
|---|---|
| `filter_polarity(p)` | Keep only ON (`1`/`True`) or OFF (`0`/`False`) events. |
| `invert_polarity()` | Swap ON and OFF. |

## Noise filters

| Method | Effect |
|---|---|
| `hot_pixel_filter(n_std=3.0)` | Drop pixels that fire more than `mean + n_std·std`. |
| `background_activity_filter(dt)` | Keep an event only if a neighbour fired within `dt` µs. |
| `refractory_filter(dt)` | Ignore a pixel's events for `dt` µs after it fires. |

`background_activity_filter` takes `neighbourhood="block"` (8 neighbours, default) or `"cross"` (4,
stricter). `refractory_filter` takes `refresh="kept"` (default) or `"all"`, which restarts the dead
time on every event, as Tonic does.

## Combining and thinning

| Method | Effect |
|---|---|
| `concat([a, b, …])` | Append other streams. The sensor grows to fit all of them. |
| `decimate(k)` | Keep every `k`-th event. |

(roi-masking)=
## Regions of interest

A mask is an `(H, W)` boolean array that is `True` where events are kept. Build one, combine masks
with NumPy operators, then apply it:

```python
aperture = ecv.circle_mask((640, 480), cx=320, cy=240, r=230)
aperture &= ~ecv.rect_mask((640, 480), 0, 0, 64, 64)  # drop a corner

stream.mask(aperture)
ecv.open("huge.h5", dt_ms=30).mask(aperture)  # every window
ecv.save_mask(aperture, "aperture.png")
```

| Function | Shape |
|---|---|
| {func}`~eventcv.circle_mask` | `(sensor_size, cx, cy, r)` |
| {func}`~eventcv.ellipse_mask` | `(sensor_size, cx, cy, rx, ry)` |
| {func}`~eventcv.rect_mask` | `(sensor_size, x0, y0, width, height)` |
| {func}`~eventcv.polygon_mask` | `(sensor_size, points)` |
| {func}`~eventcv.save_mask`, {func}`~eventcv.load_mask` | Save or load as an image. |

A pixel is kept when its centre falls inside the shape. Any 8-bit array works too: non-zero keeps the
pixel. A mask that doesn't match the sensor size raises an error.

### Draw a mask

```python
mask = stream.draw_mask()  # None if cancelled
```

This opens the viewer on the stream. Draw over the area to keep:

| Input | Action |
|---|---|
| Drag | Keep an area |
| Shift + drag | Remove an area |
| `e` / `r` / `f` | Ellipse, rectangle or freehand |
| `a` / `c` | Select all / clear |
| `z` | Undo |
| `Enter` / `Esc` | Accept / cancel |

{meth}`~eventcv.EventFrame.draw_mask` draws on a frame, and {meth}`~eventcv.EventCamera.draw_mask`
draws on a live camera. To mask a camera as it streams, see {doc}`streaming`.

## Lens undistortion

```python
camera = ecv.Camera(fx=320.0, fy=320.0, cx=320.0, cy=240.0, k1=-0.1, k2=0.01)
rectified = stream.undistort(camera)
```

{class}`~eventcv.Camera` holds pinhole intrinsics and Brown–Conrady distortion (`k1`, `k2`, `k3`,
`p1`, `p2`). {func}`~eventcv.read_camera_info` reads one from a ROS bag.
