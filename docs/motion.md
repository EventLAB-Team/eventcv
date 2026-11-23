# Motion estimation

Estimate how the camera moved with contrast maximisation, or follow objects between frames with a
tracker.

## Contrast maximisation

Events from a moving camera smear along the motion. Contrast maximisation finds the motion that,
when undone, makes the events sharpest.

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
events = stream.time_window(t0_ms=0, t1_ms=50).hot_pixel_filter()
result = events.contrast_maximise()

result["params"]  # [vx, vy] in pixels per second
result["improvement"]  # sharpness gain over assuming no motion
```

:::{warning}
Check `improvement` before trusting `params`. At or below 1.0, no motion was found and `params` is
meaningless. The slice may be too short or too sparse, or the scene may be still.
:::

:::{tip}
Always remove hot pixels first. A stuck pixel doesn't move, so it makes the unwarped image look
sharpest and the result comes back as zero motion.
:::

### Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `"translation"` | `"translation"` (`vx, vy` in px/s) or `"rotation"` (`wx, wy, wz` in rad/s). |
| `camera` | `None` | A {class}`~eventcv.Camera`. Required for `"rotation"`. |
| `objective` | `"variance"` | `"variance"`, `"sos"` or `"soe"`. `"soe"` is sharper near the answer but needs a closer start. |
| `initial` | zeros | Starting guess for `params`. |
| `initial_step` | `50.0` | First search step, in the model's units. Use about `1.0` for rotation. |
| `blur_sigma` | per objective | Blur applied before scoring. |
| `max_iterations`, `tolerance` | `200`, `1e-3` | Optimiser limits. |

Rotation fits any scene. Translation assumes everything is at a similar depth, so it suits small
patches or flat scenes.

```python
camera = ecv.Camera(fx=320, fy=320, cx=320, cy=240)
result = events.contrast_maximise(model="rotation", camera=camera, initial_step=1.0)
```

### Check the result

```python
iwe = ecv.iwe(events, result["params"])  # the sharpened image
iwe.save("sharp.png")
```

A blurry image at the recovered motion means the model doesn't fit the scene. The usual cause is
translation on a scene with large depth differences.

## Tracking

{class}`~eventcv.Tracker` follows connected blobs across frames and keeps their identities.

```python
reader = ecv.open("recording.h5", dt_ms=30, repr="binary")
tracker = ecv.Tracker(min_area=10, max_distance=25.0)

for i in range(reader.n_slices):
    for track in tracker.update(reader.slice(i)):
        print(track["id"], track["centroid"], track["velocity"])
```

| Setting | Default | Meaning |
|---|---|---|
| `min_area` | `4` | Ignore smaller blobs. The main noise control. |
| `max_distance` | `20.0` | Largest jump, in pixels, still treated as the same object. |
| `max_missed` | `3` | Frames a track survives without a match, e.g. through a brief occlusion. |
| `connectivity` | `8` | `4` or `8`. |

Each track is a dict with `id`, `centroid`, `velocity` (pixels per frame), `area`, `age` and
`missed`. IDs are never reused.

:::{note}
Blobs are matched to each track's predicted position, closest first. Two objects that pass close to
each other can swap identities.
:::
