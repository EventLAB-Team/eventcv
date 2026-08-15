# Motion estimation

Two ways of asking what moved: **contrast maximisation** recovers the camera's own motion from the
whole scene, and **tracking** follows individual objects across frames.

## Contrast maximisation

Events from a moving camera are smeared along the motion's path. Warp them by a candidate motion and
accumulate them into an *image of warped events*: guess right and the smear collapses into sharp
edges, guess wrong and it stays blurred. Maximising that sharpness recovers the motion.

```python
import eventcv as ecv

events = ecv.load("recording.h5").time_window(0, 50_000)
result = events.contrast_maximise()

print(result["params"])        # [vx, vy] in pixels per second
print(result["improvement"])   # how much sharper than assuming no motion
```

**Check `improvement` before trusting `params`.** At or below 1.0 the optimiser found no motion and
stopped wherever it happened to be — the parameters are meaningless. That happens when the slice is
too short, too sparse, or the scene genuinely was still.

### Warp models

| `model` | Parameters | When it applies |
| --- | --- | --- |
| `"translation"` | `vx, vy` (px/s) | A camera moving parallel to a flat scene, or any small patch |
| `"rotation"` | `wx, wy, wz` (rad/s) | A rotating camera — needs `camera=` intrinsics |

Rotation is where contrast maximisation is at its strongest: rotating a camera moves every ray the
same way regardless of how far away things are, so the warp is exact for *any* scene. Translation
is only exact at constant depth, which is why it is usually applied to a patch rather than a frame.

```python
camera = ecv.Camera(fx=320, fy=320, cx=320, cy=240)
result = events.contrast_maximise(model="rotation", camera=camera, initial_step=1.0)
```

`initial_step` is in the model's own units, so it needs changing between models — 50 px/s is a
sensible first probe for translation, 1 rad/s for rotation.

### Objectives

`"variance"` (default, Gallego & Scaramuzza RA-L 2017), `"sos"` and `"soe"` (Stoffregen & Kleeman
CVPR 2019). Variance is the standard choice. `soe` rewards concentration far more aggressively,
which sharpens the optimum but narrows the basin around it — better once you are close, worse when
starting from nothing.

Each blurs the accumulated image before scoring, which is not cosmetic: without it the objective is
a field of isolated spikes with no gradient between them and the optimiser has nothing to follow.
`blur_sigma` overrides the per-objective default.

### Looking at the result

```python
iwe = ecv.iwe(events, result["params"])   # the image the objective actually scored
iwe.save("sharp.png")
```

A blurred image at the "recovered" parameters is the clearest sign the warp model does not fit —
usually translation applied to a scene with real depth variation.

### Differences from the reference implementation

`event_utils` is the implementation every contrast-maximisation paper cites. Three things here
differ deliberately:

- **Events warp to the interval midpoint**, not to the last event's timestamp. Warping to the end
  gives the earliest events the largest displacement; the midpoint halves the worst case, and with
  it the error from assuming motion is linear over the interval.
- **Out-of-bounds events are dropped**, not folded onto pixel (0, 0). The reference multiplies
  coordinates by a 0/1 mask, which piles every escaped event into the corner and rewards warps that
  push events off the sensor — a bias directly against what the objective measures.
- **The optimiser is derivative-free** (Nelder-Mead). The reference uses BFGS, but its own
  documentation recommends numeric gradients as "more stable… less prone to noise", which is an
  argument for not needing gradients at all.

Cite Gallego et al., *A Unifying Contrast Maximization Framework for Event Cameras* (CVPR 2018) for
the framework itself.

## Tracking

[`connected_components`](representations.md) segments one frame but has no memory: labels are
assigned in scan order and renumber whenever anything moves. A `Tracker` adds the memory.

```python
tracker = ecv.Tracker(min_area=10, max_distance=25.0, max_missed=3)

for i in range(len(reader)):
    for track in tracker.update(reader[i]):
        print(track["id"], track["centroid"], track["velocity"])
```

Each track carries `id` (stable for its life, never reused), `centroid`, `velocity` in pixels per
frame, `area`, `age`, and `missed`.

| Setting | Does |
| --- | --- |
| `min_area` | Ignores blobs below this size — the main noise control, since a hot pixel makes a one-pixel component |
| `max_distance` | Gate beyond which a blob is a different object, not this one having moved |
| `max_missed` | Frames a track survives unmatched, which is what carries it through a brief occlusion |

Tracks are matched against their *predicted* position rather than their last one, so a fast object
stays inside the gate: something moving 20 px per frame is 20 px from where it was, but roughly zero
from where it was going.

### Where it fails

Blobs are matched greedily — closest pair first, then the next closest among what is left. **Two
objects passing close to each other can swap identities.** Greedy matching commits to the cheapest
pair before considering the rest, so at the crossing point the wrong pairing can be cheaper.

Nothing here prevents that. Resolving it needs an appearance or motion model strong enough to tell
the two apart, which is a larger piece of work than the association rule — if identity through
occlusion matters for your application, this tracker is the wrong tool rather than a tool to tune.

## Validating against known motion

Both algorithms can be checked against ground truth, because [the simulator](simulation.md) knows
exactly what motion it was given:

```python
import numpy as np

frames = np.zeros((24, 64, 96), dtype=np.uint8)
for i in range(24):
    bar = 12 + int(200.0 * i / 500.0)          # 200 px/s at 500 fps
    frames[i, :, bar:bar + 4] = 230

events = ecv.simulate(frames, fps=500)
print(events.contrast_maximise()["params"][0])  # ≈ 200
```

A recording cannot do this — it has no ground-truth motion attached, which is why the reference
implementations validate qualitatively or not at all. The test suite asserts this recovery, so a
regression in the warp, the objective or the optimiser fails the build rather than quietly
degrading the estimate.
