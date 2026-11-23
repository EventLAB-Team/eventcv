# Feature detection

Corner detectors that keep only the events on moving corners, and FEAST, a feature extractor that
learns from your data.

All of them expect events sorted by time. Call {meth}`~eventcv.EventStream.sort_by_time` first if
yours aren't.

## Corners

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
corners = stream.efast()  # events on moving corners
corners = stream.harris_corners(threshold=0.05)
corners.count().view()
```

Both return a stream with only the corner events, so any transform or representation works on the
result.

| Method | Keeps an event when | Parameters |
|---|---|---|
| {meth}`~eventcv.EventStream.efast` | Its recent neighbours form a corner, not a straight edge (Mueggler et al., BMVC 2017). | None |
| {meth}`~eventcv.EventStream.harris_corners` | Its Harris score on the time surface exceeds `threshold`. | `threshold=0.0`. The score lies in `[-0.04, 0.21]`; raise it to be stricter. |

On a reader, both apply to every window:

```python
corners = ecv.open("huge.h5", dt_ms=30).efast()
```

## FEAST

{class}`~eventcv.FEAST` learns a set of local spatio-temporal features without labels (Afshar et al.,
[Sensors 2020](https://www.mdpi.com/1424-8220/20/6/1600)). Fit it on a recording, then describe each
event by its nearest feature.

```python
feast = ecv.FEAST(n_features=25, patch=11, tau_ms=30, per_polarity=False, seed=0)
feast.fit(stream, epochs=3)  # returns the miss rate

ids = feast.transform(stream)  # (N,) feature id per event, -1 near the border
hist = feast.histogram(stream)  # feature counts, e.g. as classifier input
patches = feast.feature_images()  # (25, 11, 11) learned features
```

:::{figure} images/feast_before_after.png
:alt: FEAST features as random speckle before fitting, and as oriented edges after.
:width: 680px

25 features before fitting (left) and after (right). Each tile shows event timing, from old (cool) to
recent (warm).
:::

| Parameter | Default | Meaning |
|---|---|---|
| `n_features` | `100` | Features to learn, per polarity. |
| `patch` | `11` | Side of the square neighbourhood, in pixels. Must be odd. |
| `tau_ms` | `30.0` | Time-surface decay. Shorter suits faster motion. |
| `per_polarity` | `True` | Learn separate ON and OFF features. `feature_images()` then returns ON first, then OFF. |
| `eta` | `0.001` | Learning rate. |
| `delta_i`, `delta_e` | `0.001`, `0.003` | How fast thresholds tighten on a match and loosen on a miss. |
| `seed` | `0` | Random seed for the starting features. |

The miss rate (`feast.missed_rate`) falls as features converge; a few per cent is typical.

### Train, save and reuse

```python
for window in ecv.open("huge.h5", dt_ms=30).windows():
    feast.fit(window)  # keeps training across calls

ecv.save(feast, "feast.npz")
feast = ecv.load_feast("feast.npz")
```

`transform` and `histogram` never change the model.

:::{dropdown} Plot the learned features
```python
import matplotlib.pyplot as plt
import numpy as np

patches = feast.feature_images()
n, w, _ = patches.shape
cols = int(np.ceil(np.sqrt(n)))
fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols, cols))
for ax in axes.flat:
    ax.axis("off")
for ax, patch in zip(axes.flat, patches):
    ax.imshow(patch, cmap="turbo")
plt.show()
```

The bright centre is the triggering event. A smooth gradient across a tile is a moving edge at that
orientation. One or two nearly empty tiles are normal: they collect noise events.
:::
