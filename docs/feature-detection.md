# Feature detection

EventCV's feature tools are the event-domain counterpart of OpenCV's `features2d`:

- **Corner detectors** — stateless keypoint filters ({meth}`~eventcv.EventStream.efast`,
  {meth}`~eventcv.EventStream.harris_corners`) that return a sub-stream of the events sitting on a
  moving corner, so they chain like a denoiser and feed any representation.
- **FEAST** ({class}`~eventcv.FEAST`) — an unsupervised, *trainable* feature extractor: `fit` it on a
  recording, then `transform` events into learned-feature space. The event analogue of a learned
  descriptor.

All assume events are in ascending time order (call {meth}`~eventcv.EventStream.sort_by_time` first
if not).

## Corner detection

```python
import eventcv as ecv

stream  = ecv.load("recording.npz")
corners = stream.efast()             # events on moving corners (a sub-stream)
harris  = stream.harris_corners()    # threshold=0.0 keeps corners, rejects straight edges
corners.count().view()               # corners feed any representation
```

| Detector | Method | Keeps an event when… |
|----------|--------|----------------------|
| eFAST  | {meth}`~eventcv.EventStream.efast` | its recent neighbours form a contiguous arc on both Bresenham rings — a moving corner, not an edge (Mueggler et al., BMVC 2017). |
| Harris | {meth}`~eventcv.EventStream.harris_corners` | the normalised SAE Harris response `det/trace² - k` exceeds `threshold` (default `0.0`; the score is bounded to `[-0.04, 0.21]`, so raise `threshold` within that range to be stricter). |

Over an {class}`~eventcv.EventReader` they apply per slice: `ecv.open(...).efast()` returns a reader
of corner sub-streams, ready for {func}`~eventcv.export_png`.

## FEAST feature learning

{class}`~eventcv.FEAST` learns prototypical **spatiotemporal features** online and without labels
(Afshar et al., *Sensors* 2020 — [paper](https://www.mdpi.com/1424-8220/20/6/1600) /
[arXiv](https://arxiv.org/abs/1907.07853)). For each event it takes the local `patch × patch`
time-surface window, normalises it, and matches it to the nearest feature *within an adaptive
threshold*: a match nudges that feature toward the input and tightens its threshold, a miss loosens
every threshold. Features converge on the recording's most common local patterns.

```python
stream = ecv.load("data/test/example.npz")   # N-ImageNet: a photo scanned by a moving camera
feast  = ecv.FEAST(n_features=25, patch=11, tau_ms=30.0, per_polarity=False, seed=0)
feast.fit(stream, epochs=3)                  # unsupervised; returns the miss rate
print(feast.missed_rate)                     # ~0.013 — a convergence proxy (paper reports ~2%)

ids  = feast.transform(stream)               # (N,) nearest-feature id per event (-1 at borders)
hist = feast.histogram(stream)               # pooled feature counts (a classifier input)
imgs = feast.feature_images()                # (n_features_total, patch, patch) learned patches
```

:::{figure} images/feast_input.png
:alt: Count image of the N-ImageNet sample — a bird outlined by edge events plus background noise.
:width: 460px

The input: object contours generate events (warm = more events) against a noisy background.
:::

### Reading the features

Tile `feature_images()` into a grid to reproduce the paper's feature plots (needs `matplotlib`):

```python
import numpy as np
import matplotlib.pyplot as plt

def montage(imgs):
    n, w, _ = imgs.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.full((rows * (w + 1) - 1, cols * (w + 1) - 1), np.nan, np.float32)
    for i, patch in enumerate(imgs):
        r, c = divmod(i, cols)
        lo, hi = patch.min(), patch.max()
        grid[r*(w+1):r*(w+1)+w, c*(w+1):c*(w+1)+w] = (patch - lo) / (hi - lo) if hi > lo else 0
    return grid

plt.imshow(montage(feast.feature_images()), cmap="turbo"); plt.axis("off"); plt.show()
```

:::{figure} images/feast_features.png
:alt: 25 learned FEAST features — oriented edge patterns plus a few near-empty noise features.
:width: 400px

25 learned features. Each tile encodes local event **timing**, not intensity.
:::

- The maroon **centre dot** is the triggering event — always the newest pixel, so the peak.
- Warm→cool (`turbo`) runs **recent→old**: a smooth ramp is a **moving edge**, and because the patch
  is normalised each feature codes an *orientation* (the event-camera Gabor filter), not a speed.
- **Near-empty tiles** are **noise features** — one or two soak up uncorrelated events and act as
  free noise detectors (2–4 is healthy).

Features start as random points and `fit` sculpts them into structure:

:::{figure} images/feast_before_after.png
:alt: The same features as random speckle before fit, and as oriented edges after fit.
:width: 680px

Random init (left) → learned features (right). This transformation is the falling miss rate.
:::

```{note}
The montage stretches each tile independently, which exaggerates the flat noise features; use a
shared `vmin=0, vmax=imgs.max()` to see them render flat.
```

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_features`   | `100`   | Feature prototypes **per polarity population**. |
| `patch`        | `11`    | Side length `w` of the square ROI; must be odd. |
| `tau_ms`       | `30.0`  | Time-surface decay constant (ms). Shorter → faster motion. |
| `eta`          | `0.001` | Weight mixing rate `η` in `w ← (1−η)w + η·d`. |
| `delta_i`      | `0.001` | Threshold contraction on a match. |
| `delta_e`      | `0.003` | Threshold expansion on a miss. |
| `per_polarity` | `True`  | Train independent ON/OFF banks; `False` merges both polarities. |
| `seed`         | `0`     | RNG seed for feature init (reproducibility). |

`fit` can be called repeatedly to train across recordings (weights persist, the time surface
resets); `transform` and `histogram` never mutate the model.

### More

- **Per-polarity** (default): ON and OFF train separate banks, so `feature_images()` has
  `2 * n_features` rows — ON first, then OFF (`imgs[:n]`, `imgs[n:]`). Use `per_polarity=False` to
  merge, e.g. for ON-only data.
- **Save / load**: `ecv.save(feast, "model.npz")` and {func}`~eventcv.load_feast` round-trip the
  trained model exactly.
- **Large files**: `fit` takes a whole stream, so train across a huge recording by iterating a
  reader — `for w in ecv.open("huge.hdf5", dt_ms=30).windows(): feast.fit(w)`.
