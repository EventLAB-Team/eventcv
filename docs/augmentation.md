# Augmentation

Augmentations are the training-time counterparts of the deterministic ops in
[Quickstart](quickstart.md). Each takes an explicit `seed` and is a pure function of its inputs, so
a training run is reproducible from its configuration alone.

```python
import eventcv as ecv

stream = ecv.load("recording.h5")
noisy = stream.event_drop(0.1, seed=0).spatial_jitter(1.5, seed=0)
```

Like every other operation, each is available both as a method and as a free function:
`ecv.event_drop(stream, 0.1, seed=0)`.

## The operations

| Op | Signature | What it does |
| --- | --- | --- |
| `random_flip_x` | `(p=0.5, *, seed=0)` | Mirrors left–right with probability `p`. |
| `random_flip_y` | `(p=0.5, *, seed=0)` | Mirrors top–bottom with probability `p`. |
| `random_polarity_flip` | `(p=0.5, *, seed=0)` | Inverts **every** polarity with probability `p`. |
| `random_crop` | `(width, height, *, seed=0)` | Random `width × height` window. |
| `event_drop` | `(p=0.1, *, seed=0)` | Drops each event independently with probability `p`. |
| `pixel_dropout` | `(p=0.1, *, seed=0)` | Silences a random `p` fraction of *pixels* entirely. |
| `spatial_jitter` | `(sigma=1.0, *, seed=0)` | Gaussian position offset, `sigma` in pixels. |
| `time_jitter` | `(sigma_ms=1.0, *, seed=0)` | Gaussian timestamp offset; re-sorts afterwards. |
| `time_reversal` | `(p=0.5, *, seed=0)` | Plays backwards, inverting polarity to match. |

`time_jitter` accepts the usual time-unit variants — `sigma_s`, `sigma_ms`, `sigma_us`, `sigma_ns`.

A few of these behave differently from how their names might read:

- **`random_polarity_flip` draws once for the whole stream**, not per event. Flipping a random
  *subset* of polarities is label noise; flipping all of them is the physically meaningful
  augmentation — the same scene with the contrast direction reversed.
- **`time_reversal` inverts polarity too.** Running motion backwards without it would be
  physically wrong: an edge that brightened as it passed darkens when the motion reverses.
  Timestamps are mirrored within the stream's own span, so the result starts and ends where the
  original did.
- **`pixel_dropout` is not `event_drop`.** The first removes every event from the chosen pixels —
  a sensor with dead pixels — and is much harder for a model to average away than independent
  thinning.
- **`spatial_jitter` can thin the stream.** Events pushed off the sensor are dropped, so a large
  `sigma` removes events as well as moving them.
- **`random_crop` is a no-op** when the window is at least as large as the sensor, so it is safe to
  leave in a pipeline that also runs on smaller recordings.

## Reproducibility

The contract is that **a slice augments the same way every time it is reached**, and different
slices augment differently.

On an {class}`~eventcv.EventStream` that is simply "same seed, same output". On an
{class}`~eventcv.EventReader` it is stronger, and it is the property that matters for training: the
seed is mixed with the *index of the slice*, not carried between calls. So slice 7 augments
identically whether you reach it by `reader[7]`, inside `batch([3, 7, 1])`, or by iterating — and
identically again in a different worker process.

```python
reader = ecv.open("recording.h5", dt_ms=30).event_drop(0.1, seed=0).with_repr("voxel", bins=5)

reader[7]                    # always the same array
reader.batch([3, 7, 1])[1]   # ... including here
```

This is what makes a shuffled, multi-worker `DataLoader` reproducible:

```python
from torch.utils.data import DataLoader

loader = DataLoader(reader, batch_size=8, shuffle=True, num_workers=4)
```

Without per-slice seeding, an RNG carried between calls would make the result depend on the order
workers happened to touch slices — and two runs of the same script would differ. Change `seed` to
get a different augmentation of the same data; keep it to reproduce a run exactly.

## Composing

Augmentations chain with every other deferred op, and the order is the order you write:

```python
reader = (
    ecv.open("recording.h5", dt_ms=30)
    .hot_pixel_filter()            # clean first
    .random_flip_x(0.5, seed=0)    # then augment
    .event_drop(0.1, seed=0)
    .with_repr("voxel", bins=5)    # then render
)
```

Filter before augmenting: a hot-pixel filter estimates which pixels are stuck from their event
counts, and thinning the stream first makes that estimate worse.

## A note on cost

Every op returns a new stream, and the "did not fire" branch of a probabilistic augmentation copies
the input. On large slices, prefer chaining the ops you want over calling one you expect to be a
no-op most of the time.
