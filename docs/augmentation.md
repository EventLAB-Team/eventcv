# Augmentation

Seeded random transforms for training. Each returns a new stream, and the same seed always gives the
same result.

```python
import eventcv as ecv

stream = ecv.load("recording.h5")

dropped = stream.event_drop(0.2, seed=0)  # remove about 20% of events
print(len(stream), len(dropped))

jittered = stream.spatial_jitter(1.5, seed=0)  # move events by ~1.5 px
jittered.view()
```

Use a different seed for a different result. Each augmentation is also a free function:
`ecv.event_drop(stream, 0.2, seed=0)`.

## Operations

| Method | Default | Effect |
|---|---|---|
| `random_flip_x(p, seed=)` | `p=0.5` | Mirror left–right with probability `p`. |
| `random_flip_y(p, seed=)` | `p=0.5` | Mirror top–bottom with probability `p`. |
| `random_polarity_flip(p, seed=)` | `p=0.5` | Invert **every** polarity with probability `p`. |
| `random_crop(width, height, seed=)` | | Take a random crop. No-op if the crop is as large as the sensor. |
| `event_drop(p, seed=)` | `p=0.1` | Drop each event with probability `p`. |
| `pixel_dropout(p, seed=)` | `p=0.1` | Silence a random fraction `p` of pixels entirely, like dead pixels. |
| `spatial_jitter(sigma, seed=)` | `sigma=1.0` | Move each event by a Gaussian offset, in pixels. Events pushed off the sensor are dropped. |
| `time_jitter(sigma_ms=, seed=)` | | Shift each timestamp by a Gaussian offset, then re-sort. |
| `time_reversal(p, seed=)` | `p=0.5` | Play backwards with probability `p`, inverting polarity to match. |

`random_polarity_flip` and `time_reversal` act on the whole stream, not on individual events.

## In a training loop

Call the augmentations in your dataset's `__getitem__`, with a seed derived from the epoch and
index:

```python
import eventcv as ecv
import torch


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.reader = ecv.open(path, dt_ms=30)
        self.epoch = 0

    def __len__(self):
        return self.reader.n_slices

    def __getitem__(self, i):
        seed = self.epoch * len(self) + i
        window = self.reader[i].hot_pixel_filter()
        window = window.random_flip_x(0.5, seed=seed)
        window = window.event_drop(0.1, seed=seed)
        return window.voxel(bins=5).numpy()


dataset = EventDataset("recording.h5")
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

for epoch in range(10):
    dataset.epoch = epoch  # new augmentations each epoch
    for batch in loader:
        ...
```

The same epoch and index always give the same sample, so a run is reproducible, including with
shuffling and multiple workers.

:::{tip}
Filter hot pixels before thinning events. Thinning first makes stuck pixels harder to detect.
:::

## On a reader

For a fixed augmentation with no custom dataset, apply it to the reader. It runs on every window
before the representation:

```python
reader = ecv.open("recording.h5", dt_ms=30, repr="voxel")
reader = reader.event_drop(0.1, seed=0)

loader = torch.utils.data.DataLoader(reader, batch_size=8, shuffle=True)
```

Each window is always augmented the same way, in every epoch and every worker.
