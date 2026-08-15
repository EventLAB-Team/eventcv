# Simulation

{func}`eventcv.simulate` turns a video, or an array of frames, into the events a DVS would have
produced watching the same scene.

```python
import eventcv as ecv

events = ecv.simulate("clip.mp4")                 # realistic defaults
events = ecv.simulate(frames, fps=1000)           # from a [N, H, W] array
```

The result is an ordinary {class}`~eventcv.EventStream` — save it, slice it, build representations
from it, feed it to a `DataLoader`. Reading video goes through `ffmpeg`, so a video source needs it
on `PATH`; simulating from an array of frames needs nothing.

## The pixel model

The model follows [v2e](https://arxiv.org/abs/2006.07722) (Hu et al., CVPRW 2021). Each part exists
because leaving it out produces synthetic data with a specific, recognisable tell.

| Parameter | Default | What it models |
| --- | --- | --- |
| `pos_thres`, `neg_thres` | `0.2` | Log-contrast change that emits an event |
| `sigma_thres` | `0.03` | Per-pixel spread of those thresholds |
| `cutoff_hz` | `200` | Photoreceptor bandwidth for a white pixel |
| `leak_rate_hz` | `1.0` | Spontaneous ON events with no scene change |
| `shot_noise_rate_hz` | `10.0` | Noise floor, in the darkest pixels |
| `refractory_us` | `100` | Dead time after each event at a pixel |

**Linearisation and the lin-log map.** Video is gamma-encoded. Taking `log` of the stored 8-bit
value measures contrast in display space rather than in light, which is wrong everywhere and most
wrong in shadows. Frames are linearised out of sRGB first. Below about 20/255 the map is linear
rather than logarithmic, because `log` of a near-zero value turns single-level quantisation steps
into large apparent contrast, and the sensor emits events that no real camera would.

**Per-pixel threshold mismatch.** Real thresholds vary by a few percent across the array. With a
single fixed threshold every pixel along an edge fires simultaneously, which is the most visible way
synthetic events look synthetic. Set `sigma_thres=0` to turn it off — useful for testing, wrong for
training data.

**Photoreceptor bandwidth.** A first-order lowpass whose cutoff falls with intensity, floored at 10%
of `cutoff_hz` so dark pixels still track. This is what makes dark scenes lag, the dominant artefact
in low-light recordings.

**Leak and shot noise.** Background activity unrelated to motion. Leak decays the memorised level so
a completely still scene still produces occasional ON events; shot noise is Poisson and quieter in
bright pixels, by a factor of four at white. A model trained on noiseless events has never seen
either.

**Timestamps are interpolated.** When a pixel crosses its threshold several times between two
frames, the crossings are placed by *when they happened* within the interval, not all stamped with
the frame time. Timing is the signal in event data — collapsing it to the frame rate throws away the
temporal precision that motivates using an event camera at all.

For a clean, ideal sensor:

```python
events = ecv.simulate("clip.mp4", sigma_thres=0, cutoff_hz=0,
                      leak_rate_hz=0, shot_noise_rate_hz=0, refractory_us=0)
```

That configuration is also what makes the simulator testable: an ideal pixel's event count is
`floor(|Δ log I| / threshold)`, so halving the threshold exactly doubles the events. The test suite
asserts precisely that.

## Upsampling

Between two frames the true intensity path is unknown and is assumed linear. The more that happens
in between, the worse that assumption gets — so `upsample` subdivides the interval first.

```python
ecv.simulate("clip.mp4", upsample="adaptive")   # default
ecv.simulate("clip.mp4", upsample="off")        # straight from the source frames
ecv.simulate("clip.mp4", upsample="8")          # fixed factor
```

`"adaptive"` subdivides until no pixel would emit more than `max_events_per_pixel` (default 1) per
sub-interval. v2e derives its factor from optical flow, keeping motion under a pixel per
sub-interval; this uses **contrast** instead, because what actually bounds the timestamp error is how
many threshold crossings are being packed into one linear interpolation — and that can be measured
directly, with no flow estimate to compute or get wrong.

Upsampling also matters for the noise model. At most one noise event per polarity is emitted per
sub-interval, so a very high `shot_noise_rate_hz` relative to the frame interval saturates;
subdividing is the fix.

## Reproducibility and cost

`seed` makes a run a pure function of its configuration — the same seed gives byte-identical events,
so a training run is reproducible from its config alone.

Simulation streams: only the current frame pair is held, so a long clip costs memory proportional to
one frame interval rather than to the whole output. `scale=(w, h)` decodes at a smaller size, which
is much cheaper than decoding full-size and downsampling afterwards, and `max_frames` stops early.

## Calibration against a real recording

The parameters above are a realistic default, not *your* camera. To match a specific sensor, record
a scene with it, simulate the same scene, and compare event rate, ON/OFF ratio and inter-event
interval distribution — {meth}`~eventcv.EventStream.event_rate` gives the first two directly:

```python
real = ecv.load("recorded.h5").event_rate(bin_ms=10)
fake = ecv.simulate("same_scene.mp4", pos_thres=0.25).event_rate(bin_ms=10)
```

Raise the thresholds if the simulation is too busy, lower them if too quiet; match the quiet
stretches with `leak_rate_hz` and `shot_noise_rate_hz` before touching the thresholds, since noise
sets the floor and contrast sets the peaks.

This is deliberately a manual procedure. Automatic fitting needs a paired frames-and-events
recording of the same scene to fit against, and the quality of the fit is entirely determined by how
representative that recording is — so it is worth doing against your own data rather than trusting a
number fitted to someone else's.

## Command line

```console
$ eventcv simulate clip.mp4 events.h5
wrote events.h5 (2,481,003 events)
```

`--threshold`, `--sigma-thres`, `--cutoff-hz`, `--leak-rate-hz`, `--shot-noise-rate-hz`,
`--refractory-us`, `--upsample`, `--scale`, `--max-frames` and `--seed` map to the arguments above.
