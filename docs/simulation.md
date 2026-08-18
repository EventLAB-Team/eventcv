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

At any real resolution you want `out=` instead, which writes the events as they are produced rather
than accumulating them (see [Writing straight to a file](#writing-straight-to-a-file)):

```python
result = ecv.simulate("clip.mp4", out="events.h5", progress=True)
reader = ecv.open(result.path, dt_ms=20)
```

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

Every sub-step is a full pass over every pixel, and the **busiest** pixel sets the factor — so one
hard edge can push an otherwise quiet 1080p frame to the maximum, 64 passes over two million pixels
for that pair. `max_upsample` caps it:

```python
ecv.simulate("clip.mp4", max_upsample=8)        # bound the worst case
```

Lowering it trades timestamp accuracy on the fastest content for time, and does nothing at all to
frames that were not asking for that many sub-steps.

## Writing straight to a file

A realistic sensor emits a *lot*. A 1.6-second 1080p clip at the defaults produces about **167
million events** — roughly 5 GB as an in-memory stream, before you have saved any of it. `out=`
writes each frame interval as it is produced, so memory stays flat at one interval's worth however
long the clip is:

```python
result = ecv.simulate("clip.mp4", out="events.h5", progress=True)
result.frames, result.events, result.path
len(result)                    # the event count
```

With `out=` the return value is a `SimulationResult` rather than an `EventStream`; open the file to
work with the events. `.h5` is written incrementally; other formats are buffered and written once at
the end, so they still cost the memory `out=` exists to avoid.

`compression` picks the HDF5 filter — gzip 1 by default, `False` for none. Events compress far
better than their 13 raw bytes suggests, because `t` is monotonic and `p` is one bit in a whole
byte; on the clip above the default brings 2.2 GB down to about 410 MB.

`progress=True` prints frames and running event counts to stderr, and `Ctrl+C` stops a run and
raises `KeyboardInterrupt` with the file keeping everything written so far.

## Reproducibility and cost

`seed` makes a run a pure function of its configuration — the same seed gives byte-identical events,
so a training run is reproducible from its config alone. That holds across machines as well as
across runs: the simulation uses every core, but each block of pixels draws its noise from a stream
seeded from the configuration rather than from the thread that happens to run it, so the result does
not depend on how many cores there are.

Simulation streams: only the current frame pair is held, so a long clip costs memory proportional to
one frame interval rather than to the whole output.

**The event count is the cost.** Time and disk both scale with it, and the most effective lever on
it is resolution: `scale=(w, h)` decodes at a smaller size, which is both much cheaper than decoding
full-size and downsampling afterwards, and quarters the output for each halving of the sides.
`max_frames` stops early.

```python
ecv.simulate("clip.mp4", scale=(960, 540), out="events.h5")   # ~4x less of everything
```

Note that the noise floor is charged per pixel per second regardless of what is happening: at the
default rates, a 1080p sensor emits about 23 M events a second before the scene contributes
anything. Setting `shot_noise_rate_hz=0, leak_rate_hz=0` removes that, at the cost of the realism it
buys.

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
simulating: 39/39 frames, 167.3M events
wrote events.h5 (167,275,281 events from 39 frames)
```

The CLI always streams to the output file, and shows progress when stderr is a terminal (`-q`
silences both that and the summary).

`--threshold`, `--sigma-thres`, `--cutoff-hz`, `--leak-rate-hz`, `--shot-noise-rate-hz`,
`--refractory-us`, `--upsample`, `--max-upsample`, `--compression`, `--scale`, `--max-frames` and
`--seed` map to the arguments above.

Then look at what you made, without picking a representation first:

```console
$ eventcv play events.h5 --dt-ms 5
$ eventcv render events.h5 raw.mp4 --dt-ms 5 --fps 20
```

## Simulating on a GPU

The pixel model is the heaviest loop in the library — every sub-step is a full pass over every
pixel doing real arithmetic — which makes it the one place a GPU clearly pays. `device="gpu"` runs
it as a compute shader:

```python
events = ecv.simulate("clip.mp4", device="gpu")
```

Measured on an RTX 2080 against a 12-core CPU (release build, a moving edge, default sensor model):

| sensor | frames | CPU | GPU | |
|---|---|---|---|---|
| 346×260 | 24 | 479 ms | 219 ms | 2.2× |
| 640×480 | 24 | 1352 ms | 620 ms | 2.2× |
| 1280×720 | 16 | 2959 ms | 1488 ms | 2.0× |
| 1920×1080 | 12 | 5314 ms | 2940 ms | 1.8× |

The backend is `wgpu`, so this is Metal on macOS and Vulkan on Linux with the same shader; there is
no CUDA toolkit involved. `ecv.gpu_available()` says whether it can run here, and asking for a GPU
that is not there raises rather than falling back silently.

### The two backends are not interchangeable to the last event

Worth knowing before you compare a GPU run against a stored CPU one:

| configuration | how they compare |
|---|---|
| noise and leak off | **identical**, event for event |
| `sigma_thres` on, still noiseless | same events, same polarities; a couple of timestamps in ~25 000 land a microsecond apart |
| noise on | same event *rate* to within a few per cent, **different events** |

Threshold mismatch is drawn once on the host and uploaded, so a seed describes the same silicon on
either backend — the random part of the *model* is not the part the GPU generates. The microsecond
discrepancy is the crossing fraction, which is double precision on the CPU and cannot be in a
shader. Shot noise and leak genuinely differ: a kernel with one invocation per pixel cannot replay
a generator that the CPU consumes sequentially, so the GPU uses a counter-based one keyed on
`(seed, frame, sub-step, pixel, polarity)`. That is reproducible from run to run and across
devices, but it is a different sample path.

So compare against a stored CPU result with `shot_noise_rate_hz=0, leak_rate_hz=0`, or compare
distributions.

## Learned frame interpolation

`upsample` subdivides the interval between two frames, but the levels it subdivides are a
**straight-line blend** of them. That is right when a pixel's intensity ramps over the frame gap and
wrong when an edge crosses it and the pixel *steps* — and when it is wrong, every event from that
pixel is placed at the wrong moment, however finely the interval was subdivided. v2e reaches for
Super-SloMo here. `interpolate` reaches for whatever ONNX graph you have:

```python
events = ecv.simulate("clip.mp4", interpolate="rife_v4.onnx", interpolate_factor=4)
```

`interpolate_factor` is how many intervals each source pair becomes, so `4` inserts three frames.
Those frames are fed to the simulator as **ordinary source frames at proportional timestamps** —
the same arrangement v2e uses, and the reason the pixel model itself needs no changes. `upsample`
then refines whatever is left.

**No weights ship with eventcv.** It runs graphs; it does not carry a zoo. Export RIFE (or anything
shaped like it) yourself. What the graph has to look like:

- a frame pair in, either as one six-channel input or as two three-channel ones;
- one image out;
- ideally a scalar `timestep` input (RIFE v4 and later), which is what allows an arbitrary fraction
  in a single pass.

Without a `timestep` the network can only produce midpoints, so eventcv bisects towards the fraction
it needs — which reaches `k/2ⁿ` and refuses anything else rather than quietly returning a midpoint.
Frames are interpolated in luma, since luma is all the simulator consumes.
