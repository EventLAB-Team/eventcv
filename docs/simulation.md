# Simulation

{func}`~eventcv.simulate` turns a video, or an array of frames, into the events an event camera would
have recorded.

```python
import eventcv as ecv

events = ecv.simulate("clip.mp4")  # needs ffmpeg on your PATH
events = ecv.simulate(frames, fps=1000)  # from an (N, H, W) uint8 array
```

The result is an ordinary {class}`~eventcv.EventStream`.

## Write to a file

Simulated video produces a lot of events: a 1.6 s 1080p clip gives about 167 million. Pass `out=` to
write them as they are generated, so memory stays flat:

```python
result = ecv.simulate("clip.mp4", out="events.h5", progress=True)
reader = ecv.open(result.path, dt_ms=20)
```

With `out=`, the return value has `.path`, `.frames` and `.events` instead of the events themselves.
Use `.h5`: other formats are buffered in memory and written at the end. `compression=` sets the gzip
level (default `1`, `False` for none). `Ctrl+C` stops the run and keeps what was written.

## Sensor model

The model follows [v2e](https://arxiv.org/abs/2006.07722) (Hu et al., CVPRW 2021).

| Parameter | Default | Models |
|---|---|---|
| `pos_thres`, `neg_thres` | `0.2` | Log-contrast change that triggers an event. |
| `sigma_thres` | `0.03` | Pixel-to-pixel variation in those thresholds. |
| `cutoff_hz` | `200` | Photoreceptor bandwidth; dark pixels respond more slowly. |
| `leak_rate_hz` | `1.0` | Spontaneous ON events with no scene change. |
| `shot_noise_rate_hz` | `10.0` | Random noise events, strongest in dark pixels. |
| `refractory_us` | `100` | Dead time after each event. |
| `seed` | `0` | Seed for threshold variation and noise. The same seed gives identical output on any machine. |

Frames are converted from sRGB to linear light before taking the log, and timestamps are placed
within each frame interval rather than at the frame time.

For an ideal, noiseless sensor, turn the non-idealities off:

```python
events = ecv.simulate(
    "clip.mp4",
    sigma_thres=0,
    cutoff_hz=0,
    leak_rate_hz=0,
    shot_noise_rate_hz=0,
    refractory_us=0,
)
```

## Speed and size

Output size, and with it run time, scales with resolution. Decoding at a smaller size is the most
effective control:

```python
ecv.simulate("clip.mp4", scale=(960, 540), out="events.h5")  # about 4× smaller
```

| Option | Default | Meaning |
|---|---|---|
| `scale` | native | Decode at `(width, height)`. |
| `max_frames` | all | Stop after this many frames. |
| `upsample` | `"adaptive"` | Subdivide each frame interval: `"adaptive"`, `"off"`, or a fixed factor like `"8"`. |
| `max_events_per_pixel` | `1.0` | For `"adaptive"`: subdivide until no pixel fires more than this per step. |
| `max_upsample` | `64` | Cap on subdivisions per frame pair. |
| `device` | session default | `"gpu"` is about 2× faster. See {doc}`gpu`. |

Noise costs events regardless of the scene: at the defaults, a 1080p sensor produces about 23 million
noise events per second. Set `shot_noise_rate_hz=0, leak_rate_hz=0` to remove them.

## Frame interpolation

`upsample` interpolates linearly between frames, which places events wrongly when an edge jumps
across a pixel. For fast motion, insert frames with a learned interpolation network in ONNX format,
such as RIFE:

```python
events = ecv.simulate("clip.mp4", interpolate="rife_v4.onnx", interpolate_factor=4)
```

`interpolate_factor=4` inserts three frames between each pair. No weights ship with EventCV; export
the model yourself. The graph must take a frame pair (one 6-channel input or two 3-channel inputs) and
return one image. A scalar `timestep` input (RIFE v4+) allows any factor; without one, factors must be
powers of two.

## Match a real camera

The defaults are realistic but not specific to your camera. Record a scene, simulate the same scene,
and compare their event rates:

```python
real = ecv.load("recorded.h5").event_rate(bin_ms=10)
fake = ecv.simulate("same_scene.mp4", pos_thres=0.25).event_rate(bin_ms=10)
```

Match the quiet periods first with `leak_rate_hz` and `shot_noise_rate_hz`, then the peaks with the
thresholds: raise them if the simulation is too busy, lower them if it's too quiet.

## Command line

```console
eventcv simulate clip.mp4 events.h5 --scale 960 540
```

See {doc}`cli` for the flags.
