# Reconstruction

{func}`eventcv.reconstruct` recovers an intensity video from events — the inverse of
[Simulation](simulation.md). It runs [E2VID](https://github.com/uzh-rpg/rpg_e2vid) natively:

```python
import eventcv as ecv

frames = ecv.reconstruct("recording.h5")  # (N, H, W) float32 in [0, 1]
ecv.reconstruct("recording.h5", path="out.mp4")  # or straight to a video
```

With `path`, the output format comes from the extension (`.gif`, `.apng`, `.mp4`) and the call
returns the number of frames written. Frames are rendered at their natural scale rather than
auto-contrasted, so brightness stays consistent across the sequence.

This is distinct from {meth}`~eventcv.EventReader.save_video`, which *renders* events — a false-colour
picture of where activity was. Reconstruction estimates what the scene actually looked like.

## E2VID

{class}`~eventcv.E2VID` is the recurrent network from Rebecq et al., *High Speed and High Dynamic
Range Video with an Event Camera* (T-PAMI 2019), implemented in EventCV's Rust core. Everything
around the network follows the reference implementation's defaults, so on the CPU a frame matches
what rpg_e2vid produces for the same events to within 1e-6 (on the GPU, see below):

- **Windows** hold a fixed number of events — `0.35 × width × height`, about 15 000 on a 240×180
  sensor — rather than a fixed duration. Pass a path, `EventStream` or array and `reconstruct` cuts
  it that way; pass an {class}`~eventcv.EventReader` and its own slicing is used, so
  `ecv.open(path, dt_ms=33)` gives frames at a fixed rate instead.
- **The voxel grid** has 5 bins spanning exactly each window, normalised to zero mean and unit
  standard deviation over its nonzero cells.
- **The output** is sharpened with the reference's unsharp mask.

The network carries a hidden state from one window to the next, which is what lets it build up
detail over time. Construct it yourself to keep that state across calls — on live windows, for
example:

```python
model = ecv.E2VID(device="gpu")
for window in windows:
    image = model(window)  # (H, W) float32 in [0, 1]

model.reset()  # before an unrelated recording
```

`device` is `"cpu"` or `"gpu"` and defaults to {func}`~eventcv.get_device`. On the GPU it shares
the device the rest of EventCV's kernels use.

### Speed and precision

On the GPU the convolutions run in half precision on the tensor cores by default, which is what
makes the GPU worth using. `precision="single"` keeps everything in single precision instead, and
is what a GPU without half-precision support (a software adapter, some older drivers) runs either
way — `model.precision` says which one you got. On
an RTX 2080 and an i7-9700K, a frame — voxel grid, network, sharpening and copy back included —
takes:

| Sensor | GPU, half | GPU, single | CPU |
| --- | --- | --- | --- |
| 346×260 | 9.3 ms | 33 ms | 141 ms |
| 640×480 | 25 ms | 100 ms | 461 ms |
| 1280×720 | 75 ms | 297 ms | 1.5 s |

For comparison, the reference rpg_e2vid in PyTorch, timing the network alone, takes 9.9, 25.7 and
75.7 ms on the same GPU (CUDA, single precision) and 112, 399 and 1302 ms on the same CPU.

Single precision, and the CPU, match rpg_e2vid to ~1e-6. Half precision moves frames by ~1e-3 (well
under one 8-bit grey level), and over long sequences the recurrent state can carry that further in
a few sensitive pixels: on a 300-window sunset recording, 12 of 90 000 pixels were 0.05–0.1 off by
the end, around the saturated sun, while the mean difference stayed at 6e-4. When a frame must match
the reference — a benchmark, a published comparison — use `precision="single"`:

```python
model = ecv.E2VID(device="gpu", precision="single")
frames = ecv.reconstruct("recording.h5", model)
```

### Weights and licence

The weights are not in the wheel. The first `E2VID()` downloads them from the `e2vid/` folder of
[EventLAB/localization](https://huggingface.co/EventLAB/localization/tree/main/e2vid) on the
Hugging Face Hub, checks them against the checksum pinned in this release, and caches them under
`~/.cache/eventcv`. After that no network is needed. Set `EVENTCV_CACHE` to move the cache, `EVENTCV_OFFLINE=1` to forbid
downloading, or pass `E2VID("e2vid.safetensors")` to use a file you already have.

They are the reference `E2VID_lightweight` checkpoint, converted by `scripts/convert_e2vid.py`.
rpg_e2vid is GPL-3.0 and the weights are derived from it, so they are distributed under GPL-3.0,
separately from EventCV (Apache-2.0). If you publish results, cite the paper:

```bibtex
@Article{Rebecq19pami,
  author  = {Henri Rebecq and Ren{\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title   = {High Speed and High Dynamic Range Video with an Event Camera},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell. (T-PAMI)},
  year    = 2019
}
```

## Your own ONNX model

Any ONNX graph that maps a representation to a single-channel image works too, through
{class}`~eventcv.Model`. The reader must carry the representation the model expects:

```python
reader = ecv.open("recording.h5", dt_ms=33).with_repr("voxel", bins=5)
ecv.reconstruct(reader, ecv.Model("my_model.onnx"), "out.mp4")
```

`model.inputs` reports what the graph expects, so read that rather than guessing:

```python
>>> ecv.Model("my_model.onnx").inputs
[{'name': 'voxel', 'shape': [1, 5, 480, 640], 'dtype': 'float32'}]
```

## Recurrent ONNX models

A plain `Model` call deliberately does not carry state, so a recurrent ONNX export needs
{class}`~eventcv.StatefulModel`:

```python
model = ecv.StatefulModel(
    ecv.Model("e2vid_recurrent.onnx"),
    state_map={"new_state": "state"},  # output name -> input name
)

for i in range(len(reader)):
    image = model(reader[i])

model.reset()  # before an unrelated recording
```

`state_map` says which output feeds back into which input. It is explicit rather than inferred from
a naming convention, because getting it wrong silently produces plausible-looking garbage — the
model still runs, it just never remembers anything.

State lives in the wrapper rather than inside `Model`, so `Model` stays a pure function and
resetting is something you ask for rather than something that happens invisibly between recordings.
On the first call, and after `reset()`, each state input is seeded with zeros.

Exporting a recurrent model means exposing its hidden state as explicit inputs and outputs rather
than letting PyTorch hide it inside the module.

## The round trip

Simulation and reconstruction validate each other. Starting from real video, the reconstruction can
be compared against the frames the events were generated from:

```python
events = ecv.simulate("clip.mp4")
ecv.save(events, "synthetic.h5")

ecv.reconstruct("synthetic.h5", path="recovered.mp4")
```

`clip.mp4` and `recovered.mp4` should show the same scene. Where they differ tells you something
specific: blur in the reconstruction usually means the events were too sparse (lower the
thresholds), while structure that is present in the source but missing entirely from the
reconstruction usually means it never generated events at all — a contrast below threshold, or a
region too dark for the photoreceptor bandwidth to track.

## Command line

```console
$ eventcv reconstruct recording.h5 out.mp4 --device gpu --max-frames 300
wrote out.mp4 (300 frames at 30 fps)
$ eventcv reconstruct recording.h5 out.mp4 --dt-ms 33
$ eventcv reconstruct recording.h5 out.mp4 my_model.onnx --dt-ms 33 --bins 5
```

The model defaults to E2VID. An ONNX file covers the stateless case; recurrent ONNX models need
`StatefulModel`, so use the Python API for those.
