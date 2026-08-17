# Reconstruction

{func}`eventcv.reconstruct` recovers an intensity video from events by running a trained model over
each slice — the inverse of [Simulation](simulation.md).

```python
import eventcv as ecv

reader = ecv.open("recording.h5", dt_ms=33).with_repr("voxel", bins=5)
ecv.reconstruct(reader, ecv.Model("e2vid.onnx"), "out.mp4")
```

The reader must carry the representation the model expects, and the output format comes from the
extension (`.gif`, `.apng`, `.mp4`). Reconstructed frames are rendered at their natural scale rather
than auto-contrasted, so brightness stays consistent across the sequence.

This is distinct from {meth}`~eventcv.EventReader.save_video`, which *renders* events — a false-colour
picture of where activity was. Reconstruction estimates what the scene actually looked like.

## You supply the model

EventCV ships no weights. `Model` runs any ONNX graph mapping a representation to a single-channel
image; which network, and what it was trained on, is your decision — the same position taken in
[Models](models.md).

There is **no official ONNX export of E2VID**. The reference implementation
([rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid)) is PyTorch, so export it yourself:

```python
torch.onnx.export(model, example_inputs, "e2vid.onnx",
                  input_names=["voxel"], output_names=["image"])
```

The number of voxel bins the export expects is what `with_repr("voxel", bins=…)` has to match.
`model.inputs` reports it, so read that rather than guessing:

```python
>>> ecv.Model("e2vid.onnx").inputs
[{'name': 'voxel', 'shape': [1, 5, 480, 640], 'dtype': 'float32'}]
```

## Recurrent models

E2VID is recurrent: it carries a hidden state between frames, which is what lets it accumulate
detail over time instead of reconstructing each frame from scratch. A plain `Model` call deliberately
does not carry state, so a recurrent export needs {class}`~eventcv.StatefulModel`:

```python
model = ecv.StatefulModel(
    ecv.Model("e2vid_recurrent.onnx"),
    state_map={"new_state": "state"},     # output name -> input name
)

for i in range(len(reader)):
    image = model(reader[i])

model.reset()      # before an unrelated recording
```

`state_map` says which output feeds back into which input. It is explicit rather than inferred from
a naming convention, because getting it wrong silently produces plausible-looking garbage — the
model still runs, it just never remembers anything.

State lives in the wrapper rather than inside `Model`, so `Model` stays a pure function and
resetting is something you ask for rather than something that happens invisibly between recordings.
On the first call, and after `reset()`, each state input is seeded with zeros.

Exporting a recurrent model means exposing its hidden state as explicit inputs and outputs rather
than letting PyTorch hide it inside the module — the ConvLSTM states in E2VID's case. A stateless
export (the reference implementation's `--no-recurrent` mode) needs none of this and works with
`Model` directly, at some cost in reconstruction quality.

## The round trip

Simulation and reconstruction validate each other. Starting from real video, the reconstruction can
be compared against the frames the events were generated from:

```python
events = ecv.simulate("clip.mp4")
ecv.save(events, "synthetic.h5")

reader = ecv.open("synthetic.h5", dt_ms=33).with_repr("voxel", bins=5)
ecv.reconstruct(reader, ecv.Model("e2vid.onnx"), "recovered.mp4")
```

`clip.mp4` and `recovered.mp4` should show the same scene. Where they differ tells you something
specific: blur in the reconstruction usually means the events were too sparse (lower the
thresholds), while structure that is present in the source but missing entirely from the
reconstruction usually means it never generated events at all — a contrast below threshold, or a
region too dark for the photoreceptor bandwidth to track.

## Command line

```console
$ eventcv reconstruct recording.h5 out.mp4 e2vid.onnx --dt-ms 33 --bins 5
wrote out.mp4 (30 frames at 30 fps)
```

The CLI covers the stateless case. Recurrent models need `StatefulModel`, so use the Python API.
