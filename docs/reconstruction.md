# Reconstruction

{func}`~eventcv.reconstruct` recovers an intensity video from events, using a built-in
[E2VID](https://github.com/uzh-rpg/rpg_e2vid).

```python
import eventcv as ecv

frames = ecv.reconstruct("recording.h5")  # (N, H, W) float32 in [0, 1]
ecv.reconstruct("recording.h5", path="out.mp4")  # straight to a video
```

`source` is a path, stream, array or reader. With `path`, the format comes from the extension
(`.gif`, `.apng`, `.mp4`) and the call returns the number of frames.

| Option | Default | Meaning |
|---|---|---|
| `model` | `"e2vid"` | `"e2vid"`, an {class}`~eventcv.E2VID`, a {class}`~eventcv.Model` or a {class}`~eventcv.StatefulModel`. |
| `device` | session default | `"cpu"` or `"gpu"`. |
| `fps` | `30` | Playback rate of the written video. |
| `max_frames` | all | Stop early. |

By default each frame covers a fixed number of events, as in the original E2VID. To get frames at a
fixed rate instead, pass a reader: `ecv.reconstruct(ecv.open(path, dt_ms=33))`.

## E2VID

{class}`~eventcv.E2VID` keeps a hidden state between windows, so detail builds up over time. Create it
yourself to process windows one at a time, for example from a live camera:

```python
model = ecv.E2VID(device="gpu")

with ecv.stream(dt_ms=33) as cam:
    for window in cam:
        image = model(window)  # (H, W) float32 in [0, 1]

model.reset()  # before starting an unrelated recording
```

| Option | Default | Meaning |
|---|---|---|
| `device` | session default | `"cpu"` or `"gpu"`. |
| `precision` | `"half"` | GPU only. `"single"` matches the reference implementation exactly; `"half"` is about 4× faster. |
| `weights` | downloaded | Path to a `.safetensors` file to use instead. |

On the CPU, and on the GPU with `precision="single"`, output matches the reference rpg_e2vid to within
1e-6. See {doc}`gpu` for speed.

### Weights

The weights aren't bundled. The first `E2VID()` downloads them from the
[Hugging Face Hub](https://huggingface.co/EventLAB/localization/tree/main/e2vid), verifies the
checksum and caches them in `~/.cache/eventcv`.

| Variable | Effect |
|---|---|
| `EVENTCV_CACHE` | Use a different cache directory. |
| `EVENTCV_OFFLINE=1` | Never download. |

:::{note}
The weights are derived from rpg_e2vid and licensed under GPL-3.0, separately from EventCV
(Apache-2.0). If you publish results, cite Rebecq et al., *High Speed and High Dynamic Range Video
with an Event Camera*, T-PAMI 2019.
:::

## Your own model

Any ONNX network that turns a representation into a single-channel image works through
{class}`~eventcv.Model`. Give it a reader that produces the representation the model expects:

```python
reader = ecv.open("recording.h5", dt_ms=33).with_repr("voxel", bins=5)
ecv.reconstruct(reader, ecv.Model("my_model.onnx"), "out.mp4")
```

For a recurrent network, wrap it in {class}`~eventcv.StatefulModel` and say which output feeds back
into which input:

```python
model = ecv.StatefulModel(ecv.Model("recurrent.onnx"), state_map={"new_state": "state"})

for i in range(reader.n_slices):
    image = model(reader[i])

model.reset()
```

State inputs start at zero, and again after `reset()`.

## Command line

```console
eventcv reconstruct recording.h5 out.mp4 --device gpu
eventcv reconstruct recording.h5 out.mp4 my_model.onnx --dt-ms 33 --bins 5
```

Recurrent ONNX models need the Python API.
