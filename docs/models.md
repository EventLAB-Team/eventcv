# Models

{class}`~eventcv.Model` runs a trained ONNX network on a representation.

```python
import eventcv as ecv

model = ecv.Model("detector.onnx")
reader = ecv.open("recording.h5", dt_ms=30).with_repr("voxel", bins=5)

for i in range(reader.n_slices):
    prediction = model(reader[i])
```

ONNX Runtime is bundled with the Python package, so nothing else needs installing. To get an `.onnx`
file, export a PyTorch model with `torch.onnx.export`.

## Inputs and outputs

Check what a graph expects before choosing a representation:

```python
model.inputs  # [{'name': 'input', 'shape': [1, 5, 480, 640], 'dtype': 'float32'}]
model.outputs  # [{'name': 'output', 'shape': [1, 100, 5], 'dtype': 'float32'}]
```

Here the `5` means `with_repr("voxel", bins=5)`. `-1` marks a free dimension, usually the batch.

`model(data)` takes an {class}`~eventcv.EventFrame` or a NumPy array:

- A missing batch axis is added, so a `(C, H, W)` frame works with an `(N, C, H, W)` graph.
- Integer input, such as a count image, is converted to `float32`.
- A graph with one output returns an array; with several, a list in the order of `model.outputs`.
- Only `float32` outputs are supported.

## Performance

Create the `Model` once and reuse it: loading is far slower than running. Inference releases the GIL.
Calls on one `Model` run one at a time, so create one per thread for parallel inference.

Inference runs on the CPU. For GPU inference, run the network in PyTorch or `onnxruntime-gpu` and use
EventCV to produce the tensors; see {doc}`large-files` for the `DataLoader` setup.

## Which ONNX Runtime is used

EventCV looks for ONNX Runtime in this order, and `eventcv --version` shows which it found:

| Order | Source |
|---|---|
| 1 | The library named by `ORT_DYLIB_PATH`. |
| 2 | An `onnxruntime` already imported in the process. |
| 3 | The copy bundled with EventCV. |
| 4 | The environment's `lib` directory, e.g. conda's `onnxruntime-cpp`. |
| 5 | An installed `onnxruntime` package. |

Any version from 1.17 works. To use a specific build:

```console
export ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
```

:::{dropdown} `Model` raises a `RuntimeError`
- **"EventCV loads ONNX Runtime at run time…"**: no usable runtime was found. Reinstall EventCV,
  install one (`pip install eventcv[onnx]` or conda's `onnxruntime-cpp`), or set `ORT_DYLIB_PATH`.
- **"eventcv was built without ONNX support"**: a source build left the feature out. Rebuild with
  `maturin develop --features onnx`.
:::
