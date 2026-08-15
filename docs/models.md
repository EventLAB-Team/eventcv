# Models

{class}`~eventcv.Model` runs a trained ONNX network over a representation.

```python
import eventcv as ecv

model = ecv.Model("detector.onnx")
reader = ecv.open("recording.h5", dt_ms=30, repr="voxel", bins=5)

for i in range(len(reader)):
    prediction = model(reader[i])
```

The ONNX Runtime is compiled into the published wheels, so there is nothing to install. Check with
`eventcv --version`, which lists the features a build has.

## Scope

EventCV ships **no architectures and no weights**. `Model` takes any `.onnx` file and knows nothing
about what is inside it.

That is a deliberate boundary rather than a missing feature. Bundling models would mean tracking
every upstream project's preprocessing, weights hosting and version skew indefinitely — and the
part that actually belongs in an event-vision library is building the tensor correctly. Which is
also where the pairing pays off: a voxel grid or time surface is already a dense `float32` array in
the layout ONNX expects, so feeding one to a network is a shape check, not a conversion.

To get an `.onnx` file: export from PyTorch with `torch.onnx.export`, or convert an existing
checkpoint with the upstream project's own export script.

## Inspecting a graph

A model that will not run is nearly always a shape mismatch, so `Model` reports what the graph
declares:

```python
>>> model.inputs
[{'name': 'input', 'shape': [1, 5, 480, 640], 'dtype': 'float32'}]
>>> model.outputs
[{'name': 'output', 'shape': [1, 100, 5], 'dtype': 'float32'}]
```

`-1` marks a dimension the graph leaves free — usually the batch axis. Reading this off first is
faster than guessing at the representation parameters that produce a matching tensor: the `5` above
is what tells you to open with `repr="voxel", bins=5`.

## Inputs and outputs

`model(data)` accepts an {class}`~eventcv.EventFrame` or any NumPy array convertible to `float32`.
Two conveniences:

- **The batch axis is added for you.** A representation is `[C, H, W]` while nearly every trained
  network wants `[N, C, H, W]`, so a missing leading axis is inserted rather than making every call
  site write `arr[None]`. Only ever added, and only when the ranks differ by exactly one.
- **Non-float input is coerced.** A count image is integer; it is cast rather than rejected.

A single-output graph returns one array. A multi-output graph returns a list, in the same order as
`model.outputs`.

Only `float32` outputs are supported. A graph emitting `int64` class indices raises rather than
silently casting — a dtype mismatch is nearly always a sign the export did something unintended,
and hiding it would cost more than it saves.

## Performance

Load once, call many times — `Model.__init__` reads and optimises the graph, which is far more
expensive than a single inference. Keep the object alive across a loop rather than constructing it
per slice.

Inference releases the GIL, so it overlaps with other Python threads. Calls are serialised through a
lock because ONNX Runtime's session internals are not thread-safe; to run inference in parallel,
construct one `Model` per thread.

There is no GPU execution provider wired up: the runtime is built for CPU. For GPU inference, run
the model in PyTorch or `onnxruntime-gpu` and use EventCV for the tensors — {func}`eventcv.collate`
and the map-style {class}`~eventcv.EventReader` are built for exactly that handoff. See
[Quickstart](quickstart.md) for the `DataLoader` path.

## If `Model` is unavailable

A source build without the feature raises a `RuntimeError` naming the fix. Rebuild with:

```console
$ maturin develop --features onnx
```

The feature exists as a flag because `ort`, the ONNX Runtime binding, has no stable release yet —
keeping it switchable means a bad release candidate is a build-flag change rather than a break in
EventCV's own API.
