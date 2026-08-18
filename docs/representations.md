# Representations

A **representation** turns a sparse {class}`~eventcv.EventStream` into a dense tensor
(most are `[C, H, W]`, ready for NumPy / PyTorch) or, for {func}`~eventcv.pset`, a sparse
point array. Most are reachable three ways:

```python
stream.voxel(bins=5)                       # method
ecv.voxel(stream, bins=5)                  # free function (OpenCV-style)
stream.flatten("voxel")                    # by name (defaults; see "By name" below)
```

The **method** and **free-function** forms take the representation's parameters directly. The
**name** form (a string like `"voxel"`) selects a representation for
{meth}`~eventcv.EventStream.flatten`, {meth}`~eventcv.EventStream.view`, `open(..., repr=...)`,
and `reader.with_repr(...)` — but only `reader.with_repr(name, **opts)` also forwards the
representation's parameters; the other three use its defaults. A few representations don't fit
all three forms; each is flagged in its section below.

## Summary

| Name       | Method            | Channels | Dtype             | What it encodes |
|------------|-------------------|----------|--------------------|-----------------|
| `polarity` | *(none — default view, or by name)* | 2   | `uint64`/`uint8`   | Per-pixel event count, split into positive/negative planes. The default view when no representation is requested. |
| `binary`   | *(none — `flatten(binary=True)`, or by name)* | 1 | `uint8`         | Pixel fired at all (`1`) or not (`0`), both polarities merged. |
| `count`    | {func}`~eventcv.count`   | 1  | `uint64`/`uint8`   | Total events per pixel, both polarities summed. |
| `voxel`    | {func}`~eventcv.voxel`   | `bins` | `float32`      | Signed, time-interpolated polarity across `bins` temporal bins over the last `window_ms`. |
| `tsurf`    | {func}`~eventcv.tsurf`   | 2  | `float32`          | Time surface: `exp(-age/tau_ms)` of the *latest* event per pixel/polarity. |
| `atsurf`   | {func}`~eventcv.atsurf`  | 2  | `float32`          | Averaged (HATS-style) time surface: mean of `exp(-age/tau_ms)` over *all* events per pixel/polarity. |
| `tencode`  | {func}`~eventcv.tencode` | 3  | `uint8`            | Latest polarity + normalized age within `window_ms`, as an RGB-like image. |
| `countmask`| {func}`~eventcv.countmask` | 3 | `uint8`           | Positive/negative event counts in red/blue, jointly normalized by a percentile of the non-zero counts, plus a binary activity mask in green. Timestamps unused. |
| `mcts`     | {func}`~eventcv.mcts`    | 10 | `float32`          | Multi-channel time surface: 5 log-spaced windows up to `max_window_ms`, per polarity. |
| `flow`     | {func}`~eventcv.optical_flow` | 2 | `float32`     | Dense Lucas-Kanade optical flow `(flow_x, flow_y)` on the time surface, pixels/ms. |
| `pset`     | {func}`~eventcv.pset`    | — (`[N, 4]`) | `float32`    | Sparse, normalized `(x, y, t, p)` point array — not a dense frame. |
| `labels`   | {meth}`~eventcv.EventFrame.connected_components` | 1 | `uint64`   | Connected-component labels of an existing frame (post-processing, not stream-level). |

Two rows are special cases rather than string-selectable representations: `pset` returns a
sparse point array (not a dense frame), and `labels` is computed from an existing frame — so
neither is a valid `repr=` / `flatten` / `view` name. See their sections below.

The dtype column lists `uint64`/`uint8` for the two representations with a `normalize` option
(`count` and `polarity`): raw `uint64` counts, or `uint8` rescaled so the busiest pixel maps to
255 when `normalize=True`. The default differs by representation: `polarity` normalizes by
default (`True`), while `count` returns raw counts by default (`False`) so the frame preserves
exact event totals. The rest have no normalization option. `view` always auto-contrasts for
display regardless of the frame's dtype.

## Polarity

```python
stream.view()                  # implicit — polarity is the default when nothing is stored/named
stream.flatten("polarity")
ecv.flatten(stream, "polarity", normalize=False)
```

Two channels, `[positive, negative]`: raw event counts per pixel per polarity (`uint64`), or
`uint8` rescaled to the busiest pixel when `normalize=True` (the default). This is the frame
rendered whenever no representation is requested or stored on the stream — there is no
dedicated `.polarity()` method, only the name form and the implicit default. (A reader opened
with `open(repr=...)` renders that representation on `slice()` directly instead of falling back
to polarity.)

## Binary

```python
stream.flatten(binary=True)    # dedicated flag
stream.flatten("binary")       # equivalent, by name
```

Single channel, `uint8`, `1` where a pixel fired at all (either polarity) and `0` elsewhere.
No parameters, and — like `polarity` — no dedicated method; only reachable via
`flatten(binary=True)` or the `"binary"` name.

## Count

```python
stream.count()                    # raw uint64 counts (default)
ecv.count(stream, normalize=True) # uint8, busiest pixel → 255
```

- `normalize` (`bool`, default `False`) — when `True`, rescale to `uint8` (busiest pixel → 255)
  instead of the default raw `uint64` counts.

Single channel: total events per pixel, both polarities summed. The plainest "how much
happened here" frame — unlike `polarity`, it collapses polarity into one intensity map. The raw
`uint64` default keeps exact totals (e.g. `count().numpy().sum()` equals the event count), which
is what you want for analysis; pass `normalize=True` for a display-scaled image.

## Voxel grid

```python
stream.voxel(bins=9, window_ms=30.0)
ecv.voxel(stream, bins=5, window_ms=50.0)
```

- `bins` (`int`, default `9`) — number of temporal bins; must be at least 1.
- `window_ms` (`float`, default `30.0`) — trailing time window before the stream's last event.

`bins` channels, `float32`, signed. Each event contributes `+1`/`-1` (by polarity), linearly
interpolated between its two nearest bins by how far into `window_ms` it falls — events outside
the window are dropped.

## Time surface (`tsurf`)

```python
stream.tsurf(tau_ms=30.0)
ecv.tsurf(stream, tau_ms=10.0)
```

- `tau_ms` (`float`, default `30.0`) — exponential decay constant; must be finite and positive.

Two channels, `[positive, negative]`, `float32`. Each pixel/polarity holds
`exp(-age_ms / tau_ms)` for its **most recent** event only (`0` if that polarity never fired
there) — a snapshot of "how recently did this fire."

## Averaged time surface (`atsurf`)

```python
stream.atsurf(tau_ms=30.0)
ecv.atsurf(stream, tau_ms=30.0)
```

- `tau_ms` (`float`, default `30.0`) — exponential decay constant; must be finite and positive.

Two channels, `[positive, negative]`, `float32`. Same exponential response as `tsurf`, but
**averaged over every event** that hit the pixel (HATS-style) rather than keeping only the
latest — recurring activity reads brighter than a single stale hit.

## Tencode

```python
stream.tencode(window_ms=30.0)
ecv.tencode(stream, window_ms=30.0)
```

- `window_ms` (`float`, default `30.0`) — trailing time window; must be finite and positive.

Three channels, `[positive, age, negative]`, `uint8`. Per pixel: whichever polarity fired most
recently sets its channel to `255` (the other stays `0`), and the `age` channel holds that
event's age rescaled to `0-255` across `window_ms`. Reads like an RGB image (red/green-style
polarity planes plus a brightness-coded age plane).

## Count mask

```python
stream.countmask(pct=99.0)
ecv.countmask(stream, pct=99.0, white_frame=False)
```

- `pct` (`float`, default `99.0`) — normalization percentile; must be between `0` and `100`.
- `white_frame` (`bool`, default `False`) — invert to a white background.

Three channels, `[positive, activity, negative]`, `uint8` — the event-accumulation image of the
GEPT paper (Sec. 3.2, Eq. 2). Red holds the per-pixel count of positive events and blue the count
of negative ones; green is a binary mask, `255` wherever an event of either polarity landed and
`0` elsewhere. **Timestamps are not used at all**, which is what separates it from `tencode`
(latest polarity plus event age) and from `count` (one plane of raw totals).

Both count planes are clipped and rescaled by a **single** scale: the `pct`-th percentile of the
non-zero counts of the two planes *pooled together*. Pooling matters — a per-channel percentile
would let a quiet polarity saturate against a busy one, so the joint scale is what keeps red and
blue comparable. Counts above the percentile clamp to full scale.

Expect a dark red/blue image over a bright green mask; on typical data the green plane averages
several times brighter than the other two. `white_frame=True` inverts the whole image, background
included, and exists for parity with the reference renderer — descriptor models trained on these
frames expect the black-background default.

## MCTS (multi-channel time surface)

```python
stream.mcts(max_window_ms=30.0)
ecv.mcts(stream, max_window_ms=16.0)
```

- `max_window_ms` (`float`, default `30.0`) — largest window; must be at least `1.0`.

Ten channels, `float32`: 5 log-spaced windows (from `1 ms` up to `max_window_ms`) for each of
`negative`/`positive` (channel names like `negative_1.000ms` … `positive_30.000ms`). Each
channel holds `max(1 - age/window)` over events within that window — a multi-timescale
generalization of `tsurf`.

## Optical flow

```python
stream.optical_flow(window=3)
ecv.optical_flow(stream, window=5)
stream.view("flow")             # by name; also valid for open(repr="flow")
```

- `window` (`int`, default `3`) — half-width of the least-squares neighbourhood (a
  `(2·window+1)²` patch); must be at least 1.

Two channels, `[flow_x, flow_y]`, `float32`, in pixels/ms. Dense Lucas-Kanade on the time
surface (the Surface of Active Events): fits one velocity per pixel by least squares, falling
back to normal flow along the gradient in the aperture-problem case, and leaving flat,
event-free windows at zero. Flow is emitted only at pixels that saw an event, and assumes the
stream is sorted ascending by time.

## Sparse point set

```python
points = stream.pset()
ecv.pset(stream).numpy()        # [N, 4] float32 array: columns (x, y, t, p)
```

No parameters. Unlike the frame representations above, `pset` returns an
{class}`~eventcv.EventPointSet`, not an {class}`~eventcv.EventFrame`: an `[N, 4]` array with
`x`/`y` normalized to `[0, 1]` by sensor size, `t` normalized to `[0, 1]` over the stream's
span, and `p` as `+1`/`-1`. Because it's sparse rather than a dense `[C, H, W]` tensor, it
isn't a valid choice for `open(repr=...)` / `reader.with_repr(...)` (those require a dense
frame per slice).

## Connected-component labels

```python
frame = stream.count()
labels = frame.connected_components(connectivity=8)
```

- `connectivity` (`int`, default `8`) — `4` or `8`-connectivity for the flood fill.

Single channel, `uint64`. This is a **post-processing** step on an existing
{class}`~eventcv.EventFrame` (typically a `binary` or `count` image), not a representation
generated directly from a stream: any non-zero pixel (across all channels) is foreground, and
each maximal connected blob gets a distinct label `1..k`, with background `0`.

## By name

A representation can be picked by string anywhere a name is accepted:
{meth}`~eventcv.EventStream.flatten`, {meth}`~eventcv.EventStream.view`, `open(..., repr=...)`,
and `reader.with_repr(...)`. The name is one of `"polarity"`, `"binary"`, `"count"`,
`"countmask"`, `"voxel"`, `"tsurf"`, `"atsurf"`, `"tencode"`, `"mcts"`, or `"flow"`.

`flatten`, `view`, and `open(repr=...)` render the representation with its **default**
parameters — they don't accept a representation's own parameters (`bins`, `tau_ms`, `window_ms`,
`max_window_ms`, `window`, `pct`, `white_frame`), so `stream.flatten("voxel", bins=5)` is a
`TypeError`. The one
exception is `normalize`, which is an argument of `flatten`/`view` themselves, so
`stream.flatten("count", normalize=True)` does work (and each representation keeps its own
`normalize` default when the argument is omitted — `False` for `count`, `True` for `polarity`).
To override the other parameters by name, use `reader.with_repr(name, **opts)`, which returns a
new dataset reader:

```python
# defaults, by name:
stream.flatten("voxel")                                  # == stream.voxel()  (bins=9)
reader = ecv.open("huge.hdf5", dt_ms=30, repr="mcts")    # mcts, default max_window_ms=30

# parameters by name — only with_repr forwards them:
reader = ecv.open("huge.hdf5", dt_ms=30).with_repr("voxel", bins=5)
array = reader[0]                                        # [5, H, W] voxel for frame 0
frame = reader.slice(7)                                  # EventFrame — voxel(bins=5) applied
reader = ecv.open("huge.hdf5", dt_ms=30).with_repr("countmask", pct=95.0)

# a repr-less reader yields a stream; name the representation on the slice:
frame = ecv.open("huge.hdf5", dt_ms=30).slice(7).flatten("mcts")   # → EventFrame
```

`"pset"` and `"labels"` are not valid names here — `pset` is sparse (see above), and `labels`
is generated from a frame, not a stream.

## Running on a GPU

Every representation runs on the CPU by default, and those implementations are the reference: the
GPU kernels are checked against them, not the other way round. `device="gpu"` moves one call,
`eventcv.set_device("gpu")` moves the session, and `EVENTCV_DEVICE=gpu` moves a whole process
without touching any code.

```python
frame = stream.voxel(bins=5, device="gpu")

ecv.set_device("gpu")        # everything from here on, including reader slices
print(ecv.get_device())      # "gpu"
print(ecv.gpu_available())   # False on a build or a machine without one
```

The backend is `wgpu`, so the same shaders run on **Metal** (macOS), **Vulkan** (Linux) and
**DX12** (Windows) — there is no CUDA path to install and no separate Apple path to maintain.
Asking for a GPU that is not there raises rather than quietly falling back, because "my GPU is not
being used" should not be something you discover by timing a benchmark.

### Whether it is worth it

Usually not, and the numbers say so. Measured on an RTX 2080 against a 12-core CPU, release build,
8 M events from a 346×260 recording:

| kernel | CPU | GPU | |
|---|---|---|---|
| `count` | 10.5 ms | 19.1 ms | CPU wins |
| `countmask` | 16.7 ms | 20.3 ms | CPU wins |
| `voxel(bins=5)` | 20.8 ms | 28.8 ms | CPU wins |
| `tsurf` | 34.2 ms | 29.8 ms | about even |
| `atsurf` | 83.9 ms | 29.2 ms | **2.9× faster** |

The pattern is not about the GPU being slow. These kernels are scatter-adds that the CPU already
does at memory speed, so the cost is dominated by getting the events across the bus — and only
`atsurf`, which evaluates an exponential per event, has enough arithmetic to pay for the trip. The
place the GPU wins clearly is the **simulator**, where every sub-step is a full pass of real
arithmetic over every pixel; see {doc}`simulation`.

So: leave it on the CPU unless you have measured otherwise on your own data. `cargo bench --features
gpu -- representations_gpu` runs the comparison above on your machine.

### What changes, numerically

Nothing you would notice, and the tests pin exactly how much:

- `count`, `polarity`, `countmask` and `tsurf` are **bit-identical** to the CPU. They accumulate
  integers, and integer addition commutes exactly where float addition does not.
- `voxel` and `atsurf` accumulate in Q16.16 fixed point, which is what makes them independent of
  the order the GPU happened to run in. They agree with the CPU to about 1e-4 on a cell — well
  below `float32`'s own resolution for those values — and are bit-reproducible run to run.
- A representation with no kernel (`tencode`, `mcts`, `binary`, `flow`) simply runs on the CPU when
  you ask for the GPU. A pipeline that sets `device="gpu"` once should not have to know which steps
  were ported.
