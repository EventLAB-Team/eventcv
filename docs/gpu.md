# GPU

Representations, the simulator and E2VID can run on a GPU. Everything runs on the CPU by default.

```python
import eventcv as ecv

frame = stream.voxel(bins=5, device="gpu")  # one call
ecv.set_device("gpu")  # every call from here on
ecv.gpu_available()  # False if this build or machine has no GPU support
```

Set `EVENTCV_DEVICE=gpu` to change the default for a whole process without touching code.

The backend is wgpu: Metal on macOS, Vulkan on Linux and DX12 on Windows. No CUDA install is needed.
Asking for a GPU that isn't available raises an error rather than silently using the CPU.

## When it's faster

| Workload | GPU vs CPU |
|---|---|
| `count`, `countmask`, `voxel` | CPU is faster |
| `tsurf` | About even |
| `atsurf` | About 3× faster |
| {func}`~eventcv.simulate` | About 2× faster |
| {class}`~eventcv.E2VID` | About 15× faster in half precision (the default), 4–5× in single |

Most representations are limited by copying events to the GPU, so the CPU usually wins. Keep the
default unless you have measured a gain on your own data.

## What runs on the GPU

| Accepts `device=` | Runs on the CPU regardless |
|---|---|
| `count`, `voxel`, `tsurf`, `atsurf`, `countmask`, `redblue`, `simulate`, `E2VID` | `tencode`, `mcts`, `binary`, `flow` |

`set_device("gpu")` also covers `polarity` and reader slices. Representations without a GPU version
fall back to the CPU, so one setting works across a whole pipeline.

## Differences from the CPU

| Workload | Result |
|---|---|
| `count`, `polarity`, `countmask`, `redblue`, `tsurf` | Identical |
| `voxel`, `atsurf` | Within about 1e-4, and identical from run to run |
| `simulate` without noise | Identical events, rare 1 µs timestamp differences |
| `simulate` with noise | Same event rate, different individual events |

To compare a GPU simulation with a stored CPU one, turn noise off (`shot_noise_rate_hz=0,
leak_rate_hz=0`).
