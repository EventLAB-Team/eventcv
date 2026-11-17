"""Convert the reference E2VID checkpoint into the weights file `eventcv.E2VID` loads.

    python scripts/convert_e2vid.py --upstream ~/src/rpg_e2vid \
        --checkpoint E2VID_lightweight.pth.tar --out e2vid/

Needs torch and a checkout of https://github.com/uzh-rpg/rpg_e2vid. The architecture is imported
from that checkout at run time, so no upstream (GPL-3.0) code is copied into eventcv; the files this
writes are derived from upstream's weights and are published under upstream's licence, separately
from eventcv (https://huggingface.co/EventLAB/localization/tree/main/e2vid).

Writes into ``--out``:

- ``e2vid.safetensors`` -- BatchNorm folded into every conv, plus the unsharp-mask kernel, with the
  architecture in the metadata map. One file, so the hub pins one hash.
- ``reference.npz`` -- upstream's own reconstruction of the first ``--frames`` windows of
  ``--events``, for the parity test (``EVENTCV_E2VID_REFERENCE=<out> pytest tests/test_reconstruct.py``).
  Not for publishing.
- ``README.md`` and ``LICENSE`` for the hub repository.
"""

import argparse
import json
import math
import shutil
import struct
import sys
import types
from pathlib import Path

import numpy as np

# rpg_e2vid's defaults (options/inference_options.py), which define "classic E2VID".
EVENTS_PER_PIXEL = 0.35
UNSHARP_AMOUNT = 0.3
UNSHARP_SIGMA = 1.0
BN_EPS = 1e-5  # torch.nn.BatchNorm2d's default; upstream never overrides it


def import_upstream(root):
    """Import rpg_e2vid from a checkout, standing in for two imports its inference path never uses.

    `utils/inference_utils.py` imports cv2 at module level (only its preview helpers use it) and
    scipy only for `norm.cdf` in the unsharp kernel, which `math.erf` gives exactly.
    """
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    try:
        import scipy.stats  # noqa: F401
    except ImportError:
        cdf = np.vectorize(lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))))
        stats = types.ModuleType("scipy.stats")
        stats.norm = types.SimpleNamespace(cdf=cdf)
        scipy = types.ModuleType("scipy")
        scipy.stats = stats
        sys.modules["scipy"], sys.modules["scipy.stats"] = scipy, stats
    sys.path.insert(0, str(root))
    from model.model import E2VIDRecurrent
    from utils import inference_utils

    return E2VIDRecurrent, inference_utils


def fold(state, conv, norm=None, transposed=False):
    """Conv followed by eval-mode BatchNorm, as one conv's weight and bias.

    Upstream gives a conv a bias exactly when no BatchNorm follows it, so those pass straight through.
    """
    weight = state[f"{conv}.weight"].double()
    if norm is None:
        return weight.float(), state[f"{conv}.bias"].float()
    scale = (
        state[f"{norm}.weight"].double() / (state[f"{norm}.running_var"].double() + BN_EPS).sqrt()
    )
    shift = state[f"{norm}.bias"].double() - state[f"{norm}.running_mean"].double() * scale
    # Output channels are axis 0 of a conv weight and axis 1 of a transposed-conv weight.
    shape = (1, -1, 1, 1) if transposed else (-1, 1, 1, 1)
    return (weight * scale.reshape(shape)).float(), shift.float()


def convert(state, config, kernel):
    p = "unetrecurrent."
    tensors = {}

    def put(name, conv, norm=None, transposed=False):
        tensors[f"{name}.weight"], tensors[f"{name}.bias"] = fold(
            state, p + conv, norm and p + norm, transposed
        )

    put("head", "head.conv2d")
    for i in range(config["num_encoders"]):
        put(f"encoders.{i}.conv", f"encoders.{i}.conv.conv2d", f"encoders.{i}.conv.norm_layer")
        put(f"encoders.{i}.gates", f"encoders.{i}.recurrent_block.Gates")
    for i in range(config["num_residual_blocks"]):
        put(f"resblocks.{i}.conv1", f"resblocks.{i}.conv1", f"resblocks.{i}.bn1")
        put(f"resblocks.{i}.conv2", f"resblocks.{i}.conv2", f"resblocks.{i}.bn2")
    for i in range(config["num_encoders"]):
        put(f"decoders.{i}", f"decoders.{i}.transposed_conv2d", f"decoders.{i}.norm_layer", True)
    put("pred", "pred.conv2d", "pred.norm_layer")
    tensors["unsharp.kernel"] = kernel.reshape(1, 1, *kernel.shape[-2:]).float()
    return {name: t.contiguous().numpy() for name, t in tensors.items()}


def write_safetensors(path, tensors, metadata):
    """The safetensors layout: u64 header length, a JSON header, then the raw little-endian bytes."""
    header, offset, blobs = {"__metadata__": metadata}, 0, []
    for name, array in tensors.items():
        blob = np.ascontiguousarray(array, dtype="<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def reference(model, utils, config, events_path, frames, per_window=None):
    """Upstream's own pipeline over eventcv's event-count windows, with recurrence carried.

    Only the windowing comes from eventcv (it decides which events go together); the voxel grid,
    normalisation, padding, network and unsharp mask are all upstream's.
    """
    import eventcv as ecv
    import torch

    stream = ecv.load(str(events_path))
    width, height = stream.sensor_size
    per_window = per_window or round(EVENTS_PER_PIXEL * width * height)
    reader = ecv.open(str(events_path), max_events=per_window)
    options = types.SimpleNamespace(
        no_normalize=False,
        hot_pixels_file=None,
        flip=False,
        unsharp_mask_amount=UNSHARP_AMOUNT,
        unsharp_mask_sigma=UNSHARP_SIGMA,
    )
    preprocess = utils.EventPreprocessor(options)
    unsharp = utils.UnsharpMaskFilter(options, torch.device("cpu"))
    crop = utils.CropParameters(width, height, config["num_encoders"])
    states, out = None, []
    with torch.no_grad():
        for i in range(min(frames, len(reader))):
            x, y, t, p = reader[i].numpy().T.astype(np.float64)  # columns are x, y, t, p
            events = np.stack([t, x, y, p], axis=1)
            voxel = utils.events_to_voxel_grid_pytorch(
                events, config["num_bins"], width, height, torch.device("cpu")
            )
            image, states = model(crop.pad(preprocess(voxel.unsqueeze(0))), states)
            image = unsharp(image)[0, 0, crop.iy0 : crop.iy1, crop.ix0 : crop.ix1]
            out.append(image.clamp(0, 1).numpy())
    return np.stack(out), per_window


MODEL_CARD = """---
license: gpl-3.0
tags: [event-camera, video-reconstruction, eventcv]
---

# E2VID (lightweight) for eventcv

The pretrained `E2VID_lightweight` recurrent reconstruction network from
[rpg_e2vid](https://github.com/uzh-rpg/rpg_e2vid) (UZH Robotics and Perception Group), converted for
[eventcv](https://github.com/EventLAB-Team/eventcv) by `scripts/convert_e2vid.py`. BatchNorm is
folded into the convolutions; the network is otherwise unchanged.

```python
import eventcv as ecv
frames = ecv.reconstruct("recording.h5")       # fetches this file on first use
```

**Licence.** rpg_e2vid is released under GPL-3.0 and these weights are derived from it, so they are
distributed under GPL-3.0 here — separately from eventcv itself (Apache-2.0), which downloads them
at run time and does not bundle them.

Source checkpoint sha256: `{checkpoint_sha}`

```bibtex
@Article{Rebecq19pami,
  author  = {Henri Rebecq and Ren{\\'{e}} Ranftl and Vladlen Koltun and Davide Scaramuzza},
  title   = {High Speed and High Dynamic Range Video with an Event Camera},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell. (T-PAMI)},
  year    = 2019
}
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--upstream", type=Path, required=True, help="rpg_e2vid checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="E2VID_lightweight.pth.tar")
    parser.add_argument("--out", type=Path, required=True)
    repo = Path(__file__).resolve().parent.parent
    parser.add_argument("--events", type=Path, default=repo / "data" / "test" / "example.npz")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--max-events",
        type=int,
        default=5000,
        help="events per reference window; the default keeps example.npz (106k events) to ~20 "
        "windows so recurrence is exercised, where E2VID's 0.35/pixel would make it one",
    )
    args = parser.parse_args()

    import hashlib

    import torch

    E2VIDRecurrent, utils = import_upstream(args.upstream.resolve())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["model"]
    expected = {
        "skip_type": "sum",
        "recurrent_block_type": "convlstm",
        "use_upsample_conv": False,
        "norm": "BN",
    }
    if checkpoint["arch"] != "E2VIDRecurrent" or any(
        config.get(k) != v for k, v in expected.items()
    ):
        sys.exit(f"unsupported variant: {checkpoint['arch']} {config}")
    model = E2VIDRecurrent(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    unsharp = utils.UnsharpMaskFilter(
        types.SimpleNamespace(unsharp_mask_amount=UNSHARP_AMOUNT, unsharp_mask_sigma=UNSHARP_SIGMA),
        torch.device("cpu"),
    )
    tensors = convert(checkpoint["state_dict"], config, unsharp.gaussian_kernel)
    metadata = {
        "format": "eventcv-e2vid",
        "version": "1",
        "num_bins": str(config["num_bins"]),
        "num_encoders": str(config["num_encoders"]),
        "num_residual_blocks": str(config["num_residual_blocks"]),
        "events_per_pixel": str(EVENTS_PER_PIXEL),
        "unsharp_amount": str(UNSHARP_AMOUNT),
        "license": "GPL-3.0 (derived from uzh-rpg/rpg_e2vid)",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_safetensors(args.out / "e2vid.safetensors", tensors, metadata)

    frames, per_window = reference(model, utils, config, args.events, args.frames, args.max_events)
    np.savez(args.out / "reference.npz", frames=frames, max_events=per_window)

    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    (args.out / "README.md").write_text(MODEL_CARD.replace("{checkpoint_sha}", checkpoint_sha))
    shutil.copyfile(args.upstream / "LICENSE", args.out / "LICENSE")
    weights_sha = hashlib.sha256((args.out / "e2vid.safetensors").read_bytes()).hexdigest()
    print(
        f"wrote {args.out}: e2vid.safetensors (sha256 {weights_sha}), {len(frames)} reference frames"
    )


if __name__ == "__main__":
    main()
