"""Model weights eventcv downloads on first use, pinned by revision and checksum.

Weights are not part of the wheel: they come from the Hugging Face Hub the first time a model is
constructed, are checked against the sha256 recorded here, and are reused from a local cache after
that — so a given eventcv release always runs the same bytes, and a cached model needs no network.

Environment:

- ``EVENTCV_CACHE`` — cache root (default ``~/.cache/eventcv``).
- ``EVENTCV_OFFLINE=1`` — never download; a model that is not cached raises instead.
- ``EVENTCV_HUB_URL`` — a mirror to download from instead of ``https://huggingface.co``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# name -> where the weights live and the bytes they must be. Each file carries its own licence,
# which is printed on the download that brings it in.
MODELS = {
    "e2vid": {
        "repo": "EventLAB/localization",
        "revision": "1f86495ada0f0c1ee39d0fd16e51a38fc89bb1a2",
        "file": "e2vid/e2vid.safetensors",
        "sha256": "601e6a5d182b937a225d9ef4604b322a931a829902e3447800d2f5f9be5f9447",
        "license": "GPL-3.0, derived from uzh-rpg/rpg_e2vid",
    },
}


def cache_dir() -> Path:
    """Where downloaded weights are kept."""
    root = os.environ.get("EVENTCV_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "eventcv"


def fetch(name: str) -> Path:
    """The local path of model ``name``'s weights, downloading and verifying them if needed."""
    try:
        entry = MODELS[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from None
    target = cache_dir() / "hub" / entry["repo"] / entry["revision"] / entry["file"]
    if target.exists():
        return target
    if os.environ.get("EVENTCV_OFFLINE"):
        raise RuntimeError(f"{name} is not cached at {target}, and EVENTCV_OFFLINE is set")

    base = os.environ.get("EVENTCV_HUB_URL", "https://huggingface.co").rstrip("/")
    url = f"{base}/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"
    print(f"eventcv: downloading {name} ({entry['license']}) from {url}", file=sys.stderr)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Into a temporary file beside the target, renamed only once verified: an interrupted or
    # corrupted download never leaves something at `target` that a later call would trust.
    handle, partial = tempfile.mkstemp(dir=target.parent, suffix=".partial")
    try:
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(url) as response:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise RuntimeError(
                f"{url} does not match the checksum this eventcv expects "
                f"(got {digest.hexdigest()}, want {entry['sha256']}); nothing was cached"
            )
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)
    return target
