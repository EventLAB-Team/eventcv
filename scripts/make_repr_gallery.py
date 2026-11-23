"""Render the representation thumbnails in docs/images/representations/ from the test fixture.

python scripts/make_repr_gallery.py
"""

from pathlib import Path

import eventcv as ecv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "representations"

stream = ecv.load(str(ROOT / "data" / "test" / "example.npz"))
frames = {
    "polarity": stream.flatten(),
    "binary": stream.flatten(binary=True),
    "count": stream.count(normalize=True),
    "voxel": stream.voxel(bins=5),
    "tsurf": stream.tsurf(),
    "atsurf": stream.atsurf(),
    "mcts": stream.mcts(),
    "tencode": stream.tencode(),
    "countmask": stream.countmask(),
    "redblue": stream.redblue(),
    "flow": stream.optical_flow(),
}

OUT.mkdir(parents=True, exist_ok=True)
for name, frame in frames.items():
    frame.resize(320, 240).save(str(OUT / f"{name}.png"))
    print(OUT / f"{name}.png")
