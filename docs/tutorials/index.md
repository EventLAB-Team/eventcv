# Tutorials

Hands-on notebooks that walk through EventCV end to end. They ship pre-executed (the docs
build does not re-run them), so the outputs you see are real.

```{toctree}
:maxdepth: 1

```

*Notebooks land here as they are written.* Planned:

1. **Getting started** — load a recording, inspect it, and render a representation.
2. **Transforms & representations** — build a processing pipeline and feed it to PyTorch.
3. **Streaming a gigabyte file** — use {func}`eventcv.open` to slice a large recording
   without loading it whole.

To add one, drop a `.ipynb` (pre-run) into `docs/tutorials/` and list it in the
`toctree` above.
