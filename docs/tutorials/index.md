# Tutorials

Hands-on notebooks that walk through EventCV end to end. They ship pre-executed (the docs
build does not re-run them), so the outputs you see are real.

```{toctree}
:maxdepth: 1

01-getting-started
02-transforms-and-representations
03-streaming-large-files
```

1. **[Getting started](01-getting-started.ipynb)** — load a recording, inspect it, and
   render a representation.
2. **[Transforms & representations](02-transforms-and-representations.ipynb)** — build a
   processing pipeline and feed it to PyTorch.
3. **[Streaming large files](03-streaming-large-files.ipynb)** — use {func}`eventcv.open`
   to slice a gigabyte recording without loading it whole.

To add another, drop a pre-run `.ipynb` into `docs/tutorials/` and list it in the
`toctree` above.
