from . import _rust
from . import load as _load
from .load import *  # noqa: F401,F403  (classes, load/open/save, + generated op forwarders)

# Read from the compiled extension (baked in from its Cargo.toml), so it names the binary that is
# actually loaded rather than the metadata of whatever distribution sits beside it. `eventcv
# --version` reports it; `tests/test_version.py` keeps it in step with pyproject.toml.
__version__ = _rust.__version__

# `__all__` is defined by `load` (curated names + the auto-generated functional API, §D1).
__all__ = list(_load.__all__)
