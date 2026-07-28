"""Sphinx configuration for the EventCV Python API docs.

Built on Read the Docs (see ``.readthedocs.yaml``): RTD installs the compiled
``eventcv`` extension, then autodoc imports it and pulls docstrings straight from the
Rust-defined methods and the generated functional API — so the reference never drifts
from the code. Build locally with ``pip install -r docs/requirements.txt`` (plus an
editable ``maturin develop``) then ``sphinx-build -b html docs docs/_build/html``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

project = "EventCV"
author = "Adam Hines"
copyright = "2026, EventLAB"

try:
    release = _pkg_version("eventcv")
except PackageNotFoundError:  # docs building against an uninstalled tree
    release = "1.0.4"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",  # Markdown pages + (executable) notebook tutorials
]

# --- autodoc -------------------------------------------------------------------
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# The compiled pyclasses expose plain docstrings, not annotations; keep signatures
# as-authored rather than trying to promote types into the description.
autodoc_typehints = "signature"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# --- notebooks (myst-nb) -------------------------------------------------------
# Tutorials ship pre-executed; don't re-run them at build time (they need data files
# and a GPU for the viewer sections).
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist"]

# --- cross-references ----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# --- HTML ----------------------------------------------------------------------
html_theme = "furo"
html_title = f"EventCV {release}"
