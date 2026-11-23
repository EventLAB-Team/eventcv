"""Sphinx configuration for the EventCV docs.

autodoc imports the installed ``eventcv`` package. Build locally with
``pip install -r docs/requirements.txt`` (plus ``maturin develop`` or the PyPI wheel), then
``python -m sphinx -b html docs docs/_build/html``.
"""

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

project = "EventCV"
author = "Adam Hines"
copyright = "2026, EventLAB"

try:
    release = _pkg_version("eventcv")
except PackageNotFoundError:  # docs building against an uninstalled tree: read the workspace
    import re

    _cargo = open(os.path.join(os.path.dirname(__file__), "..", "Cargo.toml")).read()
    release = re.search(r'\[workspace\.package\]\s*version = "([^"]+)"', _cargo).group(1)
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",  # Markdown pages + (executable) notebook tutorials
    "sphinx_design",  # grids and cards on the landing page
    "sphinx_copybutton",
]

# Strip shell / REPL prompts so a copied block runs as-is.
copybutton_prompt_text = r"\$ |>>> |\.\.\. "
copybutton_prompt_is_regexp = True

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
myst_heading_anchors = 3

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
html_logo = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "source_repository": "https://github.com/EventLAB-Team/eventcv/",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
}

# Read the Docs sets this to the canonical domain and version.
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://docs.eventcv.net/en/latest/",
)
