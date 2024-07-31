import os
from iminuit import __version__ as version
import sys

sys.path.append(".")

from root_version import root_version  # noqa

# release and version are special variables used by sphinx

with open("../README.rst") as f:
    readme_content = f.read()

readme_content = readme_content.replace("doc/", "")
readme_content = readme_content.replace(
    ".. version-marker-do-not-remove",
    "**These docs are for iminuit version:** |release|",
)

with open("index.rst.in") as f:
    index_content = f.read()

with open("index.rst", "w") as f:
    f.write(readme_content + index_content)

release = f"{version} compiled with ROOT-{root_version}"

project = "iminuit"
copyright = "2022, Hans Dembinski and the iminuit team"

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    # 'matplotlib.sphinxext.only_directives',
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
nbsphinx_execute = "auto"
# use FAST=1 to speed up doc build
if bool(os.environ.get("FAST", False)):
    nbsphinx_execute = "never"

autoclass_content = "both"
autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_themes"]

# html_logo = "_static/iminuit_logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# http://read-the-docs.readthedocs.org/en/latest/theme.html#how-do-i-use-this-locally-and-on-read-the-docs
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    # Import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]

linkcheck_timeout = 3
linkcheck_allow_unauthorized = True
