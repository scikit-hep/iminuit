import os

# release and version are special variables used by sphinx
from iminuit import __version__ as version

# We set the root_version manually, because it setting it
# automatically and efficiently in CI is difficult.
# To update the number:
# - Make sure you have a full clone, not a shallow clone of ROOT.
# - Run `doc/root_version.py` and copy the string here.
root_version = "v6-25-02-9213-g754d22635f"


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
    print("Fast generation activated")
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

html_theme = "sphinx_rtd_theme"

nbsphinx_kernel_name = "python3"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]

linkcheck_timeout = 10
linkcheck_allow_unauthorized = True
linkcheck_report_timeouts_as_broken = False
linkcheck_ignore = ["https://doi.org/10.2307%2F2347496"]
