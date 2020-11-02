# -*- coding: utf-8 -*-
import sys
from pathlib import Path

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

cwd = Path(__file__).parent

# Getting the version number at this point is a bit tricky in Python:
# https://packaging.python.org/guides/single-sourcing-package-version/?highlight=single%20sourcing
with (cwd / "src/iminuit/version.py").open() as fp:
    version = {}
    exec(fp.read(), version)  # this loads __version__
    version = version["__version__"]


with (cwd / "README.rst").open() as fp:
    txt = fp.read()
    # skip everything up to the skip marker
    skip_marker = ".. skip-marker-do-not-remove"
    long_description = txt[txt.index(skip_marker) + len(skip_marker) :].lstrip()


setup(
    name="iminuit",
    version=version,
    description="Jupyter-friendly Python frontend for MINUIT2 in C++",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author="Piti Ongmongkolkul and the iminuit team",
    maintainer="Hans Dembinski",
    maintainer_email="hans.dembinski@gmail.com",
    url="http://github.com/scikit-hep/iminuit",
    project_urls={
        "Documentation": "https://iminuit.readthedocs.io",
        "Source Code": "http://github.com/scikit-hep/iminuit",
    },
    packages=["iminuit", "iminuit.tests"],
    package_dir={"": "src"},
    cmake_install_dir="src/iminuit",
)
