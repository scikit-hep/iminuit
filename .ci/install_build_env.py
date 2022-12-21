import subprocess as subp
from pathlib import Path

packages = [
    "wheel",
    "pip",
    "tomli",
    "numpy",
    "pybind11",
    "importlib_metadata",
    "distlib",
    "pathspec",
    "pyproject_metadata",
    "scikit_build_core",
]

subp.check_call(
    ["python", "-m", "pip", "install", "--upgrade", "--prefer-binary"] + packages,
    stdout=subp.DEVNULL,
)

import tomli  # noqa

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
    d = tomli.load(f)

assert d["build-system"]["requires"] == ["scikit-build-core", "pybind11"]
