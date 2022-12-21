import subprocess as subp
from pathlib import Path

packages = [
    "wheel",
    "pip",
    "numpy",
    "cmake",
    "importlib_metadata",
]

subp.check_call(
    ["python", "-m", "pip", "install", "--upgrade", "--prefer-binary"] + packages,
    stdout=subp.DEVNULL,
)

import tomli  # noqa

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
    d = tomli.load(f)

assert d["build-system"]["requires"] == [
    "setuptools>=42",
    "cmake",
]
