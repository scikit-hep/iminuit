from pathlib import Path
import sys


def install(packages):
    import subprocess as subp

    subp.check_call(
        ["python", "-m", "pip", "install", "--upgrade", "--prefer-binary"] + packages
    )


packages = [
    "tomli",
    "pip",
    "wheel",
]

install(packages)

import tomli  # noqa

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
    d = tomli.load(f)

for arg in sys.argv[1:]:
    if arg == "build":
        install(d["build-system"]["requires"])
    else:
        packages = d["project"]["optional-dependencies"][arg]
        install(packages)
