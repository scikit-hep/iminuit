import requests
from pkg_resources import parse_version
import pathlib

dir = pathlib.Path().absolute()
while dir:
    dir = dir.parent
    if dir.glob("setup.py"):
        break

with open(dir / "src" / "iminuit" / "version.py") as f:
    version = {}
    exec(f.read(), version)  # this loads __version__
    version = version["__version__"]

r = requests.get("https://pypi.org/pypi/iminuit/json")
releases = r.json()["releases"]

pypi_versions = [parse_version(v) for v in releases]
this_version = parse_version(version)

# make sure that version was updated
assert this_version not in pypi_versions

# make sure that changelog was updated
with open(dir / "doc" / "changelog.rst") as f:
    assert version in f.read()
