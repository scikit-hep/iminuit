import requests
from pkg_resources import parse_version
import sys

version_fn, changelog_fn = sys.argv[1:]

with open(version_fn) as f:
    version = {}
    exec(f.read(), version)  # this loads __version__
    version = version["__version__"]

# make sure that changelog was updated
with open(changelog_fn) as f:
    assert version in f.read()

# make sure that version itself was updated
r = requests.get("https://pypi.org/pypi/iminuit/json")
releases = r.json()["releases"]

pypi_versions = [parse_version(v) for v in releases]
this_version = parse_version(version)

assert this_version not in pypi_versions
