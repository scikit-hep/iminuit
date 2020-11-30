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
pypi_versions = []
for prefix in ("pypi", "test.pypi"):
    url = f"https://{prefix}.org/pypi/iminuit/json"
    r = requests.get(url)
    releases = r.json()["releases"]
    pypi_versions += [parse_version(v) for v in releases]

assert parse_version(version) not in pypi_versions
