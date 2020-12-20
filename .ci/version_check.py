"""Ensure that current version is not in conflict with published releases."""
import requests
from pkg_resources import parse_version
import subprocess as subp
from pathlib import PurePath

project_dir = PurePath(__file__).parent.parent
version_fn = project_dir / "src/iminuit/version.py"
changelog_fn = project_dir / "doc/changelog.rst"

with open(version_fn) as f:
    version = {}
    exec(f.read(), version)  # this loads __version__
    version = version["__version__"]

# make sure that changelog was updated
with open(changelog_fn) as f:
    assert version in f.read(), "changelog entry missing"

# make sure that version is not already tagged
tags = subp.check_output(["git", "tag"]).decode().strip().split("\n")
assert f"v{version}" not in tags, "tag exists"

# make sure that version itself was updated
r = requests.get("https://pypi.org/pypi/iminuit/json")
releases = r.json()["releases"]
pypi_versions = [parse_version(v) for v in releases]
this_version = parse_version(version)
assert this_version not in pypi_versions, "pypi version exists"
