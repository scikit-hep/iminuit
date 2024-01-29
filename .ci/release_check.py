"""Ensure that current version is not in conflict with published releases."""

from pkg_resources import parse_version
import subprocess as subp
from pathlib import PurePath
import urllib.request
import json
import warnings
import sys


project_dir = PurePath(__file__).parent.parent
changelog_fn = project_dir / "doc/changelog.rst"
version_fn = project_dir / "version.py"

version = subp.check_output([sys.executable, version_fn]).strip().decode()

with warnings.catch_warnings(record=True) as record:
    iminuit_version = parse_version(version)
if record:
    raise ValueError(record[0].message)

print("iminuit version:", iminuit_version)

# make sure that changelog was updated
with open(changelog_fn) as f:
    assert iminuit_version.base_version in f.read(), "changelog entry missing"

# make sure that version is not already tagged
tags = subp.check_output(["git", "tag"]).decode().strip().split("\n")
assert f"v{iminuit_version}" not in tags, "tag exists"

# make sure that version itself was updated
with urllib.request.urlopen("https://pypi.org/pypi/iminuit/json") as r:
    pypi_versions = [parse_version(v) for v in json.loads(r.read())["releases"]]

pypi_versions.sort()
print("PyPI    version:", pypi_versions[-1])

assert iminuit_version not in pypi_versions, "pypi version exists"
