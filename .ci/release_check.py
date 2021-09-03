"""Ensure that current version is not in conflict with published releases."""
from pkg_resources import parse_version
import subprocess as subp
from pathlib import PurePath
import urllib.request
import json

project_dir = PurePath(__file__).parent.parent
version_fn = project_dir / "src/iminuit/version.py"
changelog_fn = project_dir / "doc/changelog.rst"

with open(version_fn) as f:
    version = {}
    exec(f.read(), version)
    iminuit_version = version["version"]
    root_version = version["root_version"]

print("iminuit version:", iminuit_version, root_version)

# check that root version is up-to-date
git_submodule = subp.check_output(
    ["git", "submodule", "status"], cwd=project_dir
).decode()
for item in git_submodule.strip().split("\n"):
    parts = item.split()
    if PurePath(parts[1]) != PurePath("extern") / "root":
        continue

    assert len(parts) == 3, "module is not checked out"

    this_root_version = parts[2][1:-1]  # strip braces

    print("actual ROOT version:", this_root_version)

    assert (
        root_version == this_root_version
    ), f"ROOT version does not match: {root_version} != {this_root_version}"

# make sure that changelog was updated
with open(changelog_fn) as f:
    assert iminuit_version in f.read(), "changelog entry missing"

# make sure that version is not already tagged
tags = subp.check_output(["git", "tag"]).decode().strip().split("\n")
assert f"v{iminuit_version}" not in tags, "tag exists"

# make sure that version itself was updated
with urllib.request.urlopen("https://pypi.org/pypi/iminuit/json") as r:
    pypi_versions = [parse_version(v) for v in json.loads(r.read())["releases"]]

print("latest PyPI version:", pypi_versions[-1])

assert parse_version(iminuit_version) not in pypi_versions, "pypi version exists"
