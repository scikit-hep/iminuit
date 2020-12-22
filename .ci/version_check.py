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
    exec(f.read(), version)
    iminuit_version = version["iminuit_version"]
    root_version = version["root_version"]

# check that root version is up-to-date
git_submodule = subp.check_output(
    ["git", "submodule", "status"], cwd=project_dir
).decode()
submodule_root_version = git_submodule[git_submodule.index("extern/root") + 13 : -2]
assert (
    root_version == submodule_root_version
), f"ROOT version does not match: {root_version} != {submodule_root_version}"

# make sure that changelog was updated
with open(changelog_fn) as f:
    assert iminuit_version in f.read(), "changelog entry missing"

# make sure that version is not already tagged
tags = subp.check_output(["git", "tag"]).decode().strip().split("\n")
assert f"v{iminuit_version}" not in tags, "tag exists"

# make sure that version itself was updated
pypi_versions = [
    parse_version(v)
    for v in requests.get("https://pypi.org/pypi/iminuit/json").json()["releases"]
]
assert parse_version(iminuit_version) not in pypi_versions, "pypi version exists"
