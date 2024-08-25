import subprocess as subp
from pathlib import PurePath

project_dir = PurePath(__file__).parent.parent

# check that root version is up-to-date
git_submodule = subp.check_output(
    ["git", "submodule", "status"], cwd=project_dir
).decode()

for item in git_submodule.strip().split("\n"):
    parts = item.split()
    if PurePath(parts[1]) != PurePath("extern") / "root":
        continue

    assert len(parts) == 3, "module is not checked out"
    break

root_dir = project_dir / "extern" / "root"

# git submodule status does not yield the right state
# we must use git describe --tags
subp.check_call(["git", "pull", "--tags"], cwd=root_dir)

root_version = (
    subp.check_output(["git", "describe", "--tags"], cwd=root_dir).decode().strip()
)
print("ROOT", root_version)
