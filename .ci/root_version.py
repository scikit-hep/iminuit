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

    this_root_version = parts[2][1:-1]  # strip braces

print(this_root_version)
