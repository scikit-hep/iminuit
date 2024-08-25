import subprocess as subp
from pathlib import PurePath

project_dir = PurePath(__file__).parent.parent
root_dir = project_dir / "extern" / "root"

subp.check_call(["git", "checkout", "master"], cwd=root_dir)
git_submodule = subp.check_output(
    ["git", "submodule", "update"], cwd=project_dir
).decode()

# git submodule status does not yield the right state
# we must use git describe --tags
root_version = (
    subp.check_output(["git", "describe", "--tags"], cwd=root_dir).decode().strip()
)
print("ROOT", root_version)
