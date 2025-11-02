import subprocess as subp
from pathlib import PurePath
import sys
import ast

doc_path = PurePath(__file__).parent


def get_root_version() -> str:
    project_dir = doc_path.parent

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

    # git submodule status does not yield the right state
    # we must use git describe --tags
    root_version = (
        subp.check_output(
            ["git", "describe", "--tags"], cwd=project_dir / "extern" / "root"
        )
        .decode()
        .strip()
    )
    return root_version


def get_root_version_from_conf() -> str:
    with open(doc_path / "conf.py") as f:
        tree = ast.parse(source=f.read())

    for node in ast.walk(tree):
        # Look for: root_version = "something"
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "root_version":
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        return node.value.value

    return "UNKNOWN"


if __name__ == "__main__":
    conf_root_version = get_root_version_from_conf()
    root_version = get_root_version()

    if conf_root_version != root_version:
        print(
            f"Please update root_version in doc/conf.py from {conf_root_version} to {root_version}"
        )
        sys.exit(1)
