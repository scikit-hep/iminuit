import subprocess as subp
from pathlib import PurePath
import sys
import ast
import re

doc_path = PurePath(__file__).parent

# matches `git describe --tags` output: <tag>-<count>-g<hash>
# the tag itself may contain dashes (e.g. "v6-37-01"), so it is matched greedily
DESCRIBE_RE = re.compile(r"^(?P<tag>.+)-(?P<count>\d+)-g(?P<hash>[0-9a-f]+)$")


def describe_matches(a: str, b: str) -> bool:
    """Compare two `git describe --tags` strings, tolerating hash abbreviation length."""
    ma = DESCRIBE_RE.match(a)
    mb = DESCRIBE_RE.match(b)
    if not ma or not mb:
        return a == b
    if ma["tag"] != mb["tag"] or ma["count"] != mb["count"]:
        return False
    ha, hb = ma["hash"], mb["hash"]
    return ha.startswith(hb) or hb.startswith(ha)


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
        if len(parts) != 3:
            raise RuntimeError("module is not checked out")
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
    try:
        root_version = get_root_version()
    except (RuntimeError, subp.CalledProcessError):
        # These errors indicate that we have only a shallow ROOT checkout.
        # Since we cannot check the ROOT version then, we 'pass' the check.
        sys.exit(0)
    conf_root_version = get_root_version_from_conf()

    if not describe_matches(conf_root_version, root_version):
        print(
            f"Please update root_version in doc/conf.py from {conf_root_version} to {root_version}"
        )
        sys.exit(1)
