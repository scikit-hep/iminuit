import tomli
from pathlib import PurePath


def version():
    project_dir = PurePath(__file__).parent.parent
    with open(project_dir / "pyproject.toml", "rb") as f:
        data = tomli.load(f)
        return data["project"]["version"]


if __name__ == "__main__":
    print(version())
