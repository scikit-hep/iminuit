"""Print iminuit version."""

from pathlib import PurePath

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

with open(PurePath(__file__).parent / "pyproject.toml", "rb") as f:
    version = toml.load(f)["project"]["version"]

if __name__ == "__main__":
    print(version)
