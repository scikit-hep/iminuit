import re
from typing import Tuple, Union


def parse_version(s: str) -> Union[Tuple[int, int], Tuple[int, int, int]]:
    """
    Parse version string and return tuple of integer parts to allow for comparison.

    This does not implement the full version spec for version parsing, see
    https://packaging.python.org/en/latest/specifications/version-specifiers/. It is a
    simplified approach, so we do not have to depend on the external packaging module.

    We only support correct ordering for major, mior, and micro segments, ie. version
    strings of the form X.Y and X.Y.Z. Versions with pre- and post-release segments are
    correctly parsed, but these segments are ignored, as well as development release
    segments.
    """
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", s)
    if not match:
        msg = f"could not parse version string {s}"
        raise ValueError(msg)
    if match.group(3):
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return int(match.group(1)), int(match.group(2))
