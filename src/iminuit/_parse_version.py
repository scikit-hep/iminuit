import re
from typing import Tuple


def parse_version(s: str) -> Tuple[int, ...]:
    """
    Parse version string and return tuple of integer parts to allow for comparison.

    This does not implement RFC2119. It is a poor-mans approach, so we do not have to
    depend on the external packaging module. It should work for our purposes, though.
    """
    match = re.match("(\d+)\.(\d+)(?:\.(\d+))?", s)
    if not match:
        msg = f"could not parse version string {s}"
        raise ValueError(msg)
    if match.group(3):
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return int(match.group(1)), int(match.group(2))
