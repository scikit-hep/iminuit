from iminuit._parse_version import parse_version
import pytest


@pytest.mark.parametrize(
    "s,ref",
    [
        ("1.2", (1, 2)),
        ("1.2.3", (1, 2, 3)),
        ("1.2a1", (1, 2)),
        ("1.2.3a1", (1, 2, 3)),
        ("1.2.post1", (1, 2)),
        ("1.2.3.post1", (1, 2, 3)),
        ("1.2a1.dev1", (1, 2)),
        ("1.2.3a1.dev1", (1, 2, 3)),
    ],
)
def test_parse_version(s, ref):
    assert parse_version(s) == ref


def test_parse_version_bad():
    with pytest.raises(ValueError):
        parse_version("a.b.c")
