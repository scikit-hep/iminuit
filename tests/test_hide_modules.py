import sys

import pytest

from iminuit._hide_modules import hide_modules


def test_reload_is_undone_for_from_import():
    # The parent package caches the submodule as an attribute. Unless that
    # attribute is dropped too, `from iminuit import cost` keeps returning the
    # module that was imported while numba was hidden.
    import iminuit.cost

    before = iminuit.cost

    with hide_modules("numba", reload="iminuit.cost"):
        from iminuit import cost

        hidden = cost
        assert hidden is not before

    from iminuit import cost

    assert cost is not hidden


def test_meta_path_restored_on_exception():
    before = list(sys.meta_path)

    with pytest.raises(ValueError), hide_modules("numba", reload="iminuit.cost"):
        raise ValueError

    assert sys.meta_path == before


def test_meta_path_entry_removed_by_identity():
    # Something inside the block may insert its own finder at the front.
    class Finder:
        def find_spec(self, fullname, path, target=None):
            return None

    extra = Finder()
    before = list(sys.meta_path)

    with hide_modules("numba"):
        sys.meta_path.insert(0, extra)

    try:
        assert sys.meta_path == [extra, *before]
    finally:
        sys.meta_path.remove(extra)
