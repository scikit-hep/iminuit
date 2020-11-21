from iminuit._deprecated import deprecated
import pytest


def test_deprecated_func():
    @deprecated("bla")
    def func(x):
        pass

    with pytest.warns(DeprecationWarning, match="func is deprecated: bla"):
        func(1)
