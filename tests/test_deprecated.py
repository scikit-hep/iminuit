from iminuit._deprecated import deprecated, deprecated_parameter
import pytest
import numpy as np


def test_deprecated_func():
    @deprecated("bla")
    def func(x):
        pass

    with pytest.warns(np.VisibleDeprecationWarning, match="func is deprecated: bla"):
        func(1)


def test_deprecated_parameter():
    @deprecated_parameter(foo="bar")
    def some_function(x, y, foo):
        pass

    some_function(1, 2, foo=3)

    with pytest.warns(
        np.VisibleDeprecationWarning,
        match="keyword 'bar' is deprecated, please use 'foo'",
    ):
        some_function(1, 2, bar=3)

    with pytest.raises(TypeError):
        some_function(x=1, baz=3, y=2)
