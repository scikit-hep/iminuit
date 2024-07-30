from iminuit._deprecated import deprecated, deprecated_parameter
import pytest


def test_deprecated_func_1():
    @deprecated("bla")
    def func(x):
        pass

    with pytest.warns(
        FutureWarning,
        match="func is deprecated: bla",
    ):
        func(1)


def test_deprecated_func_2():
    @deprecated("bla", removal="1.0")
    def func(x):
        pass

    with pytest.warns(
        DeprecationWarning,
        match="func is deprecated and will be removed in version 1.0: bla",
    ):
        func(1)


def test_deprecated_parameter():
    @deprecated_parameter(foo="bar")
    def some_function(x, y, foo):
        pass

    some_function(1, 2, foo=3)

    with pytest.warns(
        FutureWarning,
        match="keyword 'bar' is deprecated, please use 'foo'",
    ):
        some_function(1, 2, bar=3)

    with pytest.raises(TypeError):
        some_function(x=1, baz=3, y=2)
