from iminuit.util import describe, make_func_code
from math import ldexp
import sys
import pytest
import platform

is_pypy = platform.python_implementation() == "PyPy"


def test_function():
    def f(x, y):
        # body is important
        a = 1
        b = 2
        return a + b

    assert describe(f, True) == ["x", "y"]


def test_class_method():
    class A:
        def f(self, x, y):
            # body is important
            a = 1
            b = 2
            return a + b

    assert describe(A().f, True) == ["x", "y"]


# unbound method
def test_class_unbound_method():
    class A:
        def f(self, x, y):
            # body is important
            a = 1
            b = 2
            return a + b

    assert describe(A.f, True) == ["self", "x", "y"]


def test_functor():
    class A:
        def __call__(self, x, y):
            # body is important
            a = 1
            b = 2
            return a + b

    assert describe(A(), True) == ["x", "y"]


@pytest.mark.skipif(
    sys.version_info >= (3, 7), reason="Not on Py 3.7; see Github issue 270"
)
def test_builtin_by_parsing_doc():
    assert describe(ldexp, True) == ["x", "i"]


def test_lambda():
    assert describe(lambda a, b: 0, True) == ["a", "b"]


def test_generic_function():
    def f(*args):
        # body is important
        a = 1
        b = 2
        return a + b

    with pytest.raises(TypeError):
        describe(f, True)


def test_generic_lambda():
    with pytest.raises(TypeError):
        describe(lambda *args: 0, True)


def test_generic_class_method():
    class A:
        def f(self, *args):
            # body is important
            a = 1
            b = 2
            return a + b

    with pytest.raises(TypeError):
        describe(A().f, True)


def test_generic_functor():
    class A:
        def __call__(self, *args):
            # body is important
            a = 1
            b = 2
            return a + b

    with pytest.raises(TypeError):
        describe(A(), True)


def test_generic_functor_with_fake_func():
    class A:
        def __init__(self):
            self.func_code = make_func_code(["x", "y"])

        def __call__(self, *args):
            # body is important
            a = 1
            b = 2
            return a + b

    assert describe(A(), True) == ["x", "y"]
