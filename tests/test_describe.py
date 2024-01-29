from iminuit.util import describe, make_func_code
from iminuit.typing import Annotated, Gt, Lt, Ge, Le, Interval
from math import ldexp
import platform
from functools import wraps
import pytest
import numpy as np
from numpy.typing import NDArray

is_pypy = platform.python_implementation() == "PyPy"


def test_function():
    def f(x, y):
        pass

    assert describe(f) == ["x", "y"]


def test_class_method():
    class A:
        def f(self, x, y):
            pass

    assert describe(A().f) == ["x", "y"]


def test_class_unbound_method():
    class A:
        def f(self, x, y):
            pass

    assert describe(A.f) == ["self", "x", "y"]


def test_functor():
    class A:
        def __call__(self, x, y):
            pass

    assert describe(A()) == ["x", "y"]


def test_builtin_by_parsing_doc():
    assert describe(ldexp) == ["x", "i"]


def test_lambda():
    assert describe(lambda a, b: 0) == ["a", "b"]


def test_generic_function():
    def f(*args):
        pass

    assert describe(f) == []


def test_generic_partial():
    from functools import partial

    def f(*args):
        pass

    partial_f = partial(f, 42, 12, 4)
    assert describe(partial_f) == []


def test_generic_lambda():
    assert describe(lambda *args: 0) == []


def test_generic_class_method():
    class A:
        def f(self, *args):
            pass

    assert describe(A().f) == []


def test_generic_functor():
    class A:
        def __call__(self, *args):
            pass

    assert describe(A()) == []


def test_generic_functor_with_fake_func():
    class A:
        def __init__(self):
            self.func_code = make_func_code(["x", "y"])

        def __call__(self, *args):
            pass

    with pytest.warns(np.VisibleDeprecationWarning):
        assert describe(A()) == ["x", "y"]


def test_decorated_function():
    def dummy_decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)

        return decorated

    @dummy_decorator
    def one_arg(x):
        pass

    @dummy_decorator
    def many_arg(x, y, z, t):
        pass

    @dummy_decorator
    def kw_only(x, *, y, z):
        pass

    assert describe(one_arg) == list("x")
    assert describe(many_arg) == list("xyzt")
    assert describe(kw_only) == list("xyz")


def test_ambiguous_1():
    def f(x, *args):
        pass

    assert describe(f) == ["x"]


def test_ambiguous_2():
    def f(x, kw=None):
        pass

    assert describe(f) == ["x", "kw"]


def test_ambiguous_3():
    def f(x, **kw):
        pass

    assert describe(f) == ["x"]


def test_ambiguous_4():
    class A:
        def __call__(self, x, **kw):
            pass

    assert describe(A()) == ["x"]


def test_from_docstring_1():
    def f(*args):
        """f(x, y, z)"""

    assert describe(f) == ["x", "y", "z"]


def test_from_docstring_2():
    class Foo:
        def bar(self, *args):
            """Foo.bar(self, int ncall_me =10000, [resume=True, int nsplit=1])"""
            pass

        def baz(self, *args):
            """Foo.baz(self: Foo, ncall_me: int =10000, arg: np.ndarray = [])"""

    assert describe(Foo().bar) == ["ncall_me", "resume", "nsplit"]
    assert describe(Foo().baz) == ["ncall_me", "arg"]


def test_from_docstring_3():
    assert describe(min) == ["iterable", "default", "key"]


def test_from_docstring_4():
    def f(*args):
        """f(a=(), b=[], *args, *[, foo=1])"""

    assert describe(f) == ["a", "b"]


def test_from_bad_docstring_2():
    def foo(*args):
        """foo is some function"""
        pass

    assert describe(foo) == []


def test_with_type_hints():
    def foo(
        x: NDArray,
        a: Annotated[float, Gt(0), Lt(1)],
        b: float,
        c: Annotated[float, 0:],
        d: Annotated[float, Ge(1)],
        e: Annotated[float, Le(2)],
        f: Annotated[float, Interval(gt=2, lt=3)],
    ): ...

    r = describe(foo, annotations=True)
    assert r == {
        "x": None,
        "a": (0, 1),
        "b": None,
        "c": (0, np.inf),
        "d": (1, np.inf),
        "e": (-np.inf, 2),
        "f": (2, 3),
    }

    class Foo:
        def __call__(self, x: NDArray, a: Annotated[float, Gt(0), Lt(1)], b: float): ...

    r = describe(Foo.__call__, annotations=True)
    assert r == {"self": None, "x": None, "a": (0, 1), "b": None}

    r = describe(Foo(), annotations=True)
    assert r == {"x": None, "a": (0, 1), "b": None}


def test_with_pydantic_types():
    tp = pytest.importorskip("pydantic.types")

    def foo(
        x: NDArray,
        a: tp.PositiveFloat,
        b: tp.NonNegativeFloat,
        c: float,
        d: Annotated[float, tp.annotated_types.Gt(1)],
        e: Annotated[float, tp.annotated_types.Interval(gt=0, lt=1)],
    ): ...

    r = describe(foo, annotations=True)
    assert r == {
        "x": None,
        "a": (0, np.inf),
        "b": (0, np.inf),
        "c": None,
        "d": (1, np.inf),
        "e": (0, 1),
    }


def test_with_annotated_types():
    tp = pytest.importorskip("annotated_types")

    def foo(
        x: NDArray,
        a: float,
        b: Annotated[float, tp.Gt(1)],
        c: Annotated[float, tp.Interval(gt=0, lt=1)],
    ): ...

    r = describe(foo, annotations=True)
    assert r == {
        "x": None,
        "a": None,
        "b": (1, np.inf),
        "c": (0, 1),
    }


class Foo(float):
    pass


def test_string_annotation_1():
    def f(x, mu: "Foo"):
        pass

    assert describe(f, annotations=True) == {"x": None, "mu": None}


def test_string_annotation_2():
    def f(x, mu: "Annotated[float, Gt(1)]"):
        pass

    assert describe(f, annotations=True) == {"x": None, "mu": (1, np.inf)}


def test_string_annotation_3():
    def f(x, mu: "Bar"):  # noqa
        pass

    assert describe(f, annotations=True) == {"x": None, "mu": None}
