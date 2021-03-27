from iminuit.util import describe, make_func_code
from math import ldexp
import platform
from functools import wraps

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

    assert describe(Foo().bar) == ["ncall_me", "resume", "nsplit"]


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
