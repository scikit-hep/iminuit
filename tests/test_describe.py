from iminuit.util import describe, make_func_code, _arguments_from_docstring
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

    assert describe(f) is None


def test_generic_lambda():
    assert describe(lambda *args: 0) is None


def test_generic_class_method():
    class A:
        def f(self, *args):
            pass

    assert describe(A().f) is None


def test_generic_functor():
    class A:
        def __call__(self, *args):
            pass

    assert describe(A()) is None


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


def test_arguments_from_docstring():
    s = "f(x, y, z)"
    args = _arguments_from_docstring(s)
    assert args == ["x", "y", "z"]
    # this is a hard one
    s = "Minuit.migrad( int ncall_me =10000, [resume=True, int nsplit=1])"
    args = _arguments_from_docstring(s)
    assert args == ["ncall_me", "resume", "nsplit"]