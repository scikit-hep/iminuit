from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit.util import describe, make_func_code
from iminuit.tests.utils import requires_dependency
from math import ldexp
import sys
import pytest
import platform
is_pypy = platform.python_implementation() == "PyPy"


def test_function():
    def f(x, y):
        pass
    assert describe(f, True) == ['x', 'y']


def test_class_method():
    class A:
        def f(self, x, y):
            pass
    assert describe(A().f, True) == ['x', 'y']


# unbound method
def test_class_unbound_method():
    class A:
        def f(self, x, y):
            pass
    assert describe(A.f, True) == ['self', 'x', 'y']


def test_functor():
    class A:
        def __call__(self, x, y):
            pass
    assert describe(A(), True) == ['x', 'y']


@pytest.mark.skipif(
    sys.version_info >= (3, 7),
    reason='Not on Py 3.7; see Github issue 270'
)
def test_builtin_by_parsing_doc():
    assert describe(ldexp, True) == ['x', 'i']


def test_lambda():
    assert describe(lambda a, b: 0, True) == ['a', 'b']


def test_generic_function():
    def f(*args):
        pass
    with pytest.raises(TypeError):
        describe(f, True)


def test_generic_lambda():
    with pytest.raises(TypeError):
        describe(lambda *args: 0, True)


def test_generic_class_method():
    class A:
        def f(self, *args):
            pass
    with pytest.raises(TypeError):
        describe(A().f, True)


def test_generic_functor():
    class A:
        def __call__(self, *args):
            pass
    with pytest.raises(TypeError):
        describe(A(), True)


def test_generic_functor_with_fake_func():
    class A:
        def __init__(self):
            self.func_code = make_func_code(['x', 'y'])

        def __call__(self, *arg):
            pass
    assert describe(A(), True) == ['x', 'y']


@requires_dependency('Cython', 'pyximport', 'cyfunc')
def test_cython_embedsig():
    import pyximport
    pyximport.install()
    from . import cyfunc
    assert describe(cyfunc.f, True) == ['a', 'b']


@requires_dependency('Cython', 'pyximport', 'cyfunc')
@pytest.mark.skipif(
    is_pypy,
    reason='Does not work in PyPy'
)
def test_cython_class_method():
    import pyximport
    pyximport.install()
    from . import cyfunc
    cc = cyfunc.CyCallable()
    assert describe(cc.test, True) == ['c', 'd']
