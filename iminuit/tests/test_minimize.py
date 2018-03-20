from __future__ import (absolute_import, division, print_function)
from iminuit import minimize
from iminuit.tests.utils import assert_allclose
import numpy as np


def func(x, *args):
    c = args[0] if args else 1
    return c + x[0] ** 2 + (x[1] - 1) ** 2 + (x[2] - 2) ** 2

def grad(x, *args):
    return np.array((2 * x[0], 2 * (x[1] - 1), 2 * (x[2] - 2)))


def test_simple():
    result = minimize(func, (1, 1, 1))
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert result.nfev > 0
    assert result.njev == 0


def test_args():
    result = minimize(func, np.ones(3), args=(5,))
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 5)
    assert result.nfev > 0
    assert result.njev == 0


def test_gradient():
    result = minimize(func, np.ones(3), jac=grad)
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert result.nfev > 0
    assert result.njev > 0


def test_callback():
    class Callback:
        trace = []
        def __call__(self, x):
            self.trace.append(x)
    callback = Callback()
    result = minimize(func, np.ones(3),
                      callback=callback)
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert result.nfev == len(callback.trace)
    assert_allclose(callback.trace[0], np.ones(3), atol=1e-2)
    assert_allclose(callback.trace[-1], result.x, atol=1e-2)
