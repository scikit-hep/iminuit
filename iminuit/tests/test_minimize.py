from __future__ import (absolute_import, division, print_function)
from iminuit import minimize
from iminuit.tests.utils import assert_allclose
import numpy as np

def func(x):
    return 1 + x[0] ** 2 + (x[1] - 1) ** 2 + (x[2] - 2) ** 2

def grad(x):
    return np.array((2 * x[0], 2 * (x[1] - 1), 2 * (x[2] - 2)))


def test_simple():
    result = minimize(func, np.ones(3))
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert(result.nfev > 0)
    assert(result.njev == 0)


def test_gradient():
    result = minimize(func, np.ones(3), jac=grad)
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert(result.nfev > 0)
    assert(result.njev > 0)
