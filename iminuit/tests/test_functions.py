from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import random
from math import sqrt, exp, cos, pi, e
from nose.tools import assert_almost_equal, assert_less
from iminuit import Minuit
from iminuit.tests.test_iminuit import assert_array_almost_equal


def rosenbrock(x, y):
    """Rosenbrock function, minimum at (1,1) with value of 0"""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def test_rosenbrock():
    random.seed(0.258)
    m = Minuit(rosenbrock, x=0, y=0, pedantic=False, print_level=1)
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)


def ackleys(x, y):
    term1 = -20 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
    return term1 + term2 + 20 + e


def test_ackleys():
    random.seed(0.258)

    m = Minuit(ackleys, x=1.5 * random.random(), y=1.5 * random.random(),
               error_x=1.7, error_y=1.7,
               pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()

    assert_less(m.fval, 1e-6)
    assert_array_almost_equal(m.args, [0, 0], decimal=3)


def beale(x, y):
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y ** 2
    term3 = 2.625 - x + x * y ** 3

    return term1 * term1 + term2 * term2 + term3 * term3


def test_beale():
    random.seed(0.258)
    m = Minuit(beale, x=random.random(), y=0.5 * random.random(),
               pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_array_almost_equal(m.args, [3, 0.5], decimal=3)
    assert m.fval < 1e-6


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def test_matyas():
    random.seed(0.258)
    m = Minuit(matyas, x=random.random(), y=random.random(),
               pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()

    assert m.fval < 1e-26
    assert_array_almost_equal(m.args, [0, 0], decimal=12)


def test_matyas_oneside():
    """One-sided limit when the minimum is in the forbidden region"""
    random.seed(0.258)
    m = Minuit(matyas, x=2 + random.random(), y=random.random(),
               limit_x=(1, None),
               pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()
    assert_array_almost_equal(m.args, [1, 0.923], decimal=3)
