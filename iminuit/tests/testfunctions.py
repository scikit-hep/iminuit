import warnings
import random
from math import sqrt, exp, sin, cos, pi, e

from nose.tools import (raises, assert_equal, assert_true, assert_false,
    assert_almost_equal)
from iminuit import Minuit

from testiminuit import assert_array_almost_equal

random.seed(0.258)

def rosenbrok(x, y):
    '''Rosenbrok function, minimum at (1,1) with value of 0'''
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def test_rosekbrok():
    m = Minuit(rosenbrok, x=0, y=0, pedantic=False, print_level=0)
    m.migrad()
    assert m.fval < 1e-7 
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)


def ackleys(x, y):
    term1 = -20 * exp(-0.2 * sqrt(0.5*(x**2 + y**2)))
    term2 = -exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
    return term1 + term2 + 20 + e


def test_ackleys():
    random.seed(0.258)
    m = Minuit(ackleys, x=1.5 * random.random(), y=1.5 * random.random(),
               error_x=1.7, error_y=1.7,
               pedantic=False, print_level=0)
    print m.values
    m.migrad()
    print m.fval
    print m.values
    assert m.fval < 1e-5
    assert_array_almost_equal(m.args, [0, 0], decimal=3)


def beale(x, y):
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y**2
    term3 = 2.625 - x + x * y**3

    return term1 * term1 + term2 * term2 + term3 * term3


def test_beale():
    random.seed(0.258)
    m = Minuit(beale, x=random.random(), y=0.5 * random.random(),
               pedantic=False, print_level=0)

    m.migrad()

    assert_array_almost_equal(m.args, [3, 0.5], decimal=3)
    assert m.fval < 1e-6


def matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y


def test_matyas():
    random.seed(0.258)
    m = Minuit(matyas, x=random.random(), y=random.random(),
               pedantic=False, print_level=0)
    m.migrad()

    print m.fval, m.args
    assert m.fval < 1e-26
    assert_array_almost_equal(m.args, [0, 0], decimal=12)


def test_matyas_oneside():
    '''One-side limit when the minimum is in the forbidden region'''
    random.seed(0.258)
    m = Minuit(matyas, x=2 + random.random(), y=random.random(),
               limit_x = (1, None),
               pedantic=False, print_level=0)

    m.migrad()
    assert_array_almost_equal(m.args, [1, 0.923], decimal=3)

