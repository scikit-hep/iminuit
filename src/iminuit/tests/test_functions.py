from math import sqrt, exp, cos, pi, e
from iminuit import Minuit
from numpy.testing import assert_allclose


def rosenbrock(x, y):
    """Rosenbrock function, minimum at (1, 1) with value of 0."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def test_rosenbrock():
    m = Minuit(rosenbrock, x=0, y=0, pedantic=False)
    m.tol = 1e-4
    m.migrad()
    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.values["x"], 1.0, atol=1e-3)
    assert_allclose(m.values["y"], 1.0, atol=1e-3)


def ackleys(x, y):
    term1 = -20 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
    return term1 + term2 + 20 + e


def test_ackleys():
    m = Minuit(ackleys, x=0.3, y=-0.2, error_x=1.7, error_y=1.7, pedantic=False)
    m.tol = 1e-4
    m.migrad()

    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.args, [0, 0], atol=1e-6)


def beale(x, y):
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y ** 2
    term3 = 2.625 - x + x * y ** 3
    return term1 * term1 + term2 * term2 + term3 * term3


def test_beale():
    m = Minuit(beale, x=0.5, y=0.25, pedantic=False)
    m.tol = 1e-4
    m.migrad()
    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.args, [3, 0.5], atol=1e-3)
    assert m.fval < 1e-6


def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def test_matyas():
    m = Minuit(matyas, x=0.5, y=0.5, pedantic=False)
    m.tol = 1e-4
    m.migrad()
    assert_allclose(m.fval, 0, atol=1e-14)
    assert_allclose(m.args, [0, 0], atol=1e-14)


def test_matyas_oneside():
    """One-sided limit when the minimum is in the forbidden region."""
    m = Minuit(matyas, x=2 + 0.5, y=0.5, limit_x=(1, None), pedantic=False)
    m.tol = 1e-4
    m.migrad()
    assert_allclose(m.args, [1, 0.923], atol=1e-3)
