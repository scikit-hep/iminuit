from iminuit import Minuit
from iminuit.testing import (
    rosenbrock,
    rosenbrock_grad,
    ackley,
    beale,
    matyas,
    sphere_np,
)
import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize("grad", (None, rosenbrock_grad))
def test_rosenbrock(grad):
    m = Minuit(rosenbrock, x=0, y=0, grad=grad)
    m.tol = 1e-4
    m.migrad()

    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.values["x"], 1.0, atol=1e-3)
    assert_allclose(m.values["y"], 1.0, atol=1e-3)


def test_ackley():
    m = Minuit(ackley, x=0.3, y=-0.2)
    # m.errors = 1.7
    m.tol = 1e-4
    m.migrad()

    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.values, [0, 0], atol=1e-6)


def test_beale():
    m = Minuit(beale, x=0.5, y=0.25)
    m.tol = 1e-4
    m.migrad()

    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.values, [3, 0.5], atol=1e-3)


def test_matyas():
    m = Minuit(matyas, x=0.5, y=0.5)
    m.tol = 1e-4
    m.migrad()

    assert_allclose(m.fval, 0, atol=1e-14)
    assert_allclose(m.values, [0, 0], atol=1e-14)


def test_matyas_oneside():
    """One-sided limit when the minimum is in the forbidden region."""
    m = Minuit(matyas, x=2.5, y=0.5)
    m.tol = 1e-4
    m.limits["x"] = (1, None)
    m.migrad()

    assert_allclose(m.values, [1, 0.923], atol=1e-3)


@pytest.mark.parametrize("start", (2, (2, 2, 2)))
def test_sphere_np(start):
    m = Minuit(sphere_np, start)
    m.migrad()

    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.values, np.zeros_like(start), atol=1e-3)
