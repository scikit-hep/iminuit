from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
from iminuit import Minuit
from iminuit.tests.utils import assert_allclose, requires_dependency

try:
    import matplotlib as mpl

    mpl.use('Agg')
except ImportError:
    pass


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


@pytest.fixture
def m():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1)
    m.tol = 1e-4
    m.migrad()
    assert_allclose(m.fval, 0, atol=1e-6)
    assert_allclose(m.values['x'], 1, atol=1e-3)
    assert_allclose(m.values['y'], 1, atol=1e-3)
    return m


@requires_dependency('matplotlib')
def test_profile(m):
    m.minos('x')
    m.draw_profile('x')


@requires_dependency('matplotlib')
def test_mnprofile(m):
    m.minos('x')
    m.draw_mnprofile('x')


@requires_dependency('matplotlib')
def test_mncontour(m):
    m.minos()
    m.draw_mncontour('x', 'y')


@requires_dependency('matplotlib')
def test_drawcontour(m):
    m.minos()
    m.draw_contour('x', 'y')


@requires_dependency('matplotlib')
def test_drawcontour_show_sigma(m):
    m.minos()
    m.draw_contour('x', 'y', show_sigma=True)
