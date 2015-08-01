from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit import Minuit
from nose.tools import (assert_almost_equal,
                        assert_less
                        )

import matplotlib as mpl

mpl.use('Agg')


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def test_mnprofile():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1)
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)
    m.minos('x')
    m.draw_mnprofile('x')


def test_mncontour():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1)
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)
    m.minos()
    m.draw_mncontour('x', 'y')


def test_drawcontour():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1)
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)
    m.minos()
    m.draw_contour('x', 'y')


def test_drawcontour_show_sigma():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1)
    m.tol = 1e-4
    m.migrad()
    assert_less(m.fval, 1e-6)
    assert_almost_equal(m.values['x'], 1., places=3)
    assert_almost_equal(m.values['y'], 1., places=3)
    m.minos()
    m.draw_contour('x', 'y', show_sigma=True)
