from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from iminuit import Minuit
from iminuit.tests.utils import assert_allclose, requires_dependency

try:
    import matplotlib as mpl

    mpl.use("Agg")
except ImportError:
    pass


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def f2(par):
    return f1(par[0], par[1])


@pytest.fixture(params=("normal", "numpy"))
def m(request):
    if request.param == "normal":
        m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1)
    else:
        m = Minuit.from_array_func(
            f2, (0, 0), name=("x", "y"), pedantic=False, print_level=1
        )
    m.migrad()
    return m


@requires_dependency("matplotlib")
def test_profile(m):
    m.minos("x")
    m.draw_profile("x")


@requires_dependency("matplotlib")
def test_mnprofile(m):
    m.minos("x")
    m.draw_mnprofile("x")


@requires_dependency("matplotlib")
def test_mncontour(m):
    m.minos()
    m.draw_mncontour("x", "y")


@requires_dependency("matplotlib")
def test_drawcontour(m):
    m.minos()
    m.draw_contour("x", "y")
    m.draw_contour("x", "x", bins=20, bound=2)
    m.draw_contour("x", "x", bins=20, bound=((-10, 10), (-10, 10)))
