import pytest
from iminuit import Minuit

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def f2(par):
    return f1(par[0], par[1])


f1.errordef = 1
f2.errordef = 1


@pytest.fixture(params=("normal", "numpy"))
def minuit(request):
    if request.param == "normal":
        m = Minuit(f1, x=0, y=0)
    else:
        m = Minuit(f2, (0, 0), name=("x", "y"))
    m.migrad()
    return m


def test_profile(minuit):
    minuit.draw_profile("x")  # plots with hesse errors
    minuit.minos()
    minuit.draw_profile("x")  # plots with minos errors


def test_mnprofile(minuit):
    minuit.draw_mnprofile("x")  # plots with hesse errors
    minuit.minos()
    minuit.draw_mnprofile("x")  # plots with minos errors


def test_mncontour(minuit):
    minuit.draw_mncontour("x", "y")


def test_drawcontour(minuit):
    minuit.draw_contour("x", "y")
    minuit.draw_contour("x", "x", size=20, bound=2)
    minuit.draw_contour("x", "x", size=20, bound=((-10, 10), (-10, 10)))
