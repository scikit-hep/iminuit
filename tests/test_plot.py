import pytest
from iminuit import Minuit

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def f2(par):
    return f1(par[0], par[1])


@pytest.fixture(params=("normal", "numpy"))
def minuit(request):
    if request.param == "normal":
        m = Minuit(f1, x=0, y=0, pedantic=False)
    else:
        m = Minuit.from_array_func(f2, (0, 0), name=("x", "y"), pedantic=False)
    m.migrad()
    return m


def test_profile(minuit):
    minuit.minos("x")
    minuit.draw_profile("x")


def test_mnprofile(minuit):
    minuit.minos("x")
    minuit.draw_mnprofile("x")


def test_mncontour(minuit):
    minuit.minos()
    minuit.draw_mncontour("x", "y")


def test_drawcontour(minuit):
    minuit.minos()
    minuit.draw_contour("x", "y")
    minuit.draw_contour("x", "x", bins=20, bound=2)
    minuit.draw_contour("x", "x", bins=20, bound=((-10, 10), (-10, 10)))
    with pytest.warns(DeprecationWarning):
        minuit.draw_contour("x", "y", show_sigma=True)
