import pytest
from iminuit import Minuit
from pathlib import Path
import numpy as np


mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
mpl.use("Agg")


def f1(x, y):
    return (1 - x) ** 2 + np.exp((y - 1) ** 2)


@pytest.fixture
def minuit():
    m = Minuit(f1, x=0, y=0)
    m.migrad()
    return m


@pytest.fixture
def fig(request):
    fig = plt.figure()
    yield fig
    p = Path(__file__).parent / "fig"
    if not p.exists():
        p.mkdir()
    fig.savefig(p / (request.node.name + ".svg"))
    del fig


@pytest.mark.parametrize("arg", ("x", "y"))
def test_profile_1(fig, minuit, arg):
    # plots with hesse errors
    minuit.draw_profile(arg)
    plt.ylim(0, 5)


@pytest.mark.parametrize("arg", ("x", "y"))
def test_profile_2(fig, minuit, arg):
    # plots with minos errors
    minuit.draw_profile(arg)
    plt.ylim(0, 5)


def test_profile_3(fig, minuit):
    minuit.draw_profile("x", grid=np.linspace(0, 5))


@pytest.mark.parametrize("arg", ("x", "y"))
def test_mnprofile_1(fig, minuit, arg):
    # plots with hesse errors
    minuit.draw_mnprofile(arg)
    plt.ylim(0, 5)


@pytest.mark.parametrize("arg", ("x", "y"))
def test_mnprofile_2(fig, minuit, arg):
    # plots with minos errors
    minuit.minos()
    minuit.draw_mnprofile(arg)
    plt.ylim(0, 5)


def test_mnprofile_3(fig, minuit):
    minuit.minos()
    minuit.draw_mnprofile("x", grid=np.linspace(0, 5))


def test_mncontour_1(fig, minuit):
    minuit.draw_mncontour("x", "y")


def test_mncontour_2(fig, minuit):
    minuit.draw_mncontour("x", "y", cl=0.68)


def test_mncontour_3(fig, minuit):
    minuit.draw_mncontour("x", "y", cl=[0.68, 0.9])


def test_contour_1(fig, minuit):
    minuit.draw_contour("x", "y")


def test_contour_2(fig, minuit):
    minuit.draw_contour("x", "y", size=20, bound=2)


def test_contour_3(fig, minuit):
    minuit.draw_contour("x", "y", size=100, bound=((-0.5, 2.5), (-1, 3)))


def test_contour_4(fig, minuit):
    minuit.draw_contour("x", "y", size=(10, 50), bound=((-0.5, 2.5), (-1, 3)))


def test_contour_5(fig, minuit):
    minuit.draw_contour("x", "y", grid=(np.linspace(-0.5, 2.5), np.linspace(-1, 3)))
