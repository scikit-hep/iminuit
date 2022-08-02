import pytest
from iminuit import Minuit
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import contextlib

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
    plt.close()


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


def test_mncontour_4(fig, minuit):
    minuit.draw_mncontour("x", "y", size=20, interpolated=200)


def test_mncontour_5(fig, minuit):
    minuit.draw_mncontour("x", "y", size=20, interpolated=10)


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


def test_mnmatrix_1(fig, minuit):
    minuit.draw_mnmatrix()


def test_mnmatrix_2(fig, minuit):
    minuit.draw_mnmatrix(cl=[0.68, 0.9])


def test_mnmatrix_3(fig):
    m = Minuit(lambda x: x**2, x=0)
    m.migrad()
    m.draw_mnmatrix()


def test_mnmatrix_4(fig, minuit):
    with pytest.raises(ValueError):
        minuit.draw_mnmatrix(cl=[])


def test_mnmatrix_5():
    m = Minuit(lambda x: x**2, x=10)
    with pytest.raises(RuntimeError, match="minimum is not valid"):
        m.draw_mnmatrix()


def test_mnmatrix_6(fig, minuit):
    minuit.fixed = True
    with pytest.raises(RuntimeError, match="all parameters are fixed"):
        minuit.draw_mnmatrix()


def test_mnmatrix_7(fig):
    # provoke an mnprofile iteration on asymmetric profile
    m = Minuit(lambda x: abs(x) ** 2 + x**4 + 10 * x, x=0)
    m.migrad()
    m.draw_mnmatrix(cl=[1, 3])


def test_interactive():
    def cost(a, b):
        return a**2 + b**2

    class Plot:
        def __init__(self):
            self.called = False
            self.raises = False

        def __call__(self, args):
            self.called = True
            if self.raises:
                raise ValueError("foo")

        @contextlib.contextmanager
        def assert_call(self):
            self.called = False
            yield
            assert self.called

    plot = Plot()

    try:
        import ipywidgets  # noqa
        import IPython  # noqa

        ipywidgets_available = True

        m = Minuit(cost, 1, 1)
        with pytest.raises(ValueError, match="no visualize method"):
            m.interactive(raise_on_exception=True)

        with plot.assert_call():
            out1 = m.interactive(plot)
        assert isinstance(out1, ipywidgets.HBox)

        # manipulate state to also check this code
        ui = out1.children[1]
        header, parameters = ui.children
        fit_button, update_button, reset_button, algo_select = header.children
        with plot.assert_call():
            fit_button.click()
        assert_allclose(m.values, (0, 0), atol=1e-5)
        with plot.assert_call():
            reset_button.click()
        assert_allclose(m.values, (1, 1), atol=1e-5)

        algo_select.value = "Scipy"
        with plot.assert_call():
            fit_button.click()

        algo_select.value = "Simplex"
        with plot.assert_call():
            fit_button.click()

        update_button.value = False
        with plot.assert_call():
            parameters.children[0].slider.value = 0.4  # change first slider
        parameters.children[0].fix.value = True
        with plot.assert_call():
            parameters.children[0].opt.value = True

        class Cost:
            def visualize(self, args):
                return plot(args)

            def __call__(self, a, b):
                return (a - 100) ** 2 + (b + 100) ** 2

        c = Cost()
        m = Minuit(c, 0, 0)
        with plot.assert_call():
            out = m.interactive(raise_on_exception=True)

        # this should modify slider range
        ui = out.children[1]
        header, parameters = ui.children
        fit_button, update_button, reset_button, algo_select = header.children
        assert parameters.children[0].slider.max < 100
        assert parameters.children[1].slider.min > -100
        with plot.assert_call():
            fit_button.click()
        assert_allclose(m.values, (100, -100), atol=1e-5)
        # this should trigger an exception
        plot.raises = True
        with plot.assert_call():
            fit_button.click()

    except ModuleNotFoundError:
        ipywidgets_available = False

    if not ipywidgets_available:
        with pytest.raises(ModuleNotFoundError, match="Please install"):
            m = Minuit(cost, 1, 1)
            m.interactive()


def test_interactive_raises():
    pytest.importorskip("ipywidgets")

    def raiser(args):
        raise ValueError

    m = Minuit(lambda x, y: 0, 0, 1)

    # by default do not raise
    m.interactive(raiser)

    with pytest.raises(ValueError):
        m.interactive(raiser, raise_on_exception=True)


def test_interactive_with_array_func():
    pytest.importorskip("ipywidgets")

    def cost(par):
        return par[0] ** 2 + (par[1] / 2) ** 2

    class TraceArgs:
        nargs = 0

        def __call__(self, par):
            self.nargs = len(par)

    trace_args = TraceArgs()
    m = Minuit(cost, (1, 2))
    m.interactive(trace_args)
    assert trace_args.nargs == 1
