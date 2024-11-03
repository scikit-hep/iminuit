import pytest
from iminuit import Minuit
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import contextlib
from unittest.mock import patch, MagicMock

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


@pytest.mark.parametrize("arg", ("x", 1))
def test_profile_1(fig, minuit, arg):
    minuit.draw_profile(arg)
    plt.ylim(0, 5)


def test_profile_2(fig, minuit):
    minuit.draw_profile("x", grid=np.linspace(0, 5))


@pytest.mark.parametrize("arg", ("x", 1))
def test_mnprofile_1(fig, minuit, arg):
    # plots with hesse errors
    minuit.draw_mnprofile(arg)
    plt.ylim(0, 5)


def test_mnprofile_2(fig, minuit):
    minuit.minos()
    minuit.draw_mnprofile("x", grid=np.linspace(0, 5))


def test_mncontour_1(fig, minuit):
    minuit.draw_mncontour("x", "y")


def test_mncontour_2(fig, minuit):
    # use 0, 1 instead of "x", "y"
    minuit.draw_mncontour(0, 1, cl=0.68)


def test_mncontour_3(fig, minuit):
    minuit.draw_mncontour("x", "y", cl=[0.68, 0.9])


def test_mncontour_4(fig, minuit):
    minuit.draw_mncontour("x", "y", size=20, interpolated=200)


def test_mncontour_5(fig, minuit):
    minuit.draw_mncontour("x", "y", size=20, interpolated=10)


def test_contour_1(fig, minuit):
    minuit.draw_contour("x", "y")


def test_contour_2(fig, minuit):
    # use 0, 1 instead of "x", "y"
    minuit.draw_contour(0, 1, size=20, bound=2)


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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_ipywidgets():
    ipywidgets = pytest.importorskip("ipywidgets")

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

    m = Minuit(cost, 1, 1)

    with patch("IPython.get_ipython") as mock_get_ipython:
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"
        mock_shell.config = {"IPKernelApp": True}
        mock_get_ipython.return_value = mock_shell

        with pytest.raises(AttributeError, match="no visualize method"):
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
            # because of implementation details, we have to trigger the slider several times
            for i in range(5):
                parameters.children[0].slider.value = i  # change first slider
        parameters.children[0].fix.value = True
        with plot.assert_call():
            parameters.children[0].fit.value = True

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
        assert parameters.children[0].slider.max == 1
        assert parameters.children[1].slider.min == -1
        with plot.assert_call():
            fit_button.click()
        assert_allclose(m.values, (100, -100), atol=1e-5)
        # this should trigger an exception
        plot.raises = True
        with plot.assert_call():
            fit_button.click()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_pyqt6(qtbot):
    PyQt6 = pytest.importorskip("PyQt6")

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

    m = Minuit(cost, 1, 1)

    with pytest.raises(AttributeError, match="no visualize method"):
        mw = m.interactive(raise_on_exception=True, qt_exec=False)
        qtbot.addWidget(mw)
        mw.close()
        mw.deleteLater()

    with plot.assert_call():
        mw1 = m.interactive(plot, qt_exec=False)
    qtbot.addWidget(mw1)
    assert isinstance(mw1, PyQt6.QtWidgets.QMainWindow)

    # manipulate state to also check this code
    with plot.assert_call():
        mw1.fit_button.click()
    assert_allclose(m.values, (0, 0), atol=1e-5)
    with plot.assert_call():
        mw1.reset_button.click()
    assert_allclose(m.values, (1, 1), atol=1e-5)

    mw1.algo_choice.setCurrentText("Scipy")
    with plot.assert_call():
        mw1.fit_button.click()

    mw1.algo_choice.setCurrentText("Simplex")
    with plot.assert_call():
        mw1.fit_button.click()

    mw1.update_button.click()
    with plot.assert_call():
        mw1.parameters[0].slider.valueChanged.emit(int(5e7))
    mw1.parameters[0].fix.click()
    with plot.assert_call():
        mw1.parameters[0].fit.click()

    mw1.close()
    mw1.deleteLater()

    class Cost:
        def visualize(self, args):
            return plot(args)

        def __call__(self, a, b):
            return (a - 100) ** 2 + (b + 100) ** 2

    c = Cost()
    m = Minuit(c, 0, 0)
    with plot.assert_call():
        mw = m.interactive(raise_on_exception=True, qt_exec=False)
    qtbot.addWidget(mw)

    # this should modify slider range
    assert mw.parameters[0].vmax == 1
    assert mw.parameters[1].vmin == -1
    with plot.assert_call():
        mw.fit_button.click()
    assert_allclose(m.values, (100, -100), atol=1e-5)
    # this should trigger an exception
    # plot.raises = True
    # with plot.assert_call():
    #    mw.fit_button.click()
    mw.close()
    mw.deleteLater()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_ipywidgets_raises():
    pytest.importorskip("ipywidgets")

    def raiser(args):
        raise ValueError

    m = Minuit(lambda x, y: 0, 0, 1)

    # by default do not raise
    with patch("IPython.get_ipython") as mock_get_ipython:
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"
        mock_shell.config = {"IPKernelApp": True}
        mock_get_ipython.return_value = mock_shell

        m.interactive(raiser)

        with pytest.raises(ValueError):
            m.interactive(raiser, raise_on_exception=True)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_pyqt6_raises(qtbot):
    pytest.importorskip("PyQt6")

    def raiser(args):
        raise ValueError

    m = Minuit(lambda x, y: 0, 0, 1)

    # by default do not raise
    mw = m.interactive(raiser, qt_exec=False)
    qtbot.addWidget(mw)
    mw.close()
    mw.deleteLater()

    with pytest.raises(ValueError):
        mw = m.interactive(raiser, raise_on_exception=True, qt_exec=False)
        qtbot.addWidget(mw)
        mw.close()
        mw.deleteLater()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_ipywidgets_with_array_func():
    pytest.importorskip("ipywidgets")

    def cost(par):
        return par[0] ** 2 + (par[1] / 2) ** 2

    class TraceArgs:
        nargs = 0

        def __call__(self, par):
            self.nargs = len(par)

    trace_args = TraceArgs()
    m = Minuit(cost, (1, 2))

    with patch("IPython.get_ipython") as mock_get_ipython:
        mock_shell = MagicMock()
        mock_shell.__class__.__name__ = "ZMQInteractiveShell"
        mock_shell.config = {"IPKernelApp": True}
        mock_get_ipython.return_value = mock_shell

        m.interactive(trace_args)
        assert trace_args.nargs > 0


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_pyqt6_with_array_func(qtbot):
    pytest.importorskip("PyQt6")

    def cost(par):
        return par[0] ** 2 + (par[1] / 2) ** 2

    class TraceArgs:
        nargs = 0

        def __call__(self, par):
            self.nargs = len(par)

    trace_args = TraceArgs()
    m = Minuit(cost, (1, 2))

    mw = m.interactive(trace_args, qt_exec=False)
    qtbot.addWidget(mw)
    assert trace_args.nargs > 0
    mw.close()
    mw.deleteLater()
