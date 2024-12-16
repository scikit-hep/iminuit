import pytest
from iminuit import Minuit
from numpy.testing import assert_allclose
import contextlib

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
PyQt6 = pytest.importorskip("PyQt6")

mpl.use("Agg")


def qtinteractive(m, plot=None, raise_on_exception=False, **kwargs):
    from iminuit.qtwidget import make_widget

    return make_widget(m, plot, kwargs, raise_on_exception, run_event_loop=False)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_pyqt6(qtbot):
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

    with plot.assert_call():
        mw1 = qtinteractive(m, plot)
    qtbot.addWidget(mw1)
    assert isinstance(mw1, PyQt6.QtWidgets.QWidget)

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

    class Cost:
        def visualize(self, args):
            return plot(args)

        def __call__(self, a, b):
            return (a - 100) ** 2 + (b + 100) ** 2

    c = Cost()
    m = Minuit(c, 0, 0)
    with plot.assert_call():
        mw = qtinteractive(m, raise_on_exception=True)
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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_pyqt6_raises(qtbot):
    def raiser(args):
        raise ValueError

    m = Minuit(lambda x, y: 0, 0, 1)

    # by default do not raise
    qtinteractive(m, raiser)

    with pytest.raises(ValueError):
        qtinteractive(m, raiser, raise_on_exception=True)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_pyqt6_with_array_func(qtbot):
    def cost(par):
        return par[0] ** 2 + (par[1] / 2) ** 2

    class TraceArgs:
        nargs = 0

        def __call__(self, par):
            self.nargs = len(par)

    trace_args = TraceArgs()
    m = Minuit(cost, (1, 2))

    qtinteractive(m, trace_args)
    assert trace_args.nargs > 0
