import pytest
from iminuit import Minuit
from numpy.testing import assert_allclose
import contextlib

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
PySide6 = pytest.importorskip("PySide6")

mpl.use("Agg")


def qtinteractive(qtbot, m, plot=None, raise_on_exception=False, **kwargs):
    from iminuit.qtwidget import make_widget

    w = make_widget(m, plot, kwargs, raise_on_exception, run_event_loop=False)
    qtbot.addWidget(w)
    return w


def test_interactive_pyside6(qtbot):
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
        mw1 = qtinteractive(qtbot, m, plot)
    assert isinstance(mw1, PySide6.QtWidgets.QWidget)

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

    # check changing of limits
    m = Minuit(cost, 0, 0)
    m.limits["a"] = (-2, 2)
    mw2 = qtinteractive(qtbot, m, plot)
    mw2.parameters[0].tmin.setValue(-1)
    mw2.parameters[0].tmax.setValue(1)
    assert_allclose(m.limits["a"], (-1, 1), atol=1e-5)
    with plot.assert_call():
        mw2.parameters[0].tmin.setValue(0.5)
    assert_allclose(m.limits["a"], (0.5, 1), atol=1e-5)
    assert_allclose(m.values, (0.5, 0), atol=1e-5)
    mw2.parameters[0].tmin.setValue(2)
    assert_allclose(m.limits["a"], (0.5, 1), atol=1e-5)
    assert_allclose(m.values, (0.5, 0), atol=1e-5)
    mw2.parameters[0].tmin.setValue(-1)
    with plot.assert_call():
        mw2.parameters[0].tmax.setValue(0)
    assert_allclose(m.limits["a"], (-1, 0), atol=1e-5)
    assert_allclose(m.values, (0, 0), atol=1e-5)
    mw2.parameters[0].tmax.setValue(-2)
    assert_allclose(m.limits["a"], (-1, 0), atol=1e-5)
    assert_allclose(m.values, (0, 0), atol=1e-5)

    class Cost:
        def visualize(self, args):
            return plot(args)

        def __call__(self, a, b):
            return (a - 100) ** 2 + (b + 100) ** 2

    c = Cost()
    m = Minuit(c, 0, 0)
    with plot.assert_call():
        mw = qtinteractive(qtbot, m, raise_on_exception=True)

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


def test_interactive_pyside6_raises(qtbot):
    def raiser(args):
        raise ValueError

    m = Minuit(lambda x, y: 0, 0, 1)

    # by default do not raise
    qtinteractive(qtbot, m, raiser)

    with pytest.raises(ValueError):
        qtinteractive(qtbot, m, raiser, raise_on_exception=True)


def test_interactive_pyside6_with_array_func(qtbot):
    def cost(par):
        return par[0] ** 2 + (par[1] / 2) ** 2

    class TraceArgs:
        nargs = 0

        def __call__(self, par):
            self.nargs = len(par)

    trace_args = TraceArgs()
    m = Minuit(cost, (1, 2))

    qtinteractive(qtbot, m, trace_args)
    assert trace_args.nargs > 0


def test_interactive_pyside6_fix_fit_exclusive(qtbot):
    def cost(a, b):
        return a**2 + b**2

    def plot(args):
        pass

    m = Minuit(cost, 1, 1)
    mw = qtinteractive(qtbot, m, plot)
    p = mw.parameters[0]

    p.fix.click()
    assert list(m.fixed) == [True, False]

    # Fit toggle unchecks Fix, this must release the parameter
    p.fit.click()
    assert p.fix.isChecked() is False
    assert list(m.fixed) == [False, False]
    assert p.slider.isEnabled() is False

    # Fix toggle unchecks Fit, this must re-enable the slider
    p.fix.click()
    assert p.fit.isChecked() is False
    assert p.slider.isEnabled() is True
    assert list(m.fixed) == [True, False]


def test_interactive_pyside6_fixed_restored_on_error(qtbot):
    class Cost:
        raises = False

        def visualize(self, args):
            pass

        def __call__(self, a, b):
            if self.raises:
                raise ValueError("foo")
            return a**2 + b**2

    c = Cost()
    m = Minuit(c, 1, 1)
    mw = qtinteractive(qtbot, m, raise_on_exception=True)

    c.raises = True
    with qtbot.capture_exceptions() as exceptions:
        mw.parameters[1].fit.click()
    assert len(exceptions) == 1
    assert exceptions[0][0] is ValueError
    assert list(m.fixed) == [False, False]


def test_interactive_pyside6_reset_fixed(qtbot):
    def cost(a, b):
        return a**2 + b**2

    def plot(args):
        pass

    m = Minuit(cost, 1, 1)
    mw = qtinteractive(qtbot, m, plot)
    p0, p1 = mw.parameters

    # before any fit
    p0.fix.click()
    assert list(m.fixed) == [True, False]
    mw.reset_button.click()
    assert list(m.fixed) == [False, False]
    assert p0.fix.isChecked() is False

    # after a fit
    p0.fix.click()
    mw.fit_button.click()
    assert_allclose(m.values, (1, 0), atol=1e-5)
    mw.reset_button.click()
    assert list(m.fixed) == [False, False]
    assert p0.fix.isChecked() is False
    assert_allclose(m.values, (1, 1), atol=1e-5)

    # fit toggle must not survive reset and must not trigger a fit
    p1.fit.click()
    assert_allclose(m.values[1], 0, atol=1e-5)
    mw.reset_button.click()
    assert p1.fit.isChecked() is False
    assert p1.slider.isEnabled() is True
    assert_allclose(m.values, (1, 1), atol=1e-5)
