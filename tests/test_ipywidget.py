import pytest
from iminuit import Minuit
from numpy.testing import assert_allclose
import contextlib

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
ipywidgets = pytest.importorskip("ipywidgets")

mpl.use("Agg")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_ipywidgets(mock_ipython):
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
def test_interactive_ipywidgets_raises(mock_ipython):
    def raiser(args):
        raise ValueError

    m = Minuit(lambda x, y: 0, 0, 1)

    # by default do not raise
    m.interactive(raiser)

    with pytest.raises(ValueError):
        m.interactive(raiser, raise_on_exception=True)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_interactive_ipywidgets_with_array_func(mock_ipython):
    def cost(par):
        return par[0] ** 2 + (par[1] / 2) ** 2

    class TraceArgs:
        nargs = 0

        def __call__(self, par):
            self.nargs = len(par)

    trace_args = TraceArgs()
    m = Minuit(cost, (1, 2))

    m.interactive(trace_args)
    assert trace_args.nargs > 0
