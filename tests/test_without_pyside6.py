from iminuit._hide_modules import hide_modules
from iminuit.cost import LeastSquares
import pytest

pytest.importorskip("matplotlib")

# matplotlib probes these in turn and only fails once none of them can be imported
QT_BINDINGS = ("PySide6", "PySide2", "PyQt5", "PyQt6")

# forget what matplotlib cached about the Qt binding it found earlier
QT_MODULES = (
    "iminuit.qtwidget",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.qt_compat",
)


def test_pyside6_interactive_with_ipython():
    pytest.importorskip("IPython")
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with hide_modules("PySide6", reload="iminuit.qtwidget"):
        with pytest.raises(ImportError, match="Please install PySide6"):
            iminuit.Minuit(cost, 1).interactive()


def test_pyside6_interactive_without_ipython():
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with hide_modules("PySide6", "IPython", reload="iminuit.qtwidget"):
        with pytest.raises(ImportError, match="Please install PySide6"):
            iminuit.Minuit(cost, 1).interactive()


def test_interactive_without_any_qt_binding():
    # matplotlib raises a plain ImportError here, not ModuleNotFoundError,
    # so the hint must still be added.
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with hide_modules(*QT_BINDINGS, "IPython", reload=QT_MODULES):
        with pytest.raises(ImportError, match="Please install PySide6"):
            iminuit.Minuit(cost, 1).interactive()
