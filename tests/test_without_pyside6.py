from iminuit._hide_modules import hide_modules
from iminuit.cost import LeastSquares
import pytest

pytest.importorskip("matplotlib")


def test_pyside6_interactive_with_ipython():
    pytest.importorskip("IPython")
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with hide_modules("PySide6", reload="iminuit.qtwidget"):
        with pytest.raises(ModuleNotFoundError, match="Please install PySide6"):
            iminuit.Minuit(cost, 1).interactive()


def test_pyside6_interactive_without_ipython():
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with hide_modules("PySide6", "IPython", reload="iminuit.qtwidget"):
        with pytest.raises(ModuleNotFoundError, match="Please install PySide6"):
            iminuit.Minuit(cost, 1).interactive()
