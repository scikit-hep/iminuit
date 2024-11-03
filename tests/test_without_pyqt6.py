from iminuit._hide_modules import hide_modules
from iminuit.cost import LeastSquares
import pytest

pytest.importorskip("PyQt6")


def test_interactive(qtbot):
    pytest.importorskip("matplotlib")
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)
    mw = iminuit.Minuit(cost, 1).interactive(qt_exec=False)
    qtbot.addWidget(mw)
    mw.close()
    mw.deleteLater()

    with hide_modules("PyQt6", reload="iminuit.qtwidget"):
        with pytest.raises(ModuleNotFoundError, match="Please install"):
            mw = iminuit.Minuit(cost, 1).interactive(qt_exec=False)
            qtbot.addWidget(mw)
            mw.close()
            mw.deleteLater()
