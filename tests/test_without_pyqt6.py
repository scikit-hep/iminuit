from iminuit._hide_modules import hide_modules
from iminuit.cost import LeastSquares
import pytest


def test_interactive():
    pytest.importorskip("matplotlib")
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    with hide_modules("PyQt6", reload="iminuit.qtwidget"):
        with pytest.raises(ModuleNotFoundError, match="Please install PyQt6"):
            iminuit.Minuit(cost, 1).interactive()
