from iminuit._hide_modules import hide_modules
from iminuit.cost import LeastSquares
import pytest

pytest.importorskip("ipywidgets")


def test_interactive(mock_ipython):
    pytest.importorskip("matplotlib")
    import iminuit

    cost = LeastSquares([1.1, 2.2], [3.3, 4.4], 1, lambda x, a: a * x)

    iminuit.Minuit(cost, 1).interactive()

    with hide_modules("ipywidgets", reload="iminuit.ipywidget"):
        with pytest.raises(ModuleNotFoundError, match="Please install ipywidgets"):
            iminuit.Minuit(cost, 1).interactive()
