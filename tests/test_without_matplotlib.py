from iminuit import cost
from iminuit._hide_modules import hide_modules
import pytest

pytest.importorskip("matplotlib.pyplot")


def test_visualize():
    import iminuit

    c = cost.LeastSquares([1, 2], [3, 4], 1, lambda x, a: a * x)

    s = iminuit.Minuit(c, 1).migrad()._repr_html_()
    assert "<svg" in s

    with hide_modules("matplotlib", reload="iminuit"):
        s = iminuit.Minuit(c, 1).migrad()._repr_html_()
        assert "<svg" not in s
