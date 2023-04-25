from iminuit._hide_modules import hide_modules
import pytest
from iminuit.warnings import OptionalDependencyWarning


def cost(x, y):
    return x**2 + y**2


def test_mncontour_cl():
    with hide_modules("scipy", reload="iminuit"):
        import iminuit

        m = iminuit.Minuit(cost, 1, 1)
        m.migrad()
        m.mncontour("x", "y")
        for cl in (0.68, 0.9, 0.95, 0.99, 1, 2, 3, 4, 5):
            m.mncontour("x", "y", cl=cl)

        with pytest.raises(ImportError):
            m.mncontour("x", "y", cl=0.1)


def test_mncontour_interpolated():
    with hide_modules("scipy", reload="iminuit"):
        import iminuit

        with pytest.warns(
            OptionalDependencyWarning,
            match="interpolation requires optional package 'scipy'",
        ):
            m = iminuit.Minuit(cost, 1, 1)
            m.migrad()
            pts = m.mncontour("x", "y", size=20, interpolated=200)
            assert len(pts) == 21


def test_minos_cl():
    with hide_modules("scipy"):
        import iminuit

        m = iminuit.Minuit(cost, 1, 1)
        m.migrad()
        m.minos()
        for cl in (0.68, 0.9, 0.95, 0.99, 1, 2, 3, 4, 5):
            m.minos(cl=cl)

        with pytest.raises(ModuleNotFoundError):
            m.minos(cl=0.1)


def test_missing_scipy():
    with hide_modules("scipy"):
        import iminuit

        m = iminuit.Minuit(cost, 1, 1)
        with pytest.raises(ModuleNotFoundError):
            m.scipy()
