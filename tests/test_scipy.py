import pytest
from numpy.testing import assert_allclose
from iminuit import Minuit

pytest.importorskip("scipy")


@pytest.fixture
def debug():
    from iminuit._core import MnPrint

    prev = MnPrint.global_level
    MnPrint.global_level = 3
    MnPrint.show_prefix_stack(True)
    yield
    MnPrint.global_level = prev
    MnPrint.show_prefix_stack(False)


def fcn(a, b):
    return a ** 2 + ((b - 1) / 2.0) ** 2 + 3


fcn.errordef = 1


def test_scipy_unbounded():

    m = Minuit(fcn, a=1, b=2)
    m.scipy()
    m.strategy = 0
    print(m.fmin)
    print(m.params)
    assert_allclose(m.values, [0, 1], atol=1e-3)
    assert_allclose(m.errors, [1, 2], atol=3e-2)


def test_scipy_bounded():

    m = Minuit(fcn, a=1, b=2)
    m.limits["a"] = (0.1, None)
    m.scipy()
    print(m.fmin)
    print(m.params)
    assert_allclose(m.values, [0.1, 1], atol=1e-3)
    assert_allclose(m.errors, [1, 2], atol=3e-2)
