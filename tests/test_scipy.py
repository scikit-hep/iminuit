import pytest
from numpy.testing import assert_allclose
from iminuit import Minuit
from iminuit.testing import rosenbrock, rosenbrock_grad

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


def grad(a, b):
    return 2 * a, b - 1


fcn.errordef = 1


@pytest.mark.parametrize("str", (0, 1))
@pytest.mark.parametrize("grad", (None, grad))
def test_scipy_unbounded(str, grad):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.strategy = str
    m.scipy()
    assert m.valid
    assert m.accurate == (str == 1)
    assert_allclose(m.values, [0, 1], atol=1e-3)
    if str == 1:
        assert_allclose(m.errors, [1, 2], atol=3e-2)
    if grad:
        assert m.fmin.ngrad > 0
    else:
        assert m.fmin.ngrad == 0


@pytest.mark.parametrize("str", (0, 1))
@pytest.mark.parametrize("grad", (None, grad))
def test_scipy_bounded(str, grad):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.limits["a"] = (0.1, None)
    m.strategy = str
    m.scipy()
    if str == 1:
        assert m.valid
        assert m.accurate
    assert_allclose(m.values, [0.1, 1], atol=1e-3)
    if str == 1:
        assert_allclose(m.errors[1], 2, atol=3e-2)
    if grad:
        assert m.fmin.ngrad > 0
    else:
        assert m.fmin.ngrad == 0


@pytest.mark.parametrize("grad", (None, grad))
def test_scipy_fixed(grad):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.fixed["a"] = True
    m.scipy()
    assert m.valid
    assert_allclose(m.values, [1, 1], atol=1e-3)
    assert_allclose(m.errors, [0.01, 2], atol=3e-2)
    if grad:
        assert m.fmin.ngrad > 0
    else:
        assert m.fmin.ngrad == 0


@pytest.mark.parametrize("str", (0, 1))
@pytest.mark.parametrize("grad", (None, grad))
def test_scipy_errordef(str, grad):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.errordef = 4
    m.strategy = str
    m.scipy()
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=1e-3)
    assert_allclose(m.errors, [2, 4], rtol=0.3)
    if grad:
        assert m.fmin.ngrad > 0
    else:
        assert m.fmin.ngrad == 0


@pytest.mark.parametrize("str", (0, 1))
@pytest.mark.parametrize("grad", (None, rosenbrock_grad))
def test_scipy_ncall(str, grad):
    m = Minuit(rosenbrock, x=10, y=10, grad=grad)
    m.strategy = str
    m.scipy()
    print(str, grad, m)
    nfcn = m.fmin.nfcn
    m.reset()
    m.scipy(ncall=1)
    assert m.fmin.nfcn < nfcn
    assert not m.valid
    print(str, grad, m)
