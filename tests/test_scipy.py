import pytest
from numpy.testing import assert_allclose
from iminuit import Minuit
from iminuit.testing import rosenbrock, rosenbrock_grad
import numpy as np

scopt = pytest.importorskip("scipy.optimize")


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


def grad(a, b):
    return 2 * a, b - 1


def hess(par):
    return [[2, 0], [0, 1]]


@pytest.mark.parametrize("stra", (0, 1))
@pytest.mark.parametrize("grad", (None, grad))
def test_scipy_unbounded(stra, grad):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.strategy = stra
    m.scipy()
    assert m.valid
    assert m.accurate == (stra == 1)
    assert_allclose(m.values, [0, 1], atol=1e-3)
    if stra == 1:
        assert_allclose(m.errors, [1, 2], atol=3e-2)
    if grad:
        assert m.fmin.ngrad > 0
    else:
        assert m.fmin.ngrad == 0


@pytest.mark.parametrize("stra", (0, 1))
@pytest.mark.parametrize("grad", (None, grad))
@pytest.mark.parametrize(
    "lower,upper",
    (
        (-0.1, None),
        (0, None),
        (0.1, None),
        (None, -0.1),
        (None, 0),
        (None, 0.1),
        (-0.1, 0.1),
    ),
)
def test_scipy_bounded(stra, grad, lower, upper):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.limits["a"] = (lower, upper)
    m.strategy = stra
    m.scipy()
    if stra == 1:
        assert m.valid
        assert m.accurate
    lower = -np.inf if lower is None else lower
    upper = np.inf if upper is None else upper
    assert_allclose(m.values, [np.clip(0, lower, upper), 1], atol=1e-3)
    if stra == 1:
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


@pytest.mark.parametrize("stra", (0, 1))
@pytest.mark.parametrize("grad", (None, grad))
def test_scipy_errordef(stra, grad):
    m = Minuit(fcn, a=1, b=2, grad=grad)
    m.errordef = 4
    m.strategy = stra
    m.scipy()
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=1e-3)
    assert_allclose(m.errors, [2, 4], rtol=0.3)
    if grad:
        assert m.fmin.ngrad > 0
    else:
        assert m.fmin.ngrad == 0


@pytest.mark.parametrize("stra", (0, 1))
@pytest.mark.parametrize("grad", (None, rosenbrock_grad))
def test_scipy_ncall(stra, grad):
    m = Minuit(rosenbrock, x=2, y=2, grad=grad)
    m.strategy = stra
    m.scipy()
    assert m.valid
    nfcn = m.fmin.nfcn
    m.reset()
    m.scipy(ncall=1)
    assert m.fmin.nfcn < nfcn
    assert not m.valid


@pytest.mark.parametrize(
    "method",
    (
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ),
)
def test_scipy_method(method):
    gr = None
    he = None
    if method in ("dogleg", "trust-ncg", "trust-exact", "trust-krylov"):
        gr = grad
        he = hess
    if method in ("Newton-CG", "trust-constr"):
        gr = grad
    m = Minuit(fcn, a=1, b=2, grad=gr)
    m.scipy(method=method, hess=he)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=1e-3)
    assert_allclose(m.errors, [1, 2], rtol=1e-2)


@pytest.mark.parametrize("ypos", (1, 1.1))
def test_scipy_constraints(ypos):
    def fcn(x, y):
        return x ** 2 + y ** 2

    m = Minuit(fcn, x=1, y=2)
    m.errordef = 1
    con = scopt.NonlinearConstraint(lambda x: x[0] ** 2 + (x[1] - ypos) ** 2, 0, 1)
    m.scipy(constraints=[con])
    if ypos == 1:
        assert m.valid
    assert_allclose(m.values, [0, ypos - 1], atol=1e-3)
    assert m.accurate
