import pytest
from numpy.testing import assert_allclose
from iminuit import Minuit
from iminuit.testing import rosenbrock, rosenbrock_grad
import numpy as np

scopt = pytest.importorskip("scipy.optimize")


def fcn(a, b):
    return a ** 2 + ((b - 1) / 2.0) ** 2 + 3


fcn.errordef = 1


def grad(a, b):
    return 2 * a, b - 1


def hess(a, b):
    return [[2, 0], [0, 0.5]]


def hessp(a, b, v):
    return np.dot(hess(a, b), v)


@pytest.mark.parametrize("array_call", (False, True))
@pytest.mark.parametrize("fixed", (False, True))
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
def test_scipy_method(array_call, fixed, method):
    fn = (lambda par: fcn(*par)) if array_call else fcn
    fn.errordef = 1

    gr = None
    he = None
    hep = None
    if method in (
        "Newton-CG",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ):
        gr = (lambda par: grad(*par)) if array_call else grad
    if method in ("Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"):
        he = (lambda par: hess(*par)) if array_call else hess
    if method in ("Newton-CG", "trust-ncg", "trust-krylov", "trust-constr"):
        hep = (lambda par, v: hessp(*par, v)) if array_call else hessp

    if array_call:
        m = Minuit(fn, (1, 2), grad=gr)
    else:
        m = Minuit(fn, a=1, b=2, grad=gr)

    m.fixed[0] = fixed

    m.scipy(method=method, hess=he)
    assert m.valid
    if fixed:
        assert_allclose(m.values, [1, 1], atol=1e-3)
        assert_allclose(m.errors[1], 2, rtol=1e-2)
    else:
        assert_allclose(m.values, [0, 1], atol=1e-3)
        assert_allclose(m.errors, [1, 2], rtol=1e-2)

    if hep:
        m.scipy(method=method, hessp=hep)
        assert m.valid
        if fixed:
            assert_allclose(m.values, [1, 1], atol=1e-3)
            assert_allclose(m.errors[1], 2, rtol=1e-2)
        else:
            assert_allclose(m.values, [0, 1], atol=1e-3)
            assert_allclose(m.errors, [1, 2], rtol=1e-2)


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


@pytest.mark.parametrize("lb", (0, 0.1))
@pytest.mark.parametrize("fixed", (False, True))
def test_scipy_constraints_1(lb, fixed):
    def fcn(a, x, y):
        return a + x ** 2 + y ** 2

    m = Minuit(fcn, a=3, x=1, y=2)
    m.fixed["a"] = fixed
    m.errordef = 1
    con_a = scopt.NonlinearConstraint(lambda a, x, y: a, lb, np.inf)
    con_x = scopt.NonlinearConstraint(lambda a, x, y: x, lb, np.inf)
    con_y = scopt.NonlinearConstraint(lambda a, x, y: y, lb, np.inf)
    m.scipy(constraints=[con_a, con_x, con_y])
    assert m.valid == (lb == 0 and fixed)
    if fixed:
        assert_allclose(m.values, [3, lb, lb], atol=1e-3)
    else:
        assert_allclose(m.values, [lb, lb, lb], atol=1e-3)
    assert m.accurate


@pytest.mark.parametrize("fixed", (False, True))
def test_scipy_constraints_2(fixed):
    def fcn(x, y):
        return x ** 2 + y ** 2

    m = Minuit(fcn, x=1, y=2)
    m.errordef = 1
    m.fixed["x"] = fixed
    con = scopt.LinearConstraint([1, 1], 0.1, np.inf)
    m.scipy(method="COBYLA", constraints=con)
    assert m.valid == fixed
    if fixed:
        assert_allclose(m.values, [1, 0.0], atol=1e-3)
        assert_allclose(m.errors[1], 1, atol=1e-3)
    else:
        assert_allclose(m.values, [0.05, 0.05], atol=1e-3)
        assert_allclose(m.errors, [1, 1], atol=1e-3)


def test_bad_constraint():
    m = Minuit(fcn, a=1, b=2)
    with pytest.raises(ValueError):
        m.scipy(constraints={})
    with pytest.raises(ValueError):
        m.scipy(constraints=[{}])


def test_high_print_level(capsys):
    m = Minuit(fcn, a=1, b=2)
    m.scipy()
    assert capsys.readouterr()[0] == ""
    m.reset()
    m.print_level = 1
    m.scipy()
    m.print_level = 0
    assert capsys.readouterr()[0] != ""


def test_on_modified_state():
    m = Minuit(fcn, a=1, b=2)
    m.scipy()
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=1e-3)
    m.fixed[1] = True  # modify latest state
    m.values = 1, 2
    m.scipy()  # used to fail
    assert m.valid
    assert_allclose(m.values, [0, 2], atol=1e-3)
