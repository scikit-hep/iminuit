import pytest
from iminuit import minimize
import numpy as np
from numpy.testing import assert_allclose

opt = pytest.importorskip("scipy.optimize")


def func(x, *args):
    c = args[0] if args else 1
    return c + x[0] ** 2 + (x[1] - 1) ** 2 + (x[2] - 2) ** 2


def grad(x, *args):
    return 2 * (x - (0, 1, 2))


def test_simple():
    result = minimize(func, (1, 1, 1))
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert result.nfev > 0
    assert result.njev == 0


def test_gradient():
    result = minimize(func, (1, 1, 1), jac=grad)
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert result.nfev > 0
    assert result.njev > 0


def test_args():
    result = minimize(func, np.ones(3), args=(5,))
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 5)
    assert result.nfev > 0
    assert result.njev == 0


def test_callback():
    trace = []
    result = minimize(func, np.ones(3), callback=lambda x: trace.append(x.copy()))
    assert_allclose(result.x, (0, 1, 2), atol=1e-8)
    assert_allclose(result.fun, 1)
    assert result.nfev == len(trace)
    assert_allclose(trace[0], np.ones(3), atol=1e-2)
    assert_allclose(trace[-1], result.x, atol=1e-2)


def test_tol():
    ref = np.ones(2)

    def rosen(par):
        x, y = par
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    r1 = minimize(rosen, (0, 0), tol=1)
    r2 = minimize(rosen, (0, 0), tol=1e-6)

    assert max(np.abs(r2.x - ref)) < max(np.abs(r1.x - ref))


def test_bad_function():
    r = minimize(lambda x: 0, 0)
    assert r.success is False


def test_disp(capsys):
    minimize(lambda x: x ** 2, 0)
    assert capsys.readouterr()[0] == ""
    minimize(lambda x: x ** 2, 0, options={"disp": True})
    assert capsys.readouterr()[0] != ""


def test_hessinv():
    r = minimize(func, (1, 1, 1))
    href = np.zeros((3, 3))
    for i in range(3):
        href[i, i] = 0.5
    assert_allclose(r.hess_inv, href, atol=1e-8)
