import pytest
from iminuit import minimize
import numpy as np
from numpy.testing import assert_allclose, assert_equal

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


def test_unsupported():
    with pytest.raises(ValueError):
        minimize(func, (1, 1, 1), constraints=[])
    with pytest.raises(ValueError):
        minimize(func, (1, 1, 1), jac=True)


def test_call_limit():
    ref = minimize(func, (1, 1, 1))
    with pytest.warns(UserWarning):
        r1 = minimize(func, (1, 1, 1), options={"maxiter": 1})
    assert r1.nfev < ref.nfev
    assert not r1.success
    assert "Call limit" in r1.message

    with pytest.warns(DeprecationWarning):
        r2 = minimize(func, (1, 1, 1), options={"maxfev": 1})
    assert not r2.success
    assert r2.nfev == r1.nfev

    r3 = minimize(func, (1, 1, 1), options={"maxfun": 1})
    assert not r3.success
    assert r3.nfev == r1.nfev


def test_eps():
    ref = minimize(func, (1, 1, 1))
    r = minimize(func, (1, 1, 1), options={"eps": 1e-10})
    assert np.any(ref.x != r.x)
    assert_allclose(r.x, ref.x, atol=1e-9)


def test_bad_function():
    class Fcn:
        n = 0

        def __call__(self, x):
            self.n += 1
            return x ** 2 + 1e-2 * (self.n % 3)

    r = minimize(Fcn(), [1], options={"maxfun": 100000000})
    assert not r.success
    assert "Estimated distance to minimum too large" in r.message


def test_bounds():
    r1 = minimize(func, (1.5, 1.7, 1.5), bounds=opt.Bounds((1, 1.5, 1), (2, 2, 2)))
    assert r1.success
    assert_allclose(r1.x, (1, 1.5, 2), atol=1e-2)
    r2 = minimize(func, (1.5, 1.7, 1.5), bounds=((1, 2), (1.5, 2), (1, 2)))
    assert r2.success
    assert_equal(r1.x, r2.x)


def test_method_warn():
    with pytest.raises(ValueError):
        minimize(func, (1.5, 1.7, 1.5), method="foo")


def test_hess_warn():
    with pytest.warns(UserWarning):
        minimize(func, (1.5, 1.7, 1.5), hess=True)


def test_unreliable_uncertainties():
    r = minimize(func, (1.5, 1.7, 1.5), options={"stra": 0})
    assert (
        r.message
        == "Optimization terminated successfully, but uncertainties are unrealiable."
    )


def test_simplex():
    r = minimize(func, (1.5, 1.7, 1.5), method="simplex", tol=1e-4)
    assert r.success
    assert_allclose(r.x, (0, 1, 2), atol=2e-3)
