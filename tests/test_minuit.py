import platform
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from iminuit import Minuit
from iminuit.util import Param, IMinuitWarning, make_func_code
from pytest import approx
from argparse import Namespace


@pytest.fixture
def debug():
    from iminuit._core import MnPrint

    prev = MnPrint.global_level
    MnPrint.global_level = 3
    MnPrint.show_prefix_stack(True)
    yield
    MnPrint.global_level = prev
    MnPrint.show_prefix_stack(False)


is_pypy = platform.python_implementation() == "PyPy"


def test_pedantic_warning_message():
    with pytest.warns(IMinuitWarning, match=r"errordef not set, using 1"):
        m = Minuit(lambda x: 0, x=0)
        m.migrad()  # MARKER


def test_version():
    import iminuit

    assert iminuit.__version__


def lsq(func):
    func.errordef = Minuit.LEAST_SQUARES
    return func


@lsq
def func0(x, y):  # values = (2.0, 5.0), errors = (2.0, 1.0)
    return (x - 2.0) ** 2 / 4.0 + np.exp((y - 5.0) ** 2) + 10


def func0_grad(x, y):
    dfdx = (x - 2.0) / 2.0
    dfdy = 2.0 * (y - 5.0) * np.exp((y - 5.0) ** 2)
    return [dfdx, dfdy]


class Func1:
    errordef = 4

    def __call__(self, x, y):
        return func0(x, y) * 4


class Func2:
    errordef = 4

    def __init__(self):
        self.func_code = make_func_code(["x", "y"])

    def __call__(self, *arg):
        return func0(arg[0], arg[1]) * 4


@lsq
def func4(x, y, z):
    return 0.2 * (x - 2.0) ** 2 + 0.1 * (y - 5.0) ** 2 + 0.25 * (z - 7.0) ** 2 + 10


def func4_grad(x, y, z):
    dfdx = 0.4 * (x - 2.0)
    dfdy = 0.2 * (y - 5.0)
    dfdz = 0.5 * (z - 7.0)
    return dfdx, dfdy, dfdz


@lsq
def func5(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    return (
        (x - 1) ** 2
        + long_variable_name_really_long_why_does_it_has_to_be_this_long**2
        + (z + 1) ** 2
    )


def func5_grad(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    dfdx = 2 * (x - 1)
    dfdy = 2 * long_variable_name_really_long_why_does_it_has_to_be_this_long
    dfdz = 2 * (z + 1)
    return dfdx, dfdy, dfdz


@lsq
def func6(x, m, s, a):
    return a / ((x - m) ** 2 + s**2)


class Correlated:
    errordef = 1

    def __init__(self):
        sx = 2
        sy = 1
        corr = 0.5
        cov = (sx**2, corr * sx * sy), (corr * sx * sy, sy**2)
        self.cinv = np.linalg.inv(cov)

    def __call__(self, x):
        return np.dot(x.T, np.dot(self.cinv, x))


@lsq
def func_np(x):  # test numpy support
    return np.sum((x - 1) ** 2)


def func_np_grad(x):  # test numpy support
    return 2 * (x - 1)


data_y = [
    0.552,
    0.735,
    0.846,
    0.875,
    1.059,
    1.675,
    1.622,
    2.928,
    3.372,
    2.377,
    4.307,
    2.784,
    3.328,
    2.143,
    1.402,
    1.44,
    1.313,
    1.682,
    0.886,
    0.0,
    0.266,
    0.3,
]
data_x = list(range(len(data_y)))


def func_test_helper(f, grad=None, errordef=None):
    m = Minuit(f, x=0, y=0, grad=grad)
    if errordef:
        m.errordef = errordef
    m.migrad()
    val = m.values
    assert_allclose(val["x"], 2.0, rtol=2e-3)
    assert_allclose(val["y"], 5.0, rtol=2e-3)
    assert_allclose(m.fval, 11.0 * m.errordef, rtol=1e-3)
    assert m.valid
    assert m.accurate
    m.hesse()
    err = m.errors
    assert_allclose(err["x"], 2.0, rtol=1e-3)
    assert_allclose(err["y"], 1.0, rtol=1e-3)
    m.errors = (1, 2)
    assert_allclose(err["x"], 1.0, rtol=1e-3)
    assert_allclose(err["y"], 2.0, rtol=1e-3)
    return m


def test_func0():  # check that providing gradient improves convergence
    m1 = func_test_helper(func0)
    m2 = func_test_helper(func0, grad=func0_grad)
    assert m1.ngrad == 0
    assert m2.ngrad > 0


def test_lambda():
    func_test_helper(lambda x, y: func0(x, y), errordef=1)


def test_Func1():
    func_test_helper(Func1())


def test_Func2():
    func_test_helper(Func2())


def test_no_signature():
    def no_signature(*args):
        x, y = args
        return (x - 1) ** 2 + (y - 2) ** 2

    no_signature.errordef = 1

    m = Minuit(no_signature, 3, 4)
    assert m.values == (3, 4)
    assert m.parameters == ("x0", "x1")

    m = Minuit(no_signature, x=1, y=2, name=("x", "y"))
    assert m.values == (1, 2)
    m.migrad()
    val = m.values
    assert_allclose((val["x"], val["y"], m.fval), (1, 2, 0), atol=1e-8)
    assert m.valid

    with pytest.raises(RuntimeError):
        Minuit(no_signature, x=1)

    with pytest.raises(RuntimeError):
        Minuit(no_signature, x=1, y=2)


def test_use_array_call():
    inf = float("infinity")
    m = Minuit(
        func_np,
        (1, 1),
        name=("a", "b"),
    )
    m.fixed = False
    m.errors = 1
    m.limits = (0, inf)
    m.migrad()
    assert m.parameters == ("a", "b")
    assert_allclose(m.values, (1, 1))
    m.hesse()
    c = m.covariance
    assert_allclose((c[("a", "a")], c[("b", "b")]), (1, 1))
    with pytest.raises(RuntimeError):
        Minuit(lambda *args: 0, [1, 2], name=["a", "b", "c"])


def test_release_with_none():
    m = Minuit(func0, x=0, y=0)
    m.fixed = (True, False)
    assert m.fixed == (True, False)
    m.fixed = None
    assert m.fixed == (False, False)


def test_parameters():
    m = Minuit(lambda a, b: 0, a=1, b=1)
    assert m.parameters == ("a", "b")
    assert m.pos2var == ("a", "b")
    assert m.var2pos["a"] == 0
    assert m.var2pos["b"] == 1


def test_covariance():
    m = Minuit(func0, x=0, y=0)
    assert m.covariance is None
    m.migrad()
    c = m.covariance
    assert_allclose((c["x", "x"], c["y", "y"]), (4, 1), rtol=1e-4)
    assert_allclose((c[0, 0], c[1, 1]), (4, 1), rtol=1e-4)

    expected = [[4.0, 0.0], [0.0, 1.0]]
    assert_allclose(c, expected, atol=1e-4)
    assert isinstance(c, np.ndarray)
    assert c.shape == (2, 2)

    c = c.correlation()
    expected = [[1.0, 0.0], [0.0, 1.0]]
    assert_allclose(c, expected, atol=1e-4)
    assert c["x", "x"] == approx(1.0)


def test_array_func_1():
    m = Minuit(func_np, (2, 1))
    m.errors = (1, 1)
    assert m.parameters == ("x0", "x1")
    assert m.values == (2, 1)
    assert m.errors == (1, 1)
    m.migrad()
    assert_allclose(m.values, (1, 1), rtol=1e-2)
    c = m.covariance
    assert_allclose(np.diag(c), (1, 1), rtol=1e-2)


def test_array_func_2():
    m = Minuit(func_np, (2, 1), grad=func_np_grad, name=("a", "b"))
    m.fixed = (False, True)
    m.errors = (0.5, 0.5)
    m.limits = ((0, 2), (-np.inf, np.inf))
    assert m.values == (2, 1)
    assert m.errors == (0.5, 0.5)
    assert m.fixed == (False, True)
    assert m.limits["a"] == (0, 2)
    m.migrad()
    assert_allclose(m.values, (1, 1), rtol=1e-2)
    c = m.covariance
    assert_allclose(c, ((1, 0), (0, 0)), rtol=1e-2)
    m.minos()
    assert len(m.merrors) == 1
    assert m.merrors[0].lower == approx(-1, abs=1e-2)
    assert m.merrors[0].name == "a"


def test_wrong_use_of_array_init():
    m = Minuit(lambda a, b: a**2 + b**2, (1, 2))
    m.errordef = Minuit.LEAST_SQUARES
    with pytest.raises(TypeError):
        m.migrad()


def test_reset():
    m = Minuit(func0, x=0, y=0)
    m.migrad()
    n = m.nfcn
    m.migrad()
    assert m.nfcn > n
    m.reset()
    m.migrad()
    assert m.nfcn == n

    m = Minuit(func0, grad=func0_grad, x=0, y=0)
    m.migrad()
    n = m.nfcn
    k = m.ngrad
    m.migrad()
    assert m.nfcn > n
    assert m.ngrad > k
    m.reset()
    m.migrad()
    assert m.nfcn == n
    assert m.ngrad == k


def test_typo():
    with pytest.raises(RuntimeError):
        Minuit(lambda x: 0, y=1)

    m = Minuit(lambda x: 0, x=0)
    with pytest.raises(KeyError):
        m.errors["y"] = 1
    with pytest.raises(KeyError):
        m.limits["y"] = (0, 1)


def test_initial_guesses():
    m = Minuit(lambda x: 0, x=0)
    assert m.values["x"] == 0
    assert m.errors["x"] == 0.1
    m = Minuit(lambda x: 0, x=1)
    assert m.values["x"] == 1
    assert m.errors["x"] == 1e-2


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_fix_param(grad):
    m = Minuit(func0, grad=grad, x=0, y=0)
    assert m.npar == 2
    assert m.nfit == 2
    m.migrad()
    m.minos()
    assert_allclose(m.values, (2, 5), rtol=2e-3)
    assert_allclose(m.errors, (2, 1), rtol=1e-4)
    assert_allclose(m.covariance, ((4, 0), (0, 1)), atol=1e-4)

    # now fix y = 10
    m = Minuit(func0, grad=grad, x=0, y=10.0)
    m.fixed["y"] = True
    assert m.npar == 2
    assert m.nfit == 1
    m.migrad()
    assert_allclose(m.values, (2, 10), rtol=1e-2)
    assert_allclose(m.fval, func0(2, 10))
    assert m.fixed == [False, True]
    assert_allclose(m.covariance, [[4, 0], [0, 0]], atol=3e-4 if grad is None else 3e-2)

    assert not m.fixed["x"]
    assert m.fixed["y"]
    m.fixed["x"] = True
    m.fixed["y"] = False
    assert m.npar == 2
    assert m.nfit == 1
    m.migrad()
    m.hesse()
    assert_allclose(m.values, (2, 5), rtol=1e-2)
    assert_allclose(m.covariance, [[0, 0], [0, 1]], atol=1e-4)

    with pytest.raises(KeyError):
        m.fixed["a"]

    # fix by setting limits
    m = Minuit(func0, x=0, y=10.0)
    m.limits["y"] = (10, 10)
    assert m.fixed["y"]
    assert m.npar == 2
    assert m.nfit == 1

    # initial value out of range is forced in range
    m = Minuit(func0, x=0, y=20.0)
    m.limits["y"] = (10, 10)
    assert m.fixed["y"]
    assert m.values["y"] == 10
    assert m.npar == 2
    assert m.nfit == 1

    m.fixed = True
    assert m.fixed == [True, True]
    m.fixed[1:] = False
    assert m.fixed == [True, False]
    assert m.fixed[:1] == [True]


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_minos(grad):
    m = Minuit(func0, grad=grad, x=0, y=0)
    m.migrad()
    m.minos()
    assert len(m.merrors) == 2
    assert m.merrors["x"].lower == approx(-m.errors["x"], abs=4e-3)
    assert m.merrors["x"].upper == approx(m.errors["x"], abs=4e-3)
    assert m.merrors[1].lower == m.merrors["y"].lower
    assert m.merrors[-1].upper == m.merrors["y"].upper


@pytest.mark.parametrize("cl", (0.68, 0.90))
@pytest.mark.parametrize("k", (10, 1000))
@pytest.mark.parametrize("limit", (False, True))
def test_minos_cl(cl, k, limit):
    opt = pytest.importorskip("scipy.optimize")
    stats = pytest.importorskip("scipy.stats")

    def nll(lambd):
        return lambd - k * np.log(lambd)

    # find location of min + up by hand
    def crossing(x):
        up = 0.5 * stats.chi2(1).ppf(cl)
        return nll(k + x) - (nll(k) + up)

    bound = 1.5 * (stats.chi2(1).ppf(cl) * k) ** 0.5
    upper = opt.root_scalar(crossing, bracket=(0, bound)).root
    lower = opt.root_scalar(crossing, bracket=(-bound, 0)).root

    m = Minuit(nll, lambd=k)
    m.limits["lambd"] = (0, None) if limit else None
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    assert m.valid
    assert m.accurate
    m.minos(cl=cl)
    assert m.values["lambd"] == approx(k)
    assert m.errors["lambd"] == approx(k**0.5, abs=2e-3 if limit else None)
    assert m.merrors["lambd"].lower == approx(lower, rel=1e-3)
    assert m.merrors["lambd"].upper == approx(upper, rel=1e-3)
    assert m.merrors[0].lower == m.merrors["lambd"].lower
    assert m.merrors[-1].upper == m.merrors["lambd"].upper

    with pytest.raises(KeyError):
        m.merrors["xy"]
    with pytest.raises(KeyError):
        m.merrors["z"]
    with pytest.raises(IndexError):
        m.merrors[1]
    with pytest.raises(IndexError):
        m.merrors[-2]


def test_minos_some_fix():
    m = Minuit(func0, x=0, y=0)
    m.fixed["x"] = True
    m.migrad()
    m.minos()
    assert "x" not in m.merrors
    me = m.merrors["y"]
    assert me.name == "y"
    assert me.lower == approx(-0.83, abs=1e-2)
    assert me.upper == approx(0.83, abs=1e-2)


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_minos_single(grad):
    m = Minuit(func0, grad=func0_grad, x=0, y=0)

    m.strategy = 0
    m.migrad()
    m.minos("x")
    assert len(m.merrors) == 1
    me = m.merrors["x"]
    assert me.name == "x"
    assert me.lower == approx(-2, rel=2e-3)
    assert me.upper == approx(2, rel=2e-3)


def test_minos_single_fixed():
    m = Minuit(func0, x=0, y=0)
    m.fixed["x"] = True
    m.migrad()
    m.minos("y")
    assert len(m.merrors) == 1
    me = m.merrors["y"]
    assert me.name == "y"
    assert me.lower == approx(-0.83, abs=1e-2)


def test_minos_single_fixed_raising():
    m = Minuit(func0, x=0, y=0)
    m.fixed["x"] = True
    m.migrad()
    with pytest.warns(RuntimeWarning):
        m.minos("x")
    assert len(m.merrors) == 0
    assert m.fixed["x"]
    m.minos()
    assert len(m.merrors) == 1
    assert "y" in m.merrors


def test_minos_single_no_migrad():
    m = Minuit(func0, x=0, y=0)
    with pytest.raises(RuntimeError):
        m.minos("x")


def test_minos_single_nonsense_variable():
    m = Minuit(func0, x=0, y=0)
    m.migrad()
    with pytest.raises(RuntimeError):
        m.minos("nonsense")


def test_minos_with_bad_fmin():
    m = Minuit(lambda x: 0, x=0)
    m.errordef = 1
    m.migrad()
    with pytest.raises(RuntimeError):
        m.minos()


@pytest.mark.parametrize("grad", (None, func5_grad))
def test_fixing_long_variable_name(grad):
    m = Minuit(
        func5,
        grad=grad,
        long_variable_name_really_long_why_does_it_has_to_be_this_long=2,
        x=0,
        z=0,
    )
    m.errordef = 1
    m.fixed["long_variable_name_really_long_why_does_it_has_to_be_this_long"] = True
    m.migrad()
    assert_allclose(m.values, [1, 2, -1], atol=1e-3)


def test_initial_value():
    m = Minuit(func0, x=1.0, y=2.0)
    assert_allclose(m.values[0], 1.0)
    assert_allclose(m.values[1], 2.0)
    assert_allclose(m.values["x"], 1.0)
    assert_allclose(m.values["y"], 2.0)

    m = Minuit(func0, 1.0, 2.0)
    assert_allclose(m.values[0], 1.0)
    assert_allclose(m.values[1], 2.0)
    assert_allclose(m.values["x"], 1.0)
    assert_allclose(m.values["y"], 2.0)

    m = Minuit(func0, (1.0, 2.0))
    assert_allclose(m.values[0], 1.0)
    assert_allclose(m.values[1], 2.0)
    assert_allclose(m.values["x"], 1.0)
    assert_allclose(m.values["y"], 2.0)

    with pytest.raises(RuntimeError):
        Minuit(func0, 1, y=2)

    with pytest.raises(RuntimeError):
        Minuit(func0)


@pytest.mark.parametrize("grad", (None, func0_grad))
@pytest.mark.parametrize("cl", (None, 0.5, 0.9))
def test_mncontour(grad, cl):
    stats = pytest.importorskip("scipy.stats")
    m = Minuit(func0, grad=grad, x=1.0, y=2.0)
    m.migrad()
    ctr = m.mncontour("x", "y", size=30, cl=cl)

    factor = stats.chi2(2).ppf(0.68 if cl is None else cl)
    cl2 = stats.chi2(1).cdf(factor)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2

    m.minos("x", "y", cl=cl2)

    xm = m.merrors["x"]
    ym = m.merrors["y"]
    cmin = np.min(ctr, axis=0)
    cmax = np.max(ctr, axis=0)

    x, y = m.values
    assert_allclose((x + xm.lower, y + ym.lower), cmin)
    assert_allclose((x + xm.upper, y + ym.upper), cmax)


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_contour(grad):
    m = Minuit(func0, grad=grad, x=1.0, y=2.0)
    m.migrad()
    x, y, v = m.contour("x", "y")
    X, Y = np.meshgrid(x, y)
    assert_allclose(func0(X, Y), v.T)


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_profile(grad):
    m = Minuit(func0, grad=grad, x=1.0, y=2.0)
    m.migrad()

    y, v = m.profile("y", subtract_min=False)
    assert_allclose(func0(m.values[0], y), v)

    v2 = m.profile("y", subtract_min=True)[1]
    assert np.min(v2) == 0
    assert_allclose(v - np.min(v), v2)


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_mnprofile(grad):
    m = Minuit(func0, grad=grad, x=1.0, y=2.0)
    m.migrad()

    with pytest.raises(ValueError):
        m.mnprofile("foo")

    y, v, _ = m.mnprofile("y", size=10, subtract_min=False)
    m2 = Minuit(func0, grad=grad, x=1.0, y=2.0)
    m2.fixed[1] = True
    v2 = []
    for yi in y:
        m2.values = (m.values[0], yi)
        m2.migrad()
        v2.append(m2.fval)

    assert_allclose(v, v2)

    y, v3, _ = m.mnprofile("y", size=10, subtract_min=True)
    assert np.min(v3) == 0
    assert_allclose(v - np.min(v), v3)


def test_contour_subtract():
    m = Minuit(func0, x=1.0, y=2.0)
    m.migrad()
    v = m.contour("x", "y", subtract_min=False)[2]
    v2 = m.contour("x", "y", subtract_min=True)[2]
    assert np.min(v2) == 0
    assert_allclose(v - np.min(v), v2)


def test_mncontour_no_fmin():
    m = Minuit(func0, x=0, y=0)

    with pytest.raises(RuntimeError):
        m.mncontour("x", "y")  # fails, because this is not a minimum

    # succeeds
    m.values = (2, 5)
    c = m.mncontour("x", "y", size=10)

    # compute reference to compare with
    m2 = Minuit(func0, x=0, y=0)
    m2.migrad()
    c2 = m.mncontour("x", "y", size=10)

    assert_allclose(c, c2)


def test_mncontour_with_fixed_var():
    m = Minuit(func0, x=0, y=0)
    m.errordef = 1
    m.fixed["x"] = True
    m.migrad()
    with pytest.raises(ValueError):
        m.mncontour("x", "y")


def test_mncontour_array_func():
    stats = pytest.importorskip("scipy.stats")

    m = Minuit(Correlated(), (0, 0), name=("x", "y"))
    m.migrad()

    cl = stats.chi2(2).cdf(1)
    ctr = m.mncontour("x", "y", size=30, cl=cl)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2

    m.minos("x", "y")
    x, y = m.values
    xm = m.merrors["x"]
    ym = m.merrors["y"]
    cmin = np.min(ctr, axis=0)
    cmax = np.max(ctr, axis=0)
    assert_allclose((x + xm.lower, y + ym.lower), cmin)
    assert_allclose((x + xm.upper, y + ym.upper), cmax)


def test_profile_array_func():
    m = Minuit(Correlated(), (0, 0), name=("x", "y"))
    m.migrad()
    m.profile("y")


def test_mnprofile_array_func():
    m = Minuit(Correlated(), (0, 0), name=("x", "y"))
    m.migrad()
    m.mnprofile("y")


def test_mnprofile_bad_func():
    m = Minuit(lambda x, y: 0, 0, 0)
    m.errordef = 1
    with pytest.warns(IMinuitWarning):
        m.mnprofile("x")


def test_fmin_uninitialized(capsys):
    m = Minuit(func0, x=0, y=0)
    assert m.fmin is None
    assert m.fval is None


def test_reverse_limit():
    # issue 94
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    with pytest.raises(ValueError):
        m = Minuit(f, x=0, y=0, z=0)
        m.limits["x"] = (3.0, 2.0)


@pytest.fixture
def minuit():
    m = Minuit(func0, x=0, y=0)
    m.migrad()
    m.hesse()
    m.minos()
    return m


def test_fcn():
    m = Minuit(func0, x=0, y=0)
    v = m.fcn([2.0, 5.0])
    assert v == func0(2.0, 5.0)


def test_grad():
    m = Minuit(func0, grad=func0_grad, x=0, y=0)
    v = m.fcn([2.0, 5.0])
    g = m.grad([2.0, 5.0])
    assert v == func0(2.0, 5.0)
    assert_equal(g, func0_grad(2.0, 5.0))


def test_values(minuit):
    expected = [2.0, 5.0]
    assert len(minuit.values) == 2
    assert_allclose(minuit.values, expected, atol=4e-3)
    minuit.values = expected
    assert minuit.values == expected
    assert minuit.values[-1] == 5
    assert minuit.values[0] == 2
    assert minuit.values[1] == 5
    assert minuit.values["x"] == 2
    assert minuit.values["y"] == 5
    assert minuit.values[:1] == [2]
    minuit.values[1:] = [3]
    assert minuit.values[:] == [2, 3]
    assert minuit.values[-1] == 3
    minuit.values = 7
    assert minuit.values[:] == [7, 7]
    with pytest.raises(KeyError):
        minuit.values["z"]
    with pytest.raises(IndexError):
        minuit.values[3]
    with pytest.raises(IndexError):
        minuit.values[-10] = 1
    with pytest.raises(ValueError):
        minuit.values[:] = [2]


def test_fmin():
    m = Minuit(lambda x, s: (x * s) ** 2, x=1, s=1)
    m.fixed["s"] = True
    m.errordef = 1
    m.migrad()
    fm1 = m.fmin
    assert fm1.is_valid

    m.values["s"] = 0

    m.migrad()
    fm2 = m.fmin

    assert fm1.is_valid
    assert not fm2.is_valid


def test_chi2_fit():
    def chi2(x, y):
        return (x - 1) ** 2 + ((y - 2) / 3) ** 2

    m = Minuit(chi2, x=0, y=0)
    m.errordef = 1
    m.migrad()
    assert_allclose(m.values, (1, 2))
    assert_allclose(m.errors, (1, 3))


def test_likelihood():
    # normal distributed
    # fmt: off
    z = np.array([-0.44712856, 1.2245077 , 0.40349164, 0.59357852, -1.09491185,
                  0.16938243, 0.74055645, -0.9537006 , -0.26621851, 0.03261455,
                  -1.37311732, 0.31515939, 0.84616065, -0.85951594, 0.35054598,
                  -1.31228341, -0.03869551, -1.61577235, 1.12141771, 0.40890054,
                  -0.02461696, -0.77516162, 1.27375593, 1.96710175, -1.85798186,
                  1.23616403, 1.62765075, 0.3380117 , -1.19926803, 0.86334532,
                  -0.1809203 , -0.60392063, -1.23005814, 0.5505375 , 0.79280687,
                  -0.62353073, 0.52057634, -1.14434139, 0.80186103, 0.0465673 ,
                  -0.18656977, -0.10174587, 0.86888616, 0.75041164, 0.52946532,
                  0.13770121, 0.07782113, 0.61838026, 0.23249456, 0.68255141,
                  -0.31011677, -2.43483776, 1.0388246 , 2.18697965, 0.44136444,
                  -0.10015523, -0.13644474, -0.11905419, 0.01740941, -1.12201873,
                  -0.51709446, -0.99702683, 0.24879916, -0.29664115, 0.49521132,
                  -0.17470316, 0.98633519, 0.2135339 , 2.19069973, -1.89636092,
                  -0.64691669, 0.90148689, 2.52832571, -0.24863478, 0.04366899,
                  -0.22631424, 1.33145711, -0.28730786, 0.68006984, -0.3198016 ,
                  -1.27255876, 0.31354772, 0.50318481, 1.29322588, -0.11044703,
                  -0.61736206, 0.5627611 , 0.24073709, 0.28066508, -0.0731127 ,
                  1.16033857, 0.36949272, 1.90465871, 1.1110567 , 0.6590498 ,
                 -1.62743834, 0.60231928, 0.4202822 , 0.81095167, 1.04444209])
    # fmt: on

    data = 2 * z + 1

    def nll(mu, sigma):
        z = (data - mu) / sigma
        logp = -0.5 * z**2 - np.log(sigma)
        return -np.sum(logp)

    m = Minuit(nll, mu=0, sigma=1)
    m.errordef = Minuit.LIKELIHOOD
    m.limits["sigma"] = (0, None)
    m.migrad()

    mu = np.mean(data)
    sigma = np.std(data)
    assert_allclose(m.values, (mu, sigma), rtol=5e-3)
    s_mu = sigma / len(data) ** 0.5
    assert_allclose(m.errors, (s_mu, 0.12047), rtol=1e-1)


def test_oneside():
    # Solution: x=2., y=5.
    m = Minuit(func0, x=0, y=0)
    m.limits["x"] = (None, 9)
    m.migrad()
    assert_allclose(m.values, (2, 5), atol=2e-2)
    m.values["x"] = 0
    m.limits["x"] = (None, 1)
    m.migrad()
    assert_allclose(m.values, (1, 5), atol=1e-3)
    m.values = (5, 0)
    m.limits["x"] = (3, None)
    m.migrad()
    assert_allclose(m.values, (3, 5), atol=4e-3)


def test_oneside_outside():
    m = Minuit(func0, x=5, y=0)
    m.limits["x"] = (None, 1)
    assert m.values["x"] == 1
    m.limits["x"] = (2, None)
    assert m.values["x"] == 2


def test_migrad_ncall():
    class Func:
        nfcn = 0
        errordef = 1

        def __call__(self, x):
            self.nfcn += 1
            return np.exp(x**2)

    # check that counting is accurate
    fcn = Func()
    m = Minuit(fcn, x=3)
    m.migrad()
    assert m.nfcn == fcn.nfcn
    fcn.nfcn = 0
    m.reset()
    m.migrad()
    assert m.nfcn == fcn.nfcn

    ncalls_without_limit = m.nfcn
    # check that ncall argument limits function calls in migrad
    # note1: Minuit only checks the ncall counter in units of one iteration
    # step, therefore the call counter is in general not equal to ncall.
    # note2: If you pass ncall=0, Minuit uses a heuristic value that depends
    # on the number of parameters.
    m.reset()
    m.migrad(ncall=1)
    assert m.nfcn < ncalls_without_limit


def test_ngrad():
    class Func:
        errordef = 1
        ngrad = 0

        def __call__(self, x):
            return x**2

        def grad(self, x):
            self.ngrad += 1
            return [2 * x]

    # check that counting is accurate
    fcn = Func()
    m = Minuit(fcn, 1)
    m.migrad()
    assert m.ngrad > 0
    assert m.ngrad == fcn.ngrad
    fcn.ngrad = 0
    m.reset()
    m.migrad()
    assert m.ngrad == fcn.ngrad

    # HESSE ignores analytical gradient
    before = m.ngrad
    m.hesse()
    assert m.ngrad == before


def test_errordef():
    m = Minuit(lambda x: x**2, 0)
    m.errordef = 4
    assert m.errordef == 4
    m.migrad()
    m.hesse()
    assert_allclose(m.errors["x"], 2)
    m.errordef = 1
    m.hesse()
    assert_allclose(m.errors["x"], 1)
    with pytest.raises(ValueError):
        m.errordef = 0


def test_print_level():
    from iminuit._core import MnPrint

    m = Minuit(lambda x: 0, x=0)
    m.print_level = 0
    assert m.print_level == 0
    assert MnPrint.global_level == 0
    m.print_level = 1
    assert MnPrint.global_level == 1
    MnPrint.global_level = 0


def test_params():
    m = Minuit(func0, x=1, y=2)
    m.errordef = Minuit.LEAST_SQUARES
    m.errors = (3, 4)
    m.fixed["x"] = True
    m.limits["y"] = (None, 10)

    # these are the initial param states
    expected = (
        Param(0, "x", 1.0, 3.0, None, False, True, None, None),
        Param(1, "y", 2.0, 4.0, None, False, False, None, 10),
    )
    assert m.params == expected

    m.migrad()
    m.minos()
    assert m.init_params == expected

    expected = [
        Namespace(number=0, name="x", value=1.0, error=3.0, merror=(-3.0, 3.0)),
        Namespace(number=1, name="y", value=5.0, error=1.0, merror=(-1.0, 1.0)),
    ]

    params = m.params
    for i, exp in enumerate(expected):
        p = params[i]
        assert p.number == exp.number
        assert p.name == exp.name
        assert p.value == approx(exp.value, rel=1e-2)
        assert p.error == approx(exp.error, rel=1e-2)
        assert p.error == approx(exp.error, rel=1e-2)


def test_non_analytical_function():
    class Func:
        errordef = 1
        i = 0

        def __call__(self, a):
            self.i += 1
            return self.i % 3

    m = Minuit(Func(), 0)
    m.migrad()
    assert m.fmin.is_valid is False
    assert m.fmin.is_above_max_edm is True


def test_non_invertible():
    m = Minuit(lambda x, y: 0, 1, 2)
    m.errordef = 1
    m.strategy = 0
    m.migrad()
    assert m.fmin.is_valid
    m.hesse()
    assert not m.fmin.is_valid
    assert m.covariance is None


def test_function_without_local_minimum():
    m = Minuit(lambda a: -a, 0)
    m.errordef = 1
    m.migrad()
    assert m.fmin.is_valid is False
    assert m.fmin.is_above_max_edm is True


def test_function_with_maximum():
    def func(a):
        return -(a**2)

    m = Minuit(func, a=0)
    m.errordef = 1
    m.migrad()
    assert m.fmin.is_valid is False


def test_perfect_correlation():
    def func(a, b):
        return (a - b) ** 2

    m = Minuit(func, a=1, b=2)
    m.errordef = 1
    m.migrad()
    assert m.fmin.is_valid is True
    assert m.fmin.has_accurate_covar is False
    assert m.fmin.has_posdef_covar is False
    assert m.fmin.has_made_posdef_covar is True


def test_modify_param_state():
    m = Minuit(func0, x=1, y=2)
    m.errors["y"] = 1
    m.fixed["y"] = True
    m.migrad()
    assert_allclose(m.values, [2, 2], atol=1e-4)
    assert_allclose(m.errors, [2, 1], atol=1e-4)
    m.fixed["y"] = False
    m.values["x"] = 1
    m.errors["x"] = 1
    assert_allclose(m.values, [1, 2], atol=1e-4)
    assert_allclose(m.errors, [1, 1], atol=1e-4)
    m.migrad()
    assert_allclose(m.values, [2, 5], atol=1e-3)
    assert_allclose(m.errors, [2, 1], atol=1e-3)
    m.values["y"] = 6
    m.hesse()
    assert_allclose(m.values, [2, 6], atol=1e-3)
    assert_allclose(m.errors, [2, 0.35], atol=1e-3)


def test_view_lifetime():
    m = Minuit(func0, x=1, y=2)
    val = m.values
    del m
    val["x"] = 3  # should not segfault
    assert val["x"] == 3


def test_hesse_without_migrad():
    m = Minuit(lambda x: x**2 + x**4, x=0)
    m.errordef = 0.5
    # second derivative: 12 x^2 + 2
    m.hesse()
    assert m.errors["x"] == approx(0.5**0.5, abs=1e-4)
    m.values["x"] = 1
    m.hesse()
    assert m.errors["x"] == approx((1.0 / 14.0) ** 0.5, abs=1e-4)
    assert m.fmin

    m = Minuit(lambda x: 0, 0)
    m.errordef = 1
    m.hesse()
    assert not m.accurate
    assert m.fmin.hesse_failed


def test_edm_goal():
    m = Minuit(func0, x=0, y=0)
    m.migrad()
    assert m.fmin.edm_goal == approx(0.0002)
    m.hesse()
    assert m.fmin.edm_goal == approx(0.0002)


def throwing(x):
    raise RuntimeError("user message")


def divide_by_zero(x):
    return 1 / 0


def returning_nan(x):
    return np.nan


def returning_garbage(x):
    return "foo"


@pytest.mark.parametrize(
    "func,expected",
    [
        (throwing, RuntimeError("user message")),
        (divide_by_zero, ZeroDivisionError("division by zero")),
        (returning_nan, RuntimeError("result is NaN")),
        (returning_garbage, RuntimeError("Unable to cast Python instance")),
    ],
)
def test_bad_functions(func, expected):
    m = Minuit(func, x=1)
    m.errordef = 1
    m.throw_nan = True
    with pytest.raises(type(expected)) as excinfo:
        m.migrad()
    assert str(expected) in str(excinfo.value)


def test_throw_nan():
    m = Minuit(returning_nan, x=1)
    m.errordef = 1
    assert not m.throw_nan
    m.migrad()
    m.throw_nan = True
    with pytest.raises(RuntimeError):
        m.migrad()
    assert m.throw_nan


def returning_nan_array(x):
    return np.array([1, np.nan])


def returning_garbage_array(x):
    return np.array([1, "foo"])


def returning_noniterable(x):
    return 0


@pytest.mark.parametrize(
    "func,expected",
    [
        (throwing, RuntimeError("user message")),
        (divide_by_zero, ZeroDivisionError("division by zero")),
        (returning_nan_array, RuntimeError("result is NaN")),
        (returning_garbage_array, RuntimeError("Unable to cast Python instance")),
        (returning_noniterable, RuntimeError()),
    ],
)
def test_bad_functions_np(func, expected):
    m = Minuit(lambda x: np.dot(x, x), (1, 1), grad=func)
    m.errordef = 1
    m.throw_nan = True
    with pytest.raises(type(expected)) as excinfo:
        m.migrad()
    assert str(expected) in str(excinfo.value)


@pytest.mark.parametrize("sign", (-1, 1))
def test_parameter_at_limit(sign):
    m = Minuit(lambda x: (x - sign * 1.2) ** 2, x=0)
    m.errordef = 1
    m.limits["x"] = (-1, 1)
    m.migrad()
    assert m.values["x"] == approx(sign * 1.0, abs=1e-3)
    assert m.fmin.has_parameters_at_limit is True

    m = Minuit(lambda x: (x - sign * 1.2) ** 2, x=0)
    m.errordef = 1
    m.migrad()
    assert m.values["x"] == approx(sign * 1.2, abs=1e-3)
    assert m.fmin.has_parameters_at_limit is False


@pytest.mark.parametrize("iterate,valid", ((1, False), (5, True)))
def test_inaccurate_fcn(iterate, valid):
    def f(x):
        return abs(x) ** 10 + 1e7

    m = Minuit(f, x=2)
    m.errordef = 1
    m.migrad(iterate=iterate)
    assert m.valid == valid


def test_migrad_iterate():
    m = Minuit(lambda x: 0, x=2)
    m.errordef = 1
    with pytest.raises(ValueError):
        m.migrad(iterate=0)


def test_precision():
    @lsq
    def fcn(x):
        return np.exp(x * x + 1)

    m = Minuit(fcn, x=-1)
    assert m.precision is None

    m.precision = 0.1
    assert m.precision == 0.1
    m.migrad()
    fm1 = m.fmin
    m.reset()
    m.precision = 1e-9
    m.migrad()
    fm2 = m.fmin
    assert fm2.edm < fm1.edm

    with pytest.raises(ValueError):
        m.precision = -1.0

    fcn.precision = 0.1
    fm3 = Minuit(fcn, x=-1).migrad().fmin
    assert fm3.edm == fm1.edm


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_scan(grad):
    m = Minuit(func0, x=0, y=0, grad=grad)
    m.errors[0] = 10
    m.limits[1] = (-10, 10)
    m.scan(ncall=99)
    assert m.fmin.nfcn == approx(99, rel=0.2)
    if grad is None:
        assert m.valid
    assert_allclose(m.values, (2, 5), atol=0.6)


def test_scan_with_fixed_par():
    m = Minuit(func0, x=3, y=0)
    m.fixed["x"] = True
    m.limits[1] = (-10, 10)
    m.scan()
    assert m.valid
    assert_allclose(m.values, (3, 5), atol=0.1)
    assert m.errors[1] == approx(1, abs=8e-3)

    m = Minuit(func0, x=5, y=4)
    m.fixed["y"] = True
    m.limits[0] = (0, 10)
    m.scan()
    assert m.valid
    assert_allclose(m.values, (2, 4), atol=0.1)
    assert m.errors[0] == approx(2, abs=1e-1)


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_simplex(grad):
    m = Minuit(func0, x=0, y=0, grad=grad)
    m.tol = 2e-4  # must decrease tolerance to get same accuracy as Migrad
    m.simplex()
    assert m.valid
    assert_allclose(m.values, (2, 5), atol=5e-3)

    m2 = Minuit(func0, x=0, y=0, grad=grad)
    m2.precision = 0.001
    m2.simplex()
    assert m2.fval != m.fval

    m3 = Minuit(func0, x=0, y=0, grad=grad)
    m3.simplex(ncall=10)
    assert 10 <= m3.fmin.nfcn < 15
    assert m3.fval > m.fval


def test_simplex_with_fixed_par_and_limits():
    m = Minuit(func0, x=3, y=0)
    m.tol = 2e-4  # must decrease tolerance to get same accuracy as Migrad
    m.fixed["x"] = True
    m.limits[1] = (-10, 10)
    m.simplex()
    assert m.valid
    assert_allclose(m.values, (3, 5), atol=2e-3)

    m = Minuit(func0, x=5, y=4)
    m.tol = 2e-4  # must decrease tolerance to get same accuracy as Migrad
    m.fixed["y"] = True
    m.limits[0] = (0, 10)
    m.simplex()
    assert m.valid
    assert_allclose(m.values, (2, 4), atol=3e-3)


def test_tolerance():
    m = Minuit(func0, x=0, y=0)
    assert m.tol == 0.1
    m.migrad()
    assert m.valid
    edm = m.fmin.edm
    m.tol = 0
    m.reset()
    m.migrad()
    assert m.fmin.edm < edm
    m.reset()
    m.tol = None
    assert m.tol == 0.1
    m.reset()
    m.migrad()
    assert m.fmin.edm == edm


def test_bad_tolerance():
    m = Minuit(func0, x=0, y=0)

    with pytest.raises(ValueError):
        m.tol = -1


def test_cfunc():
    nb = pytest.importorskip("numba")

    c_sig = nb.types.double(nb.types.uintc, nb.types.CPointer(nb.types.double))

    @lsq
    @nb.cfunc(c_sig)
    def fcn(n, x):
        x = nb.carray(x, (n,))
        r = 0.0
        for i in range(n):
            r += (x[i] - i) ** 2
        return r

    m = Minuit(fcn, (1, 2, 3))
    m.migrad()
    assert_allclose(m.values, (0, 1, 2), atol=1e-8)


@pytest.mark.parametrize("cl", (0.5, None, 0.9))
def test_confidence_level(cl):
    stats = pytest.importorskip("scipy.stats")
    mpath = pytest.importorskip("matplotlib.path")

    cov = ((1.0, 0.5), (0.5, 4.0))
    truth = (1.0, 2.0)
    d = stats.multivariate_normal(truth, cov)

    def nll(par):
        return -np.log(d.pdf(par))

    nll.errordef = 0.5

    cl_ref = 0.68 if cl is None else cl

    m = Minuit(nll, (0.0, 0.0))
    m.migrad()

    n = 10000
    r = d.rvs(n, random_state=1)

    # check that mncontour indeed contains fraction of random points equal to CL
    pts = m.mncontour("x0", "x1", cl=cl)
    p = mpath.Path(pts)
    cl2 = np.sum(p.contains_points(r)) / n
    assert cl2 == approx(cl_ref, abs=0.01)

    # check that minos interval  indeed contains fraction of random points equal to CL
    m.minos(cl=cl)
    for ipar, (v, me) in enumerate(zip(m.values, m.merrors.values())):
        a = v + me.lower
        b = v + me.upper
        cl2 = np.sum((a < r[:, ipar]) & (r[:, ipar] < b)) / n
        assert cl2 == approx(cl_ref, abs=0.01)


def test_repr():
    m = Minuit(func0, 0, 0)
    assert repr(m) == f"{m.params!r}"

    m.migrad()
    assert repr(m) == f"{m.fmin!r}\n{m.params!r}\n{m.covariance!r}"

    m.minos()
    assert repr(m) == f"{m.fmin!r}\n{m.params!r}\n{m.merrors!r}\n{m.covariance!r}"


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_pickle(grad):
    import pickle

    m = Minuit(func0, x=1, y=1, grad=grad)
    m.fixed[1] = True
    m.limits[0] = 0, 10
    m.migrad()

    pkl = pickle.dumps(m)
    m2 = pickle.loads(pkl)

    assert id(m2) != id(m)
    # check correct linking of views
    assert id(m2.values._minuit) == id(m2)
    assert id(m2.errors._minuit) == id(m2)
    assert id(m2.limits._minuit) == id(m2)
    assert id(m2.fixed._minuit) == id(m2)

    assert m2.init_params == m.init_params
    assert m2.params == m.params
    assert m2.fmin == m.fmin
    assert_equal(m2.covariance, m.covariance)

    m.fixed = False
    m2.fixed = False
    m.migrad()
    m.minos()

    m2.migrad()
    m2.minos()

    assert m2.merrors == m.merrors

    assert m2.fmin.fval == m.fmin.fval
    assert m2.fmin.edm == m.fmin.edm
    assert m2.fmin.nfcn == m.fmin.nfcn
    assert m2.fmin.ngrad == m.fmin.ngrad


def test_minos_new_min():
    xref = [1.0]
    m = Minuit(lsq(lambda x: (x - xref[0]) ** 2), x=0)
    m.migrad()
    assert m.values[0] == approx(xref[0], abs=1e-3)
    m.minos()
    assert m.merrors["x"].lower == approx(-1, abs=1e-2)
    assert m.merrors["x"].upper == approx(1, abs=1e-2)
    xref[0] = 1.1
    m.minos()
    # values are not updated...
    assert m.values[0] == approx(1.0, abs=1e-3)  # should be 1.1
    # ...but interval is correct
    assert m.merrors["x"].lower == approx(-0.9, abs=1e-2)
    assert m.merrors["x"].upper == approx(1.1, abs=1e-2)


def test_minos_without_migrad():
    m = Minuit(lsq(lambda x, y: (x - 1) ** 2 + (y / 2) ** 2), 1.001, 0.001)
    m.minos()
    me = m.merrors["x"]
    assert me.is_valid
    assert me.lower == approx(-1, abs=5e-3)
    assert me.upper == approx(1, abs=5e-3)
    me = m.merrors["y"]
    assert me.is_valid
    assert me.lower == approx(-2, abs=5e-3)
    assert me.upper == approx(2, abs=5e-3)


def test_missing_ndata():
    def fcn(a):
        return a

    fcn.errordef = 1

    m = Minuit(fcn, 1)
    assert_equal(m.ndof, np.nan)


def test_call_limit_reached_in_hesse():
    m = Minuit(lambda x: ((x - 1.2) ** 4).sum(), np.ones(10) * 10)
    m.errordef = 1
    m.migrad(ncall=200)
    assert m.fmin.has_reached_call_limit
    assert m.fmin.nfcn < 205
