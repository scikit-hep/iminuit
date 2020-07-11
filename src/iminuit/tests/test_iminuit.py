import warnings
import platform
import pytest
import numpy as np
from numpy.testing import assert_allclose
from iminuit import Minuit
from iminuit.util import Param, HesseFailedWarning
from pytest import approx

is_pypy = platform.python_implementation() == "PyPy"


def test_pedantic_warning_message():
    with warnings.catch_warnings(record=True) as w:
        Minuit(lambda x: 0)  # MARKER

    with open(__file__) as f:
        for lineno, line in enumerate(f):
            if ("Minuit(lambda x: 0)  # MARKER") in line:
                break

    assert len(w) == 2
    for i, msg in enumerate(
        (
            "Parameter x does not have neither initial value nor limits.",
            "errordef is not given, defaults to 1.",
        )
    ):
        assert str(w[i].message) == msg
        assert w[i].filename == __file__
        assert w[i].lineno == lineno + 1


def test_version():
    import iminuit

    assert iminuit.__version__


class Func_Code:
    def __init__(self, varname):
        self.co_varnames = varname
        self.co_argcount = len(varname)


def func0(x, y):
    return (x - 2.0) ** 2 / 4.0 + (y - 5.0) ** 2 + 10


def func0_grad(x, y):
    dfdx = (x - 2.0) / 2.0
    dfdy = 2.0 * (y - 5.0)
    return [dfdx, dfdy]


class Func1:
    errordef = 4

    def __call__(self, x, y):
        return func0(x, y) * 4


class Func2:
    def __init__(self):
        self.func_code = Func_Code(["x", "y"])

    def __call__(self, *arg):
        return func0(arg[0], arg[1]) * 4

    errordef = 4


def func4(x, y, z):
    return 0.2 * (x - 2.0) ** 2 + 0.1 * (y - 5.0) ** 2 + 0.25 * (z - 7.0) ** 2 + 10


def func4_grad(x, y, z):
    dfdx = 0.4 * (x - 2.0)
    dfdy = 0.2 * (y - 5.0)
    dfdz = 0.5 * (z - 7.0)
    return dfdx, dfdy, dfdz


def func5(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    return (
        (x ** 2)
        + (z ** 2)
        + long_variable_name_really_long_why_does_it_has_to_be_this_long ** 2
    )


def func5_grad(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    dfdx = 2 * x
    dfdy = 2 * long_variable_name_really_long_why_does_it_has_to_be_this_long
    dfdz = 2 * z
    return dfdx, dfdy, dfdz


def func6(x, m, s, a):
    return a / ((x - m) ** 2 + s ** 2)


def func7(x):  # test numpy support
    return np.sum((x - 1) ** 2)


def func7_grad(x):  # test numpy support
    return 2 * (x - 1)


class Func8:
    def __init__(self):
        sx = 2
        sy = 1
        corr = 0.5
        cov = (sx ** 2, corr * sx * sy), (corr * sx * sy, sy ** 2)
        self.cinv = np.linalg.inv(cov)

    def __call__(self, x):
        return np.dot(x.T, np.dot(self.cinv, x))


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


def func_test_helper(f, **kwds):
    m = Minuit(f, pedantic=False, **kwds)
    m.migrad()
    val = m.values
    assert_allclose(val["x"], 2.0, rtol=1e-3)
    assert_allclose(val["y"], 5.0, rtol=1e-3)
    assert_allclose(m.fval, 10.0 * m.errordef, rtol=1e-3)
    assert m.valid
    assert m.accurate
    m.hesse()
    err = m.errors
    assert_allclose(err["x"], 2.0, rtol=1e-3)
    assert_allclose(err["y"], 1.0, rtol=1e-3)
    return m


def test_func0():  # check that providing gradient improves convergence
    m1 = func_test_helper(func0)
    m2 = func_test_helper(func0, grad=func0_grad)
    assert m1.ngrads_total == 0
    assert m2.ngrads_total > 0
    assert m1.ncalls_total > m2.ngrads_total


def test_lambda():
    func_test_helper(lambda x, y: func0(x, y))


def test_Func1():
    func_test_helper(Func1())


def test_Func2():
    func_test_helper(Func2())


def test_no_signature():
    def no_signature(*args):
        x, y = args
        return (x - 1) ** 2 + (y - 2) ** 2

    with pytest.raises(TypeError):
        Minuit(no_signature)

    m = Minuit(no_signature, name=("x", "y"), pedantic=False)
    m.migrad()
    val = m.values
    assert_allclose((val["x"], val["y"], m.fval), (1, 2, 0), atol=1e-8)
    assert m.valid


def test_use_array_call():
    inf = float("infinity")
    m = Minuit(
        func7,
        use_array_call=True,
        a=1,
        b=1,
        error_a=1,
        error_b=1,
        limit_a=(0, inf),
        limit_b=(0, inf),
        fix_a=False,
        fix_b=False,
        errordef=Minuit.LEAST_SQUARES,
        name=("a", "b"),
    )
    m.migrad()
    v = m.values
    assert_allclose((v["a"], v["b"]), (1, 1))
    m.hesse()
    c = m.covariance
    assert_allclose((c[("a", "a")], c[("b", "b")]), (1, 1))


def test_from_array_func_1():
    m = Minuit.from_array_func(
        func7, (2, 1), error=(1, 1), errordef=Minuit.LEAST_SQUARES
    )
    assert m.fitarg == {
        "x0": 2,
        "x1": 1,
        "error_x0": 1.0,
        "error_x1": 1.0,
        "fix_x0": False,
        "fix_x1": False,
        "limit_x0": None,
        "limit_x1": None,
    }
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1), rtol=1e-2)
    c = m.np_covariance()
    assert_allclose(np.diag(c), (1, 1), rtol=1e-2)


def test_from_array_func_2():
    m = Minuit.from_array_func(
        func7,
        (2, 1),
        grad=func7_grad,
        error=(0.5, 0.5),
        limit=((0, 2), (0, 2)),
        fix=(False, True),
        name=("a", "b"),
        errordef=Minuit.LEAST_SQUARES,
    )
    assert m.fitarg == {
        "a": 2,
        "b": 1,
        "error_a": 0.5,
        "error_b": 0.5,
        "fix_a": False,
        "fix_b": True,
        "limit_a": (0, 2),
        "limit_b": (0, 2),
    }
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1), rtol=1e-2)
    c = m.np_covariance()
    assert_allclose(c, ((1, 0), (0, 0)), rtol=1e-2)


def test_from_array_func_with_broadcasting():
    m = Minuit.from_array_func(
        func7, (1, 1), error=0.5, limit=(0, 2), errordef=Minuit.LEAST_SQUARES
    )
    assert m.fitarg == {
        "x0": 1,
        "x1": 1,
        "error_x0": 0.5,
        "error_x1": 0.5,
        "fix_x0": False,
        "fix_x1": False,
        "limit_x0": (0, 2),
        "limit_x1": (0, 2),
    }
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1))
    c = m.np_covariance()
    assert_allclose(np.diag(c), (1, 1))


def test_view_repr():
    m = Minuit(func0, x=1, y=2, pedantic=False)
    mid = id(m)
    assert (
        repr(m.values)
        == (
            """
<ValueView of Minuit at %x>
  x: 1.0
  y: 2.0
"""
            % mid
        ).strip()
    )
    assert (
        repr(m.args)
        == (
            """
<ArgsView of Minuit at %x>
  1.0
  2.0
"""
            % mid
        ).strip()
    )


def test_no_resume():
    m = Minuit(func0, pedantic=False)
    m.migrad()
    n = m.ncalls_total
    m.migrad()
    assert m.ncalls_total > n
    m.migrad(resume=False)
    assert m.ncalls_total == n

    m = Minuit(func0, grad=func0_grad, pedantic=False)
    m.migrad()
    n = m.ncalls_total
    k = m.ngrads_total
    m.migrad()
    assert m.ncalls_total > n
    assert m.ngrads_total > k
    m.migrad(resume=False)
    assert m.ncalls_total == n
    assert m.ngrads_total == k


def test_typo():
    with pytest.raises(RuntimeError):
        Minuit(func4, printlevel=0)
    with pytest.raises(RuntimeError):
        Minuit(lambda x: 0, y=1)
    with pytest.raises(RuntimeError):
        Minuit(lambda x: 0, error_y=1)
    with pytest.raises(RuntimeError):
        Minuit(lambda x: 0, limit_y=1)


def test_initial_guesses():
    m = Minuit(lambda x: 0, pedantic=False)
    assert m.values["x"] == 0
    assert m.errors["x"] == 0.1
    m = Minuit(lambda x: 0, limit_x=(1, 2), pedantic=False)
    assert m.values["x"] == 1.5
    assert m.errors["x"] == 1.5e-2


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_fix_param(grad):
    kwds = {"pedantic": False, "grad": grad}
    m = Minuit(func0, **kwds)
    assert m.narg == 2
    assert m.nfit == 2
    m.migrad()
    m.minos()
    assert_allclose(m.np_values(), (2, 5), rtol=1e-2)
    assert_allclose(m.np_errors(), (2, 1))
    assert_allclose(m.matrix(), ((4, 0), (0, 1)), atol=1e-4)
    for b in (True, False):
        assert_allclose(m.matrix(skip_fixed=b), [[4, 0], [0, 1]], atol=1e-4)

    # now fix y = 10
    m = Minuit(func0, y=10.0, fix_y=True, **kwds)
    assert m.narg == 2
    assert m.nfit == 1
    m.migrad()
    assert_allclose(m.np_values(), (2, 10), rtol=1e-2)
    assert_allclose(m.fval, 35)
    assert m.fixed.items() == [("x", False), ("y", True)]
    assert_allclose(m.matrix(skip_fixed=True), [[4]], atol=1e-4)
    assert_allclose(m.matrix(skip_fixed=False), [[4, 0], [0, 0]], atol=1e-4)

    assert m.fixed["x"] is False
    assert m.fixed["y"] is True
    m.fixed["x"] = True
    m.fixed["y"] = False
    assert m.narg == 2
    assert m.nfit == 1
    m.migrad()
    m.hesse()
    assert_allclose(m.np_values(), (2, 5), rtol=1e-2)
    assert_allclose(m.matrix(skip_fixed=True), [[1]], atol=1e-4)
    assert_allclose(m.matrix(skip_fixed=False), [[0, 0], [0, 1]], atol=1e-4)

    with pytest.raises(KeyError):
        m.fixed["a"]

    with pytest.warns(DeprecationWarning):
        assert m.is_fixed("x") is True
    with pytest.warns(DeprecationWarning):
        assert m.is_fixed("y") is False
    with pytest.warns(DeprecationWarning):
        with pytest.raises(KeyError):
            m.is_fixed("a")

    # fix by setting limits
    m = Minuit(func0, y=10.0, limit_y=(10, 10), pedantic=False)
    assert m.fixed["y"]
    assert m.narg == 2
    assert m.nfit == 1

    # initial value out of range is forced in range
    m = Minuit(func0, y=20.0, limit_y=(10, 10), pedantic=False)
    assert m.fixed["y"]
    assert m.values["y"] == 10
    assert m.narg == 2
    assert m.nfit == 1

    m.fixed[:] = True
    assert m.fixed.values() == [True, True]
    m.fixed[1:] = False
    assert m.fixed.values() == [True, False]
    assert m.fixed[:1] == [True]


def test_fitarg_oneside():
    m = Minuit(func4, y=10.0, fix_y=True, limit_x=(None, 20.0), pedantic=False)
    fitarg = m.fitarg
    assert_allclose(fitarg["y"], 10.0)
    assert fitarg["fix_y"]
    assert fitarg["limit_x"] == (-np.inf, 20)
    m.migrad()

    fitarg = m.fitarg

    assert_allclose(fitarg["x"], 2.0, atol=1e-2)
    assert_allclose(fitarg["y"], 10.0, atol=1e-2)
    assert_allclose(fitarg["z"], 7.0, atol=1e-2)

    assert "error_y" in fitarg
    assert "error_x" in fitarg
    assert "error_z" in fitarg

    assert fitarg["fix_y"]
    assert fitarg["limit_x"] == (-np.inf, 20)


def test_fitarg():
    m = Minuit(func4, y=10, fix_y=True, limit_x=(0, 20), pedantic=False)
    fitarg = m.fitarg
    assert_allclose(fitarg["y"], 10)
    assert fitarg["fix_y"] is True
    assert fitarg["limit_x"] == (0, 20)
    m.migrad()

    fitarg = m.fitarg

    assert_allclose(fitarg["y"], 10)
    assert_allclose(fitarg["x"], 2, atol=1e-2)
    assert_allclose(fitarg["z"], 7, atol=1e-2)

    assert "error_y" in fitarg
    assert "error_x" in fitarg
    assert "error_z" in fitarg

    assert fitarg["fix_y"] is True
    assert fitarg["limit_x"] == (0, 20)


@pytest.mark.parametrize("grad", (None, func0_grad))
@pytest.mark.parametrize("sigma", (1, 4))
def test_minos_all(grad, sigma):
    m = Minuit(func0, grad=func0_grad, pedantic=False)
    m.migrad()
    m.minos(sigma=sigma)
    assert m.merrors["x"].lower == approx(-sigma * 2)
    assert m.merrors["x"].upper == approx(sigma * 2)
    assert m.merrors["y"].upper == approx(sigma * 1)
    assert m.merrors[0].lower == approx(-sigma * 2)
    assert m.merrors[1].upper == approx(sigma * 1)
    assert m.merrors[-1].upper == approx(sigma * 1)

    with pytest.raises(KeyError):
        m.merrors["xy"]
    with pytest.raises(KeyError):
        m.merrors["z"]
    with pytest.raises(IndexError):
        m.merrors[-3]


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_minos_single(grad):
    m = Minuit(func0, grad=func0_grad, pedantic=False)

    m.strategy = 0
    m.migrad()
    assert m.ncalls < 15


def test_minos_single_fixed_raising():
    m = Minuit(func0, pedantic=False, fix_x=True)
    m.migrad()
    with pytest.warns(RuntimeWarning):
        ret = m.minos("x")
    assert ret is None


def test_minos_single_no_migrad():
    m = Minuit(func0, pedantic=False)
    with pytest.raises(RuntimeError):
        m.minos("x")


def test_minos_single_nonsense_variable():
    m = Minuit(func0, pedantic=False)
    m.migrad()
    with pytest.raises(RuntimeError):
        m.minos("nonsense")


@pytest.mark.parametrize("grad", (None, func5_grad))
def test_fixing_long_variable_name(grad):
    m = Minuit(
        func5,
        grad=grad,
        pedantic=False,
        fix_long_variable_name_really_long_why_does_it_has_to_be_this_long=True,
        long_variable_name_really_long_why_does_it_has_to_be_this_long=0,
    )
    m.migrad()


def test_initial_value():
    m = Minuit(func0, pedantic=False, x=1.0, y=2.0, error_x=3.0)
    assert_allclose(m.args[0], 1.0)
    assert_allclose(m.args[1], 2.0)
    assert_allclose(m.values["x"], 1.0)
    assert_allclose(m.values["y"], 2.0)
    assert_allclose(m.errors["x"], 3.0)


@pytest.mark.parametrize("grad", (None, func0_grad))
@pytest.mark.parametrize("sigma", (1, 2))
def test_mncontour(grad, sigma):
    m = Minuit(func0, grad=grad, pedantic=False, x=1.0, y=2.0, error_x=3.0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour("x", "y", numpoints=30, sigma=sigma)
    xminos_t = m.minos("x", sigma=sigma)["x"]
    yminos_t = m.minos("y", sigma=sigma)["y"]
    assert_allclose(xminos.upper, xminos_t.upper)
    assert_allclose(xminos.lower, xminos_t.lower)
    assert_allclose(yminos.upper, yminos_t.upper)
    assert_allclose(yminos.lower, yminos_t.lower)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_contour(grad):
    # FIXME: check the result
    m = Minuit(func0, grad=grad, pedantic=False, x=1.0, y=2.0, error_x=3.0)
    m.migrad()
    m.contour("x", "y")


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_profile(grad):
    # FIXME: check the result
    m = Minuit(func0, grad=grad, pedantic=False, x=1.0, y=2.0, error_x=3.0)
    m.migrad()
    m.profile("y")


@pytest.mark.parametrize("grad", (None, func0_grad))
def test_mnprofile(grad):
    # FIXME: check the result
    m = Minuit(func0, grad=grad, pedantic=False, x=1.0, y=2.0, error_x=3.0)
    m.migrad()
    m.mnprofile("y")


def test_mncontour_array_func():
    m = Minuit.from_array_func(Func8(), (0, 0), name=("x", "y"), pedantic=False)
    m.migrad()
    xminos, yminos, ctr = m.mncontour("x", "y", numpoints=30, sigma=1)
    xminos_t = m.minos("x", sigma=1)["x"]
    yminos_t = m.minos("y", sigma=1)["y"]
    assert_allclose(xminos.upper, xminos_t.upper)
    assert_allclose(xminos.lower, xminos_t.lower)
    assert_allclose(yminos.upper, yminos_t.upper)
    assert_allclose(yminos.lower, yminos_t.lower)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2


def test_profile_array_func():
    m = Minuit.from_array_func(Func8(), (0, 0), name=("x", "y"), pedantic=False)
    m.migrad()
    m.profile("y")


def test_mnprofile_array_func():
    m = Minuit.from_array_func(Func8(), (0, 0), name=("x", "y"), pedantic=False)
    m.migrad()
    m.mnprofile("y")


def test_fmin_uninitialized(capsys):
    m = Minuit(func0, pedantic=False)
    assert m.fmin is None
    assert m.fval is None


def test_reverse_limit():
    # issue 94
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    with pytest.raises(ValueError):
        Minuit(f, limit_x=(3.0, 2.0), pedantic=False)


@pytest.fixture
def minuit():
    m = Minuit(func0, pedantic=False)
    m.migrad()
    m.hesse()
    m.minos()
    return m


def test_args(minuit):
    expected = [2.0, 5.0]
    assert_allclose(minuit.args, expected, atol=5e-6)
    minuit.args[:] = [1, 2]
    assert_allclose(minuit.args, [1, 2])
    assert_allclose(minuit.args[0], 1)
    assert_allclose(minuit.args[-1], 2)
    assert_allclose(minuit.args[1:], [2])
    assert_allclose(minuit.args[:-1], [1])
    minuit.args[:1] = [3]
    assert_allclose(minuit.args, [3, 2])
    assert_allclose(minuit.args[1:], [2])


def test_values(minuit):
    expected = [2.0, 5.0]
    assert len(minuit.values) == 2
    assert_allclose(minuit.values.values(), expected, atol=5e-6)
    minuit.values[:] = expected
    assert minuit.values[:] == expected
    assert minuit.values[0] == 2
    assert minuit.values[1] == 5
    assert minuit.values["x"] == 2
    assert minuit.values["y"] == 5
    assert minuit.values[:1] == [2]
    minuit.values[1:] = [3]
    assert minuit.values[:] == [2, 3]
    assert minuit.values[-1] == 3
    minuit.values[:] = 7
    assert minuit.values[:] == [7, 7]
    with pytest.raises(KeyError):
        minuit.values["z"]
    with pytest.raises(IndexError):
        minuit.values[3]


def test_matrix(minuit):
    actual = minuit.matrix()
    expected = [[4.0, 0.0], [0.0, 1.0]]
    assert_allclose(actual, expected, atol=1e-7)


def test_matrix_correlation(minuit):
    actual = minuit.matrix(correlation=True)
    expected = [[1.0, 0.0], [0.0, 1.0]]
    assert_allclose(actual, expected, atol=1e-8)


def test_np_matrix(minuit):
    actual = minuit.np_matrix()
    expected = [[4.0, 0.0], [0.0, 1.0]]
    assert_allclose(actual, expected, atol=1e-7)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 2)


def test_np_matrix_correlation(minuit):
    actual = minuit.np_matrix(correlation=True)
    expected = [[1.0, 0.0], [0.0, 1.0]]
    assert_allclose(actual, expected, atol=1e-7)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 2)


def test_np_values(minuit):
    actual = minuit.np_values()
    expected = [2.0, 5.0]
    assert_allclose(actual, expected, atol=5e-6)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2,)


def test_np_errors(minuit):
    actual = minuit.np_errors()
    expected = [2.0, 1.0]
    assert_allclose(actual, expected, atol=1e-6)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2,)


def test_np_merrors(minuit):
    actual = minuit.np_merrors()
    # output format is [abs(down_delta), up_delta] following
    # the matplotlib convention in matplotlib.pyplot.errorbar
    down_delta = (-2, -1)
    up_delta = (2, 1)
    assert_allclose(actual, (np.abs(down_delta), up_delta), atol=5e-6)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 2)


def test_np_covariance(minuit):
    actual = minuit.np_covariance()
    expected = [[4.0, 0.0], [0.0, 1.0]]
    assert_allclose(actual, expected, atol=1e-7)
    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 2)


def test_chi2_fit():
    def chi2(x, y):
        return (x - 1) ** 2 + ((y - 2) / 3) ** 2

    m = Minuit(chi2, pedantic=False)
    m.migrad()
    assert_allclose(m.np_values(), (1, 2))
    assert_allclose(m.np_errors(), (1, 3))


def test_likelihood():
    # try:
    #     from numpy.random import default_rng
    #
    #     rng = default_rng(seed=1)
    #     data = rng.normal(1, 2, 100)
    # except ImportError:
    from numpy.random import randn, seed

    seed(1)
    data = 2 * randn(100) + 1

    def nll(mu, sigma):
        z = (data - mu) / sigma
        logp = -0.5 * z ** 2 - np.log(sigma)
        return -np.sum(logp)

    m = Minuit(nll, errordef=Minuit.LIKELIHOOD, limit_sigma=(0, None), pedantic=False)
    m.migrad()

    mu = np.mean(data)
    sigma = np.std(data)
    assert_allclose(m.np_values(), (mu, sigma), rtol=5e-3)
    s_mu = sigma / len(data) ** 0.5
    assert_allclose(m.np_errors(), (s_mu, 0.12047), rtol=1e-1)


def test_oneside():
    m_limit = Minuit(func0, limit_x=(None, 9), pedantic=False)
    m_nolimit = Minuit(func0, pedantic=False)
    # Solution: x=2., y=5.
    m_limit.migrad()
    m_nolimit.migrad()
    assert_allclose(
        list(m_limit.values.values()), list(m_nolimit.values.values()), atol=5e-3
    )


def test_oneside_outside():
    m = Minuit(func0, limit_x=(None, 1), pedantic=False)
    m.migrad()
    assert_allclose(m.values["x"], 1)


def test_ncalls():
    class Func:
        ncalls = 0

        def __call__(self, x):
            self.ncalls += 1
            return x ** 2

    # check that counting is accurate
    func = Func()
    m = Minuit(func, pedantic=False)
    m.migrad()
    assert m.ncalls_total == func.ncalls
    assert m.ncalls == func.ncalls
    m.migrad()
    assert m.ncalls_total == func.ncalls
    assert m.ncalls < func.ncalls
    func.ncalls = 0
    m.migrad(resume=False)
    assert m.ncalls_total == func.ncalls
    assert m.ncalls == func.ncalls

    ncalls_without_limit = m.ncalls
    # check that ncall argument limits function calls in migrad
    # note1: Minuit only checks the ncall counter in units of one iteration
    # step, therefore the call counter is in general not equal to ncall.
    # note2: If you pass ncall=0, Minuit uses a heuristic value that depends
    # on the number of parameters.
    m.migrad(ncall=1, resume=False)
    assert m.ncalls < ncalls_without_limit


def test_ngrads():
    class Func:
        ngrads = 0

        def __call__(self, x):
            return x ** 2

        def grad(self, x):
            self.ngrads += 1
            return [2 * x]

    # check that counting is accurate
    func = Func()
    m = Minuit(func, grad=func.grad, pedantic=False)
    m.migrad()
    assert m.ngrads > 0
    assert m.ngrads_total == func.ngrads
    assert m.ngrads == func.ngrads
    m.migrad()
    assert m.ncalls > 0
    assert m.ngrads_total == func.ngrads
    assert m.ngrads < func.ngrads
    func.ngrads = 0
    m.migrad(resume=False)
    assert m.ngrads_total == func.ngrads
    assert m.ngrads == func.ngrads

    # HESSE ignores analytical gradient
    m.hesse()
    assert m.ngrads == 0
    assert m.ngrads_total > 0


def test_errordef():
    m = Minuit(lambda x: x ** 2, pedantic=False, errordef=4)
    assert m.errordef == 4
    m.migrad()
    m.hesse()
    assert_allclose(m.errors["x"], 2)
    m.errordef = 1
    m.hesse()
    assert_allclose(m.errors["x"], 1)


def test_params():
    m = Minuit(
        func0,
        x=1,
        y=2,
        error_x=3,
        error_y=4,
        fix_x=True,
        limit_y=(None, 10),
        pedantic=False,
        errordef=Minuit.LEAST_SQUARES,
    )
    # these are the initial param states
    expected = [
        Param(0, "x", 1.0, 3.0, False, True, False, False, False, None, None),
        Param(1, "y", 2.0, 4.0, False, False, True, False, True, None, 10),
    ]
    assert m.params == expected

    m.migrad()
    assert m.init_params == expected

    expected = [
        Param(0, "x", 1.0, 3.0, False, True, False, False, False, None, None),
        Param(1, "y", 5.0, 1.0, False, False, True, False, True, None, 10),
    ]

    params = m.params
    for i, exp in enumerate(expected):
        p = params[i]
        assert set(p._fields) == set(exp._fields)
        for key in exp._fields:
            if key in ("value", "error"):
                assert_allclose(getattr(p, key), getattr(exp, key), rtol=1e-2)
            else:
                assert getattr(p, key) == getattr(exp, key)


def test_latex_matrix():
    m = Minuit.from_array_func(Func8(), (0, 0), name=("x", "y"), pedantic=False)
    m.migrad()
    # hotfix for ManyLinux 32Bit, where rounding changes result
    assert str(m.latex_matrix()) in (
        r"""%\usepackage[table]{xcolor} % include this for color
%\usepackage{rotating} % include this for rotate header
%\documentclass[xcolor=table]{beamer} % for beamer
\begin{tabular}{|c|c|c|}
\hline
\rotatebox{90}{} & \rotatebox{90}{x} & \rotatebox{90}{y}\\
\hline
x & \cellcolor[RGB]{250,100,100} 1 & \cellcolor[RGB]{250,174,174} 0.5\\
\hline
y & \cellcolor[RGB]{250,174,174} 0.5 & \cellcolor[RGB]{250,100,100} 1\\
\hline
\end{tabular}""",
        r"""%\usepackage[table]{xcolor} % include this for color
%\usepackage{rotating} % include this for rotate header
%\documentclass[xcolor=table]{beamer} % for beamer
\begin{tabular}{|c|c|c|}
\hline
\rotatebox{90}{} & \rotatebox{90}{x} & \rotatebox{90}{y}\\
\hline
x & \cellcolor[RGB]{250,100,100} 1 & \cellcolor[RGB]{250,175,175} 0.5\\
\hline
y & \cellcolor[RGB]{250,175,175} 0.5 & \cellcolor[RGB]{250,100,100} 1\\
\hline
\end{tabular}""",
    )


def test_non_analytical_function():
    class Func:
        i = 0

        def __call__(self, a):
            self.i += 1
            return self.i % 3

    m = Minuit(Func(), pedantic=False)
    fmin, _ = m.migrad()
    assert fmin.is_valid is False
    assert fmin.is_above_max_edm is True


def test_non_invertible():
    m = Minuit(lambda x, y: 0, pedantic=False)
    m.strategy = 0
    m.migrad()
    assert m.fmin.is_valid
    with pytest.warns(HesseFailedWarning):
        m.hesse()
    assert not m.fmin.is_valid
    with pytest.raises(RuntimeError):
        m.matrix()


def test_function_without_local_minimum():
    m = Minuit(lambda a: -a, pedantic=False)
    fmin, _ = m.migrad()
    assert fmin.is_valid is False
    assert fmin.is_above_max_edm is True


def test_function_with_maximum():
    def func(a):
        return -(a ** 2)

    m = Minuit(func, pedantic=False)
    fmin, _ = m.migrad()
    assert fmin.is_valid is False


def test_perfect_correlation():
    def func(a, b):
        return (a - b) ** 2

    m = Minuit(func, pedantic=False)
    fmin, _ = m.migrad()
    assert fmin.is_valid is True
    assert fmin.has_accurate_covar is False
    assert fmin.has_posdef_covar is False
    assert fmin.has_made_posdef_covar is True


def test_modify_param_state():
    m = Minuit(func0, x=1, y=2, error_y=1, fix_y=True, pedantic=False)
    m.migrad()
    assert_allclose(m.np_values(), [2, 2], atol=1e-4)
    assert_allclose(m.np_errors(), [2, 1], atol=1e-4)
    m.fixed["y"] = False
    m.values["x"] = 1
    m.errors["x"] = 1
    assert_allclose(m.np_values(), [1, 2], atol=1e-4)
    assert_allclose(m.np_errors(), [1, 1], atol=1e-4)
    m.migrad()
    assert_allclose(m.np_values(), [2, 5], atol=1e-4)
    assert_allclose(m.np_errors(), [2, 1], atol=1e-4)
    m.values["y"] = 6
    m.hesse()  # hesse ignores change in value if migrad was run before
    assert_allclose(m.np_values(), [2, 6], atol=1e-4)
    assert_allclose(m.np_errors(), [2, 1], atol=1e-4)


def test_view_lifetime():
    m = Minuit(func0, x=1, y=2, pedantic=False)
    val = m.values
    arg = m.args
    del m
    val["x"] = 3  # should not segfault
    assert val["x"] == 3
    arg[0] = 5  # should not segfault
    assert arg[0] == 5


def test_hesse_without_migrad():
    m = Minuit(lambda x: x ** 2 + x ** 4, x=0, errordef=0.5, pedantic=False)
    # second derivative: 12 x^2 + 2
    m.hesse()
    assert m.errors["x"] == approx(0.5 ** 0.5, abs=1e-4)
    m.values["x"] = 1
    m.hesse()
    assert m.errors["x"] == approx((1.0 / 14.0) ** 0.5, abs=1e-4)
    assert m.fmin is None

    m = Minuit(lambda x: 0, pedantic=False)
    with pytest.raises(RuntimeError):
        m.hesse()


def test_bad_functions():
    def throwing(x):
        raise RuntimeError("user message")

    def divide_by_zero(x):
        return 1 / 0

    def returning_nan(x):
        return np.nan

    def returning_garbage(x):
        return "foo"

    for func, expected in (
        (throwing, 'RuntimeError("user message")'),
        (divide_by_zero, "ZeroDivisionError"),
        (returning_nan, "result is NaN"),
        (returning_garbage, "TypeError"),
    ):
        m = Minuit(func, x=1, pedantic=False, throw_nan=True)
        with pytest.raises(RuntimeError) as excinfo:
            m.migrad()
        assert expected in excinfo.value.args[0]

    def returning_nan_array(x):
        return np.array([1, np.nan])

    def returning_garbage_array(x):
        return np.array([1, "foo"])

    def returning_noniterable(x):
        return 0

    for func, expected in (
        (throwing, 'RuntimeError("user message")'),
        (divide_by_zero, "ZeroDivisionError"),
        (returning_nan_array, "result is NaN"),
        (returning_garbage_array, "ValueError" if is_pypy else "TypeError"),
        (returning_noniterable, "TypeError"),
    ):
        m = Minuit.from_array_func(
            lambda x: 0, (1, 1), grad=func, pedantic=False, throw_nan=True
        )
        with pytest.raises(RuntimeError) as excinfo:
            m.migrad()
        if is_pypy and func is returning_garbage:
            pass
        else:
            assert expected in excinfo.value.args[0]


def test_issue_424():
    from iminuit import Minuit

    def lsq(x, y, z):
        return (x - 1) ** 2 + (y - 4) ** 2 / 2 + (z - 9) ** 2 / 3

    m = Minuit(lsq, errordef=1.0, x=0.0, y=0.0, z=0.0)
    m.migrad()

    m.fixed["x"] = True
    m.errors["x"] = 2
    m.hesse()
    assert m.fixed["x"] is True
    assert m.errors["x"] == 2
