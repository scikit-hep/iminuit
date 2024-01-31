from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from iminuit import Minuit
from iminuit.cost import (
    CostSum,
    UnbinnedNLL,
    BinnedNLL,
    ExtendedUnbinnedNLL,
    ExtendedBinnedNLL,
    LeastSquares,
    Constant,
    NormalConstraint,
    Template,
    multinominal_chi2,
    PerformanceWarning,
)
from iminuit.util import describe
from iminuit.typing import Annotated, Gt, Lt
from typing import Sequence
import pickle


def norm_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * (np.log(2 * np.pi) + z**2) - np.log(sigma)


def norm_pdf(x, mu, sigma):
    return np.exp(norm_logpdf(x, mu, sigma))


def norm_cdf(x, mu, sigma):
    from math import erf

    z = (x - mu) / (np.sqrt(2) * sigma)

    return (1 + np.vectorize(erf)(z)) * 0.5


def mvnorm(mux, muy, sx, sy, rho):
    stats = pytest.importorskip("scipy.stats")
    C = np.empty((2, 2))
    C[0, 0] = sx**2
    C[0, 1] = C[1, 0] = sx * sy * rho
    C[1, 1] = sy**2
    m = [mux, muy]
    return stats.multivariate_normal(m, C)


def expon_cdf(x, a):
    with np.errstate(over="ignore"):
        return -np.expm1(-x / a)


def numerical_cost_gradient(fcn):
    jacobi = pytest.importorskip("jacobi").jacobi
    return lambda *args: jacobi(lambda p: fcn(*p), args)[0]


def numerical_model_gradient(fcn):
    jacobi = pytest.importorskip("jacobi").jacobi
    return lambda x, *args: jacobi(lambda p: fcn(x, *p), args)[0].T


def numerical_extended_model_gradient(fcn):
    jacobi = pytest.importorskip("jacobi").jacobi

    def fn(x, *args):
        fint = jacobi(lambda p: fcn(x, *p)[0], args)[0]
        f = jacobi(lambda p: fcn(x, *p)[1], args)[0].T
        return fint, f

    return fn


@pytest.fixture
def unbinned():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    mle = (len(x), np.mean(x), np.std(x, ddof=1))
    return mle, x


@pytest.fixture
def binned(unbinned):
    mle, x = unbinned
    nx, xe = np.histogram(x, bins=50, range=(-3, 3))
    assert np.sum(nx == 0) > 0
    return mle, nx, xe


def logpdf(x, mu, sigma):
    return norm_logpdf(x, mu, sigma)


def pdf(x, mu, sigma):
    return norm_pdf(x, mu, sigma)


def cdf(x, mu, sigma):
    return norm_cdf(x, mu, sigma)


def scaled_cdf(x, n, mu, sigma):
    return n * norm_cdf(x, mu, sigma)


def line(x, a, b):
    return a + b * x


def test_norm_logpdf():
    norm = pytest.importorskip("scipy.stats").norm
    x = np.linspace(-3, 3)
    assert_allclose(norm_logpdf(x, 3, 2), norm.logpdf(x, 3, 2))


def test_norm_pdf():
    norm = pytest.importorskip("scipy.stats").norm
    x = np.linspace(-3, 3)
    assert_allclose(norm_pdf(x, 3, 2), norm.pdf(x, 3, 2))


def test_norm_cdf():
    norm = pytest.importorskip("scipy.stats").norm
    x = np.linspace(-3, 3)
    assert_allclose(norm_cdf(x, 3, 2), norm.cdf(x, 3, 2))


def test_describe():
    c = Constant(2.5)
    assert describe(c, annotations=True) == {}

    c = NormalConstraint("foo", 1.5, 0.1)
    assert describe(c, annotations=True) == {"foo": None}

    def model(
        x, foo: Annotated[float, Gt(1), Lt(2)], bar: float, baz: Annotated[float, 0:]
    ):
        return x

    assert describe(model, annotations=True) == {
        "x": None,
        "foo": (1, 2),
        "bar": None,
        "baz": (0, np.inf),
    }

    c = UnbinnedNLL([], model)
    assert describe(c, annotations=True) == {
        "foo": (1, 2),
        "bar": None,
        "baz": (0, np.inf),
    }

    c = BinnedNLL([], [1], model)
    assert describe(c, annotations=True) == {
        "foo": (1, 2),
        "bar": None,
        "baz": (0, np.inf),
    }


def test_Constant_1():
    c = Constant(2.5)
    assert c.value == 2.5
    assert c.ndata == 0
    assert c.has_grad
    assert_equal(c.grad(), [])


def test_Constant_2():
    c = NormalConstraint(("a", "b"), (1, 2), (3, 4)) + Constant(10.2)
    assert_allclose(c(1, 2), 10.2)
    assert_allclose(c(4, 2), 10.2 + 1)


def test_Constant_3():
    c = NormalConstraint(("a", "b"), (1, 2), (3, 4)) + Constant(10.2)
    ref = numerical_cost_gradient(c)
    assert c.has_grad
    assert_allclose(c.grad(1, 2), ref(1, 2))
    assert_allclose(c.grad(2, 3), ref(2, 3))
    assert_allclose(c.grad(-1, -2), ref(-1, -2))


@pytest.mark.parametrize("verbose", (0, 1))
@pytest.mark.parametrize("model", (logpdf, pdf))
@pytest.mark.parametrize("use_grad", (False, True))
def test_UnbinnedNLL(unbinned, verbose, model, use_grad):
    mle, x = unbinned

    cost = UnbinnedNLL(
        x,
        model,
        verbose=verbose,
        log=model is logpdf,
        grad=numerical_model_gradient(model),
    )
    assert cost.ndata == np.inf

    if use_grad:
        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(1, 2), ref(1, 2), rtol=1e-3)
        assert_allclose(cost.grad(-1, 3), ref(-1, 3), rtol=1e-3)

    m = Minuit(cost, mu=0, sigma=1, grad=use_grad)
    m.limits["sigma"] = (0, None)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, mle[1:], atol=1e-3)
    assert m.errors["mu"] == pytest.approx(1000**-0.5, rel=0.05)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0

    assert_equal(m.fmin.reduced_chi2, np.nan)

    if use_grad and verbose == 0:
        fi = cost.fisher_information(*m.values)
        cov = cost.covariance(*m.values)
        assert_allclose(cov, np.linalg.inv(fi))
        assert_allclose(np.diag(cov) ** 0.5, m.errors, rtol=5e-2)
        rho1 = m.covariance.correlation()[0, 1]
        rho2 = cov[0, 1] / (cov[0, 0] * cov[1, 1]) ** 0.5
        assert_allclose(rho1, rho2, atol=0.01)


@pytest.mark.parametrize("use_grad", (False, True))
def test_UnbinnedNLL_2D(use_grad):
    def model(x_y, mux, muy, sx, sy, rho):
        return mvnorm(mux, muy, sx, sy, rho).pdf(x_y.T)

    truth = 0.1, 0.2, 0.3, 0.4, 0.5
    x, y = mvnorm(*truth).rvs(size=100, random_state=1).T

    cost = UnbinnedNLL((x, y), model, grad=numerical_model_gradient(model))
    m = Minuit(cost, *truth, grad=use_grad)
    m.limits["sx", "sy"] = (0, None)
    m.limits["rho"] = (-1, 1)
    m.migrad()
    assert m.valid

    if use_grad:
        assert m.ngrad > 0

        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(*m.values), ref(*m.values), rtol=0.01)
    else:
        assert m.ngrad == 0

    assert_allclose(m.values, truth, atol=0.2)


@pytest.mark.parametrize("use_grad", (False, True))
def test_UnbinnedNLL_mask(use_grad):
    c = UnbinnedNLL(
        [1, np.nan, 2],
        lambda x, a: x + a**2,
        grad=lambda x, a: np.ones_like(x) * 2 * a,
    )
    assert c.mask is None

    assert np.isnan(c(0))
    c.mask = np.arange(3) != 1
    assert_equal(c.mask, (True, False, True))
    assert not np.isnan(c(0))

    assert_allclose(c.grad(0), [0])

    ref = numerical_cost_gradient(c)
    assert_allclose(c.grad(1), ref(1), atol=1e-3)


@pytest.mark.parametrize("log", (False, True))
def test_UnbinnedNLL_properties(log):
    c = UnbinnedNLL([1, 2], norm_logpdf if log else norm_pdf, log=log)

    x = np.linspace(0, 1)
    expected = norm_pdf(x, 1, 2)
    assert_allclose(c.pdf(x, 1, 2), expected)

    expected *= len(c.data)
    assert_allclose(c.scaled_pdf(x, 1, 2), expected)

    with pytest.raises(AttributeError):
        c.pdf = None

    with pytest.raises(AttributeError):
        c.scaled_pdf = None

    assert_equal(c.data, [1, 2])
    c.data = [2, 3]
    assert_equal(c.data, [2, 3])
    with pytest.raises(ValueError):
        c.data = [1, 2, 3]
    assert c.verbose == 0
    c.verbose = 1
    assert c.verbose == 1


@pytest.mark.parametrize("log", (False, True))
def test_UnbinnedNLL_visualize(log):
    pytest.importorskip("matplotlib")
    c = UnbinnedNLL([1, 2], norm_logpdf if log else norm_pdf, log=log)
    # auto-sampling
    c.visualize((1, 2))
    # linear spacing
    c.visualize((1, 2), model_points=10)
    # linear spacing and different binning
    c.visualize((1, 2), model_points=10, bins=20)
    # trigger log-spacing
    c = UnbinnedNLL([1, 100, 1000], norm_logpdf if log else norm_pdf, log=log)
    c.visualize((1, 2), model_points=10)
    c.visualize((1, 2), model_points=10, bins=20)
    # manual spacing
    c.visualize((1, 2), model_points=np.linspace(1, 1000))

    with pytest.warns(
        np.VisibleDeprecationWarning, match="keyword 'nbins' is deprecated"
    ):
        c.visualize((1, 2), nbins=20)


def test_UnbinnedNLL_visualize_2D():
    pytest.importorskip("matplotlib")

    def model(x_y, mux, muy, sx, sy, rho):
        return mvnorm(mux, muy, sx, sy, rho).pdf(x_y.T)

    truth = 0.1, 0.2, 0.3, 0.4, 0.5
    x, y = mvnorm(*truth).rvs(size=10, random_state=1).T

    c = UnbinnedNLL((x, y), model)

    with pytest.raises(ValueError, match="not implemented for multi-dimensional"):
        c.visualize(truth)


def test_UnbinnedNLL_pickle():
    c = UnbinnedNLL([1, 2], norm_pdf)
    b = pickle.dumps(c)
    c2 = pickle.loads(b)
    assert_equal(c.data, c2.data)


def test_UnbinnedNLL_annotated():
    def model(
        x: np.ndarray,
        mu: float,
        sigma: Annotated[float, 0 : np.inf],
    ) -> np.ndarray:
        return norm_pdf(x, mu, sigma)

    c = UnbinnedNLL([], model)
    m = Minuit(c, mu=1, sigma=1.5)
    assert m.limits[0] == (-np.inf, np.inf)
    assert m.limits[1] == (0, np.inf)


@pytest.mark.parametrize("verbose", (0, 1))
@pytest.mark.parametrize("model", (logpdf, pdf))
@pytest.mark.parametrize("use_grad", (False, True))
def test_ExtendedUnbinnedNLL(unbinned, verbose, model, use_grad):
    mle, x = unbinned

    log = model is logpdf

    def density(x, n, mu, sigma):
        if log:
            return n, np.log(n) + logpdf(x, mu, sigma)
        return n, n * pdf(x, mu, sigma)

    cost = ExtendedUnbinnedNLL(
        x,
        density,
        verbose=verbose,
        log=log,
        grad=numerical_extended_model_gradient(density),
    )
    assert cost.ndata == np.inf

    m = Minuit(cost, n=len(x), mu=0, sigma=1, grad=use_grad)
    m.limits["n"] = (0, None)
    m.limits["sigma"] = (0, None)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, mle, atol=1e-3)
    assert m.errors["mu"] == pytest.approx(1000**-0.5, rel=0.05)

    assert_equal(m.fmin.reduced_chi2, np.nan)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0


@pytest.mark.parametrize("use_grad", (False, True))
def test_ExtendedUnbinnedNLL_2D(use_grad):
    def model(x_y, n, mux, muy, sx, sy, rho):
        return n, n * mvnorm(mux, muy, sx, sy, rho).pdf(x_y.T)

    truth = 100.0, 0.1, 0.2, 0.3, 0.4, 0.5
    x, y = mvnorm(*truth[1:]).rvs(size=int(truth[0]), random_state=1).T

    cost = ExtendedUnbinnedNLL(
        (x, y), model, grad=numerical_extended_model_gradient(model)
    )

    m = Minuit(cost, *truth, grad=use_grad)
    m.limits["n", "sx", "sy"] = (0, None)
    m.limits["rho"] = (-1, 1)
    m.migrad()
    assert m.valid

    assert_allclose(m.values, truth, atol=0.5)

    if use_grad:
        assert m.ngrad > 0

        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(*m.values), ref(*m.values), rtol=0.01)
    else:
        assert m.ngrad == 0


def test_ExtendedUnbinnedNLL_mask():
    def model(x, a):
        return 1, x + a

    c = ExtendedUnbinnedNLL(
        [1, np.nan, 2], model, grad=numerical_extended_model_gradient(model)
    )
    assert c.ndata == np.inf

    assert np.isnan(c(0))
    c.mask = np.arange(3) != 1
    assert not np.isnan(c(0))
    assert c.ndata == np.inf

    ref = numerical_cost_gradient(c)
    assert_allclose(c.grad(0), ref(0))
    assert_allclose(c.grad(1.5), ref(1.5))


@pytest.mark.parametrize("log", (False, True))
def test_ExtendedUnbinnedNLL_properties(log):
    def log_model(x, s, mu, sigma):
        return s, np.log(s) + norm_logpdf(x, mu, sigma)

    def model(x, s, mu, sigma):
        n, y = log_model(x, s, mu, sigma)
        return n, np.exp(y)

    c = ExtendedUnbinnedNLL([1, 2, 3], log_model if log else model, log=log)

    x = np.linspace(0, 1)

    scale, expected = model(x, 1, 2, 3)
    assert_allclose(c.scaled_pdf(x, 1, 2, 3), expected)

    expected /= scale
    assert_allclose(c.pdf(x, 1, 2, 3), expected)

    with pytest.raises(AttributeError):
        c.scaled_pdf = None

    with pytest.raises(AttributeError):
        c.pdf = None


@pytest.mark.parametrize("log", (False, True))
def test_ExtendedUnbinnedNLL_visualize(log):
    pytest.importorskip("matplotlib")

    def log_model(x, s, mu, sigma):
        return s, np.log(s) + norm_logpdf(x, mu, sigma)

    def model(x, s, mu, sigma):
        return s, s * norm_pdf(x, mu, sigma)

    c = ExtendedUnbinnedNLL(
        [1, 2, 3],
        log_model if log else model,
        log=log,
    )

    c.visualize((1, 2, 3))


def test_ExtendedUnbinnedNLL_visualize_2D():
    pytest.importorskip("matplotlib")

    def model(x_y, n, mux, muy, sx, sy, rho):
        return n * 100, n * 100 * mvnorm(mux, muy, sx, sy, rho).pdf(x_y.T)

    truth = 1.0, 0.1, 0.2, 0.3, 0.4, 0.5
    x, y = mvnorm(*truth[1:]).rvs(size=int(truth[0] * 100)).T

    c = ExtendedUnbinnedNLL((x, y), model)

    with pytest.raises(ValueError, match="not implemented for multi-dimensional"):
        c.visualize(truth)


def test_ExtendedUnbinnedNLL_pickle():
    c = ExtendedUnbinnedNLL([1, 2], norm_pdf)
    b = pickle.dumps(c)
    c2 = pickle.loads(b)
    assert_equal(c.data, c2.data)


@pytest.mark.parametrize("verbose", (0, 1))
@pytest.mark.parametrize("use_grad", (False, True))
def test_BinnedNLL(binned, verbose, use_grad):
    mle, nx, xe = binned

    cost = BinnedNLL(nx, xe, cdf, verbose=verbose, grad=numerical_model_gradient(cdf))
    assert cost.ndata == len(nx)

    if use_grad:
        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(*mle[1:]), ref(*mle[1:]))
        assert_allclose(cost.grad(-1, 2), ref(-1, 2))

    m = Minuit(cost, mu=0, sigma=1, grad=use_grad)
    m.limits["sigma"] = (0, None)
    m.migrad()
    assert m.valid
    # binning loses information compared to unbinned case
    assert_allclose(m.values, mle[1:], rtol=0.15)
    assert m.errors["mu"] == pytest.approx(1000**-0.5, rel=0.05)
    assert m.ndof == len(nx) - 2

    assert_allclose(m.fmin.reduced_chi2, 1, atol=0.15)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0


def test_BinnedNLL_pulls(binned):
    mle, nx, xe = binned
    cost = BinnedNLL(nx, xe, cdf)
    m = Minuit(cost, mu=0, sigma=1)
    m.limits["sigma"] = (0, None)
    m.migrad()
    pulls = cost.pulls(m.values)
    assert np.nanmean(pulls) == pytest.approx(0, abs=0.05)
    assert np.nanvar(pulls) == pytest.approx(1, abs=0.2)


@pytest.mark.parametrize("use_grad", (False, True))
def test_BinnedNLL_weighted(use_grad):
    xe = np.array([0, 0.2, 0.4, 0.8, 1.5, 10])
    p = np.diff(expon_cdf(xe, 1))
    n = p * 1000
    c = BinnedNLL(n, xe, expon_cdf)

    assert_equal(c.data, n)
    m1 = Minuit(c, 1)
    m1.migrad()
    assert m1.values[0] == pytest.approx(1, rel=1e-2)

    # variance * 4 = (sigma * 2)^2, constructed so that
    # fitted parameter has twice the uncertainty
    w = np.transpose((n, 4 * n))
    c = BinnedNLL(w, xe, expon_cdf, grad=numerical_model_gradient(expon_cdf))
    assert_equal(c.data, w)

    if use_grad:
        ref = numerical_cost_gradient(c)
        assert_allclose(c.grad(*m1.values), ref(*m1.values))
        assert_allclose(c.grad(12), ref(12))

    m2 = Minuit(c, 1, grad=use_grad)
    m2.migrad()
    assert m2.valid
    assert m2.values[0] == pytest.approx(1, rel=1e-2)
    assert m2.errors[0] == pytest.approx(2 * m1.errors[0], rel=1e-2)

    if use_grad:
        assert m2.ngrad > 0
    else:
        assert m2.ngrad == 0


def test_BinnedNLL_bad_input_1():
    with pytest.raises(ValueError):
        BinnedNLL([1], [1], lambda x, a: 0)


def test_BinnedNLL_bad_input_2():
    with pytest.raises(ValueError):
        BinnedNLL([[[1]]], [1], lambda x, a: 0)


def test_BinnedNLL_bad_input_3():
    with pytest.raises(ValueError):
        BinnedNLL([[1, 2, 3]], [1], lambda x, a: 0)


def test_BinnedNLL_bad_input_4():
    with pytest.raises(ValueError, match="n must either have same dimension as xe"):
        BinnedNLL([[1, 2, 3]], [1, 2], lambda x, a: 0)


def test_BinnedNLL_ndof_zero():
    c = BinnedNLL([1], [0, 1], lambda x, scale: expon_cdf(x, scale))
    m = Minuit(c, scale=1)
    m.migrad()
    assert c.ndata == m.nfit
    assert np.isnan(m.fmin.reduced_chi2)


def test_BinnedNLL_bad_input_5():
    with pytest.raises(ValueError):
        BinnedNLL([[1, 2, 3]], [[1, 2], [1, 2, 3]], lambda x, a: 0)


def test_BinnedNLL_bad_input_6():
    with pytest.raises(ValueError):
        BinnedNLL(1, 2, lambda x, a: 0)


@pytest.mark.parametrize("use_grad", (False, True))
def test_BinnedNLL_2D(use_grad):
    truth = (0.1, 0.2, 0.3, 0.4, 0.5)
    x, y = mvnorm(*truth).rvs(size=1000, random_state=1).T

    w, xe, ye = np.histogram2d(x, y, bins=(20, 50))

    def model(xy, mux, muy, sx, sy, rho):
        return mvnorm(mux, muy, sx, sy, rho).cdf(xy.T)

    cost = BinnedNLL(w, (xe, ye), model, grad=numerical_model_gradient(model))
    assert cost.ndata == np.prod(w.shape)

    if use_grad:
        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(*truth), ref(*truth))

    m = Minuit(cost, *truth, grad=use_grad)
    m.limits["sx", "sy"] = (0, None)
    m.limits["rho"] = (-1, 1)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, truth, atol=0.05)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0

    assert cost.ndata == np.prod(w.shape)
    w2 = w.copy()
    w2[1, 1] += 1
    cost.n = w2
    assert cost(*m.values) > m.fval


def test_BinnedNLL_2D_with_zero_bins():
    truth = (0.1, 0.2, 0.3, 0.4, 0.5)
    x, y = mvnorm(*truth).rvs(size=1000, random_state=1).T

    w, xe, ye = np.histogram2d(x, y, bins=(50, 100), range=((-5, 5), (-5, 5)))
    assert np.mean(w == 0) > 0.25

    def model(xy, mux, muy, sx, sy, rho):
        return mvnorm(mux, muy, sx, sy, rho).cdf(xy.T)

    cost = BinnedNLL(w, (xe, ye), model)
    m = Minuit(cost, *truth)
    m.limits["sx", "sy"] = (0, None)
    m.limits["rho"] = (-1, 1)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, truth, atol=0.05)


def test_BinnedNLL_mask():
    c = BinnedNLL(
        [5, 1000, 1], [0, 1, 2, 3], expon_cdf, grad=numerical_model_gradient(expon_cdf)
    )
    assert c.ndata == 3

    ref = numerical_cost_gradient(c)
    assert_allclose(c.grad(2), ref(2))

    c_unmasked = c(1)
    c.mask = np.arange(3) != 1
    assert c(1) < c_unmasked
    assert c.ndata == 2

    ref = numerical_cost_gradient(c)
    assert_allclose(c.grad(2), ref(2))


def test_BinnedNLL_properties():
    def cdf(x, a, b):
        return 0

    c = BinnedNLL([1], [1, 2], cdf)
    assert c.cdf is cdf
    with pytest.raises(AttributeError):
        c.cdf = None
    assert_equal(c.n, [1])
    assert_equal(c.xe, [1, 2])
    c.n = [2]
    assert_equal(c.n, [2])
    with pytest.raises(ValueError):
        c.n = [1, 2]


def test_BinnedNLL_visualize():
    pytest.importorskip("matplotlib")

    c = BinnedNLL([1, 2], [1, 2, 3], expon_cdf)

    c.visualize((1,))

    c.mask = np.array([False, True])
    c.visualize((1,))


def test_BinnedNLL_visualize_2D():
    pytest.importorskip("matplotlib")

    truth = (0.1, 0.2, 0.3, 0.4, 0.5)
    x, y = mvnorm(*truth).rvs(size=10, random_state=1).T
    w, xe, ye = np.histogram2d(x, y, bins=(50, 100), range=((-5, 5), (-5, 5)))

    def model(xy, mux, muy, sx, sy, rho):
        return mvnorm(mux, muy, sx, sy, rho).cdf(xy.T)

    c = BinnedNLL(w, (xe, ye), model)

    c.visualize(truth)


def test_BinnedNLL_pickle():
    c = BinnedNLL([1], [1, 2], expon_cdf)
    b = pickle.dumps(c)
    c2 = pickle.loads(b)
    assert_equal(c.data, c2.data)


@pytest.mark.parametrize("use_pdf", ["approximate", "numerical"])
def test_BinnedNLL_with_pdf(use_pdf):
    if use_pdf == "numerical":
        pytest.importorskip("scipy")

    xe = np.array([0, 0.1, 0.2, 0.3])
    n = [1, 2, 3]
    c = BinnedNLL(n, xe, norm_pdf, use_pdf=use_pdf)

    par = [0.05, 1]
    ref = np.diff(norm_cdf(xe, *par)) * np.sum(n)
    assert_allclose(c.prediction(par), ref, rtol=1e-3)


def test_BinnedNLL_use_pdf_bad_value():
    with pytest.raises(ValueError):
        BinnedNLL([1, 2], [1, 2, 3], norm_pdf, use_pdf="foo")


@pytest.mark.parametrize("use_pdf", ["approximate", "numerical"])
def test_BinnedNLL_with_pdf_3D(use_pdf):
    xe = (np.array([0, 0.1]), np.array([0.1, 0.3, 0.4]), np.array([0, 0.1, 0.2, 0.3]))
    n = [[[1, 2, 3], [3, 4, 5]]]

    def pdf(r, mx, my, mz, sx, sy, sz):
        x, y, z = r
        return norm_pdf(x, mx, sx) * norm_pdf(y, my, sy) * norm_pdf(z, mz, sz)

    def cdf(r, mx, my, mz, sx, sy, sz):
        x, y, z = r
        return norm_cdf(x, mx, sx) * norm_cdf(y, my, sy) * norm_cdf(z, mz, sz)

    if use_pdf == "numerical":
        with pytest.raises(
            ValueError,
            match=(
                'use_pdf="numerical" is not supported '
                "for multidimensional histograms"
            ),
        ):
            BinnedNLL(n, xe, pdf, use_pdf=use_pdf)
    else:
        c1 = BinnedNLL(n, xe, pdf, use_pdf=use_pdf)
        c2 = BinnedNLL(n, xe, cdf)

        par = [0, 1, 2, 3, 4, 5]
        assert_allclose(c1.prediction(par), c2.prediction(par), rtol=1e-3)


@pytest.mark.parametrize("verbose", (0, 1))
@pytest.mark.parametrize("use_grad", (False, True))
def test_ExtendedBinnedNLL(binned, verbose, use_grad):
    mle, nx, xe = binned

    cost = ExtendedBinnedNLL(
        nx, xe, scaled_cdf, verbose=verbose, grad=numerical_model_gradient(scaled_cdf)
    )
    assert cost.ndata == len(nx)

    if use_grad:
        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(*mle), ref(*mle))

    m = Minuit(cost, n=mle[0], mu=0, sigma=1, grad=use_grad)
    m.limits["n"] = (0, None)
    m.limits["sigma"] = (0, None)
    m.migrad()
    assert m.valid
    # binning loses information compared to unbinned case
    assert_allclose(m.values, mle, rtol=0.15)
    assert m.errors["mu"] == pytest.approx(1000**-0.5, rel=0.05)
    assert m.ndof == len(nx) - 3

    assert_allclose(m.fmin.reduced_chi2, 1, 0.1)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0


@pytest.mark.parametrize("use_grad", (False, True))
def test_ExtendedBinnedNLL_weighted(use_grad):
    xe = np.array([0, 1, 10])
    n = np.diff(expon_cdf(xe, 1))
    m1 = Minuit(ExtendedBinnedNLL(n, xe, expon_cdf), 1)
    m1.migrad()
    assert_allclose(m1.values, (1,), rtol=1e-2)

    w = np.transpose((n, 4 * n))
    c = ExtendedBinnedNLL(w, xe, expon_cdf, grad=numerical_model_gradient(expon_cdf))
    if use_grad:
        ref = numerical_cost_gradient(c)
        assert_allclose(c.grad(1), ref(1), atol=1e-14)
        assert_allclose(c.grad(12), ref(12))
    m2 = Minuit(c, 1.1, grad=use_grad)
    m2.migrad()
    assert m2.valid
    assert_allclose(m2.values, (1,), rtol=1e-2)

    assert m2.errors[0] == pytest.approx(2 * m1.errors[0], rel=1e-2)

    if use_grad:
        assert m2.ngrad > 0
    else:
        assert m2.ngrad == 0


def test_ExtendedBinnedNLL_bad_input():
    with pytest.raises(ValueError):
        ExtendedBinnedNLL([1], [1], lambda x, a: 0)


@pytest.mark.parametrize("use_grad", (False, True))
def test_ExtendedBinnedNLL_2D(use_grad):
    truth = (1.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    x, y = mvnorm(*truth[1:]).rvs(size=int(truth[0] * 100), random_state=1).T

    w, xe, ye = np.histogram2d(x, y, bins=(10, 20))

    def model(xy, n, mux, muy, sx, sy, rho):
        return n * 100 * mvnorm(mux, muy, sx, sy, rho).cdf(np.transpose(xy))

    cost = ExtendedBinnedNLL(w, (xe, ye), model, grad=numerical_model_gradient(model))
    assert cost.ndata == np.prod(w.shape)

    if use_grad:
        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(*truth), ref(*truth))

    m = Minuit(cost, *truth, grad=use_grad)
    m.limits["n", "sx", "sy"] = (0, None)
    m.limits["rho"] = (-1, 1)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, truth, atol=0.1)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0


def test_ExtendedBinnedNLL_3D():
    norm = pytest.importorskip("scipy.stats").norm

    truth = (1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    n = int(truth[0] * 10000)
    x, y = mvnorm(*truth[1:-2]).rvs(size=n).T
    z = norm(truth[-2], truth[-1]).rvs(size=n)

    w, edges = np.histogramdd((x, y, z), bins=(5, 10, 20))

    def model(xyz, n, mux, muy, sx, sy, rho, muz, sz):
        *xy, z = xyz
        return (
            n
            * 10000
            * mvnorm(mux, muy, sx, sy, rho).cdf(np.transpose(xy))
            * norm(muz, sz).cdf(z)
        )

    cost = ExtendedBinnedNLL(w, edges, model)
    assert cost.ndata == np.prod(w.shape)
    m = Minuit(cost, *truth)
    m.limits["n", "sx", "sy", "sz"] = (0, None)
    m.limits["rho"] = (-1, 1)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, truth, atol=0.05)


def test_ExtendedBinnedNLL_mask():
    c = ExtendedBinnedNLL(
        [1, 1000, 2], [0, 1, 2, 3], expon_cdf, grad=numerical_model_gradient(expon_cdf)
    )
    assert c.ndata == 3

    c_unmasked = c(2)
    c.mask = np.arange(3) != 1
    assert c(2) < c_unmasked
    assert c.ndata == 2

    ref = numerical_cost_gradient(c)
    assert_allclose(c.grad(2), ref(2))


def test_ExtendedBinnedNLL_properties():
    def cdf(x, a):
        return 0

    c = ExtendedBinnedNLL([1], [1, 2], cdf)
    assert c.scaled_cdf is cdf


def test_ExtendedBinnedNLL_visualize():
    pytest.importorskip("matplotlib")

    def model(x, s, slope):
        return s * expon_cdf(x, slope)

    c = ExtendedBinnedNLL([1], [1, 2], model)
    c.visualize((1, 2))


def test_ExtendedBinnedNLL_visualize_2D():
    pytest.importorskip("matplotlib")

    truth = (1.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    x, y = mvnorm(*truth[1:]).rvs(size=int(truth[0] * 1000), random_state=1).T

    w, xe, ye = np.histogram2d(x, y, bins=(10, 20))

    def model(xy, n, mux, muy, sx, sy, rho):
        return n * 1000 * mvnorm(mux, muy, sx, sy, rho).cdf(np.transpose(xy))

    c = ExtendedBinnedNLL(w, (xe, ye), model)

    c.visualize(truth)


def test_ExtendedBinnedNLL_pickle():
    c = BinnedNLL([1], [1, 2], expon_cdf)
    b = pickle.dumps(c)
    c2 = pickle.loads(b)
    assert_equal(c.data, c2.data)


def test_ExtendedBinnedNLL_weighted_pulls():
    rng = np.random.default_rng(1)
    xe = np.linspace(0, 5, 500)
    p = np.diff(expon_cdf(xe, 2))
    m = p * 100000
    n = rng.poisson(m * 4) / 4
    nvar = m * 4 / 4**2
    w = np.transpose((n, nvar))
    c = ExtendedBinnedNLL(w, xe, lambda x, n, s: n * expon_cdf(x, s))
    m = Minuit(c, np.sum(n), 2)
    m.limits["n"] = (0.1, None)
    m.limits["s"] = (0.1, None)
    m.migrad()
    assert m.valid
    pulls = c.pulls(m.values)
    assert np.nanmean(pulls) == pytest.approx(0, abs=0.1)
    assert np.nanvar(pulls) == pytest.approx(1, abs=0.2)


@pytest.mark.parametrize(
    "loss,use_grad",
    [
        ("linear", False),
        ("linear", True),
        ("soft_l1", False),
        ("soft_l1", True),
        (np.arctan, False),
    ],
)
@pytest.mark.parametrize("verbose", (0, 1))
def test_LeastSquares(loss, verbose, use_grad):
    rng = np.random.default_rng(1)

    x = np.linspace(0, 1, 1000)
    ye = 0.1
    y = rng.normal(2 * x + 1, ye)

    def model(x, a, b):
        return a + b * x

    cost = LeastSquares(
        x,
        y,
        ye,
        model,
        loss=loss,
        verbose=verbose,
        grad=numerical_model_gradient(model),
    )
    assert cost.ndata == len(x)

    if use_grad:
        ref = numerical_cost_gradient(cost)
        assert_allclose(cost.grad(1, 2), ref(1, 2))

    m = Minuit(cost, a=0, b=0, grad=use_grad)
    m.migrad()
    assert_allclose(m.values, (1, 2), rtol=0.05)
    assert cost.loss == loss
    if loss != "linear":
        cost.loss = "linear"
        assert cost.loss != loss
    m.migrad()
    assert_allclose(m.values, (1, 2), rtol=0.05)
    assert m.ndof == len(x) - 2

    assert_allclose(m.fmin.reduced_chi2, 1, atol=5e-2)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0


def test_LeastSquares_2D():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    z = 1.5 * x + 0.2 * y
    ze = 1.5

    def model(xy, a, b):
        x, y = xy
        return a * x + b * y

    c = LeastSquares((x, y), z, ze, model, grad=numerical_model_gradient(model))
    assert c.ndata == 3

    ref = numerical_cost_gradient(c)
    assert_allclose(c.grad(1, 2), ref(1, 2))

    assert_equal(c.x, (x, y))
    assert_equal(c.y, z)
    assert_equal(c.yerror, ze)
    assert_allclose(c(1.5, 0.2), 0.0)
    assert_allclose(c(2.5, 0.2), np.sum(((z - 2.5 * x - 0.2 * y) / ze) ** 2))
    assert_allclose(c(1.5, 1.2), np.sum(((z - 1.5 * x - 1.2 * y) / ze) ** 2))

    c.y = 2 * z
    assert_equal(c.y, 2 * z)
    c.x = (y, x)
    assert_equal(c.x, (y, x))


def test_LeastSquares_bad_input():
    with pytest.raises(ValueError, match="shape mismatch"):
        LeastSquares([1, 2], [], [1], lambda x, a: 0)

    with pytest.raises(ValueError, match="shape mismatch"):
        LeastSquares([1, 2], [3, 4, 5], [1], lambda x, a: 0)

    with pytest.raises(ValueError, match="unknown loss"):
        LeastSquares([1], [1], [1], lambda x, a: 0, loss="foo")

    with pytest.raises(ValueError, match="loss must be str or LossFunction"):
        LeastSquares([1], [1], [1], lambda x, a: 0, loss=[1, 2, 3])


@pytest.mark.parametrize("use_grad", (False, True))
def test_LeastSquares_mask(use_grad):
    c = LeastSquares(
        [1, 2, 3],
        [3, np.nan, 4],
        [1, 1, 1],
        lambda x, a: x + a,
        grad=lambda x, a: np.ones_like(x),
    )
    assert c.ndata == 3
    assert np.isnan(c(0))
    assert np.all(np.isnan(c.grad(1)))

    m = Minuit(c, 1, grad=use_grad)
    assert m.ndof == 2
    m.migrad()
    assert not m.valid

    c.mask = np.arange(3) != 1
    assert c.ndata == 2
    assert not np.isnan(c(0))
    assert not np.any(np.isnan(c.grad(0)))

    assert m.ndof == 1
    m.migrad()
    assert m.valid
    assert_equal(m.values, [1.5])


def test_LeastSquares_mask_2():
    c = LeastSquares([1, 2], [1, 5], 1, lambda x, a: a * x)
    assert c(2) == pytest.approx(2)
    c.mask = [False, True]
    assert c(2) == pytest.approx(1)
    c.x = [2, 2]
    c.y = [4, 4]
    c.yerror = [2, 2]
    assert c(2) == pytest.approx(0)
    assert c(1) == pytest.approx(1)


def test_LeastSquares_properties():
    def model(x, a):
        return a

    c = LeastSquares(1, 2, 3, model)
    assert_equal(c.x, [1])
    assert_equal(c.y, [2])
    assert_equal(c.yerror, [3])
    assert c.model is model
    with pytest.raises(AttributeError):
        c.model = model
    with pytest.raises(ValueError):
        c.x = [1, 2]
    with pytest.raises(ValueError):
        c.y = [1, 2]
    with pytest.raises(ValueError):
        c.yerror = [1, 2]


def test_LeastSquares_visualize():
    pytest.importorskip("matplotlib")

    c = LeastSquares([1, 2], [2, 3], 0.1, line)

    # auto-sampling
    c.visualize((1, 2))
    # linear spacing
    c.visualize((1, 2), model_points=10)
    # trigger use of log-spacing
    c = LeastSquares([1, 10, 100], [2, 3, 4], 0.1, line)
    c.visualize((1, 2), model_points=10)
    # manual spacing
    c.visualize((1, 2), model_points=np.linspace(1, 100))


def test_LeastSquares_visualize_2D():
    pytest.importorskip("matplotlib")

    c = LeastSquares([[1, 2]], [[2, 3]], 0.1, line)

    with pytest.raises(ValueError, match="not implemented for multi-dimensional"):
        c.visualize((1, 2))


def test_LeastSquares_pickle():
    c = LeastSquares([1, 2], [2, 3], 0.1, line)

    b = pickle.dumps(c)
    c2 = pickle.loads(b)
    assert_equal(c.data, c2.data)
    assert c.model is c2.model


def test_LeastSquares_pulls():
    c = LeastSquares([1, 2], [2, 3], 0.1, line)
    # pulls are zero for parameters 1, 1
    assert_equal(c.pulls((1, 1)), [0, 0])
    # correct slope but line offset by 1 with error 0.1 produces pulls [10, 10]
    assert_equal(c.pulls((0, 1)), [10, 10])
    c.mask = [True, False]
    assert_equal(c.pulls((0, 1)), [10, np.nan])


@pytest.mark.parametrize("use_grad", (False, True))
def test_CostSum_1(use_grad):
    def model1(x, a):
        return a + x

    def grad1(x, a):
        return np.ones((1, len(x)))

    def model2(x, b, a):
        return a + b * x

    def grad2(x, b, a):
        g = np.empty((2, len(x)))
        g[0] = x
        g[1] = 1
        return g

    def model3(x, c):
        return c * np.ones_like(x)

    def grad3(x, c):
        return np.ones((1, len(x)))

    lsq1 = LeastSquares(1, 2, 3, model1, grad=grad1)
    assert describe(lsq1) == ["a"]

    if use_grad:
        assert_allclose(lsq1.grad(2), numerical_cost_gradient(lsq1)(2))

    lsq2 = LeastSquares(1, 3, 4, model2, grad=grad2)
    assert describe(lsq2) == ["b", "a"]

    if use_grad:
        assert_allclose(lsq2.grad(2, 3), numerical_cost_gradient(lsq2)(2, 3))

    lsq3 = LeastSquares(1, 1, 1, model3, grad=grad3)
    assert describe(lsq3) == ["c"]

    if use_grad:
        assert_allclose(lsq3.grad(4), numerical_cost_gradient(lsq3)(4))

    lsq12 = lsq1 + lsq2
    assert lsq12._items == [lsq1, lsq2]
    assert isinstance(lsq12, CostSum)
    assert isinstance(lsq1, LeastSquares)
    assert isinstance(lsq2, LeastSquares)
    assert describe(lsq12) == ["a", "b"]
    assert lsq12.ndata == 2

    assert lsq12(1, 2) == lsq1(1) + lsq2(2, 1)

    if use_grad:
        a = 2
        b = 3
        ref = np.zeros(2)
        ref[0] += lsq1.grad(a)[0]
        ref[[1, 0]] += lsq2.grad(b, a)
        assert_allclose(lsq12.grad(a, b), ref)

    lsq121 = lsq12 + lsq1
    assert lsq121._items == [lsq1, lsq2, lsq1]
    assert describe(lsq121) == ["a", "b"]
    assert lsq121.ndata == 3

    if use_grad:
        a = 2
        b = 3
        ref = np.zeros(2)
        ref[0] += lsq1.grad(a)[0]
        ref[[1, 0]] += lsq2.grad(b, a)
        ref[0] += lsq1.grad(a)[0]
        assert_allclose(lsq121.grad(a, b), ref)

    lsq312 = lsq3 + lsq12
    assert lsq312._items == [lsq3, lsq1, lsq2]
    assert describe(lsq312) == ["c", "a", "b"]
    assert lsq312.ndata == 3

    if use_grad:
        a = 2
        b = 3
        c = 4
        ref = np.zeros(3)
        ref[0] += lsq3.grad(c)[0]
        ref[1] += lsq1.grad(a)[0]
        ref[[2, 1]] += lsq2.grad(b, a)
        assert_allclose(lsq312.grad(c, a, b), ref)

    lsq31212 = lsq312 + lsq12
    assert lsq31212._items == [lsq3, lsq1, lsq2, lsq1, lsq2]
    assert describe(lsq31212) == ["c", "a", "b"]
    assert lsq31212.ndata == 5

    if use_grad:
        a = 2
        b = 3
        c = 4
        ref = np.zeros(3)
        ref[:] += lsq312.grad(c, a, b)
        ref[1:] += lsq12.grad(a, b)
        assert_allclose(lsq31212.grad(c, a, b), ref)

    lsq31212 += lsq1
    assert lsq31212._items == [lsq3, lsq1, lsq2, lsq1, lsq2, lsq1]
    assert describe(lsq31212) == ["c", "a", "b"]
    assert lsq31212.ndata == 6

    m = Minuit(lsq12, a=0, b=0, grad=use_grad)
    m.migrad()
    assert m.parameters == ("a", "b")
    assert_allclose(m.values, (1, 2))
    assert_allclose(m.errors, (3, 5))
    assert_allclose(m.covariance, ((9, -9), (-9, 25)), atol=1e-10)

    if use_grad:
        assert m.ngrad > 0
    else:
        assert m.ngrad == 0


def test_CostSum_2():
    ref = NormalConstraint("a", 1, 2), NormalConstraint(("b", "a"), (1, 1), (2, 2))
    cs = ref[0] + ref[1]
    assert cs.ndata == 3
    assert isinstance(cs, Sequence)
    assert len(cs) == 2
    assert cs[0] is ref[0]
    assert cs[1] is ref[1]
    for c, r in zip(cs, ref):
        assert c is r
    assert cs.index(ref[0]) == 0
    assert cs.index(ref[1]) == 1
    assert cs.count(ref[0]) == 1


def test_CostSum_3():
    def line_np(x, par):
        return par[0] + par[1] * x

    lsq = LeastSquares([1, 2, 3], [3, 4, 5], 1, line_np)
    con = NormalConstraint("par", (1, 1), (1, 1))
    cs = sum([lsq, con])
    assert cs((1, 1)) == lsq((1, 1)) + con((1, 1))

    cs = 1.5 + lsq + con
    assert cs((1, 1)) == lsq((1, 1)) + con((1, 1)) + 1.5


def test_CostSum_4():
    pytest.importorskip("scipy.special")

    t = Template([1, 2], [1, 2, 3], [[1, 1], [0, 1]], method="asy")
    assert t.errordef == Minuit.LIKELIHOOD

    m1 = Minuit(t, 1, 1)
    m1.migrad()

    cs = CostSum(t)
    assert cs.errordef == Minuit.LEAST_SQUARES

    m2 = Minuit(cs, 1, 1)
    m2.migrad()

    assert_allclose(m1.errors, m2.errors)


def test_CostSum_visualize():
    pytest.importorskip("matplotlib")

    lsq = LeastSquares([1, 2, 3], [3, 4, 5], 1, line)
    con = NormalConstraint(("a", "b"), (1, 1), (1, 1))
    c = lsq + con + 1
    c.visualize((1, 2))


def test_NormalConstraint_1():
    c1 = NormalConstraint("a", 1, 1.5)
    c2 = NormalConstraint(("a", "b"), (1, 2), (3, 4))
    c3 = NormalConstraint(("a", "b"), (1, 2), ((1, 0.5), (0.5, 4)))

    assert describe(c1) == ["a"]
    assert describe(c2) == ["a", "b"]
    assert describe(c3) == ["a", "b"]
    assert c1.ndata == 1
    assert c2.ndata == 2
    assert c3.ndata == 2
    assert c1.has_grad
    assert c2.has_grad
    assert c3.has_grad
    assert c1.value == 1
    assert_equal(c1.covariance, [1.5**2])
    assert_equal(c2.value, (1, 2))
    assert_equal(c2.covariance, (9, 16))
    assert_equal(c3.value, (1, 2))
    assert_equal(c3.covariance, ((1, 0.5), (0.5, 4)))

    assert_allclose(c1(1), 0)
    assert_allclose(c1(1 + 1.5), 1)
    assert_allclose(c1(1 - 1.5), 1)

    assert_allclose(c2(1, 2), 0)
    assert_allclose(c2(1 + 3, 2), 1)
    assert_allclose(c2(1 - 3, 2), 1)
    assert_allclose(c2(1, 2 + 4), 1)
    assert_allclose(c2(1, 2 - 4), 1)

    def ref(x, y):
        d = np.subtract((x, y), (1, 2))
        c = ((1, 0.5), (0.5, 4))
        cinv = np.linalg.inv(c)
        return d.T @ cinv @ d

    assert_allclose(c3(1, 2), 0)
    assert_allclose(c3(2, 2), ref(2, 2))
    assert_allclose(c3(3, 4), ref(3, 4))
    assert_allclose(c3(-1, -2), ref(-1, -2))

    ref = numerical_cost_gradient(c1)
    assert_allclose(c1.grad(1), [0])
    assert_allclose(c1.grad(2), ref(2))
    assert_allclose(c1.grad(-2), ref(-2))

    ref = numerical_cost_gradient(c2)
    assert_allclose(c2.grad(1, 2), [0, 0])
    assert_allclose(c2.grad(2, 3), ref(2, 3))
    assert_allclose(c2.grad(-2, -4), ref(-2, -4))

    ref = numerical_cost_gradient(c3)
    assert_allclose(c3.grad(1, 2), [0, 0])
    assert_allclose(c3.grad(2, 3), ref(2, 3))
    assert_allclose(c3.grad(-2, -4), ref(-2, -4))

    c1.value = 2
    c1.covariance = 4
    assert_equal(c1.value, 2)
    assert_equal(c1.covariance, 4)
    assert_allclose(c1(2), 0)
    assert_allclose(c1(4), 1)
    assert_allclose(c1(0), 1)

    c2.value = (2, 3)
    c2.covariance = (1, 4)
    assert_equal(c2.value, (2, 3))
    assert_equal(c2.covariance, (1, 4))
    assert_allclose(c2(2, 3), 0)
    assert_allclose(c2(1, 3), 1)
    assert_allclose(c2(2, 5), 1)

    c3.value = (2, 3)
    c3.covariance = [(4, 1), (1, 5)]
    assert_equal(c3.value, (2, 3))
    assert_equal(c3.covariance, [(4, 1), (1, 5)])

    def ref(x, y):
        d = np.subtract((x, y), (2, 3))
        c = ((4, 1), (1, 5))
        cinv = np.linalg.inv(c)
        return d.T @ cinv @ d

    assert_allclose(c3(2, 3), 0)
    assert_allclose(c3(1, 3), ref(1, 3))
    assert_allclose(c3(2, 5), ref(2, 5))


@pytest.mark.parametrize("use_grad", (False, True))
def test_NormalConstraint_2(use_grad):
    c1 = NormalConstraint("a", 1, 1)
    c2 = c1 + NormalConstraint("a", 1, 0.1)

    m = Minuit(c1, 0, grad=use_grad)
    m.migrad()
    if use_grad:
        assert m.ngrad > 0
    assert_allclose(m.values, (1,), atol=1e-2)
    assert_allclose(m.errors, (1,), rtol=1e-2)

    m = Minuit(c2, 0, grad=use_grad)
    m.migrad()
    if use_grad:
        assert m.ngrad > 0
    assert_allclose(m.values, (1,), atol=1e-2)
    assert_allclose(m.errors, (0.1,), rtol=1e-2)


def test_NormalConstraint_3():
    lsq1 = NormalConstraint(("a", "b"), (1, 2), (2, 2))
    lsq2 = lsq1 + NormalConstraint("b", 2, 0.1) + NormalConstraint("a", 1, 0.01)
    sa = 0.1
    sb = 0.02
    rho = 0.5
    cov = ((sa**2, rho * sa * sb), (rho * sa * sb, sb**2))
    lsq3 = lsq1 + NormalConstraint(("a", "b"), (1, 2), cov)
    assert describe(lsq1) == ["a", "b"]
    assert describe(lsq2) == ["a", "b"]
    assert describe(lsq3) == ["a", "b"]
    assert lsq1.ndata == 2
    assert lsq2.ndata == 4

    m = Minuit(lsq1, 0, 0)
    m.migrad()
    assert_allclose(m.values, (1, 2), atol=1e-3)
    assert_allclose(m.errors, (2, 2), rtol=1e-3)

    m = Minuit(lsq2, 0, 0)
    m.migrad()
    assert_allclose(m.values, (1, 2), atol=1e-3)
    assert_allclose(m.errors, (0.01, 0.1), rtol=1e-2)

    m = Minuit(lsq3, 0, 0)
    m.migrad()
    assert_allclose(m.values, (1, 2), atol=1e-3)
    assert_allclose(m.errors, (sa, sb), rtol=1e-2)
    assert_allclose(m.covariance, cov, rtol=1e-2)


def test_NormalConstraint_visualize():
    pytest.importorskip("matplotlib")

    c = NormalConstraint(("a", "b"), (1, 2), (3, 4))
    c.visualize((1, 2))

    c = NormalConstraint(("a", "b"), (1, 2), np.eye(2))
    c.visualize((1, 2))


def test_NormalConstraint_pickle():
    c = NormalConstraint(("a", "b"), (1, 2), (3, 4))
    b = pickle.dumps(c)
    c2 = pickle.loads(b)
    assert describe(c) == describe(c2) and describe(c) == ["a", "b"]
    assert_equal(c.value, c2.value)
    assert_equal(c.covariance, c2.covariance)


def test_NormalConstraint_bad_input_1():
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        NormalConstraint("par", [[[1, 2]]], np.eye(2))


def test_NormalConstraint_bad_input_2():
    with pytest.raises(ValueError, match="not match size of args"):
        NormalConstraint(["a", "b", "c"], [1, 2], np.eye(2))


def test_NormalConstraint_bad_input_3():
    with pytest.raises(ValueError, match="size of error does not match size of value"):
        NormalConstraint(["a", "b"], [1, 2], np.eye(3))


def test_NormalConstraint_bad_input_4():
    with pytest.raises(ValueError, match="positive definite"):
        NormalConstraint(["a", "b"], [1, 2], [[1, 1], [1, 1]])


def test_NormalConstraint_bad_input_5():
    n = NormalConstraint(["a", "b"], [1, 2], [[1, 0], [0, 1]])

    with pytest.raises(ValueError, match="positive definite"):
        n.covariance = [[1, 1], [1, 1]]


def test_NormalConstraint_bad_input_6():
    with pytest.raises(ValueError, match="cannot have more than two dimensions"):
        NormalConstraint(["a", "b"], [1, 2], np.ones((2, 2, 2)))


dtypes_to_test = [np.float32]
if hasattr(np, "float128"):  # not available on all platforms
    dtypes_to_test.append(np.float128)


def test_multinominal_chi2():
    zero = np.array(0)
    one = np.array(1)

    assert multinominal_chi2(zero, zero) == 0
    assert multinominal_chi2(zero, one) == 0
    assert multinominal_chi2(one, zero) == pytest.approx(1416, abs=1)
    n = np.array([(0.0, 0.0)])
    assert_allclose(multinominal_chi2(n, zero), 0)
    assert_allclose(multinominal_chi2(n, one), 0)


@pytest.mark.skipif(
    not hasattr(np, "float128"), reason="float128 not available on all platforms"
)
def test_model_float128():
    def model(x, a):
        x = x.astype(np.float128)
        return a + x

    for cost in (
        UnbinnedNLL([1], model),
        ExtendedUnbinnedNLL([1], lambda x, a: (1, model(x, a))),
        BinnedNLL([1], [1, 2], model),
        BinnedNLL([[1, 1]], [1, 2], model),
        ExtendedBinnedNLL([1], [1, 2], model),
        ExtendedBinnedNLL([[1, 1]], [1, 2], model),
        LeastSquares([1.0], [2.0], [3.0], model),
        LeastSquares([1.0], [2.0], [3.0], model, loss="soft_l1"),
    ):
        assert cost(1).dtype == np.float128

        Minuit(cost, a=0).migrad()  # should not raise


def test_model_performance_warning():
    def model(x, a):
        return [0 for xi in x]

    with pytest.warns(PerformanceWarning):
        BinnedNLL([1], [1.0, 2.0], model)(1)

    with pytest.warns(PerformanceWarning):
        ExtendedBinnedNLL([1], [1.0, 2.0], model)(1)

    with pytest.warns(PerformanceWarning):
        UnbinnedNLL([1], model)(1)

    with pytest.warns(PerformanceWarning):
        ExtendedUnbinnedNLL([1], lambda x, a: (1, model(x, a)))(1)


@pytest.mark.parametrize("cls", (BinnedNLL, ExtendedBinnedNLL))
def test_update_data_with_mask(cls):
    xe = np.arange(0, 4)
    nx = np.diff(expon_cdf(xe, 1))
    nx[0] += 1
    c = cls(nx.copy(), xe, expon_cdf)

    c.mask = [False, True, True]
    assert c(1) == 0
    nx[0] += 1
    c.n = nx
    assert c(1) == 0
    nx[1] += 1
    c.n = nx
    assert c(1) != 0
    nx[0] -= 2
    c.mask = (True, False, True)
    c.n = nx
    assert c(1) == 0


@pytest.mark.parametrize("method", ("jsc", "asy", "da"))
def test_Template(method):
    if method == "asy":
        pytest.importorskip("scipy.special")

    xe = np.array([0, 1, 2, 3])
    t = np.array([[1, 1, 0], [0, 1, 3]])
    n = t[0] + t[1]

    c = Template(n, xe, t, method=method)
    m = Minuit(c, 1, 1)
    assert m.limits[0] == (0, np.inf)
    assert m.limits[1] == (0, np.inf)
    m.migrad()
    assert m.valid
    assert m.ndof == 1
    if method == "asy":
        assert c.errordef == 0.5
        assert_equal(m.fmin.reduced_chi2, np.nan)
        # asy produces values far away from truth in this case
        assert_allclose(m.values, [1, 3], atol=0.2)
    else:
        assert c.errordef == 1.0
        assert_allclose(m.fval, 0, atol=1e-4)
        assert_allclose(m.fmin.reduced_chi2, 0, atol=1e-5)
        assert_allclose(m.values, [2, 4], atol=1e-2)


@pytest.mark.parametrize(
    "template",
    (
        (np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 3.0])),
        (
            np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]]),
        ),
    ),
)
def test_Template_does_not_modify_inputs(template):
    from copy import deepcopy

    xe = np.array([0, 1, 2, 3])
    template_copy = deepcopy(template)
    n = template[0] + template[1]

    Template(n, xe, template, method="da")
    assert_equal(template, template_copy)


def generate(rng, nmc, truth, bins, tf=1, df=1):
    stats = pytest.importorskip("scipy.stats")
    xe = np.linspace(0, 2, bins + 1)
    b = np.diff(stats.truncexpon(1, 0, 2).cdf(xe))
    s = np.diff(stats.norm(1, 0.1).cdf(xe))
    n = b * truth[0] + s * truth[1]
    t = b * nmc, s * nmc
    if rng is not None:
        n = rng.poisson(n / df) * df
        if df != 1:
            n = np.transpose((n, n * df))
        t = [rng.poisson(ti / tf) * tf for ti in t]
        if tf != 1:
            t = [np.transpose((tj, tj * tf)) for tj in t]
    return n, xe, np.array(t)


@pytest.mark.parametrize("method", ("jsc", "asy", "da"))
@pytest.mark.parametrize("with_mask", (False, True))
@pytest.mark.parametrize("weighted_data", (False, True))
def test_Template_weighted(method, with_mask, weighted_data):
    rng = np.random.default_rng(1)
    truth = 750, 250
    z = []
    rng = np.random.default_rng(1)
    for itoy in range(100):
        ni, xe, ti = generate(rng, 400, truth, 15, 1.5, 1.5 if weighted_data else 1)
        c = Template(ni, xe, ti, method=method)
        if with_mask:
            cx = 0.5 * (xe[1:] + xe[:-1])
            c.mask = cx != 1.5
        m = Minuit(c, *truth)
        m.limits = (0, None)
        m.strategy = 0
        for iter in range(10):
            m.migrad(iterate=1)
            m.hesse()
            if m.valid and m.accurate:
                break
        assert m.valid
        z.append((m.values[1] - truth[1]) / m.errors[1])
    assert_allclose(np.mean(z), 0, atol=0.3)
    assert_allclose(np.std(z), 1, rtol=0.1)


def test_Template_bad_input():
    with pytest.raises(ValueError):
        Template([1, 2], [1, 2, 3], [])

    with pytest.raises(ValueError, match="do not match"):
        Template([1, 2], [1, 2, 3], [[1, 2, 3], [1, 2, 3]])

    with pytest.raises(ValueError, match="do not match"):
        Template(
            [1, 2],
            [1, 2, 3],
            [[[1, 2], [3, 4]], [[1, 2], [3, 4], [5, 6]]],
        )

    with pytest.raises(ValueError, match="not understood"):
        Template([1], [1, 2], [[1]], method="foo")

    with pytest.raises(ValueError, match="number of names"):
        Template([1], [1, 2], [[1]], name=("b", "s"))

    with pytest.raises(ValueError, match="model_or_template"):
        Template([1], [1, 2], [[1], None])


def test_Template_visualize():
    pytest.importorskip("matplotlib")

    xe = [0, 1, 2]
    n = [1, 2]
    t = [[1, 2], [5, 4]]

    c = Template(n, xe, t)

    c.visualize((1, 2))

    c.mask = np.array([False, True])
    c.visualize((1, 2))


def test_Template_visualize_2D():
    pytest.importorskip("matplotlib")

    xe = ([0, 1, 2], [0, 1, 2])
    n = [[1, 2], [3, 4]]
    t = [[[1, 2], [1, 2]], [[5, 4], [5, 4]]]

    c = Template(n, xe, t)
    c.visualize((1, 2))


def test_Template_pickle():
    n = np.array([1, 2, 3])
    xe = np.array([0, 1, 2, 3])
    t = np.array([[1, 1, 0], [0, 1, 3]])

    c = Template(n, xe, t)
    b = pickle.dumps(c)
    c2 = pickle.loads(b)

    assert_equal(c.data, c2.data)


def test_Template_with_model():
    n = np.array([3, 2, 3])
    xe = np.array([0, 1, 2, 3])
    t = np.array([0.1, 0, 1])

    c = Template(n, xe, (t, scaled_cdf))
    assert describe(c) == ["x0", "x1_n", "x1_mu", "x1_sigma"]

    m = Minuit(c, 1, 1, 0.5, 1)
    m.limits["x0", "x1_n"] = (0, None)
    m.limits["x1_sigma"] = (0.1, None)
    m.limits["x1_mu"] = (xe[0], xe[-1])
    m.migrad()
    assert m.valid

    c2 = Template(n, xe, (t, scaled_cdf), name=("a", "b", "c", "d"))
    assert describe(c2) == ["a", "b", "c", "d"]
    m = Minuit(c2, 1, 1, 0.5, 1)
    m.limits["d"] = (0.1, None)
    m.migrad()
    assert m.valid


def test_Template_with_model_2D():
    truth1 = (1.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    x1, y1 = mvnorm(*truth1[1:]).rvs(size=int(truth1[0] * 1000), random_state=1).T
    truth2 = (1.0, 0.2, 0.1, 0.4, 0.3, 0.0)
    x2, y2 = mvnorm(*truth2[1:]).rvs(size=int(truth2[0] * 1000), random_state=1).T

    x = np.append(x1, x2)
    y = np.append(y1, y2)
    w, xe, ye = np.histogram2d(x, y, bins=(3, 5))

    def model(xy, n, mux, muy, sx, sy, rho):
        return n * 1000 * mvnorm(mux, muy, sx, sy, rho).cdf(np.transpose(xy))

    x3, y3 = mvnorm(*truth2[1:]).rvs(size=int(truth2[0] * 10000), random_state=2).T
    template = np.histogram2d(x3, y3, bins=(xe, ye))[0]

    cost = Template(w, (xe, ye), (model, template))
    assert cost.ndata == np.prod(w.shape)
    m = Minuit(cost, *truth1, 1)
    m.limits["x0_n", "x0_sx", "x0_sy"] = (0, None)
    m.limits["x0_rho"] = (-1, 1)
    m.migrad()
    assert m.valid
    assert_allclose(m.values, truth1 + (1e3,), rtol=0.1)


def test_Template_with_only_models():
    n = np.array([3, 2, 3])
    xe = np.array([0, 1, 2, 3])

    c = Template(n, xe, (lambda x, n, mu: n * expon_cdf(x, mu), scaled_cdf))
    assert describe(c) == ["x0_n", "x0_mu", "x1_n", "x1_mu", "x1_sigma"]

    m = Minuit(c, 1, 0.5, 1, 0.5, 1)
    m.limits["x0_n", "x1_n", "x1_sigma"] = (0, None)
    m.limits["x0_mu", "x1_mu"] = (xe[0], xe[-1])
    m.migrad()
    assert m.valid


def test_Template_pulls():
    rng = np.random.default_rng(1)
    xe = np.linspace(0, 1, 200)
    tmu = np.array(
        [
            np.diff(norm_cdf(xe, 0.5, 0.1)) * 1000,
            np.diff(expon_cdf(xe, 1)) * 10000,
        ]
    )
    truth = (1000, 2000)
    mu = sum(truth[i] * (tmu[i] / np.sum(tmu[i])) for i in range(2))
    n = rng.poisson(mu)
    t = rng.poisson(tmu)
    c = Template(n, xe, t)
    m = Minuit(c, *truth)
    m.migrad()
    assert m.valid
    pulls = c.pulls(m.values)
    assert np.nanmean(pulls) == pytest.approx(0, abs=0.1)
    assert np.nanvar(pulls) == pytest.approx(1, abs=0.1)


def test_deprecated():
    from iminuit import cost

    with pytest.warns(np.VisibleDeprecationWarning):
        from iminuit.cost import BarlowBeestonLite
    assert BarlowBeestonLite is cost.Template

    with pytest.warns(np.VisibleDeprecationWarning):
        from iminuit.cost import barlow_beeston_lite_chi2_jsc
    assert barlow_beeston_lite_chi2_jsc is cost.template_chi2_jsc

    with pytest.warns(np.VisibleDeprecationWarning):
        from iminuit.cost import barlow_beeston_lite_chi2_hpd
    assert barlow_beeston_lite_chi2_hpd is cost.template_chi2_da


def test_deprecated_Template_method():
    from iminuit import cost

    with pytest.warns(np.VisibleDeprecationWarning):
        t = Template([1], [2, 3], [[1], [2]], method="hpd")
        t._impl is cost.template_chi2_da


@pytest.mark.parametrize("cost", (BinnedNLL, ExtendedBinnedNLL))
def test_binned_cost_with_model_shape_error_message_1D(cost):
    n = [1, 2]
    edges = [0.1, 0.2, 0.3]

    # cdf should return same length as xe, not same length as n
    def cdf(xe, a):
        return xe[:-1]

    c = cost(n, edges, cdf)
    with pytest.raises(
        ValueError,
        match=(r"output of model has shape \(2,\), but \(3,\) is required"),
    ):
        c(1)


@pytest.mark.parametrize("cost", (BinnedNLL, ExtendedBinnedNLL))
def test_binned_cost_with_model_shape_error_message_2D(cost):
    n = [[1, 2, 3], [4, 5, 6]]
    edges = [0.5, 0.6, 0.7], [0.1, 0.2, 0.3, 0.4]

    def cdf(xye, a):
        # xye has shape (2, N), returned array should be (N,)
        return np.ones(xye.shape[1] - 1)

    c = cost(n, edges, cdf)
    with pytest.raises(
        ValueError,
        match=(r"output of model has shape \(11,\), but \(12,\) is required"),
    ):
        c(1)


def test_BohmZechTransform():
    from iminuit.cost import BohmZechTransform

    with pytest.warns(np.VisibleDeprecationWarning):
        val = np.array([1.0, 2.0])
        var = np.array([3.0, 4.0])
        tr = BohmZechTransform(val, var)
        s = val / var
        mu = np.array([2.0, 2.0])
        ns, mus = tr(mu)
        assert_allclose(ns, val * s)
        assert_allclose(mus, mu * s)
        var2 = mu**2
        ns, mus, vars = tr(mu, var2)
        assert_allclose(ns, val * s)
        assert_allclose(mus, mu * s)
        assert_allclose(vars, var2 * s**2)
