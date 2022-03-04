from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
import numpy as np
import numba as nb
from numba_stats import norm
import pytest
from numpy.testing import assert_allclose

N = [int(x) for x in np.geomspace(10, 1e6, 11)]


@pytest.mark.parametrize("n", N)
def test_UnbinnedNLL(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    cost = UnbinnedNLL(x, norm.pdf)
    benchmark(cost, 0, 1)


@pytest.mark.parametrize("n", N)
def test_sum_log(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    def cost(x, mu, sigma):
        return -np.sum(np.log(norm.pdf(x, mu, sigma)))

    benchmark(cost, x, 0, 1)


@pytest.mark.parametrize("n", N)
def test_numba_sum_log_pdf(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit
    def cost(x, mu, sigma):
        return -np.sum(np.log(norm.pdf(x, mu, sigma)))

    cost(x, 0, 1)

    benchmark(cost, x, 0, 1)


@pytest.mark.parametrize("n", N)
def test_numba_sum_logpdf(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit
    def cost(x, mu, sigma):
        return -np.sum(norm.logpdf(x, mu, sigma))

    cost(x, 0, 1)

    benchmark(cost, x, 0, 1)


@pytest.mark.parametrize("n", N)
def test_numba_sum_log_parallel(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit(parallel=True)
    def cost(x, mu, sigma):
        return -np.sum(norm.logpdf(x, mu, sigma))

    cost(x, 0, 1)

    benchmark(cost, x, 0, 1)


@pytest.mark.parametrize("n", N[:-1])
def test_minuit_sum_log(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit
    def cost(x, mu, sigma):
        return -np.sum(norm.logpdf(x, mu, sigma))

    cost(x, 0, 1)

    m = Minuit(lambda loc, scale: cost(x, loc, scale), 0, 1)
    m.errordef = Minuit.LIKELIHOOD
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N[:-1])
def test_minuit_UnbinnedNLL(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    cost = UnbinnedNLL(x, norm.pdf)
    m = Minuit(cost, 0, 1)
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N[:-1])
def test_minuit_UnbinnedNLL_log(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    cost = UnbinnedNLL(x, norm.logpdf, log=True)
    m = Minuit(cost, 0, 1)
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)
