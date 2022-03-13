from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
import numpy as np
import numba as nb
from numba_stats import norm
import pytest
from numpy.testing import assert_allclose

N = [int(x) for x in np.geomspace(10, 1e6, 11)]


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
def test_UnbinnedNLL(benchmark, n, log):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    cost = UnbinnedNLL(x, norm.logpdf if log else norm.pdf, log=log)
    benchmark(cost, 0.0, 1.0)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
def test_simple(benchmark, n, log):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    if log:

        def cost(x, mu, sigma):
            return -np.sum(norm.logpdf(x, mu, sigma))

    else:

        def cost(x, mu, sigma):
            return -np.sum(np.log(norm.pdf(x, mu, sigma)))

    benchmark(cost, x, 0.0, 1.0)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("fastmath", [False, True])
def test_numba_sum_logpdf(benchmark, n, parallel, fastmath):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit(parallel=parallel, fastmath=fastmath)
    def cost(x, mu, sigma):
        return -np.sum(norm.logpdf(x, mu, sigma))

    cost(x, 0.0, 1.0)  # jit warm-up

    benchmark(cost, x, 0.0, 1.0)


@pytest.mark.parametrize("n", N[:-1])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("numba", [False, True])
def test_minuit_simple(benchmark, n, log, numba):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    if log:

        def cost(x, mu, sigma):
            return -np.sum(norm.logpdf(x, mu, sigma))

    else:

        def cost(x, mu, sigma):
            return -np.sum(np.log(norm.pdf(x, mu, sigma)))

    if numba:
        cost = nb.njit(cost)
        cost(x, 0.0, 1.0)  # jit warm-up

        # print(f"{numba=} {log=}")
        # for v in cost.inspect_asm().values():
        #     print(v)

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
def test_minuit_numba_sum_logpdf_parallel_fastmath(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit(parallel=True, fastmath=True)
    def cost(x, mu, sigma):
        return -np.sum(norm.logpdf(x, mu, sigma))

    cost(x, 0.0, 1.0)  # jit warm-up

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
def test_minuit_numba_handtuned_parallel_fastmath(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.njit(parallel=True, fastmath=True)
    def cost(x, mu, sigma):
        inv_sigma = 1 / sigma
        z = (x - mu) * inv_sigma
        y = 0.5 * z * z - np.log(inv_sigma)
        return np.sum(y)

    cost(x, 0.0, 1.0)  # jit warm-up

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
def test_minuit_cfunc_sum_logpdf(benchmark, n):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    @nb.cfunc(nb.double(nb.uintc, nb.types.CPointer(nb.double)))
    def cost(n, par):
        par = nb.carray(par, (n,))
        return -np.sum(norm.logpdf(x, par[0], par[1]))

    cost.errordef = Minuit.LIKELIHOOD

    m = Minuit(cost, 0, 1, name=["loc", "scale"])
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N[:-1])
@pytest.mark.parametrize("log", [False, True])
def test_minuit_UnbinnedNLL(benchmark, n, log):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    cost = UnbinnedNLL(x, norm.logpdf if log else norm.pdf, log=log)
    m = Minuit(cost, 0, 1)
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)
