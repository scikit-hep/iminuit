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
@pytest.mark.parametrize("lib", ["numba_stats", "scipy"])
@pytest.mark.parametrize("log", [False, True])
def test_custom(benchmark, n, log, lib):
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)

    if lib == "numba_stats":
        from numba_stats import norm
    else:
        from scipy.stats import norm

    if log:

        def cost(x, mu, sigma):
            return -np.sum(norm.logpdf(x, mu, sigma))

    else:

        def cost(x, mu, sigma):
            return -np.sum(np.log(norm.pdf(x, mu, sigma)))

    benchmark(cost, x, 0.0, 1.0)


@pytest.mark.parametrize("n", N[:-1])
@pytest.mark.parametrize("numba", [False, True])
@pytest.mark.parametrize("log", [False, True])
def test_minuit_custom(benchmark, n, log, numba):
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
def test_minuit_custom_log_numba_parallel_fastmath(benchmark, n):
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
def test_minuit_handtuned_log_numba_parallel_fastmath(benchmark, n):
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
def test_minuit_custom_log_cfunc(benchmark, n):
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


@pytest.mark.parametrize("n", N[:-1])
@pytest.mark.parametrize("BatchMode", [False, True])
@pytest.mark.parametrize("NumCPU", [0, nb.get_num_threads()])
def test_RooFit(benchmark, n, BatchMode, NumCPU):
    import ROOT as R

    x = R.RooRealVar("x", "x", -10, 10)
    mu = R.RooRealVar("mu", "mu", 0, -10, 10)
    sigma = R.RooRealVar("sigma", "sigma", 1, 0, 10)
    gauss = R.RooGaussian("gauss", "pdf", x, mu, sigma)

    data = gauss.generate(x, n)

    def run():
        mu.setVal(0)
        sigma.setVal(1)
        args = [R.RooFit.PrintLevel(-1), R.RooFit.BatchMode(BatchMode)]
        if NumCPU:
            args.append(R.RooFit.NumCPU(NumCPU))
        gauss.fitTo(data, *args)

    benchmark(run)
    assert_allclose(mu.getVal(), 0, atol=5 / n**0.5)
    assert_allclose(sigma.getVal(), 1, atol=5 / n**0.5)
