from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
import numpy as np
import numba as nb
from numba_stats import norm, truncexpon
import pytest
from numpy.testing import assert_allclose

N = [int(x) for x in np.geomspace(10, 1e6, 11)]
N = N[:-1]


def make_data(size, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.normal(0.5, 0.1, size=size // 2)
    b = rng.exponential(1, size=2 * size)
    b = b[b < 1]
    b = b[: size // 2]
    x = np.append(s, b)
    return x


# we inline following functions to profit from parallel=True,fastmath=True


@nb.njit(inline="always")
def mixture(x, z, mu, sigma, slope):
    b = truncexpon.pdf(x, 0.0, 1.0, 0.0, slope)
    s = norm.pdf(x, mu, sigma)
    return (1 - z) * b + z * s


@nb.njit(inline="always")
def logsumexp(a, b):
    r = np.empty_like(a)
    for i in nb.prange(len(r)):
        if a[i] > b[i]:
            r[i] = a[i] + np.log(1 + np.exp(b[i] - a[i]))
        else:
            r[i] = b[i] + np.log(1 + np.exp(a[i] - b[i]))
    return r


ARGS = (0.5, 0.5, 0.1, 1.0)


@nb.njit(inline="always")
def log_mixture(x, z, mu, sigma, slope):
    log_b = truncexpon.logpdf(x, 0.0, 1.0, 0.0, slope)
    log_s = norm.logpdf(x, mu, sigma)
    b = np.log(1 - z) + log_b
    s = np.log(z) + log_s
    return logsumexp(b, s)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
def test_UnbinnedNLL(benchmark, n, log):
    x = make_data(size=n)
    fn = log_mixture if log else mixture
    fn(x, *ARGS)  # warm-up JIT
    cost = UnbinnedNLL(x, fn, log=log)
    benchmark(cost, *ARGS)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("lib", ["numba_stats", "scipy.stats"])
def test_nll(benchmark, n, lib):
    x = make_data(size=n)

    if lib == "scipy.stats":
        from scipy.stats import truncexpon, norm

        def fn(x, z, mu, sigma, slope):
            b = truncexpon.pdf(x, 1.0, 0.0, slope)
            s = norm.pdf(x, mu, sigma)
            return (1 - z) * b + z * s

    else:
        fn = mixture

    def cost(x, z, mu, sigma, slope):
        return -np.sum(np.log(fn(x, z, mu, sigma, slope)))

    benchmark(cost, x, *ARGS)


@pytest.mark.parametrize("n", N)
def test_nll_numba(benchmark, n):
    x = make_data(size=n)

    @nb.njit
    def cost(x, z, mu, sigma, slope):
        return -np.sum(np.log(mixture(x, z, mu, sigma, slope)))

    benchmark(cost, x, *ARGS)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("numba", [False, True])
@pytest.mark.parametrize("log", [False, True])
def test_minuit(benchmark, n, log, numba):
    x = make_data(size=n)

    if log:

        def cost(x, z, mu, sigma, slope):
            return -np.sum(log_mixture(x, z, mu, sigma, slope))

    else:

        def cost(x, z, mu, sigma, slope):
            return -np.sum(np.log(mixture(x, z, mu, sigma, slope)))

    if numba:
        cost = nb.njit(cost)
        cost(x, *ARGS)  # jit warm-up

    m = Minuit(lambda *args: cost(x, *args), *ARGS)
    m.errordef = Minuit.LIKELIHOOD
    m.limits[0, 1] = (0, 1)
    m.limits[2, 3] = (0, 10)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    # ignore slope
    assert_allclose(m.values[:-1], ARGS[:-1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
def test_minuit_parallel_fastmath(benchmark, n, log):
    x = make_data(size=n)

    if log:

        @nb.njit(parallel=True, fastmath=True)
        def cost(x, z, mu, sigma, slope):
            return -np.sum(log_mixture(x, z, mu, sigma, slope))

    else:

        @nb.njit(parallel=True, fastmath=True)
        def cost(x, z, mu, sigma, slope):
            p = mixture(x, z, mu, sigma, slope)
            return -np.sum(np.log(p))

    cost(x, *ARGS)  # jit warm-up

    m = Minuit(lambda *args: cost(x, *args), *ARGS)
    m.errordef = Minuit.LIKELIHOOD
    m.limits[0, 1] = (0, 1)
    m.limits[2, 3] = (0, 10)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    # ignore slope
    assert_allclose(m.values[:-1], ARGS[:-1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
def test_minuit_cfunc(benchmark, n):
    x = make_data(size=n)

    @nb.cfunc(nb.double(nb.uintc, nb.types.CPointer(nb.double)))
    def cost(n, par):
        z, mu, sigma, slope = nb.carray(par, (n,))
        return -np.sum(np.log(mixture(x, z, mu, sigma, slope)))

    m = Minuit(cost, *ARGS)
    m.errordef = Minuit.LIKELIHOOD
    m.limits[0, 1] = (0, 1)
    m.limits[2, 3] = (0, 10)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    # ignore slope
    assert_allclose(m.values[:-1], ARGS[:-1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
def test_minuit_UnbinnedNLL(benchmark, n, log):
    x = make_data(size=n)

    cost = UnbinnedNLL(x, log_mixture if log else mixture, log=log)
    m = Minuit(cost, *ARGS)
    m.limits[0, 1] = (0, 1)
    m.limits[2, 3] = (0, 10)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    # ignore slope
    assert_allclose(m.values[:-1], ARGS[:-1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("BatchMode", [False, True])
@pytest.mark.parametrize("NumCPU", [0, nb.get_num_threads()])
def test_RooFit(benchmark, n, BatchMode, NumCPU):
    import ROOT as R

    x = R.RooRealVar("x", "x", 0, 1)
    z = R.RooRealVar("z", "z", 0.5, 0, 1)
    mu = R.RooRealVar("mu", "mu", 0.5, 0, 1)
    sigma = R.RooRealVar("sigma", "sigma", 0.1, 0, 10)
    slope = R.RooRealVar("slope", "slope", 1.0, 0, 10)
    pdf1 = R.RooGaussian("gauss", "gauss", x, mu, sigma)
    pdf2 = R.RooExponential("expon", "expon", x, slope)
    pdf = R.RooAddPdf("pdf", "pdf", [pdf1, pdf2], [z])

    data = pdf.generate(x, n)

    def run():
        mu.setVal(0.5)
        sigma.setVal(0.1)
        slope.setVal(1)
        z.setVal(0.5)
        args = [R.RooFit.PrintLevel(-1), R.RooFit.BatchMode(BatchMode)]
        if NumCPU:
            args.append(R.RooFit.NumCPU(NumCPU))
        pdf.fitTo(data, *args)

    benchmark(run)
    assert_allclose(z.getVal(), 0.5, atol=5 / n**0.5)
    assert_allclose(mu.getVal(), 0.5, atol=5 / n**0.5)
    assert_allclose(sigma.getVal(), 0.1, atol=5 / n**0.5)
