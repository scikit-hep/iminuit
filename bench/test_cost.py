from iminuit.cost import UnbinnedNLL
from iminuit import Minuit
import numpy as np
import numba as nb
from numba_stats import norm, truncexpon
import pytest
from numpy.testing import assert_allclose

N = [int(x) for x in np.geomspace(10, 1e6, 11)]


def make_data(size, seed=1):
    rng = np.random.default_rng(seed)
    s = rng.normal(0.5, 0.1, size=size // 2)
    b = rng.exponential(1, size=2 * size)
    b = b[b < 1]
    b = b[: size // 2]
    x = np.append(s, b)
    return x


@nb.njit
def mixture(x, z, mu, sigma, slope):
    s = norm.pdf(x, mu, sigma)
    b = truncexpon.pdf(x, 0.0, 1.0, 0.0, slope)
    return (1 - z) * b + z * s


@nb.njit
def logsumexp(a, b):
    r = np.empty_like(a)
    for i in nb.prange(len(r)):
        if a[i] > b[i]:
            r[i] = a[i] + np.log(1 + np.exp(b[i] - a[i]))
        else:
            r[i] = b[i] + np.log(1 + np.exp(a[i] - b[i]))
    return r


@nb.njit
def log_mixture(x, z, mu, sigma, slope):
    log_b = truncexpon.logpdf(x, 0.0, 1.0, 0.0, slope)
    log_s = norm.logpdf(x, mu, sigma)
    b = np.log(1 - z) + log_b
    s = np.log(z) + log_s
    return logsumexp(b, s)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("model", ["norm", "norm+truncexpon"])
def test_UnbinnedNLL(benchmark, n, log, model):
    if model == "norm":
        x = np.random.default_rng(1).normal(size=n)
        fn = log_mixture if log else mixture
        args = (0.5, 0.5, 0.1, 1.0)
    else:
        x = make_data(size=n)
        fn = norm.logpdf if log else norm.pdf
        args = (0.0, 1.0)
    cost = UnbinnedNLL(x, fn, log=log)
    benchmark(cost, *args)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("lib", ["numba_stats", "scipy.stats"])
@pytest.mark.parametrize("log", [False, True])
def test_custom(benchmark, n, log, lib):
    x = np.random.default_rng(1).normal(size=n)

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


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("numba", [False, True])
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("model", ["norm", "norm+truncexpon"])
def test_minuit(benchmark, n, log, numba, model):
    if model == "norm":
        if log:

            def cost(x, mu, sigma):
                return -np.sum(norm.logpdf(x, mu, sigma))

        else:

            def cost(x, mu, sigma):
                return -np.sum(np.log(norm.pdf(x, mu, sigma)))

        args = (0.0, 1.0)
        x = np.random.default_rng(1).normal(size=n)
    else:
        if log:

            def cost(x, z, mu, sigma, slope):
                return -np.sum(log_mixture(x, z, mu, sigma, slope))

        else:

            def cost(x, z, mu, sigma, slope):
                return -np.sum(np.log(mixture(x, z, mu, sigma, slope)))

        args = (0.5, 0.5, 0.1, 1.0)
        x = make_data(size=n)

    if numba:
        cost = nb.njit(cost)
        cost(x, *args)  # jit warm-up

    m = Minuit(lambda *args: cost(x, *args), *args)
    m.errordef = Minuit.LIKELIHOOD
    if model == "norm":
        m.limits[1] = (0, None)
    else:
        m.limits[0, 1] = (0, 1)
        m.limits[2, 3] = (0, 10)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    if model == "norm":
        assert_allclose(m.values, args, atol=2 / n**0.5)
    else:
        # ignore slope
        assert_allclose(m.values[:-1], args[:-1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("model", ["norm", "norm+truncexpon"])
def test_minuit_parallel_fastmath(benchmark, n, log, model):
    if model == "norm":
        if log:

            @nb.njit(parallel=True, fastmath=True)
            def cost(x, mu, sigma):
                return -np.sum(norm.logpdf(x, mu, sigma))

        else:

            @nb.njit(parallel=True, fastmath=True)
            def cost(x, mu, sigma):
                return -np.sum(np.log(norm.pdf(x, mu, sigma)))

        x = np.random.default_rng(1).normal(size=n)
        args = (0.0, 1.0)
    else:
        if log:

            @nb.njit(parallel=True, fastmath=True)
            def cost(x, z, mu, sigma, slope):
                return -np.sum(log_mixture(x, z, mu, sigma, slope))

        else:

            @nb.njit(parallel=True, fastmath=True)
            def cost(x, z, mu, sigma, slope):
                return -np.sum(np.log(mixture(x, z, mu, sigma, slope)))

        x = make_data(size=n)
        args = (0.5, 0.5, 0.1, 1.0)

    cost(x, *args)  # jit warm-up

    m = Minuit(lambda *args: cost(x, *args), *args)
    m.errordef = Minuit.LIKELIHOOD
    if model == "norm":
        m.limits[1] = (0, None)
    else:
        m.limits[0, 1] = (0, 1)
        m.limits[2, 3] = (0, 10)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    if model == "norm":
        assert_allclose(m.values, args, atol=2 / n**0.5)
    else:
        # ignore slope
        assert_allclose(m.values[:-1], args[:-1], atol=2 / n**0.5)


@pytest.mark.parametrize("handtuned", [False, True])
@pytest.mark.parametrize("n", N)
def test_minuit_log_parallel_fastmath(benchmark, n, handtuned):
    x = np.random.default_rng(1).normal(size=n)

    if handtuned:

        @nb.njit(parallel=True, fastmath=True)
        def cost(x, mu, sigma):
            inv_sigma = 1 / sigma
            z = (x - mu) * inv_sigma
            y = 0.5 * z * z - np.log(inv_sigma)
            return np.sum(y)

    else:

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


@pytest.mark.parametrize("n", N)
def test_cfunc(benchmark, n):
    x = np.random.default_rng(1).normal(size=n)

    @nb.cfunc(nb.double(nb.uintc, nb.types.CPointer(nb.double)))
    def cost(n, par):
        a, b = nb.carray(par, (n,))
        return -np.sum(np.log(norm.pdf(x, a, b)))

    cost.errordef = Minuit.LIKELIHOOD

    m = Minuit(cost, 0, 1, name=["loc", "scale"])
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("log", [False, True])
def test_minuit_UnbinnedNLL(benchmark, n, log):
    x = np.random.default_rng(1).normal(size=n)

    cost = UnbinnedNLL(x, norm.logpdf if log else norm.pdf, log=log)
    m = Minuit(cost, 0, 1)
    m.limits["scale"] = (0, None)

    def run():
        m.reset()
        return m.migrad()

    m = benchmark(run)
    assert m.valid
    assert_allclose(m.values, [0, 1], atol=2 / n**0.5)


@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("BatchMode", [False, True])
@pytest.mark.parametrize("NumCPU", [0, nb.get_num_threads()])
@pytest.mark.parametrize("model", ["norm", "norm+truncexpon"])
def test_RooFit(benchmark, n, BatchMode, NumCPU, model):
    import ROOT as R

    if model == "norm":
        x = R.RooRealVar("x", "x", -10, 10)
        mu = R.RooRealVar("mu", "mu", 0, -10, 10)
        sigma = R.RooRealVar("sigma", "sigma", 1, 0, 10)
        pdf = R.RooGaussian("gauss", "pdf", x, mu, sigma)
    else:
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
        if model == "norm":
            mu.setVal(0)
            sigma.setVal(1)
        else:
            mu.setVal(0.5)
            sigma.setVal(0.1)
            slope.setVal(1)
            z.setVal(0.5)
        args = [R.RooFit.PrintLevel(-1), R.RooFit.BatchMode(BatchMode)]
        if NumCPU:
            args.append(R.RooFit.NumCPU(NumCPU))
        pdf.fitTo(data, *args)

    benchmark(run)
    if model == "norm":
        assert_allclose(mu.getVal(), 0, atol=5 / n**0.5)
        assert_allclose(sigma.getVal(), 1, atol=5 / n**0.5)
    else:
        assert_allclose(mu.getVal(), 0.5, atol=5 / n**0.5)
        assert_allclose(sigma.getVal(), 0.1, atol=5 / n**0.5)
        assert_allclose(z.getVal(), 0.5, atol=5 / n**0.5)
