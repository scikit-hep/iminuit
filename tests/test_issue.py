import numpy as np
import warnings
import pytest


def test_issue_424():
    from iminuit import Minuit

    def fcn(x, y, z):
        return (x - 1) ** 2 + (y - 4) ** 2 / 2 + (z - 9) ** 2 / 3

    m = Minuit(fcn, x=0.0, y=0.0, z=0.0)
    m.migrad()

    m.fixed["x"] = True
    m.errors["x"] = 2
    m.hesse()  # this used to release x
    assert m.fixed["x"]
    assert m.errors["x"] == 2


def test_issue_544():
    import pytest
    from iminuit import Minuit
    from iminuit.util import IMinuitWarning

    def fcn(x, y):
        return x**2 + y**2

    m = Minuit(fcn, x=0, y=0)
    m.fixed = True
    with pytest.warns(IMinuitWarning):
        m.hesse()  # this used to cause a segfault


def test_issue_648():
    from iminuit import Minuit

    class F:
        first = True

        def __call__(self, a, b):
            if self.first:
                assert a == 1.0 and b == 2.0
                self.first = False
            return a**2 + b**2

    m = Minuit(F(), a=1, b=2)
    m.fixed["a"] = False  # this used to change a to b
    m.migrad()


def test_issue_643():
    from iminuit import Minuit

    def fcn(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(fcn, x=2, y=3, z=4)
    m.migrad()

    m2 = Minuit(fcn, x=m.values["x"], y=m.values["y"], z=m.values["z"])
    # used to call MnHesse when it was not needed and quickly exhaust call limit
    for i in range(10):
        m2.minos()

    m2.reset()
    # used to exhaust call limit, because calls to MnHesse did not reset call count
    for i in range(10):
        m2.values = m.values
        m2.minos()


def test_issue_669():
    from iminuit import Minuit

    def fcn(x, y):
        return x**2 + (y / 2) ** 2

    m = Minuit(fcn, x=0, y=0)

    m.migrad()

    xy1 = m.mncontour(x="x", y="y", size=10)
    xy2 = m.mncontour(x="y", y="x", size=10)  # used to fail

    # needs better way to compare polygons
    for x, y in xy1:
        match = False
        for y2, x2 in xy2:
            if abs(x - x2) < 1e-3 and abs(y - y2) < 1e-3:
                match = True
                break
        assert match


# cannot define this inside function, pickle will not allow it
def fcn(par):
    return np.sum(par**2)


# cannot define this inside function, pickle will not allow it
def grad(par):
    return 2 * par


def test_issue_687():
    import pickle
    import numpy as np
    from iminuit import Minuit

    start = np.zeros(3)
    m = Minuit(fcn, start)

    m.migrad()
    s_m = str(m)

    s = pickle.dumps(m)
    m2 = pickle.loads(s)

    s_m2 = str(m2)  # this used to fail
    assert s_m == s_m2


def test_issue_694():
    import pytest
    import numpy as np
    from iminuit import Minuit
    from iminuit.cost import ExtendedUnbinnedNLL

    stats = pytest.importorskip("scipy.stats")

    xmus = 1.0
    xmub = 5.0
    xsigma = 1.0
    ymu = 0.5
    ysigma = 0.2
    ytau = 0.1

    for seed in range(100):
        rng = np.random.default_rng(seed)

        xs = rng.normal(xmus, xsigma, size=33)
        xb = rng.normal(xmub, xsigma, size=66)
        x = np.append(xs, xb)

        def model(x, sig_n, sig_mu, sig_sigma, bkg_n, bkg_tau):
            return sig_n + bkg_n, (
                sig_n * stats.norm.pdf(x, sig_mu, sig_sigma)
                + bkg_n * stats.expon.pdf(x, 0, bkg_tau)
            )

        nll = ExtendedUnbinnedNLL(x, model)

        m = Minuit(nll, sig_n=33, sig_mu=ymu, sig_sigma=ysigma, bkg_n=66, bkg_tau=ytau)
        # with Simplex the fit never yields NaN, which is good but not what we want here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m.migrad(use_simplex=False)

        if np.isnan(m.fmin.edm):
            assert not m.valid
            assert m.fmin.is_above_max_edm
            break
    else:
        assert False


def test_issue_923():
    from iminuit import Minuit
    from iminuit.cost import LeastSquares
    import numpy as np
    import pytest

    # implicitly needed by visualize
    pytest.importorskip("matplotlib")

    def model(x, c1):
        c2 = 100
        res = np.zeros(len(x))
        mask = x < 47
        res[mask] = c1
        res[~mask] = c2
        return res

    xtest = np.linspace(0, 74)
    ytest = xtest * 0 + 1
    ytesterr = ytest

    least_squares = LeastSquares(xtest, ytest, ytesterr, model)

    m = Minuit(least_squares, c1=1)
    m.migrad()
    # this used to trigger an endless (?) loop
    m.visualize()


def test_jax_hessian(debug):
    pytest.importorskip("jax")
    import jax
    from jax import numpy as jnp
    from jax.scipy.special import erf
    from iminuit import Minuit

    jax.config.update(
        "jax_enable_x64", True
    )  # enable float64 precision, default is float32

    # generate some toy data
    rng = np.random.default_rng(seed=1)
    n, xe = np.histogram(rng.normal(size=10000), bins=1000)

    def cdf(x, mu, sigma):
        # cdf of a normal distribution, needed to compute the expected counts per bin
        # better alternative for real code: from jax.scipy.stats.norm import cdf
        z = (x - mu) / sigma
        return 0.5 * (1 + erf(z / np.sqrt(2)))

    def nll(par):  # negative log-likelihood with constants stripped
        amp = par[0]
        mu, sigma = par[1:]
        p = cdf(xe, mu, sigma)
        mu = amp * jnp.diff(p)
        result = jnp.sum(mu - n + n * jnp.log(n / (mu + 1e-100) + 1e-100))
        return result

    start_values = (1.5 * np.sum(n), 1.0, 2.0)
    limits = ((0, None), (None, None), (0, None))

    m = Minuit(
        fcn,
        start_values,
        grad=jax.grad(nll),
        hessian=jax.hessian(nll),
        name=("amp", "mu", "sigma"),
    )
    m.errordef = Minuit.LIKELIHOOD
    m.limits = limits
    m.strategy = 0  # do not explicitly compute hessian after minimisation
    m.migrad()
    assert m.ngrad > 0
    assert m.nhessian > 0
