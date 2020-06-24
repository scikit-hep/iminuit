import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL, BinnedNLL, ExtendedUnbinnedNLL, ExtendedBinnedNLL
from scipy.stats import norm

np.random.seed(1)
x = np.random.randn(1000)
truth = (len(x), np.mean(x), np.std(x, ddof=1))


def test_UnbinnedNLL():
    def pdf(x, mu, sigma):
        return norm(mu, sigma).pdf(x)

    m = Minuit(UnbinnedNLL(x, pdf, verbose=2), mu=0, sigma=1, limit_sigma=(0, None))
    m.migrad()
    assert_allclose(m.args, truth[1:], atol=1e-3)


def test_BinnedNLL():
    def cdf(x, mu, sigma):
        return norm(mu, sigma).cdf(x)

    n, xe = np.histogram(x, bins=300)
    m = Minuit(BinnedNLL(n, xe, cdf), mu=0, sigma=1, limit_sigma=(0, None))
    m.migrad()
    assert_allclose(m.args, truth[1:], atol=1e-3)


def test_ExtendedUnbinnedNLL():
    def scaled_pdf(x, n, mu, sigma):
        return norm(mu, sigma).pdf(x) * n

    scaled_pdf.integral = lambda n, mu, sigma: n

    m = Minuit(
        ExtendedUnbinnedNLL(x, scaled_pdf),
        n=len(x),
        mu=0,
        sigma=1,
        limit_n=(0, None),
        limit_sigma=(0, None),
    )
    m.migrad()
    assert_allclose(m.args, truth, atol=1e-3)


def test_ExtendedBinnedNLL():
    def scaled_cdf(x, n, mu, sigma):
        return norm(mu, sigma).cdf(x) * n

    n, xe = np.histogram(x, bins=1000)
    m = Minuit(
        ExtendedBinnedNLL(n, xe, scaled_cdf),
        n=len(x),
        mu=0,
        sigma=1,
        limit_n=(0, None),
        limit_sigma=(0, None),
    )
    m.migrad()
    assert_allclose(m.args, truth, atol=1e-3)
