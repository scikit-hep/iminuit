"""Example for testing the QtWidget interactively."""

from iminuit import cost
from iminuit import Minuit
from numba_stats import t
import numpy as np


def model(xe, s, mu, sigma, nuinv):
    """Return model cdf."""
    nu = 1 / nuinv
    return s * t.cdf(xe, nu, mu, sigma)


truth = 100.0, 0.5, 0.1, 0.5

rng = np.random.default_rng(1)
xe = np.linspace(0, 1, 20)
m = np.diff(model(xe, *truth))
n = rng.poisson(m)

c = cost.ExtendedBinnedNLL(n, xe, model) + cost.NormalConstraint(
    ["mu", "sigma"], [0.5, 0.1], [0.1, 0.1]
)

m = Minuit(c, *truth)
m.limits["sigma", "s", "nuinv"] = (0, None)

m.interactive()
