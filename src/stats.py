"""
Wrappers which fix unpractical conventions in scipy.stats.

The truncated distributions in scipy use an unpractical calling convention
for historical reasons. They expect the truncation limits to be for the
function in standardized form (location = 0 and scale = 1) but users
generally want to provide them in their normal coordinates.

The derived classes here which fix these quirks.
"""
from scipy import stats


class expon(stats.expon):
    def __init__(self, mu):
        super().__init__(0, mu)


class truncnorm(stats.truncnorm):
    def __init__(self, xmin, xmax, mu, sigma):
        zmin = (xmin - mu) / sigma
        zmax = (xmax - mu) / sigma
        super().__init__(zmin, zmax, mu, sigma)


class truncexpon(stats.truncexpon):
    def __init__(self, xmin, xmax, mu):
        zmin = xmin / mu
        zmax = xmax / mu
        super().__init__(zmin, zmax, 0, mu)
