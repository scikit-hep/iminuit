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
    """
    Wrapper for scipy.stats.expon.

    The location parameter of scipy.stats.expon makes no sense in practice,
    so this wrapper hides it.
    """

    def __init__(self, mu: float):
        """
        Initialize distribution.

        Parameters
        ----------
        mu : float
            Expectation value.
        """
        super().__init__(0, mu)


class truncnorm(stats.truncnorm):
    """
    Wrapper for scipy.stats.truncnorm.

    The wrapper allows you to provide the truncation limits in normal user coordinates.
    """

    def __init__(self, xmin: float, xmax: float, mu: float, sigma: float):
        """
        Initialize distribution.

        Parameters
        ----------
        xmin : float
            Lower limit of truncated range.
        xmax : float
            Upper limit of truncated range.
        mu : float
            Expectation value.
        sigma : float
            Width of the normal distribution.
        """
        zmin = (xmin - mu) / sigma
        zmax = (xmax - mu) / sigma
        super().__init__(zmin, zmax, mu, sigma)


class truncexpon(stats.truncexpon):
    """
    Wrapper for scipy.stats.truncexpon.

    The location parameter of scipy.stats.expon makes no sense in practice, so this
    wrapper hides it. It further allows you to provide lower and upper truncation limits
    in normal user coordinates.
    """

    def __init__(self, xmin: float, xmax: float, slope: float):
        """
        Initialize distribution.

        Parameters
        ----------
        xmin : float
            Lower limit of truncated range.
        xmax : float
            Upper limit of truncated range.
        slope : float
            Slope parameter.
        """
        zmin = xmin / slope
        zmax = xmax / slope
        super().__init__(zmax, zmin, slope)
