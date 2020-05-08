from __future__ import absolute_import, division, print_function, unicode_literals
import warnings

__all__ = ["IMinuitWarning", "InitialParamWarning", "HesseFailedWarning"]


class IMinuitWarning(RuntimeWarning):
    """iminuit warning.
    """


class InitialParamWarning(IMinuitWarning):
    """Initial parameter warning.
    """


class HesseFailedWarning(IMinuitWarning):
    """HESSE failed warning.
    """


warnings.simplefilter("always", InitialParamWarning, append=True)
