import warnings


class IMinuitWarning(RuntimeWarning):
    """iminuit warning.
    """
    pass


class InitialParamWarning(IMinuitWarning):
    """Initial parameter warning.
    """
    pass


class HesseFailedWarning(IMinuitWarning):
    """HESSE failed warning.
    """
    pass


warnings.simplefilter('always', InitialParamWarning, append=True)
