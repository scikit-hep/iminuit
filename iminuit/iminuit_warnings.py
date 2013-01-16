import warnings


class IMinuitWarning(RuntimeWarning):
    pass


class InitialParamWarning(IMinuitWarning):
    pass


class HesseFailedWarning(IMinuitWarning):
    pass


warnings.simplefilter('always', InitialParamWarning, append=True)
