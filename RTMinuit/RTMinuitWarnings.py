import warnings

class RTMinuitWarning(RuntimeWarning):
    pass

class RTMinuitInitialParamWarning(RuntimeWarning):
    pass

class RTMinuitHesseFailedWarning(RuntimeWarning):
    pass
warnings.simplefilter('always', RTMinuitInitialParamWarning, append=True);
