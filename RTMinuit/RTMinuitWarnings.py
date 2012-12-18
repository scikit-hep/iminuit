import warnings

class RTMinuitWarning(RuntimeWarning):
    pass

class RTMinuitInitialParamWarning(RuntimeWarning):
    pass

warnings.simplefilter('always', RTMinuitInitialParamWarning, append=True);
