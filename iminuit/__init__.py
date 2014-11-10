"""IMinuit is a nice Python interface to MINUIT.
"""

__all__ = [
    'Minuit',
    'describe',
    'Struct',
    'InitialParamWarning',
    '__version__'
]

from iminuit._libiminuit import *
from iminuit.util import describe, Struct
from iminuit.iminuit_warnings import *
from iminuit.info import __version__

# Define a test function that runs the `iminuit` tests.
# This seems to work OK
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.Tester.test.html
try:
    from numpy.testing import Tester
    test = Tester().test
except:
    pass
# An alternative would be to use `nose` directly to avoid the `numpy` depencency
# `scikit-image` is a nice example we could maybe copy & paste 
# https://github.com/scikit-image/scikit-image/blob/master/skimage/__init__.py
