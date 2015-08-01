"""IMinuit is a nice Python interface to MINUIT.

Basic usage example::

    from iminuit import Minuit
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2
    m = Minuit(f)
    m.migrad()
    print(m.values)  # {'x': 2,'y': 3,'z': 4}
    print(m.errors)  # {'x': 1,'y': 1,'z': 1}

Further information:

* Code: https://github.com/iminuit/iminuit
* Docs: http://iminuit.readthedocs.org/
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
# try:
#     from numpy.testing import Tester
#     test = Tester().test
# except:
#     pass
# An alternative would be to use `nose` directly to avoid the `numpy` depencency
# `scikit-image` is a nice example we could maybe copy & paste 
# https://github.com/scikit-image/scikit-image/blob/master/skimage/__init__.py
