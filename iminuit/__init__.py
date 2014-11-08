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
