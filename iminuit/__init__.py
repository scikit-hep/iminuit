"""IMinuit is a nice Python interface to MINUIT.
"""

__all__ = [
    'Minuit',
    'describe',
    'Struct',
    'InitialParamWarning',
    '__version__'
]
from ._libiminuit import *
from .util import describe, Struct
from .iminuit_warnings import *
from .info import __version__
