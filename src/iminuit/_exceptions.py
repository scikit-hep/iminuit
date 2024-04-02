# NumPy 2 and recent 1.x place this in exceptions
try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:
    from numpy import VisibleDeprecationWarning

__all__ = ["VisibleDeprecationWarning"]
