"""Jupyter-friendly Python interface for the Minuit2 library in C++.

Basic usage example::

    from iminuit import Minuit

    def fcn(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(fcn, x=0, y=0, z=0)
    m.migrad()
    m.hesse()

    print(m.values)  # 'x': 2, 'y': 3, 'z': 4
    print(m.errors)  # 'x': 1, 'y': 1, 'z': 1

Further information:

* Code: https://github.com/scikit-hep/iminuit
* Docs: https://iminuit.readthedocs.io
"""
import sys
from .minuit import Minuit
from .minimize import minimize
from .util import describe

if sys.version_info >= (3, 8):
    from importlib import metadata  # pragma: no cover
else:
    import importlib_metadata as metadata  # pragma: no cover

__version__ = metadata.version("iminuit")

__all__ = ["Minuit", "minimize", "describe", "__version__"]
