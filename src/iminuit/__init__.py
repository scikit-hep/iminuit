"""
Jupyter-friendly Python interface for the Minuit2 library in C++.

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
* Docs: https://scikit-hep.org/iminuit
"""

from iminuit.minuit import Minuit
from iminuit.minimize import minimize
from iminuit.util import describe
from importlib import metadata

__version__ = metadata.version("iminuit")

__all__ = ["Minuit", "minimize", "describe", "__version__"]
