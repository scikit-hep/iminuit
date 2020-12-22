"""Jupyter-friendly Python interface for the Minuit2 library in C++.

Basic usage example::

    from iminuit import Minuit

    def fcn(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    fcn.errordef = Minuit.LEAST_SQUARES

    m = Minuit(fcn, x=0, y=0, z=0)
    m.migrad()
    m.hesse()

    print(m.values)  # 'x': 2, 'y': 3, 'z': 4
    print(m.errors)  # 'x': 1, 'y': 1, 'z': 1

Further information:

* Code: https://github.com/scikit-hep/iminuit
* Docs: https://iminuit.readthedocs.io
"""

__all__ = ["Minuit", "minimize", "describe", "__version__"]

from .minuit import Minuit
from .minimize import minimize
from .util import describe
from .version import version as __version__
