"""MINUIT from Python - Fitting like a boss

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
    'minimize',
    'describe',
    'Struct',
    'InitialParamWarning',
    '__version__',
    'test',
]

from iminuit._libiminuit import *
from iminuit.minimize import minimize
from iminuit.util import describe, Struct
from iminuit.iminuit_warnings import *
from iminuit.info import __version__


def test(args=None):
    """Execute the iminuit tests.

    Requires pytest.

    From the command line:

        python -c 'import iminuit; iminuit.test()
    """
    # http://pytest.org/latest/usage.html#calling-pytest-from-python-code
    import pytest
    args = '--pyargs iminuit'
    pytest.main(args)
