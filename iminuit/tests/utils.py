from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
import os

__all__ = [
    'requires_dependency',
    'assert_allclose',
]


def requires_dependency(*names):
    """Decorator to declare required dependencies for tests.

    Parameters
    ----------
    name : str
        Package name, e.g. 'numpy' or 'ipython'

    Examples
    --------

    ::

        from iminuit.tests.utils import requires_dependency

        @requires_dependency('numpy')
        def test_using_numpy():
            import numpy
            ...
    """
    skip_it = False
    for name in names:
        path = os.path.dirname(__file__)
        p = os.path.join(path, name + ".pyx")
        if os.path.exists(p):
            continue
        try:
            __import__(name)
        except ImportError:
            skip_it = True

    reason = 'Missing dependency: {}'.format(name)
    import pytest
    return pytest.mark.skipif(skip_it, reason=reason)
