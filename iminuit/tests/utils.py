from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__all__ = ['requires_dependency']


def requires_dependency(name):
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
    try:
        __import__(name)
        skip_it = False
    except ImportError:
        skip_it = True

    reason = 'Missing dependency: {}'.format(name)
    import pytest
    return pytest.mark.skipif(skip_it, reason=reason)
