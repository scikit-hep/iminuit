from iminuit._deprecated import deprecated
import pytest
import numpy as np


def test_deprecated_func():
    @deprecated("bla")
    def func(x):
        pass

    with pytest.warns(np.VisibleDeprecationWarning, match="func is deprecated: bla"):
        func(1)
