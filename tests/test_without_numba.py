from iminuit._hide_modules import hide_modules
import numpy as np
import pytest
from numpy.testing import assert_allclose

pytest.importorskip("numba")


def npa(*args, **kwargs):
    return np.array(args, **kwargs)


@pytest.mark.parametrize(
    "func,args",
    (
        ("_safe_log", (npa(1.2, 0),)),
        ("_z_squared", (npa(1.2, 0), npa(1.2, 1.0), npa(1.2, 1))),
        ("_unbinned_nll", (npa(1.2, 0),)),
        ("multinominal_chi2", (npa(1, 0), npa(1.2, 0))),
        ("chi2", (npa(1.2, 0), npa(1.2, 1.0), npa(1.2, 0.1))),
        ("poisson_chi2", (npa(1, 0), npa(1.2, 0.1))),
        ("_soft_l1_cost", (npa(1.2, 0), npa(1.2, 0.1), npa(1.0, 1.0))),
    ),
)
def test_no_numba(func, args):
    import iminuit.cost as cost

    assert "jit" in dir(cost)

    expected = getattr(cost, func)(*args)

    with hide_modules("numba", reload="iminuit.cost"):
        import iminuit.cost as cost

        assert "jit" not in dir(cost)

        got = getattr(cost, func)(*args)

    assert_allclose(got, expected)
