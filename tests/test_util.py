from iminuit import util
import pytest
from argparse import Namespace
from numpy.testing import assert_equal, assert_allclose


def test_ndim():
    ndim = util._ndim
    assert ndim(1) == 0
    assert ndim([]) == 1
    assert ndim([[]]) == 2
    assert ndim(None) == 0
    assert ndim((None, None)) == 1
    assert ndim(((1, 2), None)) == 2
    assert ndim((None, (1, 2))) == 2


def test_Matrix():
    x = util.Matrix({"a": 0, "b": 1})
    x[:] = [[1, 1], [1, 4]]
    assert_equal(x, ((1, 1), (1, 4)))
    assert repr(x) == "Matrix([[1., 1.],\n        [1., 4.]])"
    c = x.correlation()
    assert_allclose(c, ((1.0, 0.5), (0.5, 1.0)))


def test_Param():
    values = 3, "foo", 1.2, 3.4, False, False, True, True, False, 42, None
    p = util.Param(*values)

    assert p.number == 3
    assert p.name == "foo"
    assert p.value == 1.2
    assert p.error == 3.4
    assert p.is_const == False
    assert p.is_fixed == False
    assert p.has_limits == True
    assert p.has_lower_limit is True
    assert p.has_upper_limit is False
    assert p.lower_limit == 42
    assert p.upper_limit is None

    assert (
        str(p)
        == "Param(number=3, name='foo', value=1.2, error=3.4, is_const=False, is_fixed=False, has_limits=True, has_lower_limit=True, has_upper_limit=False, lower_limit=42, upper_limit=None)"  # noqa: E501
    )


def test_FMin():
    fm = Namespace(
        fval=1.23456e-10,
        edm=1.23456e-10,
        up=0.5,
        is_valid=True,
        has_valid_parameters=True,
        has_accurate_covar=True,
        has_posdef_covar=True,
        has_made_posdef_covar=False,
        hesse_failed=False,
        has_covariance=True,
        is_above_max_edm=False,
        has_reached_call_limit=False,
        has_parameters_at_limit=False,
        state=[],
    )
    fmin = util.FMin(fm, 1, 2, 0.1)
    assert {x for x in dir(fmin) if not x.startswith("_")} == {
        "edm",
        "fval",
        "nfcn",
        "ngrad",
        "up",
        "is_valid",
        "tolerance",
        "has_accurate_covar",
        "has_valid_parameters",
        "has_posdef_covar",
        "has_made_posdef_covar",
        "hesse_failed",
        "has_covariance",
        "is_above_max_edm",
        "has_reached_call_limit",
        "has_parameters_at_limit",
    }
    assert fmin.edm == 1.23456e-10
    assert fmin.has_parameters_at_limit == False


def test_normalize_limit():
    assert util._normalize_limit(None) == (-util.inf, util.inf)
    assert util._normalize_limit((None, 2)) == (-util.inf, 2)
    assert util._normalize_limit((2, None)) == (2, util.inf)
    assert util._normalize_limit((None, None)) == (-util.inf, util.inf)
    with pytest.raises(ValueError):
        util._normalize_limit((3, 2))


def test_guess_initial_step():
    assert util._guess_initial_step(0) == 0.1
    assert util._guess_initial_step(1) == 0.01
