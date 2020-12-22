from iminuit import util
import pytest
from argparse import Namespace
from numpy.testing import assert_equal, assert_allclose
import numpy as np
from iminuit._core import MnUserParameterState


def test_ndim():
    ndim = util._ndim
    assert ndim(1) == 0
    assert ndim([]) == 1
    assert ndim([[]]) == 2
    assert ndim(None) == 0
    assert ndim((None, None)) == 1
    assert ndim(((1, 2), None)) == 2
    assert ndim((None, (1, 2))) == 2


def test_ValueView():
    state = MnUserParameterState()
    state.add("x", 1.0, 0.1)
    state.add("y", 2.2, 0.1)
    state.add("z", 3.3, 0.1)

    v = util.ValueView(
        Namespace(
            _var2pos={"x": 0, "y": 1, "z": 2},
            _pos2var=("x", "y", "z"),
            npar=3,
            _last_state=state,
            _copy_state_if_needed=lambda: None,
        )
    )

    assert repr(v) == "<ValueView x=1.0 y=2.2 z=3.3>"
    assert str(v) == repr(v)

    v[:] = (1, 2, 3)
    assert_equal(v, (1, 2, 3))
    v[1:] = 4
    assert_equal(v, (1, 4, 4))
    v["y"] = 2
    assert_equal(v, (1, 2, 4))
    v["y":] = 3
    assert_equal(v, (1, 3, 3))
    v[:"z"] = 2
    assert_equal(v, (2, 2, 3))


def test_Matrix():
    m = util.Matrix(("a", "b"))
    m[:] = [[1, 1], [1, 4]]
    assert_equal(m, ((1, 1), (1, 4)))
    assert repr(m) == "[[1. 1.]\n [1. 4.]]"
    c = m.correlation()
    assert_allclose(c, ((1.0, 0.5), (0.5, 1.0)))
    assert m["a", "b"] == 1.0
    assert m["a", 1] == 1.0
    assert m[1, 1] == 4.0
    assert_equal(m["b"], (1, 4))

    m *= 2
    assert_equal(m, ((2, 2), (2, 8)))
    assert_allclose(np.dot(m, (1, 1)), (4, 10))

    with pytest.raises(TypeError):
        util.Matrix("ab")

    with pytest.raises(TypeError):
        util.Matrix(1)


def test_Param():
    values = 3, "foo", 1.2, 3.4, None, False, False, True, True, False, 42, None
    p = util.Param(*values)

    assert p.number == 3
    assert p.name == "foo"
    assert p.value == 1.2
    assert p.error == 3.4
    assert p.merror is None
    assert p.is_const == False
    assert p.is_fixed == False
    assert p.has_limits == True
    assert p.has_lower_limit is True
    assert p.has_upper_limit is False
    assert p.lower_limit == 42
    assert p.upper_limit is None

    assert repr(p) == (
        "Param(number=3, name='foo', value=1.2, error=3.4, merror=None, "
        "is_const=False, is_fixed=False, has_limits=True, has_lower_limit=True, "
        "has_upper_limit=False, lower_limit=42, upper_limit=None)"
    )


def test_Params():
    p = util.Params(
        [
            util.Param(
                0, "foo", 1.2, 3.4, None, False, False, True, True, False, 42, None
            ),
            util.Param(
                1, "bar", 3.4, 4.5, None, False, False, True, True, False, 42, None
            ),
        ]
    )

    assert repr(p) == repr((p[0], p[1]))
    assert p[0].number == 0
    assert p[1].number == 1
    assert p["foo"].number == 0
    assert p["bar"].number == 1


def test_MError():
    me = util.MError(
        1,
        "x",
        0.1,
        0.2,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        11,
        0.7,
    )

    assert repr(me) == (
        "<MError number=1 name='x' lower=0.1 upper=0.2 is_valid=True lower_valid=True "
        "upper_valid=True at_lower_limit=False at_upper_limit=False "
        "at_lower_max_fcn=False at_upper_max_fcn=False lower_new_min=False "
        "upper_new_min=False nfcn=11 min=0.7>"
    )

    assert me == util.MError(
        1,
        "x",
        0.1,
        0.2,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        11,
        0.7,
    )

    assert me != util.MError(
        1,
        "x",
        0.1,
        0.2,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        11,
        0.8,
    )


def test_MErrors():
    mes = util.MErrors(
        x=util.MError(
            1,
            "x",
            0.1,
            0.2,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            11,
            0.7,
        )
    )

    assert repr(mes) == f"<MErrors\n  {mes['x']!r}\n>"


def test_FMin():
    fm = Namespace(
        fval=1.23456e-10,
        edm=1.23456e-10,
        errordef=0.5,
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
        "edm_goal",
        "errordef",
        "fval",
        "nfcn",
        "ngrad",
        "is_valid",
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
    assert fmin.edm_goal == 0.1
    assert fmin.has_parameters_at_limit == False

    assert fmin == util.FMin(fm, 1, 2, 0.1)
    assert fmin != util.FMin(fm, 1, 2, 0.3)

    assert repr(fmin) == (
        "<FMin edm=1.23456e-10 edm_goal=0.1 errordef=0.5 fval=1.23456e-10"
        " has_accurate_covar=True has_covariance=True has_made_posdef_covar=False"
        " has_parameters_at_limit=False has_posdef_covar=True"
        " has_reached_call_limit=False has_valid_parameters=True"
        " hesse_failed=False is_above_max_edm=False is_valid=True"
        " nfcn=1 ngrad=2>"
    )


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


def test_address_of_cfunc():
    nb = pytest.importorskip("numba")

    nb_sig = nb.types.double(nb.types.uintc, nb.types.CPointer(nb.types.double))

    @nb.cfunc(nb_sig)
    def fcn(n, x):
        x = nb.carray(x, (n,))
        r = 0.0
        for i in range(n):
            r += (x[i] - i) ** 2
        return r

    from ctypes import cast, c_void_p, CFUNCTYPE, POINTER, c_double, c_uint32

    address = cast(fcn.ctypes, c_void_p).value
    assert util._address_of_cfunc(fcn) == address

    # let's see if we can call the function pointer, going full circle
    c_sig = CFUNCTYPE(c_double, c_uint32, POINTER(c_double))
    c_fcn = cast(address, c_sig)

    v = np.array((1.0, 2.0))
    assert c_fcn(2, v.ctypes.data_as(POINTER(c_double))) == 2.0


def test_address_of_cfunc_bad_signature():
    nb = pytest.importorskip("numba")

    nb_sig = nb.types.double(nb.types.double, nb.types.CPointer(nb.types.double))

    @nb.cfunc(nb_sig)
    def fcn(y, x):
        return 0

    assert util._address_of_cfunc(fcn) == 0
