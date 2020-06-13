from iminuit import util
import pytest


def test_fitarg_rename():
    fitarg = {"x": 1, "limit_x": (2, 3), "fix_x": True, "error_x": 10}

    def ren(x):
        return "z_" + x

    newfa = util.fitarg_rename(fitarg, ren)
    assert "z_x" in newfa
    assert "limit_z_x" in newfa
    assert "error_z_x" in newfa
    assert "fix_z_x" in newfa
    assert len(newfa) == 4


def test_fitarg_rename_strprefix():
    fitarg = {"x": 1, "limit_x": (2, 3), "fix_x": True, "error_x": 10}
    newfa = util.fitarg_rename(fitarg, "z")
    assert "z_x" in newfa
    assert "limit_z_x" in newfa
    assert "error_z_x" in newfa
    assert "fix_z_x" in newfa
    assert len(newfa) == 4


def test_arguments_from_docstring():
    s = "f(x, y, z)"
    ok, a = util.arguments_from_docstring(s)
    assert ok
    assert a == ["x", "y", "z"]
    # this is a hard one
    s = "Minuit.migrad( int ncall_me =10000, [resume=True, int nsplit=1])"
    ok, a = util.arguments_from_docstring(s)
    assert ok
    assert a == ["ncall_me", "resume", "nsplit"]


def test_Matrix():
    x = util.Matrix(("a", "b"), [[1, 2], [3, 4]])
    assert x[0] == (1, 2)
    assert x[1] == (3, 4)
    assert x == ((1, 2), (3, 4))
    assert repr(x) == "((1, 2), (3, 4))"
    with pytest.raises(TypeError):
        x[0][0] = 1


def test_Param():
    keys = "number name value error is_const is_fixed has_limits has_lower_limit has_upper_limit lower_limit upper_limit".split()  # noqa: E501
    values = 3, "foo", 1.2, 3.4, False, False, True, True, False, 42, None
    p = util.Param(*values)

    assert p.has_lower_limit is True
    assert p.has_upper_limit is False
    assert p["has_lower_limit"] is True
    assert p["lower_limit"] == 42
    assert p["upper_limit"] is None
    assert "upper_limit" in p
    assert "foo" not in p

    assert list(p) == keys
    assert p.keys() == tuple(keys)
    assert p.values() == values
    assert p.items() == tuple((k, v) for k, v in zip(keys, values))

    assert (
        str(p)
        == "Param(number=3, name='foo', value=1.2, error=3.4, is_const=False, is_fixed=False, has_limits=True, has_lower_limit=True, has_upper_limit=False, lower_limit=42, upper_limit=None)"  # noqa: E501
    )


def test_MError():
    keys = "name is_valid lower upper lower_valid upper_valid at_lower_limit at_upper_limit at_lower_max_fcn at_upper_max_fcn lower_new_min upper_new_min nfcn min".split()  # noqa: E501
    values = (
        "Foo",
        True,
        0.1,
        1.2,
        True,
        True,
        False,
        False,
        False,
        False,
        0.1,
        1.2,
        42,
        0.2,
    )

    m = util.MError(*values)

    assert m.keys() == tuple(keys)
    assert m.values() == values
    assert m.name == "Foo"
    assert m["name"] == "Foo"


def test_FMin():
    keys = "fval edm tolerance nfcn ncalls up is_valid has_valid_parameters has_accurate_covar has_posdef_covar has_made_posdef_covar hesse_failed has_covariance is_above_max_edm has_reached_call_limit".split()  # noqa: E501
    values = (
        0.2,
        1e-3,
        0.1,
        10,
        10,
        1.2,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )

    f = util.FMin(*values)

    assert f.keys() == tuple(keys)
    assert f.values() == values
    assert f.fval == 0.2
    assert f["fval"] == 0.2


def test_MigradResult():
    fmin = util.FMin(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    params = util.Params([], None)
    mr = util.MigradResult(fmin, params)
    assert mr.fmin is fmin
    assert mr[0] is fmin
    assert mr.params is params
    assert mr[1] is params
    a, b = mr
    assert a is fmin
    assert b is params


def test_normalize_limit():
    assert util._normalize_limit(None) is None
    assert util._normalize_limit((None, 2)) == (-util.inf, 2)
    assert util._normalize_limit((2, None)) == (2, util.inf)
    assert util._normalize_limit((None, None)) == (-util.inf, util.inf)
    with pytest.raises(ValueError):
        util._normalize_limit((3, 2))


def test_guess_initial_value():
    assert util._guess_initial_value(None) == 0
    assert util._guess_initial_value((-util.inf, 0)) == -1
    assert util._guess_initial_value((0, util.inf)) == 1
    assert util._guess_initial_value((1.0, 2.0)) == 1.5


def test_guess_initial_step():
    assert util._guess_initial_step(0) == 0.1
    assert util._guess_initial_step(1) == 0.01
