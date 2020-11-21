from iminuit import util, _repr_text
import pytest
from argparse import Namespace
from numpy.testing import assert_equal, assert_allclose
from contextlib import contextmanager


class Printer:
    data = ""

    def text(self, arg):
        self.data += arg

    def pretty(self, arg):
        if hasattr(arg, "_repr_pretty_"):
            arg._repr_pretty_(self, False)
        else:
            self.data += repr(arg)

    def breakable(self):
        self.data += " "

    @contextmanager
    def group(self, n, start, stop):
        self.data += start
        yield
        self.data += stop


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
    v = util.ValueView(
        Namespace(
            _var2pos={"x": 0, "y": 1},
            _pos2var=("x", "y"),
            npar=2,
            _last_state=(
                Namespace(value=1.0),
                Namespace(value=2.02),
            ),
        )
    )

    assert repr(v) == "<ValueView x=1.0 y=2.02>"
    assert str(v) == repr(v)


def test_Matrix():
    m = util.Matrix(("a", "b"))
    m[:] = [[1, 1], [1, 4]]
    assert_equal(m, ((1, 1), (1, 4)))
    assert repr(m) == "<Matrix a=0 b=1\n[[1. 1.]\n [1. 4.]]\n>"
    c = m.correlation()
    assert_allclose(c, ((1.0, 0.5), (0.5, 1.0)))
    assert m["a", "b"] == 1.0
    assert m["a", 1] == 1.0
    assert m[1, 1] == 4.0
    assert_equal(m["b"], (1, 4))
    assert str(m) == repr(m)

    p = Printer()
    m._repr_pretty_(p, False)
    assert p.data == _repr_text.matrix(m)
    p = Printer()
    m._repr_pretty_(p, True)
    assert p.data == "<Matrix ...>"

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
    assert str(p) == repr(p)

    pr = Printer()
    p._repr_pretty_(pr, False)
    assert pr.data == _repr_text.params([p])
    pr = Printer()
    p._repr_pretty_(pr, True)
    assert pr.data == "Param(...)"


def test_Params():
    p = util.Params(
        [
            util.Param(
                3, "foo", 1.2, 3.4, None, False, False, True, True, False, 42, None
            )
        ]
    )

    pr = Printer()
    p._repr_pretty_(pr, False)
    assert pr.data == _repr_text.params(p)
    pr = Printer()
    p._repr_pretty_(pr, True)
    assert pr.data == "Params(...)"

    assert repr(p) == repr((p[0],))
    assert str(p) == repr(p)


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

    pr = Printer()
    me._repr_pretty_(pr, False)
    assert pr.data == _repr_text.merrors({None: me})
    pr = Printer()
    me._repr_pretty_(pr, True)
    assert pr.data == "<MError ...>"

    assert repr(me) == (
        "<MError number=1 name='x' lower=0.1 upper=0.2 is_valid=True lower_valid=True "
        "upper_valid=True at_lower_limit=False at_upper_limit=False "
        "at_lower_max_fcn=False at_upper_max_fcn=False lower_new_min=False "
        "upper_new_min=False nfcn=11 min=0.7>"
    )
    assert str(me) == repr(me)

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

    pr = Printer()
    mes._repr_pretty_(pr, False)
    assert pr.data == _repr_text.merrors(mes)
    pr = Printer()
    mes._repr_pretty_(pr, True)
    assert pr.data == "<MErrors ...>"

    assert repr(mes) == f"<MErrors\n  {mes['x']!r}\n>"
    assert str(mes) == repr(mes)


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

    pr = Printer()
    fmin._repr_pretty_(pr, False)
    assert pr.data == _repr_text.fmin(fmin)
    pr = Printer()
    fmin._repr_pretty_(pr, True)
    assert pr.data == "<FMin ...>"

    assert repr(fmin) == (
        "<FMin edm=1.23456e-10 fval=1.23456e-10 has_accurate_covar=True"
        " has_covariance=True has_made_posdef_covar=False has_parameters_at_limit=False"
        " has_posdef_covar=True has_reached_call_limit=False has_valid_parameters=True"
        " hesse_failed=False is_above_max_edm=False is_valid=True"
        " nfcn=1 ngrad=2 tolerance=0.1 up=0.5>"
    )
    assert str(fmin) == repr(fmin)


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
