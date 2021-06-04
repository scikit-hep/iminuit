from contextlib import contextmanager
from iminuit.testing import sphere_np
from iminuit import util, _repr_text as tx, Minuit
from argparse import Namespace


class PrintAssert:
    data = ""

    def __init__(self, expected):
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, *args):
        assert self.data == self.expected

    def text(self, arg):
        self.data += arg

    def pretty(self, arg):
        if hasattr(arg, "_repr_pretty_"):
            arg._repr_pretty_(self, False)
        else:
            self.data += repr(arg)

    def breakable(self):
        self.data += "\n"

    @contextmanager
    def group(self, n, start, stop):
        self.data += start
        yield
        self.data += stop


def test_Minuit():
    m = Minuit(sphere_np, (0, 0))
    with PrintAssert("<Minuit ...>") as pr:
        m._repr_pretty_(pr, True)

    with PrintAssert(tx.params(m.params)) as pr:
        m._repr_pretty_(pr, False)

    m.migrad()

    expected = (
        tx.fmin(m.fmin) + "\n" + tx.params(m.params) + "\n" + tx.matrix(m.covariance)
    )
    with PrintAssert(expected) as pr:
        m._repr_pretty_(pr, False)

    m.minos()

    expected = (
        tx.fmin(m.fmin)
        + "\n"
        + tx.params(m.params)
        + "\n"
        + tx.merrors(m.merrors)
        + "\n"
        + tx.matrix(m.covariance)
    )
    with PrintAssert(expected) as pr:
        m._repr_pretty_(pr, False)


def test_Matrix():
    m = util.Matrix(("a", "b"))
    m[:] = [[1, 2], [3, 4]]

    with PrintAssert("<Matrix ...>") as pr:
        m._repr_pretty_(pr, True)

    with PrintAssert(tx.matrix(m)) as pr:
        m._repr_pretty_(pr, False)


def test_Param():
    values = 3, "foo", 1.2, 3.4, None, False, False, 42, None
    p = util.Param(*values)

    with PrintAssert("Param(...)") as pr:
        p._repr_pretty_(pr, True)

    with PrintAssert(tx.params([p])) as pr:
        p._repr_pretty_(pr, False)


def test_Params():
    p = util.Params([util.Param(3, "foo", 1.2, 3.4, None, False, False, 42, None)])

    with PrintAssert("Params(...)") as pr:
        p._repr_pretty_(pr, True)

    with PrintAssert(tx.params(p)) as pr:
        p._repr_pretty_(pr, False)


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

    with PrintAssert("<MError ...>") as pr:
        me._repr_pretty_(pr, True)

    with PrintAssert(tx.merrors({None: me})) as pr:
        me._repr_pretty_(pr, False)


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

    with PrintAssert("<MErrors ...>") as pr:
        mes._repr_pretty_(pr, True)

    with PrintAssert(tx.merrors(mes)) as pr:
        mes._repr_pretty_(pr, False)


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
        errordef=1,
        state=[],
    )
    fmin = util.FMin(fm, "foo", 1, 2, 1, 0.1)

    with PrintAssert("<FMin ...>") as pr:
        fmin._repr_pretty_(pr, True)

    with PrintAssert(tx.fmin(fmin)) as pr:
        fmin._repr_pretty_(pr, False)
