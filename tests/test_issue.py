from iminuit import Minuit
from iminuit.util import IMinuitWarning
import pickle
import pytest
import numpy as np


def lsq(func):
    func.errordef = Minuit.LEAST_SQUARES
    return func


def test_issue_424():
    @lsq
    def fcn(x, y, z):
        return (x - 1) ** 2 + (y - 4) ** 2 / 2 + (z - 9) ** 2 / 3

    m = Minuit(fcn, x=0.0, y=0.0, z=0.0)
    m.migrad()

    m.fixed["x"] = True
    m.errors["x"] = 2
    m.hesse()  # this used to release x
    assert m.fixed["x"] is True
    assert m.errors["x"] == 2


def test_issue_544():
    @lsq
    def fcn(x, y):
        return x**2 + y**2

    m = Minuit(fcn, x=0, y=0)
    m.fixed = True
    with pytest.warns(IMinuitWarning):
        m.hesse()  # this used to cause a segfault


def test_issue_648():
    class F:
        errordef = 1
        first = True

        def __call__(self, a, b):
            if self.first:
                assert a == 1.0 and b == 2.0
                self.first = False
            return a**2 + b**2

    m = Minuit(F(), a=1, b=2)
    m.fixed["a"] = False  # this used to change a to b
    m.migrad()


def test_issue_643():
    @lsq
    def fcn(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(fcn, x=2, y=3, z=4)
    m.migrad()

    m2 = Minuit(fcn, x=m.values["x"], y=m.values["y"], z=m.values["z"])
    # used to call MnHesse when it was not needed and quickly exhaust call limit
    for i in range(10):
        m2.minos()

    m2.reset()
    # used to exhaust call limit, because calls to MnHesse did not reset call count
    for i in range(10):
        m2.values = m.values
        m2.minos()


def test_issue_669():
    @lsq
    def fcn(x, y):
        return x**2 + (y / 2) ** 2

    m = Minuit(fcn, x=0, y=0)

    m.migrad()

    xy1 = m.mncontour(x="x", y="y", size=10)
    xy2 = m.mncontour(x="y", y="x", size=10)  # used to fail

    # needs better way to compare polygons
    for x, y in xy1:
        match = False
        for (y2, x2) in xy2:
            if abs(x - x2) < 1e-3 and abs(y - y2) < 1e-3:
                match = True
                break
        assert match


@lsq
def fcn(par):
    return np.sum(par**2)


def grad(par):
    return 2 * par


def test_issue_687():

    start = np.zeros(3)
    m = Minuit(fcn, start)

    m.migrad()
    s_m = str(m)

    s = pickle.dumps(m)
    m2 = pickle.loads(s)

    s_m2 = str(m2)  # this used to fail
    assert s_m == s_m2
