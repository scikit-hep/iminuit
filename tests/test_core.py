import gc

from iminuit._core import (
    FCN,
    MnUserParameterState,
    MnUserTransformation,
    MnMigrad,
    MnStrategy,
    MnScan,
    MnMinos,
    FunctionMinimum,
    MnSimplex,
    MnUserCovariance,
    MnPrint,
    MinimumState,
)
from pytest import approx
import pytest
import pickle
import numpy as np


def test_MnStrategy():
    assert MnStrategy() == 1
    assert MnStrategy(0) == 0
    assert MnStrategy(2) == 2
    s = MnStrategy()
    s.strategy = 2
    assert s.strategy == 2
    assert s != 1
    assert not (s != 2)


def test_MnUserCovariance():
    c = MnUserCovariance((1, 2, 3), 2)
    assert c.nrow == 2

    assert c[(0, 0)] == 1
    assert c[(1, 0)] == 2
    assert c[(0, 1)] == 2
    assert c[(1, 1)] == 3

    # valid negative indices
    assert c[(-1, -1)] == 3
    assert c[(-2, -2)] == 1

    # out-of-range indices raise instead of crashing
    for bad in ((-3, 0), (0, -3), (2, 0), (0, 2)):
        with pytest.raises(IndexError):
            c[bad]

    pkl = pickle.dumps(c)
    c2 = pickle.loads(pkl)
    assert c2.nrow == 2
    assert c2[(0, 0)] == 1
    assert c2[(1, 1)] == 3
    assert c2 == c


def fn(x, y):
    return 10 + x**2 + ((y - 1) / 2) ** 2


def fn_grad(x, y):
    return (2 * x, y - 1)


def test_MnUserParameterState():
    st = MnUserParameterState()
    st.add("x", 1, 0.2)
    st.add("😁", 3, 0.3, 1, 4)
    assert len(st) == 2
    assert st[0].number == 0
    assert st[0].name == "x"
    assert st[0].value == 1
    assert st[0].error == 0.2
    assert st[1].number == 1
    assert st[1].name == "😁"
    assert st[1].value == 3
    assert st[1].error == 0.3
    assert st[1].lower_limit == 1
    assert st[1].upper_limit == 4

    st2 = MnUserParameterState(st)
    assert st2 == st

    st2.set_value(0, 1.1)

    assert st2 != st

    # valid negative index works
    assert st[-1].name == "😁"
    assert st[-2].name == "x"

    # out-of-range indices raise IndexError instead of crashing (SIGBUS)
    for bad in (-(len(st) + 1), len(st)):
        with pytest.raises(IndexError):
            st[bad]


def test_MnUserParameterState_limit_equality():
    # parameters/states differing only in their limit values must not compare equal
    a = MnUserParameterState()
    a.add("x", 0, 1, -1, 5)
    b = MnUserParameterState()
    b.add("x", 0, 1, -2, 7)
    assert a[0] != b[0]
    assert a != b

    c = MnUserParameterState()
    c.add("x", 0, 1, -1, 5)
    assert a[0] == c[0]
    assert a == c


def test_MnUserParameterState_iter_keep_alive():
    # iterating over a temporary state must keep the state alive (keep_alive<0, 1>)
    def make_params():
        st = MnUserParameterState()
        st.add("a", 1, 0.1)
        st.add("b", 2, 0.1)
        return iter(st)

    it = make_params()
    gc.collect()
    names = [p.name for p in it]
    assert names == ["a", "b"]


def test_MnMinos_keep_alive():
    # MnMinos stores references to the FCN and FunctionMinimum; constructing it
    # from temporaries and dropping all other references must not crash.
    def make_minos():
        fcn = FCN(lambda x: 10 + x**2, None, None, None, False, 1)
        state = MnUserParameterState()
        state.add("x", 5, 0.1)
        fmin = MnMigrad(fcn, state, 1)(0, 0.1)
        return MnMinos(
            FCN(lambda x: 10 + x**2, None, None, None, False, 1),
            fmin,
            MnStrategy(1),
        )

    minos = make_minos()
    gc.collect()
    err = minos(0, 100, 0.01)
    assert err.lower == approx(-1, abs=1e-2)
    assert err.upper == approx(1, abs=1e-2)


def test_MnPrint_prefix():
    # the prefix string must stay alive for the lifetime of the MnPrint object
    prev = MnPrint.global_level
    try:
        p = MnPrint("some_prefix", 3)
        # exercise the logging methods; they must not produce garbage/crash
        p.info("hello")
        p.warn("warning")
    finally:
        MnPrint.global_level = prev


def test_MnMigrad():
    fcn = FCN(fn, None, None, None, False, 1)
    state = MnUserParameterState()
    state.add("x", 5, 0.1)
    state.add("y", 3, 0.2, -5, 5)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    assert fmin.is_valid
    state = fmin.state
    assert state[0].value == approx(0, abs=5e-3)
    assert state[0].error == approx(1, abs=5e-3)
    assert state[1].value == approx(1, abs=5e-3)
    assert state[1].error == approx(2, abs=6e-2)
    assert fcn._nfcn > 0
    assert fcn._ngrad == 0


def test_MnMigrad_grad():
    fcn = FCN(lambda x: 10 + x**2, lambda x: [2 * x], None, None, False, 1)
    state = MnUserParameterState()
    state.add("x", 5, 0.1)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == 1
    assert state[0].number == 0
    assert state[0].name == "x"
    assert state[0].value == approx(0, abs=1e-3)
    assert state[0].error == approx(1, abs=1e-3)
    assert fcn._nfcn > 0
    assert fcn._ngrad > 0


def test_MnMigrad_g2():
    # g2 is diagonal of hessian(chi2)
    fcn = FCN(
        lambda x: 10 + (x / 4) ** 2,
        lambda x: [x / 8],
        lambda x: [1 / 8],
        None,
        False,
        1,
    )
    state = MnUserParameterState()
    state.add("x", 5, 0.1)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == 1
    assert state[0].number == 0
    assert state[0].name == "x"
    assert state[0].value == approx(0, abs=1e-3)
    assert state[0].error == approx(4, abs=1e-3)
    assert fcn._nfcn > 0
    assert fcn._ngrad > 0
    assert fcn._ng2 > 0


def test_MnMigrad_hessian():
    fcn = FCN(
        lambda x: 10 + (x / 4) ** 2,
        lambda x: [x / 8],
        None,
        lambda x: [[1 / 8]],
        False,
        1,
    )
    state = MnUserParameterState()
    state.add("x", 5, 0.1)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == 1
    assert state[0].number == 0
    assert state[0].name == "x"
    assert state[0].value == approx(0, abs=1e-3)
    assert state[0].error == approx(4, abs=1e-3)
    assert fcn._nfcn > 0
    assert fcn._ngrad > 0
    assert fcn._ng2 == 0
    assert fcn._nhessian > 0


def test_MnMigrad_g2_hessian():
    # if both g2 and hessian are available, hessian is used
    fcn = FCN(
        lambda x, y: 10 + (2 * x) ** 2 + ((y - 1) / 4) ** 2,
        lambda x, y: [8 * x, (y - 1) / 8],
        lambda x, y: [-1, -1],  # not used
        lambda x, y: [[8.0, 0], [0, 1 / 8]],
        False,
        1,
    )
    # cov = inv(hessian / 2) = [[1/4, 0], [0, 16]]
    # err[0] = cov[0, 0]**0.5 = 0.5
    # err[1] = cov[1, 1]**0.5 = 4
    state = MnUserParameterState()
    state.add("x", 5, 0.1)
    state.add("y", 5, 0.1)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == 2
    assert state[0].number == 0
    assert state[0].name == "x"
    assert state[0].value == approx(0, abs=1e-2)
    assert state[0].error == approx(0.5, abs=1e-2)
    assert state[1].number == 1
    assert state[1].name == "y"
    assert state[1].value == approx(1, abs=1e-2)
    assert state[1].error == approx(4, abs=1e-2)
    assert fcn._nfcn > 0
    assert fcn._ngrad > 0
    assert fcn._ng2 == 0
    assert fcn._nhessian > 0


def test_MnMigrad_cfunc():
    nb = pytest.importorskip("numba")

    c_sig = nb.types.double(nb.types.uintc, nb.types.CPointer(nb.types.double))

    y = np.arange(5)

    @nb.cfunc(c_sig)
    def fcn(n, x):
        x = nb.carray(x, (n,))
        r = 0.0
        for i in range(n):
            r += (y[i] - x[i]) ** 2
        return r

    fcn = FCN(fcn, None, None, None, True, 1)
    assert fcn._cfcn is True
    state = MnUserParameterState()
    for i in range(len(y)):
        state.add(f"x{i}", 5, 0.1)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == len(y)
    for i, p in enumerate(state):
        assert p.number == i
        assert p.value == approx(i, abs=1e-3)
        assert p.error == approx(1, abs=1e-3)


def test_MnMigrad_np():
    fcn = FCN(
        lambda xy: 10 + xy[0] ** 2 + ((xy[1] - 1) / 2) ** 2,
        lambda xy: [2 * xy[0], (xy[1] - 1)],
        None,
        None,
        True,
        1,
    )
    state = MnUserParameterState()
    state.add("x", 5, 0.1)
    state.add("😁", 3, 0.2, -5, 5)
    assert len(state) == 2
    str = MnStrategy(2)
    migrad = MnMigrad(fcn, state, str)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == 2
    assert state[0].number == 0
    assert state[0].name == "x"
    assert state[0].value == approx(0, abs=1e-2)
    assert state[0].error == approx(1, abs=1e-2)
    assert state[1].number == 1
    assert state[1].name == "😁"
    assert state[1].value == approx(1, abs=1e-2)
    assert state[1].error == approx(2, abs=6e-2)
    assert fcn._nfcn > 0
    assert fcn._ngrad > 0


def test_MnScan():
    fcn = FCN(lambda x: 10 + x**2, None, None, None, False, 1)
    state = MnUserParameterState()
    state.add("x", 2, 5)
    scan = MnScan(fcn, state, 1)
    fmin = scan(0, 0.1)
    assert fmin.is_valid
    state = fmin.state
    assert len(state) == 1
    assert state[0].value == approx(0, abs=1e-2)


def test_MnSimplex():
    fcn = FCN(lambda x: 10 + x**2, None, None, None, False, 1)
    state = MnUserParameterState()
    state.add("x", 2, 5)
    simplex = MnSimplex(fcn, state, 1)
    fmin = simplex(0, 0.1)
    assert fmin.is_valid
    state = fmin.state
    assert len(state) == 1
    assert state[0].value == approx(0, abs=5e-2)


def test_FunctionMinimum():
    fcn = FCN(lambda x: 10 + x**2, None, None, None, False, 1)
    st = MnUserParameterState()
    st.add("x", 0.01, 5)
    str = MnStrategy(1)
    fm1 = FunctionMinimum(fcn, st, str, 0.2)
    assert fm1.is_valid
    assert len(fm1.state) == 1
    assert fm1.fval == 10.0001
    fm2 = FunctionMinimum(fcn, st, str, 0)
    assert not fm2.is_valid


def test_FunctionMinimum_pickle():
    st = MnUserParameterState()
    st.add("x", 1, 0.1)
    st.add("y", 2, 0.1, 1, 3)
    fm = FunctionMinimum(FCN(fn, None, None, None, False, 1), st, 1, 0.1)

    pkl = pickle.dumps(fm)
    fm2 = pickle.loads(pkl)

    assert len(fm.state) == len(fm2.state)
    assert fm.state == fm2.state
    assert fm.edm == fm2.edm
    assert fm.fval == fm2.fval
    assert fm.is_valid == fm2.is_valid
    assert fm.has_accurate_covar == fm2.has_accurate_covar
    assert fm.has_posdef_covar == fm2.has_posdef_covar
    assert fm.has_made_posdef_covar == fm2.has_made_posdef_covar
    assert fm.hesse_failed == fm2.hesse_failed
    assert fm.has_covariance == fm2.has_covariance
    assert fm.is_above_max_edm == fm2.is_above_max_edm
    assert fm.has_reached_call_limit == fm2.has_reached_call_limit
    assert fm.errordef == fm2.errordef


def test_MnUserTransformation_pickle():
    tr = MnUserTransformation()

    pkl = pickle.dumps(tr)
    tr2 = pickle.loads(pkl)

    assert len(tr2) == len(tr)


def test_MinimumState_pickle():
    st = MinimumState(3)

    pkl = pickle.dumps(st)
    st2 = pickle.loads(pkl)

    assert st.vec == st2.vec
    assert st.fval == st2.fval
    assert st.edm == st2.edm
    assert st.nfcn == st2.nfcn
    assert st.is_valid == st2.is_valid
    assert st.has_parameters == st2.has_parameters
    assert st.has_covariance == st2.has_covariance
