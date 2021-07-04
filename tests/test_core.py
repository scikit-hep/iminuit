from iminuit._core import (
    FCN,
    MnUserParameterState,
    MnUserTransformation,
    MnMigrad,
    MnStrategy,
    MnScan,
    FunctionMinimum,
    MnSimplex,
    MnPrint,
    MnUserCovariance,
    MinimumState,
)
from pytest import approx
import pytest
import pickle


@pytest.fixture
def debug():
    prev = MnPrint.global_level
    MnPrint.global_level = 3
    MnPrint.show_prefix_stack(True)
    yield
    MnPrint.global_level = prev
    MnPrint.show_prefix_stack(False)


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

    pkl = pickle.dumps(c)
    c2 = pickle.loads(pkl)
    assert c2.nrow == 2
    assert c2[(0, 0)] == 1
    assert c2[(1, 1)] == 3
    assert c2 == c


def fn(x, y):
    return 10 + x ** 2 + ((y - 1) / 2) ** 2


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


def test_MnMigrad():
    fcn = FCN(fn, None, False, 1)
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
    fcn = FCN(lambda x: 10 + x ** 2, lambda x: [2 * x], False, 1)
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


@pytest.mark.parametrize("npar", (1, 2, 3))
def test_MnMigrad_cfunc(npar):
    nb = pytest.importorskip("numba")

    c_sig = nb.types.double(nb.types.uintc, nb.types.CPointer(nb.types.double))

    @nb.cfunc(c_sig)
    def fcn(n, x):
        x = nb.carray(x, (n,))
        r = 0.0
        for i in range(n):
            r += (x[i] - i) ** 2
        return r

    fcn = FCN(fcn, None, True, 1)
    state = MnUserParameterState()
    for i in range(npar):
        state.add(f"x{i}", 5, 0.1)
    migrad = MnMigrad(fcn, state, 1)
    fmin = migrad(0, 0.1)
    state = fmin.state
    assert len(state) == npar
    for i, p in enumerate(state):
        assert p.number == i
        assert p.value == approx(i, abs=1e-3)
        assert p.error == approx(1, abs=1e-3)


def test_MnMigrad_np():
    fcn = FCN(
        lambda xy: 10 + xy[0] ** 2 + ((xy[1] - 1) / 2) ** 2,
        lambda xy: [2 * xy[0], (xy[1] - 1)],
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
    fcn = FCN(lambda x: 10 + x ** 2, None, False, 1)
    state = MnUserParameterState()
    state.add("x", 2, 5)
    scan = MnScan(fcn, state, 1)
    fmin = scan(0, 0.1)
    assert fmin.is_valid
    state = fmin.state
    assert len(state) == 1
    assert state[0].value == approx(0, abs=1e-2)


def test_MnSimplex():
    fcn = FCN(lambda x: 10 + x ** 2, None, False, 1)
    state = MnUserParameterState()
    state.add("x", 2, 5)
    simplex = MnSimplex(fcn, state, 1)
    fmin = simplex(0, 0.1)
    assert fmin.is_valid
    state = fmin.state
    assert len(state) == 1
    assert state[0].value == approx(0, abs=5e-2)


def test_FunctionMinimum():
    fcn = FCN(lambda x: 10 + x ** 2, None, False, 1)
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
    fm = FunctionMinimum(FCN(fn, None, False, 1), st, 1, 0.1)

    pkl = pickle.dumps(fm)
    fm2 = pickle.loads(pkl)

    assert len(fm.state) == len(fm2.state)
    assert fm.state == fm2.state
    assert fm.edm == fm2.edm
    assert fm.fval == fm2.fval
    assert fm.is_valid == fm2.is_valid
    assert fm.has_valid_parameters == fm2.has_valid_parameters
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
