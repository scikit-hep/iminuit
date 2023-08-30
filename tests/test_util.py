from iminuit import util
import pytest
from argparse import Namespace
from numpy.testing import assert_equal, assert_allclose
import numpy as np
from iminuit._core import MnUserParameterState
from iminuit._optional_dependencies import optional_module_for
import pickle
from iminuit._hide_modules import hide_modules

try:
    import scipy  # noqa

    scipy_available = True
except ModuleNotFoundError:
    scipy_available = False


def test_ndim():
    ndim = util._ndim
    assert ndim(1) == 0
    assert ndim([]) == 1
    assert ndim([[]]) == 2
    assert ndim(None) == 0
    assert ndim((None, None)) == 1
    assert ndim(((1, 2), None)) == 2
    assert ndim((None, (1, 2))) == 2


def test_BasicView():
    with pytest.raises(TypeError):
        util.BasicView(None, 2)


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

    assert v == v
    assert v == (1.0, 2.2, 3.3)
    assert v != (1.0, 2.1, 3.3)
    assert v != 0

    assert repr(v) == "<ValueView x=1.0 y=2.2 z=3.3>"
    assert str(v) == repr(v)

    v[:] = (1, 2, 3)

    assert_equal(v[:3], (1, 2, 3))
    assert_equal(v[0:3], (1, 2, 3))
    assert_equal(v[0:10], (1, 2, 3))

    assert_equal(v, (1, 2, 3))
    v[1:] = 4
    assert_equal(v, (1, 4, 4))
    v["y"] = 2
    assert_equal(v, (1, 2, 4))
    v["y":] = 3
    assert_equal(v, (1, 3, 3))
    v[:"z"] = 2
    assert_equal(v, (2, 2, 3))

    v_dict = v.to_dict()
    assert isinstance(v_dict, dict)
    assert v_dict["x"] == v["x"]
    assert v_dict["y"] == v["y"]
    assert v_dict["z"] == v["z"]

    v[:] = (1, 2, 3)
    assert_equal(v[["x", "z"]], (1, 3))
    assert_equal(v[[2, 0]], (3, 1))
    v[["x", "z"]] = (3, 1)
    assert_equal(v, (3, 2, 1))


def test_FixedView_as_mask_for_other_views():
    state = MnUserParameterState()
    state.add("x", 1, 0.1)
    state.add("y", 2, 0.1)
    state.add("z", 3, 0.1)

    fake_minuit = Namespace(
        _var2pos={"x": 0, "y": 1, "z": 2},
        _pos2var=("x", "y", "z"),
        npar=3,
        _last_state=state,
        _copy_state_if_needed=lambda: None,
    )

    v = util.ValueView(fake_minuit)
    f = util.FixedView(fake_minuit)
    f[1] = True

    assert_equal(f, [False, True, False])
    assert_equal(v[f], [2])
    assert_equal(v[~f], [1, 3])
    v[f] = 5
    assert_equal(v, [1, 5, 3])
    v[~f] = [2, 4]
    assert_equal(v, [2, 5, 4])


def test_FixedView_comparison_with_broadcasting():
    state = MnUserParameterState()
    state.add("x", 1, 0.1)
    state.add("y", 2, 0.1)
    state.add("z", 3, 0.1)

    fake_minuit = Namespace(
        _var2pos={"x": 0, "y": 1, "z": 2},
        _pos2var=("x", "y", "z"),
        npar=3,
        _last_state=state,
        _copy_state_if_needed=lambda: None,
    )

    f = util.FixedView(fake_minuit)

    assert_equal(f, [False, False, False])

    # broadcasting
    assert f == False  # noqa
    f[0] = True
    assert_equal(f, [True, False, False])
    assert f != False  # noqa


def test_Matrix():
    m = util.Matrix(("a", "b"))
    m[:] = [[1, 2], [2, 8]]
    assert_equal(m, ((1, 2), (2, 8)))
    assert repr(m) == "[[1. 2.]\n [2. 8.]]"
    c = m.correlation()
    assert_allclose(c, ((1.0, 0.5**0.5), (0.5**0.5, 1.0)))
    assert m["a", "b"] == 2.0
    assert m["a", 1] == 2.0
    assert m[1, "a"] == 2.0
    assert m[1, 1] == 8.0
    assert_equal(m[0], [1, 2])
    assert_equal(m["b"], (2, 8))
    assert_equal(m[:], [[1, 2], [2, 8]])
    assert_equal(m[:, 0], [1, 2])
    assert_equal(m[:, 1], [2, 8])

    # this swaps rows and cols
    assert_equal(m[[1, 0]], [[8, 2], [2, 1]])

    m *= 2
    assert_equal(m, ((2, 4), (4, 16)))

    m2 = np.dot(m, (1, 1))
    assert repr(m2) == "[ 6. 20.]"
    assert str(m2) == "[ 6. 20.]"
    assert_allclose(m2, (6, 20))

    # matrix is always square

    m = util.Matrix(("a", "b", "c"))
    m[:] = np.arange(9).reshape((3, 3))
    # [0 1 2
    #  3 4 5
    #  6 7 8]

    # m1 = m[:2]
    # assert_equal(m1, [[0, 1], [3, 4]])
    m2 = m[[0, 2]]
    assert_equal(m2, [[0, 2], [6, 8]])
    m3 = m[["a", "c"]]
    assert_equal(m3, [[0, 2], [6, 8]])

    d = m.to_dict()
    assert list(d.keys()) == [
        ("a", "a"),
        ("a", "b"),
        ("a", "c"),
        ("b", "b"),
        ("b", "c"),
        ("c", "c"),
    ]
    for k, v in d.items():
        assert v == m[k]

    with pytest.raises(TypeError):
        util.Matrix("ab")

    with pytest.raises(TypeError):
        util.Matrix(1)

    m2 = pickle.loads(pickle.dumps(m))
    assert type(m2) is util.Matrix

    assert_equal(m2, m)
    assert m2._var2pos == m._var2pos


def test_Param():
    values = 3, "foo", 1.2, 3.4, None, False, False, 42, None
    p = util.Param(*values)

    assert p.number == 3
    assert p.name == "foo"
    assert p.value == 1.2
    assert p.error == 3.4
    assert p.merror is None
    assert not p.is_const
    assert not p.is_fixed
    assert p.has_limits
    assert p.has_lower_limit
    assert not p.has_upper_limit
    assert p.lower_limit == 42
    assert p.upper_limit is None

    assert repr(p) == (
        "Param(number=3, name='foo', value=1.2, error=3.4, merror=None, "
        "is_const=False, is_fixed=False, lower_limit=42, upper_limit=None)"
    )


def test_Params():
    p = util.Params(
        [
            util.Param(0, "foo", 1.2, 3.4, None, False, False, 42, None),
            util.Param(1, "bar", 3.4, 4.5, None, False, False, 42, None),
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


@pytest.mark.parametrize("errordef", (0.5, 1.0))
def test_FMin(errordef):
    fm = Namespace(
        fval=1.23456e-10,
        edm=1.23456e-10,
        errordef=errordef,
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
    fmin = util.FMin(fm, "foo", 1, 2, 1, 0.1, 1.2)
    assert {x for x in dir(fmin) if not x.startswith("_")} == {
        "algorithm",
        "edm",
        "edm_goal",
        "errordef",
        "fval",
        "reduced_chi2",
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
        "time",
    }
    assert fmin.algorithm == "foo"
    assert fmin.edm == 1.23456e-10
    assert fmin.edm_goal == 0.1
    assert not fmin.has_parameters_at_limit
    assert fmin.time == 1.2

    assert fmin == util.FMin(fm, "foo", 1, 2, 1, 0.1, 1.2)
    assert fmin != util.FMin(fm, "foo", 1, 2, 1, 0.3, 1.2)
    assert fmin != util.FMin(fm, "bar", 1, 2, 1, 0.1, 1.2)
    assert fmin != util.FMin(fm, "foo", 1, 2, 1, 0.1, 1.5)

    if errordef == 1:
        reduced_chi2 = fmin.fval
    else:
        reduced_chi2 = np.nan

    assert repr(fmin) == (
        f"<FMin algorithm='foo' edm=1.23456e-10 edm_goal=0.1 errordef={errordef}"
        " fval=1.23456e-10"
        " has_accurate_covar=True has_covariance=True has_made_posdef_covar=False"
        " has_parameters_at_limit=False has_posdef_covar=True"
        " has_reached_call_limit=False has_valid_parameters=True"
        " hesse_failed=False is_above_max_edm=False is_valid=True"
        f" nfcn=1 ngrad=2 reduced_chi2={reduced_chi2} time=1.2>"
    )


def test_normalize_limit():
    assert util._normalize_limit(None) == (-np.inf, np.inf)
    assert util._normalize_limit((None, 2)) == (-np.inf, 2)
    assert util._normalize_limit((2, None)) == (2, np.inf)
    assert util._normalize_limit((None, None)) == (-np.inf, np.inf)
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


def test_make_func_code():
    with pytest.warns(np.VisibleDeprecationWarning):
        fc = util.make_func_code(["a", "b"])
    assert fc.co_varnames == ("a", "b")
    assert fc.co_argcount == 2

    with pytest.warns(np.VisibleDeprecationWarning):
        fc = util.make_func_code(("x",))
    assert fc.co_varnames == ("x",)
    assert fc.co_argcount == 1


def test_make_with_signature():
    def f(a, b):
        return a + b

    f1 = util.make_with_signature(f, "x", "y")
    assert util.describe(f1) == ["x", "y"]
    assert f1(1, 2) == f(1, 2)
    f2 = util.make_with_signature(f, b="z")
    assert util.describe(f2) == ["a", "z"]
    assert f2(1, 2) == f(1, 2)
    assert f1 is not f2
    f3 = util.make_with_signature(f, "x", b="z")
    assert util.describe(f3) == ["x", "z"]
    assert f3(1, 2) == f(1, 2)

    # check that arguments are not overridden
    assert util.describe(f1) == ["x", "y"]
    assert util.describe(f) == ["a", "b"]

    with pytest.raises(ValueError):
        util.make_with_signature(f, "a", "b", "c")

    with pytest.raises(ValueError):
        util.make_with_signature(f, "a", "b", "c", b="z")


def test_make_with_signature_on_func_without_code_object():
    class Fcn:
        def __call__(self, x, y):
            return x + y

    f = Fcn()
    assert not hasattr(f, "__code__")

    f1 = util.make_with_signature(f, "x", "y")
    assert util.describe(f1) == ["x", "y"]
    assert f1(1, 2) == f(1, 2)
    assert f1 is not f

    f2 = util.make_with_signature(f1, x="a")
    assert util.describe(f2) == ["a", "y"]
    assert f2(1, 2) == f(1, 2)


def test_merge_signatures():
    def f(x, y, z):
        return x + y + z

    def g(x, a, b):
        return x + a + b

    args, (pf, pg) = util.merge_signatures([f, g])
    assert args == ["x", "y", "z", "a", "b"]
    assert pf == [0, 1, 2]
    assert pg == [0, 3, 4]


@pytest.mark.skipif(not scipy_available, reason="needs scipy")
def test_propagate_1():
    cov = [
        [1.0, 0.1, 0.2],
        [0.1, 2.0, 0.3],
        [0.2, 0.3, 3.0],
    ]
    x = [1, 2, 3]

    def fn(x):
        return 2 * x + 1

    with pytest.warns(np.VisibleDeprecationWarning):
        y, ycov = util.propagate(fn, x, cov)
    np.testing.assert_allclose(y, [3, 5, 7])
    np.testing.assert_allclose(
        ycov, [[4, 0.4, 0.8], [0.4, 8, 1.2], [0.8, 1.2, 12]], rtol=1e-3
    )

    with pytest.warns(np.VisibleDeprecationWarning):
        y, ycov = util.propagate(fn, [1], [[2]])
    np.testing.assert_allclose(y, 3)
    np.testing.assert_allclose(ycov, 8, rtol=1e-3)


@pytest.mark.skipif(not scipy_available, reason="needs scipy")
def test_propagate_2():
    cov = [
        [1.0, 0.1, 0.2],
        [0.1, 2.0, 0.3],
        [0.2, 0.3, 3.0],
    ]
    x = [1.0, 2.0, 3.0]

    a = 0.5 * np.arange(30).reshape((10, 3))

    def fn(x):
        return np.dot(a, x)

    with pytest.warns(np.VisibleDeprecationWarning):
        y, ycov = util.propagate(fn, x, cov)
    np.testing.assert_equal(y, fn(x))
    np.testing.assert_allclose(ycov, np.einsum("ij,kl,jl", a, a, cov), rtol=1e-3)

    def fn(x):
        return np.linalg.multi_dot([x.T, cov, x])

    with pytest.warns(np.VisibleDeprecationWarning):
        y, ycov = util.propagate(fn, x, cov)
    np.testing.assert_equal(y, fn(np.array(x)))
    jac = 2 * np.dot(cov, x)
    np.testing.assert_allclose(ycov, np.einsum("i,k,ik", jac, jac, cov), rtol=1e-3)


@pytest.mark.skipif(not scipy_available, reason="needs scipy")
def test_propagate_3():
    # matrices with full zero rows and columns are supported
    cov = [
        [1.0, 0.0, 0.2],
        [0.0, 0.0, 0.0],
        [0.2, 0.0, 3.0],
    ]
    x = [1.0, 2.0, 3.0]

    def fn(x):
        return 2 * x + 1

    with pytest.warns(np.VisibleDeprecationWarning):
        y, ycov = util.propagate(fn, x, cov)
    np.testing.assert_allclose(y, [3, 5, 7])
    np.testing.assert_allclose(ycov, [[4, 0.0, 0.8], [0.0, 0.0, 0.0], [0.8, 0.0, 12]])


@pytest.mark.skipif(not scipy_available, reason="needs scipy")
def test_propagate_on_bad_input():
    cov = [[np.nan, 0.0], [0.0, 1.0]]
    x = [1.0, 2.0]

    def fn(x):
        return 2 * x + 1

    with pytest.warns(np.VisibleDeprecationWarning):
        with pytest.raises(ValueError):
            util.propagate(fn, x, cov)

    with pytest.warns(np.VisibleDeprecationWarning):
        with pytest.raises(ValueError):
            util.propagate(fn, x, 1)

    cov = [[1.0], [1.0]]
    with pytest.warns(np.VisibleDeprecationWarning):
        with pytest.raises(ValueError):
            util.propagate(fn, x, cov)


def test_iterate():
    assert list(util._iterate(1)) == [1]
    assert list(util._iterate([1, 2])) == [1, 2]


def test_replace_none():
    assert util._replace_none(None, 2) == 2
    assert util._replace_none(3, 2) == 3


def test_progressbar(capsys):
    with util.ProgressBar(max_value=4) as bar:
        for i in range(4):
            bar += 1
    stdout, stderr = capsys.readouterr()
    assert stdout == "\r0 %\r25 %\r50 %\r75 %\r100 %\r     "


def test_progressbar_no_ipykernel(capsys):
    with hide_modules("ipykernel"):
        with util.ProgressBar(max_value=4) as bar:
            for i in range(4):
                bar += 1
        stdout, stderr = capsys.readouterr()
        assert stdout == "\r0 %\r25 %\r50 %\r75 %\r100 %\r     "


def test_progressbar_html(capsys):
    import sys

    m_iostream = pytest.importorskip("ipykernel.iostream")
    m_display = pytest.importorskip("IPython.display")

    class OutStream:
        def write(self, s):
            original_stdout.write(s)

        def flush(self):
            original_stdout.flush()

    def display(msg, **kwargs):
        sys.stdout.write(msg._repr_html_())

    original_stdout = sys.stdout
    # make ProgressBar think it is running in Jupyter
    sys.stdout = OutStream()

    # monkey-patching our mockups
    m_iostream.OutStream = OutStream
    m_display.display = display

    with util.ProgressBar(max_value=4) as bar:
        for i in range(4):
            bar += 1

    sys.stdout = original_stdout
    stdout, stderr = capsys.readouterr()

    assert stdout == (
        "<progress value='0' max='100'></progress> 0 %"
        "<progress value='25' max='100'></progress> 25 %"
        "<progress value='50' max='100'></progress> 50 %"
        "<progress value='75' max='100'></progress> 75 %"
        "<progress value='100' max='100'></progress> 100 %"
    )


@pytest.mark.parametrize(
    "mask_expected",
    (
        (None, [[0, 1, 2]]),
        ([True, False], [[0]]),
        ([False, True], [[1]]),
        ([True, True, True, False, True, True, False, True], [[0, 1, 2], [4, 5], [7]]),
    ),
)
def test_histogram_segments(mask_expected):
    mask, expected = mask_expected
    if mask is None:
        xe = np.linspace(0, 1, 4)
        masked = np.arange(3)
    else:
        xe = np.linspace(0, 1, len(mask))
        masked = np.arange(len(mask))[np.array(mask)]
    segments = util._histogram_segments(mask, xe, masked)
    assert_equal([s[0] for s in segments], expected)


@pytest.mark.parametrize(
    "fn_expected", ((lambda x: x, 15), (lambda x: x**11, 40), (np.log, 80))
)
def test_smart_sampling_1(fn_expected):
    fn, expected = fn_expected
    x, y = util._smart_sampling(fn, 1e-10, 5)
    assert len(y) < expected


def test_smart_sampling_2():
    # should not raise a warning
    x, y = util._smart_sampling(np.log, 1e-10, 1, tol=1e-5)
    assert 0 < len(x) < 1000


def test_smart_sampling_3():
    def step(x):
        return np.where(x > 0.5, 0, 1)

    with pytest.warns(RuntimeWarning, match="Iteration limit"):
        x, y = util._smart_sampling(step, 0, 1, tol=0)
    assert 0 < len(x) < 80


def test_smart_sampling_4():
    from time import sleep

    def step(x):
        sleep(0.1)
        return np.where(x > 0.5, 0, 1)

    with pytest.warns(RuntimeWarning, match="Time limit"):
        x, y = util._smart_sampling(step, 0, 1, maxtime=0)
    assert 0 < len(x) < 10


@pytest.mark.parametrize(
    "x,expected",
    [([1, 2, 3], False), ([-1, 1000, 100000], False), ([1, 10, 100], True)],
)
def test_detect_log_spacing_1(x, expected):
    assert util._detect_log_spacing(x) == expected


def test_optional_module_for_1():
    with optional_module_for("foo"):
        import iminuit  # noqa


def test_optional_module_for_2():
    from iminuit.warnings import OptionalDependencyWarning

    with pytest.warns(
        OptionalDependencyWarning, match="foo requires optional package 'foobarbaz'"
    ):
        with optional_module_for("foo"):
            import foobarbaz  # noqa


def test_optional_module_for_3():
    from iminuit.warnings import OptionalDependencyWarning

    with pytest.warns(
        OptionalDependencyWarning, match="foo requires optional package 'foo'"
    ):
        with optional_module_for("foo", replace={"foobarbaz": "foo"}):
            import foobarbaz  # noqa


def test_positive_definite():
    assert util.is_positive_definite([[1, 0], [0, 1]])
    assert not util.is_positive_definite([[1, 1], [1, 1]])
    assert not util.is_positive_definite([[1, 0], [1, 1]])
