import pytest
from pytest import approx
from iminuit import Minuit, util
from sys import version_info as pyver


class Fcn:
    def __call__(self, x):
        return x ** 2

    def default_errordef(self):
        return 4


@pytest.fixture
def minuit():
    m = Minuit(lambda x: x ** 2, pedantic=False)
    m.migrad()
    return m


def test_forced_parameters():
    with pytest.warns(DeprecationWarning):
        Minuit(lambda x: 0, forced_parameters="x", pedantic=False)


def test_minos_merrors(minuit):
    minuit.minos()
    m = minuit.merrors
    with pytest.warns(DeprecationWarning):
        assert m[("x", -1)] == approx(-1)
    with pytest.warns(DeprecationWarning):
        assert m[("x", 1)] == approx(1)
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError):
            m[("x", 2)]


def test_default_errordef():
    with pytest.warns(DeprecationWarning):
        Minuit(Fcn(), pedantic=False)


def test_edm(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.edm == approx(0)


def test_matrix_accurate(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.matrix_accurate() == minuit.accurate


def test_get_fmin(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.get_fmin() == minuit.fmin


def test_get_param_states(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.get_param_states() == minuit.params


def test_get_initial_param_states(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.get_initial_param_states() == minuit.init_params


def test_get_num_call_fcn(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.get_num_call_fcn() == minuit.ncalls_total


def test_num_call_grad(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.get_num_call_grad() == minuit.ngrads_total


def test_is_fixed(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.is_fixed("x") is False

    minuit.fixed["x"] = True
    with pytest.warns(DeprecationWarning):
        assert minuit.is_fixed("x") is True
    minuit.fixed["x"] = False


def test_migrad_ok(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.migrad_ok() == minuit.valid


def test_print_matrix(minuit):
    minuit.hesse()
    with pytest.warns(DeprecationWarning):
        minuit.print_matrix()


def test_print_fmin(minuit):
    with pytest.warns(DeprecationWarning):
        minuit.print_fmin()


def test_print_all_minos(minuit):
    with pytest.warns(DeprecationWarning):
        minuit.print_all_minos()


def test_print_param(minuit):
    with pytest.warns(DeprecationWarning):
        minuit.print_param()


def test_print_initial_param(minuit):
    with pytest.warns(DeprecationWarning):
        minuit.print_initial_param()


def test_set_errordef(minuit):
    assert minuit.errordef == 1
    with pytest.warns(DeprecationWarning):
        minuit.set_errordef(4)
    assert minuit.errordef == 4


def test_set_up(minuit):
    assert minuit.errordef == 1
    with pytest.warns(DeprecationWarning):
        minuit.set_up(4)
    assert minuit.errordef == 4


def test_set_strategy(minuit):
    assert minuit.strategy == 1
    with pytest.warns(DeprecationWarning):
        minuit.set_strategy(2)
    assert minuit.strategy == 2


def test_set_print_level(minuit):
    minuit.print_level = 0
    with pytest.warns(DeprecationWarning):
        minuit.set_print_level(2)
    assert minuit.print_level == 2


def test_hesse_maxcall(minuit):
    with pytest.warns(DeprecationWarning):
        minuit.hesse(maxcall=10)


def test_list_of_vary_param(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.list_of_vary_param() == [
            k for (k, v) in minuit.fixed.items() if not v
        ]


def test_list_of_fixed_param(minuit):
    with pytest.warns(DeprecationWarning):
        assert minuit.list_of_fixed_param() == [
            k for (k, v) in minuit.fixed.items() if v
        ]


def test_import_iminuit_warnings():
    with pytest.warns(DeprecationWarning):
        import iminuit.iminuit_warnings  # noqa: F401


@pytest.mark.skipif(
    pyver < (3, 7),
    reason="Deprecating module-level objects requires python-3.7 or newer",
)
def test_import_from_iminuit_warnings():
    with pytest.warns(DeprecationWarning):
        from iminuit.iminuit_warnings import IMinuitWarning  # noqa: F401


def test_true_param():
    with pytest.warns(DeprecationWarning):
        assert util.true_param("N") is True
        assert util.true_param("limit_N") is False
        assert util.true_param("error_N") is False
        assert util.true_param("fix_N") is False


def test_param_name():
    with pytest.warns(DeprecationWarning):
        assert util.param_name("N") == "N"
        assert util.param_name("limit_N") == "N"
        assert util.param_name("error_N") == "N"
        assert util.param_name("fix_N") == "N"


def test_extract_iv():
    d = dict(k=1.0, limit_k=1.0, error_k=1.0, fix_k=1.0)
    with pytest.warns(DeprecationWarning):
        ret = util.extract_iv(d)
    assert "k" in ret
    assert "limit_k" not in ret
    assert "error_k" not in ret
    assert "fix_k" not in ret


def test_extract_limit():
    d = dict(k=1.0, limit_k=1.0, error_k=1.0, fix_k=1.0)
    with pytest.warns(DeprecationWarning):
        ret = util.extract_limit(d)
    assert "k" not in ret
    assert "limit_k" in ret
    assert "error_k" not in ret
    assert "fix_k" not in ret


def test_extract_error():
    d = dict(k=1.0, limit_k=1.0, error_k=1.0, fix_k=1.0)
    with pytest.warns(DeprecationWarning):
        ret = util.extract_error(d)
    assert "k" not in ret
    assert "limit_k" not in ret
    assert "error_k" in ret
    assert "fix_k" not in ret


def test_extract_fix():
    d = dict(k=1.0, limit_k=1.0, error_k=1.0, fix_k=1.0)
    with pytest.warns(DeprecationWarning):
        ret = util.extract_fix(d)
    assert "k" not in ret
    assert "limit_k" not in ret
    assert "error_k" not in ret
    assert "fix_k" in ret


def test_remove_var():
    dk = dict(k=1, limit_k=1, error_k=1, fix_k=1)
    dm = dict(m=1, limit_m=1, error_m=1, fix_m=1)
    dn = dict(n=1, limit_n=1, error_n=1, fix_n=1)
    d = {}
    d.update(dk)
    d.update(dm)
    d.update(dn)

    with pytest.warns(DeprecationWarning):
        d = util.remove_var(d, ["k", "m"])
    assert set(d.keys()) & set(dk.keys()) == set()
    assert set(d.keys()) & set(dm.keys()) == set()
