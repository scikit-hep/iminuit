import pytest
from pytest import approx
from iminuit import Minuit
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
