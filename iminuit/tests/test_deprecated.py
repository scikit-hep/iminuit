from __future__ import absolute_import, division, print_function
import pytest
from pytest import approx
from iminuit import Minuit


class Fcn:
    def __call__(self, x):
        return x ** 2

    def default_errordef(self):
        return 4


def test_deprecated(capsys):
    with pytest.warns(DeprecationWarning):
        m = Minuit(Fcn(), pedantic=False)
    m.migrad()
    assert m.errors["x"] == 2

    with pytest.warns(DeprecationWarning):
        assert m.edm == approx(0)

    with pytest.warns(DeprecationWarning):
        assert m.get_fmin() == m.fmin

    with pytest.warns(DeprecationWarning):
        assert m.get_param_states() == m.params

    with pytest.warns(DeprecationWarning):
        assert m.get_initial_param_states() == m.init_params

    with pytest.warns(DeprecationWarning):
        assert m.get_num_call_fcn() == m.ncalls_total

    with pytest.warns(DeprecationWarning):
        assert m.get_num_call_grad() == m.ngrads_total

    with pytest.warns(DeprecationWarning):
        assert m.is_fixed("x") is False

    m.fixed["x"] = True
    with pytest.warns(DeprecationWarning):
        assert m.is_fixed("x") is True
    m.fixed["x"] = False

    with pytest.warns(DeprecationWarning):
        assert m.migrad_ok() == m.valid

    m.hesse()
    with pytest.warns(DeprecationWarning):
        m.print_matrix()

    with pytest.warns(DeprecationWarning):
        m.print_fmin()

    with pytest.warns(DeprecationWarning):
        m.print_all_minos()

    with pytest.warns(DeprecationWarning):
        m.print_param()

    with pytest.warns(DeprecationWarning):
        m.print_initial_param()

    m.errordef = 4
    assert m.errordef == 4
    with pytest.warns(DeprecationWarning):
        m.set_errordef(1)
    assert m.errordef == 1

    with pytest.warns(DeprecationWarning):
        m.set_up(4)
    assert m.errordef == 4

    assert m.strategy == 1
    with pytest.warns(DeprecationWarning):
        m.set_strategy(2)
    assert m.strategy == 2

    m.print_level = 0
    with pytest.warns(DeprecationWarning):
        m.set_print_level(2)
    assert m.print_level == 2

    with pytest.warns(DeprecationWarning):
        m.hesse(maxcall=10)

    with pytest.warns(DeprecationWarning):
        import iminuit.iminuit_warnings

    with pytest.warns(DeprecationWarning):
        from iminuit.iminuit_warnings import IMinuitWarning
