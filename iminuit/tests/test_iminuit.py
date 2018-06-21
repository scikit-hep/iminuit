from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings
from math import sqrt
import pytest
from iminuit.tests.utils import assert_allclose
from iminuit import Minuit
import numpy as np


class Func_Code:
    def __init__(self, varname):
        self.co_varnames = varname
        self.co_argcount = len(varname)


class Func1:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return (x - 2.) ** 2 + (y - 5.) ** 2 + 10


class Func2:
    def __init__(self):
        self.func_code = Func_Code(['x', 'y'])

    def __call__(self, *arg):
        return (arg[0] - 2.) ** 2 + (arg[1] - 5.) ** 2 + 10


def func3(x, y):
    return 0.2 * (x - 2.) ** 2 + (y - 5.) ** 2 + 10


def func3_grad(x, y):
    dfdx = 0.4 * (x - 2.)
    dfdy = 2 * (y - 5.)
    return [dfdx, dfdy]


def func4(x, y, z):
    return 0.2 * (x - 2.) ** 2 + 0.1 * (y - 5.) ** 2 + 0.25 * (z - 7.) ** 2 + 10


def func4_grad(x, y, z):
    dfdx = 0.4 * (x - 2.)
    dfdy = 0.2 * (y - 5.)
    dfdz = 0.5 * (z - 7.)
    return (dfdx, dfdy, dfdz)


def func5(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    return (x ** 2) + (z ** 2) + long_variable_name_really_long_why_does_it_has_to_be_this_long ** 2


def func5_grad(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    dfdx = 2 * x
    dfdy = 2 * long_variable_name_really_long_why_does_it_has_to_be_this_long
    dfdz = 2 * z
    return (dfdx, dfdy, dfdz)


def func6(x, m, s, A):
    return A / ((x - m) ** 2 + s ** 2)


def func7(*args): # no signature
    return (args[0] - 1) ** 2 + (args[1] - 2) ** 2


def func8(x): # test numpy support
    return np.sum((x - 1) ** 2)


def func8_grad(x): # test numpy support
    return 2 * (x - 1)


data_y = [0.552, 0.735, 0.846, 0.875, 1.059, 1.675, 1.622, 2.928,
          3.372, 2.377, 4.307, 2.784, 3.328, 2.143, 1.402, 1.44,
          1.313, 1.682, 0.886, 0.0, 0.266, 0.3]
data_x = list(range(len(data_y)))


def chi2(m, s, A):
    """Chi2 fitting routine"""
    return sum(((func6(x, m, s, A) - y) ** 2 for x, y in zip(data_x, data_y)))


def functesthelper(f, **kwds):
    m = Minuit(f, print_level=0, pedantic=False, **kwds)
    m.migrad()
    val = m.values
    assert_allclose(val['x'], 2., rtol=1e-3)
    assert_allclose(val['y'], 5., rtol=1e-3)
    assert_allclose(m.fval, 10., rtol=1e-3)
    assert m.matrix_accurate()
    assert m.migrad_ok()
    return m


def test_f1():
    functesthelper(Func1())


def test_f2():
    functesthelper(Func2())


def test_f3():
    functesthelper(func3)
    m = functesthelper(func3, grad_fcn=func3_grad)
    assert m.get_num_call_grad() > 0


def test_lambda():
    functesthelper(lambda x, y: (x - 2.) ** 2 + (y - 5.) ** 2 + 10)


def test_nosignature():
    with pytest.raises(TypeError):
        Minuit(func7)
    m = Minuit(func7, forced_parameters=('x', 'y'),
               pedantic=False, print_level=0)
    m.migrad()
    val = m.values
    assert_allclose((val['x'], val['y'], m.fval), (1, 2, 0), atol=1e-8)
    assert m.migrad_ok()


def test_array_call():
    inf = float("infinity")
    m = Minuit(func8, a=1, b=1,
               error_a=1, error_b=1,
               limit_a=(0, inf),
               limit_b=(0, inf),
               fix_a=False,
               fix_b=False,
               print_level=0,
               errordef=1,
               use_array_call=True,
               forced_parameters=("a", "b"))
    m.migrad()
    v = m.values
    assert_allclose((v["a"], v["b"]),
                    (1, 1))
    m.hesse()
    c = m.covariance
    assert_allclose((c[("a", "a")],
                     c[("b", "b")]),
                    (1, 1))


def test_from_array_func():
    m = Minuit.from_array_func(func8, (1, 1),
                               grad_fcn=func8_grad,
                               error=(0.5, 0.5),
                               limit=((0, 2), (0, 2)),
                               name=("a", "b"),
                               errordef=1,
                               print_level=0)
    assert m.fitarg == {"a": 1,
                        "b": 1,
                        "error_a": 0.5,
                        "error_b": 0.5,
                        "fix_a": False,
                        "fix_b": False,
                        "limit_a": (0, 2),
                        "limit_b": (0, 2)}
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1), rtol=1e-3)
    c = m.np_covariance()
    assert_allclose(np.diag(c), (1, 1), rtol=1e-3)


def test_from_array_func_with_broadcasting():
    m = Minuit.from_array_func(func8, (1, 1),
                               error=0.5,
                               limit=(0, 2),
                               errordef=1,
                               print_level=0)
    assert m.fitarg == {"x0": 1,
                        "x1": 1,
                        "error_x0": 0.5,
                        "error_x1": 0.5,
                        "fix_x0": False,
                        "fix_x1": False,
                        "limit_x0": (0, 2),
                        "limit_x1": (0, 2)}
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1))
    c = m.np_covariance()
    assert_allclose(np.diag(c), (1, 1))


def test_no_resume():
    m = Minuit(func3, pedantic=False)
    m.migrad()
    n = m.get_num_call_fcn()
    m.migrad()
    assert m.get_num_call_fcn() > n
    m.migrad(resume=False)
    assert m.get_num_call_fcn() == n

    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False)
    m.migrad()
    n = m.get_num_call_fcn()
    k = m.get_num_call_grad()
    m.migrad()
    assert m.get_num_call_fcn() > n
    assert m.get_num_call_grad() > k
    m.migrad(resume=False)
    assert m.get_num_call_fcn() == n
    assert m.get_num_call_grad() == k


def test_typo():
    with pytest.raises(RuntimeError):
        Minuit(func4, printlevel=0)
        # self.assertRaises(RuntimeError,Minuit,func4,printlevel=0)


def test_non_invertible():
    # making sure it doesn't crash
    def f(x, y):
        return (x * y) ** 2

    m = Minuit(f, pedantic=False, print_level=0)
    m.migrad()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            m.hesse()
            raise RuntimeError('Hesse did not raise a warning')
        except Warning:
            pass
    try:
        m.matrix()
        raise RuntimeError('Matrix did not raise an error')  # shouldn't reach here
    except RuntimeError:
        pass


def test_fix_param():
    m = Minuit(func4, print_level=0, pedantic=False)
    m.migrad()
    m.minos()
    val = m.values
    assert_allclose(val['x'], 2)
    assert_allclose(val['y'], 5)
    assert_allclose(val['z'], 7)
    err = m.errors  # second derivative
    assert_allclose(err['x'], 2.236067977354171)
    assert_allclose(err['y'], 3.1622776602288)
    assert_allclose(err['z'], 2)
    m.print_all_minos()
    # now fix z = 10
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, pedantic=False)
    m.migrad()
    val = m.values
    assert_allclose(val['x'], 2)
    assert_allclose(val['y'], 10)
    assert_allclose(val['z'], 7)
    assert_allclose(m.fval, 10. + 2.5)
    free_param = m.list_of_vary_param()
    fix_param = m.list_of_fixed_param()
    assert 'x' in free_param
    assert 'x' not in fix_param
    assert 'y' in fix_param
    assert 'y' not in free_param
    assert 'z' not in fix_param


def test_fix_param_with_gradient():
    m = Minuit(func4, grad_fcn=func4_grad, print_level=0, pedantic=False)
    m.migrad()
    m.minos()
    val = m.values
    assert_allclose(val['x'], 2)
    assert_allclose(val['y'], 5)
    assert_allclose(val['z'], 7)
    err = m.errors  # second derivative
    assert_allclose(err['x'], 2.236067977354171)
    assert_allclose(err['y'], 3.1622776602288)
    assert_allclose(err['z'], 2)
    m.print_all_minos()
    # now fix z = 10
    m = Minuit(func4, grad_fcn=func4_grad, print_level=-1, y=10., fix_y=True, pedantic=False)
    m.migrad()
    val = m.values
    assert_allclose(val['x'], 2)
    assert_allclose(val['y'], 10)
    assert_allclose(val['z'], 7)
    assert_allclose(m.fval, 10. + 2.5)
    free_param = m.list_of_vary_param()
    fix_param = m.list_of_fixed_param()
    assert 'x' in free_param
    assert 'x' not in fix_param
    assert 'y' in fix_param
    assert 'y' not in free_param
    assert 'z' not in fix_param


def test_fitarg_oneside():
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, limit_x=(None, 20.),
               pedantic=False)
    fitarg = m.fitarg
    assert_allclose(fitarg['y'], 10.)
    assert fitarg['fix_y']
    assert fitarg['limit_x'] == (None, 20)
    m.migrad()

    fitarg = m.fitarg

    assert_allclose(fitarg['y'], 10.)
    assert_allclose(fitarg['x'], 2., atol=1e-2)
    assert_allclose(fitarg['z'], 7., atol=1e-2)

    assert 'error_y' in fitarg
    assert 'error_x' in fitarg
    assert 'error_z' in fitarg

    assert fitarg['fix_y']
    assert fitarg['limit_x'] == (None, 20)


def test_fitarg():
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, limit_x=(0, 20.),
               pedantic=False)
    fitarg = m.fitarg
    assert_allclose(fitarg['y'], 10.)
    assert fitarg['fix_y'] is True
    assert fitarg['limit_x'] == (0, 20)
    m.migrad()

    fitarg = m.fitarg

    assert_allclose(fitarg['y'], 10.)
    assert_allclose(fitarg['x'], 2., atol=1e-2)
    assert_allclose(fitarg['z'], 7., atol=1e-2)

    assert 'error_y' in fitarg
    assert 'error_x' in fitarg
    assert 'error_z' in fitarg

    assert fitarg['fix_y'] is True
    assert fitarg['limit_x'] == (0, 20)


def test_minos_all():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    for sigma in range(1, 4):
        m.minos(sigma=sigma)
        assert_allclose(m.merrors[('x', -1.0)], -sigma*sqrt(5))
        assert_allclose(m.merrors[('x', 1.0)], sigma*sqrt(5))
        assert_allclose(m.merrors[('y', 1.0)], sigma*1.)


def test_minos_all_with_gradient():
    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False, print_level=0)
    m.set_strategy(2)
    m.migrad()
    for sigma in range(1, 4):
        m.minos(sigma=sigma)
        assert_allclose(m.merrors[('x', -1.0)], -sigma*sqrt(5))
        assert_allclose(m.merrors[('x', 1.0)], sigma*sqrt(5))
        assert_allclose(m.merrors[('y', 1.0)], sigma*1.)


def test_minos_single():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    m.minos('x')
    assert_allclose(m.merrors[('x', -1.0)], -sqrt(5))
    assert_allclose(m.merrors[('x', 1.0)], sqrt(5))


def test_minos_single_with_gradient():
    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False, print_level=0)
    m.set_strategy(2)
    m.migrad()
    m.minos('x')
    assert_allclose(m.merrors[('x', -1.0)], -sqrt(5))
    assert_allclose(m.merrors[('x', 1.0)], sqrt(5))


def test_minos_single_fixed_raising():
    m = Minuit(func3, pedantic=False, print_level=0, fix_x=True)
    m.migrad()

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(RuntimeWarning):
            m.minos('x')


def test_minos_single_fixed_result():
    m = Minuit(func3, pedantic=False, print_level=0, fix_x=True)
    m.migrad()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ret = m.minos('x')
    assert ret is None


def test_minos_single_no_migrad():
    m = Minuit(func3, pedantic=False, print_level=0)
    with pytest.raises(RuntimeError):
        m.minos('x')


def test_minos_single_nonsense_variable():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    with pytest.raises(RuntimeError):
        m.minos('nonsense')


def test_fixing_long_variable_name():
    m = Minuit(func5, pedantic=False, print_level=0,
               fix_long_variable_name_really_long_why_does_it_has_to_be_this_long=True,
               long_variable_name_really_long_why_does_it_has_to_be_this_long=0)
    m.migrad()


def test_fixing_long_variable_name_with_gradient():
    m = Minuit(func5, grad_fcn=func5_grad, pedantic=False, print_level=0,
               fix_long_variable_name_really_long_why_does_it_has_to_be_this_long=True,
               long_variable_name_really_long_why_does_it_has_to_be_this_long=0)
    m.migrad()


def test_initial_value():
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    assert_allclose(m.args[0], 1.)
    assert_allclose(m.args[1], 2.)
    assert_allclose(m.values['x'], 1.)
    assert_allclose(m.values['y'], 2.)
    assert_allclose(m.errors['x'], 3.)


def test_mncontour():
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour('x', 'y', numpoints=30)
    xminos_t = m.minos('x')['x']
    yminos_t = m.minos('y')['y']
    assert_allclose(xminos.upper, xminos_t.upper)
    assert_allclose(xminos.lower, xminos_t.lower)
    assert_allclose(yminos.upper, yminos_t.upper)
    assert_allclose(yminos.lower, yminos_t.lower)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2


def test_mncontour_with_gradient():
    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour('x', 'y', numpoints=30)
    xminos_t = m.minos('x')['x']
    yminos_t = m.minos('y')['y']
    assert_allclose(xminos.upper, xminos_t.upper)
    assert_allclose(xminos.lower, xminos_t.lower)
    assert_allclose(yminos.upper, yminos_t.upper)
    assert_allclose(yminos.lower, yminos_t.lower)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2


def test_mncontour_sigma():
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour('x', 'y', numpoints=30, sigma=2.0)
    xminos_t = m.minos('x', sigma=2.0)['x']
    yminos_t = m.minos('y', sigma=2.0)['y']
    assert_allclose(xminos.upper, xminos_t.upper)
    assert_allclose(xminos.lower, xminos_t.lower)
    assert_allclose(yminos.upper, yminos_t.upper)
    assert_allclose(yminos.lower, yminos_t.lower)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2


def test_contour():
    # FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.contour('x', 'y')


def test_contour_with_gradient():
    # FIXME: check the result
    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.contour('x', 'y')


def test_profile():
    # FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.profile('y')


def test_profile_with_gradient():
    # FIXME: check the result
    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.profile('y')


def test_mnprofile():
    # FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.mnprofile('y')


def test_mnprofile_with_gradient():
    # FIXME: check the result
    m = Minuit(func3, grad_fcn=func3_grad, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.mnprofile('y')


def test_printfmin_uninitialized():
    # issue 85
    def f(x):
        return 2 + 3 * x

    fitter = Minuit(f, pedantic=False)
    with pytest.raises(RuntimeError):
        fitter.print_fmin()


def test_reverse_limit():
    # issue 94
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(f, limit_x=(3., 2.), pedantic=False, print_level=0)
    with pytest.raises(ValueError):
        m.migrad()


class TestOutputInterface:
    def setup(self):
        self.m = Minuit(func3, print_level=0, pedantic=False)
        self.m.migrad()
        self.m.hesse()
        self.m.minos()

    def test_args(self):
        actual = self.m.args
        expected = [2., 5.]
        assert_allclose(actual, expected, atol=1e-8)

    def test_matrix(self):
        actual = self.m.matrix()
        expected = [[5., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)

    def test_matrix_correlation(self):
        actual = self.m.matrix(correlation=True)
        expected = [[1., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)

    def test_np_matrix(self):
        actual = self.m.np_matrix()
        expected = [[5., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 2)

    def test_np_matrix_correlation(self):
        actual = self.m.np_matrix(correlation=True)
        expected = [[1., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 2)

    def test_np_values(self):
        actual = self.m.np_values()
        expected = [2., 5.]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2,)

    def test_np_errors(self):
        actual = self.m.np_errors()
        expected = [5.**0.5, 1.]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2,)

    def test_np_merrors(self):
        actual = self.m.np_merrors()
        expected = [[5.**0.5, 1.], [5.**0.5, 1.]]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 2)

    def test_np_covariance(self):
        actual = self.m.np_covariance()
        expected = [[5., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 2)


def test_chi2_fit():
    """Fit a curve to data."""
    m = Minuit(chi2, s=2., error_A=0.1, errordef=0.01,
               print_level=0, pedantic=False)
    m.migrad()
    output = [round(10 * m.values['A']), round(100 * m.values['s']),
              round(100 * m.values['m'])]
    expected = [round(10 * 64.375993), round(100 * 4.267970),
                round(100 * 9.839172)]
    assert_allclose(output, expected)


def test_oneside():
    m_limit = Minuit(func3, limit_x=(None, 9), pedantic=False, print_level=0)
    m_nolimit = Minuit(func3, pedantic=False, print_level=0)
    # Solution: x=2., y=5.
    m_limit.tol = 1e-4
    m_nolimit.tol = 1e-4
    m_limit.migrad()
    m_nolimit.migrad()
    assert_allclose(list(m_limit.values.values()),
                    list(m_nolimit.values.values()), atol=1e-4)


def test_oneside_outside():
    m = Minuit(func3, limit_x=(None, 1), pedantic=False, print_level=0)
    m.migrad()
    assert_allclose(m.values['x'], 1)


def test_num_call():
    class Func:
        ncall = 0
        def __call__(self, x):
            self.ncall += 1
            return x ** 2

    # check that counting is accurate
    func = Func()
    m = Minuit(func, pedantic=False, print_level=0)
    m.migrad()
    assert m.get_num_call_fcn() == func.ncall
    m.migrad()
    assert m.get_num_call_fcn() == func.ncall
    func.ncall = 0
    m.migrad(resume=False)
    assert func.ncall ==  m.get_num_call_fcn()

    ncall_without_limit = m.get_num_call_fcn()
    # check that ncall argument limits function calls in migrad
    # note1: Minuit only checks the ncall counter in units of one iteration
    # step, therefore the call counter is in general not equal to ncall.
    # note2: If you pass ncall=0, Minuit uses a heuristic value that depends
    # on the number of parameters.
    m.migrad(ncall=1, resume=False)
    assert m.get_num_call_fcn() < ncall_without_limit


def test_set_error_def():
    m = Minuit(lambda x: x**2, pedantic=False, print_level=0, errordef=1)
    m.migrad()
    m.hesse()
    assert_allclose(m.errors["x"], 1)
    m.set_errordef(4)
    m.hesse()
    assert_allclose(m.errors["x"], 2)
