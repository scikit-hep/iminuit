from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings
from math import sqrt
from unittest import TestCase
from nose.tools import (raises,
                        assert_equal,
                        assert_true,
                        assert_false,
                        assert_almost_equal,
                        )
from iminuit import Minuit

try:
    from numpy.testing import assert_array_almost_equal
except ImportError:
    def assert_array_almost_equal(actual, expected, decimal=6):
        """
        Helper function to test if all elements of a list of lists
        are almost equal.
        A replacement for numpy.testing.assert_array_almost_equal,
        if it is not installed
        """
        for row in range(len(actual)):
            for col in range(len(actual[0])):
                assert_almost_equal(actual[row][col], expected[row][col], places=decimal)


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


def func4(x, y, z):
    return 0.2 * (x - 2.) ** 2 + 0.1 * (y - 5.) ** 2 + 0.25 * (z - 7.) ** 2 + 10


def func5(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    return (x ** 2) + (z ** 2) + long_variable_name_really_long_why_does_it_has_to_be_this_long ** 2


def func6(x, m, s, A):
    return A / ((x - m) ** 2 + s ** 2)


data_y = [0.552, 0.735, 0.846, 0.875, 1.059, 1.675, 1.622, 2.928,
          3.372, 2.377, 4.307, 2.784, 3.328, 2.143, 1.402, 1.44,
          1.313, 1.682, 0.886, 0.0, 0.266, 0.3]
data_x = list(range(len(data_y)))


def chi2(m, s, A):
    """Chi2 fitting routine"""
    return sum(((func6(x, m, s, A) - y) ** 2 for x, y in zip(data_x, data_y)))


def functesthelper(f):
    m = Minuit(f, print_level=0, pedantic=False)
    m.migrad()
    val = m.values
    assert_almost_equal(val['x'], 2.)
    assert_almost_equal(val['y'], 5.)
    assert_almost_equal(m.fval, 10.)
    assert m.matrix_accurate()
    assert m.migrad_ok()
    return m


def test_f1():
    functesthelper(Func1())


def test_f2():
    functesthelper(Func2())


def test_f3():
    functesthelper(func3)


@raises(RuntimeError)
def test_typo():
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
    except RuntimeError as e:
        pass


def test_fix_param():
    m = Minuit(func4, print_level=0, pedantic=False)
    m.migrad()
    m.minos()
    val = m.values
    assert_almost_equal(val['x'], 2.)
    assert_almost_equal(val['y'], 5.)
    assert_almost_equal(val['z'], 7.)
    err = m.errors  # second derivative
    m.print_all_minos()
    # now fix z = 10
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, pedantic=False)
    m.migrad()
    val = m.values
    assert_almost_equal(val['x'], 2.)
    assert_almost_equal(val['y'], 10.)
    assert_almost_equal(val['z'], 7.)
    assert_almost_equal(m.fval, 10. + 2.5)
    free_param = m.list_of_vary_param()
    fix_param = m.list_of_fixed_param()
    assert_true('x' in free_param)
    assert_false('x' in fix_param)
    assert_true('y' in fix_param)
    assert_false('y' in free_param)
    assert_false('z' in fix_param)


def test_fitarg_oneside():
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, limit_x=(None, 20.),
               pedantic=False)
    fitarg = m.fitarg
    assert_almost_equal(fitarg['y'], 10.)
    assert_true(fitarg['fix_y'])
    assert_equal(fitarg['limit_x'], (None, 20))
    m.migrad()

    fitarg = m.fitarg

    assert_almost_equal(fitarg['y'], 10.)
    assert_almost_equal(fitarg['x'], 2., places=2)
    assert_almost_equal(fitarg['z'], 7., places=2)

    assert_true('error_y' in fitarg)
    assert_true('error_x' in fitarg)
    assert_true('error_z' in fitarg)

    assert_true(fitarg['fix_y'])
    assert_equal(fitarg['limit_x'], (None, 20))


def test_fitarg():
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, limit_x=(0, 20.),
               pedantic=False)
    fitarg = m.fitarg
    assert_almost_equal(fitarg['y'], 10.)
    assert_true(fitarg['fix_y'])
    assert_equal(fitarg['limit_x'], (0, 20))
    m.migrad()

    fitarg = m.fitarg

    assert_almost_equal(fitarg['y'], 10.)
    assert_almost_equal(fitarg['x'], 2., places=2)
    assert_almost_equal(fitarg['z'], 7., places=2)

    assert_true('error_y' in fitarg)
    assert_true('error_x' in fitarg)
    assert_true('error_z' in fitarg)

    assert_true(fitarg['fix_y'])
    assert_equal(fitarg['limit_x'], (0, 20))


def test_minos_all():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    m.minos()
    assert_almost_equal(m.merrors[('x', -1.0)], -sqrt(5))
    assert_almost_equal(m.merrors[('x', 1.0)], sqrt(5))
    assert_almost_equal(m.merrors[('y', 1.0)], 1.)


def test_minos_single():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    m.minos('x')
    assert_almost_equal(m.merrors[('x', -1.0)], -sqrt(5))
    assert_almost_equal(m.merrors[('x', 1.0)], sqrt(5))


@raises(RuntimeWarning)
def test_minos_single_fixed_raising():
    m = Minuit(func3, pedantic=False, print_level=0, fix_x=True)
    m.migrad()

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ret = m.minos('x')


def test_minos_single_fixed_result():
    m = Minuit(func3, pedantic=False, print_level=0, fix_x=True)
    m.migrad()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ret = m.minos('x')
    assert_equal(ret, None)


@raises(RuntimeError)
def test_minos_single_no_migrad():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.minos('x')


@raises(RuntimeError)
def test_minos_single_nonsense_variable():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    m.minos('nonsense')


def test_fixing_long_variable_name():
    m = Minuit(func5, pedantic=False, print_level=0,
               fix_long_variable_name_really_long_why_does_it_has_to_be_this_long=True,
               long_variable_name_really_long_why_does_it_has_to_be_this_long=0)
    m.migrad()


def test_initial_value():
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    assert_almost_equal(m.args[0], 1.)
    assert_almost_equal(m.args[1], 2.)
    assert_almost_equal(m.values['x'], 1.)
    assert_almost_equal(m.values['y'], 2.)
    assert_almost_equal(m.errors['x'], 3.)


def test_mncontour():
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour('x', 'y', numpoints=30)
    xminos_t = m.minos('x')['x']
    yminos_t = m.minos('y')['y']
    assert_almost_equal(xminos.upper, xminos_t.upper)
    assert_almost_equal(xminos.lower, xminos_t.lower)
    assert_almost_equal(yminos.upper, yminos_t.upper)
    assert_almost_equal(yminos.lower, yminos_t.lower)
    assert_equal(len(ctr), 30)
    assert_equal(len(ctr[0]), 2)


def test_mncontour_sigma():
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour('x', 'y', numpoints=30, sigma=2.0)
    xminos_t = m.minos('x', sigma=2.0)['x']
    yminos_t = m.minos('y', sigma=2.0)['y']
    assert_almost_equal(xminos.upper, xminos_t.upper)
    assert_almost_equal(xminos.lower, xminos_t.lower)
    assert_almost_equal(yminos.upper, yminos_t.upper)
    assert_almost_equal(yminos.lower, yminos_t.lower)
    assert_equal(len(ctr), 30)
    assert_equal(len(ctr[0]), 2)


def test_contour():
    # FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.contour('x', 'y')


def test_profile():
    # FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.profile('y')


def test_mnprofile():
    # FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.mnprofile('y')


@raises(RuntimeError)
def test_printfmin_uninitialized():
    # issue 85
    def f(x): return 2 + 3 * x

    fitter = Minuit(f, pedantic=False)
    fitter.print_fmin()


@raises(ValueError)
def test_reverse_limit():
    # issue 94
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(f, limit_x=(3., 2.), pedantic=False)
    m.migrad()


class TestMatrix(TestCase):
    def setUp(self):
        self.m = Minuit(func3, print_level=0, pedantic=False)
        self.m.migrad()

    def test_matrix(self):
        actual = self.m.np_matrix()
        expected = [[5., 0.], [0., 1.]]
        assert_array_almost_equal(actual, expected)

    def test_np_matrix(self):
        import numpy as np
        actual = self.m.np_matrix()
        expected = [[5., 0.], [0., 1.]]
        assert_array_almost_equal(actual, expected)
        assert isinstance(actual, np.ndarray)

    def test_matrix_correlation(self):
        actual = self.m.matrix(correlation=True)
        expected = [[1., 0.], [0., 1.]]
        assert_array_almost_equal(actual, expected)

    def test_np_matrix_correlation(self):
        import numpy as np
        actual = self.m.np_matrix(correlation=True)
        expected = [[1., 0.], [0., 1.]]
        assert_array_almost_equal(actual, expected)
        assert isinstance(actual, np.ndarray)


def test_chi2_fit():
    """Fit a curve to data."""
    m = Minuit(chi2, s=2., error_A=0.1, errordef=0.01,
               print_level=0, pedantic=False)
    m.migrad()
    output = [round(10 * m.values['A']), round(100 * m.values['s']),
              round(100 * m.values['m'])]
    expected = [round(10 * 64.375993), round(100 * 4.267970),
                round(100 * 9.839172)]
    assert_array_almost_equal(output, expected)


def test_oneside():
    m_limit = Minuit(func3, limit_x=(None, 9), pedantic=False, print_level=0)
    m_nolimit = Minuit(func3, pedantic=False, print_level=0)
    # Solution: x=2., y=5.
    m_limit.tol = 1e-4
    m_nolimit.tol = 1e-4
    m_limit.migrad()
    m_nolimit.migrad()
    assert_array_almost_equal(list(m_limit.values.values()),
                              list(m_nolimit.values.values()), decimal=4)


def test_oneside_outside():
    m = Minuit(func3, limit_x=(None, 1), pedantic=False, print_level=0)
    m.migrad()
    assert_almost_equal(m.values['x'], 1)
