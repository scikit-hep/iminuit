from unittest import TestCase
from nose.tools import (raises, assert_equal, assert_true, assert_false,
    assert_almost_equal)
from iminuit import Minuit
from math import sqrt


def assert_array_almost_equal(actual, expected):
    """
    Helper function to test if all elements of a list of lists
    are almost equal.
    Like numpy.testing.assert_array_almost_equal,
    which we can't use here because we don't
    want to depend on numpy.
    """
    for row in range(len(actual)):
        for col in range(len(actual[0])):
            assert_almost_equal(actual[row][col], expected[row][col])


class Func_Code:
    def __init__(self, varname):
        self.co_varnames=varname
        self.co_argcount=len(varname)


class Func1:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return (x-2.)**2 + (y-5.)**2 + 10


class Func2:
    def __init__(self):
        self.func_code = Func_Code(['x', 'y'])

    def __call__(self, *arg):
        return (arg[0]-2.)**2 + (arg[1]-5.)**2 + 10


def func3(x, y):
    return 0.2*(x-2.)**2 + (y-5.)**2 + 10


def func4(x, y, z):
    return 0.2*(x-2.)**2 + 0.1*(y-5.)**2 + 0.25*(z-7.)**2 + 10


def func5(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    return (x**2)+(z**2)+long_variable_name_really_long_why_does_it_has_to_be_this_long**2


def functesthelper(f):
    m = Minuit(f, print_level=0, pedantic=False)
    m.migrad()
    val = m.values
    print val
    assert_almost_equal(val['x'], 2.)
    assert_almost_equal(val['y'], 5.)
    assert_almost_equal(m.fval, 10.)
    assert(m.matrix_accurate())
    assert(m.migrad_ok())


def test_f1():
    functesthelper(Func1())


def test_f2():
    functesthelper(Func2())


def test_f3():
    functesthelper(func3)


@raises(RuntimeError)
def test_typo():
    Minuit(func4, printlevel=0)
    #self.assertRaises(RuntimeError,Minuit,func4,printlevel=0)


def test_non_invertible():
    #making sure it doesn't crash
    def f(x, y):
        return (x*y)**2
    m = Minuit(f, pedantic=False, print_level=0)
    result = m.migrad()
    m.hesse()
    try:
        m.matrix()
        assert False #shouldn't reach here
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
    err = m.errors#second derivative
    m.print_all_minos()
    #now fix z = 10
    m = Minuit(func4, print_level=-1, y=10., fix_y=True, pedantic=False)
    m.migrad()
    val = m.values
    assert_almost_equal(val['x'], 2.)
    assert_almost_equal(val['y'], 10.)
    assert_almost_equal(val['z'], 7.)
    assert_almost_equal(m.fval, 10.+2.5)
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


def test_minos_single_fixed():
    m = Minuit(func3, pedantic=False, print_level=0, fix_x=True)
    m.migrad()
    ret = m.minos('x')
    assert_equal(ret, None)
    #assert_almost_equal(m.merrors[('x',-1.0)],-sqrt(5))
    #assert_almost_equal(m.merrors[('x',1.0)],sqrt(5))


@raises(RuntimeError)
def test_minos_single_no_migrad():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.minos('x')


@raises(RuntimeError)
def test_minos_single_nonsense_variable():
    m = Minuit(func3, pedantic=False, print_level=0)
    m.migrad()
    m.minos('nonsense')


def test_fixing_long_variablename():
    m = Minuit(func5, pedantic=False, print_level=0,
    fix_long_variable_name_really_long_why_does_it_has_to_be_this_long=True,
    long_variable_name_really_long_why_does_it_has_to_be_this_long=0)
    m.migrad()


def test_initalvalue():
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
    #FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.contour('x', 'y')


def test_profile():
    #FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.profile('y')


def test_mnprofile():
    #FIXME: check the result
    m = Minuit(func3, pedantic=False, x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.mnprofile('y')

@raises(RuntimeError)
def test_printfmin_uninitialized():
    #issue 85
    def f(x): return 2 + 3 * x
    fitter = Minuit(f)
    fitter.print_fmin()

@raises(ValueError)
def test_reverse_limit():
    #issue 94
    def f(x,y,z):
        return (x-2)**2 + (y-3)**2 + (z-4)**2
    m = Minuit(f, limit_x=(3., 2.))
    m.migrad()

class TestErrorMatrix(TestCase):

    def setUp(self):
        self.m = Minuit(func3, print_level=0, pedantic=False)
        self.m.migrad()

    def test_error_matrix(self):
        actual = self.m.matrix()
        expected = [[5., 0.], [0., 1.]]
        assert_array_almost_equal(actual, expected)

    def test_error_matrix_correlation(self):
        actual = self.m.matrix(correlation=True)
        expected = [[1., 0.], [0., 1.]]
        assert_array_almost_equal(actual, expected)
