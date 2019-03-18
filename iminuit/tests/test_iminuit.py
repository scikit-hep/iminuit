from __future__ import (absolute_import, division, print_function)
import warnings
import pytest
from iminuit.tests.utils import assert_allclose
from iminuit import Minuit
from iminuit.util import Param
from iminuit.iminuit_warnings import InitialParamWarning
import numpy as np
import platform
is_pypy = platform.python_implementation()

parametrize = pytest.mark.parametrize


def test_pedantic_warning_message():
    with warnings.catch_warnings(record=True) as w:
        # use lineno of the next line for the test
        m = Minuit(lambda x: 0) # beware: this line is referred to in the test below!

        assert len(w) == 3
        for i, msg in enumerate((
            "Parameter x does not have initial value. Assume 0.",
            "Parameter x is floating but does not have initial step size. Assume 1.",
            "errordef is not given. Default to 1.")):
            assert str(w[i].message) == msg
            assert w[i].filename == __file__
            assert w[i].lineno == 18 # the lineno of "m = Minuit(lambda x: 0)"


class Func_Code:
    def __init__(self, varname):
        self.co_varnames = varname
        self.co_argcount = len(varname)


class Func1:
    def __init__(self):
        pass

    def __call__(self, x, y):
        return (x - 2.) ** 2 / 4.0 + (y - 5.) ** 2 + 10


class Func2:
    def __init__(self):
        self.func_code = Func_Code(['x', 'y'])

    def __call__(self, *arg):
        return (arg[0] - 2.) ** 2 / 4.0 + (arg[1] - 5.) ** 2 + 10


def func3(x, y):
    return (x - 2.) ** 2 / 4.0 + (y - 5.) ** 2 + 10


def func3_grad(x, y):
    dfdx = (x - 2.) / 2.0
    dfdy = 2.0 * (y - 5.)
    return [dfdx, dfdy]


def func4(x, y, z):
    return 0.2 * (x - 2.) ** 2 + 0.1 * (y - 5.) ** 2 + 0.25 * (z - 7.) ** 2 + 10


def func4_grad(x, y, z):
    dfdx = 0.4 * (x - 2.)
    dfdy = 0.2 * (y - 5.)
    dfdz = 0.5 * (z - 7.)
    return dfdx, dfdy, dfdz


def func5(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    return (x ** 2) + (z ** 2) + long_variable_name_really_long_why_does_it_has_to_be_this_long ** 2


def func5_grad(x, long_variable_name_really_long_why_does_it_has_to_be_this_long, z):
    dfdx = 2 * x
    dfdy = 2 * long_variable_name_really_long_why_does_it_has_to_be_this_long
    dfdz = 2 * z
    return dfdx, dfdy, dfdz


def func6(x, m, s, a):
    return a / ((x - m) ** 2 + s ** 2)


def func7(*args):  # no signature
    return (args[0] - 1) ** 2 + (args[1] - 2) ** 2


def func8(x):  # test numpy support
    return np.sum((x - 1) ** 2)


def func8_grad(x):  # test numpy support
    return 2 * (x - 1)


class Func9:
    def __init__(self):
        sx = 2
        sy = 1
        corr = 0.5
        cov = (sx ** 2, corr * sx * sy), (corr * sx * sy, sy ** 2)
        self.cinv = np.linalg.inv(cov)

    def __call__(self, x):
        return np.dot(x.T, np.dot(self.cinv, x))


data_y = [0.552, 0.735, 0.846, 0.875, 1.059, 1.675, 1.622, 2.928,
          3.372, 2.377, 4.307, 2.784, 3.328, 2.143, 1.402, 1.44,
          1.313, 1.682, 0.886, 0.0, 0.266, 0.3]
data_x = list(range(len(data_y)))


def chi2(m, s, a):
    """Chi2 fitting routine"""
    return sum(((func6(x, m, s, a) - y) ** 2 for x, y in zip(data_x, data_y)))


def functesthelper(f, **kwds):
    m = Minuit(f, print_level=0, pedantic=False, **kwds)
    m.migrad()
    val = m.values
    assert_allclose(val['x'], 2., rtol=1e-3)
    assert_allclose(val['y'], 5., rtol=1e-3)
    assert_allclose(m.fval, 10., rtol=1e-3)
    assert m.matrix_accurate()
    assert m.migrad_ok()
    m.hesse()
    err = m.errors
    assert_allclose(err['x'], 2., rtol=1e-3)
    assert_allclose(err['y'], 1., rtol=1e-3)
    return m


def test_f1():
    functesthelper(Func1())


def test_f2():
    functesthelper(Func2())


def test_f3(): # check that providing gradient improves convergence
    m1 = functesthelper(func3)
    m2 = functesthelper(func3, grad=func3_grad)
    assert m1.get_num_call_grad() == 0
    assert m2.get_num_call_grad() > 0
    assert m1.get_num_call_fcn() > m2.get_num_call_fcn()


def test_lambda():
    functesthelper(lambda x, y: (x - 2.) ** 2 / 4.0 + (y - 5.) ** 2 + 10)


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


def test_from_array_func_1():
    m = Minuit.from_array_func(func8, (2, 1),
                               error=(1, 1),
                               errordef=1,
                               print_level=0)
    assert m.fitarg == {"x0": 2,
                        "x1": 1,
                        "error_x0": 1.0,
                        "error_x1": 1.0,
                        "fix_x0": False,
                        "fix_x1": False,
                        "limit_x0": None,
                        "limit_x1": None}
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1), rtol=1e-2)
    c = m.np_covariance()
    assert_allclose(np.diag(c), (1, 1), rtol=1e-2)


def test_from_array_func_2():
    m = Minuit.from_array_func(func8, (2, 1),
                               grad=func8_grad,
                               error=(0.5, 0.5),
                               limit=((0, 2), (0, 2)),
                               fix=(False, True),
                               name=("a", "b"),
                               errordef=1,
                               print_level=0)
    assert m.fitarg == {"a": 2,
                        "b": 1,
                        "error_a": 0.5,
                        "error_b": 0.5,
                        "fix_a": False,
                        "fix_b": True,
                        "limit_a": (0, 2),
                        "limit_b": (0, 2)}
    m.migrad()
    v = m.np_values()
    assert_allclose(v, (1, 1), rtol=1e-2)
    c = m.np_covariance()
    assert_allclose(c, ((1, 0), (0, 0)), rtol=1e-2)


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


def test_view_repr():
    m = Minuit(func3, print_level=0, errordef=1,
               x=1, y=2, error_x=3, error_y=4)
    mid = id(m)
    assert repr(m.values) == ("""
<ValueView of Minuit at %x>
  x: 1.0
  y: 2.0
""" % mid).strip()
    assert repr(m.args) == ("""
<ArgsView of Minuit at %x>
  1.0
  2.0
""" % mid).strip()


def test_no_resume():
    m = Minuit(func3, print_level=0, pedantic=False)
    m.migrad()
    n = m.get_num_call_fcn()
    m.migrad()
    assert m.get_num_call_fcn() > n
    m.migrad(resume=False)
    assert m.get_num_call_fcn() == n

    m = Minuit(func3, grad=func3_grad, print_level=0, pedantic=False)
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


@parametrize('grad', (None, func3_grad))
def test_fix_param(grad):
    kwds = {'print_level': 0, 'pedantic': False, 'grad': grad}
    m = Minuit(func3, **kwds)
    m.migrad()
    m.minos()
    assert_allclose(m.np_values(), (2, 5), rtol=1e-2)
    assert_allclose(m.np_errors(), (2, 1))
    assert_allclose(m.matrix(), ((4, 0), (0, 1)), atol=1e-4)
    for b in (True, False):
        assert_allclose(m.matrix(skip_fixed=b), [[4, 0], [0, 1]], atol=1e-4)
    # now fix y = 10
    m = Minuit(func3, y=10., fix_y=True, **kwds)
    m.migrad()
    assert_allclose(m.np_values(), (2, 10), rtol=1e-2)
    assert_allclose(m.fval, 35)
    assert m.list_of_vary_param() == ['x']
    assert m.list_of_fixed_param() == ['y']
    assert_allclose(m.matrix(skip_fixed=True), [[4]], atol=1e-4)
    assert_allclose(m.matrix(skip_fixed=False), [[4, 0], [0, 0]], atol=1e-4)

    assert m.is_fixed('x') is False
    assert m.is_fixed('y') is True

    m.fixed['x'] = True
    m.fixed['y'] = False
    m.migrad()
    m.hesse()
    assert_allclose(m.np_values(), (2, 5), rtol=1e-2)
    assert_allclose(m.matrix(skip_fixed=True), [[1]], atol=1e-4)
    assert_allclose(m.matrix(skip_fixed=False), [[0, 0], [0, 1]], atol=1e-4)

    with pytest.raises(KeyError):
        m.is_fixed('a')

    # fix by setting limits
    m = Minuit(func3, y=10., limit_y=(10, 10), pedantic=False, print_level=0)
    assert m.fixed['y']

    # initial value out of range is forced in range
    m = Minuit(func3, y=20., limit_y=(10, 10), pedantic=False, print_level=0)
    assert m.fixed['y']
    assert m.values['y'] == 10


def test_fitarg_oneside():
    m = Minuit(func4, print_level=0, y=10., fix_y=True, limit_x=(None, 20.),
               pedantic=False)
    fitarg = m.fitarg
    assert_allclose(fitarg['y'], 10.)
    assert fitarg['fix_y']
    assert fitarg['limit_x'] == (None, 20)
    m.migrad()

    fitarg = m.fitarg

    assert_allclose(fitarg['x'], 2., atol=1e-2)
    assert_allclose(fitarg['y'], 10., atol=1e-2)
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


@parametrize('grad', (None, func3_grad))
@parametrize('sigma', (1, 4))
def test_minos_all(grad, sigma):
    m = Minuit(func3, grad=func3_grad, pedantic=False, print_level=0)
    m.migrad()
    m.minos(sigma=sigma)
    assert_allclose(m.merrors[('x', -1.0)], -sigma * 2, rtol=1e-2)
    assert_allclose(m.merrors[('x', 1.0)], sigma * 2, rtol=1e-2)
    assert_allclose(m.merrors[('y', 1.0)], sigma * 1, rtol=1e-2)


@parametrize('grad', (None, func3_grad))
def test_minos_single(grad):
    m = Minuit(func3, grad=func3_grad, pedantic=False, print_level=0)
    m.set_strategy(2)
    m.migrad()
    m.minos('x')
    assert_allclose(m.merrors[('x', -1.0)], -2)
    assert_allclose(m.merrors[('x', 1.0)], 2)


def test_minos_single_fixed_raising():
    m = Minuit(func3, pedantic=False, print_level=0, fix_x=True)
    m.migrad()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(RuntimeWarning):
            m.minos('x')

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


@parametrize('grad', (None, func5_grad))
def test_fixing_long_variable_name(grad):
    m = Minuit(func5, grad=grad, pedantic=False, print_level=0,
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


@parametrize('grad', (None, func3_grad))
@parametrize('sigma', (1, 2))
def test_mncontour(grad, sigma):
    m = Minuit(func3, grad=grad, pedantic=False,
               x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    xminos, yminos, ctr = m.mncontour('x', 'y', numpoints=30, sigma=sigma)
    xminos_t = m.minos('x', sigma=sigma)['x']
    yminos_t = m.minos('y', sigma=sigma)['y']
    assert_allclose(xminos.upper, xminos_t.upper)
    assert_allclose(xminos.lower, xminos_t.lower)
    assert_allclose(yminos.upper, yminos_t.upper)
    assert_allclose(yminos.lower, yminos_t.lower)
    assert len(ctr) == 30
    assert len(ctr[0]) == 2


@parametrize('grad', (None, func3_grad))
def test_contour(grad):
    # FIXME: check the result
    m = Minuit(func3, grad=grad, pedantic=False,
               x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.contour('x', 'y')


@parametrize('grad', (None, func3_grad))
def test_profile(grad):
    # FIXME: check the result
    m = Minuit(func3, grad=grad, pedantic=False,
               x=1., y=2., error_x=3., print_level=0)
    m.migrad()
    m.profile('y')


@parametrize('grad', (None, func3_grad))
def test_mnprofile(grad):
    # FIXME: check the result
    m = Minuit(func3, grad=grad, pedantic=False, x=1., y=2., error_x=3., print_level=0)
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

    with pytest.raises(ValueError):
        Minuit(f, limit_x=(3., 2.), pedantic=False, print_level=0)


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
        expected = [[4., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)

    def test_matrix_correlation(self):
        actual = self.m.matrix(correlation=True)
        expected = [[1., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)

    def test_np_matrix(self):
        actual = self.m.np_matrix()
        expected = [[4., 0.], [0., 1.]]
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
        expected = [2., 1.]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2,)

    def test_np_merrors(self):
        actual = self.m.np_merrors()
        # output format is [abs(down_delta), up_delta] following
        # the matplotlib convention in matplotlib.pyplot.errorbar
        down_delta = (-2, -1)
        up_delta = (2, 1)
        assert_allclose(actual, (np.abs(down_delta), up_delta), atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 2)

    def test_np_covariance(self):
        actual = self.m.np_covariance()
        expected = [[4., 0.], [0., 1.]]
        assert_allclose(actual, expected, atol=1e-8)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 2)


def test_chi2_fit():
    """Fit a curve to data."""
    m = Minuit(chi2, s=2., error_a=0.1, errordef=0.01,
               print_level=0, pedantic=False)
    m.migrad()
    output = [round(10 * m.values['a']), round(100 * m.values['s']),
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
    assert func.ncall == m.get_num_call_fcn()

    ncall_without_limit = m.get_num_call_fcn()
    # check that ncall argument limits function calls in migrad
    # note1: Minuit only checks the ncall counter in units of one iteration
    # step, therefore the call counter is in general not equal to ncall.
    # note2: If you pass ncall=0, Minuit uses a heuristic value that depends
    # on the number of parameters.
    m.migrad(ncall=1, resume=False)
    assert m.get_num_call_fcn() < ncall_without_limit


def test_set_error_def():
    m = Minuit(lambda x: x ** 2, pedantic=False, print_level=0, errordef=1)
    m.migrad()
    m.hesse()
    assert_allclose(m.errors["x"], 1)
    m.set_errordef(4)
    m.hesse()
    assert_allclose(m.errors["x"], 2)


def test_get_param_states():
    m = Minuit(func3, x=1, y=2, error_x=3, error_y=4,
               fix_x=True, limit_y=(None, 10),
               pedantic=False, errordef=1, print_level=0)
    # these are the initial param states
    expected = [Param(0, 'x', 1.0, 3.0, False, True, False, False, False, None, None),
                Param(1, 'y', 2.0, 4.0, False, False, True, False, True, None, 10)]
    assert m.get_param_states() == expected

    m.migrad()
    assert m.get_initial_param_states() == expected

    expected = [Param(0, 'x', 1.0, 3.0, False, True, False, False, False, None, None),
                Param(1, 'y', 5.0, 1.0, False, False, True, False, True, None, 10)]

    params = m.get_param_states()
    for i, exp in enumerate(expected):
        p = params[i]
        assert set(p._fields) == set(exp._fields)
        for key in exp._fields:
            if key in ('value', 'error'):
                assert_allclose(getattr(p, key), getattr(exp, key), rtol=1e-2)
            else:
                assert getattr(p, key) == getattr(exp, key)


def test_latex_matrix():
    m = Minuit.from_array_func(Func9(), (0, 0), name=('x', 'y'),
                               pedantic=False, print_level=0)
    m.migrad()
    assert r"""%\usepackage[table]{xcolor} % include this for color
%\usepackage{rotating} % include this for rotate header
%\documentclass[xcolor=table]{beamer} % for beamer
\begin{tabular}{|c|c|c|}
\hline
\rotatebox{90}{} & \rotatebox{90}{x} & \rotatebox{90}{y}\\
\hline
x & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{209,185,152} 0.50\\
\hline
y & \cellcolor[RGB]{209,185,152} 0.50 & \cellcolor[RGB]{255,117,117} 1.00\\
\hline
\end{tabular}""" == str(m.latex_matrix())


def test_non_analytical_function():
    class Func:
        i = 0

        def __call__(self, a):
            self.i += 1
            return self.i % 3

    m = Minuit(Func(), pedantic=False, print_level=0)
    fmin, param = m.migrad()
    assert fmin.is_valid is False
    assert fmin.is_above_max_edm is True


def test_function_without_local_minimum():
    m = Minuit(lambda a: -a, pedantic=False, print_level=0)
    fmin, param = m.migrad()
    assert fmin.is_valid is False
    assert fmin.is_above_max_edm is True


# Bug in Minuit2, this needs to be fixed in upstream ROOT
@pytest.mark.xfail(strict=True)
def test_function_with_maximum():
    def func(a):
        return -a ** 2

    m = Minuit(func, pedantic=False, print_level=0)
    fmin, param = m.migrad()
    assert fmin.is_valid is False


def test_perfect_correlation():
    def func(a, b):
        return (a - b) ** 2

    m = Minuit(func, pedantic=False, print_level=0)
    fmin, param = m.migrad()
    assert fmin.is_valid is True
    assert fmin.has_accurate_covar is False
    assert fmin.has_posdef_covar is False
    assert fmin.has_made_posdef_covar is True


def test_modify_param_state():
    m = Minuit(func3, x=1, y=2, fix_y=True, pedantic=False,
               print_level=0)
    m.migrad()
    assert_allclose(m.np_values(), [2, 2], atol=1e-2)
    assert_allclose(m.np_errors(), [2, 1], atol=1e-2)
    m.fixed['y'] = False
    m.values['x'] = 1
    m.errors['x'] = 1
    assert_allclose(m.np_values(), [1, 2], atol=1e-2)
    assert_allclose(m.np_errors(), [1, 1], atol=1e-2)
    m.migrad()
    assert_allclose(m.np_values(), [2, 5], atol=1e-2)
    assert_allclose(m.np_errors(), [2, 1], atol=1e-2)
    m.values['y'] = 6
    m.hesse()  # does not change minimum
    assert_allclose(m.np_values(), [2, 6], atol=1e-2)
    assert_allclose(m.np_errors(), [2, 1], atol=1e-2)


def test_view_lifetime():
    m = Minuit(func3, x=1, y=2, pedantic=False, print_level=0)
    val = m.values
    arg = m.args
    del m
    val['x'] = 3  # should not segfault
    assert val['x'] == 3
    arg[0] = 5  # should not segfault
    assert arg[0] == 5


def test_bad_functions():
    def throwing(x):
        raise RuntimeError("user message")
    divide_by_zero = lambda x: 1/0
    returning_nan = lambda x: np.nan
    returning_garbage = lambda x: "foo"

    for func, expected in (
            (throwing, 'RuntimeError("user message")'),
            (divide_by_zero, "ZeroDivisionError"),
            (returning_nan, "result is NaN"),
            (returning_garbage, "TypeError")):
        m = Minuit(func, x=1, pedantic=False, throw_nan=True,
                   print_level=0)
        with pytest.raises(RuntimeError) as excinfo:
            m.migrad()
        assert expected in excinfo.value.args[0]

    returning_nan = lambda x: np.array([1, np.nan])
    returning_noniterable = lambda x: 0
    returning_garbage = lambda x: np.array([1, "foo"])

    for func, expected in (
            (throwing, 'RuntimeError("user message")'),
            (divide_by_zero, "ZeroDivisionError"),
            (returning_nan, "result is NaN"),
            (returning_garbage, "TypeError"),
            (returning_noniterable, "TypeError")):
        m = Minuit.from_array_func(lambda x: 0, (1, 1), grad=func,
                                   pedantic=False, throw_nan=True,
                                   print_level=0)
        with pytest.raises(RuntimeError) as excinfo:
            m.migrad()
        if is_pypy and func is returning_garbage:
            pass
        else:
            assert expected in excinfo.value.args[0]
