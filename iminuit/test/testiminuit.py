from unittest import TestCase
from nose.tools import (raises, assert_equal, assert_true, assert_false,
    assert_almost_equal)
from iminuit import Minuit

def assert_array_almost_equal(actual, expected):
    """
    Helper function to test if all elements of a list of lists
    are almost equal.
    Like numpy.testing.assert_array_almost_equal,
    which we can't user here because we don't
    want to depend on numpy.
    """
    for row in range(len(actual)):
        for col in range(len(actual[0])):
            assert_almost_equal(actual[row][col], expected[row][col])

class Func_Code:
    def __init__(self,varname):
        self.co_varnames=varname
        self.co_argcount=len(varname)

class Func1:
    def __init__(self):
        pass
    def __call__(self,x,y):
        return (x-2.)**2 + (y-5.)**2 + 10

class Func2:
    def __init__(self):
        self.func_code = Func_Code(['x','y'])
    def __call__(self,*arg):
        return (arg[0]-2.)**2 + (arg[1]-5.)**2 + 10

def func3(x,y):
    return 0.2*(x-2.)**2 + (y-5.)**2 + 10


def func4(x,y,z):
    return 0.2*(x-2.)**2 + 0.1*(y-5.)**2 + 0.25*(z-7.)**2 + 10

class TestMinuit(TestCase):

    def functesthelper(self,f):
        m = Minuit(f,print_level=0)
        m.migrad()
        val = m.values
        assert_almost_equal(val['x'],2.)
        assert_almost_equal(val['y'],5.)
        assert_almost_equal(m.fval,10.)


    def test_f1(self):
        self.functesthelper(Func1())


    def test_f2(self):
        self.functesthelper(Func2())


    def test_f3(self):
        self.functesthelper(func3)

    @raises(RuntimeError)
    def test_typo(self):
        Minuit(func4, printlevel=0)
        #self.assertRaises(RuntimeError,Minuit,func4,printlevel=0)

    def test_fix_param(self):
        m = Minuit(func4,print_level=0)
        m.migrad()
        m.minos()
        val = m.values
        assert_almost_equal(val['x'],2.)
        assert_almost_equal(val['y'],5.)
        assert_almost_equal(val['z'],7.)
        err = m.errors#second derivative
        # self.assertAlmostEqual(err['x'],5.)
        # self.assertAlmostEqual(err['y'],10.)
        # self.assertAlmostEqual(err['z'],4.)

        #now fix z = 10
        m = Minuit(func4,print_level=-1,y=10.,fix_y=True)
        m.migrad()
        val = m.values
        assert_almost_equal(val['x'],2.)
        assert_almost_equal(val['y'],10.)
        assert_almost_equal(val['z'],7.)
        assert_almost_equal(m.fval,10.+2.5)
        free_param = m.list_of_vary_param()
        fix_param = m.list_of_fixed_param()
        assert_true('x' in free_param)
        assert_false('x' in fix_param)
        assert_true('y' in fix_param)
        assert_false('y' in free_param)
        assert_false('z' in fix_param)


    def test_fitarg(self):
        m = Minuit(func4,print_level=-1,y=10.,fix_y=True,limit_x=(0,20.))
        fitarg = m.fitarg
        assert_almost_equal(fitarg['y'],10.)
        assert_true(fitarg['fix_y'])
        assert_equal(fitarg['limit_x'],(0,20))
        m.migrad()

        fitarg = m.fitarg

        assert_almost_equal(fitarg['y'],10.)
        assert_almost_equal(fitarg['x'],2.,places=2)
        assert_almost_equal(fitarg['z'],7.,places=2)

        assert_true('error_y' in fitarg)
        assert_true('error_x' in fitarg)
        assert_true('error_z' in fitarg)

        assert_true(fitarg['fix_y'])
        assert_equal(fitarg['limit_x'],(0,20))


class TestErrorMatrix(TestCase):

    def setUp(self):
        self.m = Minuit(func3,print_level=0)
        self.m.migrad()


    def test_error_matrix(self):
        actual = self.m.matrix()
        expected = [[5.,0.],[0.,1.]]
        assert_array_almost_equal(actual, expected)


    def test_error_matrix_correlation(self):
        actual = self.m.matrix(correlation=True)
        expected = [[1.,0.],[0.,1.]]
        assert_array_almost_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
