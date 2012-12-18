import unittest
from iminuit import Minuit
import numpy as np
from numpy.testing import assert_array_almost_equal

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

class TestRTMinuit(unittest.TestCase):


    def functesthelper(self,f):
        m = Minuit(f,print_level=0)
        m.migrad()
        val = m.values
        self.assertAlmostEqual(val['x'],2.)
        self.assertAlmostEqual(val['y'],5.)
        self.assertAlmostEqual(m.fval,10.)


    def test_f1(self):
        self.functesthelper(Func1())


    def test_f2(self):
        self.functesthelper(Func2())


    def test_f3(self):
        self.functesthelper(func3)

    def test_typo(self):
        self.assertRaises(RuntimeError,Minuit,func4,printlevel=0)

    def test_fix_param(self):
        m = Minuit(func4,print_level=0)
        m.migrad()
        m.minos()
        val = m.values
        self.assertAlmostEqual(val['x'],2.)
        self.assertAlmostEqual(val['y'],5.)
        self.assertAlmostEqual(val['z'],7.)
        err = m.errors#second derivative
        # self.assertAlmostEqual(err['x'],5.)
        # self.assertAlmostEqual(err['y'],10.)
        # self.assertAlmostEqual(err['z'],4.)

        #now fix z = 10
        m = Minuit(func4,print_level=-1,y=10.,fix_y=True)
        m.migrad()
        val = m.values
        self.assertAlmostEqual(val['x'],2.)
        self.assertAlmostEqual(val['y'],10.)
        self.assertAlmostEqual(val['z'],7.)
        self.assertAlmostEqual(m.fval,10.+2.5)
        free_param = m.list_of_vary_param()
        fix_param = m.list_of_fixed_param()
        self.assertIn('x', free_param) 
        self.assertNotIn('x', fix_param) 
        self.assertIn('y', fix_param) 
        self.assertNotIn('y', free_param) 
        self.assertNotIn('z', fix_param)


    def test_fitarg(self):
        m = Minuit(func4,print_level=-1,y=10.,fix_y=True,limit_x=(0,20.))
        fitarg = m.fitarg
        self.assertAlmostEqual(fitarg['y'],10.)
        self.assertTrue(fitarg['fix_y'])
        self.assertEqual(fitarg['limit_x'],(0,20))
        m.migrad()

        fitarg = m.fitarg

        self.assertAlmostEqual(fitarg['y'],10.)
        self.assertAlmostEqual(fitarg['x'],2.,delta=1)
        self.assertAlmostEqual(fitarg['z'],7.,delta=1)

        self.assertIn('error_y',fitarg)
        self.assertIn('error_x',fitarg)
        self.assertIn('error_z',fitarg)

        self.assertTrue(fitarg['fix_y'])
        self.assertEqual(fitarg['limit_x'],(0,20))

class TestErrorMatrix(unittest.TestCase):

    def setUp(self):
        self.m = Minuit(func3,print_level=0)
        self.m.migrad()


    def test_error_matrix(self):
        m = np.array(self.m.matrix())
        expected = np.array([[5.,0.],[0.,1.]])
        assert_array_almost_equal(m,expected)


    def test_error_matrix_correlation(self):
        m = np.array(self.m.matrix(correlation=True))
        expected = np.array([[1.,0.],[0.,1.]])
        assert_array_almost_equal(m,expected)


if __name__ == '__main__':
    unittest.main()   