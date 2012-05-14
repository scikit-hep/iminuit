import unittest
from RTMinuit import Minuit
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

class TestRTMinuit(unittest.TestCase):
    def setUp(self):
        self.m = Minuit(func3,printlevel=-1)
        self.m.migrad()
        
    def functesthelper(self,f):
        m = Minuit(f,printlevel=-1)
        m.migrad()
        val = m.values
        self.assertAlmostEqual(val['x'],2.)
        self.assertAlmostEqual(val['y'],5.)
        self.assertAlmostEqual(m.fmin(),10.)
           
    def test_f1(self):
        self.functesthelper(Func1())
    
    def test_f2(self):
        self.functesthelper(Func2())
    
    def test_f3(self):
        self.functesthelper(func3)

    def test_error_matrix(self):
        m = self.m.error_matrix()
        expected = np.array([[5.,0.],[0.,1.]])
        assert_array_almost_equal(m,expected)

    def test_error_matrix_correlation(self):
        m = self.m.error_matrix(correlation=True)
        expected = np.array([[1.,0.],[0.,1.]])
        assert_array_almost_equal(m,expected)

if __name__ == '__main__':
    unittest.main()   