import unittest
from RTMinuit.util import *
class TestUtil(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_fitarg_rename(self):
        fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
        ren = lambda x: 'z_'+x
        newfa = fitarg_rename(fitarg,ren)
        self.assertIn('z_x',newfa)
        self.assertIn('limit_z_x',newfa)
        self.assertIn('error_z_x',newfa)
        self.assertIn('fix_z_x',newfa)
        self.assertEqual(len(newfa),4)
        
    def test_fitarg_rename_strprefix(self):
        fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
        newfa = fitarg_rename(fitarg,'z')
        self.assertIn('z_x',newfa)
        self.assertIn('limit_z_x',newfa)
        self.assertIn('error_z_x',newfa)
        self.assertIn('fix_z_x',newfa)
        self.assertEqual(len(newfa),4)

if __name__ == '__main__':
    unittest.main()
    