import unittest
from iminuit.util import *
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

    def test_true_param(self):
        self.assertTrue(true_param('N'))
        self.assertFalse(true_param('limit_N'))
        self.assertFalse(true_param('error_N'))
        self.assertFalse(true_param('fix_N'))

    def test_param_name(self):
        self.assertEqual(param_name('N'),'N')
        self.assertEqual(param_name('limit_N'),'N')
        self.assertEqual(param_name('error_N'),'N')
        self.assertEqual(param_name('fix_N'),'N')

    def test_extract_iv(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_iv(d)
        self.assertIn('k',ret)
        self.assertNotIn('limit_k',ret)
        self.assertNotIn('error_k',ret)
        self.assertNotIn('fix_k',ret)

    def test_extract_limit(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_limit(d)
        self.assertNotIn('k',ret)
        self.assertIn('limit_k',ret)
        self.assertNotIn('error_k',ret)
        self.assertNotIn('fix_k',ret)

    def test_extract_error(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_error(d)
        self.assertNotIn('k',ret)
        self.assertNotIn('limit_k',ret)
        self.assertIn('error_k',ret)
        self.assertNotIn('fix_k',ret)

    def test_extract_fix(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_fix(d)
        self.assertNotIn('k',ret)
        self.assertNotIn('limit_k',ret)
        self.assertNotIn('error_k',ret)
        self.assertIn('fix_k',ret)

    def test_remove_var(self):
        dk = dict(k=1,limit_k=1,error_k=1,fix_k=1)
        dl = dict(l=1,limit_l=1,error_l=1,fix_l=1)
        dm = dict(m=1,limit_m=1,error_m=1,fix_m=1)
        dn = dict(n=1,limit_n=1,error_n=1,fix_n=1)
        d = {}
        d.update(dk)
        d.update(dl)
        d.update(dm)
        d.update(dn)

        ret = remove_var(d,['k','m'])
        for k in dk: self.assertNotIn(k,ret)
        for k in dl: self.assertIn(k,ret)
        for k in dm: self.assertNotIn(k,ret)
        for k in dn: self.assertIn(k,ret)

if __name__ == '__main__':
    unittest.main()
