import unittest
from iminuit.util import *
class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_fitarg_rename(self):
        fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
        ren = lambda x: 'z_'+x
        newfa = fitarg_rename(fitarg,ren)
        self.assertTrue('z_x' in newfa)
        self.assertTrue('limit_z_x' in newfa)
        self.assertTrue('error_z_x' in newfa)
        self.assertTrue('fix_z_x' in newfa)
        self.assertEqual(len(newfa),4)

    def test_fitarg_rename_strprefix(self):
        fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
        newfa = fitarg_rename(fitarg,'z')
        self.assertTrue('z_x' in newfa)
        self.assertTrue('limit_z_x' in newfa)
        self.assertTrue('error_z_x' in newfa)
        self.assertTrue('fix_z_x' in newfa)
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
        self.assertTrue('k' in ret)
        self.assertFalse('limit_k' in ret)
        self.assertFalse('error_k' in ret)
        self.assertFalse('fix_k' in ret)

    def test_extract_limit(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_limit(d)
        self.assertFalse('k' in ret)
        self.assertTrue('limit_k' in ret)
        self.assertFalse('error_k' in ret)
        self.assertFalse('fix_k' in ret)

    def test_extract_error(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_error(d)
        self.assertFalse('k' in ret)
        self.assertFalse('limit_k' in ret)
        self.assertTrue('error_k' in ret)
        self.assertFalse('fix_k' in ret)

    def test_extract_fix(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_fix(d)
        self.assertFalse('k' in ret)
        self.assertFalse('limit_k' in ret)
        self.assertFalse('error_k' in ret)
        self.assertTrue('fix_k' in ret)

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
        for k in dk: self.assertFalse(k in ret)
        for k in dl: self.assertTrue(k in ret)
        for k in dm: self.assertFalse(k in ret)
        for k in dn: self.assertTrue(k in ret)

if __name__ == '__main__':
    unittest.main()
