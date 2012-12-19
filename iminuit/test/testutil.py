from unittest import TestCase
from nose.tools import (assert_in, assert_not_in,
    assert_equal, assert_true, assert_false)
from iminuit.util import *

class TestUtil(TestCase):

    def test_fitarg_rename(self):
        fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
        ren = lambda x: 'z_'+x
        newfa = fitarg_rename(fitarg,ren)
        assert_in('z_x', newfa)
        assert_in('limit_z_x', newfa)
        assert_in('error_z_x', newfa)
        assert_in('fix_z_x', newfa)
        assert_equal(len(newfa),4)

    def test_fitarg_rename_strprefix(self):
        fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
        newfa = fitarg_rename(fitarg,'z')
        assert_in('z_x', newfa)
        assert_in('limit_z_x', newfa)
        assert_in('error_z_x', newfa)
        assert_in('fix_z_x', newfa)
        assert_equal(len(newfa),4)

    def test_true_param(self):
        assert_true(true_param('N'))
        assert_false(true_param('limit_N'))
        assert_false(true_param('error_N'))
        assert_false(true_param('fix_N'))

    def test_param_name(self):
        assert_equal(param_name('N'),'N')
        assert_equal(param_name('limit_N'),'N')
        assert_equal(param_name('error_N'),'N')
        assert_equal(param_name('fix_N'),'N')

    def test_extract_iv(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_iv(d)
        assert_in('k', ret)
        assert_not_in('limit_k', ret)
        assert_not_in('error_k', ret)
        assert_not_in('fix_k', ret)

    def test_extract_limit(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_limit(d)
        assert_not_in('k', ret)
        assert_in('limit_k', ret)
        assert_not_in('error_k', ret)
        assert_not_in('fix_k', ret)

    def test_extract_error(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_error(d)
        assert_not_in('k', ret)
        assert_not_in('limit_k', ret)
        assert_in('error_k', ret)
        assert_not_in('fix_k', ret)

    def test_extract_fix(self):
        d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
        ret = extract_fix(d)
        assert_not_in('k', ret)
        assert_not_in('limit_k', ret)
        assert_not_in('error_k', ret)
        assert_in('fix_k', ret)

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
        for k in dk: assert_not_in(k, ret)
        for k in dl: assert_in(k, ret)
        for k in dm: assert_not_in(k, ret)
        for k in dn: assert_in(k, ret)

if __name__ == '__main__':
    unittest.main()
