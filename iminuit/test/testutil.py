from nose.tools import *
from iminuit.util import *

def test_fitarg_rename():
    fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
    ren = lambda x: 'z_'+x
    newfa = fitarg_rename(fitarg,ren)
    assert_true('z_x' in newfa)
    assert_true('limit_z_x' in newfa)
    assert_true('error_z_x' in newfa)
    assert_true('fix_z_x' in newfa)
    assert_equal(len(newfa),4)

def test_fitarg_rename_strprefix():
    fitarg = {'x':1,'limit_x':(2,3),'fix_x':True,'error_x':10}
    newfa = fitarg_rename(fitarg,'z')
    assert_true('z_x' in newfa)
    assert_true('limit_z_x' in newfa)
    assert_true('error_z_x' in newfa)
    assert_true('fix_z_x' in newfa)
    assert_equal(len(newfa),4)

def test_true_param():
    assert_true(true_param('N'))
    assert_false(true_param('limit_N'))
    assert_false(true_param('error_N'))
    assert_false(true_param('fix_N'))

def test_param_name():
    assert_equal(param_name('N'),'N')
    assert_equal(param_name('limit_N'),'N')
    assert_equal(param_name('error_N'),'N')
    assert_equal(param_name('fix_N'),'N')

def test_extract_iv():
    d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
    ret = extract_iv(d)
    assert_true('k' in ret)
    assert_false('limit_k' in ret)
    assert_false('error_k' in ret)
    assert_false('fix_k' in ret)

def test_extract_limit():
    d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
    ret = extract_limit(d)
    assert_false('k' in ret)
    assert_true('limit_k' in ret)
    assert_false('error_k' in ret)
    assert_false('fix_k' in ret)

def test_extract_error():
    d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
    ret = extract_error(d)
    assert_false('k' in ret)
    assert_false('limit_k' in ret)
    assert_true('error_k' in ret)
    assert_false('fix_k' in ret)

def test_extract_fix():
    d = dict(k=1.,limit_k=1.,error_k=1.,fix_k=1.)
    ret = extract_fix(d)
    assert_false('k' in ret)
    assert_false('limit_k' in ret)
    assert_false('error_k' in ret)
    assert_true('fix_k' in ret)

def test_remove_var():
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
    for k in dk: assert_false(k in ret)
    for k in dl: assert_true(k in ret)
    for k in dm: assert_false(k in ret)
    for k in dn: assert_true(k in ret)

def test_arguments_from_docstring():
    s = 'f(x, y, z)'
    a = arguments_from_docstring(s)
    assert_equal(a, ['x','y','z'])
    #this is a hard one
    s = 'Minuit.migrad( int ncall_me =10000, [resume=True, int nsplit=1])'
    a = arguments_from_docstring(s)
    assert_equal(a, ['ncall_me','resume','nsplit'])
