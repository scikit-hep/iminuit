from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit.util import (fitarg_rename,
                          true_param,
                          param_name,
                          extract_iv,
                          extract_limit,
                          extract_error,
                          extract_fix,
                          remove_var,
                          arguments_from_docstring,
                          Struct)
import pytest


def test_fitarg_rename():
    fitarg = {'x': 1, 'limit_x': (2, 3), 'fix_x': True, 'error_x': 10}

    def ren(x):
        return 'z_' + x

    newfa = fitarg_rename(fitarg, ren)
    assert 'z_x' in newfa
    assert 'limit_z_x' in newfa
    assert 'error_z_x' in newfa
    assert 'fix_z_x' in newfa
    assert len(newfa) == 4


def test_fitarg_rename_strprefix():
    fitarg = {'x': 1, 'limit_x': (2, 3), 'fix_x': True, 'error_x': 10}
    newfa = fitarg_rename(fitarg, 'z')
    assert 'z_x' in newfa
    assert 'limit_z_x' in newfa
    assert 'error_z_x' in newfa
    assert 'fix_z_x' in newfa
    assert len(newfa) == 4


def test_true_param():
    assert true_param('N') is True
    assert true_param('limit_N') is False
    assert true_param('error_N') is False
    assert true_param('fix_N') is False


def test_param_name():
    assert param_name('N') == 'N'
    assert param_name('limit_N') == 'N'
    assert param_name('error_N') == 'N'
    assert param_name('fix_N') == 'N'


def test_extract_iv():
    d = dict(k=1., limit_k=1., error_k=1., fix_k=1.)
    ret = extract_iv(d)
    assert 'k' in ret
    assert 'limit_k' not in ret
    assert 'error_k' not in ret
    assert 'fix_k' not in ret


def test_extract_limit():
    d = dict(k=1., limit_k=1., error_k=1., fix_k=1.)
    ret = extract_limit(d)
    assert 'k' not in ret
    assert 'limit_k' in ret
    assert 'error_k' not in ret
    assert 'fix_k' not in ret


def test_extract_error():
    d = dict(k=1., limit_k=1., error_k=1., fix_k=1.)
    ret = extract_error(d)
    assert 'k' not in ret
    assert 'limit_k' not in ret
    assert 'error_k' in ret
    assert 'fix_k' not in ret


def test_extract_fix():
    d = dict(k=1., limit_k=1., error_k=1., fix_k=1.)
    ret = extract_fix(d)
    assert 'k' not in ret
    assert 'limit_k' not in ret
    assert 'error_k' not in ret
    assert 'fix_k' in ret


def test_remove_var():
    dk = dict(k=1, limit_k=1, error_k=1, fix_k=1)
    dl = dict(l=1, limit_l=1, error_l=1, fix_l=1)
    dm = dict(m=1, limit_m=1, error_m=1, fix_m=1)
    dn = dict(n=1, limit_n=1, error_n=1, fix_n=1)
    d = {}
    d.update(dk)
    d.update(dl)
    d.update(dm)
    d.update(dn)

    ret = remove_var(d, ['k', 'm'])
    for k in dk:
        assert k not in ret
    for k in dl:
        assert k in ret
    for k in dm:
        assert k not in ret
    for k in dn:
        assert k in ret


def test_arguments_from_docstring():
    s = 'f(x, y, z)'
    a = arguments_from_docstring(s)
    assert a == ['x', 'y', 'z']
    # this is a hard one
    s = 'Minuit.migrad( int ncall_me =10000, [resume=True, int nsplit=1])'
    a = arguments_from_docstring(s)
    assert a == ['ncall_me', 'resume', 'nsplit']


def test_Struct():
    s = Struct(a=1, b=2)
    assert set(s.keys()) == {'a', 'b'}
    assert s.a == 1
    assert s.b == 2
    s.a = 3
    assert s.a == 3
    with pytest.raises(AttributeError):
        s.c
    with pytest.raises(KeyError):
        s['c']
    assert s == Struct(a=3, b=2)
