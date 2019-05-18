from __future__ import (absolute_import, division, print_function)
from iminuit.util import (fitarg_rename,
                          true_param,
                          param_name,
                          extract_iv,
                          extract_limit,
                          extract_error,
                          extract_fix,
                          remove_var,
                          arguments_from_docstring,
                          Matrix,
                          FMin,
                          Param,
                          MError,
                          Params,
                          MigradResult)
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


def test_Matrix():
    x = Matrix(("a", "b"), [[1, 2],[3, 4]])
    assert x[0] == (1, 2)
    assert x[1] == (3, 4)
    assert x == ((1, 2), (3, 4))
    assert repr(x) == '((1, 2), (3, 4))'
    with pytest.raises(TypeError):
        x[0] = (1, 2)
    with pytest.raises(TypeError):
        x[0][0] = 1


def test_Param():
    # number name value error is_const is_fixed has_limits
    # has_lower_limit has_upper_limit lower_limit upper_limit
    p = Param(3, "foo", 1.2, 3.4, False, False, True, True, False, 42, None)

    assert p.has_lower_limit == True
    assert p.has_upper_limit == False
    assert p["has_lower_limit"] == True
    assert p["lower_limit"] == 42
    assert p["upper_limit"] == None
    assert "upper_limit" in p
    assert "foo" not in p

    fields = [key for key in p]
    assert fields == "number name value error is_const is_fixed has_limits has_lower_limit has_upper_limit lower_limit upper_limit".split()
    assert p.keys() == tuple(fields)
    assert p.values() == tuple(p[k] for k in fields)
    assert p.items() == tuple((k, p[k]) for k in fields)

    assert str(p).startswith("Param(number=3, name='foo'")


def test_MError():
    pass


def test_FMin():
    pass


def test_MigradResult():
    fmin = FMin(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    params = Params([], None)
    mr = MigradResult(fmin, params)
    assert mr.fmin is fmin
    assert mr[0] is fmin
    assert mr.params is params
    assert mr[1] is params
    a, b = mr
    assert a is fmin
    assert b is params
