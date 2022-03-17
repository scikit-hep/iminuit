# flake8: noqa E501
from iminuit import Minuit
from iminuit.util import Params, Param, Matrix, FMin, MError
from iminuit import _repr_html, _repr_text
import pytest
from argparse import Namespace
from pathlib import Path
import numpy as np

nan = float("nan")
inf = float("infinity")


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


f1.errordef = 1


def test_color_1():
    g = _repr_html.ColorGradient((-1, 10, 10, 20), (2, 20, 20, 10))
    assert g.rgb(-1) == "rgb(10,10,20)"
    assert g.rgb(2) == "rgb(20,20,10)"
    assert g.rgb(-1.00001) == "rgb(10,10,20)"
    assert g.rgb(1.99999) == "rgb(20,20,10)"
    assert g.rgb(0.5) == "rgb(15,15,15)"


def test_color_2():
    g = _repr_html.ColorGradient(
        (-1, 50, 50, 250), (0, 100, 100, 100), (1, 250, 50, 50)
    )
    assert g.rgb(-1) == "rgb(50,50,250)"
    assert g.rgb(-0.5) == "rgb(75,75,175)"
    assert g.rgb(0) == "rgb(100,100,100)"
    assert g.rgb(0.5) == "rgb(175,75,75)"
    assert g.rgb(1) == "rgb(250,50,50)"


def test_html_tag():
    tag = _repr_html.tag

    def stag(*args, **kwargs):
        return _repr_html.to_str(tag(*args, **kwargs))

    # fmt: off
    assert stag('foo', 'bar', baz='hi', xyzzy='2') == '<foo baz="hi" xyzzy="2"> bar </foo>'
    assert stag('foo') == """<foo></foo>"""
    assert tag('foo', tag('bar', 'baz')) == ['<foo>', ['<bar> baz </bar>'], '</foo>']
    assert stag('foo', tag('bar', 'baz')) == """<foo>
    <bar> baz </bar>
</foo>"""
    # fmt: on


def ref(fn):
    with open(Path(__file__).parent / fn, encoding="utf-8") as f:
        return f.read().strip()


def test_pdg_format():
    assert _repr_text.pdg_format(1.2567, 0.1234) == ["1.26", "0.12"]
    assert _repr_text.pdg_format(1.2567e3, 0.1234e3) == ["1.26e3", "0.12e3"]
    assert _repr_text.pdg_format(1.2567e4, 0.1234e4) == ["12.6e3", "1.2e3"]
    assert _repr_text.pdg_format(1.2567e-1, 0.1234e-1) == ["0.126", "0.012"]
    assert _repr_text.pdg_format(1.2567e-2, 0.1234e-2) == ["0.0126", "0.0012"]
    assert _repr_text.pdg_format(1.0, 0.0, 0.25) == ["1.00", "0.00", "0.25"]
    assert _repr_text.pdg_format(0, 1, -1) == ["0", "1", "-1"]
    assert _repr_text.pdg_format(2, -1, 1) == ["2", "-1", "1"]
    assert _repr_text.pdg_format(2.01, -1.01, 1.01) == ["2", "-1", "1"]
    assert _repr_text.pdg_format(1.999, -0.999, 0.999) == ["2", "-1", "1"]
    assert _repr_text.pdg_format(1, 0.5, -0.5) == ["1.0", "0.5", "-0.5"]
    assert _repr_text.pdg_format(1.0, 1e-3) == ["1.000", "0.001"]
    assert _repr_text.pdg_format(1.0, 1e-4) == ["1.0000", "0.0001"]
    assert _repr_text.pdg_format(1.0, 1e-5) == ["1.00000", "0.00001"]
    assert _repr_text.pdg_format(-1.234567e-22, 1.234567e-11) == ["-0", "0.012e-9"]
    assert _repr_text.pdg_format(nan, 1.23e-2) == ["nan", "0.012"]
    assert _repr_text.pdg_format(nan, 1.23e10) == ["nan", "0.012e12"]
    assert _repr_text.pdg_format(nan, -nan) == ["nan", "nan"]
    assert _repr_text.pdg_format(inf, 1.23e10) == ["inf", "0.012e12"]


def test_matrix_format():
    def a(*args):
        return np.array(args)

    assert _repr_text.matrix_format(a(1e1, 2e2, -3e3, -4e4)) == [
        "10",
        "200",
        "-3e+03",
        "-4e+04",
    ]
    assert _repr_text.matrix_format(a(nan, 2e2, -nan, -4e4)) == [
        "nan",
        "200",
        "nan",
        "-4e+04",
    ]
    assert _repr_text.matrix_format(a(inf, 2e2, -nan, -4e4)) == [
        "inf",
        "200",
        "nan",
        "-4e+04",
    ]


@pytest.fixture
def minuit():
    m = Minuit(f1, x=0, y=0)
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    m.minos()
    return m


@pytest.fixture
def fmin_good():
    fm = Namespace(
        fval=11.456,
        edm=1.23456e-10,
        up=0.5,
        is_valid=True,
        has_valid_parameters=True,
        has_accurate_covar=True,
        has_posdef_covar=True,
        has_made_posdef_covar=False,
        hesse_failed=False,
        has_covariance=True,
        is_above_max_edm=False,
        has_reached_call_limit=False,
        has_parameters_at_limit=False,
        errordef=1,
        state=[],
    )
    return FMin(fm, "Migrad", 10, 3, 10, 1e-4, 0.01)


@pytest.fixture
def fmin_bad():
    fm = Namespace(
        fval=nan,
        edm=1.23456e-10,
        up=0.5,
        is_valid=False,
        has_valid_parameters=False,
        has_accurate_covar=False,
        has_posdef_covar=False,
        has_made_posdef_covar=True,
        hesse_failed=True,
        has_covariance=False,
        is_above_max_edm=True,
        has_reached_call_limit=True,
        has_parameters_at_limit=True,
        errordef=1,
        state=[
            Namespace(
                has_limits=True,
                is_fixed=False,
                value=0,
                error=0.5,
                lower_limit=0,
                upper_limit=1,
                has_lower_limit=True,
                has_upper_limit=True,
            )
        ],
    )
    return FMin(fm, "SciPy[L-BFGS-B]", 100000, 200000, 1, 1e-5, 1.2)


def test_html_fmin_good(fmin_good):
    assert fmin_good._repr_html_() == ref("fmin_good.html").format(
        good=_repr_html.good_style
    )


def test_html_fmin_bad(fmin_bad):
    assert fmin_bad._repr_html_() == ref("fmin_bad.html").format(
        bad=_repr_html.bad_style, warn=_repr_html.warn_style
    )


def test_html_params(minuit):
    assert minuit.init_params._repr_html_() == ref("params_init.html")
    assert minuit.params._repr_html_() == ref("params.html")


def test_html_params_with_limits():
    m = Minuit(f1, x=3, y=5)
    m.fixed["x"] = True
    m.errors = (0.2, 0.1)
    m.limits = ((0, None), (0, 10))
    assert m.init_params._repr_html_() == ref("params_with_limits.html")


def test_html_merror(minuit):
    me = minuit.merrors[0]
    assert me._repr_html_() == ref("merror.html").format(good=_repr_html.good_style)


def test_html_merrors(minuit):
    mes = minuit.merrors
    assert mes._repr_html_() == ref("merrors.html").format(good=_repr_html.good_style)


def test_html_matrix():
    matrix = Matrix(("x", "y"))
    matrix[:] = ((1.0, 0.0), (0.0, 0.25))
    assert matrix._repr_html_() == ref("matrix.html")


def test_html_correlation_matrix():
    matrix = Matrix(("x", "y"))
    matrix[:] = ((1.0, 0.707), (0.707, 1.0))
    assert matrix._repr_html_() == ref("matrix_2.html")


def test_html_minuit():
    m = Minuit(lambda x, y: x**2 + 4 * y**2, x=0, y=0)
    m.errordef = 1
    assert m._repr_html_() == m.params._repr_html_()
    m.migrad()
    assert (
        m._repr_html_()
        == m.fmin._repr_html_() + m.params._repr_html_() + m.covariance._repr_html_()
    )
    m.minos()
    assert (
        m._repr_html_()
        == m.fmin._repr_html_()
        + m.params._repr_html_()
        + m.merrors._repr_html_()
        + m.covariance._repr_html_()
    )


def test_text_fmin_good(fmin_good):
    assert _repr_text.fmin(fmin_good) == ref("fmin_good.txt")


def test_text_fmin_bad(fmin_bad):
    assert _repr_text.fmin(fmin_bad) == ref("fmin_bad.txt")


def test_text_params(minuit):
    assert _repr_text.params(minuit.params) == ref("params.txt")


def test_text_params_with_long_names():
    mps = [
        Param(
            0,
            "super-long-name",
            0,
            0,
            None,
            False,
            False,
            None,
            None,
        )
    ]
    assert _repr_text.params(mps) == ref("params_long_names.txt")


def test_text_params_difficult_values():
    mps = [
        Param(
            0,
            "x",
            -1.234567e-22,
            1.234567e-11,
            None,
            True,
            False,
            None,
            None,
        )
    ]
    assert _repr_text.params(mps) == ref("params_difficult_values.txt")


def test_text_params_with_limits():
    m = Minuit(
        f1,
        x=3,
        y=5,
    )
    m.fixed["x"] = True
    m.errors = (0.2, 0.1)
    m.limits = ((0, None), (0, 10))
    assert _repr_text.params(m.params) == ref("params_with_limits.txt")


def test_text_merror():
    me = MError(
        0,
        "x",
        -1.0,
        1.0,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        42,
        0.123,
    )
    assert _repr_text.merrors({None: me}) == ref("merror.txt")


def test_text_merrors(minuit):
    assert _repr_text.merrors(minuit.merrors) == ref("merrors.txt")


def test_text_matrix():
    m = Matrix({"x": 0, "y": 1})
    m[:] = ((1.0, -0.0), (-0.0, 0.25))
    assert _repr_text.matrix(m) == ref("matrix.txt")


def test_text_matrix_mini():
    m = Matrix({"x": 0})
    m[:] = [1.0]
    assert _repr_text.matrix(m) == ref("matrix_mini.txt")


def test_text_matrix_large():
    m = Matrix({"x": 0, "y": 1, "z": 2})
    m[:] = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    assert _repr_text.matrix(m) == ref("matrix_large.txt")


def test_text_matrix_with_long_names():
    m = Matrix({"super-long-name": 0, "x": 1})
    m[:] = ((1.0, 0.1), (0.1, 1.0))
    assert _repr_text.matrix(m) == ref("matrix_long_names.txt")


def test_text_matrix_difficult_values():
    m = Matrix({"x": 0, "y": 1})
    m[:] = ((-1.23456, 0), (0, 0))
    assert _repr_text.matrix(m) == ref("matrix_difficult_values.txt")


def test_text_minuit():
    m = Minuit(lambda x, y: x**2 + 4 * y**2, x=0, y=0)
    m.errordef = 1
    assert str(m) == str(m.params)
    m.migrad()
    assert str(m) == str(m.fmin) + "\n" + str(m.params) + "\n" + str(m.covariance)
    m.minos()
    assert str(m) == str(m.fmin) + "\n" + str(m.params) + "\n" + str(
        m.merrors
    ) + "\n" + str(m.covariance)
