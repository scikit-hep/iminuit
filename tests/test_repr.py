# flake8: noqa E501
from iminuit import Minuit
from iminuit.util import Params, Param, Matrix, FMin
from iminuit import _repr_html as repr_html, _repr_text as repr_text
import pytest
from argparse import Namespace
from pathlib import Path

nan = float("nan")
inf = float("infinity")


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


f1.errordef = 1


def test_color_1():
    g = repr_html.ColorGradient((-1, 10, 10, 20), (2, 20, 20, 10))
    assert g.rgb(-1) == "rgb(10,10,20)"
    assert g.rgb(2) == "rgb(20,20,10)"
    assert g.rgb(-1.00001) == "rgb(10,10,20)"
    assert g.rgb(1.99999) == "rgb(20,20,10)"
    assert g.rgb(0.5) == "rgb(15,15,15)"


def test_color_2():
    g = repr_html.ColorGradient((-1, 50, 50, 250), (0, 100, 100, 100), (1, 250, 50, 50))
    assert g.rgb(-1) == "rgb(50,50,250)"
    assert g.rgb(-0.5) == "rgb(75,75,175)"
    assert g.rgb(0) == "rgb(100,100,100)"
    assert g.rgb(0.5) == "rgb(175,75,75)"
    assert g.rgb(1) == "rgb(250,50,50)"


def test_html_tag():
    tag = repr_html.tag

    def stag(*args, **kwargs):
        return repr_html.to_str(tag(*args, **kwargs))

    # fmt: off
    assert stag('foo', 'bar', baz='hi', xyzzy='2') == '<foo baz="hi" xyzzy="2"> bar </foo>'
    assert stag('foo') == """<foo></foo>"""
    assert tag('foo', tag('bar', 'baz')) == ['<foo>', ['<bar> baz </bar>'], '</foo>']
    assert stag('foo', tag('bar', 'baz')) == """<foo>
    <bar> baz </bar>
</foo>"""
    # fmt: on


def ref(fn):
    with open(Path(__file__).parent / f"{fn}.txt", encoding="utf-8") as f:
        return f.read()[:-1]  # strip trailing newline


def test_pdg_format():
    assert repr_text.pdg_format(1.2567, 0.1234) == ["1.26", "0.12"]
    assert repr_text.pdg_format(1.2567e3, 0.1234e3) == ["1.26e3", "0.12e3"]
    assert repr_text.pdg_format(1.2567e4, 0.1234e4) == ["12.6e3", "1.2e3"]
    assert repr_text.pdg_format(1.2567e-1, 0.1234e-1) == ["0.126", "0.012"]
    assert repr_text.pdg_format(1.2567e-2, 0.1234e-2) == ["0.0126", "0.0012"]
    assert repr_text.pdg_format(1.0, 0.0, 0.25) == ["1.00", "0.00", "0.25"]
    assert repr_text.pdg_format(0, 1, -1) == ["0", "1", "-1"]
    assert repr_text.pdg_format(2, -1, 1) == ["2", "-1", "1"]
    assert repr_text.pdg_format(2.01, -1.01, 1.01) == ["2", "-1", "1"]
    assert repr_text.pdg_format(1.999, -0.999, 0.999) == ["2", "-1", "1"]
    assert repr_text.pdg_format(1, 0.5, -0.5) == ["1.0", "0.5", "-0.5"]
    assert repr_text.pdg_format(1.0, 1e-3) == ["1.000", "0.001"]
    assert repr_text.pdg_format(1.0, 1e-4) == ["1.0000", "0.0001"]
    assert repr_text.pdg_format(1.0, 1e-5) == ["1.00000", "0.00001"]
    assert repr_text.pdg_format(-1.234567e-22, 1.234567e-11) == ["-0", "0.012e-9"]
    assert repr_text.pdg_format(nan, 1.23e-2) == ["nan", "0.012"]
    assert repr_text.pdg_format(nan, 1.23e10) == ["nan", "0.012e12"]
    assert repr_text.pdg_format(nan, -nan) == ["nan", "nan"]
    assert repr_text.pdg_format(inf, 1.23e10) == ["inf", "0.012e12"]


def test_matrix_format():
    assert repr_text.matrix_format(1e1, 2e2, -3e3, -4e4) == [
        "0.001e4",
        "0.020e4",
        "-0.300e4",
        "-4.000e4",
    ]
    assert repr_text.matrix_format(nan, 2e2, -nan, -4e4) == [
        "nan",
        "200",
        "nan",
        "-40000",
    ]
    assert repr_text.matrix_format(inf, 2e2, -nan, -4e4) == [
        "inf",
        "200",
        "nan",
        "-40000",
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
        fval=1.23456e-10,
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
        state=[],
    )
    return FMin(fm, 10, 3, 0.1)


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
    return FMin(fm, 100000, 200000, 0.01)


def test_html_fmin_good(fmin_good):
    # fmt: off
    assert fmin_good._repr_html_() == """<table>
    <tr>
        <td colspan="2" style="text-align:left" title="Minimum value of function"> FCN = 1.235e-10 </td>
        <td colspan="3" style="text-align:center" title="No. of function evaluations in last call and total number"> Nfcn = 10 </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:left" title="Estimated distance to minimum and goal"> EDM = 1.23e-10 (Goal: 0.0001) </td>
        <td colspan="3" style="text-align:center" title="No. of gradient evaluations in last call and total number"> Ngrad = 3 </td>
    </tr>
    <tr>
        <td style="text-align:center;{good}"> Valid Minimum </td>
        <td style="text-align:center;{good}"> Valid Parameters </td>
        <td colspan="3" style="text-align:center;{good}"> No Parameters at limit </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:center;{good}"> Below EDM threshold (goal x 10) </td>
        <td colspan="3" style="text-align:center;{good}"> Below call limit </td>
    </tr>
    <tr>
        <td style="text-align:center;{good}"> Hesse ok </td>
        <td style="text-align:center;{good}"> Has Covariance </td>
        <td style="text-align:center;{good}" title="Is covariance matrix accurate?"> Accurate </td>
        <td style="text-align:center;{good}" title="Is covariance matrix positive definite?"> Pos. def. </td>
        <td style="text-align:center;{good}" title="Was positive definiteness enforced by Minuit?"> Not forced </td>
    </tr>
</table>""".format(good=repr_html.good_style)
    # fmt: on


def test_html_fmin_bad(fmin_bad):
    # fmt: off
    assert fmin_bad._repr_html_() == """<table>
    <tr>
        <td colspan="2" style="text-align:left" title="Minimum value of function"> FCN = nan </td>
        <td colspan="3" style="text-align:center" title="No. of function evaluations in last call and total number"> Nfcn = 100000 </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:left" title="Estimated distance to minimum and goal"> EDM = 1.23e-10 (Goal: 1e-05) </td>
        <td colspan="3" style="text-align:center" title="No. of gradient evaluations in last call and total number"> Ngrad = 200000 </td>
    </tr>
    <tr>
        <td style="text-align:center;{bad}"> INVALID Minimum </td>
        <td style="text-align:center;{bad}"> INVALID Parameters </td>
        <td colspan="3" style="text-align:center;{warn}"> SOME Parameters at limit </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:center;{bad}"> ABOVE EDM threshold (goal x 10) </td>
        <td colspan="3" style="text-align:center;{bad}"> ABOVE call limit </td>
    </tr>
    <tr>
        <td style="text-align:center;{bad}"> Hesse FAILED </td>
        <td style="text-align:center;{bad}"> NO Covariance </td>
        <td style="text-align:center;{warn}" title="Is covariance matrix accurate?"> APPROXIMATE </td>
        <td style="text-align:center;{bad}" title="Is covariance matrix positive definite?"> NOT pos. def. </td>
        <td style="text-align:center;{bad}" title="Was positive definiteness enforced by Minuit?"> FORCED </td>
    </tr>
</table>""".format(bad=repr_html.bad_style, warn=repr_html.warn_style)
    # fmt: on


def test_html_params(minuit):
    # fmt: off
    assert minuit.init_params._repr_html_() == """<table>
    <tr>
        <td></td>
        <th title="Variable name"> Name </th>
        <th title="Value of parameter"> Value </th>
        <th title="Hesse error"> Hesse Error </th>
        <th title="Minos lower error"> Minos Error- </th>
        <th title="Minos upper error"> Minos Error+ </th>
        <th title="Lower limit of the parameter"> Limit- </th>
        <th title="Upper limit of the parameter"> Limit+ </th>
        <th title="Is the parameter fixed in the fit"> Fixed </th>
    </tr>
    <tr>
        <th> 0 </th>
        <td> x </td>
        <td> 0.0 </td>
        <td> 0.1 </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
    </tr>
    <tr>
        <th> 1 </th>
        <td> y </td>
        <td> 0.0 </td>
        <td> 0.1 </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
    </tr>
</table>"""

    assert minuit.params._repr_html_() == """<table>
    <tr>
        <td></td>
        <th title="Variable name"> Name </th>
        <th title="Value of parameter"> Value </th>
        <th title="Hesse error"> Hesse Error </th>
        <th title="Minos lower error"> Minos Error- </th>
        <th title="Minos upper error"> Minos Error+ </th>
        <th title="Lower limit of the parameter"> Limit- </th>
        <th title="Upper limit of the parameter"> Limit+ </th>
        <th title="Is the parameter fixed in the fit"> Fixed </th>
    </tr>
    <tr>
        <th> 0 </th>
        <td> x </td>
        <td> 2 </td>
        <td> 1 </td>
        <td> -1 </td>
        <td> 1 </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
    </tr>
    <tr>
        <th> 1 </th>
        <td> y </td>
        <td> 1.0 </td>
        <td> 0.5 </td>
        <td> -0.5 </td>
        <td> 0.5 </td>
        <td>  </td>
        <td>  </td>
        <td>  </td>
    </tr>
</table>"""
    # fmt: on


def test_html_params_with_limits():
    m = Minuit(f1, x=3, y=5)
    m.fixed["x"] = True
    m.errors = (0.2, 0.1)
    m.limits = ((0, None), (0, 10))
    # fmt: off
    assert m.init_params._repr_html_() == r"""<table>
    <tr>
        <td></td>
        <th title="Variable name"> Name </th>
        <th title="Value of parameter"> Value </th>
        <th title="Hesse error"> Hesse Error </th>
        <th title="Minos lower error"> Minos Error- </th>
        <th title="Minos upper error"> Minos Error+ </th>
        <th title="Lower limit of the parameter"> Limit- </th>
        <th title="Upper limit of the parameter"> Limit+ </th>
        <th title="Is the parameter fixed in the fit"> Fixed </th>
    </tr>
    <tr>
        <th> 0 </th>
        <td> x </td>
        <td> 3.0 </td>
        <td> 0.2 </td>
        <td>  </td>
        <td>  </td>
        <td> 0 </td>
        <td>  </td>
        <td> yes </td>
    </tr>
    <tr>
        <th> 1 </th>
        <td> y </td>
        <td> 5.0 </td>
        <td> 0.1 </td>
        <td>  </td>
        <td>  </td>
        <td> 0 </td>
        <td> 10 </td>
        <td>  </td>
    </tr>
</table>"""
    # fmt: on


def test_html_merrors(minuit):
    mes = minuit.merrors
    # fmt: off
    assert mes._repr_html_() == r"""<table>
    <tr>
        <td></td>
        <th colspan="2" style="text-align:center" title="Parameter name"> x </th>
        <th colspan="2" style="text-align:center" title="Parameter name"> y </th>
    </tr>
    <tr>
        <th title="Lower and upper minos error of the parameter"> Error </th>
        <td> -1 </td>
        <td> 1 </td>
        <td> -0.5 </td>
        <td> 0.5 </td>
    </tr>
    <tr>
        <th title="Validity of lower/upper minos error"> Valid </th>
        <td style="{good}"> True </td>
        <td style="{good}"> True </td>
        <td style="{good}"> True </td>
        <td style="{good}"> True </td>
    </tr>
    <tr>
        <th title="Did scan hit limit of any parameter?"> At Limit </th>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
    </tr>
    <tr>
        <th title="Did scan hit function call limit?"> Max FCN </th>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
    </tr>
    <tr>
        <th title="New minimum found when doing scan?"> New Min </th>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
        <td style="{good}"> False </td>
    </tr>
</table>""".format(good=repr_html.good_style)
    # fmt: on


def test_html_matrix():
    matrix = Matrix(["x", "y"], ((1.0, -0.0), (-0.0, 0.25)))
    # fmt: off
    assert matrix._repr_html_() == r"""<table>
    <tr>
        <td></td>
        <th> x </th>
        <th> y </th>
    </tr>
    <tr>
        <th> x </th>
        <td> 1.00 </td>
        <td style="background-color:rgb(250,250,250);color:black"> -0.00 </td>
    </tr>
    <tr>
        <th> y </th>
        <td style="background-color:rgb(250,250,250);color:black"> -0.00 </td>
        <td> 0.25 </td>
    </tr>
</table>"""
    # fmt: on


def test_text_fmin_good(fmin_good):
    assert str(fmin_good) == ref("fmin_good")


def test_text_fmin_bad(fmin_bad):
    assert str(fmin_bad) == ref("fmin_bad")


def test_text_params(minuit):
    assert str(minuit.params) == ref("params")


def test_text_params_with_limits():
    m = Minuit(
        f1,
        x=3,
        y=5,
    )
    m.fixed["x"] = True
    m.errors = (0.2, 0.1)
    m.limits = ((0, None), (0, 10))
    assert str(m.params) == ref("params_with_limits")


def test_text_merrors(minuit):
    assert str(minuit.merrors) == ref("merrors")


def test_text_matrix():
    matrix = Matrix(["x", "y"], ((1.0, -0.0), (-0.0, 0.25)))
    assert str(matrix) == ref("matrix")


def test_text_matrix_mini():
    matrix = Matrix(["x"], ((1.0,),))
    assert str(matrix) == ref("matrix_mini")


def test_text_matrix_large():
    matrix = Matrix(["x", "y", "z"], ((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    assert str(matrix) == ref("matrix_large")


def test_text_matrix_with_long_names():
    matrix = Matrix(["super-long-name", "x"], ((1.0, 0.1), (0.1, 1.0)))
    assert str(matrix) == ref("matrix_long_names")


def test_text_params_with_long_names():
    mps = Params(
        [
            Param(
                0,
                "super-long-name",
                0,
                0,
                False,
                False,
                False,
                False,
                False,
                None,
                None,
            )
        ],
        None,
    )
    assert str(mps) == ref("params_long_names")


def test_text_matrix_difficult_values():
    matrix = Matrix(("x", "y"), ((-1.23456, 0), (0, 0)))
    assert str(matrix) == ref("matrix_difficult_values")


def test_text_params_difficult_values():
    mps = Params(
        [
            Param(
                0,
                "x",
                -1.234567e-22,
                1.234567e-11,
                True,
                False,
                False,
                False,
                False,
                None,
                None,
            )
        ],
        None,
    )
    assert str(mps) == ref("params_difficult_values")