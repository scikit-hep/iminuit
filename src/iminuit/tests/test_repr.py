# flake8: noqa E501
from iminuit import Minuit
from iminuit.util import Params, Param, Matrix, FMin
from iminuit import repr_html, repr_text
import pytest

nan = float("nan")
inf = float("infinity")


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


def test_html_tag():
    tag = repr_html.tag

    def stag(*args, **kwargs):
        return repr_html.to_str(tag(*args, **kwargs))

    # fmt: off
    assert stag("foo", "bar", baz="hi", xyzzy="2") == '<foo baz="hi" xyzzy="2"> bar </foo>'
    assert stag("foo") == """<foo></foo>"""
    assert tag("foo", tag("bar", "baz")) == ["<foo>", ["<bar> baz </bar>"], "</foo>"]
    assert stag("foo", tag("bar", "baz")) == """<foo>
    <bar> baz </bar>
</foo>"""
    # fmt: on


def framed(s):
    return "\n" + str(s) + "\n"


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
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    m.minos()
    return m


@pytest.fixture
def fmin_good():
    return FMin(
        fval=1.23456e-10,
        edm=1.23456e-10,
        tolerance=0.1,
        nfcn=10,
        nfcn_total=20,
        ngrad=3,
        ngrad_total=3,
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
    )


@pytest.fixture
def fmin_bad():
    return FMin(
        fval=nan,
        edm=1.23456e-10,
        tolerance=0,
        nfcn=100000,
        nfcn_total=200000,
        ngrad=100000,
        ngrad_total=200000,
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
    )


def test_html_fmin_good(fmin_good):
    # fmt: off
    assert fmin_good._repr_html_() == """<table>
    <tr>
        <td colspan="2" style="text-align:left" title="Minimum value of function"> FCN = 1.235e-10 </td>
        <td colspan="3" style="text-align:center" title="No. of function evaluations in last call and total number"> Nfcn = 10 (20 total) </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:left" title="Estimated distance to minimum and goal"> EDM = 1.23e-10 (Goal: 0.0001) </td>
        <td colspan="3" style="text-align:center" title="No. of gradient evaluations in last call and total number"> Ngrad = 3 (3 total) </td>
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
        <td colspan="3" style="text-align:center" title="No. of function evaluations in last call and total number"> Nfcn = 100000 (200000 total) </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:left" title="Estimated distance to minimum and goal"> EDM = 1.23e-10 (Goal: 1.19e-10) </td>
        <td colspan="3" style="text-align:center" title="No. of gradient evaluations in last call and total number"> Ngrad = 100000 (200000 total) </td>
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
    m = Minuit(
        f1,
        x=3,
        y=5,
        fix_x=True,
        error_x=0.2,
        error_y=0.1,
        limit_x=(0, None),
        limit_y=(0, 10),
        errordef=1,
        print_level=0,
    )
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
    # fmt: off
    assert framed(fmin_good) == r"""
┌──────────────────────────────────┬──────────────────────────────────────┐
│ FCN = 1.235e-10                  │         Nfcn = 10 (20 total)         │
│ EDM = 1.23e-10 (Goal: 0.0001)    │         Ngrad = 3 (3 total)          │
├───────────────┬──────────────────┼──────────────────────────────────────┤
│ Valid Minimum │ Valid Parameters │        No Parameters at limit        │
├───────────────┴──────────────────┼──────────────────────────────────────┤
│ Below EDM threshold (goal x 10)  │           Below call limit           │
├───────────────┬──────────────────┼───────────┬─────────────┬────────────┤
│   Hesse ok    │  Has Covariance  │ Accurate  │  Pos. def.  │ Not forced │
└───────────────┴──────────────────┴───────────┴─────────────┴────────────┘
"""
    # fmt: on


def test_text_fmin_bad(fmin_bad):
    # fmt: off
    assert framed(fmin_bad) == r"""
┌──────────────────────────────────┬──────────────────────────────────────┐
│ FCN = nan                        │     Nfcn = 100000 (200000 total)     │
│ EDM = 1.23e-10 (Goal: 1.19e-10)  │    Ngrad = 100000 (200000 total)     │
├───────────────┬──────────────────┼──────────────────────────────────────┤
│INVALID Minimum│INVALID Parameters│       SOME Parameters at limit       │
├───────────────┴──────────────────┼──────────────────────────────────────┤
│ ABOVE EDM threshold (goal x 10)  │           ABOVE call limit           │
├───────────────┬──────────────────┼───────────┬─────────────┬────────────┤
│ Hesse FAILED  │  NO Covariance   │APPROXIMATE│NOT pos. def.│   FORCED   │
└───────────────┴──────────────────┴───────────┴─────────────┴────────────┘
"""
    # fmt: on


def test_text_params(minuit):
    # fmt: off
    assert framed(minuit.params) == """
┌───┬──────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼──────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ x    │     2     │     1     │     -1     │     1      │         │         │       │
│ 1 │ y    │    1.0    │    0.5    │    -0.5    │    0.5     │         │         │       │
└───┴──────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
"""
    # fmt: on


def test_text_params_with_limits():
    m = Minuit(
        f1,
        x=3,
        y=5,
        fix_x=True,
        error_x=0.2,
        error_y=0.1,
        limit_x=(0, None),
        limit_y=(0, 10),
        errordef=1,
    )
    # fmt: off
    assert framed(m.params) == """
┌───┬──────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼──────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ x    │    3.0    │    0.2    │            │            │    0    │         │  yes  │
│ 1 │ y    │    5.0    │    0.1    │            │            │    0    │   10    │       │
└───┴──────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
"""
    # fmt: on


def test_text_merrors(minuit):
    # fmt: off
    assert framed(minuit.minos()) == r"""
┌──────────┬───────────────────────┬───────────────────────┐
│          │           x           │           y           │
├──────────┼───────────┬───────────┼───────────┬───────────┤
│  Error   │    -1     │     1     │   -0.5    │    0.5    │
│  Valid   │   True    │   True    │   True    │   True    │
│ At Limit │   False   │   False   │   False   │   False   │
│ Max FCN  │   False   │   False   │   False   │   False   │
│ New Min  │   False   │   False   │   False   │   False   │
└──────────┴───────────┴───────────┴───────────┴───────────┘
"""
    # fmt: on


def test_text_matrix():
    matrix = Matrix(["x", "y"], ((1.0, -0.0), (-0.0, 0.25)))
    # fmt: off
    assert framed(matrix) == r"""
┌───┬─────────────┐
│   │     x     y │
├───┼─────────────┤
│ x │  1.00 -0.00 │
│ y │ -0.00  0.25 │
└───┴─────────────┘
"""
    # fmt: on


def test_text_matrix_mini():
    matrix = Matrix(["x"], ((1.0,),))
    # fmt: off
    assert framed(matrix) == r"""
┌───┬───┐
│   │ x │
├───┼───┤
│ x │ 1 │
└───┴───┘
"""
    # fmt: on


def test_text_matrix_large():
    matrix = Matrix(["x", "y", "z"], ((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    # fmt: off
    assert framed(matrix) == r"""
┌───┬───────┐
│   │ x y z │
├───┼───────┤
│ x │ 1 2 3 │
│ y │ 4 5 6 │
│ z │ 7 8 9 │
└───┴───────┘
"""
    # fmt: on


def test_text_matrix_with_long_names():

    matrix = Matrix(["super-long-name", "x"], ((1.0, 0.1), (0.1, 1.0)))
    # fmt: off
    assert framed(matrix) == r"""
┌─────────────────┬─────────────────────────────────┐
│                 │ super-long-name               x │
├─────────────────┼─────────────────────────────────┤
│ super-long-name │             1.0             0.1 │
│               x │             0.1             1.0 │
└─────────────────┴─────────────────────────────────┘
"""

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
    assert framed(mps) == r"""
┌───┬─────────────────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name            │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼─────────────────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ super-long-name │     0     │     0     │            │            │         │         │       │
└───┴─────────────────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
"""
    # fmt: on


def test_console_frontend_with_difficult_values():
    matrix = Matrix(("x", "y"), ((-1.23456, 0), (0, 0)))
    # fmt: off
    assert framed(matrix) == r"""
┌───┬───────────────┐
│   │      x      y │
├───┼───────────────┤
│ x │ -1.235  0.000 │
│ y │  0.000  0.000 │
└───┴───────────────┘
"""
    # fmt: on

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

    # fmt: off
    assert framed(mps) == r"""
┌───┬──────┬───────────┬───────────┬────────────┬────────────┬─────────┬─────────┬───────┐
│   │ Name │   Value   │ Hesse Err │ Minos Err- │ Minos Err+ │ Limit-  │ Limit+  │ Fixed │
├───┼──────┼───────────┼───────────┼────────────┼────────────┼─────────┼─────────┼───────┤
│ 0 │ x    │    -0     │ 0.012e-9  │            │            │         │         │ CONST │
└───┴──────┴───────────┴───────────┴────────────┴────────────┴─────────┴─────────┴───────┘
"""
    # fmt: on
