# flake8: noqa E501
from iminuit import Minuit
from iminuit.util import Params, Param, Matrix
from iminuit import repr_html, repr_text
import pytest
from xml.etree import ElementTree

nan = float("nan")
inf = float("infinity")


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


def test_html_tag():
    tag = repr_html.tag("foo", "bar", baz="hi", xyzzy="2")
    assert tag == '<foo baz="hi" xyzzy="2"> bar </foo>'
    assert repr_html.tag("foo") == "<foo> </foo>"


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


def test_html_fmin(minuit):
    fmin = minuit.fmin
    assert (
        fmin._repr_html_()
        == r"""<table>
<tr> <td colspan="2" style="text-align:center" title="Minimum value of function"> FCN = 1 </td> <td colspan="3" title="No. of calls in last algorithm and total number of calls"> Ncalls = 32 (56 total) </td> </tr>
<tr> <td colspan="2" style="text-align:center" title="Estimated distance to minimum and target threshold"> EDM = %.3g (Goal: 2e-07) </td> <td colspan="3" title="Increase in FCN which corresponds to 1 standard deviation"> up = 1.0 </td> </tr>
<tr> <td title="Validity of the migrad call"> Valid Min. </td> <td title="Validity of parameters"> Valid Param. </td> <td title="Is EDM above goal EDM?"> Above EDM </td> <td colspan="2" title="Did last migrad call reach max call limit?"> Reached call limit </td> </tr>
<tr> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> False </td> <td colspan="2" style="background-color:#92CCA6;"> False </td> </tr>
<tr> <td title="Did Hesse fail?"> Hesse failed </td> <td title="Has covariance matrix"> Has cov. </td> <td title="Is covariance matrix accurate?"> Accurate </td> <td title="Is covariance matrix positive definite?"> Pos. def. </td> <td title="Was positive definiteness enforced by Minuit?"> Forced </td> </tr>
<tr> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> False </td> </tr>
</table>
"""
        % fmin.edm
    )


def test_html_params(minuit):
    assert (
        r"""<table>
<tr style="background-color:#F4F4F4;"> <td/> <th title="Variable name"> Name </th> <th title="Value of parameter"> Value </th> <th title="Hesse error"> Hesse Error </th> <th title="Minos lower error"> Minos Error- </th> <th title="Minos upper error"> Minos Error+ </th> <th title="Lower limit of the parameter"> Limit- </th> <th title="Upper limit of the parameter"> Limit+ </th> <th title="Is the parameter fixed in the fit"> Fixed </th> </tr>
<tr style="background-color:#FFFFFF;"> <th> 0 </th> <td> x </td> <td> 0.0 </td> <td> 0.1 </td> <td>  </td> <td>  </td> <td>  </td> <td>  </td> <td>  </td> </tr>
<tr style="background-color:#F4F4F4;"> <th> 1 </th> <td> y </td> <td> 0.0 </td> <td> 0.1 </td> <td>  </td> <td>  </td> <td>  </td> <td>  </td> <td>  </td> </tr>
</table>
"""
        == minuit.init_params._repr_html_()
    )

    assert (
        """<table>
<tr style="background-color:#F4F4F4;"> <td/> <th title="Variable name"> Name </th> <th title="Value of parameter"> Value </th> <th title="Hesse error"> Hesse Error </th> <th title="Minos lower error"> Minos Error- </th> <th title="Minos upper error"> Minos Error+ </th> <th title="Lower limit of the parameter"> Limit- </th> <th title="Upper limit of the parameter"> Limit+ </th> <th title="Is the parameter fixed in the fit"> Fixed </th> </tr>
<tr style="background-color:#FFFFFF;"> <th> 0 </th> <td> x </td> <td> 2 </td> <td> 1 </td> <td> -1 </td> <td> 1 </td> <td>  </td> <td>  </td> <td>  </td> </tr>
<tr style="background-color:#F4F4F4;"> <th> 1 </th> <td> y </td> <td> 1.0 </td> <td> 0.5 </td> <td> -0.5 </td> <td> 0.5 </td> <td>  </td> <td>  </td> <td>  </td> </tr>
</table>
"""
        == minuit.params._repr_html_()
    )


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
    assert (
        r"""<table>
<tr style="background-color:#F4F4F4;"> <td/> <th title="Variable name"> Name </th> <th title="Value of parameter"> Value </th> <th title="Hesse error"> Hesse Error </th> <th title="Minos lower error"> Minos Error- </th> <th title="Minos upper error"> Minos Error+ </th> <th title="Lower limit of the parameter"> Limit- </th> <th title="Upper limit of the parameter"> Limit+ </th> <th title="Is the parameter fixed in the fit"> Fixed </th> </tr>
<tr style="background-color:#FFFFFF;"> <th> 0 </th> <td> x </td> <td> 3.0 </td> <td> 0.2 </td> <td>  </td> <td>  </td> <td> 0 </td> <td>  </td> <td> yes </td> </tr>
<tr style="background-color:#F4F4F4;"> <th> 1 </th> <td> y </td> <td> 5.0 </td> <td> 0.1 </td> <td>  </td> <td>  </td> <td> 0 </td> <td> 10 </td> <td>  </td> </tr>
</table>
"""
        == m.init_params._repr_html_()
    )


def test_html_minos(minuit):
    mes = minuit.merrors
    assert (
        mes._repr_html_()
        == r"""<table>
<tr> <td/> <th colspan="2" style="text-align:center" title="Parameter name"> x </th> <th colspan="2" style="text-align:center" title="Parameter name"> y </th> </tr>
<tr> <th title="Lower and upper minos error of the parameter"> Error </th> <td> -1 </td> <td> 1 </td> <td> -0.5 </td> <td> 0.5 </td> </tr>
<tr> <th title="Validity of lower/upper minos error"> Valid </th> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> True </td> <td style="background-color:#92CCA6;"> True </td> </tr>
<tr> <th title="Did scan hit limit of any parameter?"> At Limit </th> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> </tr>
<tr> <th title="Did scan hit function call limit?"> Max FCN </th> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> </tr>
<tr> <th title="New minimum found when doing scan?"> New Min </th> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> <td style="background-color:#92CCA6;"> False </td> </tr>
</table>
"""
    )


def test_html_matrix():
    matrix = Matrix(["x", "y"], ((1.0, -0.0), (-0.0, 0.25)))
    assert (
        matrix._repr_html_()
        == r"""<table>
<tr> <td/> <th> x </th> <th> y </th> </tr>
<tr> <th> x </th> <td> 1.00 </td> <td style="background-color:rgb(250,250,250)"> -0.00 </td> </tr>
<tr> <th> y </th> <td style="background-color:rgb(250,250,250)"> -0.00 </td> <td> 0.25 </td> </tr>
</table>
"""
    )


def test_text_fmin(minuit):
    fmin = minuit.fmin
    assert (
        str(fmin)
        == r"""------------------------------------------------------------------
| FCN = 1                       |      Ncalls=32 (56 total)      |
| EDM = %.3g (Goal: 2e-07)  |            up = 1.0            |
------------------------------------------------------------------
|  Valid Min.   | Valid Param.  | Above EDM | Reached call limit |
------------------------------------------------------------------
|     True      |     True      |   False   |       False        |
------------------------------------------------------------------
| Hesse failed  |   Has cov.    | Accurate  | Pos. def. | Forced |
------------------------------------------------------------------
|     False     |     True      |   True    |   True    | False  |
------------------------------------------------------------------"""
        % fmin.edm
    )


def test_text_params(minuit):
    assert r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |    0.0    |    0.1    |            |            |         |         |       |
| 1 | y    |    0.0    |    0.1    |            |            |         |         |       |
------------------------------------------------------------------------------------------""" == str(
        minuit.init_params
    )

    assert r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |     2     |     1     |     -1     |     1      |         |         |       |
| 1 | y    |    1.0    |    0.5    |    -0.5    |    0.5     |         |         |       |
------------------------------------------------------------------------------------------""" == str(
        minuit.params
    )


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
    assert r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |    3.0    |    0.2    |            |            |    0    |         |  yes  |
| 1 | y    |    5.0    |    0.1    |            |            |    0    |   10    |       |
------------------------------------------------------------------------------------------""" == str(
        m.init_params
    )


def test_text_minos(minuit):
    assert (
        str(minuit.minos())
        == r"""------------------------------------------------------------
|          |           x           |           y           |
------------------------------------------------------------
|  Error   |    -1     |     1     |   -0.5    |    0.5    |
|  Valid   |   True    |   True    |   True    |   True    |
| At Limit |   False   |   False   |   False   |   False   |
| Max FCN  |   False   |   False   |   False   |   False   |
| New Min  |   False   |   False   |   False   |   False   |
------------------------------------------------------------"""
    )


def test_text_matrix():
    matrix = Matrix(["x", "y"], ((1.0, -0.0), (-0.0, 0.25)))
    assert r"""-------------------
|   |     x     y |
-------------------
| x |  1.00 -0.00 |
| y | -0.00  0.25 |
-------------------""" == str(
        matrix
    )


def test_text_with_long_names():

    matrix = Matrix(["super-long-name", "x"], ((1.0, 0.1), (0.1, 1.0)))
    assert r"""-----------------------------------------------------
|                 | super-long-name               x |
-----------------------------------------------------
| super-long-name |             1.0             0.1 |
|               x |             0.1             1.0 |
-----------------------------------------------------""" == str(
        matrix
    )

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
    assert r"""-----------------------------------------------------------------------------------------------------
|   | Name            |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
-----------------------------------------------------------------------------------------------------
| 0 | super-long-name |     0     |     0     |            |            |         |         |       |
-----------------------------------------------------------------------------------------------------""" == str(
        mps
    )


def test_console_frontend_with_difficult_values():
    matrix = Matrix(("x", "y"), ((-1.23456, 0), (0, 0)))
    assert r"""---------------------
|   |      x      y |
---------------------
| x | -1.235  0.000 |
| y |  0.000  0.000 |
---------------------""" == str(
        matrix
    )

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

    assert r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |    -0     | 0.012e-9  |            |            |         |         | CONST |
------------------------------------------------------------------------------------------""" == str(
        mps
    )
