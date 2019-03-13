from __future__ import (absolute_import, division, print_function)
# Here we don't import unicode_literals, because otherwise we cannot write
# in "\usepackage" and similar in the string literals below.
# The problem is an inconsistency regarding the effect of
# from __future__ import unicode_literals in Python 2 and 3, which is
# explained here:
# https://stackoverflow.com/questions/7602171/unicode-error-unicodeescape-codec-cant-decode-bytes-string-with-u
# We want the same code to work in Python 2 and 3 and chose this solution.
from iminuit import Minuit
from iminuit.util import Params, Param, Matrix
from iminuit import repr_html, repr_text
import pytest


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


def test_html_tag():
    s = repr_html.Html()
    with repr_html.Tag("foo", s):
        s += "bar"
    assert str(s) == "<foo>\nbar\n</foo>\n"


def test_format_numbers():
    assert repr_text.format_numbers(1.2567, 0.1234) == ("1.26", "0.12")
    assert repr_text.format_numbers(1.2567e3, 0.1234e3) == ("1260", "120")
    assert repr_text.format_numbers(1.2567e4, 0.1234e4) == ("1.26E4", "0.12E4")
    assert repr_text.format_numbers(1.2567e-1, 0.1234e-1) == ("0.126", "0.012")
    assert repr_text.format_numbers(1.2567e-2, 0.1234e-2) == ("1.26E-2", "0.12E-2")
    assert repr_text.format_numbers(0, 1, -1) == ("0.0", "1.0", "-1.0")
    assert repr_text.format_numbers(2, -1, 1) == ("2.0", "-1.0", "1.0")
    assert repr_text.format_numbers(2.01, -1.01, 1.01) == ("2.0", "-1.0", "1.0")
    assert repr_text.format_numbers(1.999, -0.999, 0.999) == ("2.0", "-1.0", "1.0")
    assert repr_text.format_numbers(1, 0.5, -0.5) == ("1.0", "0.5", "-0.5")
    assert repr_text.format_numbers(1.0, 1e-10) == ("1.000", "0.000")
    assert repr_text.format_numbers(1.234567e11, -1.234567e-11) == ("1.235E11", "-0.000E11")


@pytest.fixture
def minuit():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    m.minos()
    return m


def test_html_fmin(minuit):
    fmin = minuit.get_fmin()
    assert r"""<table>
<tr>
<td colspan="2" title="Minimum value of function">
FCN = 1
</td>
<td colspan="3" align="center" title="No. of calls in last algorithm and total number of calls">
Ncalls = 24 (67 total)
</td>
</tr>
<tr>
<td colspan="2" title="Estimated distance to minimum and target threshold">
EDM = %.3G (Goal: 1E-08)
</td>
<td colspan="3" align="center" title="Increase in FCN which corresponds to 1 standard deviation">
up = 1.0
</td>
</tr>
<tr>
<td align="center" title="Validity of the migrad call">
Valid Min.
</td>
<td align="center" title="Validity of parameters">
Valid Param.
</td>
<td align="center" title="Is EDM above goal EDM?">
Above EDM
</td>
<td colspan="2" align="center" title="Did last migrad call reach max call limit?">
Reached call limit
</td>
</tr>
<tr>
<td align="center" style="background-color:#92CCA6;">
True
</td>
<td align="center" style="background-color:#92CCA6;">
True
</td>
<td align="center" style="background-color:#92CCA6;">
False
</td>
<td colspan="2" align="center" style="background-color:#92CCA6;">
False
</td>
</tr>
<tr>
<td align="center" title="Did Hesse fail?">
Hesse failed
</td>
<td align="center" title="Has covariance matrix">
Has cov.
</td>
<td align="center" title="Is covariance matrix accurate?">
Accurate
</td>
<td align="center" title="Is covariance matrix positive definite?">
Pos. def.
</td>
<td align="center" title="Was positive definiteness enforced by Minuit?">
Forced
</td>
</tr>
<tr>
<td align="center" style="background-color:#92CCA6;">
False
</td>
<td align="center" style="background-color:#92CCA6;">
True
</td>
<td align="center" style="background-color:#92CCA6;">
True
</td>
<td align="center" style="background-color:#92CCA6;">
True
</td>
<td align="center" style="background-color:#92CCA6;">
False
</td>
</tr>
</table>
""" % fmin.edm == fmin._repr_html_()


def test_html_params(minuit):
    assert r"""<table>
<tr style="background-color:#F4F4F4;">
<td/>
<th title="Variable name">
Name
</th>
<th title="Value of parameter">
Value
</th>
<th title="Hesse error">
Hesse Error
</th>
<th title="Minos lower error">
Minos Error-
</th>
<th title="Minos upper error">
Minos Error+
</th>
<th title="Lower limit of the parameter">
Limit-
</th>
<th title="Upper limit of the parameter">
Limit+
</th>
<th title="Is the parameter fixed in the fit">
Fixed
</th>
</tr>
<tr style="background-color:#FFFFFF;">
<td>
0
</td>
<td>
x
</td>
<td>
0.0
</td>
<td>
1.0
</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
<tr style="background-color:#F4F4F4;">
<td>
1
</td>
<td>
y
</td>
<td>
0.0
</td>
<td>
1.0
</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
</table>
""" == minuit.get_initial_param_states()._repr_html_()

    assert """<table>
<tr style="background-color:#F4F4F4;">
<td/>
<th title="Variable name">
Name
</th>
<th title="Value of parameter">
Value
</th>
<th title="Hesse error">
Hesse Error
</th>
<th title="Minos lower error">
Minos Error-
</th>
<th title="Minos upper error">
Minos Error+
</th>
<th title="Lower limit of the parameter">
Limit-
</th>
<th title="Upper limit of the parameter">
Limit+
</th>
<th title="Is the parameter fixed in the fit">
Fixed
</th>
</tr>
<tr style="background-color:#FFFFFF;">
<td>
0
</td>
<td>
x
</td>
<td>
2.0
</td>
<td>
1.0
</td>
<td>
-1.0
</td>
<td>
1.0
</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
<tr style="background-color:#F4F4F4;">
<td>
1
</td>
<td>
y
</td>
<td>
1.0
</td>
<td>
0.5
</td>
<td>
-0.5
</td>
<td>
0.5
</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
</table>
""" == minuit.get_param_states()._repr_html_()


def test_html_params_with_limits():
    m = Minuit(f1, x=3, y=5, fix_x=True,
               error_x=0.2, error_y=0.1,
               limit_x=(0, None), limit_y=(0, 10),
               errordef=1, print_level=0)
    assert r"""<table>
<tr style="background-color:#F4F4F4;">
<td/>
<th title="Variable name">
Name
</th>
<th title="Value of parameter">
Value
</th>
<th title="Hesse error">
Hesse Error
</th>
<th title="Minos lower error">
Minos Error-
</th>
<th title="Minos upper error">
Minos Error+
</th>
<th title="Lower limit of the parameter">
Limit-
</th>
<th title="Upper limit of the parameter">
Limit+
</th>
<th title="Is the parameter fixed in the fit">
Fixed
</th>
</tr>
<tr style="background-color:#FFFFFF;">
<td>
0
</td>
<td>
x
</td>
<td>
3.00
</td>
<td>
0.20
</td>
<td>

</td>
<td>

</td>
<td>
0
</td>
<td>

</td>
<td>
yes
</td>
</tr>
<tr style="background-color:#F4F4F4;">
<td>
1
</td>
<td>
y
</td>
<td>
5.00
</td>
<td>
0.10
</td>
<td>

</td>
<td>

</td>
<td>
0
</td>
<td>
10
</td>
<td>

</td>
</tr>
</table>
""" == m.get_initial_param_states()._repr_html_()


def test_html_minos(minuit):
    mes = minuit.merrors_struct
    assert r"""<table>
<tr>
<th title="Parameter name">
x
</th>
<td colspan="2" style="background-color:#92CCA6;" align="center">
Valid
</td>
</tr>
<tr>
<td title="Lower and upper minos error of the parameter">
Error
</td>
<td>
-1.0
</td>
<td>
1.0
</td>
</tr>
<tr>
<td title="Validity of lower/upper minos error">
Valid
</td>
<td style="background-color:#92CCA6;">
True
</td>
<td style="background-color:#92CCA6;">
True
</td>
</tr>
<tr>
<td title="Did scan hit limit of any parameter?">
At Limit
</td>
<td style="background-color:#92CCA6;">
False
</td>
<td style="background-color:#92CCA6;">
False
</td>
</tr>
<tr>
<td title="Did scan hit function call limit?">
Max FCN
</td>
<td style="background-color:#92CCA6;">
False
</td>
<td style="background-color:#92CCA6;">
False
</td>
</tr>
<tr>
<td title="New minimum found when doing scan?">
New Min
</td>
<td style="background-color:#92CCA6;">
False
</td>
<td style="background-color:#92CCA6;">
False
</td>
</tr>
</table>

<table>
<tr>
<th title="Parameter name">
y
</th>
<td colspan="2" style="background-color:#92CCA6;" align="center">
Valid
</td>
</tr>
<tr>
<td title="Lower and upper minos error of the parameter">
Error
</td>
<td>
-0.5
</td>
<td>
0.5
</td>
</tr>
<tr>
<td title="Validity of lower/upper minos error">
Valid
</td>
<td style="background-color:#92CCA6;">
True
</td>
<td style="background-color:#92CCA6;">
True
</td>
</tr>
<tr>
<td title="Did scan hit limit of any parameter?">
At Limit
</td>
<td style="background-color:#92CCA6;">
False
</td>
<td style="background-color:#92CCA6;">
False
</td>
</tr>
<tr>
<td title="Did scan hit function call limit?">
Max FCN
</td>
<td style="background-color:#92CCA6;">
False
</td>
<td style="background-color:#92CCA6;">
False
</td>
</tr>
<tr>
<td title="New minimum found when doing scan?">
New Min
</td>
<td style="background-color:#92CCA6;">
False
</td>
<td style="background-color:#92CCA6;">
False
</td>
</tr>
</table>
""" == mes._repr_html_()


def test_html_matrix(minuit):
    assert r"""<table>
<tr>
<td/>

<th>
x
</th>
<th>
y
</th>
</tr>
<tr>
<th>
x
</th>
<td>
1.000
</td>
<td style="background-color:rgb(163,254,186)">
0.000
</td>
</tr>
<tr>
<th>
y
</th>
<td style="background-color:rgb(163,254,186)">
0.000
</td>
<td>
0.250
</td>
</tr>
</table>
""" == minuit.matrix()._repr_html_()


def test_text_fmin(minuit):
    assert \
r"""------------------------------------------------------------------
| FCN = 1                       |      Ncalls=24 (67 total)      |
| EDM = %.3G (Goal: 1E-08)  |            up = 1.0            |
------------------------------------------------------------------
|  Valid Min.   | Valid Param.  | Above EDM | Reached call limit |
------------------------------------------------------------------
|     True      |     True      |   False   |       False        |
------------------------------------------------------------------
| Hesse failed  |   Has cov.    | Accurate  | Pos. def. | Forced |
------------------------------------------------------------------
|     False     |     True      |   True    |   True    | False  |
------------------------------------------------------------------""" \
    % minuit.get_fmin().edm == str(minuit.get_fmin())


def test_text_params(minuit):
    assert \
r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |    0.0    |    1.0    |            |            |         |         |       |
| 1 | y    |    0.0    |    1.0    |            |            |         |         |       |
------------------------------------------------------------------------------------------""" \
    == str(minuit.get_initial_param_states())

    assert \
r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |    2.0    |    1.0    |    -1.0    |    1.0     |         |         |       |
| 1 | y    |    1.0    |    0.5    |    -0.5    |    0.5     |         |         |       |
------------------------------------------------------------------------------------------""" \
    == str(minuit.get_param_states())


def test_text_params_with_limits():
    m = Minuit(f1, x=3, y=5, fix_x=True,
               error_x=0.2, error_y=0.1,
               limit_x=(0, None), limit_y=(0, 10),
               errordef=1)
    assert \
r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |   3.00    |   0.20    |            |            |    0    |         |  yes  |
| 1 | y    |   5.00    |   0.10    |            |            |    0    |   10    |       |
------------------------------------------------------------------------------------------""" \
    == str(m.get_initial_param_states())


def test_text_minos(minuit):
    assert \
r"""-------------------------------------------------
|        x        |            Valid            |
-------------------------------------------------
|      Error      |     -1.0     |     1.0      |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------
-------------------------------------------------
|        y        |            Valid            |
-------------------------------------------------
|      Error      |     -0.5     |     0.5      |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------""" \
    == str(minuit.minos())


def test_text_matrix(minuit):
    assert \
r"""---------------------
|   |      x      y |
---------------------
| x |  1.000  0.000 |
| y |  0.000  0.250 |
---------------------""" \
    == str(minuit.matrix())


def test_text_with_long_names():

    matrix = Matrix(["super-long-name", "x"], ((1.0, 0.1), (0.1, 1.0)))
    assert \
r"""-----------------------------------------------------
|                 | super-long-name               x |
-----------------------------------------------------
| super-long-name |            1.00            0.10 |
|               x |            0.10            1.00 |
-----------------------------------------------------""" \
    == str(matrix)

    mps = Params([Param(0, "super-long-name", 0, 0, False, False, False, False, False, None, None)], None)
    assert \
r"""-----------------------------------------------------------------------------------------------------
|   | Name            |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
-----------------------------------------------------------------------------------------------------
| 0 | super-long-name |     0     |     0     |            |            |         |         |       |
-----------------------------------------------------------------------------------------------------""" \
    == str(mps)


def test_console_frontend_with_difficult_values():
    matrix = Matrix(("x", "y"), ((-1.23456, 0), (0, 0)))
    assert \
r"""---------------------
|   |      x      y |
---------------------
| x |   -1.2    0.0 |
| y |    0.0    0.0 |
---------------------""" \
    == str(matrix)

    mps = Params([Param(0, "x",  -1.234567e-22, 1.234567e-11, True, False, False, False, False, None, None)], None)

    assert \
r"""------------------------------------------------------------------------------------------
|   | Name |   Value   | Hesse Err | Minos Err- | Minos Err+ | Limit-  | Limit+  | Fixed |
------------------------------------------------------------------------------------------
| 0 | x    |-0.000E-11 | 1.235E-11 |            |            |         |         | CONST |
------------------------------------------------------------------------------------------""" \
    == str(mps)
