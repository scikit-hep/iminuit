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
from iminuit import repr_html
import pytest


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


def test_html_tag():
    s = repr_html.Html()
    with repr_html.Tag("foo", s):
        s += "bar"
    assert str(s) == "<foo>\nbar\n</foo>\n"


@pytest.fixture
def minuit():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=0)
    m.tol = 1e-4
    m.migrad()
    m.hesse()
    m.minos()
    return m


def format_html(x):
    lines = x.split("\n")
    is_td_context = False
    s = ""
    for line in lines:
        if not line: continue
        if line.startswith("<td") and not line.endswith("/>"):
            is_td_context = True
        elif line == "</td>":
            is_td_context = False
        delim = "" if is_td_context else "\n"
        s += line + delim
    return s


def test_html_params(minuit):
    assert format_html(minuit.get_initial_param_states()._repr_html_()) == \
r"""<table>
<tr>
<td/>
<td title="Variable name">Name</td>
<td title="Value of parameter">Value</td>
<td title="Hesse error">Hesse Error</td>
<td title="Minos lower error">Minos Error-</td>
<td title="Minos upper error">Minos Error+</td>
<td title="Lower limit of the parameter">Limit-</td>
<td title="Upper limit of the parameter">Limit+</td>
<td title="Is the parameter fixed in the fit">Fixed</td>
</tr>
<tr>
<td>0</td>
<td>x</td>
<td>0</td>
<td>1</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>1</td>
<td>y</td>
<td>0</td>
<td>1</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</table>
"""

    assert format_html(minuit.get_param_states()._repr_html_()) == \
"""<table>
<tr>
<td/>
<td title="Variable name">Name</td>
<td title="Value of parameter">Value</td>
<td title="Hesse error">Hesse Error</td>
<td title="Minos lower error">Minos Error-</td>
<td title="Minos upper error">Minos Error+</td>
<td title="Lower limit of the parameter">Limit-</td>
<td title="Upper limit of the parameter">Limit+</td>
<td title="Is the parameter fixed in the fit">Fixed</td>
</tr>
<tr>
<td>0</td>
<td>x</td>
<td>2</td>
<td>1</td>
<td>-1</td>
<td>1</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>1</td>
<td>y</td>
<td>1</td>
<td>0.5</td>
<td>-0.5</td>
<td>0.5</td>
<td></td>
<td></td>
<td></td>
</tr>
</table>
"""


def test_html_fmin(minuit):
    fmin = minuit.get_fmin()
    assert format_html(fmin._repr_html_()) == \
r"""<table>
<tr>
<td title="Minimum value of function">FCN = 1</td>
<td title="Total number of call to FCN so far">TOTAL NCALL = 67</td>
<td title="Number of call in last migrad">NCALLS = 24</td>
</tr>
<tr>
<td title="Estimated distance to minimum">EDM = %.3g</td>
<td title="Maximum EDM definition of convergence">GOAL EDM = 1e-08</td>
<td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">UP = 1.0</td>
</tr>
</table>
<table>
<tr>
<td align="center" title="Validity of the migrad call">Valid</td>
<td align="center" title="Validity of parameters">Valid Param</td>
<td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
<td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
<td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
</tr>
<tr>
<td align="center" style="background-color:#92CCA6">True</td>
<td align="center" style="background-color:#92CCA6">True</td>
<td align="center" style="background-color:#92CCA6">True</td>
<td align="center" style="background-color:#92CCA6">True</td>
<td align="center" style="background-color:#92CCA6">False</td>
</tr>
<tr>
<td align="center" title="Was last hesse call fail?">Hesse Fail</td>
<td align="center" title="Validity of covariance">HasCov</td>
<td align="center" title="Is EDM above goal EDM?">Above EDM</td>
<td align="center"></td>
<td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
</tr>
<tr>
<td align="center" style="background-color:#92CCA6">False</td>
<td align="center" style="background-color:#92CCA6">True</td>
<td align="center" style="background-color:#92CCA6">False</td>
<td align="center"></td>
<td align="center" style="background-color:#92CCA6">False</td>
</tr>
</table>
""" % fmin.edm


def test_html_minos(minuit):
    mes = minuit.merrors_struct
    assert format_html(mes._repr_html_()) == \
r"""<span>
Minos status for x: 
<span style="background-color:#92CCA6">
Valid
</span>
</span>
<table>
<tr>
<td title="Lower and upper minos error of the parameter">Error</td>
<td>%.3g</td>
<td>%.3g</td>
</tr>
<tr>
<td title="Validity of lower/upper minos error" style="background-color:#92CCA6">Valid</td>
<td style="background-color:#92CCA6">True</td>
<td style="background-color:#92CCA6">True</td>
</tr>
<tr>
<td title="Did scan hit limit of any parameter?">At Limit</td>
<td style="background-color:#92CCA6">False</td>
<td style="background-color:#92CCA6">False</td>
</tr>
<tr>
<td title="Did scan hit function call limit?">Max FCN</td>
<td style="background-color:#92CCA6">False</td>
<td style="background-color:#92CCA6">False</td>
</tr>
<tr>
<td title="New minimum found when doing scan?">New Min</td>
<td style="background-color:#92CCA6">False</td>
<td style="background-color:#92CCA6">False</td>
</tr>
</table>
<span>
Minos status for y: 
<span style="background-color:#92CCA6">
Valid
</span>
</span>
<table>
<tr>
<td title="Lower and upper minos error of the parameter">Error</td>
<td>%.3g</td>
<td>%.3g</td>
</tr>
<tr>
<td title="Validity of lower/upper minos error" style="background-color:#92CCA6">Valid</td>
<td style="background-color:#92CCA6">True</td>
<td style="background-color:#92CCA6">True</td>
</tr>
<tr>
<td title="Did scan hit limit of any parameter?">At Limit</td>
<td style="background-color:#92CCA6">False</td>
<td style="background-color:#92CCA6">False</td>
</tr>
<tr>
<td title="Did scan hit function call limit?">Max FCN</td>
<td style="background-color:#92CCA6">False</td>
<td style="background-color:#92CCA6">False</td>
</tr>
<tr>
<td title="New minimum found when doing scan?">New Min</td>
<td style="background-color:#92CCA6">False</td>
<td style="background-color:#92CCA6">False</td>
</tr>
</table>
""" % (minuit.merrors[('x', -1.0)], minuit.merrors[('x', 1.0)],
       minuit.merrors[('y', -1.0)], minuit.merrors[('y', 1.0)])


def test_html_matrix(minuit):
    assert format_html(minuit.matrix()._repr_html_()) == \
r"""<table>
<tr>
<td/>
<td>x</td>
<td>y</td>
</tr>
<tr>
<td>x</td>
<td style="background-color:rgb(255,117,117)">1.00</td>
<td style="background-color:rgb(163,254,186)">0.00</td>
</tr>
<tr>
<td>y</td>
<td style="background-color:rgb(163,254,186)">0.00</td>
<td style="background-color:rgb(186,220,169)">0.25</td>
</tr>
</table>
"""


def test_html_params_with_limits():
    m = Minuit(f1, x=3, y=5, fix_x=True,
               error_x=0.2, error_y=0.1,
               limit_x=(0, None), limit_y=(0, 10),
               errordef=1, print_level=0)
    assert format_html(m.get_initial_param_states()._repr_html_()) == \
r"""<table>
<tr>
<td/>
<td title="Variable name">Name</td>
<td title="Value of parameter">Value</td>
<td title="Hesse error">Hesse Error</td>
<td title="Minos lower error">Minos Error-</td>
<td title="Minos upper error">Minos Error+</td>
<td title="Lower limit of the parameter">Limit-</td>
<td title="Upper limit of the parameter">Limit+</td>
<td title="Is the parameter fixed in the fit">Fixed</td>
</tr>
<tr>
<td>0</td>
<td>x</td>
<td>3</td>
<td>0.2</td>
<td></td>
<td></td>
<td>0</td>
<td></td>
<td>yes</td>
</tr>
<tr>
<td>1</td>
<td>y</td>
<td>5</td>
<td>0.1</td>
<td></td>
<td></td>
<td>0</td>
<td>10</td>
<td></td>
</tr>
</table>
"""


def test_text_params(minuit):
    assert str(minuit.get_initial_param_states()) == \
r"""---------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed |
---------------------------------------------------------------------------------------
|  0 |    x | 0        | 1        |          |          |          |          |       |
|  1 |    y | 0        | 1        |          |          |          |          |       |
---------------------------------------------------------------------------------------"""

    assert str(minuit.get_param_states()) == \
r"""---------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed |
---------------------------------------------------------------------------------------
|  0 |    x | 2        | 1        | -1       | 1        |          |          |       |
|  1 |    y | 1        | 0.5      | -0.5     | 0.5      |          |          |       |
---------------------------------------------------------------------------------------"""


def test_text_fmin(minuit):
    assert str(minuit.get_fmin()) == \
r"""--------------------------------------------------------------------------------------
fval = 1 | total call = 67 | ncalls = 24
edm = %.3g (Goal: 1e-08) | up = 1.0
--------------------------------------------------------------------------------------
|          Valid |    Valid Param | Accurate Covar |         Posdef |    Made Posdef |
--------------------------------------------------------------------------------------
|           True |           True |           True |           True |          False |
--------------------------------------------------------------------------------------
|     Hesse Fail |        Has Cov |      Above EDM |                |  Reach calllim |
--------------------------------------------------------------------------------------
|          False |           True |          False |                |          False |
--------------------------------------------------------------------------------------""" % minuit.get_fmin().edm


def test_text_minos(minuit):
    assert str(minuit.minos()) == \
r"""-------------------------------------------------
| Minos Status for x: Valid                     |
-------------------------------------------------
|      Error      |      -1      |      1       |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------
-------------------------------------------------
| Minos Status for y: Valid                     |
-------------------------------------------------
|      Error      |     -0.5     |     0.5      |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------"""


def test_text_matrix(minuit):
    assert str(minuit.matrix()) == \
r"""-------------------
|   |     x     y |
-------------------
| x |  1.00  0.00 |
| y |  0.00  0.25 |
-------------------"""


def test_text_params_with_limits():
    m = Minuit(f1, x=3, y=5, fix_x=True,
               error_x=0.2, error_y=0.1,
               limit_x=(0, None), limit_y=(0, 10),
               errordef=1)
    assert str(m.get_initial_param_states()) == \
r"""---------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed |
---------------------------------------------------------------------------------------
|  0 |    x | 3        | 0.2      |          |          | 0        |          |  yes  |
|  1 |    y | 5        | 0.1      |          |          | 0        | 10       |       |
---------------------------------------------------------------------------------------"""


def test_text_with_long_names():

    matrix = Matrix(["super-long-name", "x"], ((1.0, 0.1), (0.1, 1.0)))
    assert str(matrix) == \
r"""-----------------------------------------------------
|                 | super-long-name               x |
-----------------------------------------------------
| super-long-name |            1.00            0.10 |
|               x |            0.10            1.00 |
-----------------------------------------------------"""

    mps = Params([Param(0, "super-long-name", 0, 0, False, False, False, False, False, None, None)], None)
    assert str(mps) == \
r"""--------------------------------------------------------------------------------------------------
| No |      Name       |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed |
--------------------------------------------------------------------------------------------------
|  0 | super-long-name | 0        | 0        |          |          |          |          |       |
--------------------------------------------------------------------------------------------------"""


def test_console_frontend_with_difficult_values():
    matrix = Matrix(("x", "y"), ((-1.23456, 0), (0, 0)))
    assert str(matrix) == \
r"""-------------------
|   |     x     y |
-------------------
| x | -1.23  0.00 |
| y |  0.00  0.00 |
-------------------"""

    mps = Params([Param(0, "x",  -1.234567e-11, 1.234567e11, True, False, False, False, False, None, None)], None)

    assert str(mps) == \
r"""---------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed |
---------------------------------------------------------------------------------------
|  0 |    x | -1.23E-11| 1.23E+11 |          |          |          |          | CONST |
---------------------------------------------------------------------------------------"""
