from __future__ import (absolute_import, division, print_function)
# Here we don't import unicode_literals, because otherwise we cannot write
# in "\usepackage" and similar in the string literals below.
# The problem is an inconsistency regarding the effect of
# from __future__ import unicode_literals in Python 2 and 3, which is
# explained here:
# https://stackoverflow.com/questions/7602171/unicode-error-unicodeescape-codec-cant-decode-bytes-string-with-u
# We want the same code to work in Python 2 and 3 and chose this solution.
import sys
from iminuit import Minuit
import iminuit.frontends.html as html
import iminuit.frontends.console as console
from iminuit.tests.utils import requires_dependency


def f1(x, y):
    return (x - 2) ** 2 + (y - 1) ** 2 / 0.25 + 1


@requires_dependency('IPython')
def test_html(capsys):
    def out():
        return capsys.readouterr()[0]

    class FakeRandom(object):
        def choice(self, s):
            return s[0]

    class Frontend(html.HtmlFrontend):
        rng = FakeRandom()  # for reproducability

        def display(self, *args):  # text to stdout
            sys.stdout.write('\n'.join(args) + '\n')

    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1, frontend=Frontend())
    m.tol = 1e-4

    m.print_initial_param()
    assert r"""<table>
    <tr>
        <td><a href="#" onclick="$('#aaaaaaaaaa').toggle()">+</a></td>
        <td title="Variable name">Name</td>
        <td title="Value of parameter">Value</td>
        <td title="Hesse error">Hesse Error</td>
        <td title="Minos lower error">Minos Error-</td>
        <td title="Minos upper error">Minos Error+</td>
        <td title="Lower limit of the parameter">Limit-</td>
        <td title="Upper limit of the parameter">Limit+</td>
        <td title="Is the parameter fixed in the fit">Fixed?</td>
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
        <td>No</td>
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
        <td>No</td>
    </tr>
</table>
<pre id="aaaaaaaaaa" style="display:none;">
<textarea rows="10" cols="50" onclick="this.select()" readonly>
\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
\hline
 & Name & Value & Hesse Error & Minos Error- & Minos Error+ & Limit- & Limit+ & Fixed?\\
\hline
0 & x & 0 & 1 &  &  &  &  & No\\
\hline
1 & y & 0 & 1 &  &  &  &  & No\\
\hline
\end{tabular}
</textarea>
</pre>
""" == out()

    m.migrad()
    fmin = m.get_fmin()

    assert r"""<hr>
<table>
    <tr>
        <td title="Minimum value of function">FCN = 1.0</td>
        <td title="Total number of call to FCN so far">TOTAL NCALL = 24</td>
        <td title="Number of call in last migrad">NCALLS = 24</td>
    </tr>
    <tr>
        <td title="Estimated distance to minimum">EDM = %s</td>
        <td title="Maximum EDM definition of convergence">GOAL EDM = 1e-08</td>
        <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
        UP = 1.0</td>
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
<table>
    <tr>
        <td><a href="#" onclick="$('#aaaaaaaaaa').toggle()">+</a></td>
        <td title="Variable name">Name</td>
        <td title="Value of parameter">Value</td>
        <td title="Hesse error">Hesse Error</td>
        <td title="Minos lower error">Minos Error-</td>
        <td title="Minos upper error">Minos Error+</td>
        <td title="Lower limit of the parameter">Limit-</td>
        <td title="Upper limit of the parameter">Limit+</td>
        <td title="Is the parameter fixed in the fit">Fixed?</td>
    </tr>
    <tr>
        <td>0</td>
        <td>x</td>
        <td>2</td>
        <td>1</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>No</td>
    </tr>
    <tr>
        <td>1</td>
        <td>y</td>
        <td>1</td>
        <td>0.5</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td>No</td>
    </tr>
</table>
<pre id="aaaaaaaaaa" style="display:none;">
<textarea rows="10" cols="50" onclick="this.select()" readonly>
\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
\hline
 & Name & Value & Hesse Error & Minos Error- & Minos Error+ & Limit- & Limit+ & Fixed?\\
\hline
0 & x & 2 & 1 &  &  &  &  & No\\
\hline
1 & y & 1 & 0.5 &  &  &  &  & No\\
\hline
\end{tabular}
</textarea>
</pre>
<hr>
""" % fmin.edm == out()

    m.minos()
    assert r"""<span>Minos status for x: <span style="background-color:#92CCA6">VALID</span></span>
<table>
    <tr>
        <td title="lower and upper minos error of the parameter">Error</td>
        <td>%s</td>
        <td>%s</td>
    </tr>
    <tr>
        <td title="Validity of minos error">Valid</td>
        <td style="background-color:#92CCA6">True</td>
        <td style="background-color:#92CCA6">True</td>
    </tr>
    <tr>
        <td title="Did minos error search hit limit of any parameter?">At Limit</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="I don't really know what this one means... Post it in issue if you know">Max FCN</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="New minimum found when doing minos scan.">New Min</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
</table>
<span>Minos status for y: <span style="background-color:#92CCA6">VALID</span></span>
<table>
    <tr>
        <td title="lower and upper minos error of the parameter">Error</td>
        <td>%s</td>
        <td>%s</td>
    </tr>
    <tr>
        <td title="Validity of minos error">Valid</td>
        <td style="background-color:#92CCA6">True</td>
        <td style="background-color:#92CCA6">True</td>
    </tr>
    <tr>
        <td title="Did minos error search hit limit of any parameter?">At Limit</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="I don't really know what this one means... Post it in issue if you know">Max FCN</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="New minimum found when doing minos scan.">New Min</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
</table>
""" % (m.merrors[('x', -1.0)], m.merrors[('x', 1.0)],
       m.merrors[('y', -1.0)], m.merrors[('y', 1.0)]) == out()

    m.print_matrix()
    assert r"""<table>
    <tr>
        <td><a onclick="$('#aaaaaaaaaa').toggle()" href="#">+</a></td> <td>x</td> <td>y</td>
    </tr>
    <tr>
        <td>x</td> <td style="background-color:rgb(255,117,117)">1.00</td> <td style="background-color:rgb(163,254,186)">0.00</td>
    </tr>
    <tr>
        <td>y</td> <td style="background-color:rgb(163,254,186)">0.00</td> <td style="background-color:rgb(255,117,117)">1.00</td>
    </tr>
</table>
<pre id="aaaaaaaaaa" style="display:none;">
<textarea rows="13" cols="50" onclick="this.select()" readonly>
%\usepackage[table]{xcolor} % include this for color
%\usepackage{rotating} % include this for rotate header
%\documentclass[xcolor=table]{beamer} % for beamer
\begin{tabular}{|c|c|c|}
\hline
\rotatebox{90}{} & \rotatebox{90}{x} & \rotatebox{90}{y}\\
\hline
x & \cellcolor[RGB]{255,117,117} 1.00 & \cellcolor[RGB]{163,254,186} 0.00\\
\hline
y & \cellcolor[RGB]{163,254,186} 0.00 & \cellcolor[RGB]{255,117,117} 1.00\\
\hline
\end{tabular}
</textarea>
</pre>
""" == out()

    m.print_all_minos()
    assert r"""<span>Minos status for x: <span style="background-color:#92CCA6">VALID</span></span>
<table>
    <tr>
        <td title="lower and upper minos error of the parameter">Error</td>
        <td>%s</td>
        <td>%s</td>
    </tr>
    <tr>
        <td title="Validity of minos error">Valid</td>
        <td style="background-color:#92CCA6">True</td>
        <td style="background-color:#92CCA6">True</td>
    </tr>
    <tr>
        <td title="Did minos error search hit limit of any parameter?">At Limit</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="I don't really know what this one means... Post it in issue if you know">Max FCN</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="New minimum found when doing minos scan.">New Min</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
</table>
<span>Minos status for y: <span style="background-color:#92CCA6">VALID</span></span>
<table>
    <tr>
        <td title="lower and upper minos error of the parameter">Error</td>
        <td>%s</td>
        <td>%s</td>
    </tr>
    <tr>
        <td title="Validity of minos error">Valid</td>
        <td style="background-color:#92CCA6">True</td>
        <td style="background-color:#92CCA6">True</td>
    </tr>
    <tr>
        <td title="Did minos error search hit limit of any parameter?">At Limit</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="I don't really know what this one means... Post it in issue if you know">Max FCN</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
    <tr>
        <td title="New minimum found when doing minos scan.">New Min</td>
        <td style="background-color:#92CCA6">False</td>
        <td style="background-color:#92CCA6">False</td>
    </tr>
</table>
""" % (m.merrors[('x', -1.0)], m.merrors[('x', 1.0)],
       m.merrors[('y', -1.0)], m.merrors[('y', 1.0)]) == out()

    m = Minuit(f1, x=5, y=5,
               error_x=0.1, error_y=0.1,
               limit_x=(0, None), limit_y=(0, 10),
               errordef=1, frontend=Frontend())
    m.print_param()
    assert r"""<table>
    <tr>
        <td><a href="#" onclick="$('#aaaaaaaaaa').toggle()">+</a></td>
        <td title="Variable name">Name</td>
        <td title="Value of parameter">Value</td>
        <td title="Hesse error">Hesse Error</td>
        <td title="Minos lower error">Minos Error-</td>
        <td title="Minos upper error">Minos Error+</td>
        <td title="Lower limit of the parameter">Limit-</td>
        <td title="Upper limit of the parameter">Limit+</td>
        <td title="Is the parameter fixed in the fit">Fixed?</td>
    </tr>
    <tr>
        <td>0</td>
        <td>x</td>
        <td>5</td>
        <td>0.1</td>
        <td></td>
        <td></td>
        <td>0</td>
        <td></td>
        <td>No</td>
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
        <td>No</td>
    </tr>
</table>
<pre id="aaaaaaaaaa" style="display:none;">
<textarea rows="10" cols="50" onclick="this.select()" readonly>
\begin{tabular}{|c|r|r|r|r|r|r|r|c|}
\hline
 & Name & Value & Hesse Error & Minos Error- & Minos Error+ & Limit- & Limit+ & Fixed?\\
\hline
0 & x & 5 & 0.1 &  &  & 0.0 &  & No\\
\hline
1 & y & 5 & 0.1 &  &  & 0.0 & 10 & No\\
\hline
\end{tabular}
</textarea>
</pre>
""" == out()


def test_console(capsys):
    def out():
        return capsys.readouterr()[0]

    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1,
               frontend=console.ConsoleFrontend())
    m.tol = 1e-4

    m.print_initial_param()
    assert r"""----------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed? |
----------------------------------------------------------------------------------------
|  0 |    x | 0        | 1        |          |          |          |          |   No   |
|  1 |    y | 0        | 1        |          |          |          |          |   No   |
----------------------------------------------------------------------------------------
""" == out()

    m.migrad()
    assert r"""**************************************************
*                     MIGRAD                     *
**************************************************

**************************************************************************************
--------------------------------------------------------------------------------------
fval = 1.0 | total call = 24 | ncalls = 24
edm = %s (Goal: 1e-08) | up = 1.0
--------------------------------------------------------------------------------------
|          Valid |    Valid Param | Accurate Covar |         Posdef |    Made Posdef |
--------------------------------------------------------------------------------------
|           True |           True |           True |           True |          False |
--------------------------------------------------------------------------------------
|     Hesse Fail |        Has Cov |      Above EDM |                |  Reach calllim |
--------------------------------------------------------------------------------------
|          False |           True |          False |                |          False |
--------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed? |
----------------------------------------------------------------------------------------
|  0 |    x | 2        | 1        |          |          |          |          |   No   |
|  1 |    y | 1        | 0.5      |          |          |          |          |   No   |
----------------------------------------------------------------------------------------
**************************************************************************************
""" % m.get_fmin().edm == out()

    m.minos()
    assert r"""**************************************************
*                     MINOS                      *
**************************************************

-------------------------------------------------
Minos Status for x: VALID
-------------------------------------------------
|      Error      |      -1      |      1       |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------
-------------------------------------------------
Minos Status for y: VALID
-------------------------------------------------
|      Error      |     -0.5     |     0.5      |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------
""" == out()

    m.print_matrix()
    assert r"""-------------------
Correlation
-------------------
       |    0    1 
-------------------
x    0 | 1.00 0.00 
y    1 | 0.00 1.00 
-------------------
""" == out()

    m.print_all_minos()
    assert r"""-------------------------------------------------
Minos Status for x: VALID
-------------------------------------------------
|      Error      |      -1      |      1       |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------
-------------------------------------------------
Minos Status for y: VALID
-------------------------------------------------
|      Error      |     -0.5     |     0.5      |
|      Valid      |     True     |     True     |
|    At Limit     |    False     |    False     |
|     Max FCN     |    False     |    False     |
|     New Min     |    False     |    False     |
-------------------------------------------------
""" == out()

    m = Minuit(f1, x=5, y=5,
               error_x=0.1, error_y=0.1,
               limit_x=(0, None), limit_y=(0, 10),
               errordef=1, frontend=console.ConsoleFrontend())
    m.print_param()
    assert r"""----------------------------------------------------------------------------------------
| No | Name |  Value   | Sym. Err |   Err-   |   Err+   | Limit-   | Limit+   | Fixed? |
----------------------------------------------------------------------------------------
|  0 |    x | 5        | 0.1      |          |          | 0        |          |   No   |
|  1 |    y | 5        | 0.1      |          |          | 0        | 10       |   No   |
----------------------------------------------------------------------------------------
""" == out()
