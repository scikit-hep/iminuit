from nose.tools import *
from iminuit.latex import LatexTable

def test_table():
    ltt = LatexTable(data=[['alpha',10.123],['alpha_s',30]])
    assert_equal(str(ltt),
    '\\begin{tabular}{|c|c|}\n\\hline\n$\\alpha$ &     10.123\\\\\n\\hline\n$\\alpha_{s}$ & 30\\\\\n\\hline\n\\end{tabular}'
    )

def test_table_color():
    ltt = LatexTable(data=[['alpha',10.123],['alpha_s',30]])
    ltt.set_cell_color(0,0,(111,123,95))
    assert_equal(str(ltt),
    '%\\usepackage[table]{xcolor} % include this for color\n%\\usepackage{rotating} % include this for rotate header\n%\\documentclass[xcolor=table]{beamer} % for beamer\n\\begin{tabular}{|c|c|}\n\\hline\n\\cellcolor[RGB]{111,123,95} $\\alpha$ &     10.123\\\\\n\\hline\n$\\alpha_{s}$ & 30\\\\\n\\hline\n\\end{tabular}'
    )

def test_latexmap():
    ltt = LatexTable(data=[['alpha',10.123,20],['alpha_s',30,40]],
                     latex_map={'alpha':'beta'})
    assert_equal(
        ltt._format('alpha'),
        'beta'
        )

def test_smartlatex():
    ltt = LatexTable(data=[['alpha',10.123,20],['alpha_s',30,40]])
    assert_equal(
        ltt._convert_smart_latex('alpha'),
        r'$\alpha$'
        )
    assert_equal(
        ltt._convert_smart_latex('alpha_beta'),
        r'$\alpha_{\beta}$'
        )
    assert_equal(
        ltt._convert_smart_latex('a_b'),
        r'$a_{b}$'
        )
    assert_equal(
        ltt._convert_smart_latex('a_b_c'),
        r'a $b_{c}$'
        )
    assert_equal(
        ltt._convert_smart_latex('a'),
        r'a'
        )
    assert_equal(
        ltt._convert_smart_latex('a_alpha_beta'),
        r'a $\alpha_{\beta}$'
        )