from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit.latex import LatexTable


def test_table():
    ltt = LatexTable(data=[['alpha', 10.123], ['alpha_s', 30]])
    expected = '\\begin{tabular}{|c|c|}\n\\hline\n$\\alpha$ &     10.123\\\\\n\\hline\n$\\alpha_{s}$ & 30\\\\\n\\hline\n\\end{tabular}'
    assert str(ltt) == expected


def test_table_color():
    ltt = LatexTable(data=[['alpha', 10.123], ['alpha_s', 30]])
    ltt.set_cell_color(0, 0, (111, 123, 95))
    expected = '%\\usepackage[table]{xcolor} % include this for color\n%\\usepackage{rotating} % include this for rotate header\n%\\documentclass[xcolor=table]{beamer} % for beamer\n\\begin{tabular}{|c|c|}\n\\hline\n\\cellcolor[RGB]{111,123,95} $\\alpha$ &     10.123\\\\\n\\hline\n$\\alpha_{s}$ & 30\\\\\n\\hline\n\\end{tabular}'
    assert str(ltt) == expected


def test_latexmap():
    ltt = LatexTable(data=[['alpha', 10.123, 20], ['alpha_s', 30, 40]],
                     latex_map={'alpha': 'beta'})
    assert ltt._format('alpha') == 'beta'


def test_smartlatex():
    ltt = LatexTable(data=[['alpha', 10.123, 20], ['alpha_s', 30, 40]])
    assert ltt._convert_smart_latex('alpha') == r'$\alpha$'
    assert ltt._convert_smart_latex('alpha_beta') == r'$\alpha_{\beta}$'
    assert ltt._convert_smart_latex('a_b') == r'$a_{b}$'
    assert ltt._convert_smart_latex('a_b_c') == r'a $b_{c}$'
    assert ltt._convert_smart_latex('a') == r'a'
    assert ltt._convert_smart_latex('a_alpha_beta') == r'a $\alpha_{\beta}$'


def test_format():
    ltt = LatexTable(data=[['alpha', 10.123, 20], ['alpha_s', 30, 40]],
                     smart_latex=False)
    assert ltt._format('a_b') == r'a\_b'
    ltt2 = LatexTable(data=[['alpha', 10.123, 20], ['alpha_s', 30, 40]],
                      smart_latex=False, escape_under_score=False)
    assert ltt2._format('a_b') == r'a_b'


def test_empty_table():
    ltt = LatexTable(data=[])
    expected = '\n'.join([
        r'\begin{tabular}{|c|}',
        r'\hline',
        r'\end{tabular}',
    ])
    assert str(ltt) == expected


def test_empty_table_with_headers():
    ltt = LatexTable(data=[], headers=['Foo', 'Bar'])
    expected = '\n'.join([
        r'\begin{tabular}{|c|c|}',
        r'\hline',
        r'Foo & Bar\\',
        r'\hline',
        r'\end{tabular}',
    ])
    assert str(ltt) == expected
