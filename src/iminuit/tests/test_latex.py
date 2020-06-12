from iminuit.latex import LatexTable


def test_table():
    ltt = LatexTable(data=[["alpha", 10.123], ["alpha_s", 30]])
    expected = """\\begin{tabular}{|c|c|}
\\hline
$\\alpha$ &     10.123\\\\
\\hline
$\\alpha_{s}$ & 30\\\\
\\hline
\\end{tabular}"""
    assert str(ltt) == expected


def test_table_color():
    ltt = LatexTable(data=[["alpha", 10.123], ["alpha_s", 30]])
    ltt.set_cell_color(0, 0, (111, 123, 95))
    expected = """%\\usepackage[table]{xcolor} % include this for color
%\\usepackage{rotating} % include this for rotate header
%\\documentclass[xcolor=table]{beamer} % for beamer
\\begin{tabular}{|c|c|}
\\hline
\\cellcolor[RGB]{111,123,95} $\\alpha$ &     10.123\\\\
\\hline
$\\alpha_{s}$ & 30\\\\
\\hline
\\end{tabular}"""
    assert str(ltt) == expected


def test_latexmap():
    ltt = LatexTable(
        data=[["alpha", 10.123, 20], ["alpha_s", 30, 40]], latex_map={"alpha": "beta"}
    )
    assert ltt._format("alpha") == "beta"


def test_smartlatex():
    ltt = LatexTable(data=[["alpha", 10.123, 20], ["alpha_s", 30, 40]])
    assert ltt._convert_smart_latex("alpha") == r"$\alpha$"
    assert ltt._convert_smart_latex("alpha_beta") == r"$\alpha_{\beta}$"
    assert ltt._convert_smart_latex("a_b") == r"$a_{b}$"
    assert ltt._convert_smart_latex("a_b_c") == r"a $b_{c}$"
    assert ltt._convert_smart_latex("a") == r"a"
    assert ltt._convert_smart_latex("a_alpha_beta") == r"a $\alpha_{\beta}$"


def test_format():
    ltt = LatexTable(
        data=[["alpha", 10.123, 20], ["alpha_s", 30, 40]], smart_latex=False
    )
    assert ltt._format("a_b") == r"a\_b"
    ltt2 = LatexTable(
        data=[["alpha", 10.123, 20], ["alpha_s", 30, 40]],
        smart_latex=False,
        escape_under_score=False,
    )
    assert ltt2._format("a_b") == r"a_b"


def test_empty_table():
    ltt = LatexTable(data=[])
    expected = """\\begin{tabular}{|c|}
\\hline
\\end{tabular}"""
    assert str(ltt) == expected


def test_empty_table_with_headers():
    ltt = LatexTable(data=[], headers=["Foo", "Bar"])
    expected = """\\begin{tabular}{|c|c|}
\\hline
Foo & Bar\\\\
\\hline
\\end{tabular}"""
    assert str(ltt) == expected
