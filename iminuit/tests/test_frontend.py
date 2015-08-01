from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from iminuit import Minuit
from iminuit.frontends.html import HtmlFrontend
from iminuit.frontends.console import ConsoleFrontend


def f1(x, y):
    return (1 - x) ** 2 + 100 * (y - 1) ** 2


def test_html():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1, frontend=HtmlFrontend())
    m.tol = 1e-4
    m.migrad()
    m.minos()
    m.print_matrix()
    m.print_initial_param()
    m.print_fmin()
    m.print_all_minos()
    m.latex_matrix()


def test_console():
    m = Minuit(f1, x=0, y=0, pedantic=False, print_level=1, frontend=ConsoleFrontend())
    m.tol = 1e-4
    m.migrad()
    m.minos()
    m.print_matrix()
    m.print_initial_param()
    m.print_fmin()
    m.print_all_minos()
    m.latex_matrix()
