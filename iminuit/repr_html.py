from __future__ import (absolute_import, division, unicode_literals)
from iminuit.color import Gradient

good_style = 'background-color:#92CCA6'
bad_style = 'background-color:#FF7878'
warn_style = 'background-color:#FFF79A'


class Html:    
    def __init__(self):
        self.lines = []
    def __iadd__(self, s):
        self.lines.append(s)
        return self
    def __str__(self):
        s = ""
        for x in self.lines:
            s += x + "\n"
        return s
    def __repr__(self):
        return repr(self.lines)


class Tag:
    def __init__(self, name, html, **kwargs):
        self.name = name
        self.html = html
        self.html += " ".join(("<%s" % name,) + tuple('%s="%s"' % p for p in kwargs.items())) + ">"
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.html += "</" + self.name + ">"


def table(html, **kwargs):
    return Tag("table", html, **kwargs)


def tr(html):
    return Tag("tr", html)


def td(html, **kwargs):
    return Tag("td", html, **kwargs)


def span(html, **kwargs):
    return Tag("span", html, **kwargs)


def good(x, should_be):
    return good_style if x == should_be else bad_style


def caution(x, should_be):
    return good_style if x == should_be else warn_style


def fmin(sfmin):
    """Display FunctionMinum in html representation"""
    goaledm = 1e-4 * sfmin.tolerance * sfmin.up
    s = Html()
    with table(s):
        with tr(s):
            with td(s, title="Minimum value of function"):
                s += "FCN = %.3g" % sfmin.fval
            with td(s, title="Total number of call to FCN so far"):
                s += "TOTAL NCALL = %i" % sfmin.ncalls
            with td(s, title="Number of call in last migrad"):
                s += "NCALLS = %i" % sfmin.nfcn
        with tr(s):
            with td(s, title="Estimated distance to minimum"):
                s += "EDM = %.3g" % sfmin.edm
            with td(s, title="Maximum EDM definition of convergence"):
                s += "GOAL EDM = %.3g" % goaledm
            with td(s, title="Error def. Amount of increase in FCN to be defined as 1 standard deviation"):
                s += "UP = %.1f" % sfmin.up
    with table(s):
        with tr(s):
            with td(s, align="center", title="Validity of the migrad call"):
                s += "Valid"
            with td(s, align="center", title="Validity of parameters"):
                s += "Valid Param"
            with td(s, align="center", title="Is Covariance matrix accurate?"):
                s += "Accurate Covar"
            with td(s, align="center", title="Positive definiteness of covariance matrix"):
                s += "PosDef"
            with td(s, align="center", title="Was covariance matrix made posdef by adding diagonal element"):
                s += "Made PosDef"
        with tr(s):
            with td(s, align="center", style=good(sfmin.is_valid, True)):
                s += "%s" % sfmin.is_valid
            with td(s, align="center", style=good(sfmin.has_valid_parameters, True)):
                s += "%s" % sfmin.has_valid_parameters
            with td(s, align="center", style=good(sfmin.has_accurate_covar, True)):
                s += "%s" % sfmin.has_accurate_covar
            with td(s, align="center", style=good(sfmin.has_posdef_covar, True)):
                s += "%s" % sfmin.has_posdef_covar
            with td(s, align="center", style=good(sfmin.has_made_posdef_covar, False)):
                s += "%s" % sfmin.has_made_posdef_covar
        with tr(s):
            with td(s, align="center", title="Was last hesse call fail?"):
                s += "Hesse Fail"
            with td(s, align="center", title="Validity of covariance"):
                s += "HasCov"
            with td(s, align="center", title="Is EDM above goal EDM?"):
                s += "Above EDM"
            with td(s, align="center"):
                pass
            with td(s, align="center", title="Did last migrad call reach max call limit?"):
                s += "Reach calllim"
        with tr(s):
            with td(s, align="center", style=good(sfmin.hesse_failed, False)):
                s += "%s" % sfmin.hesse_failed
            with td(s, align="center", style=good(sfmin.has_covariance, True)):
                s += "%s" % sfmin.has_covariance
            with td(s, align="center", style=good(sfmin.is_above_max_edm, False)):
                s += "%s" % sfmin.is_above_max_edm
            with td(s, align="center"):
                pass
            with td(s, align="center", style=caution(sfmin.has_reached_call_limit, False)):
                s += "%s" % sfmin.has_reached_call_limit
    return str(s)


def merror(me):    
    s = Html()
    with span(s):
        s += "Minos status for %s: " % me.name
        with span(s, style=good(me.is_valid, True)):
            s += 'Valid' if me.is_valid else 'Invalid'
    with table(s):
        with tr(s):
            with td(s, title="Lower and upper minos error of the parameter"):
                s += "Error"
            with td(s):
                s += "%.3g" % me.lower
            with td(s):
                s += "%.3g" % me.upper
        with tr(s):
            with td(s, title="Validity of lower/upper minos error", style=good(me.is_valid, True)):
                s += "Valid"
            with td(s, style=good(me.lower_valid, True)):
                s += "%s" % me.lower_valid
            with td(s, style=good(me.upper_valid, True)):
                s += "%s" % me.upper_valid
        with tr(s):
            with td(s, title="Did scan hit limit of any parameter?"):
                s += "At Limit"
            with td(s, style=good(me.at_lower_limit, False)):
                s += "%s" % me.at_lower_limit
            with td(s, style=good(me.at_upper_limit, False)):
                s += "%s" % me.at_upper_limit
        with tr(s):
            with td(s, title="Did scan hit function call limit?"):
                s += "Max FCN"
            with td(s, style=good(me.at_lower_max_fcn, False)):
                s += "%s" % me.at_lower_max_fcn
            with td(s, style=good(me.at_upper_max_fcn, False)):
                s += "%s" % me.at_upper_max_fcn
        with tr(s):
            with td(s, title="New minimum found when doing scan?"):
                s += "New Min" 
            with td(s, style=good(me.lower_new_min, False)):
                s += "%s" % me.lower_new_min
            with td(s, style=good(me.upper_new_min, False)):
                s += "%s" % me.upper_new_min
    return str(s)
    

def params(mps):    
    s = Html()
    with table(s):
        # header
        with tr(s):
            s += "<td/>"
            with td(s, title="Variable name"):
                s += "Name"
            with td(s, title="Value of parameter"):
                s += "Value"
            with td(s, title="Hesse error"):
                s += "Hesse Error"
            with td(s, title="Minos lower error"):
                s += "Minos Error-"
            with td(s, title="Minos upper error"):
                s += "Minos Error+"
            with td(s, title="Lower limit of the parameter"):
                s += "Limit-"
            with td(s, title="Upper limit of the parameter"):
                s += "Limit+"
            with td(s, title="Is the parameter fixed in the fit"):
                s += "Fixed"

        mes = mps.merrors

        # body
        for i, mp in enumerate(mps):
            with tr(s):
                with td(s):
                    s += str(i)
                with td(s):
                    s += mp.name
                with td(s):
                    s += '%.3g' % mp.value
                with td(s):
                    s += '%.3g' % mp.error
                with td(s):
                    s += '%.3g' % mes[mp.name].lower if mes and mp.name in mes else ''
                with td(s):
                    s += '%.3g' % mes[mp.name].upper if mes and mp.name in mes else ''
                with td(s):
                    s += '%.3g' % mp.lower_limit if mp.lower_limit is not None else ''
                with td(s):
                    s += '%.3g' % mp.upper_limit if mp.upper_limit is not None else '' 
                with td(s):
                    s += 'yes' if mp.is_fixed else ('CONST' if mp.is_const else '')
    return str(s)


def matrix(m):
    s = Html()
    with table(s):
        with tr(s):
            s += "<td/>\n"
            for v in m.names:
                with td(s):
                    s += v
        for i, v in enumerate(m.names):
            with tr(s):
                with td(s):
                    s += v
                for j in range(len(m.names)):
                    val = m[i][j]
                    color = Gradient.rgb_color_for(val)
                    with td(s, style="background-color:"+color):
                        s += "%3.2f" % val
    return str(s)
