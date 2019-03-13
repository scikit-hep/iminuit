from __future__ import (absolute_import, division, unicode_literals)
from iminuit.color import Gradient
from iminuit.repr_text import format_numbers

good_style = 'background-color:#92CCA6;'
bad_style = 'background-color:#FF7878;'
warn_style = 'background-color:#FFF79A;'
backgrounds = ('background-color:#F4F4F4;', 'background-color:#FFFFFF;')

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


def tr(html, **kwargs):
    return Tag("tr", html, **kwargs)


def th(html, **kwargs):
    return Tag("th", html, **kwargs)


def td(html, **kwargs):
    return Tag("td", html, **kwargs)


def good(x, should_be):
    return good_style if x == should_be else bad_style


def fmin(sfmin):
    """Display FunctionMinum in html representation"""
    goaledm = 1e-4 * sfmin.tolerance * sfmin.up
    s = Html()
    with table(s):
        with tr(s):
            with td(s, colspan="2", title="Minimum value of function"):
                s += "FCN = %.4G" % sfmin.fval
            with td(s, colspan="3",align="center",
                    title="No. of calls in last algorithm and total number of calls"):
                s += "Ncalls = %i (%i total)" % (sfmin.nfcn, sfmin.ncalls)
        with tr(s):
            with td(s, colspan="2",
                    title="Estimated distance to minimum and target threshold"):
                s += "EDM = %.3G (Goal: %G)" % (sfmin.edm, goaledm)
            with td(s, colspan="3", align="center",
                    title="Increase in FCN which corresponds to 1 standard deviation"):
                s += "up = %.1f" % sfmin.up
        with tr(s):
            with td(s, align="center", title="Validity of the migrad call"):
                s += "Valid Min."
            with td(s, align="center", title="Validity of parameters"):
                s += "Valid Param."
            with td(s, align="center", title="Is EDM above goal EDM?"):
                s += "Above EDM"
            with td(s, colspan="2", align="center",
                    title="Did last migrad call reach max call limit?"):
                s += "Reached call limit"
        with tr(s):
            with td(s, align="center", style=good(sfmin.is_valid, True)):
                s += "%s" % sfmin.is_valid
            with td(s, align="center", style=good(sfmin.has_valid_parameters, True)):
                s += "%s" % sfmin.has_valid_parameters
            with td(s, align="center", style=good(sfmin.is_above_max_edm, False)):
                s += "%s" % sfmin.is_above_max_edm
            with td(s, colspan="2", align="center",
                    style=good(sfmin.has_reached_call_limit, False)):
                s += "%s" % sfmin.has_reached_call_limit
        with tr(s):
            with td(s, align="center", title="Did Hesse fail?"):
                s += "Hesse failed"
            with td(s, align="center", title="Has covariance matrix"):
                s += "Has cov."
            with td(s, align="center", title="Is covariance matrix accurate?"):
                s += "Accurate"
            with td(s, align="center", title="Is covariance matrix positive definite?"):
                s += "Pos. def."
            with td(s, align="center",
                    title="Was positive definiteness enforced by Minuit?"):
                s += "Forced"
        with tr(s):
            with td(s, align="center", style=good(sfmin.hesse_failed, False)):
                s += "%s" % sfmin.hesse_failed
            with td(s, align="center", style=good(sfmin.has_covariance, True)):
                s += "%s" % sfmin.has_covariance
            with td(s, align="center", style=good(sfmin.has_accurate_covar, True)):
                s += "%s" % sfmin.has_accurate_covar
            with td(s, align="center", style=good(sfmin.has_posdef_covar, True)):
                s += "%s" % sfmin.has_posdef_covar
            with td(s, align="center", style=good(sfmin.has_made_posdef_covar, False)):
                s += "%s" % sfmin.has_made_posdef_covar
    return str(s)


def merror(me):
    mel, meu = format_numbers(me.lower, me.upper)
    s = Html()
    with table(s):
        with tr(s):
            with th(s, title="Parameter name"):
                s += me.name
            with td(s, colspan="2", style=good(me.is_valid, True), align="center"):
                s += 'Valid' if me.is_valid else 'Invalid'
        with tr(s):
            with td(s, title="Lower and upper minos error of the parameter"):
                s += "Error"
            with td(s):
                s += mel
            with td(s):
                s += meu
        with tr(s):
            with td(s, title="Validity of lower/upper minos error"):
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
        with tr(s, style=backgrounds[0]):
            s += "<td/>"
            with th(s, title="Variable name"):
                s += "Name"
            with th(s, title="Value of parameter"):
                s += "Value"
            with th(s, title="Hesse error"):
                s += "Hesse Error"
            with th(s, title="Minos lower error"):
                s += "Minos Error-"
            with th(s, title="Minos upper error"):
                s += "Minos Error+"
            with th(s, title="Lower limit of the parameter"):
                s += "Limit-"
            with th(s, title="Upper limit of the parameter"):
                s += "Limit+"
            with th(s, title="Is the parameter fixed in the fit"):
                s += "Fixed"

        mes = mps.merrors

        # body
        for i, mp in enumerate(mps):
            if mes and mp.name in mes:
                me = mes[mp.name]
                v, e, mem, mep = format_numbers(mp.value, mp.error, me.lower, me.upper)
            else:
                e, v = format_numbers(mp.error, mp.value)
                mem = ''
                mep = ''
            with tr(s, style=backgrounds[(i + 1) % 2]):
                with td(s):
                    s += str(i)
                with td(s):
                    s += mp.name
                with td(s):
                    s += v
                with td(s):
                    s += e
                with td(s):
                    s += mem
                with td(s):
                    s += mep
                with td(s):
                    s += '%.3G' % mp.lower_limit if mp.lower_limit is not None else ''
                with td(s):
                    s += '%.3G' % mp.upper_limit if mp.upper_limit is not None else '' 
                with td(s):
                    s += 'yes' if mp.is_fixed else ('CONST' if mp.is_const else '')
    return str(s)


def matrix(m):
    is_correlation = True
    for i in range(len(m)):
        if m[i][i] != 1.0:
            is_correlation = False
            break

    if not is_correlation:
        n = len(m)
        args = []
        for mi in m:
            for mj in mi:
                args.append(mj)
        nums = format_numbers(*args)

    s = Html()
    with table(s):
        with tr(s):
            s += "<td/>\n"
            for v in m.names:
                with th(s):
                    s += v
        for i, v in enumerate(m.names):
            with tr(s):
                with th(s):
                    s += v
                for j in range(len(m.names)):
                    val = m[i][j]
                    if is_correlation:
                        if i == j:
                            with td(s):
                                s += "1.00"                        
                        else:
                            color = Gradient.rgb_color_for(val)
                            with td(s, style="background-color:"+color):
                                s += "%.2f" % val
                    else:
                        if i == j:
                            with td(s):
                                s += nums[n*i + j]
                        else:
                            color = Gradient.rgb_color_for(val / (m[i][i] * m[j][j]) ** 0.5)
                            with td(s, style="background-color:"+color):
                                s += nums[n*i + j]

    return str(s)
