from iminuit.color import Gradient
from iminuit.repr_text import pdg_format, matrix_format, goaledm

good_style = "background-color:#92CCA6;"
bad_style = "background-color:#FF7878;"
warn_style = "background-color:#FFF79A;"
backgrounds = ("background-color:#F4F4F4;", "background-color:#FFFFFF;")


def tag(name, *args, delim=" ", **kwargs):
    # sort keys so that order is same on all platforms
    s = "<" + name
    for k in sorted(kwargs):
        v = kwargs[k]
        s += ' %s="%s"' % (k, v)
    s += ">"

    def visit(x):
        if isinstance(x, str):
            return delim + x
        else:
            s = ""
            for xi in x:
                s += visit(xi)
            return s

    for x in args:
        s += visit(x)
    s += "%s</%s>" % (delim, name)
    return s


def table(*args, **kwargs):
    return tag("table", *args, delim="\n", **kwargs) + "\n"


def tr(*args, **kwargs):
    return tag("tr", *args, **kwargs)


def th(*args, **kwargs):
    return tag("th", *args, **kwargs)


def td(*args, **kwargs):
    return tag("td", *args, **kwargs)


def good(x, should_be):
    return good_style if x == should_be else bad_style


def fmin(sfmin):
    """Display FunctionMinum in html representation"""
    return table(
        tr(
            td(
                "FCN = %.4g" % sfmin.fval,
                colspan=2,
                title="Minimum value of function",
                style="text-align:center",
            ),
            td(
                "Ncalls = %i (%i total)" % (sfmin.nfcn, sfmin.ncalls),
                colspan=3,
                title="No. of calls in last algorithm and total number of calls",
            ),
        ),
        tr(
            td(
                "EDM = %.3g (Goal: %g)" % (sfmin.edm, goaledm(sfmin)),
                colspan=2,
                style="text-align:center",
                title="Estimated distance to minimum and target threshold",
            ),
            td(
                "up = %.1f" % sfmin.up,
                colspan=3,
                title="Increase in FCN which corresponds to 1 standard deviation",
            ),
        ),
        tr(
            td("Valid Min.", title="Validity of the migrad call"),
            td("Valid Param.", title="Validity of parameters"),
            td("Above EDM", title="Is EDM above goal EDM?"),
            td(
                "Reached call limit",
                colspan=2,
                title="Did last migrad call reach max call limit?",
            ),
        ),
        tr(
            td(str(sfmin.is_valid), style=good(sfmin.is_valid, True)),
            td(
                str(sfmin.has_valid_parameters),
                style=good(sfmin.has_valid_parameters, True),
            ),
            td(str(sfmin.is_above_max_edm), style=good(sfmin.is_above_max_edm, False)),
            td(
                str(sfmin.has_reached_call_limit),
                colspan=2,
                style=good(sfmin.has_reached_call_limit, False),
            ),
        ),
        tr(
            td("Hesse failed", title="Did Hesse fail?"),
            td("Has cov.", title="Has covariance matrix"),
            td("Accurate", title="Is covariance matrix accurate?"),
            td("Pos. def.", title="Is covariance matrix positive definite?"),
            td("Forced", title="Was positive definiteness enforced by Minuit?"),
        ),
        tr(
            td(str(sfmin.hesse_failed), style=good(sfmin.hesse_failed, False)),
            td(str(sfmin.has_covariance), style=good(sfmin.has_covariance, True)),
            td(
                str(sfmin.has_accurate_covar),
                style=good(sfmin.has_accurate_covar, True),
            ),
            td(str(sfmin.has_posdef_covar), style=good(sfmin.has_posdef_covar, True)),
            td(
                str(sfmin.has_made_posdef_covar),
                style=good(sfmin.has_made_posdef_covar, False),
            ),
        ),
    )


def params(mps):
    mes = mps.merrors

    # body
    rows = []
    for i, mp in enumerate(mps):
        if mes and mp.name in mes:
            me = mes[mp.name]
            v, e, mem, mep = pdg_format(mp.value, mp.error, me.lower, me.upper)
        else:
            v, e = pdg_format(mp.value, mp.error)
            mem = ""
            mep = ""
        rows.append(
            tr(
                th(str(i)),
                td(mp.name),
                td(v),
                td(e),
                td(mem),
                td(mep),
                td("%.3G" % mp.lower_limit if mp.lower_limit is not None else ""),
                td("%.3G" % mp.upper_limit if mp.upper_limit is not None else ""),
                td("yes" if mp.is_fixed else ("CONST" if mp.is_const else "")),
                style=backgrounds[(i + 1) % 2],
            )
        )

    return table(
        # header
        tr(
            "<td/>",
            th("Name", title="Variable name"),
            th("Value", title="Value of parameter"),
            th("Hesse Error", title="Hesse error"),
            th("Minos Error-", title="Minos lower error"),
            th("Minos Error+", title="Minos upper error"),
            th("Limit-", title="Lower limit of the parameter"),
            th("Limit+", title="Upper limit of the parameter"),
            th("Fixed", title="Is the parameter fixed in the fit"),
            style=backgrounds[0],
        ),
        # body
        *rows,
    )


def merrors(mes):
    return table(
        tr(
            "<td/>",
            (
                th(
                    me.name,
                    colspan=2,
                    title="Parameter name",
                    style="text-align:center",
                )
                for me in mes
            ),
        ),
        tr(
            th("Error", title="Lower and upper minos error of the parameter"),
            ((td(x) for x in pdg_format(None, me.lower, me.upper)) for me in mes),
        ),
        tr(
            th("Valid", title="Validity of lower/upper minos error"),
            (
                (
                    td(str(x), style=good(x, True))
                    for x in (me.lower_valid, me.upper_valid)
                )
                for me in mes
            ),
        ),
        tr(
            th("At Limit", title="Did scan hit limit of any parameter?"),
            (
                (
                    td(str(x), style=good(x, False))
                    for x in (me.at_lower_limit, me.at_upper_limit)
                )
                for me in mes
            ),
        ),
        tr(
            th("Max FCN", title="Did scan hit function call limit?"),
            (
                (
                    td(str(x), style=good(x, False))
                    for x in (me.at_lower_max_fcn, me.at_upper_max_fcn)
                )
                for me in mes
            ),
        ),
        tr(
            th("New Min", title="New minimum found when doing scan?"),
            (
                (
                    td(str(x), style=good(x, False))
                    for x in (me.lower_new_min, me.upper_new_min)
                )
                for me in mes
            ),
        ),
    )


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
        nums = matrix_format(*args)

    grad = Gradient(
        (-1.0, 120.0, 120.0, 250.0),
        (0.0, 250.0, 250.0, 250.0),
        (1.0, 250.0, 100.0, 100.0),
    )

    rows = []
    for i, v in enumerate(m.names):
        cols = [th(v)]
        for j in range(len(m.names)):
            val = m[i][j]
            if is_correlation:
                if i == j:
                    t = td("1.00")
                else:
                    color = grad.rgb(val)
                    t = td("%5.2f" % val, style="background-color:" + color)
            else:
                if i == j:
                    t = td(nums[n * i + j])
                else:
                    color = grad.rgb(val / (m[i][i] * m[j][j]) ** 0.5)
                    t = td(nums[n * i + j], style="background-color:" + color)
            cols.append(t)
        rows.append(tr(cols))

    return table(tr("<td/>", (th(v) for v in m.names)), rows)
