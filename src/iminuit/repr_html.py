from iminuit.color import Gradient
from iminuit.repr_text import pdg_format, matrix_format, fmin_fields

good_style = "background-color:#92CCA6;color:black"
bad_style = "background-color:#c15ef7;color:black"
warn_style = "background-color:#FFF79A;color:black"


def to_str(tag):
    lines = []

    def visit(x, level):
        indent = "    " * level
        if len(x) == 1:
            lines.append(indent + x[0])
        else:
            b, *ch, e = x
            lines.append(indent + b)
            for x in ch:
                visit(x, level + 1)
            lines.append(indent + e)

    visit(tag, 0)

    return "\n".join(lines)


def tag(name, *args, **kwargs):
    # sort keys so that order is same on all platforms
    head = "<" + name
    for k in sorted(kwargs):
        v = kwargs[k]
        head += ' %s="%s"' % (k, v)
    head += ">"
    tail = "</%s>" % name
    if len(args) == 0:
        return [head + tail]
    if len(args) == 1 and isinstance(args[0], str):
        return ["{0} {1} {2}".format(head, args[0], tail)]
    return [head, *args, tail]


def table(*args, **kwargs):
    return tag("table", *args, **kwargs)


def tr(*args, **kwargs):
    return tag("tr", *args, **kwargs)


def th(*args, **kwargs):
    return tag("th", *args, **kwargs)


def td(*args, **kwargs):
    return tag("td", *args, **kwargs)


def good(x, should_be, alt_style=bad_style):
    return good_style if x == should_be else alt_style


def fmin(fm):
    ff = fmin_fields(fm)
    return to_str(
        table(
            tr(
                td(
                    ff[0],
                    colspan=2,
                    style="text-align:left",
                    title="Minimum value of function",
                ),
                td(
                    ff[1],
                    colspan=3,
                    style="text-align:center",
                    title="No. of function evaluations in last call and total number",
                ),
            ),
            tr(
                td(
                    ff[2],
                    colspan=2,
                    style="text-align:left",
                    title="Estimated distance to minimum and goal",
                ),
                td(
                    ff[3],
                    colspan=3,
                    style="text-align:center",
                    title="No. of gradient evaluations in last call and total number",
                ),
            ),
            tr(
                td(ff[4], style="text-align:center;" + good(fm.is_valid, True),),
                td(
                    ff[5],
                    style="text-align:center;" + good(fm.has_valid_parameters, True),
                ),
                td(
                    ff[6],
                    colspan="3",
                    style="text-align:center;"
                    + good(fm.has_parameters_at_limit, False, warn_style),
                ),
            ),
            tr(
                td(
                    ff[7],
                    colspan="2",
                    style="text-align:center;" + good(fm.is_above_max_edm, False),
                ),
                td(
                    ff[8],
                    colspan=3,
                    style="text-align:center;" + good(fm.has_reached_call_limit, False),
                ),
            ),
            tr(
                td(ff[9], style="text-align:center;" + good(fm.hesse_failed, False),),
                td(ff[10], style="text-align:center;" + good(fm.has_covariance, True),),
                td(
                    ff[11],
                    title="Is covariance matrix accurate?",
                    style="text-align:center;"
                    + good(fm.has_accurate_covar, True, warn_style),
                ),
                td(
                    ff[12],
                    style="text-align:center;" + good(fm.has_posdef_covar, True),
                    title="Is covariance matrix positive definite?",
                ),
                td(
                    ff[13],
                    style="text-align:center;" + good(fm.has_made_posdef_covar, False),
                    title="Was positive definiteness enforced by Minuit?",
                ),
            ),
        )
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
            )
        )

    return to_str(
        table(
            # header
            tr(
                td(),
                th("Name", title="Variable name"),
                th("Value", title="Value of parameter"),
                th("Hesse Error", title="Hesse error"),
                th("Minos Error-", title="Minos lower error"),
                th("Minos Error+", title="Minos upper error"),
                th("Limit-", title="Lower limit of the parameter"),
                th("Limit+", title="Upper limit of the parameter"),
                th("Fixed", title="Is the parameter fixed in the fit"),
            ),
            # body
            *rows,
        )
    )


def merrors(mes):
    header = [td()]
    error = [th("Error", title="Lower and upper minos error of the parameter")]
    valid = [th("Valid", title="Validity of lower/upper minos error")]
    limit = [th("At Limit", title="Did scan hit limit of any parameter?")]
    maxfcn = [th("Max FCN", title="Did scan hit function call limit?")]
    newmin = [th("New Min", title="New minimum found when doing scan?")]
    for me in mes:
        header.append(
            th(me.name, colspan=2, title="Parameter name", style="text-align:center",)
        )
        error += [td(x) for x in pdg_format(None, me.lower, me.upper)]
        valid += [
            td(str(x), style=good(x, True)) for x in (me.lower_valid, me.upper_valid)
        ]
        limit += [
            td(str(x), style=good(x, False))
            for x in (me.at_lower_limit, me.at_upper_limit)
        ]
        maxfcn += [
            td(str(x), style=good(x, False))
            for x in (me.at_lower_max_fcn, me.at_upper_max_fcn)
        ]
        newmin += [
            td(str(x), style=good(x, False))
            for x in (me.lower_new_min, me.upper_new_min)
        ]

    return to_str(
        table(
            tr(*header), tr(*error), tr(*valid), tr(*limit), tr(*maxfcn), tr(*newmin),
        )
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
                    t = td(
                        "%5.2f" % val,
                        style="background-color:" + color + ";color:black",
                    )
            else:
                if i == j:
                    t = td(nums[n * i + j])
                else:
                    num = m[i][i] * m[j][j]
                    color = grad.rgb(val / num ** 0.5 if num > 0 else 0)
                    t = td(
                        nums[n * i + j],
                        style="background-color:" + color + ";color:black",
                    )
            cols.append(t)
        rows.append(tr(*cols))

    return to_str(table(tr(td(), *[th(v) for v in m.names]), *rows))
