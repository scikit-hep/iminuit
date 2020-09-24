from .pdg_format import _round, _strip
import numpy as np
import re


def pdg_format(value, *errors):
    if value is None:
        strings, nexp = _round((0, *errors), None, None)
        strings = strings[1:]
    else:
        strings, nexp = _round((value, *errors), None, None)
    strings = _strip(strings)
    if nexp != 0:
        for i, s in enumerate(strings):
            if s[-1] in "fn":
                continue
            m = None
            if i == 0:
                m = re.match(r"(-?)0\.0+$", s)
                if m:
                    s = m.group(1) + "0"
            suffix = ""
            if not m:
                suffix = "e%i" % nexp
            s += suffix
            strings[i] = s
    return strings


def matrix_format(*values):
    vs = np.array(values)
    mv = np.max(np.abs(vs))
    smv = "%.3g" % mv
    try:
        i = smv.index("e")
        sexp = smv[i + 1 :]
        exp = int(sexp)
        vs /= 10 ** exp
        s = [("%.3fe%i" % (v, exp) if np.isfinite(v) else str(v)) for v in vs]
    except ValueError:
        s = ["%.3f" % v for v in vs]
    return _strip(s)


def goaledm(fm):
    # - taken from the source code, see VariableMeticBuilder::Minimum and
    #   ModularFunctionMinimizer::Minimize
    # - goal is used to detect convergence but violations by 10x are also accepted;
    #   see VariableMetricBuilder.cxx:425
    mn_eps_2 = 4 * np.sqrt(np.finfo("float").eps)
    return 2e-3 * max(fm.tolerance * fm.up, mn_eps_2)


def format_row(widths, *args):
    return (
        "".join(
            ("│{0:^%i}" % w if w > 0 else "│ {0:%i}" % (-w - 1)).format(a)
            for (w, a) in zip(widths, args)
        )
        + "│"
    )


def format_line(widths, edges):
    s = edges[0]
    for w, e in zip(widths, edges[1:]):
        s += "─" * abs(w)
        s += e
    return s


def fmin_fields(fm):
    return [
        "FCN = %.4g" % fm.fval,
        "Nfcn = %i (%i total)" % (fm.nfcn, fm.nfcn_total),
        "EDM = %.3g (Goal: %.3g)" % (fm.edm, goaledm(fm)),
        "Ngrad = %i (%i total)" % (fm.ngrad, fm.ngrad_total)
        if fm.ngrad_total > 0
        else "",
        ("Valid" if fm.is_valid else "INVALID") + " Minimum",
        ("Valid" if fm.has_valid_parameters else "INVALID") + " Parameters",
        ("SOME" if fm.has_parameters_at_limit else "No") + " Parameters at limit",
        ("ABOVE" if fm.is_above_max_edm else "Below") + " EDM threshold (goal x 10)",
        ("ABOVE" if fm.has_reached_call_limit else "Below") + " call limit",
        "Hesse " + ("FAILED" if fm.hesse_failed else "ok"),
        ("Has" if fm.has_covariance else "NO") + " Covariance",
        "Accurate" if fm.has_accurate_covar else "APPROXIMATE",
        "Pos. def." if fm.has_posdef_covar else "NOT pos. def.",
        "FORCED" if fm.has_made_posdef_covar else "Not forced",
    ]


def fmin(fm):
    ff = fmin_fields(fm)
    w = (-34, 38)
    l1 = format_line(w, "┌┬┐")
    i1 = format_row(w, *ff[0:2])
    i2 = format_row(w, *ff[2:4])
    w = (15, 18, 38)
    l2 = format_line(w, "├┬┼┤")
    v1 = format_row(w, *ff[4:7])
    l3 = format_line(w, "├┴┼┤")
    v2 = format_row((34, 38), *ff[7:9])
    w = (15, 18, 11, 13, 12)
    l4 = format_line(w, "├┬┼┬┬┤")
    v3 = format_row(w, *ff[9:14])
    l5 = format_line(w, "└┴┴┴┴┘")
    return "\n".join((l1, i1, i2, l2, v1, l3, v2, l4, v3, l5))


def params(mps):
    vnames = [mp.name for mp in mps]
    name_width = max([4] + [len(x) for x in vnames])
    num_width = max(2, len("%i" % (len(vnames) - 1)))

    ws = (-num_width - 1, -name_width - 2, 11, 11, 12, 12, 9, 9, 7)
    h = format_row(
        ws,
        "",
        "Name",
        "Value",
        "Hesse Err",
        "Minos Err-",
        "Minos Err+",
        "Limit-",
        "Limit+",
        "Fixed",
    )
    ni = len(ws) - 1
    l1 = format_line(ws, "┌" + "┬" * ni + "┐")
    l2 = format_line(ws, "├" + "┼" * ni + "┤")
    lines = [l1, h, l2]
    mes = mps.merrors
    for i, mp in enumerate(mps):
        if mes and mp.name in mes:
            me = mes[mp.name]
            val, err, mel, meu = pdg_format(mp.value, mp.error, me.lower, me.upper)
        else:
            val, err = pdg_format(mp.value, mp.error)
            mel = ""
            meu = ""
        lines.append(
            format_row(
                ws,
                str(i),
                mp.name,
                val,
                err,
                mel,
                meu,
                "%g" % mp.lower_limit if mp.lower_limit is not None else "",
                "%g" % mp.upper_limit if mp.upper_limit is not None else "",
                "yes" if mp.is_fixed else "CONST" if mp.is_const else "",
            )
        )
    ln3 = format_line(ws, "└" + "┴" * ni + "┘")
    lines.append(ln3)
    return "\n".join(lines)


def merrors(mes):
    n = len(mes)
    ws = [10] + [23] * n
    l1 = format_line(ws, "┌" + "┬" * n + "┐")
    header = format_row(ws, "", *(m.name for m in mes))
    ws = [10] + [11] * (2 * n)
    l2 = format_line(ws, "├" + "┼┬" * n + "┤")
    l3 = format_line(ws, "└" + "┴" * n * 2 + "┘")
    x = []
    for m in mes:
        mel, meu = pdg_format(None, m.lower, m.upper)
        x.append(mel)
        x.append(meu)
    error = format_row(ws, "Error", *x)
    x = []
    for m in mes:
        x.append(str(m.lower_valid))
        x.append(str(m.upper_valid))
    valid = format_row(ws, "Valid", *x)
    x = []
    for m in mes:
        x.append(str(m.at_lower_limit))
        x.append(str(m.at_upper_limit))
    at_limit = format_row(ws, "At Limit", *x)
    x = []
    for m in mes:
        x.append(str(m.at_lower_max_fcn))
        x.append(str(m.at_upper_max_fcn))
    max_fcn = format_row(ws, "Max FCN", *x)
    x = []
    for m in mes:
        x.append(str(m.lower_new_min))
        x.append(str(m.upper_new_min))
    new_min = format_row(ws, "New Min", *x)
    return "\n".join((l1, header, l2, error, valid, at_limit, max_fcn, new_min, l3))


def matrix(m):
    n = len(m)

    args = []
    for mi in m:
        for mj in mi:
            args.append(mj)
    nums = matrix_format(*args)

    def row_fmt(args):
        s = "│ " + args[0] + " │"
        for x in args[1:]:
            s += " " + x
        s += " │"
        return s

    first_row_width = max(len(v) for v in m.names)
    row_width = max(first_row_width, max(len(v) for v in nums))
    v_names = [("{:>%is}" % first_row_width).format(x) for x in m.names]
    h_names = [("{:>%is}" % row_width).format(x) for x in m.names]
    val_fmt = ("{:>%is}" % row_width).format

    w = (first_row_width + 2, (row_width + 1) * len(m.names) + 1)
    l1 = format_line(w, "┌┬┐")
    l2 = format_line(w, "├┼┤")
    l3 = format_line(w, "└┴┘")

    header = row_fmt([" " * first_row_width] + h_names)
    lines = [l1, header, l2]

    for i, vn in enumerate(v_names):
        lines.append(row_fmt([vn] + [val_fmt(nums[n * i + j]) for j in range(n)]))
    lines.append(l3)
    return "\n".join(lines)
