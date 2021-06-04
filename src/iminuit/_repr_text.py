from .pdg_format import _round, _strip
import re
import numpy as np


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


def format_row(widths, *args) -> str:
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
    rc = fm.reduced_chi2
    return [
        fm.algorithm,
        f"FCN = {fm.fval:.4g}"
        + (f" (chi2/ndof = {rc:.1f})" if not np.isnan(rc) else ""),
        f"Nfcn = {fm.nfcn}",
        f"EDM = {fm.edm:.3g} (Goal: {fm.edm_goal:.3g})",
        f"Ngrad = {fm.ngrad}" if fm.ngrad > 0 else "",
        f"{'Valid' if fm.is_valid else 'INVALID'} Minimum",
        f"{'SOME' if fm.has_parameters_at_limit else 'No'} Parameters at limit",
        f"{'ABOVE' if fm.is_above_max_edm else 'Below'} EDM threshold (goal x 10)",
        f"{'ABOVE' if fm.has_reached_call_limit else 'Below'} call limit",
        f"{'' if fm.has_covariance else 'NO '}Covariance",
        f"Hesse {'FAILED' if fm.hesse_failed else 'ok'}",
        "Accurate" if fm.has_accurate_covar else "APPROXIMATE",
        "Pos. def." if fm.has_posdef_covar else "NOT pos. def.",
        "FORCED" if fm.has_made_posdef_covar else "Not forced",
    ]


def fmin(fm):
    ff = fmin_fields(fm)
    w = (73,)
    l1 = format_line(w, "┌┐")
    i1 = format_row(w, ff[0] + "   ")
    w = (-34, 38)
    l2 = format_line(w, "├┬┤")
    i2 = format_row(w, *ff[1:3])
    i3 = format_row(w, *ff[3:5])
    w = (34, 38)
    l3 = format_line(w, "├┼┤")
    v1 = format_row(w, *ff[5:7])
    l4 = format_line(w, "├┼┤")
    v2 = format_row((34, 38), *ff[7:9])
    w = (15, 18, 11, 13, 12)
    l5 = format_line(w, "├┬┼┬┬┤")
    v3 = format_row(w, *ff[9:14])
    l6 = format_line(w, "└┴┴┴┴┘")
    return "\n".join((l1, i1, l2, i2, i3, l3, v1, l4, v2, l5, v3, l6))


def params(mps):
    vnames = (mp.name for mp in mps)
    name_width = max([4] + [len(x) for x in vnames])
    num_width = max(2, len(f"{len(mps) - 1}"))

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
    for i, mp in enumerate(mps):
        me = mp.merror
        if me:
            val, err, mel, meu = pdg_format(mp.value, mp.error, *me)
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
    mes = mes.values()
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


def matrix(arr):
    names = tuple(arr._var2pos)

    n = len(arr)
    nums = matrix_format(arr.flatten())

    def row_fmt(args):
        s = "│ " + args[0] + " │"
        for x in args[1:]:
            s += " " + x
        s += " │"
        return s

    first_row_width = max(len(v) for v in names)
    row_width = max(first_row_width, max(len(v) for v in nums))
    v_names = [("{:>%is}" % first_row_width).format(x) for x in names]
    h_names = [("{:>%is}" % row_width).format(x) for x in names]
    val_fmt = ("{:>%is}" % row_width).format

    w = (first_row_width + 2, (row_width + 1) * len(names) + 1)
    l1 = format_line(w, "┌┬┐")
    l2 = format_line(w, "├┼┤")
    l3 = format_line(w, "└┴┘")

    header = row_fmt([" " * first_row_width] + h_names)
    lines = [l1, header, l2]

    for i, vn in enumerate(v_names):
        lines.append(row_fmt([vn] + [val_fmt(nums[n * i + j]) for j in range(n)]))
    lines.append(l3)
    return "\n".join(lines)


def matrix_format(values):
    s = [f"{v:.3g}" % v for v in values]
    return _strip(s)
