from .pdg_format import _round, _strip
from iminuit._optional_dependencies import optional_module_for
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
    s_ncall = f"Nfcn = {fm.nfcn}"
    if fm.ngrad > 0:
        s_ncall += f", Ngrad = {fm.ngrad}"

    if fm.hesse_failed:
        covariance_msg1 = "Hesse FAILED"
        if fm.has_reached_call_limit:
            covariance_msg2 = "ABOVE call limit"
        else:
            assert not fm.has_posdef_covar
            covariance_msg2 = "Covariance NOT pos. def."
    else:
        if fm.has_covariance:
            covariance_msg1 = "Hesse ok"
            if fm.has_accurate_covar:
                covariance_msg2 = "Covariance accurate"
            elif fm.has_made_posdef_covar:
                covariance_msg2 = "Covariance FORCED pos. def."
            else:
                covariance_msg2 = "Covariance APPROXIMATE"
        else:
            covariance_msg1 = "Hesse not run"
            covariance_msg2 = "NO covariance"

    return [
        f"{fm.algorithm}",
        f"FCN = {fm.fval:.4g}" + (f" (χ²/ndof = {rc:.1f})" if not np.isnan(rc) else ""),
        s_ncall,
        f"EDM = {fm.edm:.3g} (Goal: {fm.edm_goal:.3g})",
        f"time = {fm.time:.1f} sec" if fm.time >= 0.1 else "",
        f"{'Valid' if fm.is_valid else 'INVALID'} Minimum",
        f"{'ABOVE' if fm.is_above_max_edm else 'Below'} EDM threshold (goal x 10)",
        f"{'SOME' if fm.has_parameters_at_limit else 'No'} parameters at limit",
        f"{'ABOVE' if fm.has_reached_call_limit else 'Below'} call limit",
        covariance_msg1,
        covariance_msg2,
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
    v2 = format_row(w, *ff[7:9])
    l5 = format_line(w, "├┼┤")
    v3 = format_row(w, *ff[9:])
    l6 = format_line(w, "└┴┘")
    return "\n".join((l1, i1, l2, i2, i3, l3, v1, l4, v2, l5, v3, l6))


def params(mps):
    vnames = [_parse_latex(mp.name) for mp in mps]

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
                vnames[i],
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
    header = format_row(ws, "", *(_parse_latex(m.name) for m in mes))
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
    names = [_parse_latex(x) for x in arr._var2pos]

    n = len(arr)
    nums = matrix_format(arr)

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


def matrix_format(matrix):
    r = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j:
                r.append(f"{matrix[i, i]:.3g}")
            else:
                x = pdg_format(matrix[i, j], matrix[i, i], matrix[j, j])[0]
                r.append(x)
    return r


def _parse_latex(s):
    if s.startswith("$") and s.endswith("$"):
        with optional_module_for("rendering simple LaTeX"):
            import unicodeitplus

            return unicodeitplus.parse(s)
    return s
