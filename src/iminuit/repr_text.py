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
    # taken from the source code, see VariableMeticBuilder.cxx
    return 2e-3 * fm.tolerance * fm.up


def format_row(widths, *args):
    return (
        "".join(
            ("|{0:^%i}" % (w - 1) if w > 0 else "| {0:%i}" % (-w - 2)).format(a)
            for (w, a) in zip(widths, args)
        )
        + "|"
    )


def fmin(fm):
    ws = (-32, 33)
    i1 = format_row(
        ws, "FCN = %.4g" % fm.fval, "Ncalls=%i (%i total)" % (fm.nfcn, fm.ncalls)
    )
    i2 = format_row(
        ws, "EDM = %.3g (Goal: %g)" % (fm.edm, goaledm(fm)), "up = %.1f" % fm.up
    )
    ws = (16, 16, 12, 21)
    h1 = format_row(ws, "Valid Min.", "Valid Param.", "Above EDM", "Reached call limit")
    v1 = format_row(
        ws,
        repr(fm.is_valid),
        repr(fm.has_valid_parameters),
        repr(fm.is_above_max_edm),
        repr(fm.has_reached_call_limit),
    )
    ws = (16, 16, 12, 12, 9)
    h2 = format_row(ws, "Hesse failed", "Has cov.", "Accurate", "Pos. def.", "Forced")
    v2 = format_row(
        ws,
        repr(fm.hesse_failed),
        repr(fm.has_covariance),
        repr(fm.has_accurate_covar),
        repr(fm.has_posdef_covar),
        repr(fm.has_made_posdef_covar),
    )

    ln = len(h1) * "-"
    return "\n".join((ln, i1, i2, ln, h1, ln, v1, ln, h2, ln, v2, ln))


def params(mps):
    vnames = [mp.name for mp in mps]
    name_width = max([4] + [len(x) for x in vnames])
    num_width = max(2, len("%i" % (len(vnames) - 1)))

    ws = (-num_width - 2, -name_width - 3, 12, 12, 13, 13, 10, 10, 8)
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
    ln = "-" * len(h)
    lines = [ln, h, ln]
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
    lines.append(ln)
    return "\n".join(lines)


def merrors(mes):
    n = len(mes)
    ws = [11] + [24] * n
    header = format_row(ws, "", *(m.name for m in mes))
    ws = [11] + [12] * (2 * n)
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
    line = "-" * len(header)
    return "\n".join(
        (line, header, line, error, valid, at_limit, max_fcn, new_min, line)
    )


def matrix(m):
    n = len(m)

    args = []
    for mi in m:
        for mj in mi:
            args.append(mj)
    nums = matrix_format(*args)

    def row_fmt(args):
        s = "| " + args[0] + " |"
        for x in args[1:]:
            s += " " + x
        s += " |"
        return s

    first_row_width = max(len(v) for v in m.names)
    row_width = max(first_row_width, max(len(v) for v in nums))
    v_names = [("{:>%is}" % first_row_width).format(x) for x in m.names]
    h_names = [("{:>%is}" % row_width).format(x) for x in m.names]
    val_fmt = ("{:>%is}" % row_width).format

    header = row_fmt([" " * first_row_width] + h_names)
    hline = "-" * len(header)
    lines = [hline, header, hline]

    for i, vn in enumerate(v_names):
        lines.append(row_fmt([vn] + [val_fmt(nums[n * i + j]) for j in range(n)]))
    lines.append(hline)
    return "\n".join(lines)
