from __future__ import (absolute_import, division, unicode_literals)
from math import log10, floor


def format_numbers(*args):
    scales = tuple(int(round(log10(abs(x)))) if x != 0 else 1 for x in args)
    nmax = max(scales)
    nsig = min(scales) - 1
    if nmax <= 3 and nsig >= -3:
        if nsig >= 0:
            return tuple("%i" % round(x, -nsig) for x in args)
        else:
            return tuple(("%%.%if" % min(-nsig, 3)) % x for x in args)
    result = ["%%.%if" % min(nmax-nsig, 3) % (x / 10 ** nmax) for x in args]
    if nmax != 0:
        return tuple(x + 'E%i' % nmax for x in result)
    return tuple(result)


def format_row(widths, *args):
    return "".join(('|{0:^%i}' % (w-1) if w > 0 else '| {0:%i}' % (-w-2)).format(a) for (w, a) in zip(widths, args)) + "|"


def fmin(fmin):
    goaledm = 1e-4 * fmin.tolerance * fmin.up
    # despite what the doc said the code is actually 1e-4
    # http://wwwasdoc.web.cern.ch/wwwasdoc/hbook_html3/node125.html

    ws = (-32, 33)
    i1 = format_row(ws, 'FCN = %.4G' % fmin.fval,
        'Ncalls=%i (%i total)' % (fmin.nfcn, fmin.ncalls))
    i2 = format_row(ws, 'EDM = %.3G (Goal: %G)' % (fmin.edm, goaledm),
        'up = %.1f' % fmin.up)
    ws = (16, 16, 12, 21)
    h1 = format_row(ws, "Valid Min.", "Valid Param.", "Above EDM", "Reached call limit")
    v1 = format_row(ws, repr(fmin.is_valid), repr(fmin.has_valid_parameters),
                         repr(fmin.is_above_max_edm), repr(fmin.has_reached_call_limit))
    ws = (16, 16, 12, 12, 9)
    h2 = format_row(ws, "Hesse failed", "Has cov.", "Accurate", "Pos. def.", "Forced")
    v2 = format_row(ws, repr(fmin.hesse_failed), repr(fmin.has_covariance),
        repr(fmin.has_accurate_covar), repr(fmin.has_posdef_covar),
        repr(fmin.has_made_posdef_covar))
    
    l = len(h1) * "-"
    return "\n".join((l, i1, i2, l, h1, l, v1, l, h2, l, v2, l))


def params(mps):
    vnames = [mp.name for mp in mps]
    name_width = max([4] + [len(x) for x in vnames])
    num_width = max(2, len("%i" % (len(vnames) - 1)))

    ws = (-num_width - 2, -name_width - 3, 12, 12, 13, 13, 10, 10, 8)
    h = format_row(ws, '', 'Name', 'Value', 'Hesse Err',
        "Minos Err-", "Minos Err+", "Limit-", "Limit+", "Fixed")
    l = '-' * len(h)
    lines = [l, h, l]
    mes = mps.merrors
    for i, mp in enumerate(mps):
        if mes and mp.name in mes:
            me = mes[mp.name]
            val, err, mel, meu = format_numbers(mp.value, mp.error, me.lower, me.upper)
        else:
            val, err = format_numbers(mp.value, mp.error)
            mel = ''
            meu = ''
        lines.append(format_row(ws,
            str(i), mp.name,
            val,
            err,
            mel,
            meu,
            '%g' % mp.lower_limit if mp.lower_limit is not None else '',
            '%g' % mp.upper_limit if mp.upper_limit is not None else '',
            'yes' if mp.is_fixed else 'CONST' if mp.is_const else ''
        ))
    lines.append(l)
    return "\n".join(lines)


def merror(me):
    mel, meu = format_numbers(me.lower, me.upper)
    stat = 'Valid' if me.is_valid else 'Invalid'
    summary = '| {0:^15s} | {1:^27s} |'.format(me.name, stat)
    error = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
        'Error', mel, meu)
    valid = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
        'Valid', str(me.lower_valid), str(me.upper_valid))
    at_limit = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
        'At Limit', str(me.at_lower_limit), str(me.at_upper_limit))
    max_fcn = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
        'Max FCN', str(me.at_lower_max_fcn), str(me.at_upper_max_fcn))
    new_min = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
        'New Min', str(me.lower_new_min), str(me.upper_new_min))
    hline = '-' * len(error)
    return "\n".join((hline, summary, hline, error, valid,
                      at_limit, max_fcn, new_min, hline))


def matrix(m):
    n = len(m)

    is_correlation = True
    for i in range(n):
        if m[i][i] != 1.0:
            is_correlation = False
            break

    if not is_correlation:
        args = []
        for mi in m:
            for mj in mi:
                args.append(mj)
        nums = format_numbers(*args)        

    def row_fmt(args):
        s = '| ' + args[0] + ' |'
        for x in args[1:]:
            s += ' ' + x
        s += ' |'
        return s

    first_row_width = max(len(v) for v in m.names)
    row_width = max(first_row_width, 6)
    v_names = [('{:>%is}' % first_row_width).format(x) for x in m.names]
    h_names = [('{:>%is}' % row_width).format(x) for x in m.names]
    val_fmt = (('{:%i.2f}' if is_correlation else '{:>%is}') % row_width).format

    header = row_fmt([' ' * first_row_width] + h_names)
    hline = '-' * len(header)
    lines = [hline, header, hline]
    
    for i, vn in enumerate(v_names):
        lines.append(row_fmt([vn] + [val_fmt(m[i][j] if is_correlation else nums[n*i + j]) for j in range(n)]))
    lines.append(hline)
    return "\n".join(lines)
