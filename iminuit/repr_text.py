from __future__ import (absolute_import, division, unicode_literals)
from math import log10


def fmin(fmin):
    goaledm = 1e-4 * fmin.tolerance * fmin.up
    # despite what the doc said the code is actually 1e-4
    # http://wwwasdoc.web.cern.ch/wwwasdoc/hbook_html3/node125.html
    info1 = 'fval = %.3g | total call = %i | ncalls = %i' % (fmin.fval, fmin.ncalls, fmin.nfcn)
    info2 = 'edm = %.3g (Goal: %.3g) | up = %.1f' % (fmin.edm, goaledm, fmin.up)
    header1 = '|' + (' %14s |' * 5) % (
        'Valid',
        'Valid Param',
        'Accurate Covar',
        'Posdef',
        'Made Posdef')
    hline = '-' * len(header1)
    status1 = '|' + (' %14r |' * 5) % (
        fmin.is_valid,
        fmin.has_valid_parameters,
        fmin.has_accurate_covar,
        fmin.has_posdef_covar,
        fmin.has_made_posdef_covar)
    header2 = '|' + (' %14s |' * 5) % (
        'Hesse Fail',
        'Has Cov',
        'Above EDM',
        '',
        'Reach calllim')
    status2 = '|' + (' %14s |' * 5) % (
        fmin.hesse_failed,
        fmin.has_covariance,
        fmin.is_above_max_edm,
        '',
        fmin.has_reached_call_limit)

    return "\n".join((hline, info1, info2,
                 hline, header1, hline, status1,
                 hline, header2, hline, status2,
                 hline))

    
def merror(me):
    stat = 'Valid' if me.is_valid else 'Invalid'
    summary = '| Minos Status for %s: %s' % (me.name, stat)
    summary += max(48 - len(summary), 0) * ' ' + '|'

    error = '| {0:^15s} | {1:^12g} | {2:^12g} |'.format(
        'Error', me.lower, me.upper)
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
    def row_fmt(args):
        s = '| ' + args[0] + ' |'
        for x in args[1:]:
            s += ' ' + x
        s += ' |'
        return s

    first_row_width = max(len(v) for v in m.names)
    row_width = max(first_row_width, 5)
    v_names = [('{:>%is}' % first_row_width).format(x) for x in m.names]
    h_names = [('{:>%is}' % row_width).format(x) for x in m.names]
    val_fmt = '{:%i.2f}' % row_width

    header = row_fmt([' ' * first_row_width] + h_names)
    hline = '-' * len(header)
    lines = [hline, header, hline]
    for (vn, row) in zip(v_names, m):
        lines.append(row_fmt([vn] + [val_fmt.format(x) for x in row]))
    lines.append(hline)
    return "\n".join(lines)
    
    
def params(mps, mes=None):
    # TODO include Minos data
    vnames = [mp.name for mp in mps]
    name_width = max([4] + [len(x) for x in vnames])
    num_max = len(vnames) - 1
    num_width = max(2, int(log10(max(num_max, 1)) + 1))

    header = (('| {0:^%is} | {1:^%is} | {2:^8s} | {3:^8s} | {4:^8s} |'
               ' {5:^8s} | {6:8s} | {7:8s} | {8:^5s} |') %
              (num_width, name_width)).format(
        'No', 'Name', 'Value', 'Sym. Err',
        "Err-", "Err+", "Limit-", "Limit+", "Fixed")
    hline = '-' * len(header)
    linefmt = (('| {0:>%id} | {1:>%is} | {2:<9s}| {3:<9s}| {4:<9s}|'
                ' {5:<9s}| {6:9s}| {7:9s}| {8:^5s} |') %
               (num_width, name_width))
    nfmt = '{0:<9.3G}'
    nformat = nfmt.format
    blank = ' ' * 8

    lines = [hline, header, hline]
    for i, mp in enumerate(mps):
        v = mp.name
        line = linefmt.format(
            i, mp.name,
            nformat(mp.value),
            nformat(mp.error),
            nformat(mes[v].lower) if mes and v in mes else blank,
            nformat(mes[v].upper) if mes and v in mes else blank,
            nformat(mp.lower_limit) if mp.lower_limit is not None else blank,
            nformat(mp.upper_limit) if mp.upper_limit is not None else blank,
            'Yes' if mp.is_fixed else 'CONST' if mp.is_const else ''
        )
        lines.append(line)
    lines.append(hline)
    return "\n".join(lines)
