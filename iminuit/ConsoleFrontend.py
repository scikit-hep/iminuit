class ConsoleFrontend:
    """Minuit console front end class
    This class print stuff directly via print
    #TODO: add some color
    """
    def print_fmin(self, sfmin, tolerance=None, ncalls=0):
        """display function minimum information
        for FunctionMinimumStruct *sfmin*.
        It contains various migrad status.
        """
        fmin = sfmin
        goaledm = 0.0001*tolerance*fmin.up if tolerance is not None else ''
        #despite what the doc said the code is actually 1e-4
        #http://wwwasdoc.web.cern.ch/wwwasdoc/hbook_html3/node125.html
        flatlocal = dict(locals().items()+fmin.__dict__.items())
        info1 = 'fval = %(fval)r | total call = %(ncalls)r | ncalls = %(nfcn)r\n' %\
                flatlocal
        info2 = 'edm = %(edm)r (Goal: %(goaledm)r) | up = %(up)r\n' % flatlocal
        header1 = '|' + (' %14s |'*5) % (
                    'Valid',
                    'Valid Param',
                    'Accurate Covar',
                    'Posdef',
                    'Made Posdef')+'\n'
        hline = '-'*len(header1)+'\n'
        status1 = '|' + (' %14r |'*5) % (
                    fmin.is_valid,
                    fmin.has_valid_parameters,
                    fmin.has_accurate_covar,
                    fmin.has_posdef_covar,
                    fmin.has_made_posdef_covar)+'\n'
        header2 = '|' + (' %14s |'*5) % (
                    'Hesse Fail',
                    'Has Cov',
                    'Above EDM',
                    '',
                    'Reach calllim')+'\n'
        status2 = '|' + (' %14r |'*5) % (
                    fmin.hesse_failed,
                    fmin.has_covariance,
                    fmin.is_above_max_edm,
                    '',
                    fmin.has_reached_call_limit)+'\n'

        print hline + info1 + info2 +\
            hline + header1 + hline + status1 +\
            hline + header2 + hline+ status2 +\
            hline

    def print_merror(self, vname, smerr):
        """print minos error for varname"""
        stat = 'VALID' if smerr.is_valid else 'PROBLEM'

        summary = 'Minos Status for %s: %s\n'%\
                (vname, stat)

        error = '| {0:^15s} | {1: >12g} | {2: >12g} |\n'.format(
                    'Error',
                    smerr.
                    lower,
                    smerr.upper)
        valid = '| {0:^15s} | {1:^12s} | {2:^12s} |\n'.format(
                    'Valid',
                    str(smerr.lower_valid),
                    str(smerr.upper_valid))
        at_limit='| {0:^15s} | {1:^12s} | {2:^12s} |\n'.format(
                    'At Limit',
                    str(smerr.at_lower_limit),
                    str(smerr.at_upper_limit))
        max_fcn='| {0:^15s} | {1:^12s} | {2:^12s} |\n'.format(
                    'Max FCN',
                    str(smerr.at_lower_max_fcn),
                    str(smerr.at_upper_max_fcn))
        new_min='| {0:^15s} | {1:^12s} | {2:^12s} |\n'.format(
                    'New Min',
                    str(smerr.lower_new_min),
                    str(smerr.upper_new_min))
        hline = '-'*len(error)+'\n'
        print hline +\
              summary +\
              hline +\
              error +\
              valid +\
              at_limit +\
              max_fcn +\
              new_min +\
              hline

    def print_param(self, mps, merr=None, float_format=None):
        """Print parameter states

        Arguments:

            *mps*: list of MinuitParameter struct

            *merr*: dictionary of vname->minos error struct

            *float_format*: ignored
        """
        def lud(m, k, d):
            #lookup map with default
            return m[k] if k in m else d
        merr = {} if merr is None else merr
        vnames = [mp.name for mp in mps]
        maxlength = max([len(x) for x in vnames])
        maxlength = max(5, maxlength)

        header = (('| {0:^4s} | {1:^%ds} | {2:^8s} | {3:^8s} | {4:^8s} |'
                  ' {5:^8s} | {6:^8s} | {7:^8s} | {8:^8s} |\n')%maxlength).format(
                    '', 'Name', 'Value', 'Para Err',
                    "Err-", "Err+", "Limit-", "Limit+", " ")
        hline = '-'*len(header)+'\n'
        linefmt = ('| {0:>4d} | {1:>%ds} = {2:<8s} | {3:<8s} | {4:<8s} |'
                  ' {5:<8s} | {6:<8s} | {7:<8s} | {8:^8s} |\n')%maxlength
        nfmt = '{0:< 8.4G}'
        nformat = nfmt.format
        blank = ' '*8

        ret = hline+header+hline
        for i, (v, mp) in enumerate(zip(vnames, mps)):
            tmp = [i, v]

            tmp.append(nfmt.format(mp.value))
            tmp.append(nfmt.format(mp.error))

            tmp.append(nformat(merr[v].lower) if v in merr else blank)
            tmp.append(nformat(merr[v].upper) if v in merr else blank)

            tmp.append(nformat(mp.lower_limit) if mp.has_limits else blank)
            tmp.append(nformat(mp.upper_limit) if mp.has_limits else blank)

            tmp.append(
                'FIXED' if mp.is_fixed else 'CONST' if mp.is_const else '')

            line = linefmt.format(*tmp)
            ret+=line
        ret+=hline
        print ret

    def print_banner(self, cmd):
        """show banner of command"""
        ret = '*'*50+'\n'
        ret += '*{:^48}*'.format(cmd)+'\n'
        ret += '*'*50+'\n'
        print ret

    def print_matrix(self, vnames, matrix):
        """TODO: properly implement this"""
        print vnames
        print matrix
        maxlen = max(len(v) for v in vnames)
        narg = len(matrix)
        vfmt = '%%%ds'%maxlen
        vblank = ' '*maxlen
        fmt = '%3.2f ' # 4char
        dfmt = '%4d '
        tmp = ''
        header = vblank+' '*4+'  | '+(dfmt*narg)%tuple(range(narg))+'\n'
        blank_line = '-'*len(header)+'\n'
        tmp += header + blank_line
        for i, (v, row) in enumerate(zip(vnames, matrix)):
            fmt = '%3.2f '*narg
            head = (vfmt+' %4d | ')%(v, i)
            content = (fmt)%tuple(row)+'\n'
            tmp += head + content
        tmp += blank_line
        print tmp

    def print_hline(self):
        print '*'*70
