from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from warnings import warn
from iminuit.iminuit_warnings import InitialParamWarning
from iminuit import util as mutil
import numpy as np


def pedantic(self, parameters, kwds, errordef):
    def w(msg):
        warn(msg, InitialParamWarning, stacklevel=3)
    for vn in parameters:
        if vn not in kwds:
            w('Parameter %s does not have initial value. Assume 0.' % vn)
        if 'error_' + vn not in kwds and 'fix_' + mutil.param_name(vn) not in kwds:
            w('Parameter %s is floating but does not have initial step size. Assume 1.' % vn)
    for vlim in mutil.extract_limit(kwds):
        if mutil.param_name(vlim) not in parameters:
            w('%s is given. But there is no parameter %s. Ignore.' % (vlim, mutil.param_name(vlim)))
    for vfix in mutil.extract_fix(kwds):
        if mutil.param_name(vfix) not in parameters:
            w('%s is given. But there is no parameter %s. Ignore.' % (vfix, mutil.param_name(vfix)))
    for verr in mutil.extract_error(kwds):
        if mutil.param_name(verr) not in parameters:
            w('%s float. But there is no parameter %s. Ignore.' % (verr, mutil.param_name(verr)))
    if errordef is None:
        w('errordef is not given. Default to 1.')


def draw_profile(self, vname, x, y, s=None, band=True, text=True):
    from matplotlib import pyplot as plt

    x = np.array(x)
    y = np.array(y)
    if s is not None:
        s = np.array(s, dtype=bool)
        x = x[s]
        y = y[s]

    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(vname)
    plt.ylabel('FCN')

    try:
        minpos = np.argmin(y)

        # Scan to the right of minimum until greater than min + errordef.
        # Note: We need to find the *first* crossing of up, right from the
        # minimum, because there can be several. If the loop is replaced by
        # some numpy calls, make sure that this property is retained.
        yup = self.errordef + y[minpos]
        best = float("infinity")
        for i in range(minpos, len(y)):
            z = abs(y[i] - yup)
            if z < best:
                rightpos = i
                best = z
            else:
                break
        else:
            raise ValueError("right edge not found")

        # Scan to the left of minimum until greater than min + errordef.
        best = float("infinity")
        for i in range(minpos, 0, -1):
            z = abs(y[i] - yup)
            if z < best:
                leftpos = i
                best = z
            else:
                break
        else:
            raise ValueError("left edge not found")

        plt.plot([x[leftpos], x[minpos], x[rightpos]],
                 [y[leftpos], y[minpos], y[rightpos]], 'o')

        if band:
            plt.axvspan(x[leftpos], x[rightpos], facecolor='g', alpha=0.5)

        if text:
            plt.title('%s = %.3g - %.3g + %.3g (scan)' % (vname, x[minpos],
                                                          x[minpos] - x[leftpos],
                                                          x[rightpos] - x[minpos]),
                      fontsize="large")
    except ValueError:
        warn(RuntimeWarning('band and text is requested but '
                            'the bound is too narrow.'))

    return x, y, s


def draw_contour(self, x, y, bins=20, bound=2, args=None, show_sigma=False):
    from matplotlib import pyplot as plt
    vx, vy, vz = self.contour(x, y, bins, bound, args, subtract_min=True)

    v = [self.errordef * ((i + 1) ** 2) for i in range(bound)]

    CS = plt.contour(vx, vy, vz, v, colors=['b', 'k', 'r'])
    if not show_sigma:
        plt.clabel(CS, v)
    else:
        tmp = dict((vv, r'%i $\sigma$' % (i + 1)) for i, vv in enumerate(v))
        plt.clabel(CS, v, fmt=tmp, fontsize=16)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.axhline(self.values[y], color='k', ls='--')
    plt.axvline(self.values[x], color='k', ls='--')
    plt.grid(True)
    return vx, vy, vz


def draw_mncontour(self, x, y, nsigma=2, numpoints=20):
    from matplotlib import pyplot as plt
    from matplotlib.contour import ContourSet

    c_val = []
    c_pts = []
    for sigma in range(1, nsigma + 1):
        pts = self.mncontour(x, y, numpoints, sigma)[2]
        # close curve
        pts.append(pts[0])
        c_val.append(sigma)
        c_pts.append([pts])  # level can have more than one contour in mpl
    cs = ContourSet(plt.gca(), c_val, c_pts)
    plt.clabel(cs, inline=1, fontsize=10)
    plt.xlabel(x)
    plt.ylabel(y)
    return cs
