from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings
import numpy as np

__all__ = ['draw_contour',
           'draw_mncontour',
           'draw_profile']


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
        ymin = y[minpos]
        tmpy = y - ymin
        # now scan for minpos to the right until greater than one
        up = self.errordef
        righty = np.power(tmpy[minpos:] - up, 2)
        right_min = np.argmin(righty)
        rightpos = right_min + minpos
        lefty = np.power((tmpy[:minpos] - up), 2)
        left_min = np.argmin(lefty)
        leftpos = left_min
        le = x[minpos] - x[leftpos]
        re = x[rightpos] - x[minpos]

        if band:
            plt.axvspan(x[leftpos], x[rightpos], facecolor='g', alpha=0.5)

        if text:
            plt.figtext(0.5, 0.5,
                        '%s = %7.3e ( -%7.3e , +%7.3e)' % (vname, x[minpos], le, re),
                        ha='center')
    except ValueError:
        warnings.warn(RuntimeWarning('band and text is requested but'
                                     ' the bound is too narrow.'))

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
    # from matplotlib.collections import LineCollection
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
