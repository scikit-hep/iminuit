from __future__ import absolute_import, division, print_function, unicode_literals
from warnings import warn
from iminuit.iminuit_warnings import InitialParamWarning
from iminuit import util as mutil
import numpy as np


def pedantic(self, parameters, kwds, errordef):
    def w(msg):
        warn(msg, InitialParamWarning, stacklevel=3)

    for vn in parameters:
        if vn not in kwds:
            w("Parameter %s does not have initial value. Assume 0." % vn)
        if "error_" + vn not in kwds and "fix_" + mutil.param_name(vn) not in kwds:
            w(
                "Parameter %s is floating but does not have initial step size. Assume 1."
                % vn
            )
    for vlim in mutil.extract_limit(kwds):
        if mutil.param_name(vlim) not in parameters:
            w(
                "%s is given. But there is no parameter %s. Ignore."
                % (vlim, mutil.param_name(vlim))
            )
    for vfix in mutil.extract_fix(kwds):
        if mutil.param_name(vfix) not in parameters:
            w(
                "%s is given. But there is no parameter %s. Ignore."
                % (vfix, mutil.param_name(vfix))
            )
    for verr in mutil.extract_error(kwds):
        if mutil.param_name(verr) not in parameters:
            w(
                "%s float. But there is no parameter %s. Ignore."
                % (verr, mutil.param_name(verr))
            )
    if errordef is None:
        w("errordef is not given. Default to 1.")


def profile(self, vname, bins, bound, args, subtract_min):
    # center value
    val = np.linspace(bound[0], bound[1], bins, dtype=np.double)
    result = np.empty(bins, dtype=np.double)
    pos = self.var2pos[vname]
    n = val.shape[0]
    arg = list(self.args if args is None else args)
    if self.use_array_call:
        varg = np.array(arg, dtype=np.double)
        for i in range(n):
            varg[pos] = val[i]
            result[i] = self.fcn(varg)
    else:
        for i in range(n):
            arg[pos] = val[i]
            result[i] = self.fcn(*arg)
    if subtract_min:
        result -= self.fval
    return val, result


def draw_profile(self, vname, x, y, s, band, text):
    from matplotlib import pyplot as plt

    if s is not None:
        s = np.array(s, dtype=bool)
        x = x[s]
        y = y[s]

    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel(vname)
    plt.ylabel("FCN")

    if vname in self.values:
        v = self.values[vname]
    else:
        v = np.argmin(y)
    vmin = None
    vmax = None
    if (vname, 1) in self.merrors:
        vmin = v + self.merrors[(vname, -1)]
        vmax = v + self.merrors[(vname, 1)]
    if vname in self.errors:
        vmin = v - self.errors[vname]
        vmax = v + self.errors[vname]

    plt.axvline(v, color="r")

    if vmin is not None and band:
        plt.axvspan(vmin, vmax, facecolor="g", alpha=0.5)

    if text:
        plt.title(
            ("%s = %.3g" % (vname, v))
            if vmin is None
            else ("%s = %.3g - %.3g + %.3g" % (vname, v, v - vmin, vmax - v)),
            fontsize="large",
        )

    return x, y


def draw_contour(self, x, y, bins=20, bound=2, args=None, show_sigma=False):
    from matplotlib import pyplot as plt

    vx, vy, vz = self.contour(x, y, bins, bound, args, subtract_min=True)

    v = [self.errordef * ((i + 1) ** 2) for i in range(bound)]

    CS = plt.contour(vx, vy, vz, v, colors=["b", "k", "r"])
    if not show_sigma:
        plt.clabel(CS, v)
    else:
        tmp = dict((vv, r"%i $\sigma$" % (i + 1)) for i, vv in enumerate(v))
        plt.clabel(CS, v, fmt=tmp, fontsize=16)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.axhline(self.values[y], color="k", ls="--")
    plt.axvline(self.values[x], color="k", ls="--")
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
