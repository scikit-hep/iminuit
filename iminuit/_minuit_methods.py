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


def profile(self, vname, bins, bound, subtract_min):
    # center value
    val = np.linspace(bound[0], bound[1], bins, dtype=np.double)
    result = np.empty(bins, dtype=np.double)
    pos = self.var2pos[vname]
    n = val.shape[0]
    arg = list(self.args)
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


def draw_profile(self, vname, x, y, band, text):
    from matplotlib import pyplot as plt

    plt.plot(x, y)
    plt.xlabel(vname)
    plt.ylabel("FCN")

    if vname in self.values:
        v = self.values[vname]
        plt.axvline(v, color="k", linestyle="--")

        vmin = None
        vmax = None
        if (vname, 1) in self.merrors:
            vmin = v + self.merrors[(vname, -1)]
            vmax = v + self.merrors[(vname, 1)]
        if vname in self.errors:
            vmin = v - self.errors[vname]
            vmax = v + self.errors[vname]

        if vmin is not None and band:
            plt.axvspan(vmin, vmax, facecolor="0.8")

        if text:
            plt.title(
                ("%s = %.3g" % (vname, v))
                if vmin is None
                else ("%s = %.3g - %.3g + %.3g" % (vname, v, v - vmin, vmax - v)),
                fontsize="large",
            )

    return x, y


def draw_contour(self, x, y, bins, bound):
    from matplotlib import pyplot as plt

    vx, vy, vz = self.contour(x, y, bins, bound, subtract_min=True)

    v = [self.errordef * (i + 1) for i in range(4)]

    CS = plt.contour(vx, vy, vz, v)
    plt.clabel(CS, v)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.axhline(self.values[y], color="k", ls="--")
    plt.axvline(self.values[x], color="k", ls="--")
    return vx, vy, vz


def draw_mncontour(self, x, y, nsigma, numpoints):
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
    plt.clabel(cs)
    plt.xlabel(x)
    plt.ylabel(y)
    return cs
