from warnings import warn
from iminuit.util import InitialParamWarning
import numpy as np


def pedantic(self, parameters, kwds, errordef):
    def w(msg):
        warn(msg, InitialParamWarning, stacklevel=3)

    for vn in parameters:
        if vn not in kwds and "limit_%s" % vn not in kwds:
            w("Parameter %s does not have neither initial value nor limits." % vn)

    if errordef is None:
        w("errordef is not given, defaults to 1.")


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


def check_extra_args(parameters, kwd):
    """Check keyword arguments to find unwanted/typo keyword arguments"""
    fixed_param = {"fix_" + p for p in parameters}
    limit_param = {"limit_" + p for p in parameters}
    error_param = {"error_" + p for p in parameters}
    for k in kwd:
        if (
            k not in parameters
            and k not in fixed_param
            and k not in limit_param
            and k not in error_param
        ):
            raise RuntimeError(
                (
                    "Cannot understand keyword %s. May be a typo?\n"
                    "The parameters are %r"
                )
                % (k, parameters)
            )


# place all deprecated attributes and methods of the Minuit object here
def deprecated(self, key):
    deprecated = {
        "edm": (
            False,
            "this_object.fmin.edm",
            lambda self: self.fmin.edm if self.fmin else None,
        ),
        "merrors_struct": (False, "this_object.merrors", lambda self: self.merrors),
        "ncalls": (
            False,
            "this_object.fmin.nfcn",
            lambda self: self.fmin.nfcn if self.fmin else 0,
        ),
        "ngrads": (
            False,
            "this_object.fmin.ngrad",
            lambda self: self.fmin.ngrad if self.fmin else 0,
        ),
        "get_merrors": (True, "this_object.merrors", lambda: self.merrors),
        "get_fmin": (True, "this_object.fmin", lambda: self.fmin),
        "get_param_states": (True, "this_object.params", lambda: self.params),
        "get_initial_param_states": (
            True,
            "this_object.init_params",
            lambda: self.init_params,
        ),
        "get_num_call_fcn": (
            True,
            "this_object.ncalls_total",
            lambda: self.ncalls_total,
        ),
        "get_num_call_grad": (
            True,
            "this_object.ngrads_total",
            lambda: self.ngrads_total,
        ),
        "is_fixed": (True, "this_object.fixed[arg]", lambda arg: self.fixed[arg]),
        "migrad_ok": (True, "this_object.valid", lambda: self.valid),
        "print_matrix": (
            True,
            "print(this_object.matrix())",
            lambda: print(self.matrix(correlation=True, skip_fixed=True)),
        ),
        "print_fmin": (True, "print(this_object.fmin)", lambda: print(self.fmin)),
        "print_all_minos": (
            True,
            "print(this_object.merrors)",
            lambda: print(self.merrors),
        ),
        "print_param": (True, "print(this_object.params)", lambda: print(self.params)),
        "print_initial_param": (
            True,
            "print(this_object.init_params)",
            lambda: print(self.init_params),
        ),
        "set_errordef": (
            True,
            "this_object.errordef = value",
            lambda arg: setattr(self, "errordef", arg),
        ),
        "set_up": (
            True,
            "this_object.errordef = value",
            lambda arg: setattr(self, "errordef", arg),
        ),
        "set_strategy": (
            True,
            "this_object.strategy = value",
            lambda arg: setattr(self, "strategy", arg),
        ),
        "set_print_level": (
            True,
            "this_object.print_level = value",
            lambda arg: setattr(self, "print_level", arg),
        ),
        "list_of_fixed_param": (
            True,
            "`[name for (name, fix) in this_object.fixed.items() if fix]`",
            lambda: [name for (name, fix) in self.fixed.items() if fix],
        ),
        "list_of_vary_param": (
            True,
            "`[name for (name, fix) in this_object.fixed.items() if not fix]`",
            lambda: [name for (name, fix) in self.fixed.items() if not fix],
        ),
        "matrix_accurate": (True, "this_object.accurate", lambda: self.accurate),
    }

    if key in deprecated:
        is_method, replacement, lambd = deprecated[key]
        warn(
            "`{0}` is deprecated: Use `{1}` instead".format(key, replacement),
            DeprecationWarning,
            stacklevel=2,
        )

        if is_method:
            return lambd
        else:
            return lambd(self)

    raise AttributeError(
        "type object '{0}' has no attribute '{1}'".format(self.__class__.__name__, key)
    )
