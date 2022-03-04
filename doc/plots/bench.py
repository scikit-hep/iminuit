import numpy as np
from numpy.random import default_rng
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator
import os
import pickle

mpl.rcParams.update(mpl.rcParamsDefault)


class TrackingFcn:
    errordef = 1

    def __init__(self, rng, npar):
        self.ncall = 0
        self.y = 5 * rng.standard_normal(npar)

    def __call__(self, par, *args):
        self.ncall += 1
        # make problem non-linear
        z = self.y - par
        return np.sum(z**2 + 0.1 * z**4)


class Runner:
    def __init__(self, npars):
        self.npars = npars

    def __call__(self, seed):
        from iminuit import Minuit
        import nlopt
        from scipy.optimize import minimize

        data = []

        rng = default_rng(seed)
        for npar in self.npars:
            fcn = TrackingFcn(rng, npar)
            for stra in (0, 1, 2):
                key = f"Minuit2/strategy={stra}"
                print(key, npar)
                fcn.ncall = 0
                m = Minuit(fcn, np.zeros(npar))
                m.strategy = stra
                m.migrad()
                max_dev = np.max(np.abs(m.np_values() - fcn.y))
                data.append((key, npar, fcn.ncall, max_dev))

            for algo in ("BOBYQA", "NEWUOA", "PRAXIS", "SBPLX"):
                if npar == 1 and algo == "PRAXIS":
                    continue  # PRAXIS does not work for npar==1
                print(algo, npar)
                fcn.ncall = 0
                opt = nlopt.opt(getattr(nlopt, "LN_" + algo), npar)
                opt.set_min_objective(lambda par, grad: fcn(par))
                opt.set_xtol_abs(1e-2)
                try:
                    xopt = opt.optimize(np.zeros(npar))
                    max_dev = np.max(np.abs(xopt - fcn.y))
                    key = f"nlopt/{algo}"
                    data.append((key, npar, fcn.ncall, max_dev))
                except Exception:
                    pass

            for algo in ("BFGS", "CG", "Powell", "Nelder-Mead"):
                print(algo, npar)
                fcn.ncall = 0
                result = minimize(fcn, np.zeros(npar), method=algo, jac=False)
                max_dev = np.max(np.abs(result.x - fcn.y))
                key = f"scipy/{algo}"
                data.append((key, npar, fcn.ncall, max_dev))

        return data


if os.path.exists("bench.pkl"):
    with open("bench.pkl", "rb") as f:
        results = pickle.load(f)
else:
    npars = (1, 2, 3, 4, 6, 10, 20, 30, 40, 60, 100)

    from numpy.random import SeedSequence
    from concurrent.futures import ProcessPoolExecutor as Pool

    sg = SeedSequence(1)
    with Pool() as p:
        results = tuple(p.map(Runner(npars), sg.spawn(16)))

    with open("bench.pkl", "wb") as f:
        pickle.dump(results, f)


# plt.figure()
# f = TrackingFcn(default_rng(), 2)
# x = np.linspace(-10, 10)
# X, Y = np.meshgrid(x, x)
# F = np.empty_like(X)
# for i, xi in enumerate(x):
#     for j, yi in enumerate(x):
#         F[i, j] = f((xi, yi))
# plt.pcolormesh(X, Y, F.T)
# plt.colorbar()

methods = {}
for data in results:
    for key, npar, ncal, maxdev in data:
        methods.setdefault(key, {}).setdefault(npar, []).append((ncal, maxdev))

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
plt.subplots_adjust(
    top=0.96, bottom=0.14, left=0.075, right=0.81, hspace=0.2, wspace=0.25
)
handles = []
labels = []
markers = (
    ("o", 10),
    ("s", 7),
    ("D", 7),
    ("<", 7),
    (">", 7),
    ("^", 7),
    ("v", 7),
    ("*", 9),
    ("X", 7),
    ("P", 7),
    ("p", 8),
)


for method, (m, ms) in zip(sorted(methods), markers):
    ls = "-"
    lw = 1
    zorder = None
    color = None
    mfc = None
    mew = 1
    if "Minuit" in method:
        ls = "-"
        lw = 2
        zorder = 10
        color = "k"
        mfc = "w"
        mew = 2

    data = methods[method]
    npars = np.sort(list(data))
    ncalls = np.empty_like(npars)
    max_devs = np.empty_like(npars, dtype=float)
    for i, npar in enumerate(npars):
        nc, md = np.transpose(data[npar])
        ncalls[i] = np.median(nc)
        max_devs[i] = np.median(md)
    plt.sca(ax[0])
    (p,) = plt.plot(
        npars,
        ncalls / npars,
        ls=ls,
        lw=lw,
        marker=m,
        ms=ms,
        zorder=zorder,
        color=color,
        mfc=mfc,
        mew=mew,
    )
    handles.append(p)
    labels.append(method)
    plt.xlabel("$N_\\mathrm{par}$")
    plt.ylabel("$N_\\mathrm{call}$ / $N_\\mathrm{par}$")
    plt.loglog()
    plt.ylim(8, 5e2)
    plt.xlim(0.7, 150)
    plt.sca(ax[1])
    plt.xlabel("$N_\\mathrm{par}$")
    plt.ylabel("maximum deviation")
    plt.plot(
        npars,
        max_devs,
        lw=lw,
        ls=ls,
        marker=m,
        ms=ms,
        zorder=zorder,
        color=color,
        mfc=mfc,
        mew=mew,
    )
    plt.loglog()
    plt.gca().yaxis.set_major_locator(LogLocator(numticks=100))

plt.figlegend(handles, labels, loc="center right", fontsize="small")
plt.savefig("bench.svg")

plt.figure(constrained_layout=True)
plt.loglog()
for method, (m, ms) in zip(sorted(methods), markers):
    zorder = None
    color = None
    mfc = None
    mew = 1
    if "Minuit" in method:
        zorder = 10
        color = "k"
        mfc = "w"
        mew = 2

    data = methods[method]

    x = []
    y = []
    s = []
    for npar in (2, 10, 100):
        if npar not in data:
            continue
        nc, md = np.transpose(data[npar])
        x.append(np.median(nc) / npar)
        y.append(np.median(md))
        s.append(50 * npar**0.5)

    plt.scatter(x, y, s, marker=m, color=mfc, edgecolor=color, zorder=zorder)

plt.xlabel("$N_\\mathrm{call}$ / $N_\\mathrm{par}$")
plt.ylabel("maximum deviation")
plt.title("small: npar = 2, medium: npar = 10, large: npar = 100")
plt.savefig("bench2d.svg")

plt.show()
