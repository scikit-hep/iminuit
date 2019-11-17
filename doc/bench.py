import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator
import os
import pickle


class TrackingFcn:
    def __init__(self, npar):
        self.ncall = 0
        self.y = np.random.randn(npar)

    def __call__(self, par, *args):
        self.ncall += 1
        # use log to make problem non-linear
        return np.log(np.sum((self.y - par) ** 2) + 1)


class Runner:
    def __init__(self, npars):
        self.npars = npars

    def __call__(self, trial):
        from iminuit import Minuit
        import nlopt
        from scipy.optimize import minimize
        import numpy as np

        np.random.seed(1 + trial)
        data = []

        for npar in self.npars:
            fcn = TrackingFcn(npar)
            for stra in (0, 1, 2):
                key = f"Minuit2/strategy={stra}"
                print(key, npar)
                fcn.ncall = 0
                m = Minuit.from_array_func(
                    fcn, np.zeros(npar), pedantic=False, print_level=0
                )
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
                opt.set_xtol_abs(1e-3)
                try:
                    xopt = opt.optimize(np.zeros(npar))
                except:
                    pass
                max_dev = np.max(np.abs(xopt - fcn.y))

                key = f"nlopt/{algo}"
                data.append((key, npar, fcn.ncall, max_dev))

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
    from multiprocessing.pool import Pool

    with Pool() as p:
        results = p.map(Runner(npars), range(100))
    with open("bench.pkl", "wb") as f:
        pickle.dump(results, f)


methods = {}
for data in results:
    for key, npar, ncal, maxdev in data:
        methods.setdefault(key, {}).setdefault(npar, []).append((ncal, maxdev))

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
plt.subplots_adjust(left=0.12, right=0.75, wspace=0.6)
handles = []
labels = []
markers = iter(
    (
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
)

for method in sorted(methods):
    m, ms = next(markers)
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
    p, = plt.plot(
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
    # plt.xticks((1, 10, 100))

    plt.gca().xaxis.set_major_locator(LogLocator(numticks=10))
    plt.gca().xaxis.set_minor_locator(LogLocator(numticks=10, subs="all"))
    plt.xticks((1, 10, 100), ("1", "10", "100"))
    plt.gca().yaxis.set_major_locator(LogLocator(numticks=100))

plt.figlegend(handles, labels, loc="center right", fontsize="small")
plt.savefig("bench.svg")
plt.show()
