#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) == 2:
    fn = sys.argv[1]
else:
    from pathlib import Path

    fn = sorted(Path("../.benchmarks").rglob("*cost.json"))[-1]

with open(fn) as f:
    data = json.load(f)

variant = {}
for b in data["benchmarks"]:
    n = b["params"]["n"]
    n = int(n)
    name = b["name"]
    name = name[name.find("_") + 1 : name.find("[")]
    extra = [k for (k, v) in b["params"].items() if k != "n" and v]
    if extra:
        name += "_" + "_".join(extra)
    t = b["stats"]["mean"]
    if name not in variant:
        variant[name] = []
    variant[name].append((n, t))

for k in variant:
    print(k)

fig, ax = plt.subplots(
    2, 2, figsize=(13, 8), sharex=True, sharey=False, constrained_layout=True
)
names = [
    ["UnbinnedNLL", "simple", "UnbinnedNLL_log", "simple_log"],
    [
        "UnbinnedNLL_log",
        "numba_sum_logpdf",
        "numba_sum_logpdf_parallel",
        "numba_sum_logpdf_fastmath",
        "numba_sum_logpdf_fastmath_parallel",
    ],
    [
        "minuit_UnbinnedNLL",
        "minuit_UnbinnedNLL_log",
        "minuit_simple_numba",
        "minuit_simple_numba_log",
        "minuit_cfunc_sum_logpdf",
    ],
    [
        "minuit_numba_sum_logpdf_parallel_fastmath",
        "minuit_numba_handtuned_parallel_fastmath",
    ],
]

for axi, subnames in zip(ax.flat, names):
    plt.sca(axi)
    for name in subnames:
        d = variant[name]
        n, t = np.transpose(d)
        ls = "-"
        if "parallel" in name and "fastmath" in name:
            ls = "-."
        elif "parallel" in name:
            ls = "--"
        elif "fastmath" in name:
            ls = ":"
        plt.plot(n, t, ls=ls, label=name)
for axi in ax.flat:
    axi.loglog()
    axi.legend(frameon=False)
fig.suptitle("Fit of normal distribution with 2 parameters")
fig.supxlabel("number of data points")
fig.supylabel("runtime / sec")
plt.show()
