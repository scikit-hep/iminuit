#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) == 2:
    fn = sys.argv[1]
else:
    from pathlib import Path

    fn = sorted(Path(".benchmarks").rglob("*cost.json"))[-1]

with open(fn) as f:
    data = json.load(f)

variant = {}
for b in data["benchmarks"]:
    params = b["params"]
    n = params["n"]
    n = int(n)
    name = b["name"]
    name = name[name.find("_") + 1 : name.find("[")]
    extra = [k for (k, v) in params.items() if k not in ("n", "lib") and v]
    if extra:
        name += "_" + "_".join(extra)
    if "lib" in params:
        name += f"_{params['lib']}"
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
    [
        "custom_scipy",
        "custom_log_scipy",
        "custom_numba_stats",
        "custom_log_numba_stats",
    ],
    [
        "UnbinnedNLL",
        "custom_numba_stats",
        "UnbinnedNLL_log",
        "custom_log_numba_stats",
    ],
    [
        "custom_log_numba_stats",
        "custom_log_numba",
        "custom_log_numba_fastmath",
        "custom_log_numba_parallel",
        "custom_log_numba_parallel_fastmath",
    ],
    [
        "RooFit",
        "RooFit_BatchMode",
        "RooFit_NumCPU",
        "RooFit_NumCPU_BatchMode",
        "minuit_custom_numba",
        "minuit_custom_log_numba",
        "minuit_custom_log_cfunc",
        "minuit_custom_log_numba_parallel_fastmath",
        "minuit_handtuned_log_numba_parallel_fastmath",
    ],
]

for axi, subnames in zip(ax.flat, names):
    plt.sca(axi)
    for name in subnames:
        d = variant[name]
        n, t = np.transpose(d)
        ls = "-"
        if (
            "parallel" in name
            and "fastmath" in name
            or ("NumCPU" in name and "BatchMode" in name)
        ):
            ls = "-."
        elif "parallel" in name or "NumCPU" in name:
            ls = "--"
        elif "fastmath" in name or "BatchMode" in name:
            ls = ":"
        if axi is not ax[0, 0] and name.endswith("_numba_stats"):
            name = name.replace("_numba_stats", "")
        plt.plot(n, t, ls=ls, label=name)
for axi in ax.flat:
    axi.loglog()
    axi.legend(
        frameon=True,
        loc="upper left",
        framealpha=1,
        fontsize="x-small" if axi is ax[1, 1] else "medium",
    )
fig.suptitle("Fit of normal distribution with 2 parameters")
fig.supxlabel("number of data points")
fig.supylabel("runtime / sec")
plt.show()
