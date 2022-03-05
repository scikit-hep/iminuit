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
    t = b["stats"]["mean"]
    if name not in variant:
        variant[name] = []
    variant[name].append((n, t))

fig, ax = plt.subplots(
    1, 2, figsize=(13, 4), sharex=True, sharey=True, constrained_layout=True
)
for name, d in variant.items():
    n, t = np.transpose(d)
    is_minuit = int("minuit" in name)
    plt.sca(ax[is_minuit])
    if is_minuit:
        name = name[name.find("_") + 1 :]
    ls = "-"
    if "parallel" in name and "fastmath" in name:
        ls = "-."
    elif "parallel" in name:
        ls = "--"
    elif "fastmath" in name:
        ls = ":"
    plt.plot(n, t, ls=ls, label=name)
for axi in ax:
    axi.loglog()
    axi.legend(title="minuit" if axi is ax[1] else None, ncol=2, frameon=False)
fig.suptitle("Fit of 2 parameters (normal distribution)")
fig.supxlabel("number of data points")
fig.supylabel("runtime / sec")
plt.show()
