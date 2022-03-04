import matplotlib.pyplot as plt
import numpy as np
import json

with open("cost.json") as f:
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

fig, ax = plt.subplots(1, 2, figsize=(14, 4), sharex=True, sharey=True)
for name, d in variant.items():
    n, t = np.transpose(d)
    plt.sca(ax[int("minuit" in name)])
    plt.plot(n, t, label=name)
for axi in ax:
    axi.loglog()
    axi.legend()
fig.supxlabel("number of points")
plt.savefig("cost_bench.pdf")
