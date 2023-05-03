#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

if len(sys.argv) == 2:
    fn = sys.argv[1]
else:
    from pathlib import Path

    paths = []
    for p in Path(".benchmarks").rglob("*.json"):
        paths.append((p.stat().st_mtime, p))
    paths.sort()
    fn = paths[-1][1]

with open(fn) as f:
    data = json.load(f)

print(
    f"""\
benchmark results
  {data['datetime']}
  {data['machine_info']['cpu']['brand_raw']}
"""
)

variant = {}
for b in data["benchmarks"]:
    params = b["params"]
    n = params["n"]
    n = int(n)
    name = b["name"]
    name = name[name.find("_") + 1 : name.find("[")]
    extra = [k for (k, v) in params.items() if k not in ("n", "lib", "model") and v]
    if extra:
        name += "_" + "_".join(extra)
    for key in ("lib", "model"):
        if key in params:
            name += f"_{params[key]}"
    t = b["stats"]["mean"]
    if name not in variant:
        variant[name] = []
    variant[name].append((n, t))

for k in variant:
    print(k)

names = [
    {
        "custom_scipy": "scipy",
        "custom_log_scipy": "scipy logpdf",
        "custom_numba_stats": "numba_stats",
        "custom_log_numba_stats": "numba_stats logpdf",
    },
    {
        "UnbinnedNLL_norm": "UnbinnedNLL",
        "UnbinnedNLL_log_norm": "UnbinnedNLL logpdf",
        "custom_numba_stats": "custom numba_stats",
        "custom_log_numba_stats": "custom numba_stats logpdf",
    },
    {
        "RooFit_norm": "RooFit",
        "RooFit_BatchMode_norm": "RooFit with BatchMode",
        "RooFit_NumCPU_norm": "RooFit with NumCPU",
        "RooFit_NumCPU_BatchMode_norm": "RooFit with NumCPU, BatchMode",
        "minuit_custom_numba_norm": "iminuit+numba",
        "minuit_custom_numba_parallel_fastmath_norm": "iminuit+numba with parallel, fastmath",  # noqa: E501
    },
    {
        "RooFit_norm+truncexpon": "RooFit",
        "RooFit_BatchMode_norm+truncexpon": "RooFit with BatchMode",
        "RooFit_NumCPU_norm+truncexpon": "RooFit with NumCPU",
        "RooFit_NumCPU_BatchMode_norm+truncexpon": "RooFit with NumCPU, BatchMode",
        "minuit_custom_numba_norm+truncexpon": "iminuit+numba",
        "minuit_custom_numba_parallel_fastmath_norm+truncexpon": "iminuit+numba with parallel, fastmath",  # noqa: E501
    },
    {
        "minuit_custom_numba_norm": "numba",
        "minuit_custom_cfunc": "cfunc",
    },
    {
        "minuit_custom_log_numba_parallel_fastmath": "parallel fastmath",
        "minuit_custom_log_numba_parallel_fastmath_handtuned": "parallel fastmath handtuned",  # noqa: E501
    },
    {
        "minuit_custom_numba_parallel_fastmath_norm": "norm parallel fastmath",
        "minuit_custom_numba_parallel_fastmath_log_norm": "norm logpdf parallel fastmath",  # noqa: E501
        "minuit_custom_numba_parallel_fastmath_norm+truncexpon": "mix parallel fastmath",  # noqa: E501
        "minuit_custom_numba_parallel_fastmath_log_norm+truncexpon": "mix logpdf paralel fastmath",  # noqa: E501
    },
]

for subnames in names:
    plt.figure(constrained_layout=True)
    for name in subnames:
        if name not in variant:
            continue
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
        plt.plot(n, t, ls=ls, label=subnames[name])

    plt.loglog()
    plt.legend(
        frameon=False,
        fontsize="medium" if len(subnames) < 4 else "small",
        ncol=1 if len(subnames) < 3 else 2,
    )
    # plt.title("Fit of normal distribution with 2 parameters")
    plt.xlabel("number of data points")
    plt.ylabel("runtime / sec")
plt.show()
