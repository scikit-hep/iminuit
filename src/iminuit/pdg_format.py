# Copyright 2020 Hans Dembinski

# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following disclaimer in the documentation and/or other materials
# provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
PDG formatting of numbers with uncertainties.

The PDG uses a special rounding rule for quantities with uncertainties. The description
quoted from M. Tanabashi et al. (Particle Data Group), Phys. Rev. D 98, 030001 (2018),
https://doi.org/10.1103/PhysRevD.98.030001:

"The basic rule states that if the three highest order digits of the error lie between 100
and 354, we round to two significant digits. If they lie between 355 and 949, we round to
one significant digit. Finally, if they lie between 950 and 999, we round up to 1000 and
keep two significant digits. In all cases, the central value is given with a precision
that matches that of the error. So, for example, the result (coming from an average) 0.827
+- 0.119 would appear as 0.83 +- 0.12, while 0.827 +- 0.367 would turn into 0.8 +- 0.4."

In addition, the LHCb Editorial Board declared that in case of several errors, the most
precise one defines the number of digits shown.

This module offers functions that convert values and errors into a string representation.

Documentation follows these guidelines:
https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
"""

import math
from typing import List


term = (" %s", " +%s", " ± %s", " (%s)", "(%s)E%+03i", True, None)
latex = (
    " {}_{%s}",
    "^{+%s}",
    r" \pm %s",
    r" (\mathrm{%s})",
    r"(%s) \times 10^{%i}",
    True,
    (r"\mathrm{NaN}", r"\infty"),
)


def pdg_format(value, error, *errors, labels=None, format=term, leader=None, exp=None):
    r"""Return formatted value with uncertainties according to PDG rules.

    Examples
    --------
    >>> pdg_format(2.3456, 0.123, 0.0123)
    '2.346 ± 0.123 ± 0.012'
    >>> pdg_format(2.3456, 0.123, 0.0123, leader=0)
    '2.35 ± 0.12 ± 0.01'
    >>> pdg_format(1.2, (0.1, 0.2), 0.3, format=latex)
    '1.2 {}_{-0.1}^{+0.2} \\pm 0.3'
    >>> pdg_format(2.3e-09, 0.354e-09, format=latex)
    '(2.30 \\pm 0.35) \\times 10^{-9}'
    >>> pdg_format(2.3e3, 0.1e3, 0.2e3, format=latex, labels=('stat', 'sys'))
    '(2.3 \\pm 0.1 (\\mathrm{stat}) \\pm 0.2 (\\mathrm{sys})) \\times 10^{3}'
    >>> pdg_format(1.234, -0.11, 0.22, 0.3, format=latex)
    '1.23 {}_{-0.11}^{+0.22} \\pm 0.30'

    Parameters
    ----------
    value : float
        Estimated value.
    error : float or tuple of floats
        Uncertainty of value. A positive number is interpreted as a symmetric uncertainty
        (value +- error). Asymmetric uncertainties are passed as a tuple of positive
        numbers (x - xmin, xmax - x) OR as a negative number xmin - x immediately
        followed by a positive number xmax - x.
    *errors
        Optional further uncertainties.
    labels : sequence of str, optional
        Optional labels for the different uncertainties (e.g. 'sys', 'stat', 'lumi').
        A label that starts with one of the characters '_\^<' is used verbatim, otherwise
        it is wrapped according to the format spec for labels.
    format : tuple, optional
        Formatting specification. Structure: (
            <str: format spec for lower asymmetric error>,
            <str: format spec for upper asymmetric error>,
            <str: format spec for symmetric error>,
            <str: format spec for label>,
            <str: format spec for scientific notation>,
            <bool: whether to strip trailing zeros and dots>,
            <tuple of str OR None: replacement for 'nan' and 'inf'>
        )
    leader : int, optional
        Index of uncertainty that should be used to determine the number of digits shown,
        if there are several uncertainties. Default is to use the smallest uncertainty.
        If asymmetric uncertainties are passed as tuples or subsequent pairs of negative
        and positive numbers, the index is that of the pair.
    exp : int, optional
        Exponent to use for scientific notation. If omitted, a heuristic algorithm selects
        a suitable exponent.

    Returns
    -------
    str
        Formatted string.
    """
    fmt_ne, fmt_pe, fmt_se, fmt_lab, fmt_sc, strip, trans = format

    strings, nexp = _round([value, error, *errors], leader, exp)
    if strip:
        strings = _strip(strings)

    if trans:
        for i, x in enumerate(strings):
            c = x[-1]
            if c == "n":  # nan, -nan
                x = trans[0]
            elif c == "f":  # inf, -inf
                if x[0] == "-":
                    x = "-" + trans[1]
                x = trans[1]
            strings[i] = x

    s = strings[0]
    asym = False
    liter = iter(labels) if labels is not None else None
    for si in strings[1:]:
        if si[0] == "-":
            asym = True
            s += fmt_ne % si
        elif asym:
            asym = False
            s += fmt_pe % si
        else:
            s += fmt_se % si
        if liter and asym is False:
            y = next(liter)
            if y[0] in r"_\^<":
                s += y
            else:
                s += fmt_lab % y

    if nexp == 0:
        return s
    else:
        return fmt_sc % (s, nexp)


def _strip(items: List[str]) -> List[str]:
    # ignore inf and nan
    mask = tuple(i for (i, s) in enumerate(items) if "." in s)
    if mask:
        # strip common trailing "0"s
        first = items[mask[0]]
        for i in range(len(first)):
            if all(items[k][-1 - i] == "0" for k in mask):
                i += 1
            else:
                break
        # maybe strip common trailing "." after stripping "0"s
        if i > 0 and all(items[k][-1 - i] == "." for k in mask):
            i += 1
        if i > 0:
            for k in mask:
                items[k] = items[k][:-i]
    return items


def _find_smallest_nonzero_abs_value(seq):
    k = None
    xmin = float("infinity")
    for i, x in enumerate(seq):
        x = abs(x)
        if x > 0 and x < xmin:
            xmin = x
            k = i
    return k, xmin


def _is_asym(value):
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        if len(value) != 2:
            raise ValueError("sequence must have two elements")
        return True
    return False


def _unpack(values):
    assert len(values) > 0
    assert _is_asym(values[0]) is False
    result = [values[0]]
    for v in values[1:]:
        if _is_asym(v):
            result.append(-abs(v[0]))
            result.append(v[1])
        else:
            result.append(v)
    return result


def _unpacked_index(values, index):
    k = 0
    for i in range(index):
        k += 2 if values[k] < 0 else 1
    return k


def _round(values, leader, n_exp_extern):
    assert len(values) >= 1
    assert _is_asym(values[0]) is False
    values = _unpack(values)

    if leader is None:
        # select leading error that determines precision
        leader, lerror = _find_smallest_nonzero_abs_value(values[1:])
        if leader is not None:
            leader += 1
    else:
        leader = _unpacked_index(values[1:], leader) + 1
        asym = math.copysign(1, values[leader]) < 0  # also works for NaN and -0.0
        if asym:
            offset, lerror = _find_smallest_nonzero_abs_value(
                values[leader : leader + 2]
            )
            if offset is None:
                leader = None
            else:
                leader += offset
        else:
            lerror = abs(values[leader])

    def fmt(x, n_digits):
        return ("%%.%if" % max(n_digits, 0)) % x

    n_exp = None
    if math.isfinite(lerror) and lerror > 0:
        n_exp = int(math.floor(math.log10(lerror))) + 1
    if n_exp is None:
        leader = None
        if math.isfinite(values[0]) and values[0] != 0:
            n_exp = int(math.floor(math.log10(abs(values[0]))))
        else:
            # n_exp cannot be determined
            return [str(x) for x in values], 0

    if leader is None:
        # invalid leading error, cannot determine digits
        scale = 10**-n_exp
        return ([fmt(v * scale, 4) for v in values], n_exp)

    scale = 10**-n_exp
    digits = round(lerror * scale, 3)
    if digits < 0.355:
        n_digits = 2
        digits = round(digits, 2)
    else:
        n_digits = 1
        if digits < 0.95:
            digits = round(digits, 1)
        else:
            digits = 1.0

    if n_exp_extern is None:
        if abs(n_exp) > 2:
            n_exp_extern = int(round(n_exp / 3) * 3)
        else:
            n_exp_extern = 0
        if math.isfinite(values[0]) and values[0] != 0:
            n = math.floor(math.log10(abs(values[0])) / 3) * 3
            if n > n_exp_extern:
                n_exp_extern = n

    shift = n_exp - n_exp_extern
    values = [
        fmt(
            (
                round(x * scale, n_digits)
                if i != leader
                else math.copysign(digits, values[leader])
            )
            * 10**shift,
            n_digits - shift,
        )
        for (i, x) in enumerate(values)
    ]
    return values, n_exp - shift
