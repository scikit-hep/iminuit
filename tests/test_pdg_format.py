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

# test: pytest --doctest-modules
# coverage report: pytest --cov --cov-report html

import pytest
from iminuit.pdg_format import (
    _find_smallest_nonzero_abs_value,
    _is_asym,
    _unpack,
    _strip,
    _unpacked_index,
    pdg_format,
    term,
    latex,
)
import math


def test_find_smallest_nonzero_abs_value():
    assert _find_smallest_nonzero_abs_value((-1.1, 3)) == (0, 1.1)
    assert _find_smallest_nonzero_abs_value((-10, 3, 1.2, 0, 4)) == (2, 1.2)
    assert _find_smallest_nonzero_abs_value(()) == (None, math.inf)
    assert _find_smallest_nonzero_abs_value((0, math.inf, -math.inf)) == (
        None,
        math.inf,
    )


def test_is_asym():
    assert _is_asym((1, 2)) is True
    assert _is_asym(1) is False
    assert _is_asym([1, 2]) is True
    with pytest.raises(ValueError):
        _is_asym([])
    with pytest.raises(ValueError):
        _is_asym([1, 2, 3])


def test_unpack():
    assert _unpack([1, (2, 3), 4]) == [1, -2, 3, 4]
    with pytest.raises(ValueError):
        _unpack([1, [2, 3, 4]])


def test_unpacked_index():
    assert _unpacked_index([1, 1], 0) == 0
    assert _unpacked_index([1, 1], 1) == 1
    assert _unpacked_index([1, 1], 2) == 2
    assert _unpacked_index([1, -1, 1], 0) == 0
    assert _unpacked_index([1, -1, 1], 1) == 1
    assert _unpacked_index([1, -1, 1], 2) == 3
    assert _unpacked_index([1, -1, 1, 1], 3) == 4
    assert _unpacked_index([1, -1, 1, -1, 1], 3) == 5


def test_strip():
    assert _strip(["123", "20"]) == ["123", "20"]
    assert _strip(["10", "20"]) == ["10", "20"]
    assert _strip(["0.10", "0.20", "0.30"]) == ["0.1", "0.2", "0.3"]
    assert _strip(["0.11", "0.20"]) == ["0.11", "0.20"]
    assert _strip(["10.0", "20.0"]) == ["10", "20"]
    assert _strip(["1.200", "3.40"]) == ["1.20", "3.4"]


def test_term_format():
    def ft(*args, **kwargs):
        return pdg_format(*args, format=term, **kwargs)

    assert ft(2.345e-09, 0.1e-09) == "(2.35 ± 0.10)E-09"
    assert ft(2.345e-09, 0.354e-09) == "(2.35 ± 0.35)E-09"
    assert ft(2.345e-09, 0.355e-09) == "(2.3 ± 0.4)E-09"
    assert ft(2.345e-09, 0.949e-09) == "(2.3 ± 0.9)E-09"
    assert ft(2.345e-09, 0.95e-09) == "(2.3 ± 1.0)E-09"
    assert ft(2.345e-09, 0.999e-09) == "(2.3 ± 1.0)E-09"

    assert ft(2.3456, 0.001) == "2.3456 ± 0.0010"
    assert ft(2.3456, 0.00354) == "2.3456 ± 0.0035"
    assert ft(2.3456, 0.00355) == "2.346 ± 0.004"
    assert ft(2.3456, 0.00949) == "2.346 ± 0.009"
    assert ft(2.3456, 0.0095) == "2.346 ± 0.010"
    assert ft(2.3456, 0.00999) == "2.346 ± 0.010"

    assert ft(2.3456, 0.01) == "2.346 ± 0.010"
    assert ft(2.3456, 0.0354) == "2.346 ± 0.035"
    assert ft(2.3456, 0.0355) == "2.35 ± 0.04"
    assert ft(2.3456, 0.0949) == "2.35 ± 0.09"
    assert ft(2.3456, 0.095) == "2.35 ± 0.10"
    assert ft(2.3456, 0.0999) == "2.35 ± 0.10"

    assert ft(2.3456, 0.1) == "2.35 ± 0.10"
    assert ft(2.3456, 0.354) == "2.35 ± 0.35"
    assert ft(2.3456, 0.355) == "2.3 ± 0.4"
    assert ft(2.3456, 0.949) == "2.3 ± 0.9"
    assert ft(2.3456, 0.95) == "2.3 ± 1.0"
    assert ft(2.3456, 0.999) == "2.3 ± 1.0"

    assert ft(2.3456, 1) == "2.3 ± 1.0"
    assert ft(2.3456, 3.54) == "2.3 ± 3.5"
    assert ft(2.3456, 3.55) == "2 ± 4"
    assert ft(2.3456, 9.49) == "2 ± 9"
    assert ft(2.3456, 9.5) == "2 ± 10"
    assert ft(2.3456, 9.99) == "2 ± 10"

    assert ft(23.456, 10) == "23 ± 10"
    assert ft(23.456, 35.4) == "23 ± 35"
    assert ft(23.456, 35.5) == "20 ± 40"
    assert ft(23.456, 94.9) == "20 ± 90"
    assert ft(23.456, 95.0) == "20 ± 100"
    assert ft(23.456, 99.9) == "20 ± 100"

    assert ft(234.56, 100) == "(0.23 ± 0.10)E+03"
    assert ft(234.56, 354) == "(0.23 ± 0.35)E+03"
    assert ft(234.56, 355) == "(0.2 ± 0.4)E+03"
    assert ft(234.56, 949) == "(0.2 ± 0.9)E+03"
    assert ft(234.56, 950) == "(0.2 ± 1.0)E+03"
    assert ft(234.56, 999) == "(0.2 ± 1.0)E+03"

    assert ft(2345.6, 1000) == "(2.3 ± 1.0)E+03"
    assert ft(2345.6, 3540) == "(2.3 ± 3.5)E+03"
    assert ft(2345.6, 3550) == "(2 ± 4)E+03"
    assert ft(2345.6, 9490) == "(2 ± 9)E+03"
    assert ft(2345.6, 9500) == "(2 ± 10)E+03"
    assert ft(2345.6, 9990) == "(2 ± 10)E+03"

    assert ft(2.3456e12, 1e11) == "(2.35 ± 0.10)E+12"
    assert ft(2.3456e12, 3.54e11) == "(2.35 ± 0.35)E+12"
    assert ft(2.3456e12, 3.55e11) == "(2.3 ± 0.4)E+12"
    assert ft(2.3456e12, 9.49e11) == "(2.3 ± 0.9)E+12"
    assert ft(2.3456e12, 9.5e11) == "(2.3 ± 1.0)E+12"
    assert ft(2.3456e12, 9.99e11) == "(2.3 ± 1.0)E+12"

    assert ft(-2.3456e13, 1e11) == "(-23.46 ± 0.10)E+12"
    assert ft(-2.3456e13, 3.54e11) == "(-23.46 ± 0.35)E+12"
    assert ft(-2.3456e13, 3.55e11) == "(-23.5 ± 0.4)E+12"
    assert ft(-2.3456e13, 9.49e11) == "(-23.5 ± 0.9)E+12"
    assert ft(-2.3456e13, 9.5e11) == "(-23.5 ± 1.0)E+12"
    assert ft(-2.3456e13, 9.99e11) == "(-23.5 ± 1.0)E+12"

    assert ft(math.nan, 1.0) == "nan ± 1"
    assert ft(math.nan, 3.54) == "nan ± 3.5"
    assert ft(math.nan, 3.55) == "nan ± 4"
    assert ft(math.nan, 9.49) == "nan ± 9"
    assert ft(math.nan, 9.99) == "nan ± 10"

    assert ft(math.inf, 1.0) == "inf ± 1"
    assert ft(math.inf, 3.54) == "inf ± 3.5"
    assert ft(-math.inf, 3.55) == "-inf ± 4"
    assert ft(math.inf, 9.49) == "inf ± 9"
    assert ft(-math.inf, 9.99) == "-inf ± 10"

    assert ft(math.nan, 1.0e3) == "(nan ± 1)E+03"
    assert ft(math.nan, 3.54e3) == "(nan ± 3.5)E+03"
    assert ft(math.nan, 3.55e3) == "(nan ± 4)E+03"
    assert ft(math.nan, 9.49e3) == "(nan ± 9)E+03"
    assert ft(math.nan, 9.99e3) == "(nan ± 10)E+03"

    assert ft(2.3456, math.nan) == "2.3456 ± nan"
    assert ft(1.2345e9, math.nan) == "(1.2345 ± nan)E+09"
    assert ft(2.3456e4, math.inf) == "(2.3456 ± inf)E+04"

    # implementation is robust against input errors
    assert ft(2.3456e-3, -1) == "0 -1"

    assert ft(0, 0) == "0 ± 0"
    assert ft(2.3456e10, 0) == "(2.3456 ± 0.0000)E+10"
    assert ft(2.3456e10, 0) == "(2.3456 ± 0.0000)E+10"

    assert ft(1.2345e100, 1.2345e100) == "(0.012 ± 0.012)E+102"

    assert ft(1.2345, 0.123, 0.0123) == "1.234 ± 0.123 ± 0.012"
    assert ft(1.2345, 0.1, 0.4) == "1.23 ± 0.10 ± 0.40"

    assert ft(1.234, (0.11, 0.22), 0.45) == "1.23 -0.11 +0.22 ± 0.45"
    assert (
        ft(1.234, -0.11, 0.22, 0.45, labels=("a", "b"))
        == "1.23 -0.11 +0.22 (a) ± 0.45 (b)"
    )
    assert (
        ft(1.234, -0.11, 0.22, 0.45, labels=("a", "b"), leader=1)
        == "1.2 -0.1 +0.2 (a) ± 0.5 (b)"
    )
    data_as_list = [1.234, -0.11, 0.22, 0.45]
    assert ft(*data_as_list, leader=1) == "1.2 -0.1 +0.2 ± 0.5"
    assert ft(1.2, -0.0, 0.0, 0.0, leader=0) == "1.2 -0.0 +0.0 ± 0.0"

    assert ft(-1.234567e-22, 1.234567e-11) == "(-0.000 ± 0.012)E-09"


def test_latex_format():
    def ft(*args, **kwargs):
        return pdg_format(*args, format=latex, **kwargs)

    nan = math.nan
    inf = math.inf
    assert ft(234.56, 12.3) == r"235 \pm 12"
    assert ft(2345.6, 123) == r"(2.35 \pm 0.12) \times 10^{3}"
    assert ft(2345.6, 355) == r"(2.3 \pm 0.4) \times 10^{3}"
    assert ft(234.5, 123.4) == r"(0.23 \pm 0.12) \times 10^{3}"
    assert ft(23.45, 12.34) == r"23 \pm 12"
    assert ft(2.345, 1.234) == r"2.3 \pm 1.2"
    assert ft(0.2345, 0.1234) == r"0.23 \pm 0.12"
    assert ft(0.02345, 0.01234) == r"0.023 \pm 0.012"
    assert ft(0.02345, 0.09123) == r"0.02 \pm 0.09"
    assert ft(nan, 2.34e3) == r"(\mathrm{NaN} \pm 2.3) \times 10^{3}"
    assert ft(1e9, nan) == r"(1 \pm \mathrm{NaN}) \times 10^{9}"
    assert ft(inf, 2.345e3) == r"(\infty \pm 2.3) \times 10^{3}"
    assert ft(1.2345e9, inf) == r"(1.2345 \pm \infty) \times 10^{9}"
    assert ft(inf, -inf) == r"\infty \pm \infty"  # tolerance against input errors
    assert ft(0, 0) == r"0 \pm 0"
    assert ft(1.234, 0.123, 2.345) == r"1.23 \pm 0.12 \pm 2.35"
    assert ft(1.234, 0.96, 0.45) == r"1.2 \pm 1.0 \pm 0.5"
    assert (
        ft(1.234, (0.12, 0.78), (0.045, 0.067))
        == r"1.23 {}_{-0.12}^{+0.78} {}_{-0.05}^{+0.07}"
    )
    assert (
        ft(1.234, (0.12, 0.78), (0.045, 0.067))
        == r"1.23 {}_{-0.12}^{+0.78} {}_{-0.05}^{+0.07}"
    )
    assert (
        ft(1.234, (0.12, 0.78), (0.45, 0.67), leader=1)
        == r"1.2 {}_{-0.1}^{+0.8} {}_{-0.5}^{+0.7}"
    )
    assert (
        ft(1.234, -0.12, 0.78, -0.45, 0.67, leader=1)
        == r"1.2 {}_{-0.1}^{+0.8} {}_{-0.5}^{+0.7}"
    )
    assert (
        ft(1.234, 0.123, 2.345, labels=(r"_\mathrm{a}", "b"))
        == r"1.23 \pm 0.12_\mathrm{a} \pm 2.35 (\mathrm{b})"
    )
    assert (
        ft(1.234, (0.123, 2.345), 4.56, labels=("a", "b"))
        == r"1.23 {}_{-0.12}^{+2.35} (\mathrm{a}) \pm 4.56 (\mathrm{b})"
    )
