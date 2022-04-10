"""
Types for iminuit.

These are used by mypy and similar tools.
"""

import typing as _tp

UserBound = _tp.Optional[_tp.Collection[_tp.Optional[float]]]

# Key for ValueView, ErrorView, etc.
Key = _tp.Union[int, str, slice, _tp.List[_tp.Union[int, str]]]
