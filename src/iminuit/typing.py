"""
Types for iminuit.

These are used by mypy and similar tools.
"""

import typing as _tp

T = _tp.TypeVar("T")


class Indexable(_tp.Iterable, _tp.Sized, _tp.Generic[T]):
    """Indexable type for mypy."""

    def __getitem__(self, idx: int) -> T:
        """Get item at index idx."""
        ...  # pragma: no cover


UserBound = _tp.Optional[Indexable[_tp.Optional[float]]]

# Key for ValueView, ErrorView, etc.
Key = _tp.Union[int, str, slice, _tp.List[_tp.Union[int, str]]]
