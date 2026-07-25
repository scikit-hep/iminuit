"""
Types for iminuit.

These are used by mypy and similar tools.
"""

from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
    NamedTuple,
    Annotated,
)
from numpy.typing import NDArray
import numpy as np
import dataclasses

# Key for ValueView, ErrorView, etc.
Key = int | str | slice | list[int | str]


@runtime_checkable
class Model(Protocol):
    """Type for user-defined model."""

    def __call__(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Evaluate model at locations x and return results as an array."""
        ...  # pragma: no cover


@runtime_checkable
class ModelGradient(Protocol):
    """Type for user-defined model gradient."""

    def __call__(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Evaluate model gradient at locations x and return results as an array."""
        ...  # pragma: no cover


@runtime_checkable
class Cost(Protocol):
    """Type for user-defined cost function."""

    def __call__(self, *args: float) -> float:
        """Evaluate cost and return results as a float."""
        ...  # pragma: no cover


@runtime_checkable
class CostVector(Protocol):
    """Type for user-defined gradient, G2, Hessian of a cost function."""

    def __call__(self, *args: float) -> np.ndarray:
        """Evaluate gradient, G2, or Hessian and return results as an array."""
        ...  # pragma: no cover


# for backward compatibility
CostGradient = CostVector


@runtime_checkable
class LossFunction(Protocol):
    """Type for user-defined loss function for LeastSquares clas."""

    def __call__(self, z: NDArray) -> NDArray:
        """Evaluate loss function on values."""
        ...  # pragma: no cover


class UserBound(NamedTuple):
    """Type for user-defined limit."""

    min: float | None
    max: float | None


@dataclasses.dataclass
class Gt:
    """Annotation compatible with annotated-types."""

    gt: float


@dataclasses.dataclass
class Ge:
    """Annotation compatible with annotated-types."""

    ge: float


@dataclasses.dataclass
class Lt:
    """Annotation compatible with annotated-types."""

    lt: float


@dataclasses.dataclass
class Le:
    """Annotation compatible with annotated-types."""

    le: float


@dataclasses.dataclass
class Interval:
    """Annotation compatible with annotated-types."""

    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None


# common convenience types
PositiveFloat = Annotated[float, Gt(0)]
Probability = Annotated[float, Interval(ge=0, le=1)]
