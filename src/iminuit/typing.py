"""Types for iminuit.

These are used by mypy and similar tools.
"""

from typing import (
    Protocol,
    Optional,
    List,
    Union,
    NamedTuple,
    runtime_checkable,
)
from numpy.typing import NDArray
import numpy as np

# Key for ValueView, ErrorView, etc.
Key = Union[int, str, slice, List[Union[int, str]]]


@runtime_checkable
class Model(Protocol):
    """Type for user-defined model."""

    def __call__(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Evaluate model at locations x and return results as an array."""
        ...  # pragma: no cover


@runtime_checkable
class LossFunction(Protocol):
    """Type for user-defined loss function for LeastSquares clas."""

    def __call__(self, z: NDArray) -> NDArray:
        """Evaluate loss function on values."""
        ...  # pragma: no cover


class ValueRange(NamedTuple):
    """Annotation for the allowed value range of a model parameter."""

    min: float
    max: float


class UserBound(NamedTuple):
    """Type for user-defined limit."""

    min: Optional[float]
    max: Optional[float]
