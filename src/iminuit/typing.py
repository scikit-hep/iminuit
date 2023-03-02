"""Types for iminuit.

These are used by mypy and similar tools.
"""

from typing import Protocol, Optional, List, Union, runtime_checkable, NamedTuple
from numpy.typing import NDArray
import numpy as np
import dataclasses
import sys

if sys.version_info < (3, 9):
    from typing_extensions import Annotated  # noqa pragma: no cover
else:
    from typing import Annotated  # noqa pragma: no cover


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


class UserBound(NamedTuple):
    """Type for user-defined limit."""

    min: Optional[float]
    max: Optional[float]


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
