"""Types for iminuit.

These are used by mypy and similar tools.
"""

from typing import (
    Protocol,
    Callable,
    Optional,
    Collection,
    List,
    Union,
    runtime_checkable,
)
import numpy as np

# Used by LeastSquares class
LossFunction = Callable[[np.ndarray], np.ndarray]

# Used by Minuit
UserBound = Optional[Collection[Optional[float]]]

# Key for ValueView, ErrorView, etc.
Key = Union[int, str, slice, List[Union[int, str]]]


@runtime_checkable
class Model(Protocol):
    """Type for user-defined model."""

    def __call__(self, x: np.ndarray, *args: float) -> np.ndarray:
        """Evaluate model at locations x and return results as an array."""
        ...
