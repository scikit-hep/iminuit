"""Warnings used by iminuit."""


class OptionalDependencyWarning(RuntimeWarning):
    """Feature requires an optional external package."""


class IMinuitWarning(RuntimeWarning):
    """Generic iminuit warning."""


class HesseFailedWarning(IMinuitWarning):
    """HESSE failed warning."""


class PerformanceWarning(UserWarning):
    """Warning about performance issues."""
