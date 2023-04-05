"""Warnings used by iminuit."""


class IMinuitWarning(RuntimeWarning):
    """Generic iminuit warning."""


class OptionalDependencyWarning(IMinuitWarning):
    """Feature requires an optional external package."""


class HesseFailedWarning(IMinuitWarning):
    """HESSE failed warning."""


class PerformanceWarning(UserWarning):
    """Warning about performance issues."""
