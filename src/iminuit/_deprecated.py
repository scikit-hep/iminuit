import warnings
from numpy import VisibleDeprecationWarning


class deprecated:
    def __init__(self, reason):
        self._reason = reason

    def __call__(self, func):
        def decorated_func(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {self._reason}",
                category=VisibleDeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        decorated_func.__doc__ = "deprecated: " + self._reason
        return decorated_func


class deprecated_parameter:
    def __init__(self, **replacements):
        self._replacements = replacements

    def __call__(self, func):
        def decorated_func(*args, **kwargs):
            for new, old in self._replacements.items():
                if old in kwargs:
                    warnings.warn(
                        f"keyword {old!r} is deprecated, please use {new!r}",
                        category=VisibleDeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs[new] = kwargs[old]
                    del kwargs[old]
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        return decorated_func
