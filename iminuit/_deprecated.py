import warnings
import functools


class deprecated(object):
    def __init__(self, reason):
        self._reason = reason

    def __call__(self, func):
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            warnings.warn(
                "{0} is deprecated: {1}".format(func.__name__, self._reason),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        decorated_func.__doc__ = "deprecated: " + self._reason
        return decorated_func
