import warnings
import functools
import inspect


class deprecated(object):
    def __init__(self, message):
        self._message = message

    def __call__(self, func):
        @functools.wraps(func)
        def decorated_func(*args, **kwargs):
            warnings.warn(
                "{0} is deprecated: {1}".format(func.__name__, self._message),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return decorated_func
