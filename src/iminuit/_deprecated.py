import warnings
from typing import Callable, Any
from importlib.metadata import version
from iminuit._parse_version import parse_version


CURRENT_VERSION = parse_version(version("iminuit"))


class deprecated:
    def __init__(self, reason: str, removal: str = ""):
        self.reason = reason
        self.removal = parse_version(removal) if removal else None

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        category: Any = FutureWarning
        extra = ""
        if self.removal:
            vstring = ".".join(str(x) for x in self.removal)
            extra = f" and will be removed in version {vstring}"
            if CURRENT_VERSION >= self.removal:
                category = DeprecationWarning
        msg = f"{func.__name__} is deprecated{extra}: {self.reason}"

        def decorated_func(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, category=category, stacklevel=2)
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        decorated_func.__doc__ = msg
        return decorated_func


class removed_parameter(deprecated):
    def __init__(self, parameter: str, reason: str, removal: str):
        self.parameter = parameter
        super().__init__(reason, removal)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        assert self.removal is not None
        category: Any = FutureWarning
        vstring = ".".join(str(x) for x in self.removal)
        if CURRENT_VERSION >= self.removal:
            category = DeprecationWarning

        def decorated_func(*args: Any, **kwargs: Any) -> Any:
            if self.parameter in kwargs:
                warnings.warn(
                    f"keyword {self.parameter!r} is deprecated and will be removed in version {vstring}",
                    category=category,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        decorated_func.__doc__ = func.__doc__
        return decorated_func


class renamed_parameter:
    def __init__(self, **replacements: str):
        self.replacements = replacements

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        def decorated_func(*args: Any, **kwargs: Any) -> Any:
            for new, old in self.replacements.items():
                if old in kwargs:
                    warnings.warn(
                        f"keyword {old!r} is deprecated, please use {new!r}",
                        category=FutureWarning,
                        stacklevel=2,
                    )
                    kwargs[new] = kwargs[old]
                    del kwargs[old]
            return func(*args, **kwargs)

        decorated_func.__name__ = func.__name__
        decorated_func.__doc__ = func.__doc__
        return decorated_func
