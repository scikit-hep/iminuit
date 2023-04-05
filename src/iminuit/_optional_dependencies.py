import contextlib
import warnings
from iminuit.warnings import OptionalDependencyWarning


@contextlib.contextmanager
def optional_module_for(functionality, *, replace=None, stacklevel=3):
    try:
        yield
    except ModuleNotFoundError as e:
        package = e.name.split(".")[0]
        if replace:
            package = replace.get(package, package)
        msg = (
            f"{functionality} requires optional package {package!r}. "
            f"Install {package!r} manually to enable this functionality."
        )
        warnings.warn(msg, OptionalDependencyWarning, stacklevel=stacklevel)
