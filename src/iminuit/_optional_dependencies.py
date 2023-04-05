import contextlib
import warnings
from iminuit.warnings import OptionalDependencyWarning


@contextlib.contextmanager
def optional_module_for(functionality, stacklevel=3):
    try:
        yield
    except ModuleNotFoundError as e:
        msg = (
            f"{functionality} requires optional package {e.name!r}. "
            f"Install {e.name!r} manually to enable this."
        )
        warnings.warn(msg, OptionalDependencyWarning, stacklevel=stacklevel)
