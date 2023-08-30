import contextlib
import warnings
from iminuit.warnings import OptionalDependencyWarning
from typing import Dict, Optional


@contextlib.contextmanager
def optional_module_for(
    functionality: str, *, replace: Optional[Dict[str, str]] = None, stacklevel: int = 3
):
    try:
        yield
    except ModuleNotFoundError as e:
        assert e.name is not None
        package = e.name.split(".")[0]
        if replace:
            package = replace.get(package, package)
        msg = (
            f"{functionality} requires optional package {package!r}. "
            f"Install {package!r} manually to enable this functionality."
        )
        warnings.warn(msg, OptionalDependencyWarning, stacklevel=stacklevel)
