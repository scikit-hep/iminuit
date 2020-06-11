import warnings
from iminuit.version import __version__  # noqa: F401

warnings.warn("import iminuit.version instead", DeprecationWarning, stacklevel=2)
