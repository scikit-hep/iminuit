from . import util as _util
import warnings
from sys import version_info as pyver

warnings.warn(
    "importing iminuit.iminuit_warnings is deprecated, "
    "import warnings from iminuit.util instead",
    DeprecationWarning,
)

deprecated_names = ["IMinuitWarning", "InitialParamWarning", "HesseFailedWarning"]


if pyver >= (3, 7):

    def __getattr__(name):
        if name in deprecated_names:
            warnings.warn(
                (
                    "importing {} from {} is deprecated, "
                    "import from iminuit.util instead"
                ).format(name, __name__),
                DeprecationWarning,
            )
            return getattr(_util, name)
        raise AttributeError("module {} has no attribute {}".format(__name__, name))


else:
    from .util import (  # noqa: F401
        IMinuitWarning,
        InitialParamWarning,
        HesseFailedWarning,
    )
