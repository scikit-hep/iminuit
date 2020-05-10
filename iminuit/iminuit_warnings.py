from __future__ import absolute_import, division, print_function, unicode_literals
from . import util as _util
import warnings

warnings.warn(
    "importing iminuit.iminuit_warnings is deprecated, "
    "import warnings from iminuit.util instead",
    DeprecationWarning,
)

deprecated_names = ["IMinuitWarning", "InitialParamWarning", "HesseFailedWarning"]


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


warnings.warn(
    "importing iminuit.iminuit_warnings is deprecated, "
    "import warnings from iminuit.util instead",
    DeprecationWarning,
)
