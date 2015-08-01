"""Python 2 / 3 compatibility layer.
"""
import sys

py_ver = sys.version_info
PY2 = False
PY3 = False
if py_ver[0] == 2:
    PY2 = True
else:  # just in case PY4
    PY3 = True

ARRAY_DOUBLE_TYPECODE = 'd' if PY2 else u'd'


def is_string(s):
    try:  # Python 2
        return isinstance(s, basestring)
    except NameError:  # Python 3
        return isinstance(s, str)
