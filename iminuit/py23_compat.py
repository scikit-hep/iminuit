"""Python 2 / 3 compatibility layer.

The functions are copied from http://python-future.org/
to avoid the extra dependency.
"""
import sys

py_ver = sys.version_info
PY2 = False
PY3 = False
if py_ver[0] == 2:
    PY2 = True
else:#just in case PY4
    PY3 = True

ARRAY_DOUBLE_TYPECODE = 'd' if PY2 else u'd'


