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


def is_string(s):
    try:  # Python 2
        return isinstance(s, basestring)
    except NameError:  # Python 3
        return isinstance(s, str)


try:
    from abc import ABC
except ImportError:
    from abc import ABCMeta

    class ABC(object):
        __metaclass__ = ABCMeta
        pass
