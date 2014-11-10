"""Python 2 / 3 compatibility layer.

The functions are copied from http://python-future.org/
to avoid the extra dependency.
"""


# from future.utils import bytes_to_native_str
def bytes_to_native_str(b, encoding='utf-8'):
        return b.decode(encoding)

# See http://python-future.org/stdlib_incompatibilities.html#array-array
ARRAY_DOUBLE_TYPECODE = bytes_to_native_str(b'd')
