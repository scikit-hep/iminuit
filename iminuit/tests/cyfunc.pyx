#cython: embedsignature=True

cpdef f(a, b):
    return a + 2 * b

cdef class CyCallable:
    cpdef double test(self, c, d):
        return c + d
