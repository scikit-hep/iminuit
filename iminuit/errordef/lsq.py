"""
See iminuit.errordef for details.
"""

from iminuit.util import ErrorDefSigma, ErrorDefCL

sigma = ErrorDefSigma(1, 1)
cl = ErrorDefCL(1, None)

del ErrorDefSigma, ErrorDefCL
