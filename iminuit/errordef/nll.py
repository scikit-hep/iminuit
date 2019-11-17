"""
See iminuit.errordef for details.
"""

from iminuit.util import ErrorDefSigma, ErrorDefCL

sigma = ErrorDefSigma(0.5, 1)
cl = ErrorDefCL(0.5, None)

del ErrorDefSigma, ErrorDefCL
