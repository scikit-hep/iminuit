"""
This module contains special constants ``sigma`` and ``cl`` to conveniently set the error definition in the :class:`iminuit.Minuit` class. The constants are not normal numbers, they are instances of the classes :class:`iminuit.util.ErrorDefSigma` and :class:`iminuit.util.ErrorDefCL`, respectively, which have overloaded operators so that the syntax becomes straight forward. How to use these is best explained with some examples.

    from iminuit import Minuit
    
    from iminuit.errordef import lsq

    def least_squares(a, b, c):
        ...

    # for standard errors
    m = Minuit(negative_log_likelihood, ..., errordef=lsq.sigma)
    # for 2-sigma errors
    m.set_errordef(2 * lsq.sigma)  # or lsq.sigma * 2
    # for 99 % confidence level
    m.set_errordef(0.99 * lsq.cl)  # or lsq.cl * 0.99
"""

from iminuit.util import ErrorDefSigma, ErrorDefCL

sigma = ErrorDefSigma(1, 1)
cl = ErrorDefCL(1, None)

del ErrorDefSigma, ErrorDefCL
