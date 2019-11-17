"""
This module contains special constants ``sigma`` and ``cl`` to conveniently set the error definition in the :class:`iminuit.Minuit` class. The constants are not normal numbers, they are instances of the classes :class:`iminuit.util.ErrorDefSigma` and :class:`iminuit.util.ErrorDefCL`, respectively, which have overloaded operators so that the syntax becomes straight forward. How to use these is best explained with some examples.

    from iminuit import Minuit
    
    # for likelihoods
    from iminuit.errordef import ml

    def negative_log_likelihood(a, b, c):
        ...

    # for standard errors
    m = Minuit(negative_log_likelihood, ..., errordef=ml.sigma)
    # for 2-sigma errors
    m.set_errordef(2 * ml.sigma)  # or ml.sigma * 2
    # for 99 % confidence level
    m.set_errordef(0.99 * ml.cl)  # or ml.cl * 0.99
"""

from iminuit.util import ErrorDefSigma, ErrorDefCL

sigma = ErrorDefSigma(0.5, 1)
cl = ErrorDefCL(0.5, None)

del ErrorDefSigma, ErrorDefCL
