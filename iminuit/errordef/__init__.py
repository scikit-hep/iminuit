"""
The modules :mod:`iminuit.errordef.ml` (for likelihood functions) and :mod:`iminuit.errordef.lsq` (for least-squares functions a.k.a. :math:`\chi^2` functions) contain special constants ``sigma`` and ``cl`` to conveniently set the error definition in the :class:`iminuit.Minuit` class. These constants are not normal numbers, they are instances of the classes :class:`iminuit.util.ErrorDefSigma` and :class:`iminuit.util.ErrorDefCL`, respectively, which have overloaded operators so that the syntax becomes straight forward. How to use these is best explained with some examples.

.. code::

    from iminuit import Minuit
    
    # this is for likelihoods; for least-squares import from iminuit.errordef.lsq 
    from iminuit.errordef.ml import sigma, cl

    def negative_log_likelihood(a, b, c):
        ...

    # for standard errors
    m = Minuit(negative_log_likelihood, ..., errordef=1 * sigma)
    # for 2-sigma errors
    m.set_errordef(2 * sigma)  # or sigma * 2
    # for 99 % confidence level
    m.set_errordef(0.99 * cl)  # or cl * 0.99

If you are curious what kind of error definition these constants compute, convert them to floats, for example: ``float(0.68 * lsq.cl)``. 
"""
