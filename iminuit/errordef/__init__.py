"""
The modules :mod:`iminuit.errordef.ml` (for negative log-likelihood functions) and :mod:`iminuit.errordef.lsq` (for least-squares functions a.k.a. :math:`\chi^2` functions) contain special constants ``sigma`` and ``cl`` to conveniently set the error definition in the :class:`iminuit.Minuit` class. These constants are not normal numbers, they are instances of the classes :class:`iminuit.util.ErrorDefSigma` and :class:`iminuit.util.ErrorDefCL`, respectively, which have overloaded operators so that the syntax becomes straight forward. How to use these is best explained with some examples.

.. code::

    from iminuit import Minuit
    
    from iminuit.errordef import lsq.sigma, lsq.cl
    # this is for least-squares; for negative log-likelihood import nll.sigma, nll.cl 

    def lsq(mu):
        return (mu - 1) ** 2

    # for standard errors
    m = Minuit(lsq, mu=0, errordef=1 * lsq.sigma)
    m.migrad()
    # now m.errors["mu"] is approximately 1

    # for 2-sigma errors
    m.set_errordef(2 * lsq.sigma)  # or lsq.sigma * 2
    m.hesse()
    # now m.errors["mu"] is approximately 2

    # for 99 % confidence level
    m.set_errordef(0.99 * lsq.cl)  # or lsq.cl * 0.99
    m.hesse()
    # now m.errors["mu"] is approximately 2.6

If you are curious what kind of error definition these special constants compute, convert them to floats, for example: ``float(0.68 * lsq.cl)``. 
"""
