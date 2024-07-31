.. include:: bibliography.txt

.. _faq:

.. currentmodule:: iminuit

FAQ
===

Disclaimer: Read the excellent MINUIT user guide!
-------------------------------------------------

Many technical questions are nicely covered by the `Minuit reference manual <https://cds.cern.ch/record/2296388>`_ of the original MINUIT in Fortran, on which MINUIT2 is based. Still relevant is chapter 5, which covers all the questions that one encounters when using Minuit in practice. If you have trouble with the fit, there is advice in sections 5.6 and 5.7 on how to fix common issues.

Another source is the :download:`MINUIT User's guide <mnusersguide.pdf>`, which covers the C++ Minuit2 library on which iminuit is based. It also contains advice on troubleshooting, we will frequently refer to it here. However, the Fortran manual usually goes into more detail and is therefore preferred.

I don't understand :meth:`Minuit.hesse`, :meth:`Minuit.minos`, :attr:`Minuit.errordef`; what do these do?
-----------------------------------------------------------------------------------------------------------
How do I interpret the parameter errors?
----------------------------------------
How do I obtain a high-quality error matrix?
--------------------------------------------
What effect has setting limits on the parameter uncertainties?
--------------------------------------------------------------

The MINUIT2 user's guide explains all about it, see pages 6-8 and 38-40.

I want to cite iminuit in my paper. Help?
------------------------------------------------

We use Zenodo to make each iminuit release citable. You can either cite iminuit as a software or you can cite the exact version that was used for your analysis. For more details, see :ref:`citation`.

Can I have parameter limits that depend on each other (e.g. x^2 + y^2 < 3)?
---------------------------------------------------------------------------

MINUIT was only designed to handle box constrains, meaning that the limits on the parameters are independent of each other and constant during the minimization. If you want limits that depend on each other, you have three options (all with caveats), which are listed in increasing order of difficulty:

1) Change the variables so that the limits become independent. For example, transform from Cartesian coordinates to polar coordinates for a circle. This is not always possible, of course.

2) Use another minimizer to locate the minimum which supports complex boundaries. The `nlopt`_ library and `scipy.optimize`_ have such minimizers. Once the minimum is found and if it is not near the boundary, place box constraints around the minimum and run iminuit to get the uncertainties (make sure that the box constraints are not too tight around the minimum). Neither `nlopt`_ nor `scipy`_ can give you the uncertainties.

3) Artificially increase the negative log-likelihood in the forbidden region. This is not as easy as it sounds.

The third method done properly is known as the `interior point or barrier method <https://en.wikipedia.org/wiki/Interior-point_method>`_. A glance at the Wikipedia article shows that one has to either run a series of minimizations with iminuit (and find a clever way of knowing when to stop) or implement this properly at the level of a Newton step, which would require changes to the complex and convoluted internals of MINUIT2.

Warning: you cannot just add a large value to the likelihood when the parameter boundary is violated. MIGRAD expects the likelihood function to be differential everywhere, because it uses the gradient of the likelihood to go downhill. The derivative at a discrete step is infinity and zero in the forbidden region. MIGRAD does not like this at all.

What happens when I change the strategy?
----------------------------------------

See page 5 of the MINUIT2 user guide for a rough explanation what this does. Here is our detailed explanation, extracted from looking into the source code and doing benchmarks.

* ``strategy = 0`` is the fastest and the number of function calls required to minimize scales linearly with the number of fitted parameters. The Hesse matrix is not computed during the minimization (only an approximation that is continuously updated). When the number of fitted parameters > 10, you should prefer this strategy.
* ``strategy = 1`` (default) is medium in speed. The number of function calls required scales quadratically with the number of fitted parameters. The different scales come from the fact that the Hesse matrix is explicitly computed in a Newton step, if Minuit detects significant correlations between parameters.
* ``strategy = 2`` has the same quadratic scaling as strategy 1 but is even slower. The Hesse matrix is always explicitly computed in each Newton step.

If you have a function that is everywhere analytical and the Hesse matrix varies not too much, strategy 0 should give you good results. Strategy 1 and 2 are better when these conditions are not given. The approximated Hesse matrix can become distorted in this case. The explicit computation of the Hesse matrix may help Minuit to recover after passing through such a region.

How do I get Hesse errors for sigma=2, 3, ...?
----------------------------------------------

For a least-squares function, you use ``errordef = sigma ** 2`` and for a negative log-likelihood function you use ``errordef = 0.5 * sigma ** 2``.

Why do extra messages appear in the terminal when I use `print_level=2` or larger?
------------------------------------------------------------------------------------

A `print_level=2` or higher activates internal debug messages directly from C++ MINUIT, which we cannot capture and print nicely in a Jupyter notebook, sorry.

Is it possible to stop iminuit by setting a tolerance for changes in the minimized function or the parameters?
--------------------------------------------------------------------------------------------------------------------------

No. MINUIT only uses the *Estimated-Distance-to-Minimum* (EDM) stopping criterion. It stops, if the difference between the actual function value at the estimated minimum location and the estimated function value at that location, based on MINUIT's internal parabolic approximation of the function, is small. This criterion depends strongly on the numerically computed first and second derivatives. Therefore, problems with MINUIT's convergence are often related to numerical issues when the first and second derivatives are computed.

More information about the EDM criterion can be found in the `MINUIT paper`_.

I am not sure if minimization or error estimation is reliable. What can I do?
-----------------------------------------------------------------------------

Plot the likelihood profile around the minimum as explained in the basic tutorial. Check that the parameter estimate is at the bottom. If the minimum is not parabolic, you may want to use MINOS to estimate the uncertainty interval instead of HESSE.

Why do you shout at me with CAPITAL LETTERS so much?
----------------------------------------------------

Sorry, that's just how it was in the old FORTRAN days from which MINUIT originates. Computers were incredibly loud back then and everyone was shouting all the time!

Seriously though: People were using typewriters. There was only a single font and no way to make letters italic or even bold, to set apart normal text from code. So people used CAPITAL LETTERS a lot for code in technical writing. For these historic reasons, MINUIT, MIGRAD, and HESSE are often written with capital letters here.
