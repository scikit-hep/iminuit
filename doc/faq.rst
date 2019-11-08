.. include:: references.txt

.. _faq:

.. currentmodule:: iminuit

Frequently-asked questions
==========================

Disclaimer: Read the excellent MINUIT2 user guide!
--------------------------------------------------

Many technical questions are nicely covered by the user guide of MINUIT2,
:download:`MINUIT User's guide <mnusersguide.pdf>`. We will frequently refer to it here.

I don't understand :meth:`Minuit.hesse`, :meth:`Minuit.minos`, `errordef`; what do these do?
--------------------------------------------------------------------------------------------
How do I interpret the parameter errors that iminuit produces?
--------------------------------------------------------------
How do I obtain a high-quality error matrix?
--------------------------------------------
What effect has setting limits on the parameter uncertainties?
--------------------------------------------------------------

The MINUIT2 user's guide explains all about it, see pages 6-8 and 38-40.

Can I have parameter limits that depend on each other (e.g. x^2 + y^2 < 3)?
---------------------------------------------------------------------------

MINUIT was only designed to handle box constrains, meaning that the limits on the parameters are independent of each other and constant during the minimization. If you want limits that depend on each other, you have three options (all with caveats), which are listed in increasing order of difficulty:

1) Change the variables so that the limits become independent, such as going from x,y to r, phi for a circle. This is not always possible or desirable, of course.

2) Use another minimizer to locate the minimum which supports complex boundaries. The `nlopt library <https://nlopt.readthedocs.io/en/latest>`_ and `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html>`_ have such minimizers. Once the minimum is found and if it is not near the boundary, place box constraints around the minimum and run iminuit to get the uncertainties (make sure that the box constraints are not too tight around the minimum). Neither nlopt nor scipy can give you the uncertainties.

3) Artificially increase the negative log-likelihood in the forbidden region. This is not as easy as it sounds.

The third method done properly is known as the `interior point or barrier method <https://en.wikipedia.org/wiki/Interior-point_method>`_. A glance at the Wikipedia article shows that one has to either run a series of minimizations with iminuit (and find a clever way of knowing when to stop) or implement this properly at the level of a Newton step, which would require changes to the complex and convoluted internals of MINUIT2.

Warning: you cannot just add a large value to the likelihood when the parameter boundary is violated. MIGRAD expects the likelihood function to be differential everywhere, because it uses the gradient of the likelihood to go downhill. The derivative at a discrete step is infinity and zero in the forbidden region. MIGRAD does not like this at all.

What happens when I change the strategy?
----------------------------------------

See page 5 of the MINUIT2 user guide.

How do I get Hesse errors for sigma=2, 3, ...?
----------------------------------------------

For a least-squares function, you use `errordef = sigma ** 2` and for a negative log-likelihood function you use `errordef = 0.5 * sigma ** 2`.

Why do extra messages appear in the terminal when I use `print_level=2` or larger?
------------------------------------------------------------------------------------

A `print_level=2` or higher activates internal debug messages directly from C++ MINUIT, which we cannot capture and print nicely in a Jupyter notebook, sorry.

Is it possible to stop iminuit by setting a tolerance for changes in the minimized function or the parameters?
--------------------------------------------------------------------------------------------------------------------

No. MINUIT2 only uses the `Estimated Distance to Minimum`_ (EDM) stopping criterion, in which MINUIT2 compares its current local parabolic estimate of the minimized function with reality. It stops if the vertical distance of the estimate is small. More information about the EDM criterion can be found in the MINUIT paper: `MINUIT - A SYSTEM FOR FUNCTION MINIMIZATION AND ANALYSIS OF THE PARAMETER ERRORS AND CORRELATIONS`_.

I am not sure if minimization or error estimation is reliable. What can I do?
-----------------------------------------------------------------------------

Plot the likelihood profile around the minimum as explained in the basic tutorial. Check that the parameter estimate is at the bottom. If the minimum is not parabolic, you may want to use MINOS to estimate the uncertainty interval instead of HESSE.

Why do you shout at me with CAPITAL LETTERS so much?
----------------------------------------------------

Sorry, that's just how it was in the old FORTRAN days from which MINUIT originates. Computers were incredibly loud back then and everyone was shouting all the time!

Seriously though: People were using type writers. There was only a single font and no way to make letters italic or even bold, to set apart normal text from code. So people used CAPITAL LETTERS a lot for code in writing. For these historic reasons, MINUIT, MIGRAD, and HESSE are often written with capital letters here.
