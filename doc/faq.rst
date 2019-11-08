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

The MINUIT2 user's guide explains all about it, see pages 6-8 and 38-40.

How do I obtain a high-quality error matrix?
--------------------------------------------

The answer can be found again in the MINUIT2 user's guide, see previous answer.

Can I have parameter limits that depend on each other (e.g. x^2 + y^2 < 3)?
---------------------------------------------------------------------------

MINUIT was only designed to handle box constrains, meaning that the limits on the parameters are independent of each other and constant during the minimization. If you want limits that depend on each other, you have three options (all with caveats), which are listed in increasing order of difficulty:

1) Change the variables so that the limits become independent, such as going from x,y to r, phi for a circle. This is not always possible or desirable, of course.

2) Use another minimizer to locate the minimum which supports complex boundaries. The `nlopt library <https://nlopt.readthedocs.io/en/latest>`_ and `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html>`_ have such minimizers. Once the minimum is found and if it is not near the boundary, place box constraints around the minimum and run iminuit to get the uncertainties (make sure that the box constraints are not too tight around the minimum). Neither nlopt nor scipy can give you the uncertainties.

3) Artificially increase the negative log-likelihood in the forbidden region. This is not as easy as it sounds.

The third method done properly is known as the `interior point or barrier method <https://en.wikipedia.org/wiki/Interior-point_method>`_. A glance at the Wikipedia article shows that one has to either run a series of minimizations with iminuit (and find a clever way of knowing when to stop) or implement this properly at the level of a Newton step, which would require changes to the complex and convoluted internals of MINUIT2.

Warning: you cannot just add a large value to the likelihood when the parameter boundary is violated. MIGRAD expects the likelihood function to be differential everywhere, because it uses the gradient of the likelihood to go downhill. The derivative at a discrete step is infinity and zero in the forbidden region. MIGRAD does not like this at all.
