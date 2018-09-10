.. include:: references.txt

.. _about:

About
=====

What is iminuit?
----------------

iminuit is the fast interactive IPython-friendly minimizer based on `Minuit2`_ in ROOT-6.12.06.

For a hands-on introduction, see the :ref:`tutorials`.

**Easy to install**

You can install iminuit with pip. It only needs a moderately recent C++ compiler on your machine. The Minuit2 code is bundled, so you don't need to install it separately.

**Support for Python 2.7 to 3.5 and Numpy**

Whether you use the latest Python 3 or stick to classic Python 2, iminuit works for you. Numpy is supported: you can minimize functions that accept numpy arrays and get the fit results as numpy arrays. If you prefer to access parameters by name, that works as well!

**Robust optimizer and error estimator**

iminuit uses Minuit2 to minimize your functions, a battle-hardened code developed and maintained by scientists at CERN, the world's leading particle accelerator laboratory. Minuit2 has good performance compared to other minimizers, and it is one of the few codes out there which compute error estimates for your parameters. When you do statistics seriously, this is a must-have.

**Interactive convenience**

iminuit extracts the parameter names from your function signature (or the docstring) and allows you access them by their name. For example, if your function is defined as ``func(alpha, beta)``, iminuit understands that your first parameter is `alpha` and the second `beta` and will use these names in status printouts (you can override this inspection if you like). It also produces pretty messages on the console and in Jupyter notebooks.

**Support for Cython**

iminuit was designed to work with Cython functions, in order to speed up the minimization of complex functions.

**Successor of PyMinuit**

iminuit is mostly compatible with PyMinuit. Existing PyMinuit code can be ported to iminuit by just changing the import statement.

If you are interested in fitting a curve or distribution, take a look at `probfit`_.


Technical docs
--------------

When you use iminuit/Minuit2 seriously, it is a good idea to understand a bit how it works and what possible limitations are in your case. The following links help you to understand the numerical approach behind Minuit2. The links are ordered by recommended reading order.

* Wikipedia for `Quasi Newton Method`_ and `DFP formula`_. The numerical algorithm behind MIGRAD.
* `Variable Metric Method for Minimization`_ by William Davidon, 1991
* `A New Approach to Variable Metric Algorithm`_ by R. Fletcher, 1970
* Original user guide for C++ Minuit2: :download:`MINUIT User's guide <mnusersguide.pdf>` by F. James, 2004
* Original Paper: `MINUIT - A SYSTEM FOR FUNCTION MINIMIZATION AND ANALYSIS OF THE PARAMETER ERRORS AND CORRELATIONS`_ by Fred James and Matts Roos, .

Team
----

iminuit was created by **Piti Ongmongkolkul** (@piti118). It is a logical successor of pyminuit/pyminuit2, created by **Jim Pivarski** (@jpivarski).

Maintainers
~~~~~~~~~~~

* Piti Ongmongkolkul (@piti118)
* Chih-hsiang Cheng (@gitcheng)
* Christoph Deil (@cdeil)
* Hans Dembinski (@HDembinski)

Contributors
~~~~~~~~~~~~

* Jim Pivarski (@jpivarski)
* David Men\'endez Hurtado (@Dapid)
* Chris Burr (@chrisburr)
* Andrew ZP Smith (@energynumbers)
* Fabian Rost (@fabianrost84)
* Alex Pearce (@alexpearce)
* Lukas Geiger (@lgeiger)
* Omar Zapata (@omazapa)
