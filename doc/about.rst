.. include:: bibliography.txt

.. _about:

About
=====

What is iminuit?
----------------

iminuit is the fast interactive IPython-friendly minimiser based on `Minuit2`_ C++ library maintained by CERN's ROOT team. The corresponding version of the C++ code is listed after the '+' in the iminuit version string: |release|.

For a hands-on introduction, see the :ref:`tutorials`.

**Easy to install with binary wheels for CPython and PyPy**

If you use one of the currently maintained CPython versions on Windows, OSX, or Linux, installing iminuit from PyPI is a breeze. We offer pre-compiled binary wheels for these platforms. We even support the latest PyPy. On other platforms you can also install the iminuit source package if you have a C++ compiler that supports C++14 or newer. Installing the source package compiles iminuit for your machine automatically. All required dependencies are installed from PyPI automatically.

**Robust optimiser and error estimator**

iminuit uses Minuit2 to minimise your functions, a battle-hardened code developed and maintained by scientists at CERN, the world's leading particle accelerator laboratory. Minuit2 has good performance compared to other minimizers, and it is one of the few codes out there which compute error estimates for your parameters. When you do statistics seriously, this is a must-have.

**Support for NumPy**

iminuit interoperates with NumPy. You can minimize functions that accept parameters as positional arguments, but also functions that accept all parameters as a single NumPy array. Fit results are either NumPy arrays or array-like types, which mimic key aspects of the array interface and which are easily convertible to arrays.

**Interactive convenience**

iminuit extracts the parameter names from your function signature (or the docstring) and allows you access them by their name. For example, if your function is defined as ``func(alpha, beta)``, iminuit understands that your first parameter is ``alpha`` and the second ``beta`` and will use these names in status printouts (you can override this inspection if you like). It also produces pretty messages on the console and in Jupyter notebooks.

**Support for Cython, Numba, JAX, Tensorflow, ...**

iminuit was designed to work with functions implemented in Python and with compiled functions, produced by Cython, Numba, etc.

**Successor of PyMinuit**

iminuit is mostly compatible with PyMinuit. Existing PyMinuit code can be ported to iminuit by just changing the import statement.

If you are interested in fitting a curve or distribution, take a look at `probfit`_.

Who is using iminuit?
---------------------

This is a list of known users of iminuit. Please let us know if you use iminuit, we like to keep in touch.

* probfit_
* gammapy_
* flavio_
* Veusz_
* TensorProb_
* threeML_
* pyhf_
* zfit_
* ctapipe_
* lauztat_

Technical docs
--------------

When you use iminuit/Minuit2 seriously, it is a good idea to understand a bit how it works and what possible limitations are in your case. The following links help you to understand the numerical approach behind Minuit2. The links are ordered by recommended reading order.

* The `MINUIT paper`_ by Fred James and Matts Roos, 1975.
* Wikipedia articles for the `Quasi Newton Method`_ and `DFP formula`_ used by MIGRAD.
* `Variable Metric Method for Minimization`_ by William Davidon, 1991.
* Original user guide for C++ Minuit2: :download:`MINUIT User's guide <mnusersguide.pdf>` by Fred James, 2004.

Team
----

iminuit was created by **Piti Ongmongkolkul**. It is a logical successor of pyminuit/pyminuit2, created by **Jim Pivarski**. It is now maintained by **Hans Dembinski** and the Scikit-HEP_ community.

Maintainers
~~~~~~~~~~~

* Hans Dembinski (@HDembinski) [current]
* Christoph Deil (@cdeil)
* Piti Ongmongkolkul (@piti118)
* Chih-hsiang Cheng (@gitcheng)

Contributors
~~~~~~~~~~~~

* Jim Pivarski (@jpivarski)
* Henry Schreiner (@henryiii)
* David Men\'endez Hurtado (@Dapid)
* Chris Burr (@chrisburr)
* Andrew ZP Smith (@energynumbers)
* Fabian Rost (@fabianrost84)
* Alex Pearce (@alexpearce)
* Lukas Geiger (@lgeiger)
* Omar Zapata (@omazapa)
