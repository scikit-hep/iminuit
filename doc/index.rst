.. include:: references.txt

iminuit
=======

MINUIT from Python - Fitting like a boss

* Code: https://github.com/iminuit/iminuit
* Documentation: http://iminuit.readthedocs.org/
* Mailing list: https://groups.google.com/forum/#!forum/iminuit
* PyPI: https://pypi.python.org/pypi/iminuit
* License: LGPL (the iminuit source is MIT, but the bundled MINUIT is LGPL and thus the whole package is LGPL)

What is iminuit?
----------------

Interactive IPython-friendly mimizer based on `SEAL Minuit`_.

(It's included in the package, no need to install it separately.)

iminuit is designed from ground up to be fast, interactive and cython friendly. iminuit
extract function signature very permissively starting from checking *func_code*
down to last resort of parsing docstring (or you could tell iminuit to stop looking
and take your answer). The interface is inspired heavily
by PyMinuit and the status printout is inspired by ROOT Minuit. iminuit is
mostly compatible with PyMinuit (with few exceptions). Existing PyMinuit
code can be ported to iminuit by just changing the import statement.

In a nutshell
-------------

.. code-block:: python

    from iminuit import Minuit
    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2
    m = Minuit(f)
    m.migrad()
    print(m.values)  # {'x': 2,'y': 3,'z': 4}
    print(m.errors)  # {'x': 1,'y': 1,'z': 1}

If you are interested in fitting a curve or distribution, take a look at `probfit`_.


.. toctree::
    :maxdepth: 4
    :hidden:

    installation
    api


Tutorial
--------

All the tutorials are in tutorial directory. You can view them online too:

- `Quick start <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/tutorial.ipynb>`_
- `Hard Core Cython tutorial <http://nbviewer.ipython.org/urls/raw.github.com/iminuit/iminuit/master/tutorial/hard-core-tutorial.ipynb>`_.
  If you need to do a huge likelihood fit that needs speed, this is for you.
  If you don't care, just use `probfit`_.
  It's a fun read though I think.

API
---

See :ref:`api-doc`


Technical Stuff
---------------

Using it as a black box is a bad idea. Here are some fun reads; the order is given
by the order I think you should read.

* Wikipedia for `Quasi Newton Method`_ and `DFP formula`_. The magic behind MIGRAD.
* `Variable Metric Method for Minimization`_ William Davidon 1991
* `A New Approach to Variable Metric Algorithm`_ (R.Fletcher 1970)
* Original Paper: `MINUIT - A SYSTEM FOR FUNCTION MINIMIZATION AND ANALYSIS OF THE PARAMETER ERRORS AND CORRELATIONS`_ by Fred James and Matts Roos.

You can help
------------

Github allows you to contribute to this project very easily just fork the
repository, make changes and submit a pull request.

Here's the list of concrete open issues and feature requests:
https://github.com/iminuit/iminuit

More generally any contribution to the docs, tests and package itself is welcome!

* Documentation. Tell us what's missing, what's incorrect or misleading.
* Tests. If you have an example that shows a bug or problem, please file an issue!
* Performance. If you are a C/cython/python hacker go ahead and make it faster.
