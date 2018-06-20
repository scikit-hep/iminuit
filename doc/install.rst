.. include:: references.txt

.. _install:

Installation
============

- iminuit works with Python 3.5 or later, as well as legacy Python 2.7.
- Linux, macOS and Windows are supported.

To install the latest stable version from https://pypi.org/project/iminuit/ with `pip`:

.. code-block:: bash

    $ pip install iminuit

We don't distribute binary wheels for `iminuit`, so `pip install` requires that you
have a C++ compiler available.

We do provide binary packages for `conda` users via https://anaconda.org/conda-forge/iminuit:

.. code-block:: bash

    $ conda install -c conda-forge iminuit

The only required dependency for `iminuit` is `numpy`.

As explained in the documentation, using `ipython` and `jupyter` for
interactive analysis, as well as `cython` for speed is advisable,
so you might want to install those as well.

You can use the following command to check if you have ``iminuit`` installed,
and which version you have::

    $ python -m iminuit
