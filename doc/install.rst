.. include:: references.txt

.. _install:

Installation
============

- iminuit works with Python 3.5 or later, as well as legacy Python 2.7.
- Linux, macOS and Windows are supported.

pip
---

To install the latest stable version from https://pypi.org/project/iminuit/ with `pip`:

.. code-block:: bash

    $ pip install iminuit

We don't distribute binary wheels for `iminuit`, so `pip install` requires that you
have a C++ compiler available.

Conda
-----

We provide binary packages for `conda` users via https://anaconda.org/conda-forge/iminuit:

.. code-block:: bash

    $ conda install -c conda-forge iminuit

The only required dependency for `iminuit` is `numpy`.

As explained in the documentation, using `ipython` and `jupyter` for
interactive analysis, as well as `cython` for speed is advisable,
so you might want to install those as well.

Check
-----

To check your `iminuit` version number and install location::

    $ python
    >>> import iminuit
    >>> iminuit
    # install location is printed
    >>> iminuit.__version__
    # version number is printed

Usually if `import iminuit` works, everything is OK.
But in case you suspect that you have a broken `iminuit` installation,
you can run the automated tests like this::

    $ pip install pytest
    $ python
    >>> import iminuit
    >>> iminuit.test()
