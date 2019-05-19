.. include:: references.txt

.. _install:

Installation
============

- Supported Python versions: 2.7, 3.5+, PyPy3.5.
- Supported platforms: Linux, OSX and Windows.

pip
---

To install the latest stable version from https://pypi.org/project/iminuit/ with `pip`:

.. code-block:: bash

    $ pip install iminuit

If your platform is not supported by a binary wheel, `pip install` requires that you
have a C++ compiler available.

Conda
-----

We provide binary packages for `conda` users via https://anaconda.org/conda-forge/iminuit:

.. code-block:: bash

    $ conda install -c conda-forge iminuit

The only required dependency for `iminuit` is `numpy`. As explained in the documentation, using `ipython` and `jupyter` for interactive analysis, as well as `cython` for speed is advisable, so you might want to install those as well.

Installing from source
----------------------

For users
+++++++++

If you need the latest unreleased version, you can download and install directly from Github. The easiest way is to use pip.

    pip install git+https://github.com/scikit-hep/iminuit@develop#egg=iminuit

For contributors/developers
+++++++++++++++++++++++++++

See :ref:`contribute`.

Check installation
------------------

To check your `iminuit` version number and install location::

    $ python
    >>> import iminuit
    >>> iminuit
    # install location is printed
    >>> iminuit.__version__
    # version number is printed

Usually if `import iminuit` works, everything is OK. But in case you suspect that you have a broken `iminuit` installation, you can run the automated tests like this::

    $ pip install pytest
    $ python -c "import iminuit; iminuit.test()"
