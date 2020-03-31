.. include:: references.txt

.. _install:

Installation
============

Note: iminuit is tested to work with PyPy3.5 and PyPy3.6, but we do not provide binary packages for PyPy. PyPy users need to install the source package of iminuit. This happens automatically when you install it via conda or pip, but requires a working C++ compiler.

Conda
-----

We provide binary packages for `conda` users via https://anaconda.org/conda-forge/iminuit:

.. code-block:: bash

    $ conda install -c conda-forge iminuit

`iminuit` only depends on `numpy`. The conda packages are semi-automatically maintained and usually quickly support the least Python version on all platforms.

pip
---

To install the latest stable version from https://pypi.org/project/iminuit/ with `pip`:

.. code-block:: bash

    $ pip install iminuit

If your platform is not supported by a binary wheel, `pip install` requires that you
have a C++ compiler available but otherwise runs the compilation automatically. As an alternative you can try to install iminuit with conda.

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
