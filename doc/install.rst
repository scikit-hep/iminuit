.. include:: bibliography.txt

.. _install:

Installation
============

Note: iminuit is tested to work with PyPy3.5 and PyPy3.6, but we do not provide binary packages for PyPy. PyPy users need to install the source package of iminuit. This happens automatically when you install it via ``conda`` or ``pip``, but requires a working C++ compiler.

pip
---

To install the latest stable version from https://pypi.org/project/iminuit/ with ``pip``:

.. code-block:: bash

    $ pip install iminuit

If your platform is not supported by a binary wheel, ``pip install`` requires that you
have a C++ compiler available but otherwise runs the compilation automatically.

Conda
-----

We also provide binary packages for ``conda`` users via https://anaconda.org/conda-forge/iminuit:

.. code-block:: bash

    $ conda install -c conda-forge iminuit

The ``conda`` packages are semi-automatically maintained and usually quickly support the least Python version on all platforms.

Installing from source
----------------------

For users
+++++++++

If you need the latest unreleased version, you can download and install directly from Github. The easiest way is to use pip.

    pip install iminuit@git+https://github.com/scikit-hep/iminuit@develop

This requires a C++ compiler with C++14 support.

For contributors/developers
+++++++++++++++++++++++++++

See :ref:`contribute`.
