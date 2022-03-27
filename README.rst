.. |iminuit| image:: doc/_static/iminuit_logo.svg
   :alt: iminuit
   :target: http://iminuit.readthedocs.io/en/latest

|iminuit|
=========

.. skip-marker-do-not-remove

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org
.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://img.shields.io/conda/vn/conda-forge/iminuit.svg
   :target: https://github.com/conda-forge/iminuit-feedstock
.. image:: https://coveralls.io/repos/github/scikit-hep/iminuit/badge.svg?branch=develop
   :target: https://coveralls.io/github/scikit-hep/iminuit?branch=develop
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=stable
   :target: https://iminuit.readthedocs.io/en/stable
.. image:: https://img.shields.io/pypi/l/iminuit
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207
.. image:: https://img.shields.io/badge/ascl-2108.024-blue.svg?colorB=262255
   :target: https://ascl.net/2108.024
   :alt: ascl:2108.024

*iminuit* is a Jupyter-friendly Python interface for the *Minuit2* C++ library maintained by CERN's ROOT team.

Minuit was designed to minimise statistical cost functions, for likelihood and least-squares fits of parametric models to data. It provides the best-fit parameters and error estimates from likelihood profile analysis.

- Supported CPython versions: 3.6+
- Supported PyPy versions: 3.6+
- Supported platforms: Linux, OSX and Windows.

The iminuit package comes with additional features:

- Builtin cost functions for statistical fits

  - Binned and unbinned maximum-likelihood
  - Non-linear regression with (optionally robust) weighted least-squares
  - Gaussian penalty terms
  - Cost functions can be combined by adding them: ``total_cost = cost_1 + cost_2``
- Support for SciPy minimisers as alternatives to Minuit's Migrad algorithm (optional)
- Support for Numba accelerated functions (optional)

Checkout our large and comprehensive list of `tutorials`_ that take you all the way from beginner to power user. For help and how-to questions, please use the `discussions`_ on GitHub.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/develop?filepath=doc%2Ftutorial

In a nutshell
-------------

iminuit is intended to be used with a user-provided negative log-likelihood function or least-squares function. Standard functions are included in ``iminuit.cost``, so you don't have to write them yourself. The following example shows how iminuit is used with a dummy least-squares function.

.. code-block:: python

    from iminuit import Minuit

    def cost_function(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    cost_function.errordef = Minuit.LEAST_SQUARES

    m = Minuit(cost_function, x=0, y=0, z=0)

    m.migrad()  # run optimiser
    print(m.values)  # x: 2, y: 3, z: 4

    m.hesse()   # run covariance estimator
    print(m.errors)  # x: 1, y: 1, z: 1

Partner projects
----------------

* `numba_stats`_ provides faster implementations of probability density functions than scipy, and a few specific ones used in particle physics that are not in scipy.
* `jacobi`_ provides a robust, fast, and accurate calculation of the Jacobi matrix of any transformation function and building a function for generic error propagation.

Versions
--------

**The current 2.x series has introduced breaking interfaces changes with respect to the 1.x series.**

All interface changes are documented in the `changelog`_ with recommendations how to upgrade. To keep existing scripts running, pin your major iminuit version to <2, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

.. _changelog: https://iminuit.readthedocs.io/en/stable/changelog.html
.. _tutorials: https://iminuit.readthedocs.io/en/stable/tutorials.html
.. _discussions: https://github.com/scikit-hep/iminuit/discussions
.. _jacobi: https://github.com/hdembinski/jacobi
.. _numba_stats: https://github.com/HDembinski/numba-stats
