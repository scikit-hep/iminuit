.. |iminuit| image:: doc/_static/iminuit_logo.svg
   :alt: iminuit

|iminuit|
=========

.. version-marker-do-not-remove

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org
.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://img.shields.io/conda/vn/conda-forge/iminuit.svg
   :target: https://github.com/conda-forge/iminuit-feedstock
.. image:: https://coveralls.io/repos/github/scikit-hep/iminuit/badge.svg?branch=develop
   :target: https://coveralls.io/github/scikit-hep/iminuit?branch=develop
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=latest
   :target: https://iminuit.readthedocs.io/en/stable
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207
.. image:: https://img.shields.io/badge/ascl-2108.024-blue.svg?colorB=262255
   :target: https://ascl.net/2108.024
   :alt: ascl:2108.024
.. image:: https://img.shields.io/gitter/room/Scikit-HEP/iminuit
   :target: https://gitter.im/Scikit-HEP/iminuit
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/develop?filepath=doc%2Ftutorial

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

Documentation
-------------

Checkout our large and comprehensive list of `tutorials`_ that take you all the way from beginner to power user. For help and how-to questions, please use the `discussions`_ on GitHub or `gitter`_.

**Lecture by Glen Cowan**

`In the exercises to his lecture for the KMISchool 2022 <https://github.com/KMISchool2022>`_, Glen Cowan shows how to solve statistical problems in Python with iminuit. You can find the lectures and exercises on the Github page, which covers both frequentist and Bayesian methods.

`Glen Cowan <https://scholar.google.com/citations?hl=en&user=ljQwt8QAAAAJ&view_op=list_works>`_ is a known for his papers and international lectures on statistics in particle physics, as a member of the Particle Data Group, and as author of the popular book `Statistical Data Analysis <https://www.pp.rhul.ac.uk/~cowan/sda/>`_.

In a nutshell
-------------

iminuit is intended to be used with a user-provided negative log-likelihood function or least-squares function. Standard functions are included in ``iminuit.cost``, so you don't have to write them yourself. The following example shows how iminuit is used with a dummy least-squares function.

.. code-block:: python

    from iminuit import Minuit

    def cost_function(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(cost_function, x=0, y=0, z=0)

    m.migrad()  # run optimiser
    m.hesse()   # run covariance estimator

    print(m.values)  # x: 2, y: 3, z: 4
    print(m.errors)  # x: 1, y: 1, z: 1

Interactive fitting
-------------------

iminuit optionally supports an interactive fitting mode in Jupyter notebooks.

.. image:: doc/_static/interactive_demo.gif
   :alt: Animated demo of an interactive fit in a Jupyter notebook

Partner projects
----------------

* `boost-histogram` from Scikit-HEP provides fast generalized histograms that you can use with the builtin cost functions.
* `numba_stats`_ provides faster implementations of probability density functions than scipy, and a few specific ones used in particle physics that are not in scipy.
* `jacobi`_ provides a robust, fast, and accurate calculation of the Jacobi matrix of any transformation function and building a function for generic error propagation.

Versions
--------

**The current 2.x series has introduced breaking interfaces changes with respect to the 1.x series.**

All interface changes are documented in the `changelog`_ with recommendations how to upgrade. To keep existing scripts running, pin your major iminuit version to <2, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

.. _changelog: https://iminuit.readthedocs.io/en/stable/changelog.html
.. _tutorials: https://iminuit.readthedocs.io/en/stable/tutorials.html
.. _discussions: https://github.com/scikit-hep/iminuit/discussions
.. _gitter: https://gitter.im/Scikit-HEP/iminuit
.. _jacobi: https://github.com/hdembinski/jacobi
.. _numba_stats: https://github.com/HDembinski/numba-stats
