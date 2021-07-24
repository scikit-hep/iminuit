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
  :alt: License
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207

*iminuit* is a Jupyter-friendly Python interface for the *Minuit2* C++ library maintained by CERN's ROOT team.

Minuit was designed to minimise statistical cost functions, for likelihood and least-squares fits of parametric models to data. It provides the best-fit parameters and error estimates from likelihood profile analysis.

- Supported CPython versions: 3.6+
- Supported PyPy versions: 3.6
- Supported platforms: Linux, OSX and Windows.

The iminuit package comes with additional features:

- Included cost functions for binned and unbinned maximum-likelihood and (robust)
  least-squares fits
- Support for SciPy minimisers
- Numba support (optional)

Checkout the comprehensive list of `tutorials`_ that demonstrate these features.

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/develop?filepath=doc%2Ftutorial

.. image:: https://img.shields.io/gitter/room/Scikit-HEP/iminuit
   :alt: Gitter

In a nutshell
-------------

.. code-block:: python

    from iminuit import Minuit

    def cost_function(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    fcn.errordef = Minuit.LEAST_SQUARES

    m = Minuit(cost_function, x=0, y=0, z=0)

    m.migrad()  # run optimiser
    print(m.values)  # x: 2, y: 3, z: 4

    m.hesse()   # run covariance estimator
    print(m.errors)  # x: 1, y: 1, z: 1

Versions
--------

**The current 2.x series has introduced breaking interfaces changes with respect to the 1.x series.**

All interface changes are documented in the `changelog`_ with recommendations how to upgrade. To keep existing scripts running, pin your major iminuit version to <2, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

.. _changelog: https://iminuit.readthedocs.io/en/stable/changelog.html
.. _tutorials: https://iminuit.readthedocs.io/en/stable/tutorials.html
