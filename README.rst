.. |iminuit| image:: doc/_static/iminuit_logo.svg
   :alt: iminuit
   :target: http://iminuit.readthedocs.io/en/latest

|iminuit|
=========

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :alt: Scikit-HEP project package
   :target: https://scikit-hep.org
.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://img.shields.io/conda/vn/conda-forge/iminuit.svg
   :target: https://github.com/conda-forge/iminuit-feedstock
.. image:: https://github.com/scikit-hep/iminuit/workflows/Github-Actions/badge.svg
   :target: https://github.com/scikit-hep/iminuit/actions
.. image:: https://coveralls.io/repos/github/scikit-hep/iminuit/badge.svg?branch=develop
   :target: https://coveralls.io/github/scikit-hep/iminuit?branch=develop
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=latest
   :target: https://iminuit.readthedocs.io/en/latest
   :alt: Documentation Status
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/master?filepath=tutorial
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207

.. skip-marker-do-not-remove

*iminuit* is a Jupyter-friendly Python interface for the *Minuit2* C++ library maintained by CERN's ROOT team.

It can be used as a general robust function minimisation method, but is most
commonly used for likelihood fits of models to data, and to get model parameter
error estimates from likelihood profile analysis.

- Supported CPython versions: 3.6+
- Supported PyPy versions: 3.6
- Supported platforms: Linux, OSX and Windows.

* PyPI: https://pypi.org/project/iminuit
* Documentation: http://iminuit.readthedocs.org
* Source: https://github.com/scikit-hep/iminuit
* Gitter: https://gitter.im/Scikit-HEP/community
* License: *MINUIT2* is LGPL and *iminuit* is MIT
* Citation: https://doi.org/10.5281/zenodo.3949207

In a nutshell
-------------

.. code-block:: python

    from iminuit import Minuit

    def fcn(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    fcn.errordef = Minuit.LEAST_SQUARES

    m = Minuit(fcn, x=0, y=0, z=0)

    m.migrad()  # run optimiser
    print(m.values)  # x: 2, y: 3, z: 4

    m.hesse()   # run covariance estimator
    print(m.errors)  # x: 1, y: 1, z: 1

Versions
--------

**The current 2.x series has introduced breaking interfaces changes with respect to the 1.x series.**

All interface changes are documented in the `changelog`_ with recommendations how to upgrade. To keep existing scripts running, pin your major iminuit version to <2, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

.. _changelog: https://iminuit.readthedocs.io/en/stable/changelog.html
