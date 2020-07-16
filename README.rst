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
.. image:: https://dev.azure.com/scikit-hep/iMinuit/_apis/build/status/scikit-hep.iminuit?branchName=master
   :target: https://dev.azure.com/scikit-hep/iMinuit
.. image:: https://github.com/scikit-hep/iminuit/workflows/Github-Actions/badge.svg
   :target: https://github.com/scikit-hep/iminuit/actions
.. image:: https://codecov.io/gh/scikit-hep/iminuit/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/scikit-hep/iminuit
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=latest
   :target: https://iminuit.readthedocs.io/en/develop/?badge=latest
   :alt: Documentation Status
.. image:: https://mybinder.org/badge_logo.svg
  :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/master?filepath=tutorial
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207

.. skip-marker-do-not-remove

*iminuit* is a Jupyter-friendly Python frontend to the *MINUIT2* C++ library.

It can be used as a general robust function minimisation method,
but is most commonly used for likelihood fits of models to data,
and to get model parameter error estimates from likelihood profile analysis.

- Supported CPython versions: 3.5+
- Supported PyPy versions: 3.5, 3.6
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

    def f(x, y, z):
        return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2

    m = Minuit(f)

    m.migrad()  # run optimiser
    print(m.values)  # {'x': 2,'y': 3,'z': 4}

    m.hesse()   # run covariance estimator
    print(m.errors)  # {'x': 1,'y': 1,'z': 1}
