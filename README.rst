iminuit
=======

.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://dev.azure.com/scikit-hep/iMinuit/_apis/build/status/scikit-hep.iminuit?branchName=master
   :target: https://dev.azure.com/scikit-hep/iMinuit
.. image:: https://travis-ci.org/scikit-hep/iminuit.svg?branch=develop
   :target: https://travis-ci.org/scikit-hep/iminuit
.. image:: https://ci.appveyor.com/api/projects/status/g6vymxvu9ax34e7l?svg=true
   :target: https://ci.appveyor.com/project/HDembinski/iminuit-b4eg8
.. image:: https://codecov.io/gh/scikit-hep/iminuit/branch/develop/graph/badge.svg
   :target: https://codecov.io/gh/scikit-hep/iminuit
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=latest
   :target: https://iminuit.readthedocs.io/en/develop/?badge=latest
   :alt: Documentation Status

.. skip-marker-do-not-remove

*iminuit* is a Python interface to the *MINUIT2* C++ package.

It can be used as a general robust function minimization method,
but is most commonly used for likelihood fits of models to data,
and to get model parameter error estimates from likelihood profile analysis.

* Code: https://github.com/scikit-hep/iminuit
* Documentation: http://iminuit.readthedocs.org/
* Gitter: https://gitter.im/HSF/PyHEP
* Mailing list: https://groups.google.com/forum/#!forum/scikit-hep-forum
* PyPI: https://pypi.org/project/iminuit/
* License: *MINUIT2* is LGPL and *iminuit* is MIT
* Citation: https://github.com/scikit-hep/iminuit/blob/master/CITATION

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
